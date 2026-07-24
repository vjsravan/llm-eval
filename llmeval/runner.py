"""
Executes a suite against a model and records a Run.

Design notes:
  * Concurrency is bounded — an unbounded gather against a rate-limited provider turns
    a 200-case suite into 200 simultaneous 429s.
  * Failures are recorded, never raised. A provider timeout on case 47 must not lose
    the 46 results already collected; a partial run with a known failure count is
    useful, a crashed run is not.
  * Runs serialise to JSON so today's run becomes tomorrow's baseline.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from .assertions import Score
from .dataset import GoldenCase, Suite

# A model function takes the case input and returns the completion text. Sync or async
# are both accepted; sync functions are pushed to a worker thread so one slow blocking
# client cannot stall the event loop.
ModelFn = Callable[[str], "str | Awaitable[str]"]


@dataclass
class CaseResult:
    case_id: str
    output: str
    score: float
    assertion_scores: dict[str, float]
    reasons: dict[str, str]
    latency_ms: float
    tags: tuple[str, ...] = ()
    weight: float = 1.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Run:
    """One execution of a suite. `label` is what shows up in reports — use the prompt
    version or git sha, not 'test1'."""

    run_id: str
    suite: str
    label: str
    created_at: str
    results: list[CaseResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def scores(self) -> dict[str, float]:
        return {r.case_id: r.score for r in self.results}

    @property
    def mean_score(self) -> float:
        return statistics.fmean([r.score for r in self.results]) if self.results else 0.0

    @property
    def weighted_mean(self) -> float:
        """Weighted by case importance. Diverges from mean_score when your critical
        cases behave differently from your cosmetic ones — which is the situation you
        most want to notice."""
        total_w = sum(r.weight for r in self.results)
        if not total_w:
            return 0.0
        return sum(r.score * r.weight for r in self.results) / total_w

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.score >= 1.0) / len(self.results)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def p95_latency_ms(self) -> float:
        lats = sorted(r.latency_ms for r in self.results)
        if not lats:
            return 0.0
        return round(lats[min(len(lats) - 1, int(0.95 * len(lats)))], 2)

    def by_tag(self) -> dict[str, float]:
        """Mean score per tag. The aggregate can look healthy while one slice quietly
        collapses, so the per-slice view is usually where the real story is."""
        buckets: dict[str, list[float]] = {}
        for r in self.results:
            for t in r.tags:
                buckets.setdefault(t, []).append(r.score)
        return {t: round(statistics.fmean(v), 4) for t, v in sorted(buckets.items())}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Run":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["results"] = [CaseResult(**{**r, "tags": tuple(r.get("tags", []))}) for r in raw["results"]]
        return cls(**raw)


async def _invoke(model: ModelFn, prompt: str) -> str:
    result = model(prompt)
    if asyncio.iscoroutine(result):
        return await result
    # Blocking client: hand it to the default executor rather than blocking the loop.
    return await asyncio.to_thread(lambda: result)  # type: ignore[return-value]


async def _run_case(
    case: GoldenCase,
    model: ModelFn,
    *,
    retries: int,
    timeout_s: float,
) -> CaseResult:
    last_error: str | None = None
    started = time.perf_counter()

    for attempt in range(retries + 1):
        try:
            output = await asyncio.wait_for(_invoke(model, case.input), timeout=timeout_s)
            elapsed = (time.perf_counter() - started) * 1000

            scores: dict[str, Score] = {}
            for assertion in case.assertions:
                # A bad assertion must not sink the whole run; record it as a zero.
                try:
                    scores[assertion.name] = assertion.score(output, case)
                except Exception as exc:  # noqa: BLE001
                    scores[assertion.name] = Score(0.0, f"assertion raised {type(exc).__name__}: {exc}")

            # All assertions on a case must hold, so the case score is the weakest link.
            # Averaging here would let a passing format check paper over a wrong answer.
            case_score = min(s.value for s in scores.values()) if scores else 0.0

            return CaseResult(
                case_id=case.id,
                output=output,
                score=round(case_score, 4),
                assertion_scores={k: v.value for k, v in scores.items()},
                reasons={k: v.reason for k, v in scores.items()},
                latency_ms=round(elapsed, 2),
                tags=case.tags,
                weight=case.weight,
            )
        except asyncio.TimeoutError:
            last_error = f"timeout after {timeout_s}s"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            await asyncio.sleep(min(2**attempt * 0.25, 4.0))

    return CaseResult(
        case_id=case.id,
        output="",
        score=0.0,
        assertion_scores={},
        reasons={},
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        tags=case.tags,
        weight=case.weight,
        error=last_error,
    )


async def run_suite_async(
    suite: Suite,
    model: ModelFn,
    *,
    label: str = "candidate",
    concurrency: int = 8,
    retries: int = 2,
    timeout_s: float = 60.0,
    metadata: dict | None = None,
) -> Run:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(case: GoldenCase) -> CaseResult:
        async with sem:
            return await _run_case(case, model, retries=retries, timeout_s=timeout_s)

    results = await asyncio.gather(*(guarded(c) for c in suite.cases))

    return Run(
        run_id=uuid.uuid4().hex[:12],
        suite=suite.name,
        label=label,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        results=list(results),
        metadata=metadata or {},
    )


def run_suite(suite: Suite, model: ModelFn, **kwargs) -> Run:
    """Blocking wrapper for scripts and CI."""
    return asyncio.run(run_suite_async(suite, model, **kwargs))
