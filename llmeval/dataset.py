"""
Golden datasets: the versioned set of cases a prompt must keep satisfying.

A suite is plain YAML-ish JSON on disk so it diffs cleanly in review. When someone
changes a prompt, the reviewer should be able to see which cases moved and by how much.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .assertions import (
    Assertion,
    Contains,
    Equals,
    JsonFieldEquals,
    LlmJudge,
    MatchesRegex,
    MaxLength,
    Refuses,
    TokenOverlap,
    ValidJson,
)


@dataclass(frozen=True)
class GoldenCase:
    """One input the system must handle, plus how to grade the response.

    `tags` let you slice results ("how are we doing on the safety subset?") and
    `weight` lets a critical case count for more than a cosmetic one.
    """

    id: str
    input: str
    assertions: Sequence[Assertion]
    tags: tuple[str, ...] = ()
    weight: float = 1.0
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.assertions:
            raise ValueError(f"case {self.id!r} has no assertions — it can never fail, so it tests nothing")
        if self.weight <= 0:
            raise ValueError(f"case {self.id!r} has non-positive weight {self.weight}")


@dataclass
class Suite:
    name: str
    cases: list[GoldenCase]

    def __iter__(self) -> Iterator[GoldenCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def filter_by_tag(self, tag: str) -> "Suite":
        return Suite(f"{self.name}[{tag}]", [c for c in self.cases if tag in c.tags])

    @property
    def tags(self) -> set[str]:
        return {t for c in self.cases for t in c.tags}


def resolve_callable(spec: str) -> Callable:
    """Resolve 'package.module:callable' to the callable itself.

    Used for the pieces of a suite that cannot be expressed as data — currently just
    an `llm_judge` grading function.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"expected 'module:function', got {spec!r}")
    module_name, fn_name = spec.rsplit(":", 1)
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"could not import {module_name!r}: {exc}") from exc
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"{module_name!r} has no callable named {fn_name!r}")
    return fn


# Maps the on-disk assertion `type` to its constructor. Adding an assertion type means
# adding it here; unknown types fail loudly at load time rather than silently scoring 0.
_ASSERTION_TYPES = {
    "equals": lambda cfg: Equals(expected=cfg["expected"]),
    "contains": lambda cfg: Contains(required=tuple(cfg["required"])),
    "regex": lambda cfg: MatchesRegex(pattern=cfg["pattern"]),
    "valid_json": lambda cfg: ValidJson(required_keys=tuple(cfg.get("required_keys", []))),
    "json_field": lambda cfg: JsonFieldEquals(field=cfg["field"], expected=cfg["expected"]),
    "token_overlap": lambda cfg: TokenOverlap(
        reference=cfg["reference"], threshold=float(cfg.get("threshold", 0.0))
    ),
    "max_length": lambda cfg: MaxLength(limit_chars=int(cfg["limit_chars"])),
    "refuses": lambda cfg: Refuses(),
    # The grader is a callable, so a suite file names one by import path rather than
    # inlining it. Resolved at load time so a bad path fails before the run starts
    # instead of after you have paid for every completion.
    "llm_judge": lambda cfg: LlmJudge(
        rubric=cfg["rubric"], judge_fn=resolve_callable(cfg["judge_fn"])
    ),
}


def _build_assertion(cfg: dict[str, Any]) -> Assertion:
    kind = cfg.get("type")
    if kind not in _ASSERTION_TYPES:
        known = ", ".join(sorted(_ASSERTION_TYPES))
        raise ValueError(f"unknown assertion type {kind!r}; known types: {known}")
    return _ASSERTION_TYPES[kind](cfg)


def load_suite(path: str | Path) -> Suite:
    """Load a suite from JSON. Raises on malformed cases so a typo in a golden file
    surfaces at load time instead of quietly shrinking your coverage."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for entry in raw["cases"]:
        cid = entry["id"]
        if cid in seen:
            raise ValueError(f"duplicate case id {cid!r} in {p}")
        seen.add(cid)
        cases.append(
            GoldenCase(
                id=cid,
                input=entry["input"],
                assertions=[_build_assertion(a) for a in entry["assertions"]],
                tags=tuple(entry.get("tags", [])),
                weight=float(entry.get("weight", 1.0)),
                context=entry.get("context", {}),
            )
        )
    return Suite(name=raw.get("name", p.stem), cases=cases)
