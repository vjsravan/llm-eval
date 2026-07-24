"""
Assertions score a single model output against a golden case.

Every assertion returns a Score in [0.0, 1.0] plus a human-readable reason, rather
than a bare pass/fail. Graded scores matter because LLM output degrades continuously:
a prompt change that drops mean similarity from 0.94 to 0.71 has regressed badly while
still "passing" every boolean check.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class Score:
    value: float
    reason: str

    @property
    def passed(self) -> bool:
        return self.value >= 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"score must be in [0,1], got {self.value}")


class Assertion(Protocol):
    name: str

    def score(self, output: str, case: "GoldenCase") -> Score: ...  # noqa: F821


@dataclass(frozen=True)
class Equals:
    """Exact match after normalising whitespace and case."""

    expected: str
    name: str = "equals"

    def score(self, output: str, case: Any) -> Score:
        norm = lambda s: re.sub(r"\s+", " ", s).strip().casefold()
        ok = norm(output) == norm(self.expected)
        return Score(1.0 if ok else 0.0, "exact match" if ok else f"expected {self.expected!r}, got {output!r}")


@dataclass(frozen=True)
class Contains:
    """All required substrings must appear (case-insensitive). Partial credit is
    proportional, so losing one of four required fields scores 0.75, not 0."""

    required: tuple[str, ...]
    name: str = "contains"

    def score(self, output: str, case: Any) -> Score:
        low = output.casefold()
        hits = [s for s in self.required if s.casefold() in low]
        missing = [s for s in self.required if s.casefold() not in low]
        val = len(hits) / len(self.required) if self.required else 1.0
        reason = "all present" if not missing else f"missing: {', '.join(missing)}"
        return Score(val, reason)


@dataclass(frozen=True)
class Refuses:
    """Asserts the model declined. Used for safety cases where compliance is failure."""

    name: str = "refuses"
    markers: tuple[str, ...] = (
        "i can't", "i cannot", "i won't", "i'm not able", "unable to",
        "can't help", "cannot help", "not appropriate", "i don't have",
    )

    def score(self, output: str, case: Any) -> Score:
        low = output.casefold()
        hit = next((m for m in self.markers if m in low), None)
        return Score(1.0 if hit else 0.0, f"declined ({hit!r})" if hit else "did not decline")


@dataclass(frozen=True)
class MatchesRegex:
    pattern: str
    name: str = "regex"

    def score(self, output: str, case: Any) -> Score:
        ok = re.search(self.pattern, output, re.IGNORECASE | re.DOTALL) is not None
        return Score(1.0 if ok else 0.0, "matched" if ok else f"no match for /{self.pattern}/")


@dataclass(frozen=True)
class ValidJson:
    """Output must parse as JSON and contain the required keys.

    Tolerates the ```json fences models habitually add, because rejecting output for
    formatting noise measures your parser, not the model.
    """

    required_keys: tuple[str, ...] = ()
    name: str = "valid_json"

    @staticmethod
    def _strip_fences(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        return (fenced.group(1) if fenced else text).strip()

    def score(self, output: str, case: Any) -> Score:
        try:
            parsed = json.loads(self._strip_fences(output))
        except json.JSONDecodeError as exc:
            return Score(0.0, f"invalid JSON: {exc.msg} at pos {exc.pos}")
        if not isinstance(parsed, dict):
            return Score(0.5, f"parsed as {type(parsed).__name__}, expected object")
        if not self.required_keys:
            return Score(1.0, "valid JSON")
        present = [k for k in self.required_keys if k in parsed]
        missing = [k for k in self.required_keys if k not in parsed]
        # JSON validity is worth half; required keys the other half.
        val = 0.5 + 0.5 * (len(present) / len(self.required_keys))
        return Score(val, "valid, all keys present" if not missing else f"missing keys: {', '.join(missing)}")


@dataclass(frozen=True)
class JsonFieldEquals:
    """Compares one field inside a JSON response. This is the workhorse for
    classification and extraction tasks."""

    field: str
    expected: Any
    name: str = "json_field"

    def score(self, output: str, case: Any) -> Score:
        try:
            parsed = json.loads(ValidJson._strip_fences(output))
        except json.JSONDecodeError:
            return Score(0.0, "output was not valid JSON")
        if not isinstance(parsed, dict) or self.field not in parsed:
            return Score(0.0, f"field {self.field!r} absent")
        actual = parsed[self.field]
        norm = lambda v: v.strip().casefold() if isinstance(v, str) else v
        ok = norm(actual) == norm(self.expected)
        return Score(1.0 if ok else 0.0, "match" if ok else f"{self.field}={actual!r}, expected {self.expected!r}")


@dataclass(frozen=True)
class TokenOverlap:
    """Bag-of-words F1 against a reference answer.

    A deliberately cheap, dependency-free stand-in for embedding similarity: it needs
    no model call, so it runs identically in CI and offline. Swap in a real embedding
    distance when you need semantic rather than lexical agreement.
    """

    reference: str
    threshold: float = 0.0
    name: str = "token_overlap"

    _STOP = frozenset(
        "a an the is are was were be been being of to in on for with and or as at by from it its this that".split()
    )

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", text.casefold())
        return {w for w in words if w not in cls._STOP}

    def score(self, output: str, case: Any) -> Score:
        got, want = self._tokens(output), self._tokens(self.reference)
        if not want:
            return Score(1.0, "empty reference")
        if not got:
            return Score(0.0, "empty output")
        overlap = len(got & want)
        precision = overlap / len(got)
        recall = overlap / len(want)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        val = f1 if f1 >= self.threshold else 0.0
        return Score(round(val, 4), f"F1={f1:.3f} (p={precision:.2f} r={recall:.2f})")


@dataclass(frozen=True)
class MaxLength:
    """Guards against the model padding its answer. Scores down smoothly past the cap
    instead of cliff-failing, so a 5% overrun is not treated like a 500% one."""

    limit_chars: int
    name: str = "max_length"

    def score(self, output: str, case: Any) -> Score:
        n = len(output)
        if n <= self.limit_chars:
            return Score(1.0, f"{n} <= {self.limit_chars} chars")
        overrun = (n - self.limit_chars) / self.limit_chars
        return Score(round(max(0.0, math.exp(-overrun)), 4), f"{n} chars, {overrun:.0%} over limit")


@dataclass(frozen=True)
class LlmJudge:
    """Delegates grading to a model, for cases where correctness is not mechanically
    checkable. Kept last and used sparingly: it is the slowest, priciest, and least
    reproducible assertion, and a drifting judge silently rewrites your baseline.

    `judge_fn` receives (rubric, output) and returns a float in [0,1].
    """

    rubric: str
    judge_fn: Callable[[str, str], float]
    name: str = "llm_judge"

    def score(self, output: str, case: Any) -> Score:
        raw = self.judge_fn(self.rubric, output)
        val = min(1.0, max(0.0, float(raw)))
        return Score(round(val, 4), f"judge scored {val:.2f} against rubric")
