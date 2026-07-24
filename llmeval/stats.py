"""
Statistics for comparing two eval runs.

The reason this module exists: LLM outputs are noisy, so a raw score delta between
two runs is not evidence on its own. "Mean dropped from 0.91 to 0.86" might be a real
regression or might be sampling noise, and shipping on the wrong reading is expensive
in both directions — you either ignore a real quality drop or you block a good change
and lose trust in the gate.

Everything here is paired: each case is run under both the baseline and the candidate,
so we compare per-case differences rather than two independent means. Paired comparison
removes case difficulty as a source of variance and is far more sensitive on the small
suites (50-300 cases) that real teams actually maintain.

Pure standard library on purpose — an eval gate that needs a scientific-computing stack
installed is an eval gate that gets skipped in CI.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Comparison:
    """Result of comparing candidate against baseline over the same cases."""

    n: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    regressed_cases: tuple[str, ...]
    improved_cases: tuple[str, ...]

    @property
    def significant(self) -> bool:
        """True when the 95% interval excludes zero — i.e. the change is unlikely to
        be noise. Note this says nothing about whether the change *matters*; a
        statistically real drop of 0.002 is still not worth blocking a release over."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def verdict(self, min_effect: float = 0.02) -> str:
        """Combines significance with a practical floor, because those are different
        questions and conflating them is how eval gates become noise generators."""
        if not self.significant:
            return "inconclusive"
        if abs(self.delta) < min_effect:
            return "negligible"
        return "regression" if self.delta < 0 else "improvement"


def bootstrap_paired(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260724,
) -> tuple[float, float, float]:
    """Bootstrap a confidence interval for the paired mean difference.

    Resamples the per-case differences with replacement `iterations` times and reads
    the interval off the resulting distribution. This makes no normality assumption,
    which matters because eval scores are typically bounded, skewed, and spiky at 0
    and 1 — exactly where a t-test's assumptions are weakest.

    The seed is fixed so the same runs always produce the same verdict. A gate that
    flips between pass and fail on identical inputs will be disabled within a week.

    Returns (ci_low, ci_high, two_sided_p_value).
    """
    if len(baseline) != len(candidate):
        raise ValueError(f"paired comparison needs equal lengths, got {len(baseline)} and {len(candidate)}")
    if not baseline:
        raise ValueError("cannot compare empty runs")

    diffs = [c - b for b, c in zip(baseline, candidate)]
    n = len(diffs)
    rng = random.Random(seed)

    means: list[float] = []
    for _ in range(iterations):
        resample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    lo = means[max(0, int(tail * iterations) - 1)]
    hi = means[min(iterations - 1, int((1.0 - tail) * iterations))]

    # Two-sided achieved significance level: how often the resampled mean crosses zero.
    observed = sum(diffs) / n
    if observed == 0:
        p = 1.0
    else:
        crossings = sum(1 for m in means if (m <= 0) if observed > 0) or sum(
            1 for m in means if (m >= 0) if observed < 0
        )
        p = min(1.0, 2.0 * crossings / iterations)

    return lo, hi, p


def compare_runs(
    baseline_scores: dict[str, float],
    candidate_scores: dict[str, float],
    *,
    per_case_epsilon: float = 1e-9,
    iterations: int = 10_000,
) -> Comparison:
    """Compare two runs keyed by case id.

    Only cases present in both runs are compared; a case added alongside a prompt
    change has no baseline and would otherwise silently bias the delta.
    """
    shared = sorted(set(baseline_scores) & set(candidate_scores))
    if not shared:
        raise ValueError("no overlapping case ids between the two runs")

    base = [baseline_scores[k] for k in shared]
    cand = [candidate_scores[k] for k in shared]

    lo, hi, p = bootstrap_paired(base, cand, iterations=iterations)

    regressed = tuple(k for k in shared if candidate_scores[k] < baseline_scores[k] - per_case_epsilon)
    improved = tuple(k for k in shared if candidate_scores[k] > baseline_scores[k] + per_case_epsilon)

    base_mean = statistics.fmean(base)
    cand_mean = statistics.fmean(cand)

    return Comparison(
        n=len(shared),
        baseline_mean=round(base_mean, 4),
        candidate_mean=round(cand_mean, 4),
        delta=round(cand_mean - base_mean, 4),
        ci_low=round(lo, 4),
        ci_high=round(hi, 4),
        p_value=round(p, 4),
        regressed_cases=regressed,
        improved_cases=improved,
    )


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a pass rate.

    Preferred over the textbook normal approximation because eval suites are small and
    pass rates cluster near 1.0, where the normal interval produces upper bounds above
    100% and understates uncertainty exactly when you care most.
    """
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def required_sample_size(effect: float, sd: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """How many cases you need to reliably detect a score drop of `effect`.

    Useful as a sanity check before trusting a gate: teams routinely try to detect a
    0.02 regression with 30 cases, which is underpowered by roughly an order of
    magnitude, and then conclude "evals don't work" when the gate stays silent.
    """
    if effect <= 0 or sd <= 0:
        raise ValueError("effect and sd must be positive")
    z_alpha = 1.959963985 if abs(alpha - 0.05) < 1e-9 else _z_from_alpha(alpha)
    z_beta = 0.841621234 if abs(power - 0.8) < 1e-9 else _z_from_alpha(2 * (1 - power))
    return math.ceil(((z_alpha + z_beta) * sd / effect) ** 2)


def _z_from_alpha(alpha: float) -> float:
    """Inverse normal CDF at 1 - alpha/2 (Acklam's rational approximation)."""
    p = 1.0 - alpha / 2.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
