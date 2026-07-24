"""
Tests for llmeval.

Bias here is toward the failure modes that make an eval harness untrustworthy:
non-determinism, silently-passing assertions, and a gate that reports the wrong verdict.
A flaky eval tool is worse than none, because it launders noise as evidence.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from llmeval import (
    Contains,
    Equals,
    GoldenCase,
    JsonFieldEquals,
    MaxLength,
    Refuses,
    Suite,
    TokenOverlap,
    ValidJson,
    bootstrap_paired,
    compare_runs,
    load_suite,
    required_sample_size,
    run_suite,
    wilson_interval,
)
from llmeval.gate import GatePolicy
from llmeval.report import html_report, markdown_report, terminal_report
from llmeval.runner import Run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "customs_classification" / "suite.json"


class TestAssertions(unittest.TestCase):
    def test_equals_normalises_whitespace_and_case(self):
        self.assertEqual(Equals("Hold Shipment").score("  hold   SHIPMENT ", None).value, 1.0)

    def test_contains_gives_partial_credit(self):
        a = Contains(required=("alpha", "beta", "gamma", "delta"))
        self.assertEqual(a.score("alpha beta gamma", None).value, 0.75)

    def test_contains_names_what_is_missing(self):
        reason = Contains(required=("alpha", "zulu")).score("alpha only", None).reason
        self.assertIn("zulu", reason)

    def test_valid_json_tolerates_markdown_fences(self):
        out = '```json\n{"disposition": "hold"}\n```'
        self.assertEqual(ValidJson(required_keys=("disposition",)).score(out, None).value, 1.0)

    def test_valid_json_partial_credit_for_missing_keys(self):
        out = '{"disposition": "hold"}'
        score = ValidJson(required_keys=("disposition", "action")).score(out, None)
        self.assertEqual(score.value, 0.75)  # 0.5 validity + 0.5 * (1 of 2 keys)

    def test_invalid_json_scores_zero_and_explains(self):
        score = ValidJson().score("not json at all", None)
        self.assertEqual(score.value, 0.0)
        self.assertIn("invalid JSON", score.reason)

    def test_json_field_equals_is_case_insensitive(self):
        out = '{"disposition": "HOLD"}'
        self.assertEqual(JsonFieldEquals("disposition", "hold").score(out, None).value, 1.0)

    def test_json_field_equals_handles_booleans(self):
        out = '{"blocking": true}'
        self.assertEqual(JsonFieldEquals("blocking", True).score(out, None).value, 1.0)
        self.assertEqual(JsonFieldEquals("blocking", False).score(out, None).value, 0.0)

    def test_refuses_detects_decline(self):
        self.assertEqual(Refuses().score("I can't do that.", None).value, 1.0)
        self.assertEqual(Refuses().score("Sure, done!", None).value, 0.0)

    def test_max_length_decays_smoothly(self):
        a = MaxLength(limit_chars=100)
        self.assertEqual(a.score("x" * 100, None).value, 1.0)
        slight = a.score("x" * 110, None).value
        severe = a.score("x" * 400, None).value
        self.assertLess(slight, 1.0)
        self.assertLess(severe, slight)
        self.assertGreater(slight, 0.8, "a 10% overrun should not be catastrophic")

    def test_token_overlap_ignores_stopwords(self):
        a = TokenOverlap(reference="the shipment is held for a missing document")
        high = a.score("shipment held, document missing", None).value
        self.assertGreater(high, 0.6)

    def test_token_overlap_threshold_zeroes_weak_matches(self):
        a = TokenOverlap(reference="customs clearance document", threshold=0.9)
        self.assertEqual(a.score("completely unrelated text", None).value, 0.0)

    def test_score_rejects_out_of_range(self):
        from llmeval import Score
        with self.assertRaises(ValueError):
            Score(1.5, "impossible")


class TestDataset(unittest.TestCase):
    def test_example_suite_loads(self):
        suite = load_suite(EXAMPLE)
        self.assertEqual(len(suite), 10)
        self.assertIn("safety", suite.tags)

    def test_case_without_assertions_is_rejected(self):
        with self.assertRaises(ValueError):
            GoldenCase(id="empty", input="x", assertions=[])

    def test_duplicate_case_ids_are_rejected(self, ):
        import tempfile
        payload = {"name": "dup", "cases": [
            {"id": "same", "input": "a", "assertions": [{"type": "equals", "expected": "a"}]},
            {"id": "same", "input": "b", "assertions": [{"type": "equals", "expected": "b"}]},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(payload, fh)
            path = fh.name
        with self.assertRaises(ValueError) as ctx:
            load_suite(path)
        self.assertIn("duplicate", str(ctx.exception))

    def test_unknown_assertion_type_fails_loudly(self):
        import tempfile
        payload = {"name": "bad", "cases": [
            {"id": "x", "input": "a", "assertions": [{"type": "telepathy"}]},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(payload, fh)
            path = fh.name
        with self.assertRaises(ValueError) as ctx:
            load_suite(path)
        self.assertIn("telepathy", str(ctx.exception))

    def test_filter_by_tag(self):
        suite = load_suite(EXAMPLE)
        safety = suite.filter_by_tag("safety")
        self.assertTrue(len(safety) >= 2)
        self.assertTrue(all("safety" in c.tags for c in safety))


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.suite = Suite("t", [
            GoldenCase("good", "in", [Equals("yes")]),
            GoldenCase("bad", "in", [Equals("no")]),
        ])

    def test_case_score_is_the_weakest_assertion(self):
        """A passing format check must not mask a wrong answer."""
        suite = Suite("t", [GoldenCase("c", "in", [ValidJson(), JsonFieldEquals("k", "expected")])])
        run = run_suite(suite, lambda p: '{"k": "wrong"}')
        self.assertEqual(run.results[0].score, 0.0)
        self.assertEqual(run.results[0].assertion_scores["valid_json"], 1.0)

    def test_model_exception_is_recorded_not_raised(self):
        def broken(prompt: str) -> str:
            raise RuntimeError("provider exploded")

        run = run_suite(self.suite, broken, retries=0)
        self.assertEqual(len(run.results), 2, "a failing model must still produce results")
        self.assertEqual(run.error_count, 2)
        self.assertIn("provider exploded", run.results[0].error)

    def test_partial_failure_preserves_good_results(self):
        calls = {"n": 0}

        def flaky(prompt: str) -> str:
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("intermittent")
            return "yes"

        run = run_suite(self.suite, flaky, retries=0, concurrency=1)
        self.assertEqual(len(run.results), 2)

    def test_retries_eventually_succeed(self):
        attempts = {"n": 0}

        def recovers(prompt: str) -> str:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("transient")
            return "yes"

        suite = Suite("t", [GoldenCase("c", "in", [Equals("yes")])])
        run = run_suite(suite, recovers, retries=3)
        self.assertEqual(run.results[0].score, 1.0)
        self.assertIsNone(run.results[0].error)

    def test_weighted_mean_respects_weights(self):
        suite = Suite("t", [
            GoldenCase("heavy", "in", [Equals("no")], weight=9.0),
            GoldenCase("light", "in", [Equals("yes")], weight=1.0),
        ])
        run = run_suite(suite, lambda p: "yes")
        self.assertAlmostEqual(run.mean_score, 0.5)
        self.assertAlmostEqual(run.weighted_mean, 0.1, places=6)

    def test_run_roundtrips_through_json(self):
        import tempfile
        run = run_suite(self.suite, lambda p: "yes")
        path = Path(tempfile.mkdtemp()) / "run.json"
        run.save(path)
        loaded = Run.load(path)
        self.assertEqual(loaded.scores, run.scores)
        self.assertEqual(loaded.run_id, run.run_id)

    def test_by_tag_aggregation(self):
        suite = Suite("t", [
            GoldenCase("a", "in", [Equals("yes")], tags=("x",)),
            GoldenCase("b", "in", [Equals("no")], tags=("x",)),
            GoldenCase("c", "in", [Equals("yes")], tags=("y",)),
        ])
        run = run_suite(suite, lambda p: "yes")
        self.assertAlmostEqual(run.by_tag()["x"], 0.5)
        self.assertAlmostEqual(run.by_tag()["y"], 1.0)


class TestStats(unittest.TestCase):
    def test_bootstrap_is_deterministic(self):
        base = [0.9] * 40
        cand = [0.7] * 40
        self.assertEqual(bootstrap_paired(base, cand), bootstrap_paired(base, cand))

    def test_identical_runs_are_inconclusive(self):
        scores = {f"c{i}": 0.8 for i in range(30)}
        cmp = compare_runs(scores, dict(scores))
        self.assertEqual(cmp.delta, 0.0)
        self.assertFalse(cmp.significant)
        self.assertEqual(cmp.verdict(), "inconclusive")

    def test_large_consistent_drop_is_flagged(self):
        base = {f"c{i}": 1.0 for i in range(60)}
        cand = {f"c{i}": 0.5 for i in range(60)}
        cmp = compare_runs(base, cand)
        self.assertTrue(cmp.significant)
        self.assertEqual(cmp.verdict(), "regression")
        self.assertEqual(len(cmp.regressed_cases), 60)

    def test_significant_but_tiny_delta_is_negligible(self):
        base = {f"c{i}": 0.500 for i in range(200)}
        cand = {f"c{i}": 0.505 for i in range(200)}
        cmp = compare_runs(base, cand)
        self.assertEqual(cmp.verdict(min_effect=0.02), "negligible")

    def test_only_shared_cases_are_compared(self):
        cmp = compare_runs({"a": 1.0, "b": 1.0}, {"a": 0.0, "c": 1.0})
        self.assertEqual(cmp.n, 1)

    def test_no_overlap_raises(self):
        with self.assertRaises(ValueError):
            compare_runs({"a": 1.0}, {"b": 1.0})

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            bootstrap_paired([1.0, 0.5], [1.0])

    def test_wilson_interval_stays_in_bounds(self):
        lo, hi = wilson_interval(10, 10)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, 1.0, "a perfect run on 10 cases is not certainty")

    def test_required_sample_size_grows_as_effect_shrinks(self):
        self.assertGreater(required_sample_size(0.01, 0.2), required_sample_size(0.10, 0.2))


class TestGate(unittest.TestCase):
    def _run_with(self, score_map: dict[str, float], tags: dict[str, tuple[str, ...]]) -> Run:
        cases = [
            GoldenCase(cid, "in", [Equals("yes" if s >= 1.0 else "no")], tags=tags.get(cid, ()))
            for cid, s in score_map.items()
        ]
        return run_suite(Suite("t", cases), lambda p: "yes")

    def test_safety_floor_blocks_even_when_stats_inconclusive(self):
        """The central design claim: a safety failure blocks on its own authority."""
        run = self._run_with({"safe": 0.0, "other": 1.0}, {"safe": ("safety",)})
        policy = GatePolicy(tag_floors={"safety": 1.0})
        result = policy.evaluate(run, comparison=None)
        self.assertFalse(result.passed)
        self.assertIn("safety", result.failures[0])

    def test_clean_run_passes(self):
        run = self._run_with({"a": 1.0, "b": 1.0}, {"a": ("safety",)})
        self.assertTrue(GatePolicy(tag_floors={"safety": 1.0}).evaluate(run).passed)

    def test_missing_tag_warns_rather_than_silently_passing(self):
        run = self._run_with({"a": 1.0}, {})
        result = GatePolicy(tag_floors={"safety": 1.0}).evaluate(run)
        self.assertTrue(result.passed)
        self.assertTrue(any("safety" in w for w in result.warnings))

    def test_underpowered_drop_produces_a_warning(self):
        base = {f"c{i}": 1.0 for i in range(6)}
        cand = {f"c{i}": 0.5 for i in range(6)}
        cmp = compare_runs(base, cand)
        run = self._run_with({f"c{i}": 1.0 for i in range(6)}, {})
        result = GatePolicy().evaluate(run, cmp)
        if not cmp.significant:
            self.assertTrue(any("interval includes zero" in w for w in result.warnings))

    def test_min_mean_score_floor(self):
        run = self._run_with({"a": 1.0, "b": 0.0}, {})
        self.assertFalse(GatePolicy(min_mean_score=0.9).evaluate(run).passed)


class TestReports(unittest.TestCase):
    def setUp(self):
        suite = Suite("t", [
            GoldenCase("pass", "in", [Equals("yes")], tags=("alpha",)),
            GoldenCase("fail", "in", [Equals("no")], tags=("beta",)),
        ])
        self.run = run_suite(suite, lambda p: "yes")
        self.cmp = compare_runs({"pass": 1.0, "fail": 1.0}, self.run.scores)

    def test_terminal_report_mentions_failing_case(self):
        self.assertIn("fail", terminal_report(self.run, self.cmp))

    def test_markdown_report_is_valid_table(self):
        md = markdown_report(self.run, self.cmp)
        self.assertIn("| metric | value |", md)
        # Two cases cannot support a significance claim, so the honest verdict here is
        # "inconclusive" — but the regressed case must still be listed for the reviewer.
        self.assertIn("inconclusive", md.lower())
        self.assertIn("`fail`", md)

    def test_html_report_escapes_and_is_self_contained(self):
        suite = Suite("t", [GoldenCase("<script>x</script>", "in", [Equals("yes")])])
        run = run_suite(suite, lambda p: "yes")
        html = html_report(run)
        self.assertNotIn("<script>x</script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("http://", html.split("<style>")[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
