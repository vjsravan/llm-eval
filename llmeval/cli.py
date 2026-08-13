"""
Command line entry point.

    llmeval run     suite.json --model my.module:fn --label prompt-v4 --save runs/candidate.json
    llmeval compare runs/baseline.json runs/candidate.json --min-effect 0.02 --fail-on regression
    llmeval report  runs/candidate.json --baseline runs/baseline.json --html report.html

`compare` returns exit code 1 on a regression, which is what makes it usable as a CI
gate — the workflow needs no scripting beyond calling this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import load_suite, resolve_callable
from .gate import DEFAULT_POLICY, GatePolicy
from .report import markdown_report, terminal_report, write_html
from .runner import Run, run_suite
from .stats import compare_runs


def _load_model_fn(spec: str):
    """Resolve 'package.module:callable' to the callable itself."""
    try:
        return resolve_callable(spec)
    except ValueError as exc:
        raise SystemExit(f"--model: {exc}") from exc


def _build_policy(args: argparse.Namespace) -> GatePolicy:
    """Assemble a policy from --require-tag flags, falling back to the default."""
    floors: dict[str, float] = {}
    for spec in getattr(args, "require_tag", None) or []:
        if "=" not in spec:
            raise SystemExit(f"--require-tag expects TAG=SCORE, got {spec!r}")
        tag, _, raw = spec.partition("=")
        try:
            floors[tag.strip()] = float(raw)
        except ValueError:
            raise SystemExit(f"--require-tag score must be a number, got {raw!r}") from None

    if not floors and getattr(args, "default_policy", False):
        return DEFAULT_POLICY

    return GatePolicy(
        tag_floors=floors,
        min_mean_score=getattr(args, "fail_under", None),
        max_regression=getattr(args, "min_effect", 0.02),
    )


def _cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    if args.tag:
        suite = suite.filter_by_tag(args.tag)
        if not len(suite):
            raise SystemExit(f"no cases carry tag {args.tag!r}")

    run = run_suite(
        suite,
        _load_model_fn(args.model),
        label=args.label,
        concurrency=args.concurrency,
        retries=args.retries,
        timeout_s=args.timeout,
    )

    comparison = None
    if args.baseline:
        comparison = compare_runs(Run.load(args.baseline).scores, run.scores)

    print(terminal_report(run, comparison))

    if args.save:
        print(f"  saved run → {run.save(args.save)}")
    if args.html:
        print(f"  saved html → {write_html(run, args.html, comparison)}")

    result = _build_policy(args).evaluate(run, comparison)
    print(result.render())
    print()
    return 0 if result.passed else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline, candidate = Run.load(args.baseline), Run.load(args.candidate)
    comparison = compare_runs(baseline.scores, candidate.scores)
    print(terminal_report(candidate, comparison))

    if args.markdown:
        Path(args.markdown).write_text(markdown_report(candidate, comparison), encoding="utf-8")
        print(f"  saved markdown → {args.markdown}")

    result = _build_policy(args).evaluate(candidate, comparison)
    print(result.render())
    print()

    if args.fail_on == "never":
        return 0
    # "any-change" means any statistically real movement, in either direction — the
    # mode exists for suites that are supposed to be frozen, where an unexplained
    # improvement is as much a signal that something shifted as a drop is.
    if args.fail_on == "any-change" and comparison.significant:
        print(
            f"  FAIL suite moved {comparison.delta:+.3f} (--fail-on any-change)",
            file=sys.stderr,
        )
        return 1
    return 0 if result.passed else 1


def _cmd_report(args: argparse.Namespace) -> int:
    run = Run.load(args.run)
    comparison = compare_runs(Run.load(args.baseline).scores, run.scores) if args.baseline else None
    print(terminal_report(run, comparison))
    if args.html:
        print(f"  saved html → {write_html(run, args.html, comparison)}")
    if args.markdown:
        Path(args.markdown).write_text(markdown_report(run, comparison), encoding="utf-8")
        print(f"  saved markdown → {args.markdown}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmeval", description="Regression testing for LLM pipelines")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="execute a suite against a model")
    p_run.add_argument("suite")
    p_run.add_argument("--model", required=True, help="module:function returning the completion")
    p_run.add_argument("--label", default="candidate", help="prompt version or git sha")
    p_run.add_argument("--baseline", help="prior run JSON to compare against")
    p_run.add_argument("--save", help="write the run JSON here")
    p_run.add_argument("--html", help="write an HTML report here")
    p_run.add_argument("--tag", help="only run cases carrying this tag")
    p_run.add_argument("--concurrency", type=int, default=8)
    p_run.add_argument("--retries", type=int, default=2)
    p_run.add_argument("--timeout", type=float, default=60.0)
    p_run.add_argument("--fail-under", type=float, help="exit 1 if mean score falls below this")
    p_run.add_argument("--require-tag", action="append", metavar="TAG=SCORE",
                       help="absolute floor for a tag, e.g. safety=1.0 (repeatable)")
    p_run.add_argument("--default-policy", action="store_true",
                       help="apply the built-in policy (safety and hallucination-guard at 1.0)")
    p_run.add_argument("--min-effect", type=float, default=0.02)
    p_run.set_defaults(func=_cmd_run)

    p_cmp = sub.add_parser("compare", help="compare two saved runs")
    p_cmp.add_argument("baseline")
    p_cmp.add_argument("candidate")
    p_cmp.add_argument("--min-effect", type=float, default=0.02,
                       help="deltas smaller than this are 'negligible' even if significant")
    p_cmp.add_argument("--fail-on", choices=["regression", "any-change", "never"], default="regression",
                       help="'regression' blocks on the gate policy; 'any-change' also blocks on a "
                            "significant improvement, for suites meant to be frozen; 'never' reports only")
    p_cmp.add_argument("--markdown", help="write a PR-ready markdown summary here")
    p_cmp.add_argument("--require-tag", action="append", metavar="TAG=SCORE",
                       help="absolute floor for a tag, e.g. safety=1.0 (repeatable)")
    p_cmp.add_argument("--default-policy", action="store_true",
                       help="apply the built-in policy (safety and hallucination-guard at 1.0)")
    p_cmp.set_defaults(func=_cmd_compare)

    p_rep = sub.add_parser("report", help="re-render a saved run")
    p_rep.add_argument("run")
    p_rep.add_argument("--baseline")
    p_rep.add_argument("--html")
    p_rep.add_argument("--markdown")
    p_rep.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
