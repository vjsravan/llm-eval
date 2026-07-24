"""
llmeval — regression testing for LLM pipelines.

Treats prompts like code: a versioned golden suite, graded assertions, and a CI gate
that blocks a merge when output quality drops by a statistically real margin.

    from llmeval import load_suite, run_suite, compare_runs, terminal_report

    suite = load_suite("examples/customs_classification/suite.json")
    run = run_suite(suite, my_model_fn, label="prompt-v4")
    print(terminal_report(run))
"""

from .assertions import (
    Contains,
    Equals,
    JsonFieldEquals,
    LlmJudge,
    MatchesRegex,
    MaxLength,
    Refuses,
    Score,
    TokenOverlap,
    ValidJson,
)
from .dataset import GoldenCase, Suite, load_suite
from .report import html_report, markdown_report, terminal_report, write_html
from .runner import CaseResult, Run, run_suite, run_suite_async
from .stats import Comparison, bootstrap_paired, compare_runs, required_sample_size, wilson_interval

__version__ = "1.0.0"

__all__ = [
    "Contains", "Equals", "JsonFieldEquals", "LlmJudge", "MatchesRegex", "MaxLength",
    "Refuses", "Score", "TokenOverlap", "ValidJson",
    "GoldenCase", "Suite", "load_suite",
    "CaseResult", "Run", "run_suite", "run_suite_async",
    "Comparison", "bootstrap_paired", "compare_runs", "required_sample_size", "wilson_interval",
    "html_report", "markdown_report", "terminal_report", "write_html",
]
