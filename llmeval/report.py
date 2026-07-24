"""
Renders a Run (and optionally a baseline comparison) as terminal text, Markdown for a
PR comment, or a standalone HTML page.

The audience for each differs: the terminal output is for the engineer who just ran it,
the Markdown is for the reviewer who did not, and the HTML is for whoever is asked
"is the new prompt actually better?" three weeks later.
"""

from __future__ import annotations

import html
from pathlib import Path

from .runner import Run
from .stats import Comparison, wilson_interval


def _bar(value: float, width: int = 24) -> str:
    filled = round(value * width)
    return "█" * filled + "·" * (width - filled)


def terminal_report(run: Run, comparison: Comparison | None = None) -> str:
    lines: list[str] = []
    lines.append(f"\n  {run.suite}  ·  {run.label}  ·  {len(run.results)} cases")
    lines.append("  " + "─" * 62)

    lo, hi = wilson_interval(sum(1 for r in run.results if r.score >= 1.0), len(run.results))
    lines.append(f"  mean score     {run.mean_score:.3f}   {_bar(run.mean_score)}")
    lines.append(f"  weighted mean  {run.weighted_mean:.3f}")
    lines.append(f"  pass rate      {run.pass_rate:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    lines.append(f"  p95 latency    {run.p95_latency_ms:.0f} ms")
    if run.error_count:
        lines.append(f"  errors         {run.error_count}")

    by_tag = run.by_tag()
    if by_tag:
        lines.append("")
        lines.append("  by tag")
        for tag, score in sorted(by_tag.items(), key=lambda kv: kv[1]):
            lines.append(f"    {tag:<22} {score:.3f}  {_bar(score, 18)}")

    worst = sorted((r for r in run.results if r.score < 1.0), key=lambda r: r.score)[:8]
    if worst:
        lines.append("")
        lines.append("  lowest-scoring cases")
        for r in worst:
            reason = r.error or next(
                (v for k, v in r.reasons.items() if r.assertion_scores.get(k, 1.0) < 1.0),
                "",
            )
            lines.append(f"    {r.score:.2f}  {r.case_id:<28} {reason[:44]}")

    if comparison is not None:
        lines.append("")
        lines.append("  vs baseline")
        lines.append("  " + "─" * 62)
        verdict = comparison.verdict()
        arrow = "▲" if comparison.delta > 0 else "▼" if comparison.delta < 0 else "="
        lines.append(
            f"  {comparison.baseline_mean:.3f} → {comparison.candidate_mean:.3f}  "
            f"{arrow} {comparison.delta:+.3f}"
        )
        lines.append(f"  95% CI [{comparison.ci_low:+.3f}, {comparison.ci_high:+.3f}]   p={comparison.p_value:.4f}")
        lines.append(f"  verdict: {verdict.upper()}   ({comparison.n} shared cases)")
        if comparison.regressed_cases:
            shown = ", ".join(comparison.regressed_cases[:6])
            more = f" +{len(comparison.regressed_cases) - 6} more" if len(comparison.regressed_cases) > 6 else ""
            lines.append(f"  regressed: {shown}{more}")
        if comparison.improved_cases:
            lines.append(f"  improved:  {len(comparison.improved_cases)} cases")

    lines.append("")
    return "\n".join(lines)


def markdown_report(run: Run, comparison: Comparison | None = None) -> str:
    md: list[str] = []
    md.append(f"### Eval — `{run.suite}` @ `{run.label}`")
    md.append("")
    md.append("| metric | value |")
    md.append("| --- | --- |")
    md.append(f"| mean score | **{run.mean_score:.3f}** |")
    md.append(f"| weighted mean | {run.weighted_mean:.3f} |")
    md.append(f"| pass rate | {run.pass_rate:.1%} ({len(run.results)} cases) |")
    md.append(f"| p95 latency | {run.p95_latency_ms:.0f} ms |")
    if run.error_count:
        md.append(f"| errors | {run.error_count} |")
    md.append("")

    if comparison is not None:
        verdict = comparison.verdict()
        badge = {"regression": "🔴 regression", "improvement": "🟢 improvement",
                 "negligible": "⚪ negligible", "inconclusive": "⚪ inconclusive"}[verdict]
        md.append(f"**{badge}** — {comparison.baseline_mean:.3f} → {comparison.candidate_mean:.3f} "
                  f"({comparison.delta:+.3f}), 95% CI [{comparison.ci_low:+.3f}, {comparison.ci_high:+.3f}], "
                  f"p={comparison.p_value:.4f}")
        md.append("")
        if comparison.regressed_cases:
            md.append("<details><summary>Regressed cases</summary>")
            md.append("")
            for cid in comparison.regressed_cases:
                r = next((x for x in run.results if x.case_id == cid), None)
                reason = ""
                if r:
                    reason = r.error or next(
                        (v for k, v in r.reasons.items() if r.assertion_scores.get(k, 1.0) < 1.0), ""
                    )
                md.append(f"- `{cid}` — {reason}")
            md.append("")
            md.append("</details>")
    return "\n".join(md)


_HTML_CSS = """
:root { color-scheme: light dark; --fg:#111; --dim:#666; --line:#e3e3e6; --bg:#fff;
        --good:#0a7f43; --bad:#c0392b; --accent:#2b6cb0; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8ea; --dim:#9a9aa2; --line:#2a2a30; --bg:#141417; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2.5rem 1.5rem; background:var(--bg); color:var(--fg);
       font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width: 900px; margin: 0 auto; }
h1 { font-size:1.4rem; margin:0 0 .25rem; letter-spacing:-.01em; }
.sub { color:var(--dim); font-size:.85rem; margin-bottom:2rem; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.75rem; margin-bottom:2rem; }
.card { border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem; }
.card .k { font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; color:var(--dim); }
.card .v { font-size:1.6rem; font-weight:650; margin-top:.15rem; font-variant-numeric:tabular-nums; }
.verdict { padding:.8rem 1rem; border-radius:10px; margin-bottom:2rem; font-weight:600;
           border:1px solid var(--line); }
.verdict.regression { color:var(--bad); border-color:var(--bad); }
.verdict.improvement { color:var(--good); border-color:var(--good); }
.verdict .detail { font-weight:400; color:var(--dim); font-size:.85rem; margin-top:.3rem;
                   font-variant-numeric:tabular-nums; }
h2 { font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; color:var(--dim);
     margin:2rem 0 .6rem; }
.tablewrap { overflow-x:auto; border:1px solid var(--line); border-radius:10px; }
table { border-collapse:collapse; width:100%; font-size:.85rem; }
th,td { text-align:left; padding:.55rem .8rem; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-weight:600; color:var(--dim); font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; }
tr:last-child td { border-bottom:none; }
td.score { font-variant-numeric:tabular-nums; font-weight:600; width:1%; white-space:nowrap; }
td.score.fail { color:var(--bad); }
td.score.pass { color:var(--good); }
code { font:.82em ui-monospace,"SF Mono",Menlo,monospace; background:color-mix(in srgb,var(--fg) 7%,transparent);
       padding:.1em .35em; border-radius:4px; }
.meter { height:6px; border-radius:3px; background:color-mix(in srgb,var(--fg) 10%,transparent); overflow:hidden; }
.meter > i { display:block; height:100%; background:var(--accent); }
"""


def html_report(run: Run, comparison: Comparison | None = None) -> str:
    e = html.escape

    cards = [
        ("mean score", f"{run.mean_score:.3f}"),
        ("pass rate", f"{run.pass_rate:.0%}"),
        ("cases", str(len(run.results))),
        ("p95 latency", f"{run.p95_latency_ms:.0f} ms"),
    ]
    if run.error_count:
        cards.append(("errors", str(run.error_count)))

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Eval — {e(run.suite)} @ {e(run.label)}</title>",
        f"<style>{_HTML_CSS}</style></head><body><div class='wrap'>",
        f"<h1>{e(run.suite)}</h1>",
        f"<div class='sub'>{e(run.label)} · run <code>{e(run.run_id)}</code> · {e(run.created_at)}</div>",
        "<div class='cards'>",
        *[f"<div class='card'><div class='k'>{e(k)}</div><div class='v'>{e(v)}</div></div>" for k, v in cards],
        "</div>",
    ]

    if comparison is not None:
        verdict = comparison.verdict()
        parts.append(
            f"<div class='verdict {verdict}'>"
            f"{verdict.upper()} — {comparison.baseline_mean:.3f} → {comparison.candidate_mean:.3f} "
            f"({comparison.delta:+.3f})"
            f"<div class='detail'>95% CI [{comparison.ci_low:+.3f}, {comparison.ci_high:+.3f}] · "
            f"p={comparison.p_value:.4f} · {comparison.n} shared cases · "
            f"{len(comparison.regressed_cases)} regressed, {len(comparison.improved_cases)} improved</div>"
            f"</div>"
        )

    by_tag = run.by_tag()
    if by_tag:
        parts.append("<h2>Score by tag</h2><div class='tablewrap'><table><tr><th>tag</th><th>score</th><th></th></tr>")
        for tag, score in sorted(by_tag.items(), key=lambda kv: kv[1]):
            parts.append(
                f"<tr><td><code>{e(tag)}</code></td>"
                f"<td class='score'>{score:.3f}</td>"
                f"<td><div class='meter'><i style='width:{score*100:.1f}%'></i></div></td></tr>"
            )
        parts.append("</table></div>")

    parts.append("<h2>Cases</h2><div class='tablewrap'><table><tr><th>score</th><th>case</th><th>detail</th></tr>")
    for r in sorted(run.results, key=lambda r: r.score):
        cls = "pass" if r.score >= 1.0 else "fail"
        reason = r.error or next(
            (v for k, v in r.reasons.items() if r.assertion_scores.get(k, 1.0) < 1.0), "all assertions held"
        )
        parts.append(
            f"<tr><td class='score {cls}'>{r.score:.2f}</td>"
            f"<td><code>{e(r.case_id)}</code></td>"
            f"<td>{e(reason)}</td></tr>"
        )
    parts.append("</table></div></div></body></html>")
    return "".join(parts)


def write_html(run: Run, path: str | Path, comparison: Comparison | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_report(run, comparison), encoding="utf-8")
    return p
