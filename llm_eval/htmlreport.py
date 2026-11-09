"""Self-contained HTML report for a finished or partial run: one file, no network, light and dark.

Built from ``state.json`` and ``plan.json`` only. If ``analysis.html`` sits beside the report, its
contents are embedded under the results, so a person can add the reading of a run to the page.
"""
from html import escape
from pathlib import Path

CSS = """
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#6b6b66;--card:#fff;--line:#e2e2dc;--pass:#2e8b57;--part:#b57d0c;--fail:#c4453c;--pass-bg:#dff1e6;--fail-bg:#f8e0dd;--part-bg:#faefd3}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ececE6;--mut:#9a9a92;--card:#1f1f1c;--line:#34342f;--pass:#4fb581;--part:#e0ac3c;--fail:#e0675e;--pass-bg:#1d3a2b;--fail-bg:#442321;--part-bg:#40331a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
main{max-width:880px;margin:0 auto;padding:24px 16px 56px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}h3{font-size:15px;margin:20px 0 6px}
.sub,.note{color:var(--mut)}.sub{margin:0 0 20px}.note{font-size:13px;margin-top:8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.card b{display:block;font-size:22px}.card span{color:var(--mut);font-size:13px}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:8px 10px;text-align:center;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{font-size:12px;color:var(--mut);font-weight:600}td:first-child,th:first-child{text-align:left}
tr:last-child td{border-bottom:0}
.c{display:inline-block;min-width:3.2em;border-radius:6px;padding:0 6px;font-weight:600}
.p{background:var(--pass-bg);color:var(--pass)}.f{background:var(--fail-bg);color:var(--fail)}.m{background:var(--part-bg);color:var(--part)}
.sm{font-size:12px;color:var(--mut);font-weight:400;display:block}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden;min-width:70px}.bar i{display:block;height:100%}
.analysis{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:4px 18px 14px}
.analysis li{margin:4px 0}
"""

OK_STATES = ("pass", "capacity_fallback")


def _cls(status):
    return "p" if status in OK_STATES else "m" if status == "skipped" else "f"


def _cell(trial):
    if not trial:
        return '<span class="sm">not run</span>'
    grading = trial.get("grading") or {}
    if "correct_positions" in grading:
        text = f"{grading['correct_positions']}/{grading['expected_count']}"
        note = ""
        if grading["actual_count"] != grading["expected_count"]:
            note = f'<span class="sm">{grading["actual_count"]} returned</span>'
    else:
        text, note = escape(str(trial["status"])), ""
    return f'<span class="c {_cls(trial["status"])}">{text}</span>{note}'


def _tokens(n):
    return "n/a" if n is None else f"{n:,}"


def _model_html(model, sections):
    prefix = model["id"] + "__"
    own = {k[len(prefix):]: v for k, v in sections.items() if k.startswith(prefix)}
    trials = {k: v for k, v in own.items() if k.startswith("trial-")}
    named = {k: v for k, v in own.items() if not k.startswith("trial-")}

    def kind(prefix_):
        return [t for k, t in trials.items() if k.startswith("trial-" + prefix_)]

    recall, tools, smoke = named.get("long_context_recall", {}), named.get("structured_tools", {}), named.get("smoke", {})
    descent = {d["context"]: d for d in recall.get("descent") or []}
    anchor = descent.get(model["anchor"])
    passing = [t["prompt_tokens"] for t in kind("recall") if t["status"] == "pass"]
    peaks = {name: max((t["output_tokens"] for t in kind(key) if t.get("output_tokens") is not None), default=None)
             for name, key in (("smoke", "smoke"), ("recall", "recall"), ("tools", "tool"))}
    cards = [
        (smoke.get("status", "not run"), "smoke", _cls(smoke.get("status"))),
        (f"{tools.get('passed')}/{tools.get('total')}" if tools else "not run", "tools passed", _cls(tools.get("status"))),
        (f"{anchor['passed']}/{anchor['total']}" if anchor else "not run", f"recall at anchor {model['anchor']:,}",
         "p" if anchor and anchor["passed"] == anchor["total"] else "f"),
        (_tokens(recall.get("verified_context")), "verified recall context", ""),
        (_tokens(max(passing, default=None)), "largest passing recall prompt (tokens)", ""),
        (_tokens(peaks["recall"]), "peak recall output tokens", ""),
    ]
    thinking = _thinking_sent(model.get("workload") or {})
    if thinking:
        cards.append((thinking, "thinking sent (pi)", ""))
    out = [f"<h2>{escape(model['id'])}</h2>", '<div class="cards">']
    for value, label, cls in cards:
        color = {"p": "var(--pass)", "f": "var(--fail)"}.get(cls, "inherit")
        out.append(f'<div class="card"><b style="color:{color}">{escape(str(value))}</b><span>{escape(label)}</span></div>')
    out.append("</div>")

    by_context = {}
    for name, trial in trials.items():
        if name.startswith("trial-recall-"):
            by_context.setdefault(trial["context"], {})[int(name.split("-")[2])] = trial
    if by_context:
        count = max(len(v) for v in by_context.values())
        out += ["<h3>Recall by context</h3>", '<div class="wrap"><table>',
                "<tr><th>Context</th><th>Prompt tokens</th>" + "".join(f"<th>Trial {i}</th>" for i in range(1, count + 1))
                + "<th>Passed</th><th></th></tr>"]
        for context in sorted(by_context, reverse=True):
            row = by_context[context]
            passed = sum(1 for t in row.values() if t["status"] == "pass")
            share = round(100 * passed / len(row))
            color = "var(--pass)" if passed == len(row) else "var(--fail)" if not passed else "var(--part)"
            tag = ' <span class="sm">anchor</span>' if context == model["anchor"] else ""
            prompt = sum(t["prompt_tokens"] for t in row.values()) // len(row)
            out.append(f"<tr><td>{context:,}{tag}</td><td>~{prompt / 1000:.1f}K</td>"
                       + "".join(f"<td>{_cell(row.get(i))}</td>" for i in range(1, count + 1))
                       + f'<td>{passed} of {len(row)}</td><td><div class="bar"><i style="width:{share}%;background:{color}"></i></div></td></tr>')
        out += ["</table></div>", '<p class="note">Cells are records in the correct position out of those expected; a trial passes only when all are right. "Returned" is shown when the count differed from expected.</p>']
        out += ["<h3>Output tokens per recall trial</h3>", '<div class="wrap"><table>',
                "<tr><th>Context</th>" + "".join(f"<th>Trial {i}</th>" for i in range(1, count + 1)) + "</tr>"]
        for context in sorted(by_context, reverse=True):
            out.append(f"<tr><td>{context:,}</td>" + "".join(
                f"<td>{_tokens((by_context[context].get(i) or {}).get('output_tokens'))}</td>" for i in range(1, count + 1)) + "</tr>")
        capped = [t for t in kind("") if t.get("finish_reason") == "length"]
        out += ["</table></div>", f'<p class="note">{len(capped)} trial(s) ended on the output cap.</p>' if capped
                else '<p class="note">No trial ended on the output cap.</p>']

    out += _workload_html(named, trials)
    out += ["<h3>Sections</h3>", '<div class="wrap"><table><tr><th>Section</th><th>Status</th></tr>']
    for name in sorted(named):
        out.append(f'<tr><td>{escape(name)}</td><td><span class="c {_cls(named[name]["status"])}">{escape(str(named[name]["status"]))}</span></td></tr>')
    out.append("</table></div>")
    return "\n".join(out)


def _thinking_sent(workload):
    """What pi sends for thinking on every turn of this model's workload: the template switches and the
    thinking budget, as the plan's frozen pi entry resolves the plan's level. Empty without a pi workload."""
    entry, level = workload.get("production_entry"), workload.get("thinking")
    if not entry or not level:
        return ""
    from .workload import pi_config
    output = (workload.get("mode") or {}).get("output_limit") or entry.get("maxTokens") or 0
    try:
        fields = pi_config.thinking_request(entry, level, output)
    except ValueError:
        return ""
    sent = [f"{k}={v}" for k, v in (fields.get("chat_template_kwargs") or {}).items()]
    sent += [f"{k}={v}" for k, v in fields.items() if k != "chat_template_kwargs"]
    return ", ".join(sent) or "off"


def _cap_turns(runs):
    """Turns whose thinking reached the budget, summed over runs; None when no run recorded the count."""
    counts = [(r.get("reasoning_from_capture") or {}).get("cap_hits") for r in runs]
    counts = [c for c in counts if c is not None]
    return sum(counts) if counts else None


def _median(values):
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def _workload_html(named, trials):
    """Scorecard: one row per workload group that ran; pass rate with process numbers beside it."""
    groups = [(k.split("_", 1)[1], v) for k, v in sorted(named.items())
              if k.startswith("workload_") and v.get("status") != "skipped"]
    if not groups:
        return []
    out = ["<h3>Workload scorecard</h3>", '<div class="wrap"><table>',
           "<tr><th>Work type</th><th>Runs passed</th><th>Median steps</th><th>Median model time</th>"
           "<th>Peak context</th><th>Compactions</th><th>Thinking-cap turns</th><th>Clarifications</th><th>Web misses</th></tr>"]
    for group, section in groups:
        runs = [t for k, t in trials.items() if k.startswith("trial-wl-") and t.get("group") == group]
        metric = lambda name: [(r.get("metrics") or {}).get(name) for r in runs]
        seconds = _median(metric("model_seconds"))
        misses = sum(len((r.get("web") or {}).get("misses", [])) for r in runs)
        cls = "p" if section["status"] == "pass" else ("m" if section.get("passed") else "f")
        out.append(f"<tr><td>{escape(group)}</td>"
                   f'<td><span class="c {cls}">{section.get("passed", 0)}/{section.get("total", 0)}</span></td>'
                   f"<td>{_tokens(_median(metric('steps')))}</td>"
                   f"<td>{'n/a' if seconds is None else f'{seconds / 60:.1f} min'}</td>"
                   f"<td>{_tokens(max((m for m in metric('peak_context_tokens') if m is not None), default=None))}</td>"
                   f"<td>{sum(m or 0 for m in metric('compactions'))}</td>"
                   f"<td>{_tokens(_cap_turns(runs))}</td>"
                   f"<td>{sum(r.get('clarifications', 0) for r in runs)}</td><td>{misses}</td></tr>")
    out += ["</table></div>", '<p class="note">A run passes only when every code check passes. The process '
            "numbers are reported apart from pass or fail, so a slow correct run is not mistaken for an efficient one. "
            "Thinking-cap turns are model turns whose thinking reached the request's thinking budget, where the server "
            "forces the thinking closed (inferred from token counts; n/a for runs recorded before the count existed). "
            "Each failed run's checks are in its section file.</p>"]
    return out


def render(plan, state, analysis_html=""):
    restore = state.get("last_restore") or {}
    restored = "restoration required" if state.get("restoration_required") else (
        "production restored and verified" if restore.get("health") and restore.get("service_matches") else "restore not recorded")
    body = [f"<h1>{escape(', '.join(m['id'] for m in plan['models']))}</h1>",
            f'<p class="sub">run status {escape(str(state["status"]))} &middot; {restored} &middot; plan {escape(plan["plan_hash"][:12])}</p>']
    body += [_model_html(m, state["sections"]) for m in plan["models"]]
    if analysis_html:
        body += ["<h2>Analysis</h2>", f'<div class="analysis">{analysis_html}</div>']
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>llm-eval run report</title><style>{CSS}</style></head><body><main>{''.join(body)}</main></body></html>")


def write(run_dir, plan, state):
    """Write report.html into the run folder, embedding analysis.html when present. Returns the path."""
    run_dir = Path(run_dir)
    analysis = run_dir / "analysis.html"
    html = render(plan, state, analysis.read_text() if analysis.exists() else "")
    path = run_dir / "report.html"
    path.write_text(html)
    return path
