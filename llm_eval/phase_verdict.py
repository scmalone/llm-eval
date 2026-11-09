"""A finished phase's verdict and summary line, read from its sections.

Kept out of ``tools/run_test_plan.py`` on purpose: that file is restore code (``RESTORE_CODE``), so any byte changed
in it needs a fresh signal drill. Nothing here starts, stops or signals anything; it only reads a finished run's
records, so its wording can change without a drill (DECISIONS 2026-11-01).
"""


def verdict(summary, sections, expect=None, planned=(), target=None):
    """(status, one line) for a finished run. Only a ``complete`` run with every expectation met can pass.

    ``complete``: every trial passed. ``quality_fail``: the phase measured what it promised and some trials
    failed (a result). ``assert_fail``: the phase did not measure what it promised (wrong context, missing
    trials, recall not verified at the target). ``infra_fail``: the run stopped or a section failed on the
    harness or the machine. ``blocked``: nothing ran."""
    status = summary.get("status")
    if status == "blocked":
        return "blocked", summary.get("next") or "blocked"
    if status != "complete":
        return "infra_fail", f"run status {status!r}: {str(summary.get('failure') or summary.get('next') or '')[:240]}"
    infra = [k for k, v in sections.items() if v.get("status") in ("infra_fail", "capacity_fail")]
    if infra:
        return "infra_fail", ", ".join(f"{k}={sections[k]['status']}" for k in infra[:6])
    problems = assess(sections, expect or {}, planned, target)
    if problems:
        return "assert_fail", "; ".join(problems[:6])
    trials = {k: v for k, v in sections.items() if "__trial-" in k}
    bad = [k for k, v in trials.items() if v.get("status") != "pass"]
    bad += [k for k, v in sections.items() if "__trial-" not in k and v.get("status") == "quality_fail"]
    if bad:
        return "quality_fail", ", ".join(f"{k.split('__', 1)[-1]}={sections[k]['status']}" for k in bad[:8])
    return "complete", "all pass"


def assess(sections, expect, planned=(), target=None):
    """What a finished run's sections fail to show of what the phase promised; ``target``: the plan's context."""
    problems = []
    for key, section in sections.items():
        if key.endswith("__context_qualification"):
            if section.get("status") != "pass" or section.get("selected_context") != target:
                problems.append(f"ran at {section.get('selected_context')} ({section.get('status')}), "
                                f"not {target}")
    if not any(k.endswith("__context_qualification") for k in sections):
        problems.append("no context qualification section")
    if expect.get("graded"):
        graded = {k.split("__", 1)[-1] for k, v in sections.items()
                  if "__trial-wl-" in k and v.get("status") in ("pass", "quality_fail", "timeout")}
        # Unmeasured: graded, but the session stopped short of its depth floor, so the task was not asked at depth.
        unmeasured = sorted(k.split("__", 1)[-1] for k, v in sections.items()
                            if "__trial-wl-" in k and v.get("status") == "unmeasured")
        missing = sorted(set(planned) - graded - set(unmeasured))
        if unmeasured:
            problems.append(f"{len(unmeasured)} task-runs unmeasured (stopped short of the depth floor): "
                            f"{', '.join(unmeasured[:5])}")
        if not graded and not unmeasured:
            problems.append("no workload task was graded")
        elif missing:
            problems.append(f"{len(missing)} planned task-runs have no verdict: {', '.join(missing[:5])}")
    if expect.get("recall"):
        recall = next((v for k, v in sections.items() if k.endswith("__long_context_recall")), {})
        # A recall that ran at the target and missed is a measured result (quality_fail), not a phase that failed
        # to measure (review 2026-11-06, section 3); one verified at another rung, or never run there, is not.
        measured_miss = recall.get("status") == "quality_fail" and target in recall_rungs(recall)
        if not measured_miss and (recall.get("status") != "pass" or recall.get("verified_context") != target):
            problems.append(f"{recall_text(recall, sections)}, not verified at {target}")
    if expect.get("tools"):
        tools = next((v for k, v in sections.items() if k.endswith("__structured_tools")), {})
        # Tool calls missed at the target are a measured result, like a recall miss (2026-11-06).
        measured_miss = tools.get("status") == "quality_fail" and tools.get("context") in (None, target)
        if not measured_miss and (tools.get("status") != "pass" or tools.get("passed") != tools.get("required")):
            problems.append(f"tools {tools.get('passed')}/{tools.get('required')} ({tools.get('status')})")
    return problems


def describe(sections):
    """One line of numbers for the summary: graded, passes, incomplete, model time, per category, cache."""
    trials = [v for k, v in sections.items() if "__trial-wl-" in k]
    passed = sum(t.get("status") == "pass" for t in trials)
    incomplete = sum(bool(t.get("incomplete")) for t in trials)
    seconds = round(sum((t.get("metrics") or {}).get("model_seconds") or 0 for t in trials), 1)
    groups = {k.rsplit("__workload_", 1)[1]: v for k, v in sections.items()
              if "__workload_" in k and v.get("status") != "skipped"}
    categories = ", ".join(f"{g} {v.get('passed')}/{v.get('total')}" for g, v in sorted(groups.items()))
    qualification = next((v for k, v in sections.items() if k.endswith("__context_qualification")), {})
    memory = qualification.get("memory") or {}
    cache = f"{memory.get('cache_slots')} slots" if memory.get("cache_slots") else "cache not reported"
    recall = next((v for k, v in sections.items() if k.endswith("__long_context_recall")), {})
    tools = next((v for k, v in sections.items() if k.endswith("__structured_tools")), {})
    extra = []
    if recall.get("status") not in (None, "skipped"):
        extra.append(recall_text(recall, sections))
    if tools.get("status") not in (None, "skipped"):
        extra.append(f"tools {tools.get('passed')}/{tools.get('required')}")
    return (f"{passed}/{len(trials)} pass, {incomplete} incomplete, {seconds} s model time"
            + (f"; {categories}" if categories else "") + (f"; {'; '.join(extra)}" if extra else "")
            + f"; {cache}")


def recall_rungs(recall):
    return [d.get("context") for d in recall.get("descent") or []]


def recall_text(recall, sections):
    """"recall quality_fail (2/3 at 131072; 32/36 records; verified none)": each rung tried, the records the trials
    got right, and the verified context. "verified at None" alone hid how close a model came."""
    rungs = ", ".join(f"{d.get('passed')}/{d.get('total')} at {d.get('context')}" for d in recall.get("descent") or [])
    grades = [v.get("grading") or {} for k, v in sections.items() if "__trial-recall-" in k]
    right = sum(g.get("correct_positions") or 0 for g in grades)
    expected = sum(g.get("expected_count") or 0 for g in grades)
    records = f"; {right}/{expected} records" if expected else ""
    return (f"recall {recall.get('status')} ({rungs or 'no rung'}{records}; "
            f"verified {recall.get('verified_context') or 'none'})")
