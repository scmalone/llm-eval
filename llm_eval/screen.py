"""The screen matrix and the top-3 selection (plan 2026-10-04), read from the metrics corpus.

The screen is a fixed list of tasks (every task of the ladder's 'screen' rungs), run once per profile.
This module only reads: it builds the profile-by-task matrix from runs of the exact current version of
each task, in the bare entry, with no context override and the screen's output limit (so every profile is
compared like with like), and ranks the profiles that finished it.

Outcomes: ``pass``, ``fail`` (a substantive failure), ``timeout`` (the task's own wall cap; a failure, kept
visible), ``infra`` and ``capacity`` (not the model's verdict; the result is still owed) and ``missing``. A
profile is ranked only when every screen task has a pass, fail or timeout. Missing results stay missing.
Rank by substantive passes, then category coverage, then elapsed time on the tasks the tied profiles all
passed. Where the third place is a near-tie, the same extra task is run for each boundary candidate; the
extra tasks are named in the ladder file (``tiebreak``). No quality-based elimination in the first screen.
"""
import json
from pathlib import Path
import sqlite3

from . import ladder
from .workload import prepare as workload

OUTCOMES = {"pass": "pass", "quality_fail": "fail", "timeout": "timeout", "infra_fail": "infra",
            "capacity_fail": "capacity", "unmeasured": "unmeasured"}
COMPLETE = ("pass", "fail", "timeout")
# A regex verdict on these is not a quality verdict (red-team EF-003, RT2-010): the owner or agent reads each answer,
# failures too (a gate can reject a right answer), before the ranking is trusted; the report lists the files to read.
AUDIT_CATEGORIES = ("review", "advisory")


def screen_specs(ladder_file=None, task_dir=None):
    """[(task id, category, spec sha)] in screen order."""
    from .metrics.ingest import spec_sha
    ids = ladder.screen_tasks(ladder_file, task_dir)
    _, tasks = workload.select("core", task_dir=task_dir or workload.TASK_DIR)
    by_id = {t["id"]: t for t in tasks}
    return [(i, by_id[i]["category"], spec_sha(by_id[i])) for i in ids]


def matrix(root, profiles, specs):
    """{profile: {task id: {"outcome", "wall_seconds", "context"}}} from the newest valid run of each task."""
    db = Path(root) / "metrics.sqlite"
    cells = {p: {} for p in profiles}
    if not db.exists():
        return cells
    conn = sqlite3.connect(db)
    try:
        for profile in profiles:
            for task, _, sha in specs:
                row = conn.execute(
                    "SELECT t.status, t.wall_seconds, t.context, t.export_path FROM task_runs t JOIN runs r ON r.run_id = t.run_id "
                    "JOIN run_models m ON m.run_id = t.run_id AND m.profile = t.profile "
                    "WHERE t.profile = ? AND m.mode_entry = ? AND r.valid_for_analysis = 1 "
                    "AND COALESCE(m.mode_context_overridden, 0) = 0 AND m.mode_output_limit = ? AND t.task_id = ? "
                    "AND t.task_spec_sha = ? ORDER BY t.recorded_at DESC LIMIT 1",
                    (profile, ladder.entry_for(profile, "bare"), workload.SCREEN_OUTPUT, task, sha)).fetchone()
                if row:
                    cells[profile][task] = {"outcome": OUTCOMES.get(row[0], "missing"), "wall_seconds": row[1], "context": row[2],
                                            "result": str(Path(row[3]).parent / "result.json") if row[3] else None}
    except sqlite3.OperationalError:  # a corpus that predates task_runs or the spec column
        pass
    finally:
        conn.close()
    return cells


def _outcome(cells, profile, task):
    return cells.get(profile, {}).get(task, {}).get("outcome", "missing")


def rank(cells, specs):
    """Rank the profiles that finished the screen; list the rest with what they still owe."""
    tasks = [t for t, _, _ in specs]
    category = {t: c for t, c, _ in specs}
    owed, finished = {}, []
    for profile in cells:
        outcomes = {t: _outcome(cells, profile, t) for t in tasks}
        missing = {t: o for t, o in outcomes.items() if o not in COMPLETE}
        if missing:
            owed[profile] = missing
            continue
        passed = [t for t in tasks if outcomes[t] == "pass"]
        finished.append({"profile": profile, "passes": len(passed), "coverage": len({category[t] for t in passed}),
                         "passed": passed, "timeouts": [t for t in tasks if outcomes[t] == "timeout"],
                         "audit": {t: {"outcome": outcomes[t], "result": cells[profile][t].get("result")} for t in tasks
                                   if category[t] in AUDIT_CATEGORIES and outcomes[t] in ("pass", "fail")}})
    finished.sort(key=lambda e: (-e["passes"], -e["coverage"], e["profile"]))
    # Ties on passes and coverage: faster on the tasks every tied profile passed, when there are any.
    i = 0
    while i < len(finished):
        j = i
        while j + 1 < len(finished) and (finished[j + 1]["passes"], finished[j + 1]["coverage"]) == (finished[i]["passes"], finished[i]["coverage"]):
            j += 1
        group = finished[i:j + 1]
        if len(group) > 1:
            common = set.intersection(*(set(e["passed"]) for e in group))
            for e in group:
                e["elapsed_common_seconds"] = round(sum(cells[e["profile"]][t]["wall_seconds"] or 0 for t in common), 1) if common else None
            if common:
                group.sort(key=lambda e: e["elapsed_common_seconds"])
                finished[i:j + 1] = group
            for e in group:
                e["tied_with"] = [o["profile"] for o in group if o is not e]
        i = j + 1
    for position, e in enumerate(finished, 1):
        e["position"] = position
    top3 = [e["profile"] for e in finished[:3]]
    # Contested when a profile outside the top 3 is within one pass of the third place: it is the slot that is in
    # question, so the boundary is the third place and those at or below it that close.
    boundary, third = [], finished[2] if len(finished) >= 3 else None
    if third:
        near = [e["profile"] for e in finished[2:] if e["passes"] >= third["passes"] - 1]
        boundary = near if len(near) > 1 else []
    # a tie that the elapsed time could not break leaves the third place provisional
    provisional = bool(third and ((third.get("tied_with") and third.get("elapsed_common_seconds") is None) or boundary))
    return {"ranking": finished, "top3": top3, "boundary_candidates": boundary, "third_place_provisional": provisional,
            "owes_results": owed}


def tiebreak_specs(ladder_file=None, task_dir=None):
    """[(task id, category, spec sha)] for the ladder's `tiebreak` tasks, in the order they are to be run."""
    from .metrics.ingest import spec_sha
    ids = json.loads(Path(ladder_file or ladder.LADDER_FILE).read_text()).get("tiebreak", [])
    _, tasks = workload.select("core", task_dir=task_dir or workload.TASK_DIR, task_ids=ids) if ids else (None, [])
    return [(t["id"], t["category"], spec_sha(t)) for t in tasks]


def settle(result, tb_cells, tb_specs):
    """Fold the tiebreak runs into a contested third place (red-team RT2-004). The boundary candidates (the third
    place and those within one pass below it) compete for one slot. Extra tasks are used in order, and only once
    every candidate has a completed result for one; the best total of screen passes plus extra passes takes the
    slot when it is strictly ahead. Otherwise the next extra is owed, and when none are left the slot goes to the
    faster candidate and stays provisional."""
    boundary = result["boundary_candidates"]
    if not boundary:
        return result
    entries = {e["profile"]: e for e in result["ranking"]}
    counted, owed, settled = [], None, False

    def totals():
        return {p: entries[p]["passes"] + sum(tb_cells.get(p, {}).get(t, {}).get("outcome") == "pass" for t in counted)
                for p in boundary}

    for task, _, _ in tb_specs:
        outcomes = {p: tb_cells.get(p, {}).get(task, {}).get("outcome", "missing") for p in boundary}
        incomplete = {p: o for p, o in outcomes.items() if o not in COMPLETE}
        if incomplete:
            owed = {"task": task, "profiles": sorted(incomplete), "outcomes": incomplete}
            break
        counted.append(task)
        ordered = sorted(totals().values(), reverse=True)
        if ordered[0] > ordered[1]:
            settled = True
            break
    grand = totals()
    position = {e["profile"]: e["position"] for e in result["ranking"]}
    best_first = sorted(boundary, key=lambda p: (-grand[p], position[p]))
    return {**result, "top3": result["top3"][:2] + [best_first[0]], "boundary_totals": grand,
            "tiebreak_counted": counted, "tiebreak_owed": owed, "third_place_provisional": not settled}


def cannot_run(root, profiles):
    """{profile: reason} for profiles whose newest valid run could not load or qualify at any context and that
    have no task results: a capacity result the owner decides on, not a result still owed."""
    db = Path(root) / "metrics.sqlite"
    if not db.exists():
        return {}
    conn = sqlite3.connect(db)
    found = {}
    try:
        for key, reason in conn.execute(
                "SELECT s.key, s.reason FROM sections s JOIN runs r ON r.run_id = s.run_id WHERE r.valid_for_analysis = 1 "
                "AND s.status = 'capacity_fail' AND s.key LIKE '%\\_\\_failure' ESCAPE '\\' ORDER BY s.time"):
            profile = key[:-len("__failure")]
            if profile in profiles:
                found[profile] = reason
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()
    return found


def report(root=None, preset=None, ladder_file=None, task_dir=None):
    from .preflight import DEFAULT_PRESET, distinct_profiles, read_presets
    root = Path(root) if root else None
    _, presets = read_presets(preset or DEFAULT_PRESET)
    profiles = [name for name, _ in distinct_profiles(presets)]
    specs = screen_specs(ladder_file, task_dir)
    cells = matrix(root, profiles, specs)
    result = rank(cells, specs)
    blocked = {p: r for p, r in cannot_run(root, profiles).items() if not cells[p]}
    result["owes_results"] = {p: v for p, v in result["owes_results"].items() if p not in blocked}
    tiebreak = tiebreak_specs(ladder_file, task_dir)
    if result["boundary_candidates"]:
        result = settle(result, matrix(root, result["boundary_candidates"], tiebreak), tiebreak)
    started = [p for p in profiles if cells[p]]
    not_started = [p for p in profiles if p not in started and p not in blocked]
    return {"screen_tasks": [t for t, _, _ in specs],
            "matrix": {p: {t: cells[p].get(t, {}).get("outcome", "missing") for t, _, _ in specs} for p in profiles},
            "not_started": not_started, "cannot_run": blocked,
            **result,
            "tiebreak_tasks": [t for t, _, _ in tiebreak] if result["boundary_candidates"] else [],
            "next": _next(result, not_started, blocked)}


def _next(result, not_started, blocked=None):
    note = (f" {len(blocked)} profile(s) could not load at any context ({', '.join(sorted(blocked))}): that is a capacity "
            "result, show it to the owner with the reason.") if blocked else ""
    if not_started:
        return (f"{len(not_started)} profile(s) have no screen results yet: run `prepare --next --profile <name>` for each. "
                "Do not run the full stage for any profile until every profile has been screened." + note)
    if result["owes_results"]:
        return "Some profiles still owe screen results (infrastructure, capacity or missing): find the cause and finish them first." + note
    owed = result.get("tiebreak_owed")
    if owed:
        return (f"The third place is contested: run `{owed['task']}` once for each of {', '.join(owed['profiles'])} (manual form: "
                f"`prepare --suite core --task {owed['task']} --profile <name> --dir <new dir>`), then run `acceptance screen` again." + note)
    if result["boundary_candidates"] and result.get("third_place_provisional"):
        return ("The third place is still tied after every tiebreak task: it goes to the faster boundary candidate and stays "
                "provisional. Read the audit files, then show the owner the matrix and say so." + note)
    return ("Read each ranked profile's `audit` files (the review and advisory answers, passes and failures) and confirm "
            "the gates judged them right before trusting the ranking. If they hold, show the owner the matrix and ranking; on "
            "their yes run `prepare --next --stage full` for each of the top 3." + note)
