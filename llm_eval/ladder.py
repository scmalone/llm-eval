"""What to run next: the rung ladder (``suites/ladder.json``) read against the metrics corpus.

``next_step`` is read-only. A rung is done for a (profile, mode) when every task in it has its runs
recorded in the corpus, valid for analysis and not an infrastructure or capacity failure (model
failures are data and count). It never advances past a rung whose latest attempt was stopped by an
infrastructure failure, a capacity failure or an unfinished run until the owner says ``--retry``.
"""
import json
from pathlib import Path
import re
import sqlite3

from .metrics.ingest import spec_sha
from .workload import prepare as workload

LADDER_FILE = Path(__file__).resolve().parents[1] / "suites" / "ladder.json"
DEFAULT_PROFILE = "muse-glimmer-30b-q4"
MODES = ("bare",)  # the -agent entries are the hosted-delegation use case, backlogged (plan 2026-10-04)
PENDING = "bench-pending.json"
PROJECT = Path(__file__).resolve().parents[1]


def _select_ids(rung, task_dir=None):
    _, tasks = workload.select(rung["suite"], task_dir=task_dir or workload.TASK_DIR, task_ids=rung.get("tasks"))
    return [t["id"] for t in tasks]


def load(path=None, task_dir=None):
    """The rungs, each with its resolved ``task_ids``. ``except_rungs`` names earlier rungs whose tasks it
    leaves out, so a later rung can be "everything else in the suite"."""
    rungs = json.loads(Path(path or LADDER_FILE).read_text())["rungs"]
    seen, resolved = set(), {}
    for rung in rungs:
        if rung["id"] in seen or not re.fullmatch(r"[a-z0-9-]+", rung["id"]):
            raise ValueError(f"rung ids must be unique slugs: {rung['id']}")
        ids = _select_ids(rung, task_dir)  # raises for an unknown suite or task
        unknown = [r for r in rung.get("except_rungs", []) if r not in resolved]
        if unknown:
            raise ValueError(f"rung {rung['id']} excepts rungs that do not come before it: {', '.join(unknown)}")
        excluded = {t for r in rung.get("except_rungs", []) for t in resolved[r]}
        rung["task_ids"] = resolved[rung["id"]] = [t for t in ids if t not in excluded]
        seen.add(rung["id"])
    return rungs


def requirements(rung, task_dir=None):
    """{task id: runs needed} for a rung."""
    return {t: need for t, (need, _) in _needs(rung, task_dir).items()}


CHECK_FILE = LADDER_FILE.with_name("check.json")


def check_tasks(check_file=None):
    """The check suite's task ids (suites/check.json): its fixed tasks, then every filled slot."""
    spec = json.loads(Path(check_file or CHECK_FILE).read_text())
    return list(spec["tasks"]) + [s["task"] for s in spec.get("slots", []) if s.get("task")]


def screen_tasks(ladder_file=None, task_dir=None):
    """The screen's task ids in ladder order: every task of every rung in the 'screen' stage."""
    seen = []
    for rung in load(ladder_file, task_dir):
        if rung.get("stage", "screen") == "screen":
            seen += [t for t in rung["task_ids"] if t not in seen]
    return seen


def _needs(rung, task_dir=None):
    """{task id: (runs needed, spec sha)} for a rung."""
    ids = rung["task_ids"] if "task_ids" in rung else _select_ids(rung, task_dir)
    if not ids:
        return {}
    _, tasks = workload.select(rung["suite"], task_dir=task_dir or workload.TASK_DIR, task_ids=ids)
    return {t["id"]: (rung.get("runs") or t.get("runs", 1), spec_sha(t)) for t in tasks}


def entry_for(profile, mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    return profile


def modes_of(mode):
    return list(MODES) if mode == "both" else [mode]


def _counts(root, profile, entry, specs, output=None):
    """Runs on record per task, ``specs`` being {task id: spec sha}. Only runs of this exact version of the
    task, in the default mode (no ``--context`` override), count: an edited task or another context must be rerun.
    ``output`` pins the output limit (the screen rungs, so progress agrees with `acceptance screen`)."""
    db = Path(root) / "metrics.sqlite"
    if not db.exists() or not specs:
        return {t: 0 for t in specs}
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT t.task_id, t.task_spec_sha, COUNT(*) FROM task_runs t JOIN runs r ON r.run_id = t.run_id "
            "JOIN run_models m ON m.run_id = t.run_id AND m.profile = t.profile "
            "WHERE t.profile = ? AND m.mode_entry = ? AND r.valid_for_analysis = 1 "
            "AND COALESCE(m.mode_context_overridden, 0) = 0 "
            + ("AND m.mode_output_limit = ? " if output is not None else "") +
            "AND (t.failure_class IS NULL OR t.failure_class IN ('model', 'timeout')) GROUP BY t.task_id, t.task_spec_sha",
            (profile, entry, *([output] if output is not None else []))).fetchall()
    except sqlite3.OperationalError:  # a corpus that predates task_runs or the spec column
        rows = []
    finally:
        conn.close()
    return {t: sum(n for task, sha, n in rows if task == t and sha == specs[t]) for t in specs}


def table(root, profile=DEFAULT_PROFILE, modes=("bare",), rungs=None, task_dir=None, ladder_file=None):
    """One row per (rung, mode): tasks done of needed, in ladder order."""
    rows = []
    for rung in rungs or load(ladder_file, task_dir):
        needs = _needs(rung, task_dir)
        need = {t: n for t, (n, _) in needs.items()}
        specs = {t: sha for t, (_, sha) in needs.items()}
        for mode in modes:
            have = _counts(root, profile, entry_for(profile, mode), specs,
                           workload.SCREEN_OUTPUT if rung.get("stage", "screen") == "screen" else None)
            done = sum(min(have[t], need[t]) for t in need)
            rows.append({"rung": rung["id"], "mode": mode, "runs_done": done, "runs_needed": sum(need.values()),
                         "complete": all(have[t] >= need[t] for t in need), "adds": rung.get("adds", "")})
    return rows


def attempt_dir(root, profile, mode, rung_id, n):
    return Path(root) / f"bench-{profile}-{mode}-{rung_id}-{n}"


def attempts(root, profile, mode, rung_id):
    found = []
    for path in Path(root).glob(f"bench-{profile}-{mode}-{rung_id}-*"):
        match = re.fullmatch(rf"bench-{re.escape(profile)}-{mode}-{re.escape(rung_id)}-(\d+)", path.name)
        if match and path.is_dir():
            found.append(int(match[1]))
    return sorted(found)


def _latest_run(directory):
    states = sorted(Path(directory).glob("runs/*/state.json"))
    return json.loads(states[-1].read_text()) if states else None


REFUSED_NO_LOCK = "the last attempt was refused for want of a sleep lock (status 'refused_no_inhibit')"


def blocker(directory):
    """Why the latest attempt must not simply be repeated, or None. An attempt that was prepared but
    never started is not a blocker (it is reused)."""
    state = _latest_run(directory)
    if state is None:
        return None
    if state.get("status") == "refused_no_inhibit":
        return REFUSED_NO_LOCK
    if state.get("status") != "complete":
        return f"the run did not finish (status {state.get('status')!r})"
    # A task-run that stopped on the harness or capacity, or a model-level failure (it could not load or qualify at
    # any context: a `<profile>__failure` section), is not a result to repeat blindly (red-team RT2-007).
    bad = [(k, v) for k, v in state["sections"].items()
           if ("trial-wl-" in k and v.get("status") in ("infra_fail", "capacity_fail"))
           or (k.endswith("__failure") and v.get("status") in ("infra_fail", "capacity_fail", "timeout"))]
    if bad:
        key, value = bad[0]
        return f"{value['status']} in {key.split('__', 1)[-1]}: {str(value.get('stop') or value.get('reason'))[:300]}"
    return None


def token_of(directory):
    return json.loads((Path(directory) / "plan.json").read_text())["plan_hash"][:12]


def next_step(root, profile=DEFAULT_PROFILE, modes=("bare",), retry=False, task_dir=None, ladder_file=None,
              stage="screen"):
    """The first incomplete (rung, mode) in ladder order, with the directory its attempt would use. ``stage`` is
    "screen" (the default: only the screen rungs, so a profile is never fully run by accident) or "full" (every
    rung, for a profile the owner has chosen after reading `acceptance screen`)."""
    rungs = [r for r in load(ladder_file, task_dir) if stage == "full" or r.get("stage", "screen") == "screen"]
    rows = table(root, profile, modes, rungs, task_dir)
    by_id = {r["id"]: r for r in rungs}
    for row in rows:
        if row["complete"]:
            continue
        rung, mode = by_id[row["rung"]], row["mode"]
        used = attempts(root, profile, mode, rung["id"])
        if used and not retry:
            why = blocker(attempt_dir(root, profile, mode, rung["id"], used[-1]))
            if why == REFUSED_NO_LOCK:  # recovered with `resume`, not by starting a new attempt
                return {"status": "blocked", "rung": rung["id"], "mode": mode, "why": why,
                        "attempt_dir": str(attempt_dir(root, profile, mode, rung["id"], used[-1])),
                        "next": "STOP: the last attempt of this rung was refused because no sleep lock could be taken; "
                                "production was not touched. Tell the owner. On their say-so recover it with `resume` "
                                "(the refusal printed the exact command), not `--retry`."}
            if why:
                return {"status": "blocked", "rung": rung["id"], "mode": mode, "why": why,
                        "attempt_dir": str(attempt_dir(root, profile, mode, rung["id"], used[-1])),
                        "next": "STOP: the last attempt of this rung ended on a harness or capacity problem. Report "
                                "the reason and the run folder to the owner and find the cause; do not repeat blindly. "
                                "Only on the owner's say-so, rerun `prepare --next --retry`."}
        started = used and _latest_run(attempt_dir(root, profile, mode, rung["id"], used[-1])) is not None
        n = (used[-1] + 1) if started else (used[-1] if used else 1)
        return {"status": "next", "rung": rung["id"], "mode": mode, "rung_spec": rung,
                "needs": requirements(rung, task_dir), "progress": rows,
                "dir": str(attempt_dir(root, profile, mode, rung["id"], n))}
    done = "screen" if stage != "full" else "ladder"
    return {"status": "ladder_complete", "stage": stage, "progress": rows,
            "next": (f"The {done} is done for this profile. " + (
                "Run `acceptance screen` once every profile has its screen results." if stage != "full" else
                "Summarise the scorecard from the corpus for the owner."))}


def record_draft(directory, summary, code, rung, mode, profile, today):
    """Pre-filled rows for the record step, saved beside the attempt and returned. Facts only: the
    operator adds the judgment (what the result means) before committing."""
    token = token_of(directory)
    folder = str(Path(directory).resolve())
    cards = summary.get("scorecard") or {}
    score = "; ".join(f"{g} {c['passed']}" for g, c in cards.items()) or "none"
    failing = ", ".join(summary.get("failing_sections") or []) or "none"
    result = f"exit {code}, status {summary.get('status')}, failing: {failing}"
    restore = summary.get("production_restore") or {}
    restored = "production restored and verified" if restore.get("health") and restore.get("service_matches") else "restore NOT confirmed"
    report = f"reports/{today}-{profile}-{rung}-{mode}.html"
    draft = {
        "runs_md_row": f"| {today} | `{profile}` ({mode}, {rung}, token `{token}`) | {result}; {restored} | n/a | n/a | {score} "
                       f"| [report]({report}) | `{folder}/runs/acceptance-{token}/` |",
        "state_md_row": f"| {today} | Ladder rung {rung}, mode {mode}, `{profile}` (token `{token}`): {result}; {restored}. "
                        f"Run folder `{folder}`. <what this means, in one sentence> | <why it matters for the next rung> |",
        "copy_report": f"cp {summary.get('report_html')} {PROJECT / report}" if summary.get("report_html") else None,
        "token": token,
    }
    (Path(directory) / "record-draft.md").write_text(
        "Draft rows. Add the judgment, then put each row where it belongs (RUNBOOK, Recording).\n\n"
        f"RUNS.md, Runs table:\n{draft['runs_md_row']}\n\nSTATE.md, Major Decisions:\n{draft['state_md_row']}\n\n"
        f"Report copy: {draft['copy_report']}\n")
    return draft


def write_pending(root, **values):
    (Path(root) / PENDING).write_text(json.dumps(values, indent=2))


def read_pending(root):
    path = Path(root) / PENDING
    if not path.exists():
        raise PermissionError("nothing is prepared; run `prepare --next` first")
    return json.loads(path.read_text())


def clear_pending(root):
    (Path(root) / PENDING).unlink(missing_ok=True)
