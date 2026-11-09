#!/usr/bin/env python3
"""Launch, watch and report an unattended test plan for an agent that is not watching it (RUNCARD.md).

    python3 tools/plan_ctl.py launch -- [run_test_plan.py arguments]
    python3 tools/plan_ctl.py watch            # blocks until the plan ends; start it in the background
    python3 tools/plan_ctl.py report           # one short status: unit, phases, production, next step
    python3 tools/plan_ctl.py preflight        # the machine and every open queue row, in seconds; touches nothing
    python3 tools/plan_ctl.py stop             # only when the owner asks: SIGTERM, which restores production

``launch`` runs the cheap refusals first, then starts ``tools/run_test_plan.py`` as the transient user unit
``llm-eval-plan``: it outlives the agent's session, holds one sleep and idle lock for the whole plan (the
campaigns' own locks lapse between phases), and on stop sends SIGTERM to the runner alone, which forwards it to
the running campaign as SIGINT so production is restored. ``watch`` polls the unit every 30 seconds and spends
no tokens while it waits. Nothing here changes the plan, its phases or its verdicts.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("run_test_plan", REPO / "tools" / "run_test_plan.py")
plan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plan)

UNIT = "llm-eval-plan"
STATE = plan.RUNS / "plan-control.json"
POLL_SECONDS = 30
# Restore wait plus margin: systemd must not SIGKILL a campaign that is still restoring production.
STOP_TIMEOUT_SECONDS = plan.RESTORE_WAIT_SECONDS + 300
KEY_WORDS = (": complete", "quality_fail", "finished", "STOP", "refused", "PRODUCTION", "stopping", "infra_fail", "assert_fail", "received",
             "plan starting", "drill-", "skipped")
# Variables a user unit does not inherit and the runner needs (user bus for systemctl, the session's PATH).
PASS_ENV = ("PATH", "HOME", "LANG", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS")


def _systemctl(*args):
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def unit_state():
    """(ActiveState, Result, ExecMainStatus) of the plan unit; ('inactive', '', '') when it never ran."""
    out = _systemctl("show", UNIT, "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus").stdout
    fields = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return fields.get("ActiveState", "inactive"), fields.get("Result", ""), fields.get("ExecMainStatus", "")


def running(state=None):
    return (state or unit_state())[0] in ("active", "activating", "deactivating", "reloading")


def target_args(argv):
    """The target options run_test_plan.py would read, so the plan folder is known before it starts."""
    parser = argparse.ArgumentParser(add_help=False)
    for flag, kind in (("--runtime", str), ("--profile", str), ("--context", int), ("--harness", str),
                       ("--runs", int), ("--output", int), ("--only", str), ("--suite", str)):
        parser.add_argument(flag, type=kind)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", action="append", default=[])
    known, _ = parser.parse_known_args(argv)
    return known


def refusals(args):
    """The runner's own refusals, checked before the unit starts so a refused plan never looks launched."""
    from llm_eval import environment
    why = []
    if running():
        why.append(f"{UNIT} is already running; `report` says where it is")
    if args.dry_run:
        return why
    dirty = environment.dirty_paths()
    if dirty:
        why.append(f"uncommitted harness changes: {dirty[:5]}")
    if not plan.production_ok():
        why.append(f"{plan.ROUTER} is not active and healthy")
    from llm_eval import ramcheck
    low_ram, _ = ramcheck.floor_blocker(plan.RUNTIME)  # refused here in seconds, not after a stopped router
    if low_ram:
        why.append(low_ram)
    phases = [p.strip() for p in (args.only or "").split(",") if p.strip()]
    try:
        phases = phases or plan.narrow(args.task)
    except ValueError as exc:
        why.append(str(exc))
    if (not phases or any(p != "D" for p in phases)) and not plan.drill_passed():
        why.append("no passing SIGINT and SIGTERM drill on the current restore code: run `launch -- --only D` first")
    return why


def queue_rows(text):
    """Open rows of QUEUE.md, in order, up to the first `owner:` row: [(row, status, command)]. A plan row's
    command is its runner arguments; a sweep row's starts with python3."""
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5 or not cells[0][:1].isalpha() or cells[0] == "Row":
            continue
        row, command, status = cells[0], cells[2].strip("`"), cells[3]
        if status.startswith("owner:"):
            break
        if status.startswith(("todo", "partial")):
            rows.append((row, status, command))
    return rows


def preflight(queue=None):
    """Everything the unattended night needs, checked in seconds without launching or stopping anything: the
    machine (RAM floor, GPU, router, harness, disk) and each open row (target resolves, drill state, sweep
    folder free). Run it first; any problem here would have cost the night."""
    from llm_eval import environment, ramcheck
    text = (queue or REPO / "QUEUE.md").read_text()
    rows = queue_rows(text)
    problems, notes = [], []
    dirty = environment.dirty_paths()
    if dirty:
        problems.append(f"uncommitted harness changes: {dirty[:5]}")
    if running():
        problems.append(f"{UNIT} is already running")
    if not plan.production_ok():
        notes.append(f"{plan.ROUTER} is not active and healthy (start it before launching)")
    holders = [h for h in plan.gpu_holders() or [] if h[1] > plan.GPU_GATE_MIB]
    if holders:
        problems.append(f"another program holds the GPU: {holders}")
    free_gib = shutil.disk_usage(plan.RUNS if plan.RUNS.exists() else REPO).free / 2**30
    if free_gib < DISK_FLOOR_GIB:
        problems.append(f"only {free_gib:.0f} GiB free for run data (floor {DISK_FLOOR_GIB})")
    drill_first = bool(rows) and rows[0][2].replace(" ", "") == "--onlyD"
    for row, status, command in rows:
        if command.startswith("python3"):
            if "tools/speed_sweep.py" in command and "--out " in command:
                out = Path(command.split("--out ")[1].split()[0]).expanduser()
                if out.exists():
                    notes.append(f"{row}: sweep folder {out} exists; the operator renames it first")
            continue
        argv = command.split()
        args = target_args(argv)
        try:
            plan.configure(args.runtime or "strata", args.profile, args.context, args.harness or "pi", args.runs,
                           args.output, args.suite or "plan")
        except Exception as exc:  # a bad profile, a stale pi entry, a missing frozen config
            problems.append(f"{row}: {exc}")
            continue
        if plan.RUNTIME == "strata":
            low, available = ramcheck.floor_blocker("strata")
            if low:
                problems.append(f"{row}: {low}")
        if args.only != "D" and not plan.drill_passed() and not drill_first:
            problems.append(f"{row}: no passing signal drill on the current restore code and the first open row is not the drill")
    return {"ok": not problems, "open_rows": [r[0] for r in rows], "problems": problems, "notes": notes,
            "next": "Fix the problems before the night starts." if problems else "Clear to launch the first open row."}


DISK_FLOOR_GIB = 50


def launch(argv):
    args = target_args(argv)
    try:
        plan.configure(args.runtime or "strata", args.profile, args.context, args.harness or "pi", args.runs,
                       args.output, args.suite or "plan")
    except ValueError as exc:
        return {"launched": False, "refused": [str(exc)]}
    why = refusals(args)
    if why:
        return {"launched": False, "refused": why}
    from llm_eval import integration
    notes = []
    if integration.blocker() and not args.dry_run:
        notes.append(f"integration tier: {integration.blocker()}; the runner runs it first (about 4 minutes)")
    holders = [h for h in plan.gpu_holders() or [] if h[1] > plan.GPU_GATE_MIB]
    if holders:
        notes.append(f"another program holds the GPU {holders}; the runner waits up to {plan.GPU_WAIT_MINUTES} min")
    plan.RUNS.mkdir(parents=True, exist_ok=True)
    plan.PLAN_DIR.mkdir(parents=True, exist_ok=True)
    log = plan.PLAN_DIR / "runner.log"
    offset = plan.SUMMARY.stat().st_size if plan.SUMMARY.exists() else 0
    _systemctl("reset-failed", UNIT)  # a finished unit that failed keeps its name until reset
    env = [f"--setenv={name}" for name in PASS_ENV if name in os.environ]
    # The inhibitor watches the runner's PID: bash execs into python, so $$ is the runner itself. systemd
    # expands $ in a unit's command line, so each one is doubled here.
    script = ('systemd-inhibit --what=sleep:idle --who=llm-eval-plan --why="unattended test plan" --mode=block '
              'tail --pid=$$ -f /dev/null & exec python3 tools/run_test_plan.py "$@"').replace("$", "$$")
    command = ["systemd-run", "--user", f"--unit={UNIT}", f"--working-directory={REPO}", *env,
               "-p", "KillMode=mixed", "-p", f"TimeoutStopSec={STOP_TIMEOUT_SECONDS}",
               "-p", f"StandardOutput=append:{log}", "-p", f"StandardError=append:{log}",
               "bash", "-c", script, "bash", *argv]
    started = subprocess.run(command, capture_output=True, text=True)
    if started.returncode:
        return {"launched": False, "refused": [f"systemd-run failed: {started.stderr.strip()[-400:]}"]}
    record = {"unit": UNIT, "argv": argv, "plan_dir": str(plan.PLAN_DIR), "summary": str(plan.SUMMARY),
              "summary_offset": offset, "log": str(log), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "started_epoch": time.time()}
    STATE.write_text(json.dumps(record, indent=1) + "\n")
    time.sleep(5)
    inhibited = UNIT in subprocess.run(["systemd-inhibit", "--list", "--no-pager"], capture_output=True,
                                       text=True).stdout
    if running() and not inhibited:
        notes.append("the sleep lock is not listed: the desktop may suspend between phases")
    return {"launched": running() or unit_state()[1] == "success", "unit": UNIT, "plan_dir": str(plan.PLAN_DIR),
            "log": str(log), "notes": notes, "state": unit_state()[0],
            "next": "Start `python3 tools/plan_ctl.py watch` in the background and end the turn."}


def _record():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return None


def summary_lines(record, full=False):
    """Lines the runner wrote to SUMMARY.md since this launch; only the ones a reader acts on unless ``full``."""
    try:
        with open(record["summary"], "rb") as handle:
            handle.seek(record.get("summary_offset", 0))
            text = handle.read().decode(errors="replace")
    except OSError:
        return []
    lines = [line for line in text.splitlines() if line.startswith("- ")]
    return lines if full else [line for line in lines if any(word in line for word in KEY_WORDS)]


def report(full=False):
    record = _record()
    if not record:
        return {"error": "no plan has been launched through plan_ctl"}
    active, result, status = unit_state()
    elapsed = round((time.time() - record.get("started_epoch", time.time())) / 3600, 2)
    lines = summary_lines(record, full)
    try:
        log_lines = Path(record["log"]).read_text(errors="replace").splitlines()
    except OSError:
        log_lines = []
    out = {"unit": f"{UNIT}: {active}" + (f" ({result}, exit {status})" if not running() else ""),
           "started": record["started"], "hours": elapsed, "args": " ".join(record["argv"]) or "(defaults)",
           "plan_dir": record["plan_dir"], "lines": lines[-40:],
           "production": "active and healthy" if plan.production_ok() else "NOT active and healthy"}
    if not lines:
        out["log_tail"] = log_lines[-15:]  # a dry run, or a refusal before the summary was opened
    ended = not running()
    finished = any("plan finished" in line for line in lines + log_lines[-5:])
    out["next"] = ("still running: keep waiting" if not ended else
                   "production is down: capture it, then start the router (RUNCARD.md sections 5 and 8)"
                   if out["production"].startswith("NOT") else
                   "ended without `plan finished`: capture and handle it (RUNCARD.md section 5)" if not finished else
                   "finished: record and report the phase results (RUNCARD.md section 4)")
    return out


def watch(timeout_hours, each_phase=False):
    """Block until the unit stops (or, with ``each_phase``, a phase finishes); print the report once."""
    record = _record()
    deadline = time.time() + timeout_hours * 3600
    seen = len(summary_lines(record)) if record else 0
    while time.time() < deadline:
        if not running():
            break
        if each_phase and record:
            lines = summary_lines(record)
            if any("finished:" in line for line in lines[seen:]):
                break
        time.sleep(POLL_SECONDS)
    return report() | ({"watch": "timed out, plan still running"} if running() and not each_phase else {})


def stop():
    if not running():
        return {"stopped": False, "why": f"{UNIT} is not running"}
    _systemctl("stop", "--no-block", UNIT)
    return {"stopped": True, "how": "SIGTERM to the runner, forwarded to the campaign as SIGINT; production "
            f"restores within {plan.RESTORE_WAIT_SECONDS} s. `watch` then `report`."}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="plan_ctl.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("launch", help="refusal checks, then start the plan unit; pass runner args after --")
    watching = commands.add_parser("watch", help="wait until the plan ends, then report")
    watching.add_argument("--timeout-hours", type=float, default=24)
    watching.add_argument("--each-phase", action="store_true", help="also return when a phase finishes")
    reporting = commands.add_parser("report", help="one short status")
    reporting.add_argument("--full", action="store_true", help="every summary line since launch")
    commands.add_parser("stop", help="only when the owner asks")
    commands.add_parser("preflight", help="check the machine and every open QUEUE.md row; touches nothing")
    if argv[:1] == ["launch"]:
        rest = argv[1:]
        result = launch(rest[1:] if rest[:1] == ["--"] else rest)
    else:
        args = parser.parse_args(argv)
        result = (watch(args.timeout_hours, args.each_phase) if args.command == "watch" else
                  report(args.full) if args.command == "report" else
                  preflight() if args.command == "preflight" else stop())
    print(json.dumps(result, indent=1))
    return 0 if result.get("launched", True) and result.get("ok", True) and "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
