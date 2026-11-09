"""Attended acceptance run: one profile, live, with an owner approval token.

Separate from ``llm_eval.cli`` on purpose: the main CLI stays fail-closed for live plans.
Every command prints one JSON object whose ``next`` field says what the operator does next,
and the exit code is the stop signal:

  0  done, production restored        1  evaluation failures, production restored
  2  refused before touching anything 3  STOP: production not verified restored
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from . import inhibit, integration, ladder, rung0, screen
from .htmlreport import write as write_html
from .live import LiveAdapter, RealHost
from .runtime import adapter_class
from .plan import compile_plan
from .policy import RESERVE
from .preflight import DEFAULT_PRESET, eval_manifest
from .workload import prepare as workload
from .runner import execute
from .state import atomic_json, machine_lock, safe_id

DEFAULT_RESTORE = "muse-glimmer-30b-q4"
RUNS_ROOT = Path.home() / "llm-eval-runs"  # profile folders, the corpus and the pending file live here
MACHINE_LOCK = Path.home() / ".cache" / "llm-eval" / "machine.lock"  # the lock execute() takes for live runs
OK, EVAL_FAILURES, REFUSED, STOP = 0, 1, 2, 3
# One profile takes 25-40 minutes (Muse Q4: 2,863 tok/s prefill, 45 tok/s decode, so a
# 112K-token prompt is ~40 s and a full 16,384-token reasoning budget ~6 min). These are hang
# caps that bound how long production can stay down.
LIMITS = {"campaign_seconds": 4500, "model_seconds": 3600, "load_seconds": 240, "request_seconds": 600}


def token(plan):
    """What the owner hands back to approve exactly this plan."""
    return plan["plan_hash"][:12]


def run_id(plan):
    return f"acceptance-{token(plan)}"


def prepare(directory, profile, restore_model=DEFAULT_RESTORE, preset=DEFAULT_PRESET, host=None,
            suite="classic", tasks=None, runs=None, entry=None, context=None, keep_going=False, output=None,
            harness="opencode", thinking=None, compaction_at=None, **workload_inputs):
    """Read-only on the machine: pin one profile and a suite, compile the plan, describe the run.

    Workload suites also build task snapshots (git) and read opencode's version and production
    config; nothing here starts, stops or changes a service."""
    directory = Path(directory)
    runtime = workload_inputs.pop("runtime", "llama.cpp")
    single_rung = workload_inputs.pop("single_rung", False)
    if single_rung and context is None:
        raise ValueError("--no-step-down needs --context: it pins the run to that one rung")
    if runtime == "strata":
        from .strata import manifest as strata_manifest
        manifest = strata_manifest(workload_inputs.pop("strata_config"), workload_inputs.pop("strata_python"),
            workload_inputs.pop("strata_server"), profile, context or workload.TARGET_CONTEXT, preset, restore_model,
            workload_inputs.pop("request_settings", None))
    elif runtime == "llama.cpp":
        manifest = eval_manifest(preset, restore_model=restore_model, profiles=[profile])
        from .preflight import pinned_artifacts, preset_request, read_presets
        request = workload_inputs.pop("request_settings", None)
        overrides = workload_inputs.pop("preset_overrides", None) or {}
        for model in manifest["models"]:
            if overrides:
                # A paired arm's one change to the preset (a server flag); the production preset file is untouched
                model["settings"] = {k: v for k, v in {**model["settings"], **overrides}.items() if v is not None}
                model["preset_overrides"] = dict(overrides)
                model["artifacts"] = pinned_artifacts(model["settings"], read_presets(preset)[0].parent, {})
            model["request"] = dict(request) if request is not None else preset_request(model["settings"])
    else:
        raise ValueError("unsupported inference runtime")
    if runtime != "llama.cpp" and workload_inputs.pop("preset_overrides", None):
        raise ValueError("--preset-set and --preset-unset change a llama.cpp preset; Strata arms change their config")
    if single_rung:
        # One rung, no fallback: a model that cannot hold this context fails capacity instead of passing at a
        # lower one, and paired campaigns (which refuse a ladder) can freeze the plan.
        # A llama.cpp preset's own ctx-size is a rung too: the 262K presets sit above the ladder's 131,072 top.
        loads = int(manifest["models"][0].get("settings", {}).get("ctx-size", 0)) if runtime == "llama.cpp" else 0
        if context not in manifest["models"][0].get("contexts", [context]) and context != loads:
            raise ValueError(f"context {context} is not a rung this profile can load")
        manifest["models"][0].update(anchor=context, contexts=[context])
    narrowing = {"--task": tasks, "--runs": runs, "--entry": entry, "--context": context, "--keep-going": keep_going or None}
    if runtime == "strata":
        # The Strata manifest anchors the context ladder on --context for every suite, so it is not a
        # workload-only flag here: without it the classic suite would silently anchor at the 131,072
        # default instead of the 262,144 target the plan asks for.
        narrowing.pop("--context")
    refused = [flag for flag, value in narrowing.items() if value is not None]
    if suite == "classic" and refused:
        raise ValueError(f"{', '.join(refused)} {'applies' if len(refused) == 1 else 'apply'} to workload suites only")
    if suite in ('strata', 'strata-depth'):
        if runtime != 'strata':
            raise ValueError('the strata cohort requires the Strata runtime')
        from . import cohort
        roster = cohort.ROSTER if suite == 'strata' else cohort.DEPTH_ROSTER
        settings = manifest['models'][0]['settings']
        with cohort.runtime_counter(settings['python'], settings['config'], settings['server']) as count:
            selected = cohort.prepare(directory, count, roster=roster)
        known = {t['id'] for t in selected}
        if tasks and set(tasks) - known:
            raise ValueError('unknown Strata cohort task')
        fixtures, tasks = [], [t for t in selected if not tasks or t['id'] in tasks]
    elif suite in ("documents", "accuracy"):
        # The document tasks sized by this profile's own tokenizer (Strata's, or the GGUF vocabulary), so every
        # model reads the same 32K and 100K; the accuracy suite adds core, depth and the collections.
        from . import cohort
        wanted = tasks
        if runtime == "strata":
            settings = manifest['models'][0]['settings']
            with cohort.runtime_counter(settings['python'], settings['config'], settings['server']) as count:
                docs = cohort.documents(directory, count)
        else:
            with cohort.llama_counter(manifest["models"][0]["settings"]["model"]) as count:
                docs = cohort.documents(directory, count)
        fixtures, rest = ([], []) if suite == "documents" else workload.select("accuracy")
        tasks = [t for t in rest + docs if not wanted or t["id"] in wanted]
        if wanted and set(wanted) - {t["id"] for t in tasks}:
            raise ValueError(f"unknown task for suite {suite}: {sorted(set(wanted) - {t['id'] for t in tasks})}")
    else:
        fixtures, tasks = workload.select(suite, task_ids=tasks, **({"task_dir": workload_inputs["task_dir"]} if "task_dir" in workload_inputs else {}))
    model_entry = manifest["models"][0]
    if not _sees_images(runtime, profile, model_entry):
        # a task that needs image input is not this profile's work: left out, not failed (the vision collection)
        tasks = [t for t in tasks if "image" not in t.get("requires", [])]
    tasks = workload.with_runs(tasks, runs, directory)
    model_entry["fixtures"] = fixtures + [t["_path"] for t in tasks]
    if runtime == "strata":
        model_entry["runtime"] = runtime
    minutes = workload.estimate_minutes(suite, tasks)
    if tasks:
        block_inputs = {k: v for k, v in workload_inputs.items() if k in ("production_config", "binary")}
        block_inputs.update(entry=entry, context=context, stop_on_infra_fail=not keep_going, output=output,
                            harness=harness, thinking=thinking, compaction_at=compaction_at)
        model_entry["workload"] = workload.workload_block(directory, model_entry, tasks, **block_inputs)
    manifest["limits"] = LIMITS if suite == "classic" else workload.limits(minutes, tasks, context or max(model_entry.get("contexts") or [0]) or None)
    atomic_json(directory / "manifest.json", manifest)
    plan = compile_plan(directory / "manifest.json")
    atomic_json(directory / "plan.json", plan)
    snapshot = (LiveAdapter if runtime == "llama.cpp" else adapter_class(plan))(plan, directory / "unused", host or RealHost(), run_id=run_id(plan)).snapshot()
    model = plan["models"][0]
    card = {
        "profile": model["id"], "covers": model.get("covers", [model["id"]]),
        "anchor": model["anchor"], "context_ladder": model["contexts"],
        "suite": suite, "estimated_minutes": minutes,
        "fixtures": sorted(f["kind"] for f in model["fixtures"]),
        # Each recall prompt fills the window minus the output budget: the state opencode is in
        # just before it compacts, if limit.output equals this budget.
        "output_budget": {f["kind"]: f["max_output_tokens"] for f in model["fixtures"] if "max_output_tokens" in f},
        "recall_prompt_tokens_by_context": {
            ctx: ctx - max(f["max_output_tokens"] for f in model["fixtures"] if f["kind"] == "recall") - RESERVE
            for ctx in model["contexts"]} if any(f["kind"] == "recall" for f in model["fixtures"]) else {},
        "production_unit": plan["production"]["unit"], "restore_model": plan["production"]["restore_model"],
        "production_active": snapshot["was_active"], "gpu_free_mib_now": snapshot["free_mib"],
        "unknown_gpu_clients": snapshot["unknown_gpu_clients"],
        "artifacts": [a["path"] for a in model["artifacts"]],
        "limits_seconds": plan["limits"],
        "plan": str((directory / "plan.json").resolve()), "approval_token": token(plan),
    }
    if runtime == "strata":
        card["runtime"] = runtime
        card["capabilities"] = model["settings"]["capabilities"]
        card["effective_request"] = model["settings"]["request"]
    elif "request" in model:
        card["effective_request"] = model["request"]
    if model.get("workload"):
        block = model["workload"]
        harness = block.get("harness", "opencode")
        card["workload"] = {"harness": harness,
                            "client": block["binary_version"] if harness == "pi" else block["opencode_version"],
                            "model_entry": block["ref"],
                            "context": block["context"], "mode": block["mode"],
                            "stop_on_infra_fail": block["stop_on_infra_fail"],
                            "tasks": sorted(f["id"] for f in model["fixtures"] if f["kind"] == "workload"),
                            "runs": {f["id"]: f.get("runs", 1) for f in model["fixtures"] if f["kind"] == "workload"}}
        runs = [f.get("runs", 1) for f in model["fixtures"] if f["kind"] == "workload"]
        if runs and max(runs) == 1 and suite in ("strata", "strata-depth"):
            # The review of 2026-10-31: four of eighteen tasks flipped between arms at one run each, which
            # cannot be told from noise. Say so on the card rather than let the arms be compared anyway.
            card["warning"] = ("one run per task: this arm cannot be distinguished from another at n=1. "
                               "Pass --runs 2 to compare arms; the campaign reports passes and attempts per task.")
    if snapshot["unknown_gpu_clients"]:
        card["next"] = ("STOP: another program holds the GPU. Tell the owner which, and wait until "
                        "they close it; then run prepare again.")
        return card, REFUSED
    card["next"] = ("Show this card to the owner and wait for a yes; then run with approval_token.")
    return card, OK


def _sees_images(runtime, profile, model):
    """Whether this profile takes image input: its pi entry says so (``input`` includes ``image``), or, without an
    entry, a llama.cpp preset loads a projector."""
    from .workload import pi_config
    entry = pi_config.production_entry(profile)
    if entry is not None:
        return "image" in (entry.get("input") or [])
    return runtime == "llama.cpp" and bool((model.get("settings") or {}).get("mmproj"))


def preset_overrides(sets=(), unsets=()):
    """``KEY=VALUE`` and ``KEY`` flags as one override map; ``None`` drops a setting."""
    overrides = {}
    for item in sets:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--preset-set needs KEY=VALUE, got {item!r}")
        overrides[key.strip()] = value.strip()
    for key in unsets:
        overrides[key.strip()] = None
    return overrides


def _first_live_workload(root):
    """True while the corpus holds no valid workload run at all."""
    db = Path(root) / "metrics.sqlite"
    if not db.exists():
        return True
    import sqlite3
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM task_runs t JOIN runs r ON r.run_id = t.run_id "
                            "WHERE r.valid_for_analysis = 1").fetchone()[0] == 0
    except sqlite3.OperationalError:
        return True
    finally:
        conn.close()


def prepare_next(profile=ladder.DEFAULT_PROFILE, mode="bare", retry=False, root=RUNS_ROOT,
                 restore_model=DEFAULT_RESTORE, preset=DEFAULT_PRESET, host=None, ladder_file=None,
                 integration_stamp=None, stage="screen", rung0_gate=None, **inputs):
    """Prepare whatever the ladder says is next for this profile and mode; the owner never picks
    flags, task ids or folders. Same read-only guarantees and exit codes as ``prepare``.

    ``stage`` is "screen" (default) or "full" (only for a profile chosen from `acceptance screen`). The CLI
    passes the rung-0 gate and integration stamp; passing None for either disables that gate in direct calls."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    step = ladder.next_step(root, profile, ladder.modes_of(mode), retry, inputs.get("task_dir"), ladder_file, stage)
    if step["status"] == "ladder_complete":
        return step, OK
    if step["status"] == "blocked":
        return step, REFUSED
    if rung0_gate is not None:
        report, gate = rung0.run(LIMITS)
        if gate != OK:
            return {"status": "blocked_rung0", "why": report["next"], "report": report,
                    "next": "STOP: rung 0 (the pre-pilot gate) failed. Fix the problems in `report` "
                            "(run `python3 -m llm_eval.acceptance rung0` for the detail), then `prepare --next` again."}, REFUSED
    if integration_stamp is not None and _first_live_workload(root):
        why = integration.blocker(integration_stamp)
        if why:
            return {"status": "blocked_integration", "why": why,
                    "next": f"STOP: {why}. Run `python3 -m llm_eval.acceptance integration` (bare; it runs the tier "
                            "outside the agent sandbox and stamps this source), then `prepare --next` again."}, REFUSED
    rung = step["rung_spec"]
    card, code = prepare(step["dir"], profile, restore_model, preset, host, suite=rung["suite"],
                         tasks=rung["task_ids"], runs=rung.get("runs"),
                         entry=ladder.entry_for(profile, step["mode"]), **inputs)
    done = sum(r["complete"] for r in step["progress"])
    result = {"status": "prepared" if code == OK else "refused", "rung": rung["id"], "mode": step["mode"],
              "rung_adds": rung.get("adds"), "tasks": step["needs"], "ladder": f"{done} of {len(step['progress'])} rungs done",
              "card": card}
    if code == OK:
        ladder.write_pending(root, dir=step["dir"], token=card["approval_token"], rung=rung["id"], mode=step["mode"],
                             profile=profile, stage=stage)
        events = f"{step['dir']}/runs/acceptance-{card['approval_token']}/events.jsonl"
        result["watch"] = (f"tail -n +1 -F {events} | grep --line-buffered -E "
                           "'section_complete|retry|interrupted|restoration|wl_task|wl_stop'")
        mode = card["workload"]["mode"]
        result["next"] = (f"Tell the owner in one line: {profile}, rung {rung['id']} ({rung.get('adds')}), entry {mode['entry']}, "
                          f"target context {mode['context']}, output limit {mode['output_limit']}; production stops and is restored. "
                          "If the owner has already said to run this stage, go straight on; otherwise ask \"Start it?\" once and "
                          "treat the yes as covering every remaining profile in this stage. Then run: "
                          f"python3 -m llm_eval.acceptance run --next --approve {card['approval_token']}")
    else:
        result["next"] = card["next"]
    return result, code


def run_next(approve, root=RUNS_ROOT, host=None, lock=MACHINE_LOCK, ladder_file=None, task_dir=None,
             today=None, integration_stamp=None, stage="screen"):
    """Run what ``prepare --next`` prepared. The token still binds the owner's yes to that exact plan."""
    root = Path(root)
    pending = ladder.read_pending(root)
    summary, code = run(pending["dir"], approve, False, host, lock, integration_stamp)
    if (Path(pending["dir"]) / "runs").exists():
        ladder.clear_pending(root)
    summary["rung"], summary["mode"] = pending["rung"], pending["mode"]
    try:  # the pre-filled record rows and the one-time follow-ups for this rung; bookkeeping never changes the outcome
        import time
        rung = next(r for r in ladder.load(ladder_file, task_dir) if r["id"] == pending["rung"])
        summary["record"] = ladder.record_draft(pending["dir"], summary, code, pending["rung"], pending["mode"],
                                                pending["profile"], today or time.strftime("%Y-%m-%d"))
        if rung.get("after"):
            summary["after_rung"] = rung["after"]
    except Exception as exc:
        summary["record_error"] = str(exc)[:200]
    try:
        upcoming = ladder.next_step(root, pending["profile"], [pending["mode"]], task_dir=task_dir, ladder_file=ladder_file,
                                    stage=pending.get("stage", stage))
        summary["after_this"] = {k: upcoming.get(k) for k in ("status", "rung", "why", "next") if upcoming.get(k)}
    except Exception as exc:  # a bookkeeping failure never changes the run's outcome
        summary["after_this_error"] = str(exc)[:200]
    return summary, code


def _load(directory, approve):
    plan = json.loads((Path(directory) / "plan.json").read_text())
    if approve != token(plan):
        raise PermissionError("approval token does not match this plan; ask the owner")
    return plan


def report(state, plan):
    model = plan["models"][0]["id"]
    sections = {k.split("__", 1)[1]: v for k, v in state["sections"].items() if k.startswith(model + "__")}
    def status(name):
        return sections.get(name, {}).get("status")
    qualification = sections.get("context_qualification", {})
    recall = sections.get("long_context_recall", {})
    tools = sections.get("structured_tools", {})
    trials = [v for k, v in sections.items() if k.startswith("trial-")]
    def peak(kind):
        used = [t["output_tokens"] for t in trials if t.get("output_tokens") is not None
                and t["section"].split("trial-", 1)[1].startswith(kind)]
        return max(used) if used else None
    passing_recall = [t["prompt_tokens"] for t in trials
                      if t["status"] == "pass" and t["section"].split("trial-", 1)[1].startswith("recall")]
    failing = sorted(name for name, value in sections.items()
                     if value["status"] not in ("pass", "skipped", "capacity_fallback"))
    summary = {
        "status": state["status"], "restoration_required": state["restoration_required"],
        "production_restore": state.get("last_restore"),
        "selected_context": qualification.get("selected_context"),
        "memory_class": qualification.get("memory_class"),
        "verified_recall_context": recall.get("verified_context"),
        "recall_descent": recall.get("descent"),
        "tools_passed": f"{tools.get('passed')}/{tools.get('total')}" if tools else None,
        "largest_passing_recall_prompt_tokens": max(passing_recall) if passing_recall else None,
        "peak_output_tokens": {"smoke": peak("smoke"), "recall": peak("recall"), "tools": peak("tool")},
        "hit_output_cap": sorted(t["section"].split("__", 1)[1] for t in trials if t.get("finish_reason") == "length"),
        "workload_output": _workload_output(sections),
        "sections": {name: status(name) for name in sorted(sections) if not name.startswith("trial-")},
        "failing_sections": failing,
        "failure": sections.get("failure", {}).get("reason"),
        "scorecard": scorecard(sections),
    }
    if state["restoration_required"] or state["status"] in ("restore_failed", "restoring"):
        summary["next"] = ("STOP: production is not verified restored. Do not retry, do not touch "
                           "systemd. Tell the owner now and include this output.")
        return summary, STOP
    if state["status"] == "blocked":
        summary["next"] = "Another program holds the GPU. Tell the owner; nothing was changed."
        return summary, REFUSED
    clean = state["status"] == "complete" and not failing
    summary["next"] = ("Report this summary to the owner." if clean else
                       "Production is restored. Report this summary and the failing sections to the owner; "
                       "do not rerun without them.")
    return summary, OK if clean else EVAL_FAILURES


def _workload_output(sections):
    """The largest single generation any workload step used, and the task-runs where a step ended on a length
    cutoff: what the output limit has to cover (red-team EF-004)."""
    runs = [v for k, v in sections.items() if k.startswith("trial-wl-")]
    peaks = [(r.get("metrics") or {}).get("peak_step_output_tokens") or 0 for r in runs]
    return {"peak_step_output_tokens": max(peaks, default=None),
            "length_finishes": sorted(r["section"].split("__", 1)[1] for r in runs
                                      if ((r.get("metrics") or {}).get("length_finishes") or 0) > 0)}


def scorecard(sections):
    """Per workload group: runs passed and the process numbers beside them. No single total."""
    card = {}
    for name, value in sorted(sections.items()):
        if not name.startswith("workload_") or value["status"] == "skipped":
            continue
        group = name.split("_", 1)[1]
        runs = [v for k, v in sections.items() if k.startswith("trial-wl-") and v.get("group") == group]
        metric = lambda m: sorted(r["metrics"].get(m) or 0 for r in runs)
        card[group] = {"passed": f"{value.get('passed')}/{value.get('total')}", "status": value["status"],
                       "median_steps": metric("steps")[len(runs) // 2] if runs else None,
                       "max_peak_context_tokens": max(metric("peak_context_tokens"), default=None),
                       "peak_step_output_tokens": max(metric("peak_step_output_tokens"), default=None),
                       "timeouts": sum(r.get("stop") == "wall_cap" for r in runs),
                       "incomplete": sum(bool(r.get("incomplete")) for r in runs),
                       "compactions": sum(metric("compactions")), "clarifications": sum(r.get("clarifications", 0) for r in runs)}
    return card


def _with_html(summary, code, run_dir, plan, state):
    """Best effort: a report that cannot be written never changes the outcome or the exit code."""
    try:
        summary["report_html"] = str(write_html(run_dir, plan, state).resolve())
    except OSError:
        pass
    # The corpus (llm_eval.metrics) is derived from the run folder; ingest is idempotent and never
    # changes the outcome. The database sits beside the profile folders, e.g. ~/llm-eval-runs/.
    try:
        from .metrics.ingest import ingest_all
        db = Path(run_dir).resolve().parents[2] / "metrics.sqlite"
        ingest_all(db, [run_dir])
        summary["metrics_db"] = str(db)
    except Exception as exc:
        summary["metrics_ingest_error"] = str(exc)[:200]
    return summary, code


def run(directory, approve, resume=False, host=None, lock=MACHINE_LOCK, integration_stamp=None):
    directory = Path(directory)
    plan = _load(directory, approve)
    identifier = safe_id(run_id(plan))
    started = (directory / "runs" / identifier).exists()
    if started != resume:
        raise PermissionError("this run has already started; use `status`, and `resume` only if the owner says so"
                              if started else "nothing to resume; this run has not started")
    if any(f["kind"] == "workload" for m in plan["models"] for f in m["fixtures"]) and not shutil.which("bwrap"):
        raise PermissionError("workload suites need bwrap; nothing was changed")
    if (integration_stamp is not None and not resume and plan["backend"] == "live"
            and any(f["kind"] == "workload" for m in plan["models"] for f in m["fixtures"])
            and _first_live_workload(directory.parent)):
        why = integration.blocker(integration_stamp)
        if why:  # the same gate as `prepare --next`, at the boundary both the ladder and the manual form share
            raise PermissionError(f"{why}; run `python3 -m llm_eval.acceptance integration` first. Nothing was changed")
    adapter = (LiveAdapter if plan.get("runtime", "llama.cpp") == "llama.cpp" else adapter_class(plan))(plan, directory / "runs" / identifier / "server-logs", host or RealHost(),
                          run_id=identifier)
    state = execute(plan, directory / "runs", identifier, resume=resume, adapter=adapter, lock_path=lock)
    return _with_html(*report(state, plan), directory / "runs" / identifier, plan, state)


def status(directory, lock=MACHINE_LOCK):
    """Read-only: summarize the run so far without touching the machine."""
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    state_path = directory / "runs" / run_id(plan) / "state.json"
    if not state_path.exists():
        return {"status": "not_started", "next": "Nothing has run yet."}, OK
    state = json.loads(state_path.read_text())
    if state["status"] == "refused_no_inhibit":
        touched = ("Production was restored first; the campaign was not started."
                   if state.get("last_restore") else "Production was not touched.")
        return {"status": "refused_no_inhibit",
                "next": f"The run was refused before it armed: no sleep lock could be taken. {touched} "
                        "Fix the cause, then `resume` with the owner's approval."}, REFUSED
    if state["status"] in ("planned", "running", "restoring"):
        try:
            with machine_lock(lock):
                pass
        except RuntimeError:
            done = sorted(k.split("__", 1)[1] for k in state["sections"] if "__" in k and "__trial-" not in k)
            return {"status": "running", "sections_done": done,
                    "next": "Still running. Check `status` again in 10 minutes; do nothing else."}, OK
        return {"status": "crashed", "last_status": state["status"],
                "restoration_required": state["restoration_required"],
                "next": "STOP: the runner died mid-run and production may be stopped. Tell the owner now; "
                        "`resume` (with the approval token) restores production first, but only on their say-so."}, STOP
    return _with_html(*report(state, plan), state_path.parent, plan, state)


def _inhibit_refusal(args, exc):
    """The sleep lock could not be taken. The run folder exists, so a plain `run` would say it already started;
    the way forward is `resume` (which also restores production first if a restore was pending)."""
    try:
        directory = args.dir or ladder.read_pending(RUNS_ROOT)["dir"]
    except (PermissionError, OSError, KeyError, ValueError):
        directory = "<run dir>"
    lead = ("Production was restored first if a restore was pending; the campaign was not started."
            if args.command == "resume" else "The campaign was not started and production was not touched.")
    return {"refused": f"{lead} No sleep lock: {exc}",
            "next": "Fix the cause (is systemd-inhibit installed and the system bus reachable?), tell the owner, and "
                    "on their say-so run: "
                    f"python3 -m llm_eval.acceptance resume --dir {directory} --approve {args.approve}"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m llm_eval.acceptance")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--runtime", choices=("llama.cpp", "strata"), default="llama.cpp")
    prep.add_argument("--strata-config")
    prep.add_argument("--strata-python")
    prep.add_argument("--strata-server")
    prep.add_argument('--request-settings', help='JSON file of explicit request settings (default: Strata\'s '
                      'pinned request, or a llama.cpp profile\'s preset sampling)')
    prep.add_argument("--profile", default=None, help="default muse-glimmer-30b-q4 with --next; required otherwise")
    prep.add_argument("--restore-model", default=DEFAULT_RESTORE)
    prep.add_argument("--preset", default=str(DEFAULT_PRESET))
    prep.add_argument("--preset-set", action="append", default=[], metavar="KEY=VALUE",
                      help="llama.cpp: change one preset setting for this plan only (a paired arm's knob)")
    prep.add_argument("--preset-unset", action="append", default=[], metavar="KEY",
                      help="llama.cpp: drop one preset setting for this plan only")
    prep.add_argument("--suite", default="classic", choices=workload.SUITES)
    prep.add_argument("--task", action="append", help="run only this task id from the suite (repeatable)")
    prep.add_argument("--runs", type=int, help="runs per task, overriding the task's own count")
    prep.add_argument("--entry", help="opencode model entry to use as the mode (default: the profile's own bare entry)")
    prep.add_argument("--harness", default="opencode", choices=("opencode", "pi"),
                      help="which agent harness runs the workload tasks (default opencode)")
    prep.add_argument("--thinking",
                      help="with --harness pi: pi's thinking level (default: the owner's defaultThinkingLevel for a "
                           "production entry, else high)")
    prep.add_argument("--compaction-at", type=int,
                      help="with --harness pi: the context size at which pi compacts (default: context - output - margin)")
    prep.add_argument("--context", type=int, help="override the target context (default 131,072, stepping down to what the model qualifies at)")
    prep.add_argument("--output", type=int, help="override the output limit (default 16,384)")
    prep.add_argument("--no-step-down", action="store_true",
                      help="with --context: load only that context; a model that cannot hold it fails capacity "
                           "instead of falling back to a lower rung (needed for paired campaigns)")
    prep.add_argument("--keep-going", action="store_true", help="continue after an infrastructure failure instead of stopping")
    prep.add_argument("--next", action="store_true", help="prepare the next rung of the ladder; no other flags needed")
    prep.add_argument("--mode", default="bare", choices=ladder.MODES, help="with --next: which opencode entry")
    prep.add_argument("--retry", action="store_true", help="with --next: repeat a rung whose last attempt was stopped")
    prep.add_argument("--stage", default="screen", choices=("screen", "full"),
                      help="with --next: 'screen' (default, every profile) or 'full' (only a profile chosen from `screen`)")
    for name in ("run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--approve", required=True)
    commands.choices["run"].add_argument("--next", action="store_true", help="run what `prepare --next` prepared")
    commands.add_parser("status")
    commands.add_parser("integration", help="run the integration tier and stamp this source (first live run needs it)")
    screen_cmd = commands.add_parser("screen", help="read-only: the screen matrix, the ranking and the top 3")
    screen_cmd.add_argument("--preset", default=str(DEFAULT_PRESET))
    commands.add_parser("rung0", help="pre-pilot gate: grading isolation and output caps (no GPU)")
    table = commands.add_parser("ladder", help="read-only: which rungs are done for a profile")
    table.add_argument("--profile", default=ladder.DEFAULT_PROFILE)
    table.add_argument("--mode", default="bare", choices=ladder.MODES)
    for command in commands.choices.values():
        command.add_argument("--dir", required=False)
    args = parser.parse_args(argv)
    try:
        if args.command == "ladder":
            modes = ladder.modes_of(args.mode)
            result, code = {"profile": args.profile, "rungs": ladder.table(RUNS_ROOT, args.profile, modes),
                            "next": ladder.next_step(RUNS_ROOT, args.profile, modes, stage="full").get("rung")}, OK
        elif args.command == "screen":
            result, code = screen.report(RUNS_ROOT, args.preset), OK
        elif args.command == "prepare" and args.next:
            if args.runtime != "llama.cpp":
                parser.error("Strata comparisons use manual prepare, not the llama.cpp ladder")
            if args.context is not None:
                parser.error("--context is not supported with --next (those runs would not advance the ladder); use the manual form")
            if args.output is not None and args.stage != "full":
                parser.error("--output applies to --stage full only: the screen is run at its fixed output limit")
            result, code = prepare_next(args.profile or ladder.DEFAULT_PROFILE, args.mode, args.retry,
                                        restore_model=args.restore_model, preset=args.preset,
                                        integration_stamp=integration.STAMP, stage=args.stage,
                                        rung0_gate=rung0.GATE,
                                        **({"output": args.output} if args.output is not None else {}))
        elif args.command == "run" and args.next:
            result, code = run_next(args.approve, integration_stamp=integration.STAMP)
        elif args.command == "integration":
            result, code = integration.run_and_stamp()
        elif args.command == "rung0":
            result, code = rung0.run(LIMITS)
        elif args.command in ("prepare", "status", "run", "resume") and not args.dir:
            parser.error(f"{args.command} needs --dir (or use `prepare --next` / `run --next`)")
        elif args.command == "prepare":
            if not args.profile:
                parser.error("prepare needs --profile")
            result, code = prepare(args.dir, args.profile, args.restore_model, args.preset, suite=args.suite,
                                   tasks=args.task, runs=args.runs, entry=args.entry, context=args.context,
                                   keep_going=args.keep_going, output=args.output, runtime=args.runtime,
                                   harness=args.harness, thinking=args.thinking, compaction_at=args.compaction_at,
                                   strata_config=args.strata_config, strata_python=args.strata_python,
                                   strata_server=args.strata_server, single_rung=args.no_step_down,
                                   preset_overrides=preset_overrides(args.preset_set, args.preset_unset),
                                   request_settings=json.loads(Path(args.request_settings).read_text()) if args.request_settings else None)
        elif args.command == "status":
            result, code = status(args.dir)
        else:
            result, code = run(args.dir, args.approve, resume=args.command == "resume",
                               integration_stamp=integration.STAMP)
    except inhibit.InhibitUnavailable as exc:  # before the generic PermissionError clause: the text differs
        result, code = _inhibit_refusal(args, exc), REFUSED
    except PermissionError as exc:
        result, code = {"refused": str(exc), "next": "Nothing was changed. Tell the owner and wait."}, REFUSED
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        # execute() restores in its own finally; anything reaching here may still have stopped production.
        if args.command in ("run", "resume"):
            result, code = {"error": str(exc), "next": "STOP: run `status` and tell the owner, including this output."}, STOP
        else:
            result, code = {"error": str(exc), "next": "Nothing was changed. Tell the owner, including this output."}, REFUSED
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
