"""Run folder -> warehouse rows. Idempotent: ingesting a run replaces all of its rows in one
transaction, so a resumed or re-reported run converges and an ingest fix is a re-ingest."""
import hashlib
import json
from pathlib import Path
import re
import time

from .logparse import last_stamp, parse_server_log
from .schema import TABLES, connect

# Runs known to be unusable for statistics, keyed by the start of their plan hash. A run folder can
# also carry its own ``analysis-notes.json`` ({"valid_for_analysis": false, "reason": "..."}).
KNOWN_INVALID = {
    "07fd060bc4e3": "first live run: 768-token recall and 64-token smoke caps starved reasoning, so failures are "
                    "a methodology gap, not a model verdict (STATE 2026-10-03)",
}
QUANT = re.compile(r"(?i)[-_.]((?:I?Q\d(?:_[A-Z0-9]+)*)|BF16|F16|FP8|MXFP4)(?=[-_.])")
LOG_CONTEXT = re.compile(r"-(\d+)-\d+\.log$")
TIMING_KEYS = {"prompt_n": "prompt_n", "cache_n": "cache_n", "prompt_ms": "prompt_ms", "predicted_n": "predicted_n",
               "predicted_ms": "predicted_ms", "prompt_per_second": "prompt_tps", "predicted_per_second": "predicted_tps"}


REWRITTEN = Path(__file__).resolve().parents[2] / "configs" / "rewritten-shas.json"


def main_sha(sha, mapping=None):
    """The main-branch commit for a harness SHA a run recorded. History rewritten on 2026-11-06 left runs naming
    commits that live only on a backup branch; each maps to the main commit with the same tree."""
    if mapping is None:
        mapping = (_read(REWRITTEN, {}) or {}).get("map", {})
    return mapping.get(sha, sha) if sha else sha


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _dump(value):
    return None if value is None else json.dumps(value, sort_keys=True)


def _task_hash(task, plan_files):
    """The task spec plus the pinned digest of every snapshot file it was built from."""
    needle = f"/snapshots/{task['id']}/"
    pinned = sorted((p.split(needle, 1)[1], d) for p, d in plan_files.items() if needle in p)
    return canonical_sha({"task": task, "snapshot": pinned})


def spec_sha(task):
    """The task's own spec, without per-run settings (``runs``, the plan's ``_path``) or its snapshot: what the
    ladder can recompute from a fixture file to tell whether a recorded run was of this version of the task."""
    return canonical_sha({k: v for k, v in task.items() if k not in ("runs", "_path")})


def task_identity_sha(task, plan_files):
    """V1 fixture identity: normalized spec and pinned snapshot, not repeat allocation.

    Keep the original task_hash for legacy consumers; re-ingestion derives this
    identity from saved plans rather than guessing equivalence from old hashes.
    """
    needle = f"/snapshots/{task['id']}/"
    pinned = sorted((p.split(needle, 1)[1], d) for p, d in plan_files.items() if needle in p)
    return canonical_sha({"version": 1, "spec_sha": spec_sha(task), "snapshot": pinned})


def _runtime(environment, sections):
    """The runtime record, with the version the engine itself printed at load. environment.json is written
    before the load and its build file can be stale, so the load's own line wins when they disagree."""
    runtime = environment.get("runtime")
    if not isinstance(runtime, dict):
        return runtime
    runtime = dict(runtime)
    for key, section in sections.items():
        if key.endswith("__anchor_load"):
            for attempt in section.get("attempts") or []:
                version = (attempt.get("memory") or {}).get("engine_version")
                if version:
                    runtime["engine_version"] = version
    built = (runtime.get("engine_build") or {}).get("version")
    if runtime.get("engine_version") and built and built != runtime["engine_version"]:
        runtime["engine_build_stale"] = True
    return runtime


def _suite(plan):
    kinds = {f["kind"] for m in plan["models"] for f in m["fixtures"]}
    categories = sorted({f["category"] for m in plan["models"] for f in m["fixtures"] if f["kind"] == "workload"})
    if "workload" not in kinds:
        return "classic", "classic"
    return "workload", "workload:" + ",".join(categories)


def _validity(run_dir, plan, state):
    notes = _read(Path(run_dir) / "analysis-notes.json", {})
    if "valid_for_analysis" in notes:
        return int(bool(notes["valid_for_analysis"])), notes.get("reason")
    token = plan.get("plan_hash", "")
    for prefix, reason in KNOWN_INVALID.items():
        if token.startswith(prefix):
            return 0, reason
    if state.get("status") != "complete":
        return 0, f"run status is {state.get('status')!r}, not complete"
    return 1, None


def put(conn, table, replace=False, **columns):
    """Named-column insert, so a schema change cannot silently shift values into the wrong column."""
    names = ",".join(columns)
    marks = ",".join("?" * len(columns))
    conn.execute(f"INSERT {'OR REPLACE ' if replace else ''}INTO {table} ({names}) VALUES ({marks})", tuple(columns.values()))


def _delete(conn, run_id):
    for table in TABLES:
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))


def _profile_of(key, profiles):
    head = key.split("__", 1)[0]
    return head if "__" in key and head in profiles else None


def ingest_run(conn, run_dir, *, now=None):
    """Replace one run's rows. Returns a summary of what was written."""
    run_dir = Path(run_dir).resolve()  # one spelling per run, whichever name of the data folder was given
    state, plan = _read(run_dir / "state.json"), _read(run_dir / "plan.json")
    if not state or not plan:
        raise ValueError(f"{run_dir} is not a run folder (needs state.json and plan.json)")
    run_id = run_dir.name
    environment = _read(run_dir / "environment.json", {})
    sections = state.get("sections", {})
    times = [s["time"] for s in sections.values() if s.get("time")]
    kind, label = _suite(plan)
    valid, reason = _validity(run_dir, plan, state)
    workload = next((m["workload"] for m in plan["models"] if m.get("workload")), {})
    server = environment.get("llama_server") or {}
    runtime = plan.get('runtime', 'llama.cpp')
    gpu = environment.get("gpu") or {}
    harness = environment.get("harness") or {}
    restore = sections.get("06_restore_and_verify", {}).get("verification") or state.get("last_restore") or {}
    verified = int(bool(restore) and all(restore.get(k) for k in ("service_matches", "health", "four_token_probe")))
    profiles = {m["id"] for m in plan["models"]}
    counts = {}
    with conn:
        _delete(conn, run_id)
        put(conn, "runs", run_id=run_id, run_dir=str(run_dir), plan_hash=plan.get("plan_hash"),
            backend=plan.get("backend"), suite_kind=kind, suite_label=label, status=state.get("status"),
            restoration_required=int(bool(state.get("restoration_required"))), restore_verified=verified,
            started_at=min(times, default=None), ended_at=max(times, default=None),
            wall_seconds=(max(times) - min(times)) if times else None, harness_sha=main_sha(harness.get("sha")),
            harness_dirty=None if harness.get("dirty") is None else int(bool(harness["dirty"])),
            opencode_version=workload.get("opencode_version"),
            harness_name=(workload.get("harness") or "opencode") if workload else None,
            harness_version=(workload.get("binary_version") if workload.get("harness") == "pi"
                             else workload.get("opencode_version")) if workload else None,
            runtime_name=runtime, runtime_json=_dump(_runtime(environment, sections)),
            llama_server_path=(server.get("path") or _llama_path(state)) if runtime == 'llama.cpp' else None,
            llama_server_version=server.get("version"), gpu_name=gpu.get("name"), gpu_driver=gpu.get("driver"),
            gpu_memory_mib=gpu.get("memory_total_mib"), kernel=environment.get("kernel"), valid_for_analysis=valid,
            invalid_reason=reason, limits_json=_dump(plan.get("limits")),
            environment_json=_dump(environment) if environment else None, ingested_at=now or time.time())
        for model in plan["models"]:
            _model_rows(conn, run_id, plan, model, sections)
        _section_rows(conn, run_id, plan, sections, profiles, run_dir, counts)
        _event_rows(conn, run_id, run_dir)
        counts["gpu_samples"] = _gpu_rows(conn, run_id, run_dir)
        counts["server_requests"] = _log_rows(conn, run_id, run_dir) if runtime == 'llama.cpp' else 0
        counts['inference_requests'] = _inference_rows(conn, run_id, run_dir)
        counts.update(_judge_rows(conn, run_id, run_dir))
    return {"run_id": run_id, "valid_for_analysis": bool(valid), **counts}


def _llama_path(state):
    exec_start = (state.get("snapshot") or {}).get("exec_start") or ""
    match = re.search(r"path=(\S+)", exec_start)
    return match[1] if match else None


def _model_rows(conn, run_id, plan, model, sections):
    from ..runner import _group
    name = model["id"]
    settings = model.get("settings") or {}
    mode = (model.get("workload") or {}).get("mode") or {}
    path = settings.get("model")
    if plan.get('runtime') == 'strata':
        args = settings.get('resolved_config', {}).get('args', [])
        path = args[args.index('--native') + 1] if '--native' in args else None
    quant = QUANT.search(Path(path).name) if path else None
    qualification = sections.get(f"{name}__context_qualification", {})
    recall = sections.get(f"{name}__long_context_recall", {})
    workload_plan = {"version": 1, "tasks": [
        {"id": f["id"], "category": f["category"], "group": _group(f), "runs": f.get("runs", 1),
         "task_spec_sha": spec_sha(f), "task_identity_sha": task_identity_sha(f, plan.get("files", {}))}
        for f in model["fixtures"] if f["kind"] == "workload"]}
    put(conn, "run_models", run_id=run_id, profile=name, covers_json=_dump(model.get("covers")),
        anchor=model.get("anchor"), contexts_json=_dump(model.get("contexts")), model_path=path,
        quant=quant[1].upper() if quant else None, artifacts_json=_dump(model.get("artifacts")),
        settings_json=_dump(settings), preset_sha256=(plan.get("production", {}).get("preset") or {}).get("sha256"),
        restore_model=plan.get("production", {}).get("restore_model"), mode_entry=mode.get("entry"),
        mode_context=mode.get("context"), mode_output_limit=mode.get("output_limit"),
        mode_compaction_point=mode.get("compaction_point"),
        mode_entry_explicit=None if "entry_explicit" not in mode else int(mode["entry_explicit"]),
        mode_context_overridden=None if "context_overridden" not in mode else int(mode["context_overridden"]),
        selected_context=qualification.get("selected_context"), verified_recall_context=recall.get("verified_context"),
        workload_plan_json=_dump(workload_plan),
        effort=(model.get("workload") or {}).get("thinking_effective") or (model.get("workload") or {}).get("thinking"))
    for attempt in qualification.get("attempts", []):
        put(conn, "load_attempts", run_id=run_id, profile=name, context=attempt.get("context"),
            status=attempt.get("status"), free_mib=attempt.get("free_mib"), fixed_mib=attempt.get("fixed_mib"),
            kv_mib=attempt.get("kv_mib"), load_seconds=attempt.get("load_seconds"),
            predicted_free_mib=attempt.get("predicted_free_mib"), buffers_source=attempt.get("buffers_source"))


def _section_rows(conn, run_id, plan, sections, profiles, run_dir, counts):
    files = plan.get("files", {})
    fixtures = {(m["id"], f["id"]): f for m in plan["models"] for f in m["fixtures"]}
    counts.update(trials=0, task_runs=0, checks=0, sections=len(sections))
    for key, value in sections.items():
        profile = _profile_of(key, profiles)
        name = key.split("__", 1)[1] if profile else key
        put(conn, "sections", run_id=run_id, key=key, status=value.get("status"), time=value.get("time"),
            reason=value.get("reason"))
        if profile and name.startswith("trial-wl-"):
            _task_run(conn, run_id, profile, key, value, fixtures, files, run_dir)
            counts["task_runs"] += 1
            counts["checks"] += len(value.get("checks") or [])
        elif profile and name.startswith("trial-"):
            _trial(conn, run_id, profile, key, value, fixtures)
            counts["trials"] += 1


def _trial(conn, run_id, profile, key, value, fixtures):
    fixture = fixtures.get((profile, value.get("fixture")))
    timings = value.get("timings") or {}
    mapped = {ours: timings.get(theirs) for theirs, ours in TIMING_KEYS.items()}
    put(conn, "trials", run_id=run_id, profile=profile, section=key, fixture_id=value.get("fixture"),
        fixture_kind=fixture.get("kind") if fixture else None, fixture_hash=canonical_sha(fixture) if fixture else None,
        context=value.get("context"), status=value.get("status"), prompt_tokens=value.get("prompt_tokens"),
        output_tokens=value.get("output_tokens"), max_output_tokens=value.get("max_output_tokens"),
        finish_reason=value.get("finish_reason"), fits=None if value.get("fits") is None else int(bool(value["fits"])),
        request_seconds=value.get("request_seconds"), usage_json=_dump(value.get("usage")),
        grading_json=_dump(value.get("grading")), recorded_at=value.get("time"), **mapped)


def _task_run(conn, run_id, profile, key, value, fixtures, files, run_dir):
    task = fixtures.get((profile, value.get("task")))
    metrics = dict(value.get("metrics") or {})
    answer, commands = metrics.pop("answer", ""), metrics.pop("commands", [])
    web = value.get("web") or {}
    status = value.get("status")
    failure = {"pass": None, "quality_fail": "model", "infra_fail": "infra", "capacity_fail": "capacity"}.get(status, status)
    if value.get("incomplete"):  # cut off by a serving limit: not a wrong answer, and not a pass either
        failure = "incomplete"
    export = Path(value["output"]) / "export.json" if value.get("output") else None
    if export and not export.exists() and value.get("output"):  # a moved run folder: look beside the section files
        export = Path(run_dir) / "workload" / Path(value["output"]).name / "export.json"
    put(conn, "task_runs", run_id=run_id, profile=profile, section=key, task_id=value.get("task"),
        task_hash=_task_hash(task, files) if task else None,
        task_spec_sha=spec_sha(task) if task else None, category=value.get("category"),
        task_identity_sha=task_identity_sha(task, files) if task else None,
        group_name=value.get("group"), run_n=value.get("run"), context=value.get("context"), status=status,
        passed=int(status == "pass"), failure_class=failure, stop_reason=value.get("stop"), turns=value.get("turns"),
        clarifications=value.get("clarifications"), tests_pasted=value.get("tests_pasted"), steps=metrics.get("steps"),
        tool_calls=metrics.get("tool_calls"), tools_json=_dump(metrics.get("tools")),
        invalid_calls=metrics.get("invalid_calls"), tool_errors=metrics.get("tool_errors"),
        compactions=metrics.get("compactions"), peak_context_tokens=metrics.get("peak_context_tokens"),
        output_tokens=metrics.get("output_tokens"), reasoning_tokens=metrics.get("reasoning_tokens"),
        peak_step_output_tokens=metrics.get("peak_step_output_tokens"), length_finishes=metrics.get("length_finishes"),
        target_context=value.get("target_context"), compaction_point=value.get("compaction_point"),
        model_seconds=metrics.get("model_seconds"), tool_seconds=metrics.get("tool_seconds"),
        wall_seconds=metrics.get("wall_seconds"), user_turns=metrics.get("user_turns"),
        ended_with_question=None if "ended_with_question" not in metrics else int(bool(metrics["ended_with_question"])),
        harness_restarts=metrics.get("harness_restarts"), harness_lingered_turns=metrics.get("harness_lingered_turns"),
        web_hits=web.get("hits"), web_misses=len(web["misses"]) if web.get("misses") is not None else None,
        answer_chars=len(answer or ""), command_count=len(commands or []),
        export_path=str(export) if export and export.exists() else None,
        export_sha256=_file_sha(export) if export else None, metrics_json=_dump(metrics), recorded_at=value.get("time"),
        graders_sha=value.get("graders_sha"))
    for index, check in enumerate(value.get("checks") or []):
        detail = {k: v for k, v in check.items() if k not in ("type", "pass")}
        put(conn, "checks", run_id=run_id, section=key, idx=index, type=check.get("type"),
            passed=int(bool(check.get("pass"))), detail_json=_dump(detail))


def _event_rows(conn, run_id, run_dir):
    path = run_dir / "events.jsonl"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            break  # a torn tail; the journal truncates it on resume
        extra = {k: v for k, v in item.items() if k not in ("sequence", "time", "event", "section", "status")}
        put(conn, "events", replace=True, run_id=run_id, sequence=item["sequence"], time=item.get("time"),
            event=item.get("event"), section=item.get("section"), status=item.get("status"),
            payload_json=_dump(extra) if extra else None)


def _gpu_rows(conn, run_id, run_dir):
    path = run_dir / "telemetry" / "gpu.jsonl"
    count = 0
    if not path.exists():
        return count
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        put(conn, "gpu_samples", run_id=run_id, t=row.get("t"), tag=row.get("tag"), mem_used_mib=row.get("mem_used_mib"),
            mem_free_mib=row.get("mem_free_mib"), util_gpu_pct=row.get("util_gpu_pct"),
            util_mem_pct=row.get("util_mem_pct"), temp_c=row.get("temp_c"), power_w=row.get("power_w"),
            clock_sm_mhz=row.get("clock_sm_mhz"), clock_mem_mhz=row.get("clock_mem_mhz"), error=row.get("error"))
        count += 1
    return count


def _log_rows(conn, run_id, run_dir):
    count = 0
    for path in sorted((run_dir / "server-logs").glob("*.log")):
        text = path.read_text(errors="replace")
        end = last_stamp(text)
        context = LOG_CONTEXT.search(path.name)
        mtime = path.stat().st_mtime
        for r in parse_server_log(text):
            epoch = None if end is None or r["t_log_s"] is None else round(mtime - end + r["t_log_s"], 3)
            put(conn, "server_requests", replace=True, run_id=run_id, log_file=path.name,
                load_context=int(context[1]) if context else None, task=r["task"], slot_similarity=r["slot_similarity"],
                prompt_n=r["prompt_n"], prompt_ms=r["prompt_ms"], prompt_tps=r["prompt_tps"],
                predicted_n=r["predicted_n"], predicted_ms=r["predicted_ms"], predicted_tps=r["predicted_tps"],
                total_ms=r["total_ms"], total_tokens=r["total_tokens"], ctx_tokens_after=r["ctx_tokens_after"],
                cache_n_est=r["cache_n_est"], graphs_reused=r["graphs_reused"], truncated=r["truncated"],
                t_epoch_est=epoch)
            count += 1
    return count


def _inference_rows(conn, run_id, run_dir):
    count = 0
    for path in sorted(Path(run_dir).glob('workload/*/raw/inference/*/request.json')):
        row = _read(path)
        if not row or row.get('path') != '/v1/chat/completions':
            continue
        response = row.get('response', {})
        put(conn, 'inference_requests', run_id=run_id, file=str(path.relative_to(run_dir)),
            request_n=row.get('request'), turn_n=row.get('turn'), status=row.get('status'),
            started=row.get('started'), ended=row.get('ended'), ttft_seconds=response.get('first_token_seconds'),
            first_answer_seconds=response.get('first_answer_seconds'), first_tool_seconds=response.get('first_tool_seconds'),
            finish_reason=response.get('finish_reason'), usage_json=_dump(response.get('usage')),
            timings_json=_dump(response.get('timings')), contract_json=_dump({
                k: row.get(k) for k in ('effective_body', 'supplied', 'conflicts', 'error')}))
        count += 1
    return count


def _judge_rows(conn, run_id, run_dir):
    """The offline judge's output, if it has been run on this folder. Scores sit beside the gates."""
    data = _read(run_dir / "judge.json")
    if not data:
        return {"judge_scores": 0}
    endpoint = data.get("endpoint") or {}
    common = dict(judge_model=endpoint.get("model"), judge_base_url=endpoint.get("base_url"),
                  against_run_id=data.get("against"), rubric_sha=data.get("rubric_sha"), judged_at=data.get("judged_at"))
    count = 0
    for result in data.get("results", []):
        if "verdict" in result:  # pairwise: first = this run, second = the opponent
            verdict = dict(result["verdict"])
            overall = verdict.pop("overall", None)
            for criterion, value in verdict.items():
                put(conn, "judge_scores", run_id=run_id, task_id=result["task"], mode="pairwise", criterion=criterion,
                    value=value, overall=overall, reason=None, **common)
                count += 1
        else:
            for criterion, value in (result.get("scores") or {}).items():
                put(conn, "judge_scores", run_id=run_id, task_id=result["task"], mode="reference", criterion=criterion,
                    value=None if value is None else str(value), overall=None, reason=result.get("reason"), **common)
                count += 1
    return {"judge_scores": count}


def find_runs(root):
    """Every run folder under ``root`` (a ``runs/<id>`` directory holding state.json and plan.json)."""
    root = Path(root)
    return sorted(p.parent for p in root.glob("**/state.json") if (p.parent / "plan.json").exists()
                  and "snapshots" not in p.parts and "workload" not in p.parts)


def ingest_all(db_path, runs):
    conn = connect(db_path)
    try:
        return [ingest_run(conn, run) for run in runs]
    finally:
        conn.close()
