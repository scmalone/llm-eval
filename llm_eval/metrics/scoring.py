"""Equal-task fixed-suite scores from explicit, version-pinned warehouse cohorts.

No schema writes, inference of independent attempts, or automatic cohort selection.
See plans/2026-10-05-scoring-and-coverage-spec.md for the statistical contract.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

from ..workload.tasks import CATEGORIES, COLLECTIONS

METHOD = "equal-task-campaign-hoeffding-v2"
RUN_FIELDS = ("backend", "harness_sha", "harness_dirty", "opencode_version",
              "llama_server_version", "gpu_name", "gpu_driver", "gpu_memory_mib", "kernel", "limits_json",
              "runtime_name", "runtime_json")
MODEL_FIELDS = ("profile", "model_path", "quant", "artifacts_json", "settings_json",
                "mode_entry", "mode_output_limit")
CHANGE_FIELDS = set(MODEL_FIELDS) | {"context", "target_context", "compaction_point", "llama_server_version", "runtime_json"}
PROVENANCE = ("backend", "harness_sha", "opencode_version", "llama_server_version", "gpu_name",
              "gpu_driver", "gpu_memory_mib", "kernel", "limits_json", "artifacts_json", "settings_json",
              "mode_entry", "mode_output_limit", "context", "target_context", "compaction_point")
OUTCOMES = {"pass": 1, "quality_fail": 0, "timeout": 0}
FAILURES = {"pass": None, "quality_fail": "model", "timeout": "timeout"}
# A trial cut off by output_cap or wall_cap keeps its quality_fail status and is ingested as "incomplete": it is a
# failure in the quality denominator (adapter spec: "record any truncation as a failure"; scoring spec: timeouts
# count as failures), reported under its own label so it is not read as a wrong answer.
TRUNCATED = "incomplete"


def _failure_classes(status):
    expected = FAILURES[status]
    return (expected, TRUNCATED) if status == "quality_fail" else (expected,)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a nonempty string")
    return value


def _keys(value, allowed, where):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{where} must be an object with only {sorted(allowed)}")


def _group(group):
    if not isinstance(group, str):
        return None
    if group in CATEGORIES or group == "core":
        return "core"
    if group in COLLECTIONS:  # mined, game and vision score apart from the 35-task core
        return group
    return {"chat_sweep": "diagnostic", "diagnostic": "diagnostic", "depth": "depth"}.get(group)


def _identity_key(request):
    return "task_identity_sha" if request["version"] == 2 else "task_hash"


def validate(request):
    _keys(request, {"version", "suite", "candidate", "baseline", "allowed_changes", "inference"}, "request")
    if type(request.get("version")) is not int or request["version"] not in (1, 2):
        raise ValueError("request version must be 1 (legacy) or 2")
    suite = request.get("suite")
    _keys(suite, {"id", "group", "tasks"}, "suite")
    _text(suite.get("id"), "suite.id")
    if suite.get("group") not in ("core", "depth", "diagnostic"):
        raise ValueError("suite.group must be core, depth, or diagnostic")
    if not isinstance(suite.get("tasks"), list) or not suite["tasks"]:
        raise ValueError("suite.tasks must be a nonempty list")
    ids = set()
    for task in suite["tasks"]:
        identity = _identity_key(request)
        _keys(task, {"id", "category", "task_spec_sha", identity, "family"}, "suite task")
        task_id = _text(task.get("id"), "task.id")
        _text(task.get("category"), "task.category")
        if task_id in ids:
            raise ValueError(f"duplicate task {task_id}")
        ids.add(task_id)
        for key in ("task_spec_sha", identity):
            if not re.fullmatch(r"[0-9a-f]{64}", str(task.get(key, ""))):
                raise ValueError(f"{task_id}.{key} must be a SHA256 digest")
        if "family" in task:
            _text(task["family"], "task.family")
    changes = request.get("allowed_changes", [])
    if not isinstance(changes, list) or any(not isinstance(k, str) or k not in CHANGE_FIELDS for k in changes):
        raise ValueError(f"allowed_changes must be a list drawn from {sorted(CHANGE_FIELDS)}")
    inference = request.get("inference", {})
    _keys(inference, {"confidence", "independent_blocks", "fixed_design", "note"}, "inference")
    confidence = inference.get("confidence", 0.95)
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    for key in ("independent_blocks", "fixed_design"):
        if key in inference and type(inference[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if "note" in inference:
        _text(inference["note"], "inference.note")
    selected = set()
    for name in ("candidate", "baseline"):
        if name == "baseline" and name not in request:
            continue
        arm = request.get(name)
        _keys(arm, {"label", "selections"}, name)
        _text(arm.get("label"), name + ".label")
        if not isinstance(arm.get("selections"), list) or not arm["selections"]:
            raise ValueError(f"{name}.selections must be nonempty")
        for selection in arm["selections"]:
            _keys(selection, {"run_id", "profile", "block", "expected_attempts"}, "selection")
            pair = (_text(selection.get("run_id"), "run_id"), _text(selection.get("profile"), "profile"))
            if pair in selected:
                raise ValueError(f"duplicate or overlapping selection {pair}")
            selected.add(pair)
            if "block" in selection:
                _text(selection["block"], "block")
            if "expected_attempts" in selection:
                counts = selection["expected_attempts"]
                if not isinstance(counts, dict) or not counts or set(counts) - ids or any(
                        type(n) is not int or n < 1 for n in counts.values()):
                    raise ValueError("expected_attempts needs known task IDs and positive integer counts")
    return request


def _config(run, model, task):
    config = {k: run[k] if k in run.keys() else None for k in RUN_FIELDS}
    config['runtime_name'] = config['runtime_name'] or 'llama.cpp'
    config.update({k: model[k] for k in MODEL_FIELDS})
    config.update({k: task[k] for k in ("context", "target_context", "compaction_point")})
    for key in ("limits_json", "artifacts_json", "settings_json"):
        if config[key] is not None:
            config[key] = json.loads(config[key])
    return config


def _provenance(config):
    warnings = set()
    numeric = {"gpu_memory_mib", "mode_output_limit", "context", "target_context", "compaction_point"}
    for key in PROVENANCE:
        if key == 'llama_server_version' and config['runtime_name'] == 'strata':
            continue
        if key.endswith("_json"):
            continue
        value = config[key]
        if value is None or value == "":
            warnings.add("missing_provenance:" + key)
        elif key in numeric:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid numeric provenance: {key}")
            if key != "gpu_memory_mib" and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        elif not isinstance(value, str):
            raise ValueError(f"invalid string provenance: {key}")
        elif not value.strip():
            warnings.add("missing_provenance:" + key)
    for key in ("settings_json", "limits_json", "artifacts_json"):
        value = config[key]
        kind = list if key == "artifacts_json" else dict
        if value is None:
            warnings.add("missing_provenance:" + key)
            continue
        if not isinstance(value, kind):
            raise ValueError(f"{key} must contain a JSON {kind.__name__}")
        if not value:
            warnings.add("missing_provenance:" + key)
            continue
        if key == "artifacts_json":
            for artifact in value:
                if not isinstance(artifact, dict):
                    raise ValueError("artifacts_json entries must be objects")
                if (not all(isinstance(artifact.get(k), str) and artifact[k].strip() for k in ("path", "role"))
                        or not isinstance(artifact.get("sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])):
                    raise ValueError("artifact provenance needs path, role, and SHA256")
            if not any(a["role"] == "model" for a in value) and config['runtime_name'] != 'strata':
                warnings.add("missing_provenance:model_artifact")
        elif key == "settings_json":
            if config['runtime_name'] == 'strata':
                if not value.get('pinned_files') or not value.get('resolved_config'):
                    warnings.add('missing_provenance:strata_pinned_settings')
            elif "model" not in value:
                warnings.add("missing_provenance:settings_json.model")
            else:
                _text(value["model"], "settings_json.model")
        else:
            for limit in ("campaign_seconds", "model_seconds", "load_seconds", "request_seconds"):
                if limit not in value:
                    warnings.add("missing_provenance:limits_json." + limit)
                elif type(value[limit]) not in (int, float) or not math.isfinite(value[limit]) or value[limit] <= 0:
                    raise ValueError(f"invalid limit provenance: {limit}")
    if config["harness_dirty"] != 0:
        warnings.add("unverified_clean_harness")
    if config['runtime_name'] == 'strata':
        runtime = json.loads(config['runtime_json']) if config.get('runtime_json') else {}
        if not runtime.get('sha256') or not runtime.get('engine_build'):
            warnings.add('missing_provenance:strata_engine_build')
    return warnings


def _coverage(model, rows, selection, suite, identity, warnings):
    expected = {t["id"]: t for t in suite["tasks"]}
    raw = model["workload_plan_json"] if "workload_plan_json" in model.keys() else None
    planned = None
    if raw is not None:
        plan = json.loads(raw)
        if not isinstance(plan, dict) or type(plan.get("version")) is not int or plan["version"] != 1 or not isinstance(plan.get("tasks"), list):
            raise ValueError("invalid workload_plan_json")
        planned = {}
        plan_ids = set()
        for task in plan["tasks"]:
            if not isinstance(task, dict) or not isinstance(task.get("id"), str):
                raise ValueError("invalid planned task")
            task_id = task["id"]
            if (task_id in plan_ids or task.get("category") not in CATEGORIES + COLLECTIONS or _group(task.get("group")) is None
                    or type(task.get("runs")) is not int or task["runs"] < 1):
                raise ValueError("invalid or duplicate planned task metadata")
            plan_ids.add(task_id)
            for key in ("task_spec_sha", "task_identity_sha"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(task.get(key, ""))):
                    raise ValueError(f"invalid planned {key}")
            if task_id not in expected or _group(task.get("group")) != suite["group"]:
                continue
            if any(task.get(k) != expected[task_id][k] for k in ("category", "task_spec_sha")) or (
                    identity == "task_identity_sha" and task.get(identity) != expected[task_id][identity]):
                raise ValueError(f"planned task version/category mismatch: {task_id}")
            planned[task_id] = task["runs"]
    declared = selection.get("expected_attempts")
    if declared is not None and planned is not None and declared != planned:
        raise ValueError("declared expected_attempts disagrees with saved plan")
    counts = planned if planned is not None else declared
    if counts is None:
        warnings.add("missing_planned_coverage")
        counts = {task_id: 1 for task_id in expected}
    if not counts:
        warnings.add("no_planned_suite_tasks")
    recorded = defaultdict(set)
    for row in rows:
        task_id = row["task_id"]
        if task_id not in expected or _group(row["group_name"]) != suite["group"]:
            continue
        n = row["run_n"]
        if type(n) is not int or n < 1 or n in recorded[task_id]:
            raise ValueError(f"invalid or duplicate attempt: {selection['run_id']}/{task_id}/{n}")
        if planned is not None or declared is not None:
            if task_id not in counts or n > counts[task_id]:
                raise ValueError(f"attempt outside saved/declared plan: {task_id}/{n}")
        recorded[task_id].add(n)
    missing = [{"task_id": task_id, "repeats": sorted(set(range(1, n + 1)) - recorded[task_id])}
               for task_id, n in sorted(counts.items()) if set(range(1, n + 1)) - recorded[task_id]]
    if missing:
        warnings.add("missing_planned_attempts")
    return {"run_id": selection["run_id"], "profile": selection["profile"],
            "expected_attempts": counts, "recorded_attempts": {k: len(v) for k, v in sorted(recorded.items())},
            "missing_attempts": missing}


def _arm(conn, arm, suite, identity):
    expected = {t["id"]: t for t in suite["tasks"]}
    samples = {task_id: [] for task_id in expected}
    excluded, evidence, configs, warnings, coverage = [], [], {}, set(), []
    if identity == "task_hash":
        warnings.add("legacy_task_identity_reingest_and_use_request_v2")
    for selection in arm["selections"]:
        run_id, profile = selection["run_id"], selection["profile"]
        run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        model = conn.execute("SELECT * FROM run_models WHERE run_id=? AND profile=?", (run_id, profile)).fetchone()
        if run is None or model is None:
            raise ValueError(f"unknown run/profile {run_id}/{profile}")
        rows = conn.execute("SELECT * FROM task_runs WHERE run_id=? AND profile=? ORDER BY section",
                            (run_id, profile)).fetchall()
        coverage.append(_coverage(model, rows, selection, suite, identity, warnings))
        for missing in coverage[-1]["missing_attempts"]:
            excluded.append({"run_id": run_id, "profile": profile, "reason": "missing_attempts", **missing})
        if not coverage[-1]["expected_attempts"]:
            excluded.append({"run_id": run_id, "profile": profile, "reason": "no_planned_suite_tasks"})
        failure = conn.execute("SELECT status, reason FROM sections WHERE run_id=? AND key=?",
                               (run_id, profile + "__failure")).fetchone()
        if failure:
            excluded.append({"run_id": run_id, "profile": profile, "reason": "profile_failure", "detail": dict(failure)})
            warnings.add("filtered_attempts")
        if not run["valid_for_analysis"]:
            excluded.append({"run_id": run_id, "profile": profile, "reason": "invalid_run",
                             "detail": run["invalid_reason"], "attempts": len(rows)})
            warnings.add("filtered_attempts")
            evidence.append({"run": dict(run), "model": dict(model), "rows": [dict(t) for t in rows]})
            continue
        seen = set()
        for row in rows:
            task_id = row["task_id"]
            if task_id not in expected or _group(row["group_name"]) != suite["group"]:
                excluded.append({"run_id": run_id, "section": row["section"], "reason": "outside_suite"})
                continue
            task = expected[task_id]
            if identity not in row.keys() or row[identity] is None:
                raise ValueError("normalized identity unavailable; re-ingest saved runs before request v2")
            if any(row[k] != task[k] for k in ("task_spec_sha", identity, "category")):
                raise ValueError(f"task version/category mismatch: {run_id}/{task_id}")
            repeat = row["run_n"]
            if type(repeat) is not int or repeat < 1 or (task_id, repeat) in seen:
                raise ValueError(f"invalid or duplicate attempt: {run_id}/{task_id}/{repeat}")
            seen.add((task_id, repeat))
            config = _config(run, model, row)
            configs[_sha(config)] = config
            warnings.update(_provenance(config))
            if row["status"] not in OUTCOMES:
                excluded.append({"run_id": run_id, "section": row["section"], "reason": row["status"] or "unknown_status"})
                warnings.add("filtered_attempts")
                continue
            if row["passed"] != OUTCOMES[row["status"]] or row["failure_class"] not in _failure_classes(row["status"]):
                raise ValueError(f"inconsistent outcome: {run_id}/{row['section']}")
            block = selection.get("block")
            if block is None:
                warnings.add("missing_campaign_blocks")
            samples[task_id].append({"passed": row["passed"], "block": block,
                                     "run_id": run_id, "section": row["section"]})
        evidence.append({"run": dict(run), "model": dict(model), "rows": [dict(t) for t in rows]})
    if len(configs) > 1:
        raise ValueError(f"mixed configurations in cohort {arm['label']}")
    return {"label": arm["label"], "samples": samples, "excluded": excluded, "warnings": warnings,
            "config": next(iter(configs.values()), None), "evidence": evidence, "coverage": coverage}


def _summary(arm, tasks):
    details = []
    for task in tasks:
        samples = arm["samples"][task["id"]]
        n, passes = len(samples), sum(s["passed"] for s in samples)
        details.append({**task, "attempts": n, "passes": passes,
                        "success_rate": passes / n if n else None,
                        "blocks": len({s["block"] for s in samples if s["block"] is not None})})
    observed = [d for d in details if d["attempts"]]
    rate = sum(d["success_rate"] for d in observed) / len(observed) if observed else None
    complete = len(observed) == len(tasks) and not any(c["missing_attempts"] for c in arm["coverage"]) and not (
        {"no_planned_suite_tasks", "missing_planned_coverage"} & arm["warnings"])
    categories = {}
    for category in sorted({t["category"] for t in tasks}):
        members = [t for t in details if t["category"] == category]
        rates = [t["success_rate"] for t in members if t["attempts"]]
        categories[category] = {"expected_tasks": len(members), "observed_tasks": len(rates),
                                "observed_success_rate": sum(rates) / len(rates) if rates else None}
    return {"label": arm["label"], "complete": complete, "expected_tasks": len(tasks),
            "observed_tasks": len(observed), "missing_tasks": [t["id"] for t in details if not t["attempts"]],
            "success_rate": rate if complete else None, "observed_success_rate": rate,
            "percentage": 100 * rate if complete else None,
            "attempts": sum(t["attempts"] for t in details), "passes": sum(t["passes"] for t in details),
            "repeat_distribution": dict(sorted(Counter(t["attempts"] for t in details).items())),
            "tasks": details, "categories": categories, "excluded": arm["excluded"], "config": arm["config"],
            "selection_coverage": arm["coverage"]}


def campaign_bound(estimate, widths, confidence=0.95, limits=(0.0, 1.0)):
    """Two-sided weighted Hoeffding bound for independent bounded block sums.

    widths are sums of absolute observation weights within each campaign, not
    sample variances. Outcome-identical blocks therefore never give zero width.
    """
    if not math.isfinite(estimate) or not 0 < confidence < 1 or not widths:
        raise ValueError("need a finite estimate, confidence in (0,1), and block widths")
    if any(not math.isfinite(w) or w <= 0 for w in widths):
        raise ValueError("block widths must be finite and positive")
    radius = math.sqrt(math.log(2 / (1 - confidence)) * sum(w * w for w in widths) / 2)
    return {"low": max(limits[0], estimate - radius), "high": min(limits[1], estimate + radius),
            "half_width": radius, "blocks": len(widths), "confidence": confidence}


def _uncertainty(arms, summaries, tasks, inference, estimate, limits):
    reasons = set().union(*(a["warnings"] for a in arms))
    if not all(s["complete"] for s in summaries):
        reasons.add("incomplete_suite")
    if not (inference.get("independent_blocks") is True and inference.get("fixed_design") is True
            and inference.get("note")):
        reasons.add("independent_fixed_design_not_confirmed")
    if any(t["blocks"] < 2 for s in summaries for t in s["tasks"]):
        reasons.add("fewer_than_two_blocks_per_task")
    result = {"method": METHOD, "scope": "fixed_suite_repeatability", "confidence": inference.get("confidence", 0.95),
              "status": "insufficient_evidence" if reasons else "available", "reasons": sorted(reasons),
              "interval": None}
    if reasons:
        return result
    widths = defaultdict(float)
    for arm in arms:
        for task in tasks:
            samples = arm["samples"][task["id"]]
            for sample in samples:
                widths[sample["block"]] += 1 / (len(tasks) * len(samples))
    result["interval"] = campaign_bound(estimate, list(widths.values()), result["confidence"], limits)
    return result


def analyze(conn, request):
    """Return auditable JSON-ready scores. Caller must provide a Row connection."""
    validate(request)
    suite, inference = request["suite"], request.get("inference", {})
    tasks = sorted(suite["tasks"], key=lambda t: t["id"])
    names = [n for n in ("candidate", "baseline") if n in request]
    arms = {n: _arm(conn, request[n], suite, _identity_key(request)) for n in names}
    summaries = {n: _summary(arms[n], tasks) for n in names}
    for name in names:
        summaries[name]["uncertainty"] = _uncertainty(
            [arms[name]], [summaries[name]], tasks, inference, summaries[name]["success_rate"], (0, 1))
    comparison = None
    if "baseline" in arms:
        candidate, baseline = arms["candidate"], arms["baseline"]
        differences = {}
        if candidate["config"] is not None and baseline["config"] is not None:
            differences = {k: {"baseline": baseline["config"][k], "candidate": v}
                           for k, v in candidate["config"].items() if baseline["config"][k] != v}
        undeclared = set(differences) - set(request.get("allowed_changes", []))
        if undeclared:
            raise ValueError(f"undeclared cohort differences: {sorted(undeclared)}")
        complete = all(s["complete"] for s in summaries.values())
        delta = summaries["candidate"]["success_rate"] - summaries["baseline"]["success_rate"] if complete else None
        per_task = []
        for c, b in zip(summaries["candidate"]["tasks"], summaries["baseline"]["tasks"]):
            diff = c["success_rate"] - b["success_rate"] if c["attempts"] and b["attempts"] else None
            per_task.append({"id": c["id"], "delta_percentage_points": 100 * diff if diff is not None else None})
        bound = _uncertainty(list(arms.values()), list(summaries.values()), tasks, inference, delta, (-1, 1))
        interval = bound["interval"]
        direction = "insufficient_evidence"
        if interval:
            direction = "increase" if interval["low"] > 0 else "decrease" if interval["high"] < 0 else "unresolved"
        comparison = {"baseline": baseline["label"], "candidate": candidate["label"],
                      "delta_percentage_points": 100 * delta if delta is not None else None,
                      "interval_percentage_points": {k: 100 * interval[k] for k in ("low", "high")}
                      if interval else None,
                      "uncertainty": bound, "direction": direction, "declared_differences": differences,
                      "tasks": per_task}
    return {"method": METHOD, "request_sha256": _sha(request), "evidence_sha256": _sha(
                {n: arms[n]["evidence"] for n in names}), "suite": suite, "cohorts": summaries,
            "comparison": comparison, "inference": inference,
            "transfer": {"status": "unavailable", "reason": "authored_suite_not_population_sample",
                         "families": dict(sorted(Counter(t.get("family", "unclassified") for t in tasks).items()))}}


def score(db_path, request):
    """Read an existing warehouse in a read transaction; never create/migrate it."""
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN")
        return analyze(conn, request)
    finally:
        conn.close()
