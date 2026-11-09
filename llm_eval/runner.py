"""One campaign owner, durable section commits, and restoration-first resume."""
import contextlib
import json
from pathlib import Path
import re
import signal
import threading
import time
from .adapters import FakeAdapter, CapacityError, HostMemoryError, InfrastructureError
from .graders import recall, tool_call, evidence
from .haystack import build
from .images import data_uri, quadrants_png
from .plan import verify_plan
from .policy import RESERVE, classification, forecast, fits, context_label
from .environment import capture
from . import inhibit
from .state import Journal, atomic_json, machine_lock, safe_id
from .telemetry import DEFAULT_INTERVAL, GpuSampler, sampling


# The owner's floor for usable context (the compaction point): a mode below it cannot be asked a scaled depth task.
MIN_USABLE_CONTEXT = 64000
COMPACTION_MARGIN = 1024
WORKLOAD_GROUPS = ("resume", "implementation", "ops", "surgery", "lookup", "advisory", "review", "depth", "chat_sweep",
                   "mined", "game", "vision")
MODEL_SECTIONS = ("preflight", "anchor_load", "context_qualification", "fixture_validation",
                  "smoke", "long_context_recall", "structured_tools", "vision",
                  *(f"workload_{g}" for g in WORKLOAD_GROUPS), "operational", "teardown")


def _recorded(write, *args, **values):
    """Bookkeeping around a restore must never stop the restore itself (a full disk would leave
    production down with no machine call made); a failed write costs only the record."""
    try:
        write(*args, **values)
    except Exception:
        pass


@contextlib.contextmanager
def _uninterruptible():
    """Nothing may abort a restore or the bookkeeping after it: while this is active SIGINT, SIGTERM and SIGHUP are
    ignored (red-team RT2-002: a first signal during teardown used to raise inside ``restore`` and leave
    production stopped). A SIGKILL cannot be caught; ``status`` then says STOP and ``resume`` restores."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    wanted = [s for s in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)) if s is not None]
    previous = {s: signal.signal(s, signal.SIG_IGN) for s in wanted}
    try:
        yield
    finally:
        for s, old in previous.items():
            signal.signal(s, old)


def restore(journal, adapter):
    with _uninterruptible():
        return _restore(journal, adapter)


def _restore(journal, adapter):
    if not journal.state["restoration_required"]:
        return True
    journal.state["status"] = "restoring"
    _recorded(journal.save)
    for attempt in range(2):
        try:
            adapter.stop()
            verification = adapter.restore(journal.state["snapshot"])
            if not verification.get("service_matches"):
                raise InfrastructureError("service does not match snapshot")
            if journal.state["snapshot"]["was_active"] and not (
                    verification.get("health") and verification.get("four_token_probe")):
                raise InfrastructureError("production health/probe failed")
            journal.state["restoration_required"] = False
            journal.state["last_restore"] = verification
            _recorded(journal.event, "restoration_complete", attempt=attempt + 1, verification=verification)
            _recorded(journal.save)
            return True
        except Exception as exc:  # Anything escaping here would leave production stopped, unrecorded.
            _recorded(journal.event, "restoration_failed", attempt=attempt + 1, error=str(exc))
    journal.state["status"] = "restore_failed"
    _recorded(journal.save)
    return False


def retried(operation, journal, label):
    for attempt in range(2):
        try:
            return operation()
        except InfrastructureError as exc:
            journal.event("infrastructure_retry", operation=label, attempt=attempt + 1, error=str(exc))
            if attempt or isinstance(exc, HostMemoryError):
                raise


def qualify(model, adapter, journal, free_before):
    anchor = model["anchor"]
    above = [c for c in model["contexts"] if c >= anchor]
    below = sorted((c for c in model["contexts"] if c < anchor), reverse=True)
    attempts, selected, buffers = [], None, None
    queue = list(above)
    while queue:
        context = queue.pop(0)
        predicted = None
        if buffers is not None and buffers.get("kv_mib") is not None and buffers.get("fixed_mib") is not None:
            predicted = forecast(free_before, buffers["fixed_mib"], buffers["kv_mib"],
                                 buffers["context"], context)
            if predicted < 1024:
                attempts.append({"context": context, "status": "forecast_rejected", "predicted_free_mib": predicted})
                break
        adapter.stop()
        try:
            def load():
                try:
                    return adapter.load(model, context)
                except InfrastructureError:
                    adapter.stop()
                    raise
            result = retried(load, journal, f"load:{model['id']}:{context}")
            if result["free_mib"] < 1024:
                raise CapacityError("actual free memory below 1024 MiB")
            selected = context
            buffers = {**result, "context": context}
            attempts.append({"context": context, "status": "qualified", **result,
                             "predicted_free_mib": predicted if predicted is not None else result.get("predicted_free_mib")})
            if context < anchor:
                break
        except CapacityError as exc:
            attempts.append({"context": context, "status": "capacity_fail", "reason": str(exc)})
            adapter.stop()
            if selected is None and below:
                queue = [below.pop(0)]
                continue
            break
    if selected is None:
        raise CapacityError(json.dumps(attempts))
    # A rejected higher rung may have unloaded the selected rung.
    adapter.stop()
    active = retried(lambda: adapter.load(model, selected), journal, "selected_context_load")
    if active["free_mib"] < 1024:
        adapter.stop()
        raise CapacityError("selected context no longer has safe memory")
    # Under an auto-sized cache free VRAM is the reserve the config asked for, so the class reads the setting;
    # the engine's cache report (slots, MiB) is what a context or a knob actually costs.
    return {"selected_context": selected, "attempts": attempts, "memory": active.get("memory"),
            "memory_class": classification(active["free_mib"]), "free_mib": active["free_mib"],
            "context_class": context_label(selected),
            "capacity_fallback": selected < max(model["contexts"])}


def grade(fixture, response):
    # An answer cut off at the output cap would be a truncated turn in real use.
    if response.get("finish_reason") == "length":
        return {"pass": False, "reason": "output cap reached before the model finished"}
    if fixture["kind"] in ("recall", "vision"):
        return recall(response.get("content", ""), fixture["expected"])
    if fixture["kind"] == "smoke":
        return {"pass": re.fullmatch(r"\W*ok\W*", response.get("content", ""), re.IGNORECASE) is not None}
    if fixture["kind"] == "tools":
        return tool_call(response, fixture["tool_name"], fixture["arguments"])
    return evidence(response.get("content", ""), fixture["required_fact_ids"], fixture["known_sources"])


def materialize(fixture, context, adapter):
    """Concrete request for one context; haystack fixtures grow to the context's token envelope."""
    if fixture["kind"] == "vision":
        concrete = {k: v for k, v in fixture.items() if k != "image"}
        return {**concrete, "image_uri": data_uri(quadrants_png(fixture["image"]["quadrants"]))}
    if "haystack" not in fixture:
        return dict(fixture)
    target = context - fixture["max_output_tokens"] - RESERVE
    built = build(fixture["haystack"], target, lambda text: adapter.tokenize({"prompt": text}))
    concrete = {k: v for k, v in fixture.items() if k != "haystack"}
    return {**concrete, "prompt": built["prompt"], "expected": built["expected"],
            "prompt_tokens": built["prompt_tokens"]}


def run_trial(fixture, context, adapter, journal, key, deadline):
    trial_key = key(f"trial-{fixture['id']}-at-{context}")
    if trial_key in journal.state["sections"]:
        return journal.state["sections"][trial_key]
    if time.monotonic() > deadline:
        raise TimeoutError("model/campaign budget exceeded")
    concrete = materialize(fixture, context, adapter)
    tokens = concrete.get("prompt_tokens") or adapter.tokenize(concrete)
    validation = {"fixture": fixture["id"], "context": context, "prompt_tokens": tokens,
                  "fits": fits(tokens, concrete["max_output_tokens"], context)}
    if not validation["fits"]:
        return journal.section(trial_key, "capacity_fail", reason="prompt exceeds context envelope", **validation)
    request = {k: v for k, v in concrete.items() if k in ("prompt", "max_output_tokens", "seed", "tools", "image_uri")}
    atomic_json(journal.root / "requests" / f"{trial_key}.json", request)
    with sampling(journal, trial_key):
        response = retried(lambda: adapter.request(concrete), journal, trial_key)
    atomic_json(journal.root / "responses" / f"{trial_key}.json", response)
    verdict = grade(concrete, response)
    usage = {k: response[k] for k in ("output_tokens", "finish_reason", "usage", "timings", "request_seconds")
             if k in response}
    return journal.section(trial_key, "pass" if verdict["pass"] else
                           ("tool_fail" if fixture["kind"] == "tools" else "quality_fail"),
                           grading=verdict, max_output_tokens=concrete["max_output_tokens"],
                           **usage, **validation)


def _group(task):
    if task.get("diagnostic"):
        return "chat_sweep"
    return "depth" if task.get("depth") else task["category"]


SUMMARY_CHECK_KEYS = ("type", "pass", "missing", "missed", "problems", "forbidden", "outside_allow", "denied",
                      "returncode", "reason", "wrong", "min_peak_tokens", "peak_context_tokens",
                      "final_context_tokens", "compaction_point")


def _trial_summary(result, out):
    # A failed check keeps its whole payload (graders bound their own lists), so the section says why it failed;
    # a passing one keeps the short form.
    checks = [dict(c) if not c.get("pass") else {k: v for k, v in c.items() if k in SUMMARY_CHECK_KEYS}
              for c in result["grading"]["checks"]]
    return {"stop": result["stop"], "turns": len(result["turns"]), "clarifications": result["clarifications"],
            "tests_pasted": result["tests_pasted"], "checks": checks, "metrics": result["metrics"], "output": str(out),
            "depth": result.get("depth"), "graders_sha": result["grading"].get("graders_sha")}


def _with_context(config, ref, context):
    """A copy of the sandbox harness config with its context window set to the context actually loaded.

    opencode's config is ``provider.<p>.models.<entry>.limit.context``; pi's ``models.json`` is
    ``providers.<p>.models[]`` with ``contextWindow`` (the pi block's ``config``)."""
    import copy
    config = copy.deepcopy(config)
    provider, entry = ref.split("/", 1)
    if "providers" in config:
        for model in config["providers"][provider]["models"]:
            if model["id"] == entry:
                model["contextWindow"] = context
        return config
    config["provider"][provider]["models"][entry].setdefault("limit", {})["context"] = context
    return config


def run_workload(model, adapter, journal, key, ensure, invalidate, selected, deadline):
    """Every workload task-run, at the context real opencode sessions use for this profile.

    A task-run commits only when it finishes, so a resume reruns an unfinished one from a fresh
    snapshot (its partial output is kept aside) and never repeats a committed one."""
    from .workload.graders import unmeasured
    from .workload.session import run_task
    from .workload.webfix import CannedPages, PROXY_INSIDE, sandbox_env
    tasks = [f for f in model["fixtures"] if f["kind"] == "workload"]
    groups = {g: [t for t in tasks if _group(t) == g] for g in WORKLOAD_GROUPS}
    if not tasks:
        for group in WORKLOAD_GROUPS:
            journal.section(key(f"workload_{group}"), "skipped", reason="no workload tasks in this suite")
        return
    workload = model["workload"]
    target = workload["context"]
    # The target is a ceiling: run at the highest context the model qualified at (a memory fallback is a
    # lower-context result, not a lost one) and record the context actually used.
    context = min(target, selected)
    workload = {**workload, "config": _with_context(workload["config"], workload["ref"], context)}
    if workload.get("harness") == "pi" and workload.get("mode"):
        # pi rebuilds its models.json from the mode at install time, so the mode carries the loaded context too
        workload["mode"] = {**workload["mode"], "context": context}
    model = {**model, "workload": workload}
    output_limit = (workload.get("mode") or {}).get("output_limit") or 0
    # Leave a margin under the ceiling: opencode's own token estimate lags the server's templated count, and
    # a full-output request that lands a few hundred tokens over the ceiling is refused outright (Strata 400s
    # rather than truncating). Measured: a 148-message session compacted at 114,688 still asked at 114,981.
    compaction_point = context - min(output_limit, 32000) - COMPACTION_MARGIN
    reserve = ((workload.get("settings") or {}).get("compaction") or {}).get("reserveTokens")
    if workload.get("harness") == "pi" and reserve is not None:
        compaction_point = context - reserve  # pi compacts past contextWindow - reserveTokens
    sections = journal.state["sections"]
    home_files = {rel: Path(path).read_text() for rel, path in workload.get("home_files", {}).items()}
    for group in WORKLOAD_GROUPS:
        trials = []
        for task in groups[group]:
            for run in range(1, task.get("runs", 1) + 1):
                trial = key(f"trial-wl-{task['id']}-r{run}")
                if trial in sections:
                    trials.append(sections[trial])
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError("model/campaign budget exceeded")
                if task.get("fill") and compaction_point < MIN_USABLE_CONTEXT:
                    # A capacity result, not a quality one: this mode leaves too little room to ask the task at depth.
                    trials.append(journal.section(
                        trial, "capacity_fail", task=task["id"], category=task["category"], group=group, run=run,
                        context=context, target_context=target, compaction_point=compaction_point,
                        stop=f"usable context (compaction point) is {compaction_point}, below the {MIN_USABLE_CONTEXT} a scaled depth task needs",
                        turns=0, clarifications=0, tests_pasted=0, checks=[], metrics={}, output=None))
                    journal.event("wl_task", section=trial, status="capacity_fail", steps=None, peak_context_tokens=None)
                    continue
                ensure(context)
                out = journal.root / "workload" / trial
                if out.exists():
                    out.rename(out.with_name(f"{out.name}.aborted-{int(time.time())}"))
                served = None
                try:
                    pages = CannedPages(task["pages"], out / "ca") if task.get("pages") else contextlib.nullcontext()
                    with pages as served, sampling(journal, trial):
                        files, relays, env = dict(home_files), [], {}
                        if served:
                            files[".cache/llm-eval/canned-ca.pem"] = served.ca.read_text()
                            relays = [(PROXY_INSIDE, served.port)]
                            env = sandbox_env(str(Path.home() / ".cache/llm-eval/canned-ca.pem"))
                        result = run_task(task, out, workload["snapshots"][task["id"]],
                                          lambda home, manifest: adapter.workload_driver(home, manifest, model, relays, env),
                                          lambda home: adapter.workload_box(home, model), files,
                                          budget=deadline - time.monotonic(), compaction_point=compaction_point)
                    status = "pass" if result["pass"] else "quality_fail"
                    # A session that never reached its intended depth did not measure the model there: "unmeasured"
                    # is kept out of the quality denominator instead of counted as a wrong answer (owner, 2026-11-01).
                    if unmeasured(result.get("grading")):
                        status = "unmeasured"
                    # A response cut at the output limit is a truncation on every runtime (DECISIONS 2026-10-30); this
                    # was Strata-only, so a llama.cpp run that hit its cap scored as an ordinary wrong answer.
                    if (result.get('metrics') or {}).get('length_finishes'):
                        status, result['stop'] = 'quality_fail', 'output_cap'
                    if result.get("stop") in ("wall_cap", "tool_hang"):  # the model did not finish inside its cap,
                        # or a command it ran never returned: a model verdict, and the server is untouched
                        status = "timeout"
                    elif result.get("stop") == "disk_cap":  # a runaway writer: stopped before it fills the disk (B1)
                        status = "capacity_fail"
                    elif result.get("stop") in ("budget_cap", "stalled", "harness_stall", "harness_exit"):  # not a model verdict
                        status = "infra_fail"
                        if result["stop"] == "stalled":
                            invalidate()  # a wedged server must reload before anything else runs on it
                except (CapacityError, InfrastructureError):
                    raise
                except Exception as exc:  # harness or export failure: this run fails, the campaign goes on
                    journal.event("wl_task_error", section=trial, error=str(exc))
                    result = {"pass": False, "stop": f"harness failure: {exc}", "turns": [], "clarifications": 0,
                              "tests_pasted": 0, "grading": {"pass": False, "checks": []}, "metrics": {}}
                    status = "infra_fail"
                try:
                    adapter.check_alive()
                except CapacityError as exc:
                    status, result["stop"] = "capacity_fail", f"server died: {exc}"
                    invalidate()
                except InfrastructureError as exc:
                    status, result["stop"] = "infra_fail", f"server failed: {exc}"
                    invalidate()
                drift = getattr(adapter, "client_drift", None)
                if drift:
                    # An ops task may legitimately rewrite the client's own config, so drift is recorded and
                    # the trial is still graded: the evidence for it stays in raw/inference.
                    adapter.client_drift = None
                    journal.event("wl_client_drift", section=trial, errors=drift)
                summary = _trial_summary(result, out)
                if drift:
                    summary["contract_drift"] = drift
                if result.get("stop") in ("output_cap", "wall_cap", "tool_hang"):
                    summary["incomplete"] = result["stop"]  # cut off by a limit, not judged wrong
                capture = getattr(adapter, "inference_capture", None)
                if capture is not None:
                    reasoning = capture.reasoning_summary()
                    summary["reasoning_from_capture"] = reasoning
                    # The client's own count is the export's business and reads zero for a provider it does not
                    # model, so the server's stream wins when it saw thinking the client did not count.
                    if reasoning.get("blocks") and not (summary["metrics"].get("reasoning_tokens") or 0):
                        summary["metrics"]["reasoning_tokens_client"] = summary["metrics"].get("reasoning_tokens")
                        summary["metrics"]["reasoning_tokens"] = reasoning["tokens_estimate"]
                        summary["metrics"]["reasoning_source"] = "capture"
                if served:
                    summary["web"] = {"hits": len(served.hits), "misses": served.misses[:20]}
                trials.append(journal.section(trial, status, task=task["id"], category=task["category"], group=group,
                                              cohort_category=task.get('cohort_category'),
                                              run=run, context=context, target_context=target,
                                              compaction_point=compaction_point, **summary))
                journal.event("wl_task", section=trial, status=status, steps=summary["metrics"].get("steps"),
                              peak_context_tokens=summary["metrics"].get("peak_context_tokens"))
                if status == "infra_fail" and workload.get("stop_on_infra_fail", True):
                    # A harness failure is not a model verdict: end the block now, restore, and let the
                    # operator find the cause before more of the suite runs against a broken harness.
                    journal.event("wl_stop_on_infra_fail", section=trial, reason=result["stop"])
                    raise InfrastructureError(f"stopped at the first infrastructure failure, {trial}: {result['stop']}")
        passed = sum(t["status"] == "pass" for t in trials)
        per_task = {}
        for trial in trials:
            task_id = trial.get("task")
            entry = per_task.setdefault(task_id, {"passes": 0, "attempts": 0, "incomplete": 0})
            entry["attempts"] += 1
            entry["passes"] += trial["status"] == "pass"
            entry["incomplete"] += bool(trial.get("incomplete"))
        journal.section(key(f"workload_{group}"), "skipped" if not trials else (
            "pass" if passed == len(trials) else "quality_fail"), passed=passed, total=len(trials), context=context,
            incomplete=sum(t.get("incomplete") is not None for t in trials),
            contract_drift=sum(1 for t in trials if t.get("contract_drift")),
            per_task=per_task,
            **({} if trials else {"reason": "no tasks in this group"}))


def model_run(model, adapter, journal, deadline):
    prefix = model["id"]
    key = lambda section: f"{prefix}__{section}"
    if key("teardown") in journal.state["sections"]:
        return
    journal.section(key("preflight"), "pass", model_id=prefix)
    loaded = None
    interrupted = False
    def ensure(context):
        """Load ``context`` once; a lower rung is a new server, never a retry of an OOM rung."""
        nonlocal loaded
        if loaded == context:
            return
        adapter.stop()
        active = retried(lambda: adapter.load(model, context), journal, f"load:{prefix}:{context}")
        if active["free_mib"] < 1024:
            adapter.stop()
            raise CapacityError(f"context {context} left {active['free_mib']} MiB free")
        loaded = context
    def invalidate():
        nonlocal loaded
        adapter.stop()
        loaded = None
    try:
        previous = journal.state["sections"].get(key("context_qualification"))
        if previous and previous["status"] in ("pass", "capacity_fallback"):
            qualification = previous
        else:
            qualification = qualify(model, adapter, journal, journal.state["snapshot"]["free_mib"])
            loaded = qualification["selected_context"]
            journal.section(key("anchor_load"), "pass", attempts=qualification["attempts"])
            journal.section(key("context_qualification"),
                            "capacity_fallback" if qualification["capacity_fallback"] else "pass",
                            **qualification)
        selected = qualification["selected_context"]
        smoke_fixtures = [f for f in model["fixtures"] if f["kind"] == "smoke"]
        if any(key(f"trial-{f['id']}-at-{selected}") not in journal.state["sections"] for f in smoke_fixtures):
            ensure(selected)  # A resumed run has no scratch server yet.
        smoke = [run_trial(f, selected, adapter, journal, key, deadline) for f in smoke_fixtures]
        smoke_section = journal.section(key("smoke"), "skipped" if not smoke else
                                        ("pass" if all(s["status"] == "pass" for s in smoke) else "quality_fail"),
                                        context=selected, trials=len(smoke))
        if smoke_section["status"] == "quality_fail":
            # A model that cannot answer a one-line prompt will not pass anything deeper.
            for section in MODEL_SECTIONS[MODEL_SECTIONS.index("long_context_recall"):-1]:
                journal.section(key(section), "skipped", reason="smoke failed")
            return
        # Highest passing context: memory-safe is not the same as accurate at that depth.
        recall_fixtures = [f for f in model["fixtures"] if f["kind"] == "recall"]
        descent, verified, measured = [], None, 0
        for context in sorted((c for c in model["contexts"] if c <= selected), reverse=True):
            if not recall_fixtures:
                break
            pending = any(key(f"trial-{f['id']}-at-{context}") not in journal.state["sections"]
                          for f in recall_fixtures)
            try:
                if pending:
                    ensure(context)
                trials = [run_trial(f, context, adapter, journal, key, deadline) for f in recall_fixtures]
            except CapacityError as exc:
                # Ran out of memory at this depth, so the next rung down is a fresh server.
                descent.append({"context": context, "passed": 0, "total": len(recall_fixtures),
                                "capacity_fail": str(exc)})
                adapter.stop()
                loaded = None
                continue
            passed = sum(t["status"] == "pass" for t in trials)
            descent.append({"context": context, "passed": passed, "total": len(trials)})
            if passed == len(trials):
                verified, measured = context, min(t["prompt_tokens"] for t in trials)
                break
        recall_total = len(recall_fixtures)
        recall_status = "skipped" if not recall_total else ("pass" if verified and recall_total >= 3 else "quality_fail")
        journal.section(key("long_context_recall"), recall_status, required=3, verified_context=verified,
                        selected_context=selected, descent=descent,
                        context_class=context_label(verified, measured) if verified else None)
        tool_fixtures = [f for f in model["fixtures"] if f["kind"] == "tools"]
        tool_context = verified or (descent[-1]["context"] if descent else selected)
        if any(key(f"trial-{f['id']}-at-{tool_context}") not in journal.state["sections"] for f in tool_fixtures):
            ensure(tool_context)
        tools = [run_trial(f, tool_context, adapter, journal, key, deadline) for f in tool_fixtures]
        passed = sum(t["status"] == "pass" for t in tools)
        journal.section(key("structured_tools"), "skipped" if not tools else (
            "pass" if passed == len(tools) and len(tools) >= 10 else "quality_fail"),
            passed=passed, total=len(tools), required=10, context=tool_context)
        sees = any(a["role"] == "mmproj" for a in model.get("artifacts", [])) or "mmproj" in model["settings"]
        pictures = [f for f in model["fixtures"] if f["kind"] == "vision"] if sees else []
        if pictures and any(key(f"trial-{f['id']}-at-{tool_context}") not in journal.state["sections"] for f in pictures):
            ensure(tool_context)
        seen = [run_trial(f, tool_context, adapter, journal, key, deadline) for f in pictures]
        journal.section(key("vision"), "pass" if seen and all(s["status"] == "pass" for s in seen) else (
            "quality_fail" if seen else "skipped"), context=tool_context, trials=len(seen),
            **({} if sees else {"reason": "profile has no vision projector"}))
        run_workload(model, adapter, journal, key, ensure, invalidate, selected, deadline)
        journal.section(key("operational"), "skipped", reason="not part of the accepted workload plan")
    except (CapacityError, InfrastructureError, TimeoutError) as exc:
        status = "capacity_fail" if isinstance(exc, CapacityError) else (
            "timeout" if isinstance(exc, TimeoutError) else "infra_fail")
        journal.section(key("failure"), status, reason=str(exc))
        for section in MODEL_SECTIONS:
            if section != "teardown":
                journal.section(key(section), "skipped", reason=f"dependent section stopped: {status}")
    except BaseException:
        # An interrupt leaves the model unfinished: no teardown commit, so a resume picks it up.
        interrupted = True
        raise
    finally:
        adapter.stop()
        if not interrupted:
            journal.section(key("teardown"), "pass", owned_processes_remaining=0)


@contextlib.contextmanager
def terminate_as_interrupt():
    """SIGTERM and SIGHUP (a harness timeout, a closed terminal) take the same path as Ctrl-C, so the
    campaign restores production. The first one wins: later ones are ignored while the restore runs.
    Only the main thread can install handlers; elsewhere this does nothing."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    wanted = [s for s in (signal.SIGTERM, getattr(signal, "SIGHUP", None)) if s is not None]
    def handler(number, frame):
        for s in wanted:
            signal.signal(s, signal.SIG_IGN)
        raise KeyboardInterrupt(f"terminated by {signal.Signals(number).name}")
    previous = {s: signal.signal(s, handler) for s in wanted}
    try:
        yield
    finally:
        for s, old in previous.items():
            signal.signal(s, old)


def execute(plan, run_root, run_id, resume=False, adapter=None, lock_path=None):
    live = plan.get("backend") == "live"
    if live and adapter is None:
        raise RuntimeError("live execution disabled: real adapters and machine acceptance are pending")
    safe_id(run_id)
    root = Path(run_root).resolve()
    directory = root / run_id
    # One machine, one campaign: live runs share a per-user lock, fake runs lock their own root.
    lock = Path(lock_path) if lock_path else (
        Path.home() / ".cache" / "llm-eval" / "machine.lock" if live else root / ".fake-machine.lock")
    with machine_lock(lock), terminate_as_interrupt():
        if directory.exists() and not resume:
            raise ValueError("run exists; use resume")
        if resume and not (directory / "state.json").exists():
            raise ValueError("unknown run")
        if not resume:
            verify_plan(plan)
        journal = Journal(directory)
        original = journal.state["sections"].get("01_plan")
        if resume and original and original["plan_hash"] != plan["plan_hash"]:
            raise ValueError("plan does not match the run's original plan; resume with its plan.json")
        adapter = adapter or FakeAdapter(directory, plan.get("scenario"))
        if getattr(adapter, "run_id", run_id) != run_id:
            raise ValueError("adapter was built for a different run ID")
        if resume and journal.state.get("snapshot"):
            adapter.adopt(journal.state["snapshot"])
        if resume and not restore(journal, adapter):
            return journal.state
        # Restore stale state before rejecting changed inputs.
        verify_plan(plan)
        if journal.state["status"] == "complete":
            return journal.state
        atomic_json(directory / "plan.json", plan)
        journal.section("01_plan", "pass", plan_hash=plan["plan_hash"], backend=plan["backend"])
        snapshot = adapter.snapshot()
        if snapshot["unknown_gpu_clients"]:
            journal.state["status"] = "blocked"
            journal.section("02_lock_and_snapshot", "infra_fail", reason="unknown GPU clients", snapshot=snapshot)
            journal.save()
            return journal.state
        journal.state["snapshot"] = snapshot
        journal.section("02_lock_and_snapshot", "pass", snapshot=snapshot)
        with contextlib.ExitStack() as stack:
            lock = None
            if getattr(adapter, "holds_machine", False):
                # After the snapshot check and before the journal is armed: a refusal leaves no restore pending,
                # and the lock outlives the try/finally below, so it is released after the restore.
                try:
                    lock = stack.enter_context(inhibit.sleep_inhibit(f"llm-eval {run_id}", guard=_uninterruptible))
                except inhibit.InhibitUnavailable:
                    journal.state["status"] = "refused_no_inhibit"
                    _recorded(journal.save)  # a failed write must not replace the refusal
                    raise
                _recorded(journal.event, "sleep_inhibit_held", pid=lock.pid)
                lock.start_watcher(lambda name, **values: _recorded(journal.event, name, **values))
            journal.state.update(restoration_required=True, status="running")
            journal.save()  # Must reach disk before quiesce.
            journal.section("03_restore_arm", "pass")
            if not (directory / "environment.json").exists():  # a resume keeps what the original run ran on
                try:
                    atomic_json(directory / "environment.json", capture(adapter))
                except Exception:
                    pass  # derived data: never fails a run
            deadline = time.monotonic() + plan["limits"]["campaign_seconds"]
            sample = getattr(adapter, "sample_gpu", None)
            journal.sampler = GpuSampler(sample, directory / "telemetry" / "gpu.jsonl",
                                         plan["limits"].get("gpu_sample_seconds", DEFAULT_INTERVAL)).start() if sample else None
            host_sample = getattr(adapter, "sample_host", None)
            if host_sample:
                journal.extra_samplers = [GpuSampler(host_sample, directory / "telemetry" / "host.jsonl",
                                                     plan["limits"].get("gpu_sample_seconds", DEFAULT_INTERVAL)).start()]
            snapshot_server = getattr(adapter, "server_snapshot", None)
            if snapshot_server:
                def probe(tag, phase):
                    report = snapshot_server()
                    if report is not None:
                        atomic_json(directory / "raw" / "server" / f"{tag.replace('/', '_')}.{phase}.json", report)
                journal.server_probe = probe
            try:
                quiesced = adapter.quiesce()
                if isinstance(quiesced, dict) and "free_mib" in quiesced:
                    # Capacity is judged against the GPU with production stopped, not before.
                    journal.state["snapshot"].update(free_mib=quiesced["free_mib"], quiesced=True)
                    journal.save()
                journal.section("04_production_quiesce", "pass", **(quiesced or {}))
                for model in plan["models"]:
                    if lock is not None and lock.lost:
                        raise RuntimeError("sleep lock lost and could not be re-taken")
                    if time.monotonic() > deadline:
                        raise TimeoutError("campaign budget exceeded")
                    model_run(model, adapter, journal, min(deadline, time.monotonic() + plan["limits"]["model_seconds"]))
                journal.section("05_campaign_complete", "pass", models=len(plan["models"]))
                journal.state["status"] = "complete"
            except (Exception, KeyboardInterrupt) as exc:
                journal.state["status"] = "partial"
                journal.event("campaign_interrupted", error=str(exc))
            finally:
                with _uninterruptible():
                    if lock is not None and not lock.stop_watcher(inhibit.WATCHER_JOIN_SECONDS):
                        _recorded(journal.event, "sleep_inhibit_watcher_stuck")  # bounded: the restore never waits on it
                    outcome = journal.state["status"]
                    try:
                        restored = restore(journal, adapter)
                    finally:
                        for sampler in (journal.sampler, *journal.extra_samplers):
                            if sampler:
                                sampler.stop()  # after the restore, so recovery is on the record too
                        journal.server_probe = None
                    if restored:
                        journal.state["status"] = outcome
                        journal.section("06_restore_and_verify", "pass", verification=journal.state.get("last_restore"))
                    if lock is not None:
                        _recorded(journal.event, "sleep_inhibit_released", alive=lock.alive())
                    journal.save()
                    atomic_json(directory / "summary.json", journal.state)
            return journal.state
