#!/usr/bin/env python3
"""Run the unattended test plan (plans/2026-10-31-unattended-test-plan.md) phase by phase.

Nothing here asks a question mid-run. Each phase prepares a plan, runs it, checks the phase measured what
the plan says it measures, records the verdict and carries on. The rules the plan states are enforced here:

- a run that is not ``complete`` is never a pass, and infrastructure failures are told apart from results;
- every phase runs at 262,144 with no step-down, and a phase whose sections show another context fails;
- an infrastructure failure gets one fresh re-prepare, then the phase is skipped and the reason written down;
- the runner never SIGKILLs a campaign: an over-budget run gets SIGINT and the time to restore;
- a signal to this script is forwarded to the running campaign as SIGINT, and the stop is written down;
- production is checked before the plan and after every phase, and a failure stops the plan;
- a GPU held by another program (by usage, not by name) is waited on for 30 minutes, then the plan stops;
- the full plan refuses to start on a dirty harness or without a passing signal drill (``--only D``).

``--dry-run`` prints the phases and changes nothing; ``--check`` prepares every phase into a scratch folder
(which touches no service) to catch a refusal before the night, then deletes it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from llm_eval import environment, integration, phase_verdict, ramcheck  # noqa: E402

from llm_eval.paths import DATA as RUNS  # noqa: E402  (the data folder, outside git)
DATE = time.strftime("%Y-%m-%d")  # the day the plan starts, from the clock: names follow the run, not the plan
PLAN_DIR = RUNS / f"{DATE}-plan"  # set again by configure() for the target being measured
SUMMARY = PLAN_DIR / "SUMMARY.md"
DRILLS = RUNS / "drills.json"
STRATA = Path.home() / "Workspace" / "Strata"
CONFIGS = REPO / "configs/engine-0.1.54"
PYTHON = str(STRATA / ".venv/bin/python")
SERVER = str(STRATA / "serve/server.py")
# What is measured, set by configure(): the defaults are Strata at its 262,144 target, through pi (the owner's
# harness for every model). A llama.cpp profile runs the same phases at its own preset context, or at --context.
RUNTIME = "strata"
PROFILE = "strata-iq3s"
TARGET_CONTEXT = 262144
TARGET = CONFIGS / "iq3s-262144.json"  # the frozen Strata launch config for TARGET_CONTEXT
HARNESS = "pi"
RUNS_PER_TASK = None
OUTPUT = None  # None: the owner's pi maxTokens for the profile (configs/pi-production.json)
STRATA_ONLY = ("3",)  # the document tasks are sized by Strata's tokenizer
ROUTER = "llama-router.service"

GPU_GATE_MIB = 1024     # another program holding more than this blocks a phase (RUNBOOK: by usage, not by name)
GPU_WAIT_MINUTES = 30
RESTORE_WAIT_SECONDS = 900  # what a SIGINTed campaign gets to stop its engine and restore production
OVERRUN_SECONDS = 600       # past the plan's own campaign budget before the runner asks the campaign to stop
DRILL_DELAY_SECONDS = 60    # how long a drill trial runs before it is signalled
# The code a drill proves: a change to any of these needs a new drill before the next unattended run.
RESTORE_CODE = ("llm_eval/runner.py", "llm_eval/state.py", "llm_eval/inhibit.py", "llm_eval/live.py",
                "llm_eval/strata.py", "llm_eval/acceptance.py", "llm_eval/workload/inference.py",
                "llm_eval/workload/driver.py", "tools/run_test_plan.py")

IMPLEMENTATION = ["impl-llm-eval-eval-manifest", "impl-llm-eval-haystack", "impl-llm-eval-html-report",
                  "impl-llm-eval-output-budget", "impl-llm-eval-smoke-vision"]
DOCUMENTS = ["doc-retrieve-32k", "doc-reason-32k", "doc-unanswerable-32k",
             "doc-retrieve-100k", "doc-reason-100k", "doc-unanswerable-100k"]


def _drop(config, flag):
    """A copy of the config with one flag and its value removed."""
    args, out, skip = config.get("args", []), [], False
    for value in args:
        if skip:
            skip = False
            continue
        if value == flag:
            skip = True
            continue
        out.append(value)
    return {**config, "args": out}


def _env(config, name, value):
    return {**config, "env": {**config.get("env", {}), name: value}}


# One change per arm; the baseline arm is the frozen 262K config, run in the same paired campaign.
KNOBS = {
    "no-mtp": lambda config: _drop(config, "--mtp"),
    "tail-skip": lambda config: _env(config, "STRATA_ROUTE_TAIL_SKIP", "1"),
}
# No knob runs by default (owner, 2026-11-02): speculation is checked by the target model, so it changes speed, which
# the sweep measures, not answers (Strata's no-mtp campaign: 9/9 against 8/9 for 6 h). Name one with --knobs.
DEFAULT_KNOBS = ()
# llama.cpp knobs: preset overrides for arm B (None drops a setting), and whether a profile can take one.
# Each is one change in what it measures, though it may touch several flags (speculation needs all four).
LLAMA_KNOBS = {
    "no-spec": ({"spec-type": None, "model-draft": None, "spec-draft-n-max": None, "draft-p-min": None},
                lambda settings: bool(settings.get("spec-type"))),
    "kv-q8": ({"cache-type-k": "q8_0", "cache-type-v": "q8_0"},
              lambda settings: settings.get("cache-type-k") == "q4_0" and settings.get("cache-type-v") == "q4_0"),
}

# --variant FILE: extra llama.cpp knobs, one per ini section ({name: overrides}); "unset" drops a setting.
VARIANTS = {}
# --gate: the check suite's first tier, one paired seed instead of three (plans/2026-11-01-check-suite.md).
GATE = False


def load_variants(path):
    """{section: overrides} from an ini file whose sections each hold the preset lines arm B changes."""
    import configparser
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    if not parser.read(path):
        raise ValueError(f"cannot read the variant file {path}")
    found = {}
    for name in parser.sections():
        if not parser[name]:
            raise ValueError(f"variant {name} changes nothing")
        found[name] = {k: None if v.strip().lower() == "unset" else v.strip() for k, v in parser[name].items()}
    if not found:
        raise ValueError(f"no variants in {path}")
    return found


def llama_knobs():
    return {**LLAMA_KNOBS, **{name: (overrides, lambda settings: True) for name, overrides in VARIANTS.items()}}


def gate_verdict(lost):
    """The gate's rule: arm B (the candidate) against arm A (its baseline), one seed per task."""
    if not lost:
        return "gate passed: the candidate lost no task the baseline passed"
    if len(lost) == 1:
        return f"one task lost ({lost[0]}): run the three-seed campaign before deciding"
    return f"regression: {len(lost)} tasks lost ({', '.join(lost)})"


# Each phase: what it prepares and what its sections must show for the phase to have measured what it claims.
# ``graded``: every workload task-run in the plan has a verdict. ``recall``/``tools``: the classic sections pass
# at the target context with their required counts.
PHASES = {
    "0": {"name": "smoke", "suite": "workload:implementation", "tasks": ["impl-llm-eval-eval-manifest"],
          "expect": {"graded": True}, "single": True},
    "1": {"name": "diagnostic", "suite": "diagnostic", "expect": {"graded": True, "recall": True, "tools": True}},
    "2": {"name": "core", "suite": "core", "expect": {"graded": True}},
    "3": {"name": "documents", "suite": "strata", "tasks": DOCUMENTS, "expect": {"graded": True}},
    "4": {"name": "depth", "suite": "depth", "expect": {"graded": True}},
}
PLAN_PHASES = dict(PHASES)
# --suite accuracy (plans/2026-11-01-accuracy-suite.md): the per-model measurement, one phase per part so a part that
# fails is re-run alone (--only mined,game). Documents are sized by each profile's own tokenizer, so every runtime runs
# them; vision runs only for a profile that takes images and is skipped otherwise.
ACCURACY_PHASES = {
    "core": {"name": "accuracy-core", "suite": "core", "expect": {"graded": True}},
    "depth": {"name": "accuracy-depth", "suite": "depth", "expect": {"graded": True}},
    "documents": {"name": "accuracy-documents", "suite": "documents", "tasks": DOCUMENTS, "expect": {"graded": True}},
    "mined": {"name": "accuracy-mined", "suite": "mined", "expect": {"graded": True}},
    "game": {"name": "accuracy-game", "suite": "game", "expect": {"graded": True}},
    "vision": {"name": "accuracy-vision", "suite": "vision", "expect": {"graded": True}, "images": True},
}
SUITE = "plan"

DRY = False
STOPPING = []   # the signal that asked this script to stop, once one has
CHILD = [None]  # the running acceptance or paired process, so a signal can be forwarded to it


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    if DRY:
        return  # a dry run changes nothing, including the summary
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("a") as handle:
        handle.write(f"- {time.strftime('%Y-%m-%d %H:%M')} {message}\n")


def _on_signal(number, frame):
    """Forward the stop to the campaign as SIGINT (the path that restores production) and let it finish."""
    name = signal.Signals(number).name
    if not STOPPING:
        STOPPING.append(name)
        log(f"{name} received: asking the running campaign to stop and restore, then stopping the plan")
    child = CHILD[0]
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def _python(module, *args):
    return [sys.executable, "-m", module, *args]


def _call(argv, timeout):
    """Run a harness command to completion. Over its budget it gets SIGINT and the restore window, never
    SIGKILL: a killed campaign leaves production stopped. Returns (stdout, stderr, how it ended)."""
    process = subprocess.Popen(argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    CHILD[0] = process
    try:
        try:
            out, err = process.communicate(timeout=timeout)
            return out, err, "exited"
        except subprocess.TimeoutExpired:
            log(f"over its {timeout} s budget: SIGINT, waiting up to {RESTORE_WAIT_SECONDS} s for the restore")
            process.send_signal(signal.SIGINT)
            try:
                out, err = process.communicate(timeout=RESTORE_WAIT_SECONDS)
                return out, err, "interrupted"
            except subprocess.TimeoutExpired:
                return "", "", "stuck"  # still running: left alive on purpose, the plan stops and says so
    finally:
        CHILD[0] = None


def _json(text):
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, IndexError):
        return None


def fresh(directory):
    """A directory name nothing has used: a re-prepare never overwrites or moves an earlier attempt."""
    if not directory.exists():
        return directory
    attempt = 2
    while (candidate := directory.with_name(f"{directory.name}-attempt{attempt}")).exists():
        attempt += 1
    return candidate


def production_context(profile, settings, rungs, fallback):
    """The context the owner's pi gives this profile, if the preset can load it and it is a rung; else ``fallback``."""
    from llm_eval.workload import pi_config
    entry = pi_config.production_entry(profile)
    window = entry and entry["contextWindow"]
    loads = int(settings["ctx-size"])
    return window if window and window <= loads and (window in rungs or window == loads) else fallback


def pi_matches_preset(profile, settings):
    """Refuse a profile whose pi contextWindow (the repo's copy) differs from the context its preset loads: pi
    would compact at the wrong point, and the owner's own pi would send requests the server cannot hold."""
    from llm_eval.workload import pi_config
    entry = pi_config.production_entry(profile)
    if entry and entry.get("contextWindow") != int(settings["ctx-size"]):
        raise ValueError(f"{profile}: pi's contextWindow is {entry.get('contextWindow')} but the preset loads "
                         f"{int(settings['ctx-size'])}; run python3 tools/sync_pi_context.py --write, then commit")


def configure(runtime="strata", profile=None, context=None, harness="pi", runs=None, output=None, suite="plan"):
    """Point every phase at one target: runtime, profile, context, harness, runs per task, output limit; ``suite``
    picks the phases: the test plan's (0 to 6) or the accuracy suite's parts."""
    global RUNTIME, PROFILE, TARGET_CONTEXT, TARGET, HARNESS, RUNS_PER_TASK, OUTPUT, PLAN_DIR, SUMMARY, SUITE
    if suite not in ("plan", "accuracy"):
        raise ValueError(f"--suite must be plan or accuracy, not {suite!r}")
    RUNTIME, HARNESS, RUNS_PER_TASK, OUTPUT, SUITE = runtime, harness, runs, output, suite
    PHASES.clear()
    PHASES.update(ACCURACY_PHASES if suite == "accuracy" else PLAN_PHASES)
    if runtime == "strata":
        PROFILE = profile or "strata-iq3s"
        TARGET_CONTEXT = context or 262144
        TARGET = CONFIGS / f"iq3s-{TARGET_CONTEXT}.json"
        if not TARGET.exists():
            raise ValueError(f"no frozen Strata config for {TARGET_CONTEXT}: {TARGET}")
    else:
        if not profile:
            raise ValueError("a llama.cpp target needs --profile")
        from llm_eval.preflight import eval_manifest
        PROFILE = profile
        model = eval_manifest(profiles=[profile])["models"][0]
        pi_matches_preset(profile, model["settings"])
        TARGET_CONTEXT = context or production_context(profile, model["settings"], model["contexts"], model["anchor"])
        TARGET = None
    PLAN_DIR = RUNS / f"{label(suite)}"
    SUMMARY = PLAN_DIR / "SUMMARY.md"


def label(name):
    """``<date>-<profile>-<context>-<name>``: one name per measurement, so runs of different targets never mix."""
    return f"{DATE}-{PROFILE}-{TARGET_CONTEXT}-{name}"


def sees_images():
    """Whether this target takes image input (its pi entry, or a llama.cpp preset with a projector)."""
    from llm_eval.acceptance import _sees_images
    model = {}
    if RUNTIME != "strata":
        from llm_eval.preflight import eval_manifest
        model = eval_manifest(profiles=[PROFILE])["models"][0]
    return _sees_images(RUNTIME, PROFILE, model)


def runtime_of(profile):
    return "strata" if profile.startswith("strata") else "llama.cpp"


def narrow(picks):
    """Phases narrowed to named tasks, from ``--task PHASE:TASK`` (a re-run of a few tasks under the plan's own
    supervision). Returns the phases named, in plan order; raises ValueError on a malformed or unknown phase."""
    chosen = {}
    for pick in picks:
        number, _, task = pick.partition(":")
        if number not in PHASES or not task:
            raise ValueError(f"--task wants PHASE:TASK with a phase in {', '.join(PHASES)}: {pick!r}")
        chosen.setdefault(number, []).append(task)
    for number, tasks in chosen.items():
        spec = PHASES[number]
        known = set(spec.get("tasks") or ())
        if not known:
            from llm_eval.workload import prepare as workload
            known = {t["id"] for t in workload.select(spec["suite"])[1]}
        unknown = sorted(set(tasks) - known)
        if unknown:
            raise ValueError(f"--task names no task of phase {number} ({spec['suite']}): {', '.join(unknown)}")
        # A narrowed phase is its own measurement: its folder and summary lines say so, never the whole phase's.
        PHASES[number] = {**spec, "tasks": tasks, "name": f"{spec['name']}-subset"}
    return [n for n in PHASES if n in chosen]


def prepare(directory, suite, config=None, tasks=(), runs=None, harness=None, require_free_gpu=True,
            profile=None, overrides=None):
    """``profile``: another target than this plan's (a model comparison's arm B), at this plan's context and
    output. ``overrides``: llama.cpp preset changes for this plan only (a knob's arm B)."""
    runtime = runtime_of(profile) if profile else RUNTIME
    args = ["prepare", "--runtime", runtime, "--profile", profile or PROFILE, "--suite", suite, "--dir", str(directory),
            "--context", str(TARGET_CONTEXT), "--no-step-down", "--harness", harness or HARNESS]
    if OUTPUT:
        args += ["--output", str(OUTPUT)]
    if runtime == "strata":
        frozen = config or (TARGET if RUNTIME == "strata" else CONFIGS / f"iq3s-{TARGET_CONTEXT}.json")
        if not Path(frozen).exists():
            return None, f"no frozen Strata config for {TARGET_CONTEXT}: {frozen}"
        args += ["--strata-config", str(frozen), "--strata-python", PYTHON, "--strata-server", SERVER]
    for key, value in (overrides or {}).items():
        args += ["--preset-unset", key] if value is None else ["--preset-set", f"{key}={value}"]
    for task in tasks:
        args += ["--task", task]
    if runs or RUNS_PER_TASK:
        args += ["--runs", str(runs or RUNS_PER_TASK)]
    out, err, _ = _call(_python("llm_eval.acceptance", *args), timeout=5400)
    body = _json(out)
    if not body or "approval_token" not in body:
        return None, (body or {}).get("error") or out[-800:] or err[-800:]
    if body.get("unknown_gpu_clients") and require_free_gpu:
        return None, f"another program holds the GPU: {body['unknown_gpu_clients']}"
    return directory, body["approval_token"]


def _plan_budget(directory):
    try:
        plan = json.loads((directory / "plan.json").read_text())
        return int(plan["limits"]["campaign_seconds"]) + OVERRUN_SECONDS
    except (OSError, ValueError, KeyError):
        return 8 * 3600


def run(directory, token):
    out, err, ended = _call(_python("llm_eval.acceptance", "run", "--dir", str(directory), "--approve", token),
                            timeout=_plan_budget(directory))
    if ended == "stuck":
        return {"status": "stuck", "failure": "the campaign did not stop within the restore window"}
    return _json(out) or {"status": "unknown", "failure": out[-800:] or err[-800:]}


def run_dir(directory):
    """The newest acceptance run inside a prepared directory."""
    runs = sorted(Path(directory).glob("runs/acceptance-*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def sections_of(directory):
    folder = run_dir(directory)
    try:
        return json.loads((folder / "state.json").read_text()).get("sections", {}) if folder else {}
    except (OSError, ValueError):
        return {}


def planned_trials(directory):
    """Workload task-runs the prepared plan promised, by trial section suffix."""
    try:
        plan = json.loads((Path(directory) / "plan.json").read_text())
    except (OSError, ValueError):
        return set()
    return {f"trial-wl-{f['id']}-r{n}" for m in plan["models"] for f in m["fixtures"] if f.get("kind") == "workload"
            for n in range(1, f.get("runs", 1) + 1)}


def verdict(summary, sections, expect=None, planned=()):
    """(status, one line) for a finished run at this plan's context: ``llm_eval/phase_verdict.py``."""
    return phase_verdict.verdict(summary, sections, expect, planned, TARGET_CONTEXT)


describe = phase_verdict.describe


def gpu_holders():
    """[(pid, MiB)] of compute processes that are neither production's router (or its children) nor this
    plan's own campaign; None when nvidia-smi cannot say, which is treated as busy."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode:
        return None
    router = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", ROUTER],
                            capture_output=True, text=True).stdout.strip()
    ours = {int(router)} if router.isdigit() and int(router) else set()
    holders = []
    for line in out.stdout.strip().splitlines():
        try:
            pid, used = (int(float(cell)) for cell in line.split(","))
        except ValueError:
            continue
        if not _descends(pid, ours):
            holders.append((pid, used))
    return holders


def _descends(pid, roots):
    while pid > 1:
        if pid in roots:
            return True
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def wait_for_gpu():
    """True when the GPU is free of other programs, polling every 5 minutes for up to 30."""
    for attempt in range(GPU_WAIT_MINUTES // 5 + 1):
        holders = gpu_holders()
        if holders is not None and sum(m for _, m in holders) <= GPU_GATE_MIB:
            return True
        if attempt == GPU_WAIT_MINUTES // 5 or STOPPING:
            break
        log(f"GPU busy: {holders if holders is not None else 'nvidia-smi gave no answer'}; waiting 5 minutes")
        _sleep(300)
    return False


def _sleep(seconds):
    """A wait a stop signal cuts short."""
    end = time.monotonic() + seconds
    while time.monotonic() < end and not STOPPING:
        time.sleep(min(5, end - time.monotonic()))


def production_ok():
    active = subprocess.run(["systemctl", "--user", "is-active", ROUTER], capture_output=True, text=True)
    health = subprocess.run(["curl", "-s", "--max-time", "5", "http://127.0.0.1:8080/health"],
                            capture_output=True, text=True)
    return active.stdout.strip() == "active" and '"ok"' in health.stdout


def restore_code_sha():
    digest = hashlib.sha256()
    for relative in RESTORE_CODE:
        digest.update(relative.encode() + b"\0" + (REPO / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _drills():
    try:
        return json.loads(DRILLS.read_text())
    except (OSError, ValueError):
        return []


def drill_passed():
    """Both signals have been drilled, and passed, on the restore code as it is now."""
    current = restore_code_sha()
    passed = {d["signal"] for d in _drills() if d.get("passed") and d.get("restore_code_sha") == current}
    return {"SIGINT", "SIGTERM"} <= passed


def drill(dry, number):
    """Signal a live trial and require the campaign to restore production by itself. Meant to be run
    attended (``--only D``) before the first unattended night on new restore code."""
    sig = signal.Signals(number)
    name = f"drill-{sig.name.lower()}"
    log(f"{name}: preparing")
    if dry:
        return "dry"
    if not wait_for_gpu():
        return "gpu"
    directory, token = prepare(fresh(RUNS / label(name)), "workload:implementation",
                               tasks=["impl-llm-eval-eval-manifest"])
    if directory is None:
        log(f"{name}: prepare refused: {token}")
        return "refused"
    process = subprocess.Popen(_python("llm_eval.acceptance", "run", "--dir", str(directory), "--approve", token),
                               cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    CHILD[0] = process
    record = {"date": time.strftime("%Y-%m-%d %H:%M"), "signal": sig.name, "dir": str(directory),
              "restore_code_sha": restore_code_sha(), "harness": environment.harness_version()}
    try:
        started = time.monotonic()
        while process.poll() is None and time.monotonic() - started < 900:
            sections = sections_of(directory)
            if any(k.endswith("__vision") for k in sections):  # the classic sections are done: a trial is running
                break
            time.sleep(5)
        _sleep(DRILL_DELAY_SECONDS)
        sent = time.monotonic()
        if process.poll() is None:
            process.send_signal(number)
        else:  # nothing to interrupt: the trial ended (or failed) before the signal, so the drill proves nothing
            failure = next((v.get("reason") for k, v in sections_of(directory).items()
                            if k.endswith("__failure") and v.get("reason")), None)
            record.update(passed=False, reason="the campaign ended before the signal; " +
                          (failure if failure else "see its sections"))
        try:
            process.communicate(timeout=RESTORE_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            record.update(passed=False, reason=f"still running {RESTORE_WAIT_SECONDS} s after {sig.name}")
    finally:
        CHILD[0] = None
    folder = run_dir(directory)
    events = []
    if folder and (folder / "events.jsonl").exists():
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines() if line.strip()]
    state = json.loads((folder / "state.json").read_text()) if folder and (folder / "state.json").exists() else {}
    wall = time.time() - (time.monotonic() - sent)
    interrupted = next((e for e in events if e.get("event") == "campaign_interrupted"), None)
    restored = next((e for e in events if e.get("section") == "06_restore_and_verify" and e.get("status") == "pass"), None)
    record.update(seconds_to_interrupt=round(interrupted["time"] - wall, 1) if interrupted else None,
                  seconds_to_restore=round(restored["time"] - wall, 1) if restored else None,
                  restoration_required=state.get("restoration_required"), production_ok=production_ok())
    if "passed" not in record:
        record["passed"] = bool(interrupted and restored and state.get("restoration_required") is False
                                and record["production_ok"])
    DRILLS.write_text(json.dumps(_drills() + [record], indent=1) + "\n")
    log(f"{name}: {'passed' if record['passed'] else 'FAILED'}{' (' + record['reason'] + ')' if record.get('reason') else ''} (interrupt after {record['seconds_to_interrupt']} s, "
        f"restored after {record['seconds_to_restore']} s, production ok: {record['production_ok']}); {directory}")
    return "complete" if record["passed"] else "drill_fail"


def phase(dry, number):
    """Prepare, run, assess, check production; one fresh re-prepare after an infrastructure failure."""
    spec = PHASES[number]
    name = spec["name"]
    if spec.get("images") and not sees_images():
        log(f"{name}: {PROFILE} takes no image input; skipped")
        return "skipped"
    log(f"{name}: preparing ({spec['suite']}{', ' + str(len(spec.get('tasks', []))) + ' tasks' if spec.get('tasks') else ''})")
    if dry:
        return "dry"
    status = None
    for attempt in (1, 2):
        if not wait_for_gpu():
            return "gpu"
        directory, token = prepare(fresh(RUNS / label(name)), spec["suite"], tasks=spec.get("tasks", ()),
                                   runs=1 if spec.get("single") else None)
        if directory is None:
            log(f"{name}: prepare refused: {token}")
            return "refused"
        summary = run(directory, token)
        if summary.get("status") == "stuck":
            log(f"{name}: STOP, the campaign is still running after SIGINT; {directory}")
            return "stuck"
        sections = sections_of(directory)
        status, detail = verdict(summary, sections, spec.get("expect"), planned_trials(directory))
        log(f"{name}: {status} ({detail}); {describe(sections)}; {directory}")
        if not production_ok():
            log(f"{name}: PRODUCTION NOT RESTORED, stopping the plan")
            return "production"
        refusal = next((v["reason"] for k, v in sections.items() if k.endswith("__failure")
                        and str(v.get("reason", "")).startswith("refused:")), None)
        if refusal:  # the host lacked RAM: a re-prepare minutes later meets the same machine
            log(f"{name}: {refusal}")
            return "refused"
        if status != "infra_fail" or STOPPING:
            break
        if attempt == 1:
            log(f"{name}: infrastructure failure, one fresh re-prepare (raw/ of {run_dir(directory)} has the cause)")
    return status


def paired_changes(plan_a, plan_b):
    """The arm differences a paired freeze must be told about, refused unless they are settings only and leave
    the request (sampling, effort) alone."""
    from llm_eval import paired
    leaves = [paired._leaves(paired._contract(plan)) for plan in (plan_a, plan_b)]
    changes = sorted(k for k in leaves[0].keys() | leaves[1].keys() if leaves[0].get(k) != leaves[1].get(k))
    wrong = [k for k in changes if not k.startswith("settings.") or k.startswith("settings.request")]
    if wrong:
        raise ValueError(f"the arms differ outside the launch settings: {wrong[:6]}")
    return changes


def knob_arms(name, folder=None):
    """(arm A prepare kwargs, arm B prepare kwargs) for a knob on this target, or None if it does not apply.
    A Strata variant config is written to ``folder`` (default the plan's ``knobs/``, outside the harness repo)."""
    if RUNTIME == "strata":
        variant = (folder or PLAN_DIR / "knobs") / f"{name}.json"
        if not DRY:
            variant.parent.mkdir(parents=True, exist_ok=True)
            variant.write_text(json.dumps(KNOBS[name](json.loads(TARGET.read_text())), indent=1) + "\n")
        return {"config": TARGET}, {"config": variant}
    from llm_eval.preflight import eval_manifest
    overrides, applies = llama_knobs()[name]
    if not applies(eval_manifest(profiles=[PROFILE])["models"][0]["settings"]):
        return None
    return {}, {"overrides": overrides}


def default_knobs():
    """Phase 5's arms when --knobs is not given: a --variant file's sections, otherwise none (DEFAULT_KNOBS)."""
    return list(VARIANTS) if VARIANTS else list(DEFAULT_KNOBS)


def campaign(dry, title, arm_kwargs, tasks, kind="settings", suite="check"):
    """A paired campaign (llm_eval.paired): arm A against arm B on ``tasks`` x three seeds (one with --gate),
    alternating A/B order, a cold engine per attempt. ``kind='settings'`` admits only launch-setting differences;
    'models' admits any difference but the tasks, context and harness, and records them all."""
    seeds = 1 if GATE else 3
    log(f"{title}: preparing the paired campaign ({len(tasks)} tasks x {seeds} seed{'s' * (seeds > 1)} x 2 arms)")
    if dry:
        return "dry"
    if not wait_for_gpu():
        return "gpu"
    base = fresh(RUNS / label(title))
    arms = {}
    for arm, kwargs in zip("ab", arm_kwargs):
        directory, token = prepare(base / f"arm-{arm}", suite, tasks=tasks, **kwargs)
        if directory is None:
            log(f"{title}: arm {arm} prepare refused: {token}")
            return "refused"
        arms[arm] = json.loads((directory / "plan.json").read_text())
    allowed = []
    if kind == "settings":
        try:
            allowed = paired_changes(arms["a"], arms["b"])
        except ValueError as exc:
            log(f"{title}: refused: {exc}")
            return "refused"
    folder = base / "campaign"
    freeze = ["freeze", "--arm-a", str(base / "arm-a/plan.json"), "--arm-b", str(base / "arm-b/plan.json"),
              "--dir", str(folder), "--kind", kind, "--seeds", str(seeds)]
    for key in allowed:
        freeze += ["--allow", key]
    folder.mkdir(parents=True, exist_ok=True)
    out, err, _ = _call(_python("llm_eval.paired", *freeze), timeout=600)
    frozen = _json(out)
    if not frozen or "approval_token" not in frozen:
        log(f"{title}: freeze refused: {(out or err)[-400:]}")
        return "refused"
    changes = sorted(frozen.get("changes") or allowed)
    longest = max(int(plan["limits"]["campaign_seconds"]) for plan in arms.values())
    budget = frozen["planned"] * (longest + 120) + OVERRUN_SECONDS
    out, err, ended = _call(_python("llm_eval.paired", "run", "--dir", str(folder), "--approve",
                                    frozen["approval_token"]), timeout=budget)
    report = _json(out) or {}
    if ended == "stuck":
        log(f"{title}: STOP, the campaign is still running after SIGINT; {folder}")
        return "stuck"
    a, b = (report.get("arms") or {}).get("A", {}), (report.get("arms") or {}).get("B", {})
    log(f"{title}: {report.get('recorded')}/{report.get('planned')} attempts; A {a.get('passed')}/"
        f"{a.get('recorded')} pass, {round(a.get('all_attempt_seconds') or 0)} s; B {b.get('passed')}/"
        f"{b.get('recorded')} pass, {round(b.get('all_attempt_seconds') or 0)} s; time on tasks both solved "
        f"{round(a.get('common_success_seconds') or 0)} s against {round(b.get('common_success_seconds') or 0)} s; "
        f"{len(changes)} changes {changes[:8]}; {folder}")
    lost, gained = report.get("lost_by_b") or [], report.get("gained_by_b") or []
    if report.get("complete"):
        verdict = gate_verdict(lost) if GATE else f"B lost {len(lost)} and gained {len(gained)} paired attempts"
        log(f"{title}: {verdict}")
    if not production_ok():
        log(f"{title}: PRODUCTION NOT RESTORED, stopping the plan")
        return "production"
    if not report.get("complete"):
        return "infra_fail" if report.get("halted") else "incomplete"
    return "complete"


def knob(dry, name):
    """The frozen target (A) against one change (B), on the check suite."""
    from llm_eval.ladder import check_tasks
    arms = knob_arms(name)
    if arms is None:
        log(f"knob-{name}: does not apply to {PROFILE}; skipped")
        return "skipped"
    return campaign(dry, f"knob-{name}", arms, check_tasks())


def compare(dry, other):
    """This target (A) against another profile (B) at this plan's context, on the check suite: every category
    plus two long-context tasks, so the comparison covers the owner's kinds of work rather than one."""
    from llm_eval.ladder import check_tasks
    return campaign(dry, f"vs-{other}", ({}, {"profile": other}), check_tasks(), kind="models")


def ram_blocker():
    """Strata reads its whole GGUF into RAM: refuse at the start, before anything is stopped, when the host has
    less than the floor available. Other runtimes fit in VRAM and are only logged. Every plan logs ram.jsonl."""
    row = ramcheck.snapshot(PLAN_DIR / "ram.jsonl", "plan_start", runtime=RUNTIME)
    why, available = ramcheck.floor_blocker(RUNTIME)
    log(f"RAM available at the start: {available:,.0f} MiB" if available is not None else "RAM available: unreadable")
    return why


def preflight(full_plan):
    """Refusals before anything runs: a dirty harness, an integration tier not passed on this source, production
    not healthy, and (for the full plan) no passing signal drill on the current restore code."""
    dirty = environment.dirty_paths()
    if dirty:
        return f"the harness has uncommitted changes, so its SHA would not reproduce the run: {dirty[:5]}"
    if not production_ok():
        return f"{ROUTER} is not active and healthy before the plan; nothing was started"
    why = ram_blocker()
    if why:
        return why
    if integration.blocker():
        log(f"integration tier: {integration.blocker()}; running it (no GPU, real opencode and pi in bwrap)")
        out, err, _ = _call(_python("llm_eval.acceptance", "integration"), timeout=1800)
        if integration.blocker():
            return f"the integration tier did not pass: {(out or err)[-400:]}"
    if full_plan and not drill_passed():
        return ("no passing SIGINT and SIGTERM drill on the current restore code: run `--only D` attended first "
                f"(records in {DRILLS})")
    return None


def check_phases(knobs=(), others=()):
    """Prepare every phase into a scratch folder and report refusals; no service is touched."""
    from llm_eval.ladder import check_tasks
    scratch = Path(tempfile.mkdtemp(prefix="llm-eval-plan-check-"))
    problems = []
    try:
        for number, spec in PHASES.items():
            if RUNTIME != "strata" and number in STRATA_ONLY:
                continue
            if spec.get("images") and not sees_images():
                print(f"phase {number} {spec['name']}: skipped, {PROFILE} takes no image input")
                continue
            directory, token = prepare(scratch / spec["name"], spec["suite"], tasks=spec.get("tasks", ()),
                                       runs=1 if spec.get("single") else None, require_free_gpu=False)
            print(f"phase {number} {spec['name']}: {'ok, token ' + token if directory else 'REFUSED: ' + str(token)}")
            problems += [] if directory else [number]
        for name in knobs:
            arms = knob_arms(name, scratch)
            if arms is None:
                print(f"knob {name}: does not apply to {PROFILE}")
                continue
            plans = []
            for arm, kwargs in zip("ab", arms):
                directory, token = prepare(scratch / f"knob-{name}-{arm}", "check", tasks=check_tasks(),
                                           require_free_gpu=False, **kwargs)
                plans.append(json.loads((directory / "plan.json").read_text()) if directory else None)
                if not directory:
                    print(f"knob {name} arm {arm}: REFUSED: {token}")
            try:
                print(f"knob {name}: arms differ in {paired_changes(*plans)}" if all(plans) else f"knob {name}: REFUSED")
            except ValueError as exc:
                print(f"knob {name}: REFUSED: {exc}")
            problems += [] if all(plans) else [name]
        for other in others:
            from llm_eval import paired
            plans = []
            for arm, kwargs in zip("ab", ({}, {"profile": other})):
                directory, token = prepare(scratch / f"vs-{other}-{arm}", "check", tasks=check_tasks(),
                                           require_free_gpu=False, **kwargs)
                plans.append(json.loads((directory / "plan.json").read_text()) if directory else None)
                if not directory:
                    print(f"vs {other} arm {arm}: REFUSED: {token}")
            if all(plans):
                try:
                    frozen = paired.freeze({"A": plans[0], "B": plans[1]}, scratch / f"vs-{other}", kind="models")
                    print(f"vs {other}: {len(frozen['attempts'])} attempts; arms differ in {sorted(frozen['changes'])[:8]}")
                except ValueError as exc:
                    print(f"vs {other}: REFUSED: {exc}")
                    plans = [None]
            problems += [] if all(plans) else [other]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return 1 if problems else 0


def write_summary(results):
    lines = ["", "## Summary", "", "| phase | result |", "|---|---|"]
    lines += [f"| {number} | {result} |" for number, result in results.items()]
    passed = [n for n, r in results.items() if r == "complete"]
    lines += ["", f"{len(passed)} of {len(results)} phases complete; the per-phase lines above carry the numbers. "
              "`quality_fail` is a measured result; `assert_fail`, `infra_fail`, `refused`, `gpu`, `stuck` and "
              "`production` mean the phase did not measure what the plan says."]
    with SUMMARY.open("a") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true", help="prepare every phase into a scratch folder and exit")
    parser.add_argument("--suite", default="plan", choices=("plan", "accuracy"),
                        help="plan: the test plan's phases 0-6; accuracy: the accuracy suite, one phase per part "
                             f"({', '.join(ACCURACY_PHASES)}), no knobs or comparisons")
    parser.add_argument("--only", default="",
                        help="comma-separated phases: D (signal drill), 0-4, 5 (knobs), 6 (model comparisons); with "
                             f"--suite accuracy, D or the parts ({', '.join(ACCURACY_PHASES)})")
    parser.add_argument("--knobs", help=f"comma-separated knob arms for phase 5 (Strata: {', '.join(KNOBS)}, "
                                        f"llama.cpp: {', '.join(LLAMA_KNOBS)}; default none, so phase 5 is "
                                        "skipped unless named or a --variant file is given)")
    parser.add_argument("--variant", help="llama.cpp: an ini file, one section per variant holding the preset lines "
                                          "arm B changes ('unset' drops one); each section is a phase 5 knob")
    parser.add_argument("--gate", action="store_true",
                        help="phases 5 and 6 run the check suite's gate: one paired seed, not three")
    parser.add_argument("--compare", default="",
                        help="phase 6: comma-separated profiles to compare against this target, paired, at its "
                             "context (strata-iq3s or a llama.cpp preset name)")
    parser.add_argument("--output", type=int,
                        help="output limit for every phase (default: the owner's pi maxTokens for the profile); "
                             "pin it for a like-for-like table")
    parser.add_argument("--runtime", default="strata", choices=("strata", "llama.cpp"))
    parser.add_argument("--profile", help="the profile (default strata-iq3s; a llama.cpp preset name otherwise)")
    parser.add_argument("--context", type=int,
                        help="the one context every phase runs at (default: 262,144 for Strata, the preset's own)")
    parser.add_argument("--harness", default="pi", choices=("pi", "opencode"))
    parser.add_argument("--runs", type=int, help="runs per task in phases 1, 2 and 4 (default: each task's own, 1)")
    parser.add_argument("--task", action="append", default=[], metavar="PHASE:TASK",
                        help="run only these tasks in that phase (repeatable); with no --only, run only the phases "
                             "named here")
    args = parser.parse_args()
    global DRY, GATE
    DRY, GATE = args.dry_run, args.gate
    try:
        configure(args.runtime, args.profile, args.context, args.harness, args.runs, args.output, args.suite)
        if SUITE == "accuracy" and (args.knobs is not None or args.compare or args.variant or args.gate):
            raise ValueError("--suite accuracy runs no knobs or comparisons: drop --knobs, --compare, --variant, --gate")
        if args.variant:
            if RUNTIME == "strata":
                raise ValueError("--variant changes a llama.cpp preset; Strata knobs change its config")
            VARIANTS.update(load_variants(args.variant))
    except ValueError as exc:
        parser.error(str(exc))
    knobs = ([] if SUITE == "accuracy" else [k.strip() for k in args.knobs.split(",") if k.strip()]
             if args.knobs is not None else default_knobs())
    others = [c.strip() for c in args.compare.split(",") if c.strip()]
    known = KNOBS if RUNTIME == "strata" else llama_knobs()
    if [k for k in knobs if k not in known]:
        parser.error(f"unknown {RUNTIME} knob: {[k for k in knobs if k not in known]} (known: {', '.join(known)})")
    if PROFILE in others:
        parser.error("--compare names the target itself")
    if args.check:
        return check_phases(knobs, others)
    only = [p.strip() for p in args.only.split(",") if p.strip()]
    try:
        named = narrow(args.task)
    except ValueError as exc:
        parser.error(str(exc))
    only = only or named
    extra = ("5", "6") if SUITE == "plan" else ()
    unknown = [p for p in only if p not in ("D", *PHASES, *extra)]
    if unknown:
        parser.error(f"unknown phase or knob: {unknown}")
    if RUNTIME != "strata" and set(only) & set(STRATA_ONLY):
        parser.error(f"phases {', '.join(STRATA_ONLY)} are Strata-only")
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    if not DRY:
        PLAN_DIR.mkdir(parents=True, exist_ok=True)
        if not SUMMARY.exists():
            title = "Accuracy suite" if SUITE == "accuracy" else "Test plan"
            SUMMARY.write_text(f"# {title}, {RUNTIME} {PROFILE} at {TARGET_CONTEXT}, harness {HARNESS}, {DATE}\n\n")
    log(f"plan starting: suite {SUITE}, {RUNTIME} {PROFILE} at {TARGET_CONTEXT}, harness {HARNESS}, runs {RUNS_PER_TASK or 'per task'}, "
        f"output {OUTPUT or 'production maxTokens'} (dry run: {DRY}, phases: {only or 'all'}, knobs: {knobs or 'none'}, "
        f"compare: {others or 'none'})")
    if not DRY:
        why = preflight(full_plan=not only or any(p != "D" for p in only))
        if why:
            log(f"refused: {why}")
            return 2
    steps = {"D": lambda: next((r for r in (drill(DRY, s) for s in (signal.SIGINT, signal.SIGTERM))
                                if r not in ("complete", "dry")), "dry" if DRY else "complete")}
    steps.update({number: (lambda n=number: phase(DRY, n)) for number in PHASES})
    if SUITE == "plan":
        steps["5"] = lambda: next((r for r in (knob(DRY, k) for k in knobs) if r not in ("complete", "dry", "skipped")),
                                  "dry" if DRY else "complete" if knobs else "skipped")
        steps["6"] = lambda: next((r for r in (compare(DRY, o) for o in others) if r not in ("complete", "dry")),
                                  "dry" if DRY else "complete" if others else "skipped")
    order = only or [n for n in steps if n != "D" and (RUNTIME == "strata" or n not in STRATA_ONLY)
                     and (n != "5" or knobs) and (n != "6" or others)]
    results = {}
    for number in order:
        if STOPPING:
            log(f"stopping before phase {number}: {STOPPING[0]}")
            break
        log(f"phase {number} starting")
        if not DRY:
            ramcheck.snapshot(PLAN_DIR / "ram.jsonl", f"phase_{number}_start")
        results[number] = steps[number]()
        if not DRY:
            ramcheck.snapshot(PLAN_DIR / "ram.jsonl", f"phase_{number}_end", result=str(results[number]))
        log(f"phase {number} finished: {results[number]}")
        if results[number] in ("production", "stuck", "gpu", "drill_fail"):
            log(f"stopping the plan: {results[number]}")
            break
    if not DRY:
        write_summary(results)
        if not production_ok():
            log(f"plan end: {ROUTER} is NOT active and healthy; restore it with the run's `resume` command")
    log(f"plan finished: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
