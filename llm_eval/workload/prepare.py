"""Prepare-time work for workload suites: choose the tasks, build their snapshots, and assemble the
model's ``workload`` block that the plan pins. Runs git and ``opencode --version``; never touches
systemd, the GPU or production config (production's opencode config is only read).
"""
import json
from pathlib import Path
import re
import shutil
import subprocess

from ..plan import read_manifest
from . import opencode_config, replicas, snapshot
from .tasks import CATEGORIES, COLLECTIONS, validate

TASK_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "workload"
CLASSIC_DIR = Path(__file__).resolve().parents[2] / "fixtures"
HOME_FIXTURE_DIR = TASK_DIR / "home"
PRODUCTION_CONFIG = Path.home() / ".config/opencode/opencode.jsonc"
OPENCODE_BINARY = Path.home() / ".opencode/bin/opencode"
# The npm install itself, not ~/.local/bin/pi: the driver mounts this binary's grandparent as the box's
# ~/.npm-global, and the ~/.local/bin symlink points back at that path, which inside the box is a loop (pi was
# "not found" on its first real run, 2026-10-30).
PI_BINARY = Path.home() / ".npm-global/bin/pi"
# The owner's working agreement as pi loads it, frozen for the bench comparison (2026-11-06 review decision 1A):
# the copy in this repo is the dotfiles version of 69cbbcde, which every run since 11-04 used. The live file
# (~/Workspace/dotfiles/agents/AGENTS.md) changed that day and is not read; each pi plan pins this copy.
PI_AGENTS_MD = Path(__file__).resolve().parents[2] / "configs" / "pi-agents-md-bench.md"
NODE_HOME = Path.home() / ".local/share/fnm"
SUITES = ("classic", "diagnostic", "depth", "core", "check", "full", "strata", "strata-depth", "accuracy", "documents",
          *COLLECTIONS, *(f"workload:{c}" for c in CATEGORIES))
# The benchmark's mode (plan 2026-10-04): the bare entry at this target context, stepping down to the highest
# context the model qualifies at, with this output limit; both are recorded on every run.
TARGET_CONTEXT = 131072
SCREEN_OUTPUT = 16384
# Mean model minutes per task-run by tool-call band, from 131 local opencode sessions (plan
# evidence), plus a minute of setup and grading per run.
BAND_MINUTES = {"lookup": 1.2, "advisory": 1.2, "resume": 5.2, "review": 5.2, "ops": 9.4,
                "surgery": 13.6, "implementation": 13.6, "mined": 13.6, "game": 13.6, "vision": 2.0}
SETUP_MINUTES = 1.0
CLASSIC_MINUTES = 40
# Real sessions have a git identity; a synthetic one keeps commits working without real details.
GITCONFIG = "[user]\n\tname = Workspace Owner\n\temail = owner@example.invalid\n[init]\n\tdefaultBranch = main\n"


def load_tasks(task_dir=TASK_DIR, collections=()):
    """The task directory's own tasks, plus those of each named collection subfolder (mined, game, vision)."""
    paths = sorted(Path(task_dir).glob("*.json"))
    for name in collections:
        paths += sorted((Path(task_dir) / name).glob("*.json"))
    return [validate(read_manifest(p)) | {"_path": str(p)} for p in paths]


def select(suite, task_dir=TASK_DIR, task_ids=None):
    """(classic fixture paths, workload tasks) for a suite name; ``task_ids`` narrows it further."""
    if suite not in SUITES:
        raise ValueError(f"suite must be one of {', '.join(SUITES)}")
    if suite == 'strata':
        raise ValueError('freeze the separate Strata roster through acceptance.prepare')
    classic = sorted(str(p) for p in CLASSIC_DIR.glob("*.json"))
    extra = COLLECTIONS if suite in ("check", "accuracy", *COLLECTIONS) else ()
    tasks = load_tasks(task_dir, extra) if suite != "classic" else []
    if suite.startswith("workload:"):
        tasks = [t for t in tasks if t["category"] == suite.split(":", 1)[1] and not t.get("depth") and not t.get("diagnostic")]
    elif suite == "core":
        tasks = [t for t in tasks if not t.get("depth") and not t.get("diagnostic")]
    elif suite == "depth":
        tasks = [t for t in tasks if t.get("depth")]
    elif suite == "check":  # the check suite (suites/check.json) draws on core, depth and collection tasks
        tasks = [t for t in tasks if not t.get("diagnostic")]
    elif suite == "accuracy":  # the accuracy suite: core, depth and every collection, no classic trials
        tasks = [t for t in tasks if not t.get("diagnostic")]
    elif suite in COLLECTIONS:
        tasks = [t for t in tasks if t["category"] == suite]
    elif suite == "diagnostic":
        tasks = [t for t in tasks if t.get("diagnostic")]
    # Workload suites carry no smoke trial: the first task proves the harness, and a one-word reply gate
    # would reject a useful model before any task ran (red-team EF-005).
    fixtures = classic if suite in ("classic", "diagnostic", "full") else []
    if task_ids:
        known = {t["id"] for t in tasks}
        missing = sorted(set(task_ids) - known)
        if missing:
            raise ValueError(f"task {', '.join(missing)} is not in suite {suite} (it has: {', '.join(sorted(known)) or 'none'})")
        order = {t: i for i, t in enumerate(task_ids)}
        tasks = sorted((t for t in tasks if t["id"] in order), key=lambda t: order[t["id"]])  # the caller's order
    if suite != "classic" and not tasks:
        raise ValueError(f"no workload tasks for suite {suite} in {task_dir}")
    from .session import env_mounts
    verified = set()
    for task in tasks:
        env_mounts(task, "/repo", verified=verified)  # a missing or changed environment refuses at prepare
    return fixtures, tasks


def with_runs(tasks, runs, directory):
    """Task copies whose ``runs`` is overridden, written under the run directory so the plan hashes
    exactly what will run. Returns the tasks with their ``_path`` pointing at the copies."""
    if runs is None:
        return tasks
    if not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    out = Path(directory) / "tasks"
    out.mkdir(parents=True, exist_ok=True)
    changed = []
    for task in tasks:
        clean = {k: v for k, v in task.items() if k != "_path"} | {"runs": runs}
        path = out / f"{task['id']}.json"
        path.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")
        changed.append(clean | {"_path": str(path)})
    return changed


def estimate_minutes(suite, tasks):
    minutes = lambda t: 6.0 if t.get("diagnostic") else BAND_MINUTES[t["category"]] + (4.0 if t.get("depth") else 0)
    work = sum((minutes(t) + SETUP_MINUTES) * t.get("runs", 1) for t in tasks)
    return round(work + (CLASSIC_MINUTES if suite in ("classic", "diagnostic", "full") else 5))


# Export, capture and grading happen outside the turn deadline: about 20 s a task in the 2026-10-30 runs.
TASK_OVERHEAD_SECONDS = 180
PREFILL_FLOOR_TPS = 250  # a cold full-context prefill slower than this is a failure, not a slow model
GPU_SAMPLE_SECONDS = 1   # free memory bottoms out inside a prefill; 5 s sampling missed it


def limits(minutes, tasks=(), context=None):
    """Backstop caps, enforced mid-task (run_task gets what is left of the model budget). Each task is already
    bounded by its own wall cap, so the model cap is just their sum plus per-task overhead and a load: it can
    never cut a task that is still inside its own cap (red-team EF-007). The estimate is only a fallback for
    a suite with no tasks. Hangs are caught sooner by the stall watchdog (driver.STALL_SECONDS).
    ``request_seconds`` is a socket idle timeout, and a streamed request is silent through its whole prefill, so
    it covers a cold prefill of the full context (after a compaction or a cache miss) at PREFILL_FLOOR_TPS."""
    work = sum((t["limits"]["wall_seconds"] + TASK_OVERHEAD_SECONDS) * t.get("runs", 1) for t in tasks)
    model = work + 300 if work else max(900, int(minutes * 60 * 1.5))
    request = max(600, int(context / PREFILL_FLOOR_TPS) + 120) if context else 600
    return {"campaign_seconds": model + 900, "model_seconds": model, "load_seconds": 300,
            "request_seconds": request, "gpu_sample_seconds": GPU_SAMPLE_SECONDS}


def opencode_version(binary=OPENCODE_BINARY):
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60)
    version = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if out.returncode or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"cannot read the opencode version from {binary}")
    return version


def bundled_home_files(directory=HOME_FIXTURE_DIR):
    """Public, deterministic fake-home guidance used by every workload run."""
    root = Path(directory)
    return {path.relative_to(root).as_posix(): path.read_text()
            for path in sorted(root.rglob("*")) if path.is_file()}


def mode_of(entry_name, limit, explicit_entry, context_overridden):
    """The recorded mode: which opencode entry, loaded context, output limit and the compaction
    point those imply (opencode compacts at ``context - min(limit.output, 32,000)``, per STATE)."""
    context, output = limit.get("context"), limit.get("output")
    return {"entry": entry_name, "context": context, "output_limit": output,
            "compaction_point": context - min(output or 0, 32000) if context else None,
            "entry_explicit": explicit_entry, "context_overridden": context_overridden}


def pi_version(binary=PI_BINARY):
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=120)
    version = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    if out.returncode or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"cannot read the pi version from {binary}")
    return version


def pi_workload_block(directory, model, tasks, context=None, output=None, entry=None, thinking=None,
                      compaction_at=None, binary=PI_BINARY, stop_on_infra_fail=True, agents_md=PI_AGENTS_MD):
    """The workload block for the pi harness: one loopback provider, one model, one compaction point.

    With ``agents_md`` (the default) the run is the owner's pi: their model entry for this profile and their
    AGENTS.md. ``agents_md=None`` keeps the synthetic entry (reasoning flag, output 16,384)."""
    from . import pi_config
    directory = Path(directory)
    if not shutil.which("bwrap"):
        raise ValueError("bwrap is required for workload suites")
    # The run's pinned request (llama.cpp: model level; Strata: settings.request), so pi's own samplingParams send
    # what the capture expects instead of leaving the capture to fill them in.
    request = model.get("request") or (model.get("settings") or {}).get("request") or {}
    context = TARGET_CONTEXT if context is None else context
    explicit_output = output is not None
    production = pi_config.production_entry(model["id"]) if agents_md else None
    if thinking is None:
        # the owner's defaultThinkingLevel; pi clamps it per model (max reaches each model's highest level)
        thinking = pi_config.production_thinking_level() if production else "high"
    if output is None:
        # The owner's maxTokens for this model (32,768 for Strata), not the opencode screen's 16,384
        output = (production or {}).get("maxTokens", SCREEN_OUTPUT)
    models, settings, ref = pi_config.build(model["id"], context, output, request, thinking, compaction_at,
                                            production)
    if context not in model["contexts"]:
        raise ValueError(f"pi's context ({context}) is not a ladder rung")
    snapshots = {}
    for task in tasks:
        out = directory / "snapshots" / task["id"]
        clean = {k: v for k, v in task.items() if k != "_path"}
        (replicas.build if "replica" in task["source"] else snapshot.build)(clean, out)
        snapshots[task["id"]] = str(out)
    extra = {(pi_config.AGENT_DIR / "AGENTS.md").as_posix(): Path(agents_md).read_text()} if agents_md else {}
    home_files, block = _home_files(directory, extra)
    mode = mode_of(ref.split("/", 1)[1], {"context": context, "output": output}, entry is not None, context is not None)
    # pi compacts once the context passes contextWindow - reserveTokens, not at opencode's point
    mode["compaction_point"] = context - settings["compaction"]["reserveTokens"]
    block.update({"harness": "pi", "context": context, "config": models, "settings": settings, "ref": ref,
                  "snapshots": snapshots, "thinking": thinking, "request": request,
                  "thinking_effective": pi_config.clamp_level(production, thinking) if production else thinking,
                  "production_entry": production, "agents_md": str(agents_md) if agents_md else None,
                  "mode": mode | {"output_overridden": explicit_output},
                  "stop_on_infra_fail": stop_on_infra_fail, "binary": str(binary),
                  # The npm prefix the box mounts as ~/.npm-global. Plan compilation resolves "binary" to the
                  # bundle's cli.js (for its hash), whose grandparent is not that prefix.
                  "binary_home": str(Path(binary).parents[1]),
                  "binary_version": pi_version(binary), "home_files": home_files,
                  "read_only": [], "path": []})
    if NODE_HOME.exists():
        block["read_only"] = [[str(NODE_HOME), str(NODE_HOME)]]
        block["path"] = [str(NODE_HOME / "aliases/default/bin")]
    return block


def _home_files(directory, extra=None):
    home = Path(directory) / "home-files"
    home.mkdir(parents=True, exist_ok=True)
    contents = bundled_home_files() | {".gitconfig": GITCONFIG} | (extra or {})
    home_files = {}
    for relative, text in contents.items():
        pinned = home / relative
        pinned.parent.mkdir(parents=True, exist_ok=True)
        pinned.write_text(text)
        home_files[relative] = str(pinned)
    hits = snapshot.scan(home)
    if hits:
        raise ValueError("home files refused, likely secrets: " + ", ".join(f"{p} ({r})" for p, r in hits[:10]))
    return home_files, {"home_files": home_files}


def workload_block(directory, model, tasks, production_config=PRODUCTION_CONFIG, binary=OPENCODE_BINARY,
                   entry=None, context=None, stop_on_infra_fail=True, output=None, harness="opencode",
                   thinking=None, compaction_at=None):
    directory = Path(directory)
    if harness == "pi":
        return pi_workload_block(directory, model, tasks, context=context, output=output, entry=entry,
                                 thinking=thinking, compaction_at=compaction_at,
                                 stop_on_infra_fail=stop_on_infra_fail)
    if harness != "opencode":
        raise ValueError(f"unknown harness {harness!r}")
    if not shutil.which("bwrap"):
        raise ValueError("bwrap is required for workload suites")
    if model.get("runtime") == "strata":
        # A synthetic local-only entry avoids edits/registration in production config.
        production = {"share": "disabled", "provider": {"local-llama": {
            "npm": "@ai-sdk/openai-compatible", "name": "Local benchmark",
            "options": {"baseURL": opencode_config.SANDBOX_BASE_URL}, "models": {
                model["id"]: {"name": model["id"], "supportsToolCalls": True,
                              "limit": {"context": context or TARGET_CONTEXT, "output": output or SCREEN_OUTPUT}}}}}}
    else:
        production = opencode_config.parse_jsonc(Path(production_config).read_text())
    config, ref, limit = opencode_config.build(production, model["id"], model.get("covers", []), entry,
                                               TARGET_CONTEXT if context is None else context,
                                               SCREEN_OUTPUT if output is None else output)
    overridden = context is not None
    context = limit.get("context")
    if context not in model["contexts"]:
        raise ValueError(f"opencode's context for {ref} ({context}) is not a ladder rung")
    snapshots = {}
    for task in tasks:
        out = directory / "snapshots" / task["id"]
        clean = {k: v for k, v in task.items() if k != "_path"}
        (replicas.build if "replica" in task["source"] else snapshot.build)(clean, out)
        snapshots[task["id"]] = str(out)
    home = directory / "home-files"
    home.mkdir(parents=True, exist_ok=True)
    contents = bundled_home_files() | {".gitconfig": GITCONFIG}
    home_files = {}
    for relative, text in contents.items():
        pinned = home / relative
        pinned.parent.mkdir(parents=True, exist_ok=True)
        pinned.write_text(text)
        home_files[relative] = str(pinned)
    hits = snapshot.scan(home)
    if hits:
        raise ValueError("home files refused, likely secrets: " + ", ".join(f"{p} ({r})" for p, r in hits[:10]))
    block = {"harness": "opencode", "context": context, "config": config, "ref": ref, "snapshots": snapshots,
             "mode": mode_of(ref.split("/", 1)[1], limit, entry is not None, overridden) | {"output_overridden": output is not None},
             "stop_on_infra_fail": stop_on_infra_fail,
             "opencode_binary": str(binary), "opencode_version": opencode_version(binary),
             "production_config": None if model.get("runtime") == "strata" else str(production_config), "home_files": home_files,
             "read_only": [], "path": []}
    if NODE_HOME.exists():  # job-tracker tests need node; bound read-only at the same path
        block["read_only"] = [[str(NODE_HOME), str(NODE_HOME)]]
        block["path"] = [str(NODE_HOME / "aliases/default/bin")]
    return block
