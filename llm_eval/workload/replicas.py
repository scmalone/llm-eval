"""Synthetic machine replicas for ops tasks, packed in the same snapshot format as repo tasks.

A builder returns the files of one broken machine, all synthetic and derived from the seed:
a git repo (usually a config repo), files for the fake home (placeholder models, unit files,
the simulated machine state, wrapper scripts for the shims), where opencode starts, and extra
PATH entries. ``build`` packs that into a bundle plus ``home.tar`` and a manifest, so a run
materializes a replica exactly like a pinned commit. Builders live outside the sandbox: they
describe the fault, so the box never sees them.
"""
import json
from pathlib import Path
import random
import shutil
import subprocess
import tarfile
import tempfile

from ..plan import digest
from ..state import atomic_json
from .snapshot import BRANCH, git, scan, tree_files

SHIMS = ("systemctl", "journalctl", "nvidia-smi", "ps")
MACHINE = ".local/state/llm-eval-machine/machine.json"
SYNTHETIC_HOME = Path("~")


def _wrappers():
    return {f".local/bin/{name}": f'#!/bin/sh\nexec python3 -P -m llm_eval.workload.machine {name} "$@"\n'
            for name in SHIMS}


def router_bad_model_path(seed):
    """The router unit fails to start because a preset points at a GGUF that is not on disk."""
    rng = random.Random(seed)
    home = SYNTHETIC_HOME
    family = rng.choice(["muse-glimmer-30b", "qwen3.6-35b-a3b", "occamy-1.0"])
    actual = f"{family}-Q4_K_XL.gguf"
    wrong = rng.choice([f"{family}-Q4_K_M.gguf", f"{family}-q4_k_xl.gguf", f"{family}-Q4_K_XL-v2.gguf"])
    other = "qwen3.8-27b-Q4_K_XL.gguf"
    preset = (f"[{family}-q4-agent]\nmodel = {home}/models/{wrong}\nctx-size = 65536\njinja = true\n\n"
              f"[qwen3.8-27b-q4-agent]\nmodel = {home}/models/{other}\nctx-size = 65536\njinja = true\n")
    unit = "llama-router.service"
    machine = {"units": {unit: {
        "description": "llama.cpp router (local models for opencode)", "active": False, "enabled": True,
        "result": "exit-code",
        "requires": [{"ini_paths_exist": [f"{home}/.config/llama-cpp/models-preset.ini", "model"]}],
        "journal": {
            "failed": [f"llama-server[2211]: loading preset {home}/.config/llama-cpp/models-preset.ini",
                       f"llama-server[2211]: gguf_init_from_file: failed to open GGUF file '{home}/models/{wrong}' (No such file or directory)",
                       f"llama-server[2211]: error: failed to load model '{family}-q4-agent'",
                       f"systemd[1]: {unit}: Main process exited, code=exited, status=1/FAILURE",
                       f"systemd[1]: {unit}: Failed with result 'exit-code'."],
            "active": [f"llama-server[2290]: loaded {len(preset.split('[')) - 1} presets",
                       "llama-server[2290]: server is listening on http://127.0.0.1:8080"]}}},
        "home": str(home), "gpu": {"processes": []}, "processes": [], "log": []}
    files = {f"models/{actual}": "GGUF placeholder for the benchmark\n", f"models/{other}": "GGUF placeholder\n",
             f".config/systemd/user/{unit}": "[Service]\nExecStart=/usr/bin/llama-server --models-preset %h/.config/llama-cpp/models-preset.ini --port 8080\n",
             MACHINE: json.dumps(machine, indent=2), **_wrappers()}
    return {"place": ".config/llama-cpp", "repo_files": {"models-preset.ini": preset}, "home_files": files,
            "cwd": "", "path": [".local/bin"], "facts": {"wrong": wrong, "actual": actual, "unit": unit}}


def _router_unit(home, extra_journal_failed, requires, active=False, port=8080):
    unit = "llama-router.service"
    return unit, {
        "description": "llama.cpp router (local models for opencode)", "active": active, "enabled": True,
        "result": "success" if active else "exit-code", "requires": requires,
        "journal": {"failed": extra_journal_failed + [f"systemd[1]: {unit}: Main process exited, code=exited, status=1/FAILURE",
                                                      f"systemd[1]: {unit}: Failed with result 'exit-code'."],
                    "active": [f"llama-server[2290]: server is listening on http://127.0.0.1:{port}"]}}


def _machine(home, units, gpu=(), processes=()):
    return json.dumps({"units": units, "home": str(home), "gpu": {"processes": list(gpu)},
                       "processes": list(processes), "log": []}, indent=2)


PRESET = "[{name}]\nmodel = {home}/models/{file}\nctx-size = 65536\njinja = true\n{extra}\n"
UNIT_FILE = "[Service]\nExecStart=%h/.local/bin/llama-server --models-preset %h/.config/llama-cpp/models-preset.ini --port 8080\n"


def opencode_wrong_port(seed):
    """opencode's machine env points at a port the router does not listen on; the router is fine."""
    rng = random.Random(seed)
    home = SYNTHETIC_HOME
    wrong = rng.choice([8081, 8000, 8180])
    unit, router = _router_unit(home, [], [], active=True)
    env = (f'# opencode machine settings (desktop)\nset -gx OC_MODEL_BALANCED "local-llama/muse-glimmer-30b-q4-agent"\n'
           f'set -gx OC_LLAMA_BASEURL "http://127.0.0.1:{wrong}/v1"\nset -gx OC_SHELL "fish"\n')
    files = {".config/fish/conf.d/opencode-machine.fish": env, f".config/systemd/user/{unit}": UNIT_FILE,
             MACHINE: _machine(home, {unit: router}), **_wrappers()}
    preset = PRESET.format(name="muse-glimmer-30b-q4-agent", home=home, file="muse-glimmer-30b-Q4_K_XL.gguf", extra="")
    files["models/muse-glimmer-30b-Q4_K_XL.gguf"] = "GGUF placeholder\n"
    return {"place": ".config/llama-cpp", "repo_files": {"models-preset.ini": preset}, "home_files": files,
            "cwd": "", "path": [".local/bin"], "facts": {"wrong_port": wrong, "port": 8080}}


def gpu_hog(seed):
    """A forgotten benchmark loop, started by a user timer, holds the GPU at full load."""
    rng = random.Random(seed)
    home = SYNTHETIC_HOME
    pid = rng.randint(3000, 9000)
    unit, router = _router_unit(home, [], [], active=True)
    units = {unit: router,
             "bench-loop.service": {"description": "Benchmark loop (scratch)", "active": True, "enabled": False,
                                    "result": "success", "requires": [],
                                    "journal": {"active": [f"python3[{pid}]: iteration 48211: 2,903 tok/s"], "failed": []}},
             "bench-loop.timer": {"description": "Restart the benchmark loop every 10 minutes", "active": True,
                                  "enabled": True, "result": "success", "requires": [], "journal": {"active": [], "failed": []}}}
    gpu = [{"pid": pid, "name": "python3", "mib": 17800, "util": 98, "unit": "bench-loop.service"}]
    procs = [{"pid": pid, "cpu": 99.0, "mem": 6.1, "command": f"python3 {home}/Workspace/scratch/bench_loop.py --forever",
              "unit": "bench-loop.service"},
             {"pid": 2290, "cpu": 0.3, "mem": 1.2, "command": "llama-server --models-preset ... --port 8080", "unit": unit}]
    files = {f".config/systemd/user/{unit}": UNIT_FILE, ".config/systemd/user/bench-loop.service":
             f"[Service]\nExecStart=/usr/bin/python3 %h/Workspace/scratch/bench_loop.py --forever\n",
             ".config/systemd/user/bench-loop.timer": "[Timer]\nOnUnitInactiveSec=10min\n\n[Install]\nWantedBy=timers.target\n",
             "Workspace/scratch/bench_loop.py": "while True:\n    run_batch()\n",
             MACHINE: _machine(home, units, gpu, procs), **_wrappers()}
    return {"place": ".config/llama-cpp", "repo_files": {"models-preset.ini": "# presets\n"}, "home_files": files,
            "cwd": "", "path": [".local/bin"], "facts": {"pid": pid}}


def renamed_flag(seed):
    """After a llama.cpp update the router rejects a preset key the new build no longer accepts."""
    rng = random.Random(seed)
    home = SYNTHETIC_HOME
    flag = rng.choice(["defrag-thold", "no-kv-offload-legacy", "draft-p-split"])
    preset = PRESET.format(name="muse-glimmer-30b-q4-agent", home=home, file="muse-glimmer-30b-Q4_K_XL.gguf",
                           extra=f"{flag} = 0.1\n")
    path = f"{home}/.config/llama-cpp/models-preset.ini"
    unit, router = _router_unit(home, [f"llama-server[2211]: build 6650 (b6650-3f1c2a)",
                                       f"llama-server[2211]: error: invalid argument: --{flag}"],
                                [{"file_lacks": [path, rf"^\s*{flag}\s*="]}])
    files = {"models/muse-glimmer-30b-Q4_K_XL.gguf": "GGUF placeholder\n", f".config/systemd/user/{unit}": UNIT_FILE,
             MACHINE: _machine(home, {unit: router}), **_wrappers()}
    return {"place": ".config/llama-cpp", "repo_files": {"models-preset.ini": preset}, "home_files": files,
            "cwd": "", "path": [".local/bin"], "facts": {"flag": flag}}


def broken_symlink(seed):
    """The router's binary symlink still points at a build directory that was cleaned up."""
    rng = random.Random(seed)
    home = SYNTHETIC_HOME
    old = rng.choice(["build-old", "build-cuda12", "build-2026-08"])
    unit, router = _router_unit(home, [f"systemd[1]: {unit_name()}: Failed at step EXEC spawning {home}/.local/bin/llama-server: No such file or directory"],
                                [{"file_exists": f"{home}/.local/bin/llama-server"}])
    files = {"src/llama.cpp/build/bin/llama-server": "#!/bin/sh\necho llama-server build 6650\n",
             f".config/systemd/user/{unit}": UNIT_FILE, MACHINE: _machine(home, {unit: router}), **_wrappers()}
    preset = PRESET.format(name="muse-glimmer-30b-q4-agent", home=home, file="muse-glimmer-30b-Q4_K_XL.gguf", extra="")
    files["models/muse-glimmer-30b-Q4_K_XL.gguf"] = "GGUF placeholder\n"
    return {"place": ".config/llama-cpp", "repo_files": {"models-preset.ini": preset}, "home_files": files,
            "symlinks": {".local/bin/llama-server": f"../../src/llama.cpp/{old}/bin/llama-server"},
            "executables": ["src/llama.cpp/build/bin/llama-server"],
            "cwd": "", "path": [".local/bin"], "facts": {"old": old}}


def unit_name():
    return "llama-router.service"


def models_to_keep(seed):
    """Advisory: a models folder with several quants per family and the owner's rules in the config repo."""
    home = SYNTHETIC_HOME
    names = ["muse-glimmer-30b-Q4_K_XL.gguf", "muse-glimmer-30b-Q4_K_M.gguf", "muse-glimmer-30b-Q3_K_M.gguf",
             "muse-glimmer-30b-Q8_0.gguf", "qwen3.8-27b-Q4_K_XL.gguf", "qwen3.8-27b-Q3_K_S.gguf", "qwen3.8-27b-Q6_K.gguf"]
    files = {f"models/{n}": f"GGUF placeholder for {n}\n" for n in names}
    preset = (PRESET.format(name="muse-glimmer-30b-q4-agent", home=home, file=names[0], extra="")
              + PRESET.format(name="qwen3.8-27b-q4-agent", home=home, file=names[4], extra=""))
    readme = ("# llama-cpp config\n\nRules (owner, 2026-10-02):\n- No Q3 or smaller quants for now.\n"
              "- The restore target is muse-glimmer-30b-q4-agent; never delete its model file.\n"
              "- Disk is tight: about 60 GB free on the models volume.\n")
    return {"place": ".config/llama-cpp", "repo_files": {"models-preset.ini": preset, "README.md": readme},
            "home_files": files, "cwd": "", "path": [], "facts": {}}


def _git_repo(path, files=None, init_args=()):
    path.mkdir(parents=True, exist_ok=True)
    git(["init", "-q", *init_args], path, isolated=True)
    git(["config", "user.name", "Workspace Owner"], path, isolated=True)
    git(["config", "user.email", "owner@example.invalid"], path, isolated=True)
    return path


def _commit(repo, files, message, deletions=()):
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    for name in deletions:
        (repo / name).unlink()
    git(["add", "-A"], repo, isolated=True)
    git(["commit", "-q", "--no-verify", "-m", message], repo, isolated=True)
    return git(["rev-parse", "HEAD"], repo).strip()


def _surgery(place, setup, facts=None):
    return {"place": place, "repo_files": {}, "home_files": {}, "cwd": "", "path": [], "setup": setup,
            "facts": facts if facts is not None else {}}


def promote_experiment(seed):
    """Move an experiment out of the sandbox monorepo into its own repo, keeping its history."""
    facts = {}
    def setup(home):
        repo = _git_repo(home / "Workspace/sandbox", init_args=["-b", "main"])
        _commit(repo, {"README.md": "# sandbox\n", "experiments/other/x.py": "print('x')\n"}, "Start sandbox")
        _commit(repo, {"experiments/fanctl/fanctl.py": "def curve(t):\n    return min(100, max(30, t))\n"}, "fanctl: first curve")
        _commit(repo, {"experiments/other/x.py": "print('x2')\n"}, "other: tweak")
        _commit(repo, {"experiments/fanctl/fanctl.py": "def curve(t, floor=30):\n    return min(100, max(floor, t))\n",
                       "experiments/fanctl/README.md": "# fanctl\n"}, "fanctl: configurable floor")
        _commit(repo, {"experiments/fanctl/test_fanctl.py": "from fanctl import curve\nassert curve(10) == 30\n"}, "fanctl: add test")
        facts["subjects"] = ["fanctl: first curve", "fanctl: configurable floor", "fanctl: add test"]
    return _surgery("Workspace/sandbox", setup, facts)


def reconcile_diverged(seed):
    """The desktop clone and origin (pushed from the laptop) have both moved on."""
    facts = {}
    def setup(home):
        origin = home / "git/dots.git"
        origin.mkdir(parents=True)
        git(["init", "-q", "--bare", "-b", "main"], origin, isolated=True)
        laptop = _git_repo(home / ".cache/laptop-dots", init_args=["-b", "main"])
        _commit(laptop, {"fish/config.fish": "set -gx EDITOR nvim\n"}, "fish: editor")
        git(["push", "-q", str(origin), "main"], laptop, isolated=True)
        git(["clone", "-q", str(origin), str(home / "Workspace/dots")], home, isolated=True)
        desk = home / "Workspace/dots"
        git(["config", "user.name", "Workspace Owner"], desk, isolated=True)
        git(["config", "user.email", "owner@example.invalid"], desk, isolated=True)
        git(["remote", "set-url", "origin", "../../git/dots.git"], desk, isolated=True)
        facts["laptop"] = [_commit(laptop, {"git/config": "[pull]\n\trebase = false\n"}, "git: pull config"),
                           _commit(laptop, {"fish/abbr.fish": "abbr g git\n"}, "fish: abbreviations")]
        git(["push", "-q", str(origin), "main"], laptop, isolated=True)
        facts["desktop"] = [_commit(desk, {"nvim/init.lua": "vim.o.number = true\n"}, "nvim: line numbers"),
                            _commit(desk, {"fish/config.fish": "set -gx EDITOR nvim\nset -gx PAGER less\n"}, "fish: pager")]
        import shutil as _shutil
        _shutil.rmtree(laptop)
    return _surgery("Workspace/dots", setup, facts)


def revert_pushed(seed):
    """A pushed commit removed a function other code needs; undo it without rewriting origin."""
    facts = {}
    def setup(home):
        origin = home / "git/fancurves.git"
        origin.mkdir(parents=True)
        git(["init", "-q", "--bare", "-b", "main"], origin, isolated=True)
        repo = _git_repo(home / "Workspace/fancurves", init_args=["-b", "main"])
        _commit(repo, {"curves.py": "def quiet(t):\n    return max(25, t - 10)\n\ndef loud(t):\n    return min(100, t + 20)\n",
                       "main.py": "from curves import quiet, loud\n"}, "Curves")
        facts["bad"] = _commit(repo, {"curves.py": "def loud(t):\n    return min(100, t + 20)\n"}, "tidy helpers")
        _commit(repo, {"README.md": "# fancurves\n"}, "Add README")
        git(["remote", "add", "origin", str(origin)], repo, isolated=True)
        git(["push", "-q", "origin", "main"], repo, isolated=True)
        git(["remote", "set-url", "origin", "../../git/fancurves.git"], repo, isolated=True)
    return _surgery("Workspace/fancurves", setup, facts)


def recover_branch(seed):
    """A branch with unmerged work was force-deleted; its commits survive only in the reflog."""
    facts = {}
    def setup(home):
        repo = _git_repo(home / "Workspace/thermal", init_args=["-b", "main"])
        _commit(repo, {"thermal.py": "LIMIT = 80\n"}, "Thermal limits")
        git(["checkout", "-q", "-b", "feature/fan-curves"], repo, isolated=True)
        _commit(repo, {"curves.py": "STEPS = [30, 50, 80]\n"}, "Fan curve steps")
        facts["tip"] = _commit(repo, {"curves.py": "STEPS = [30, 50, 70, 100]\n"}, "Finer fan curve steps")
        git(["checkout", "-q", "main"], repo, isolated=True)
        git(["branch", "-q", "-D", "feature/fan-curves"], repo, isolated=True)
    return _surgery("Workspace/thermal", setup, facts)


def split_commit(seed):
    """The last, unpushed commit mixes a code change and a docs change."""
    facts = {}
    def setup(home):
        origin = home / "git/fanctl.git"
        origin.mkdir(parents=True)
        git(["init", "-q", "--bare", "-b", "main"], origin, isolated=True)
        repo = _git_repo(home / "Workspace/fanctl", init_args=["-b", "main"])
        facts["base"] = _commit(repo, {"src/fan.py": "def speed(t):\n    return t\n", "docs/usage.md": "# Usage\n"}, "Start fanctl")
        git(["remote", "add", "origin", str(origin)], repo, isolated=True)
        git(["push", "-q", "origin", "main"], repo, isolated=True)
        git(["remote", "set-url", "origin", "../../git/fanctl.git"], repo, isolated=True)
        _commit(repo, {"src/fan.py": "def speed(t):\n    return min(100, max(30, t))\n",
                       "docs/usage.md": "# Usage\n\nspeed() clamps to 30-100.\n"}, "Clamp speed and document it")
    return _surgery("Workspace/fanctl", setup, facts)


def chat_room(seed):
    """An almost empty home for conversations that need no files."""
    return {"place": "Workspace/scratch", "repo_files": {"README.md": "scratch space\n"}, "home_files": {},
            "cwd": "", "path": [], "facts": {}}


def router_with_history(seed):
    """The bad-model-path fault, in a home with long machine notes and a failed unit that is not
    the problem (a ground rule in the prelude says to leave it alone)."""
    spec = router_bad_model_path(seed)
    rng = random.Random(f"history:{seed}")
    home = SYNTHETIC_HOME
    days = []
    for day in range(1, 31):
        events = [f"- {rng.choice(['updated', 'rebuilt', 'retuned', 'benchmarked', 'pruned'])} "
                  f"{rng.choice(['llama.cpp', 'presets', 'opencode', 'fan curves', 'models folder', 'router unit'])}: "
                  + " ".join(rng.choice(WORDS_NOTES) for _ in range(40)) for _ in range(6)]
        days.append(f"## 2026-09-{day:02d}\n" + "\n".join(events))
    machine = json.loads(spec["home_files"][MACHINE])
    machine["units"]["obsidian-sync.service"] = {
        "description": "Vault sync", "active": False, "enabled": True, "result": "exit-code", "requires": [],
        "journal": {"failed": ["obsidian-sync[812]: remote unreachable"], "active": []}}
    spec["home_files"][MACHINE] = json.dumps(machine, indent=2)
    spec["home_files"]["notes/machine-history.md"] = "# Machine history\n\n" + "\n\n".join(days) + "\n"
    spec["home_files"]["notes/gpu-incidents.md"] = "# GPU incidents\n\n" + "\n".join(
        f"- 2026-09-{rng.randint(1, 30):02d}: " + " ".join(rng.choice(WORDS_NOTES) for _ in range(60)) for _ in range(80)) + "\n"
    return spec


WORDS_NOTES = ("router", "preset", "context", "VRAM", "fans", "thermal", "rebuild", "quant", "draft", "kv", "cache",
               "restart", "journal", "unit", "timeout", "latency", "prefill", "decode", "model", "symlink", "port",
               "compaction", "opencode", "llama-server", "gguf", "throughput", "baseline", "probe", "health", "budget")


def document_room(seed):
    return {'place': 'Workspace/documents', 'repo_files': {'README.txt': 'Answer only from the corpus in the user prompt.\n'},
            'home_files': {}, 'facts': {}, 'cwd': 'Workspace/documents', 'path': []}


REPLICAS = {"document-room": document_room, "router-bad-model-path": router_bad_model_path, "chat-room": chat_room,
            "router-with-history": router_with_history, "opencode-wrong-port": opencode_wrong_port,
            "gpu-hog": gpu_hog, "renamed-flag": renamed_flag, "broken-symlink": broken_symlink,
            "models-to-keep": models_to_keep, "promote-experiment": promote_experiment,
            "reconcile-diverged": reconcile_diverged, "revert-pushed": revert_pushed,
            "recover-branch": recover_branch, "split-commit": split_commit}


def build(task, out_dir):
    """Build the snapshot dir for a replica task; returns its manifest."""
    source = task["source"]
    if source["replica"] not in REPLICAS:
        raise ValueError(f"unknown replica {source['replica']}")
    spec = REPLICAS[source["replica"]](source["seed"])
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    temporary = Path(tempfile.mkdtemp(prefix="llm-eval-replica-"))
    try:
        overlay = temporary / "home"
        overlay.mkdir(parents=True, exist_ok=True)
        for relative, text in spec["home_files"].items():
            path = overlay / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            if relative.startswith(".local/bin/"):
                path.chmod(0o755)
        for relative, target in spec.get("symlinks", {}).items():
            (overlay / relative).parent.mkdir(parents=True, exist_ok=True)
            (overlay / relative).symlink_to(target)
        for relative in spec.get("executables", []):
            (overlay / relative).chmod(0o755)
        in_home = "setup" in spec
        if in_home:
            # Repos with history, branches, reflogs and bare remotes live in the home overlay itself.
            overlay.mkdir(parents=True, exist_ok=True)
            spec["setup"](overlay)
            work = overlay / spec["place"]
        else:
            work = temporary / "repo"
            for relative, text in spec["repo_files"].items():
                (work / relative).parent.mkdir(parents=True, exist_ok=True)
                (work / relative).write_text(text)
            git(["init", "-q", "-b", BRANCH], work, isolated=True)
            git(["add", "-A"], work, isolated=True)
            git(["commit", "-q", "--no-verify", "-m", "config"], work, isolated=True)
            git(["bundle", "create", "-q", str(out / "repo.bundle"), BRANCH], work, isolated=True)
        hits = scan(work) + scan(overlay)
        if hits:
            raise ValueError(f"replica refused, likely secrets: {hits[:5]}")
        for log in overlay.rglob(".git/logs/**/*"):
            # A local clone's reflog names the build's temporary path ("clone: from /tmp/..."); say ~ instead
            if log.is_file():
                log.write_text(log.read_text().replace(str(overlay), "~"))
        for index in overlay.rglob(".git/index"):
            # The tar leaves each index out (it holds inode and ctime, so two builds never match); unpack_home
            # rebuilds it from HEAD, which is only faithful when nothing was staged.
            if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=index.parent.parent).returncode:
                raise ValueError(f"replica stages changes in {index.parent.parent.relative_to(overlay)}; "
                                 "the snapshot would lose them")
        with tarfile.open(out / "home.tar", "w", format=tarfile.PAX_FORMAT) as archive:
            archive.add(overlay, arcname=".", filter=_reproducible)
        manifest = {"task": task["id"], "repo": f"replica:{source['replica']}", "replica": source["replica"],
                    "seed": source["seed"], "commit": None, "answer": None, "history": False, "workdir": ".",
                    "place": spec["place"], "cwd": spec["cwd"], "path": spec["path"], "removed": [],
                    "repo_in_home": in_home, "files": tree_files(work), "home_files": tree_files(overlay),
                    "hidden": {}, "bundle_sha256": None if in_home else digest(out / "repo.bundle"),
                    "home_sha256": digest(out / "home.tar")}
        atomic_json(out / "manifest.json", manifest)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _reproducible(member):
    """Same overlay, same bytes: the tar carries no build time or owner, so the task identity (which hashes
    ``home.tar``) changes only when the content does. Two prepares of one task used to differ by mtime alone,
    then by the git index (stat data) alone, which ``unpack_home`` rebuilds instead."""
    if member.name.endswith("/.git/index"):
        return None
    member.mtime = 0
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    member.pax_headers = {}
    return member


def facts(task):
    """The seeded facts behind a replica (what is wrong), for authoring checks and tests. Git
    replicas run their setup in a throwaway dir; fixed identities and dates make the SHAs the
    same as in the real build."""
    spec = REPLICAS[task["source"]["replica"]](task["source"]["seed"])
    if "setup" in spec:
        with tempfile.TemporaryDirectory() as tmp:
            spec["setup"](Path(tmp))
    return spec["facts"]


def unpack_home(snapshot_dir, home):
    archive = Path(snapshot_dir) / "home.tar"
    if archive.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(home, filter="data")
        for dot_git in sorted(Path(home).rglob(".git")):
            if dot_git.is_dir() and not (dot_git / "index").exists():
                git(["reset", "-q"], dot_git.parent, isolated=True)  # the index the tar left out, from HEAD
