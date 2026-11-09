"""One task-run: build its starting state, drive the harness through turns with rule-based
follow-ups, then export, measure and grade. Harness-agnostic: the driver is passed in.

Follow-ups, checked in order after each turn until none applies or a cap is hit:
1. a scripted line due after this turn (a scope change, an "undo that");
2. failing visible tests, pasted back at most ``tests.max`` times (``cwd`` is repo-relative,
   as for the ``tests`` check);
3. a question from the model: the first matching FAQ answer, else "use your judgment" once.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from .capture import RawCapture
from .graders import grade
from .replicas import unpack_home
from .snapshot import materialize
from ..state import atomic_json

DEFAULT_REPLY = "Use your judgment and continue; I won't be answering questions on this one."
# The reply a question-only core task (advisory, lookup, resume, review) sets as its ``followups.default``: the
# general reply read as a go-ahead, and every model that acted on a question did so right after it (owner,
# 2026-11-06, review 2.4).
QUESTION_REPLY = "I won't be answering questions on this one; finish with your answer and change nothing."


class FollowUps:
    def __init__(self, task, run_tests):
        self.rules = task.get("followups", {})
        self.run_tests = run_tests
        self.pasted = 0
        self.defaulted = False
        self.clarifications = 0

    def next(self, turn, answer):
        for rule in self.rules.get("scripted", []):
            if rule["after_turn"] == turn:
                return rule["say"], "scripted"
        tests = self.rules.get("tests")
        if tests and self.pasted < tests.get("max", 2):
            code, output = self.run_tests(tests["command"], tests.get("cwd", "."), tests.get("timeout", 600))
            if code != 0:
                self.pasted += 1
                return f"The tests are failing:\n```\n{output[-3000:]}\n```\nPlease fix this.", "tests"
        if answer.rstrip().endswith("?"):
            self.clarifications += 1
            for rule in self.rules.get("faq", []):
                if re.search(rule["match"], answer, re.IGNORECASE | re.MULTILINE):
                    return rule["say"], "faq"
            if not self.defaulted:
                self.defaulted = True
                return self.rules.get("default", DEFAULT_REPLY), "default"
        return None, None


def turn_stop(limits):
    """A task allowed one turn is finished when it takes it; a multi-turn task that ran out hit its cap.
    The difference matters: "turn_cap" reads as an unfinished run, and the six document tasks were all
    being recorded that way after a turn that had answered the question."""
    return "done" if limits["turns"] <= 1 else "turn_cap"


def _origin_refs(task, workdir, home):
    refs = {}
    for check in task["checks"]:
        origin = check.get("origin_fast_forward") if check["type"] == "git" else None
        if origin:
            where = Path(home) / origin[2:] if origin.startswith("~/") else Path(workdir) / origin
            out = subprocess.run(["git", "for-each-ref", "--format=%(refname) %(objectname)"],
                                 cwd=where, capture_output=True, text=True, timeout=60)
            refs[origin] = dict(line.split() for line in out.stdout.splitlines() if line.strip())
    return refs


# Stops that end a task before it can be judged; the last two are infrastructure, not model verdicts.
UNGRADED = {"wall_cap": "wall-clock cap reached", "budget_cap": "model time budget reached mid-task",
            "stalled": "no progress from opencode or the model server (hung)",
            "tool_hang": "a command the model ran never returned (the server was idle, the harness waited on it)",
            "harness_stall": "opencode never started (stalled at startup twice)",
            "harness_exit": "opencode exited with an error before doing anything",
            "disk_cap": "the task home passed its disk cap (limits.home_mib)"}
TURN_KEYS = ("returncode", "events", "lingered", "restarts", "hung", "tool_open", "max_silence_seconds", "stalled", "disk_cap",
             "home_mib")


def _stop_for(result, capped):
    """Why a turn ends the task, or None when it went through."""
    if result.get("disk_cap"):
        return "disk_cap"
    if result.get("hung"):
        # The model ran a command that blocks (north-mini started a fake server that sleeps forever): its own
        # timeout, not a wedged server (review 2026-11-06, section 4).
        return "tool_hang" if result.get("tool_open") else "stalled"
    return _harness_failure(result) or (capped if result.get("timed_out") else None)


def _harness_failure(result):
    """A turn that says nothing about the model: opencode never started, or crashed before any event."""
    if result.get("stalled"):
        return "harness_stall"
    if result.get("returncode") not in (0, None) and not result.get("events") and not result.get("timed_out") \
            and not result.get("lingered") and not result.get("hung"):
        return "harness_exit"
    return None


def env_mounts(task, repo, root=None, verified=None):
    """[(host, repo path, writable)] for the task's prebuilt environments (E2): each ``env`` entry's ``from`` under
    ``<data>/envs`` mounted read-only at ``to`` inside the repo, with a scratch tmpfs over each ``writable`` path
    inside it (a tool's cache folder, such as vite's ``.vite-temp``). An entry with ``files`` mounts only those
    files from the folder, each at ``to/<name>`` (a local-only asset beside tracked ones, such as flight-test's
    heightfield). Each folder must still hash to the entry's ``digest`` (``envs.verify``; RT-002); ``verified``
    is a set of (folder, digest) pairs already checked by this caller. A missing or changed environment is an
    error, not a quiet run without dependencies."""
    from ..paths import DATA
    from .envs import verify
    base = Path(root) if root else DATA / "envs"
    mounts = []
    for entry in task.get("env", []):
        host = (base / entry["from"]).resolve()
        if not host.is_relative_to(base.resolve()) or not host.is_dir():
            raise ValueError(f"{task['id']}: environment {entry['from']} is absent under {base}")
        key = (str(host), entry.get("digest"))
        if verified is None or key not in verified:
            if not entry.get("digest"):
                raise ValueError(f"{task['id']}: environment {entry['from']} has no digest to verify")
            verify(host, entry["digest"])
            if verified is not None:
                verified.add(key)
        if entry.get("files"):
            for name in entry["files"]:
                if not (host / name).is_file():
                    raise ValueError(f"{task['id']}: {name} is absent from environment {entry['from']}")
                mounts.append((host / name, Path(repo) / entry["to"] / name, ()))
        else:
            mounts.append((host, Path(repo) / entry["to"], tuple(entry.get("writable", ()))))
    return mounts


EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def _mount(box, mounts):
    """Add read-only mounts to a box (the grader's or the harness's); the mount points exist in the home first (an
    empty file for a file mount, which ``run_task`` counts as part of the start so it never reads as an edit)."""
    if box is None:
        return
    for host, target, writable in mounts:
        target = Path(target)
        if Path(host).is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.touch()
        else:
            target.mkdir(parents=True, exist_ok=True)
        pair = (str(host), box.inside(target))
        if pair not in box.read_only:
            box.read_only.append(pair)
        for sub in writable:
            inside = str(Path(box.inside(target)) / sub)
            if inside not in getattr(box, "tmpfs", []):
                box.tmpfs.append(inside)


def run_task(task, run_dir, snapshot_dir, driver_factory, box_factory, home_files=None, clock=time.monotonic, budget=None, compaction_point=None):
    """Run ``task`` once into ``run_dir``; returns the graded result (also saved as result.json).

    ``driver_factory(home, manifest)`` returns a driver context manager; ``box_factory(home)`` the
    sandbox used for test commands; ``home_files`` maps fake-home paths to text (PROJECTS.md and
    the like). The manifest says where the repo sits in the home (``place``), where the harness
    starts (``cwd``, home-relative; default the repo's workdir) and any replica home overlay.
    ``budget`` is what is left of the model's time budget in seconds; when it is shorter than the
    task's own wall cap it ends the task as ``budget_cap`` (an infrastructure stop, not a verdict).
    ``compaction_point`` is the context at which opencode compacts for this run's mode; a task with a ``fill``
    spec grows the session toward ``fraction`` of it before the task's own prompt (a depth scaled to the mode).
    """
    run_dir = Path(run_dir)
    home = run_dir / "home"
    manifest = json.loads((Path(snapshot_dir) / "manifest.json").read_text())
    place = manifest.get("place") or f"Workspace/{Path(manifest['repo']).name}"
    if manifest.get("repo_in_home"):  # replicas whose repos carry history, remotes and reflogs
        unpack_home(snapshot_dir, home)
        repo = home / place
    else:
        repo = materialize(snapshot_dir, _mkparent(home / place))
        unpack_home(snapshot_dir, home)
    base_files = dict(manifest["files"])
    for relative, text in task.get("plant", {}).items():  # uncommitted state the task asks about
        _mkparent(repo / relative).write_text(text)
        base_files[relative] = hashlib.sha256(text.encode()).hexdigest()  # planted is the start, not an edit
    workdir = home / manifest["cwd"] if "cwd" in manifest else repo / manifest["workdir"]
    for relative, text in (home_files or {}).items():
        _mkparent(home / relative).write_text(text)
    origin_refs = _origin_refs(task, repo, home)
    base_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                               timeout=60).stdout.strip()
    box = box_factory(home)
    mounts = env_mounts(task, repo)
    _mount(box, mounts)
    for host, target, _ in mounts:
        if Path(host).is_file():  # its empty mount point is part of the start, like a planted file
            base_files[Path(target).relative_to(repo).as_posix()] = EMPTY_SHA
    raw = RawCapture(run_dir / "raw")
    recorded_run = raw.recording(box.runner())
    run_tests = lambda command, cwd, timeout: recorded_run(command, repo / cwd, timeout)
    followups = FollowUps(task, run_tests)
    wall = task["limits"]["wall_seconds"]
    budget_bound = budget is not None and budget < wall
    capped = "budget_cap" if budget_bound else "wall_cap"
    deadline = clock() + (budget if budget_bound else wall)
    turns, answers, stop = [], [], "done"
    with driver_factory(home, manifest) as driver, _Kept(raw, driver, box, repo, home, base_head):
        if hasattr(driver, "home_cap_mib") and task["limits"].get("home_mib"):
            driver.home_cap_mib = task["limits"]["home_mib"]
        _mount(getattr(driver, "box", None), mounts)  # the agent sees the same environment the grader uses
        # Prelude: scripted opening turns the real model works through (reading, summarizing) so
        # the task starts deep in a grown session. They count toward wall time, not the turn cap.
        for text in task.get("prelude", []):
            remaining = deadline - clock()
            result = driver.turn(text, workdir, remaining) if remaining > 0 else {"timed_out": True}
            stop = _stop_for(result, capped) or "done"
            if stop != "done":
                break
            turns.append({"source": "prelude", "said": text[:200], **{k: result.get(k) for k in TURN_KEYS}})
        # Fill: grow the session toward a fraction of this mode's compaction point, reading real files, so the
        # task is asked at the depth the owner works at (red-team plan 2026-10-04: depth scaled to the mode).
        fill, depth = task.get("fill"), None
        if fill and compaction_point and stop == "done":
            target, reached, used = int(fill["fraction"] * compaction_point), 0, 0
            meter = getattr(driver, "context_tokens", None)
            while stop == "done" and reached < target and used < fill["max_turns"]:
                remaining = deadline - clock()
                if remaining <= 0:
                    stop = capped
                    break
                text = fill["prompts"][used % len(fill["prompts"])]
                result = driver.turn(text, workdir, remaining)
                stop = _stop_for(result, capped) or "done"
                turns.append({"source": "fill", "said": text[:200], **{k: result.get(k) for k in ("timed_out", *TURN_KEYS)}})
                used += 1
                if stop != "done":
                    break
                reached = (meter() if meter else None) or reached
            depth = {"target_tokens": target, "reached_tokens": reached, "fill_turns": used, "compaction_point": compaction_point}
        text, source = task["prompt"], "prompt"
        while stop == "done":
            remaining = deadline - clock()
            if remaining <= 0:
                stop = capped
                break
            result = driver.turn(text, workdir, remaining)
            answers.append(result["answer"])
            turns.append({"source": source, "said": text[:500], **{k: result.get(k) for k in ("timed_out", *TURN_KEYS)}})
            stop = _stop_for(result, capped) or "done"
            if stop != "done":
                break
            if len(answers) >= task["limits"]["turns"]:
                stop = turn_stop(task["limits"])
                break
            text, source = followups.next(len(answers), result["answer"])
            if text is None:
                break
        export = driver.export()
    atomic_json(run_dir / "export.json", export)
    metrics = driver.summarize(export)  # the driver owns its metrics: each harness reports the same numbers
    outcome = {"answer": metrics["answer"], "commands": metrics["commands"], "workdir": repo, "home": home,
               "turn_answers": answers, "metrics": metrics, "git": getattr(box, "git", None),
               "base_files": base_files, "hidden_files": list(manifest.get("hidden", {})), "base_head": base_head,
               "snapshot": Path(snapshot_dir), "origin_refs": origin_refs,
               "run": recorded_run, "compaction_point": compaction_point}
    verdict = grade(task, outcome) if stop not in UNGRADED else {
        "pass": False, "checks": [{"type": "limits", "pass": False, "reason": UNGRADED[stop]}]}
    result = {"task": task["id"], "category": task["category"], "pass": verdict["pass"], "stop": stop,
              "repo": str(repo), "home": str(home), "base_head": base_head,
              "turns": turns, "depth": depth, "clarifications": followups.clarifications, "tests_pasted": followups.pasted,
              "harness_lingered_turns": sum(bool(t.get("lingered")) for t in turns),
              "harness_restarts": sum(t.get("restarts") or 0 for t in turns),
              "grading": verdict, "metrics": {k: v for k, v in metrics.items() if k not in ("answer", "commands")},
              "answer": metrics["answer"]}  # whole: a re-grade must see what the grader saw
    atomic_json(run_dir / "result.json", result)
    return result


class _Kept:
    """On the way out of the task-run, success or not, save what the harness and the workspace hold."""
    def __init__(self, raw, driver, box, repo, home, base_head):
        self.raw, self.driver, self.box, self.repo, self.home, self.base_head = raw, driver, box, repo, home, base_head

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.raw.turns(getattr(self.driver, "raw_turns", []))
        self.raw.end_state(self.box, self.repo, self.home, self.base_head)
        self.raw.opencode_data(self.home)
        self.raw.finish()


def _mkparent(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
