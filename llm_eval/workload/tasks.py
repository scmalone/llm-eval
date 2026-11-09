"""Workload task specs: one real-work task per file, validated before anything is planned.

A task names where its starting state comes from (a pinned commit of a code repo, or a
synthetic replica), the first user turn, the rule-based follow-ups, and the code checks that
decide pass or fail. Expected answers live only in ``checks``; nothing there reaches the model.
"""
from pathlib import Path
import re

from ..plan import read_manifest
from ..state import safe_id

CATEGORIES = ("resume", "implementation", "ops", "surgery", "lookup", "advisory", "review")
# The accuracy suite's added collections (plans/2026-11-01-accuracy-suite.md), each in its own subfolder of the task
# directory so the core suite stays 35 tasks: fixes mined from the owner's repos, the game category, and vision.
COLLECTIONS = ("mined", "game", "vision")
ENV_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
CHECKS = ("answer", "box", "tests", "diff_scope", "files", "git", "state", "commands", "style", "findings", "turn_answers",
          "unit_ready", "depth")
SHA = re.compile(r"[0-9a-f]{40}")


def _regexes(values, where):
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError(f"{where} must be a list of regexes")
    for value in values:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"{where}: bad regex {value!r}: {exc}") from exc


def _check(check, index):
    where = f"checks[{index}]"
    kind = check.get("type")
    if kind not in CHECKS:
        raise ValueError(f"{where}: unknown check type {kind!r}")
    if kind == "answer":
        if not check.get("all") and not check.get("none"):
            raise ValueError(f"{where}: answer needs 'all' or 'none'")
        _regexes(check.get("all", []), where + ".all")
        _regexes(check.get("none", []), where + ".none")
        _regexes(check.get("one_value", []), where + ".one_value")
        if any(re.compile(v).groups != 1 for v in check.get("one_value", [])):
            raise ValueError(f"{where}.one_value: each regex captures the field's value as its one group")
    elif kind == "tests":
        if not check.get("command") or not all(isinstance(a, str) for a in check["command"]):
            raise ValueError(f"{where}: tests needs a command argv")
        _regexes([check[k] for k in ("success", "count", "skipped") if k in check] + check.get("expect", []), where)
    elif kind == "diff_scope":
        if not any(check.get(k) for k in ("allow", "deny", "frozen", "min_changed")):
            raise ValueError(f"{where}: diff_scope needs allow, deny, frozen or min_changed")
    elif kind == "files":
        if not any(check.get(k) for k in ("exists", "absent", "contains", "lacks")):
            raise ValueError(f"{where}: files needs exists, absent, contains or lacks")
        _regexes(list((check.get("contains") or {}).values()), where + ".contains")
        _regexes(list((check.get("lacks") or {}).values()), where + ".lacks")
    elif kind == "git":
        for sha in check.get("contains_commits", []):
            if not SHA.fullmatch(sha):
                raise ValueError(f"{where}: contains_commits needs full SHAs")
        if not any(check.get(k) for k in ("branch", "contains_commits", "origin_fast_forward", "clean", "recent_commits",
                                         "min_commits", "contains_subjects", "synced_with")):
            raise ValueError(f"{where}: git needs at least one assertion")
    elif kind == "state":
        if not check.get("file") or not isinstance(check.get("equals"), dict) or not check["equals"]:
            raise ValueError(f"{where}: state needs file and a non-empty equals")
    elif kind == "commands":
        _regexes(check.get("forbid", []), where + ".forbid")
        _regexes(check.get("require", []), where + ".require")
        if not check.get("forbid") and not check.get("require"):
            raise ValueError(f"{where}: commands needs forbid or require")
    elif kind == "unit_ready":
        if not check.get("unit") or not check.get("requires") or not check.get("replica_home"):
            raise ValueError(f"{where}: unit_ready needs unit, requires and replica_home")
    elif kind == "box":
        target = check.get("target")
        if not (isinstance(target, list) and len(target) == 4 and all(isinstance(v, (int, float)) for v in target)
                and target[0] < target[2] and target[1] < target[3]):
            raise ValueError(f"{where}: box needs a target [x1, y1, x2, y2] with x1 < x2 and y1 < y2")
    elif kind == "depth":
        floor, fraction = check.get("min_peak_tokens"), check.get("min_fraction")
        if floor is not None and (not isinstance(floor, int) or floor <= 0):
            raise ValueError(f"{where}: depth needs a positive min_peak_tokens")
        if fraction is not None and not (isinstance(fraction, (int, float)) and 0 < fraction <= 1):
            raise ValueError(f"{where}: depth min_fraction must be in (0, 1]")
        if floor is None and fraction is None:
            raise ValueError(f"{where}: depth needs min_peak_tokens or min_fraction")
    elif kind == "style":
        if not check.get("options") and not check.get("no_em_dash"):
            raise ValueError(f"{where}: style needs options or no_em_dash")
    elif kind == "turn_answers":
        if not isinstance(check.get("turns"), dict) or not check["turns"]:
            raise ValueError(f"{where}: turn_answers needs turns")
        for turn, rule in check["turns"].items():
            if not str(turn).isdigit() or int(turn) < 1:
                raise ValueError(f"{where}: turn keys are 1-based task turns")
            _regexes(rule.get("all", []), f"{where}.{turn}.all")
            _regexes(rule.get("none", []), f"{where}.{turn}.none")
            if not rule.get("all") and not rule.get("none"):
                raise ValueError(f"{where}.{turn}: needs all or none")
    elif kind == "findings":
        defects = check.get("defects")
        if not defects:
            raise ValueError(f"{where}: findings needs defects")
        for defect in defects:
            safe_id(defect["id"])
            _regexes(defect.get("match", []), f"{where}.{defect['id']}")
            _regexes(defect.get("none", []), f"{where}.{defect['id']}.none")
            if not defect.get("match"):
                raise ValueError(f"{where}.{defect['id']}: a defect needs match regexes")


def _source(source):
    if "replica" in source:
        if "repo" in source or "bundle" in source:
            raise ValueError("a source must be exactly one of replica, repo or bundle")
        safe_id(source["replica"])
        if not isinstance(source.get("seed"), int):
            raise ValueError("replica source needs an integer seed")
        return
    repo, bundle = source.get("repo"), source.get("bundle")
    if bool(repo) == bool(bundle) or not SHA.fullmatch(str(source.get("commit", ""))):
        raise ValueError("source needs exactly one repo or bundle and a full commit SHA")
    if bundle and (Path(bundle).is_absolute() or ".." in Path(bundle).parts or not str(bundle).endswith(".bundle")):
        raise ValueError("bundle must be a relative .bundle path inside fixtures/workload")
    if source.get("answer") is not None and not SHA.fullmatch(source["answer"]):
        raise ValueError("answer must be a full commit SHA")
    if source.get("withhold") and not source.get("answer"):
        raise ValueError("withheld files come from the answer commit, so withhold needs answer")
    for path in source.get("withhold", []) + source.get("deny", []):
        if path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"paths must be relative to the repo: {path}")


def validate(task):
    safe_id(task.get("id", ""))
    if task.get("kind") != "workload":
        raise ValueError("workload tasks need kind 'workload'")
    if task.get("category") not in CATEGORIES + COLLECTIONS:
        raise ValueError(f"category must be one of {', '.join(CATEGORIES + COLLECTIONS)}")
    for mount in task.get("env", []):
        # a prebuilt, read-only environment (venv, node_modules, Blender, a local-only asset) under <data>/envs,
        # mounted into the repo and pinned by its content digest (llm_eval/workload/envs.py)
        if not {"from", "to", "digest"} <= set(mount) <= {"from", "to", "digest", "writable", "files"} \
                or not ENV_NAME.fullmatch(str(mount["from"]).split("/")[0]) \
                or ".." in str(mount["from"]).split("/") or str(mount["to"]).startswith("/") \
                or ".." in str(mount["to"]).split("/") or not re.fullmatch(r"[0-9a-f]{64}", str(mount["digest"])):
            raise ValueError("env entries are {from: <name under the data folder's envs>, to: <repo-relative path>, "
                             "digest: <its ENV.json digest>, writable: [<paths inside it that get a scratch tmpfs>], "
                             "files: [<names: mount only these files, each at to/<name>>]}")
        if any(str(w).startswith("/") or ".." in str(w).split("/") for w in mount.get("writable", [])):
            raise ValueError("env writable paths are relative to the mount")
        if any(not ENV_NAME.fullmatch(str(f)) for f in mount.get("files", [])) or (mount.get("files") and mount.get("writable")):
            raise ValueError("env files are plain names in the folder, and a files mount has no writable paths")
    if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
        raise ValueError("task needs a prompt")
    _source(task.get("source") or {})
    checks = task.get("checks") or []
    if not checks:
        raise ValueError("task needs at least one check")
    for index, check in enumerate(checks):
        _check(check, index)
    limits = task.get("limits") or {}
    if not all(isinstance(limits.get(k), int) and limits[k] > 0 for k in ("turns", "wall_seconds")):
        raise ValueError("limits needs positive integer turns and wall_seconds")
    if "home_mib" in limits and not (isinstance(limits["home_mib"], int) and limits["home_mib"] > 0):
        raise ValueError("limits.home_mib must be a positive integer (MiB)")
    followups = task.get("followups", {})
    for rule in followups.get("faq", []):
        _regexes([rule.get("match", "")], "followups.faq")
    for rule in followups.get("scripted", []):
        if not isinstance(rule.get("after_turn"), int) or not rule.get("say"):
            raise ValueError("scripted follow-ups need after_turn and say")
    if not isinstance(task.get("prelude", []), list) or not all(isinstance(t, str) and t for t in task.get("prelude", [])):
        raise ValueError("prelude must be a list of user turns")
    fill = task.get("fill")
    if fill is not None:
        if not (isinstance(fill.get("prompts"), list) and fill["prompts"] and all(isinstance(p, str) and p for p in fill["prompts"])):
            raise ValueError("fill needs a list of prompts")
        if not (isinstance(fill.get("fraction"), (int, float)) and 0 < fill["fraction"] <= 1):
            raise ValueError("fill fraction must be in (0, 1]")
        if not (isinstance(fill.get("max_turns"), int) and fill["max_turns"] >= 1):
            raise ValueError("fill max_turns must be a positive integer")
    if task.get("runs", 2) < 1:
        raise ValueError("runs must be at least 1")
    return task


def load(path):
    return validate(read_manifest(path))
