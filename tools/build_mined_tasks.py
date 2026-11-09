#!/usr/bin/env python3
"""Turn hand-written packets for proven mined fixes (``tools/prove_fix.py``) into workload tasks.

    python3 tools/build_mined_tasks.py suites/mined-specs.json            # write fixtures/workload/<collection>/
    python3 tools/build_mined_tasks.py suites/mined-specs.json --check    # rebuild in memory, report drift

A spec holds what only a person can write (the packet: the issue as the commit message states it, plus every value
the hidden tests read, never the fix) and the test command. Everything else comes from git: the parent commit is
the start, the fix commit is the answer, the commit's test files are withheld and revealed at grading, and the
expected test names are the tests the commit added or changed. ``leaks`` refuses a packet that quotes a line of the
fix's source diff (E4, third proof); the no-op and solved proofs are ``tests_integration/test_collection_tasks.py``.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from llm_eval.workload.prepare import TASK_DIR  # noqa: E402
from llm_eval.paths import DATA  # noqa: E402
from llm_eval.workload.envs import recorded  # noqa: E402
from llm_eval.workload.tasks import validate  # noqa: E402

TEST_PATH = re.compile(r"(^|/)(tests?|__tests__)/|(^|/)test[-_][^/]*$|\.test\.[a-z]+$|_test\.[a-z]+$", re.I)
# How each runner reports success, its test count, one passing test by name and, where a skipped test's line could
# still match its name, any skip at all (``skipped``: red-team RT-001; pytest's PASSED and vitest's check mark already
# tell a skip apart, and the bash and checks runners have none).
RUNNERS = {
    "node": {"success": r"(?m)^(?:ℹ|#) fail 0$", "count": r"(?m)^(?:ℹ|#) tests (\d+)$",
             "skipped": r"(?m)^(?:ℹ|#) (?:skipped|todo) [1-9]",
             "name": r"(?m)^\s*(?:ok \d+ - |✔ ){}(?: \(|$)",
             "decl": r"""\b(?:test|it)\(\s*(["'`])(.+?)\1"""},
    "vitest": {"success": r"(?m)Tests\s+\d+ passed", "count": r"(?m)Tests\s+(\d+) passed",
               "name": r"(?m)✓ .*{}", "decl": r"""\b(?:test|it)\(\s*(["'`])(.+?)\1"""},
    "pytest": {"success": r"(?m)\b\d+ passed\b", "count": r"(?m)\b(\d+) passed\b",  # with -o addopts= (see specs)
               "name": r"(?m)::{} PASSED", "decl": r"(?m)^\+\s*(?:async )?def (test_\w+)"},
    # unittest -v can split a test's name from its "ok" with the test's own stderr, so the name alone does not say it
    # passed: the runner's final OK with exit code 0 says every test that ran passed, and ``skipped`` refuses any skip
    # or expected failure (source the agent edits can mark tests skipped: RT-001).
    "unittest": {"success": None, "count": None, "name": r"(?m)^{} \(",
                 "skipped": r"(?m)^(?:OK|FAILED) \(.*\b(?:skipped|expected failures)=",
                 "decl": r"(?m)^\+\s*def (test_\w+)"},
    "bash": {"success": None, "count": None, "name": r"(?m)^  ok    {}$",  # the dotfiles shell suites' ok() lines
             "decl": r'(?:\bok|case_run) "([^"]+)"'},
    "checks": {"success": r"(?m)^0/\d+ \w+ checks failed$", "count": None,  # meal-planner's PASS/FAIL scripts
               "name": r"(?m)^PASS {}$", "decl": r"""\[\s*'([^']+)',"""},
}


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout


def test_files(repo, sha):
    return [p for p in git(repo, "show", "--name-only", "--format=", sha).split() if TEST_PATH.search(p)]


def changed_tests(repo, sha, runner, files):
    """Names of the tests the commit added or changed, in file order."""
    decl = RUNNERS[runner]["decl"]
    if not decl:
        return []
    diff = git(repo, "show", "--format=", "-U0", sha, "--", *files)
    names = []
    for match in re.finditer(decl, diff):
        line_start = diff.rfind("\n", 0, match.start()) + 1
        if not diff[line_start:].startswith("+"):
            continue
        name = match.group(match.lastindex)
        if name not in names and "${" not in name:
            names.append(name)
    return names


def frozen_glob(path):
    """The test directory a withheld file sits in (``studio/tests/**``), or the file itself when it sits beside
    source (``serve/test_server.py``), so freezing tests never freezes the code the model must change."""
    parts = path.split("/")
    for i, part in enumerate(parts[:-1]):
        if re.fullmatch(r"tests?|__tests__", part, re.I):
            return "/".join(parts[:i + 1]) + "/**"
    return path


def leaks(repo, sha, prompt, files):
    """Lines of the fix's source diff (30 characters or more, not tests) that the packet quotes."""
    diff = git(repo, "show", "--format=", "-U0", sha, "--", ".", *[f":(exclude){f}" for f in files])
    added = {line[1:].strip() for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")}
    return sorted(line for line in added if len(line) >= 30 and line in prompt)


def env_digest(name):
    """The content digest an environment's ENV.json records (``python3 -m llm_eval.workload.envs`` writes it)."""
    digest = recorded(DATA / "envs" / name)
    if not digest:
        raise ValueError(f"environment {name} has no digest; run python3 -m llm_eval.workload.envs {name}")
    return digest


def build(spec):
    repo = Path(spec["repo"]).expanduser()
    sha = git(repo, "rev-parse", spec["sha"]).strip()
    parent = git(repo, "rev-parse", f"{sha}^").strip()
    files = spec.get("withhold") or test_files(repo, sha)
    runner = spec["runner"]
    preset = RUNNERS[runner]
    names = spec.get("expect_tests") or changed_tests(repo, sha, runner, files)
    expect = [preset["name"].format(re.escape(n)) for n in names] if preset["name"] else []
    check = {"type": "tests", "command": spec["command"], "cwd": spec.get("cwd", "."), "timeout": spec.get("timeout", 600),
             "expect": expect}
    for key in ("success", "count", "skipped"):
        if spec.get(key, preset.get(key)):
            check[key] = spec.get(key, preset.get(key))
    if spec.get("min_tests"):
        check["min_tests"] = spec["min_tests"]
    frozen = sorted({frozen_glob(f) for f in files})
    task = {"id": spec["id"], "kind": "workload", "category": spec["collection"],
            "prompt": spec["prompt"],
            "source": {"repo": str(repo).replace(str(Path.home()), "~"), "commit": parent, "workdir": ".",
                       "answer": sha, "withhold": files},
            "checks": [check, {"type": "diff_scope", "allow": ["**"], "frozen": spec.get("frozen", frozen),
                               "min_changed": 1, "deny": []}],
            "limits": {"turns": 4, "wall_seconds": spec.get("wall_seconds", 2700)},
            "solution": {"apply_reference": True, "answer": "Implemented; the tests pass."},
            "followups": {"tests": {"command": spec["command"], "cwd": spec.get("cwd", "."), "max": 2}},
            "mined": {"repo": repo.name, "fix": sha, "subject": git(repo, "log", "-1", "--format=%s", sha).strip(),
                      "kind": spec.get("kind", "behavior")}}
    if spec.get("env"):
        task["env"] = [{**entry, "digest": env_digest(entry["from"])} for entry in spec["env"]]
    if spec.get("deny"):
        task["source"]["deny"] = spec["deny"]
    validate(dict(task))
    found = leaks(repo, sha, spec["prompt"], files)
    if found:
        raise ValueError(f"{spec['id']}: the packet quotes the fix: {found[:3]}")
    if expect == [] and preset["name"]:
        raise ValueError(f"{spec['id']}: no changed test names found; set expect_tests")
    return task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", type=Path)
    parser.add_argument("--check", action="store_true", help="report tasks whose file differs from a rebuild")
    parser.add_argument("--only", default="", help="comma-separated spec ids")
    args = parser.parse_args(argv)
    only = set(filter(None, args.only.split(",")))
    drift = 0
    for spec in json.loads(args.specs.read_text())["tasks"]:
        if only and spec["id"] not in only:
            continue
        task = build(spec)
        path = TASK_DIR / spec["collection"] / f"{spec['id']}.json"
        text = json.dumps(task, indent=2) + "\n"
        if args.check:
            same = path.exists() and path.read_text() == text
            drift += not same
            print(f"{spec['id']}: {'ok' if same else 'DIFFERS'}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"{spec['id']}: {len(task['checks'][0]['expect'])} named tests, withheld {task['source']['withhold']}")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
