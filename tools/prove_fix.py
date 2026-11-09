"""Prove mined fix candidates (``tools/mine_fixes.py``): a candidate becomes a task only if its own tests fail at
the parent and pass at the commit, in a clean clone.

    python3 tools/prove_fix.py candidates.json --repo flight-test --top 8 --out proven.jsonl [--env repo=python]

Each candidate is cloned (``git clone --shared``, the source repo is never written), checked out at the parent
with only the commit's test files applied (the model's starting point plus the hidden tests), run, then checked
out at the commit and run again. Tests run with HOME pointed at a scratch folder so nothing touches the owner's
home. ``node_modules`` is linked from the source checkout when it has one. Repos whose tests drive terminals or
devices are refused (EXCLUDED).
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

EXCLUDED = {"firstmate": "its tests drive tmux and terminal sessions"}
TIMEOUT = 600
# A parent that fails only because the commit's tests call something not written yet: real, but the packet must
# name that API (the haystack lesson), so it is recorded apart from a behaviour failure.
API = re.compile(r"ImportError|ModuleNotFoundError|cannot import name|Cannot find module|ERR_MODULE_NOT_FOUND|"
                 r"does not provide an export named|is not exported", re.I)  # import-time only (review 2026-11-01)
# Only these reach a third-party test suite: no tokens or keys from the owner's environment.
ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "USER", "VIRTUAL_ENV", "FNM_DIR", "FNM_MULTISHELL_PATH",
            "FNM_NODE_DIST_MIRROR", "FNM_VERSION_FILE_STRATEGY")  # node via fnm still resolves; no npm_config_* tokens


def run(argv, cwd, env, timeout=TIMEOUT):
    try:
        done = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, errors="replace",
                              timeout=timeout)
        return done.returncode, (done.stdout + done.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return "timeout", ""
    except OSError as exc:
        return "error", str(exc)


def package_root(clone, path):
    here = (clone / path).parent
    while here != clone.parent:
        if (here / "package.json").exists():
            return here
        here = here.parent
    return clone


def python_root(clone, path):
    """The folder a Python test expects as its working directory: the parent of its tests folder."""
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part in ("tests", "test", "tests_integration"):
            return clone.joinpath(*parts[:index]) if index else clone
    return (clone / path).parent


def commands(clone, tests, python):
    """[(cwd, argv)] that run exactly the commit's test files."""
    groups = {}
    for path in tests:
        if not (clone / path).is_file():
            continue
        name = Path(path).name
        if path.endswith(".py") and (name.startswith("test") or name.endswith("_test.py")):
            root = python_root(clone, path)
            groups.setdefault(("py", root), []).append(str((clone / path).relative_to(root)))
        elif any(path.endswith(f".{kind}.{ext}") for kind in ("test", "spec") for ext in ("js", "mjs", "cjs", "ts", "tsx", "mts")):
            root = package_root(clone, path)
            groups.setdefault(("js", root), []).append(str((clone / path).relative_to(root)))
        elif name.startswith("test-") and path.endswith((".mjs", ".js")):
            groups.setdefault(("script", package_root(clone, path)), []).append(path)
        elif path.endswith(".sh") and ("test" in name):
            groups.setdefault(("sh", clone), []).append(path)
    out = []
    for (kind, root), files in groups.items():
        if kind == "py":
            has_pytest = subprocess.run([python, "-c", "import pytest"], capture_output=True).returncode == 0
            if has_pytest:
                out.append((root, [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files]))
            else:
                modules = [f[:-3].replace("/", ".") for f in files]
                out.append((root, [python, "-m", "unittest", *modules]))
        elif kind == "js":
            package = json.loads((root / "package.json").read_text())
            deps = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
            vitest = root / "node_modules" / ".bin" / "vitest"
            if "vitest" in deps and vitest.exists():
                out.append((root, [str(vitest), "run", *files]))
            else:
                out.append((root, ["node", "--test", *files]))
        elif kind == "script":
            for path in files:
                out.append((root, ["node", str(clone / path)]))
        else:
            for path in files:
                out.append((clone, ["bash", path]))
    return out


def suite(clone, tests, python, env):
    results = []
    for cwd, argv in commands(clone, tests, python):
        code, tail = run(argv, cwd, env)
        results.append({"argv": argv[-3:], "cwd": str(cwd.relative_to(clone)), "code": code, "tail": tail})
    return results


def prove(candidate, python, scratch):
    repo, sha = Path(candidate["repo"]), candidate["sha"]
    if repo.name in EXCLUDED:
        return {"status": "excluded", "why": EXCLUDED[repo.name]}
    clone = scratch / f"{repo.name}-{sha[:8]}"
    shutil.rmtree(clone, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(repo), str(clone)], check=True,
                   capture_output=True)
    git = lambda *a: subprocess.run(["git", "-C", str(clone), *a], capture_output=True, text=True)
    if git("checkout", "-q", f"{sha}^").returncode:
        return {"status": "error", "why": "no parent"}
    for path in candidate["tests"]:  # the hidden tests, as the commit has them; a deleted test stays deleted
        git("checkout", "-q", sha, "--", path)
    for root in {package_root(clone, p) for p in candidate["tests"]} | {clone}:
        modules = repo / root.relative_to(clone) / "node_modules"  # each package's own, as the source checkout has it
        if modules.is_dir() and not (root / "node_modules").exists():
            (root / "node_modules").symlink_to(modules)
    home = scratch / "home"
    home.mkdir(exist_ok=True)
    env = {**{k: os.environ[k] for k in ENV_KEEP if k in os.environ}, "HOME": str(home), "CI": "1", "NO_COLOR": "1"}
    before = suite(clone, candidate["tests"], python, env)
    if not before:
        return {"status": "no_runner", "why": "no test file this tool knows how to run"}
    git("checkout", "-q", "-f", sha)
    after = suite(clone, candidate["tests"], python, env)
    failed_before = any(r["code"] != 0 for r in before)
    passed_after = all(r["code"] == 0 for r in after)
    api = failed_before and API.search(" ".join(r["tail"] for r in before if r["code"] != 0))
    status = (("proven_api" if api else "proven") if failed_before and passed_after else
              "passes_at_parent" if not failed_before else "fails_at_commit")
    shutil.rmtree(clone, ignore_errors=True)
    return {"status": status, "before": before, "after": after}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates")
    parser.add_argument("--repo", action="append", default=[])
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--min-score", type=int, default=7)
    parser.add_argument("--env", action="append", default=[], help="repo=path/to/python for that repo's tests")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    pythons = dict(item.split("=", 1) for item in args.env)
    data = json.loads(Path(args.candidates).read_text())
    scratch = Path(tempfile.mkdtemp(prefix="prove-fix-"))
    with open(args.out, "a") as out:
        for repo in args.repo or list(data):
            chosen = [c for c in data.get(repo, []) if c["fix"] and c["score"] >= args.min_score][:args.top]
            for candidate in chosen:
                result = prove(candidate, pythons.get(repo, sys.executable), scratch)
                record = {**{k: candidate[k] for k in ("repo", "sha", "date", "subject", "source", "tests",
                                                        "source_lines", "score")}, **result}
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(f"{repo:20s} {candidate['sha'][:8]} {result['status']:16s} {candidate['subject'][:70]}",
                      flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
