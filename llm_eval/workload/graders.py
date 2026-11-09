"""Code checks for one finished workload task-run. Nothing here calls a model.

``grade(task, outcome)`` runs every check in the task and passes only when all pass. The
outcome carries what the run left behind:

* ``answer``: the final assistant text; ``commands``: shell commands the agent ran
* ``workdir``: the run's repo root; ``home``: its fake home (``~/`` paths in checks resolve there)
* ``base_files``: the snapshot's path -> sha256 map
* ``snapshot``: the snapshot dir (withheld files are revealed from it for ``tests``)
* ``origin_refs``: refs of local bare "origins" before the run, for ``git.origin_fast_forward``
* ``run(argv, cwd, timeout) -> (returncode, output)``: the sandboxed command runner
* ``git(args, cwd) -> (returncode, stdout)``: git run inside the sandbox (the repo is the agent's,
  and its config could name executables); ``metrics``: the session's process metrics

The final tree is hashed before any check runs, so revealing withheld tests cannot read as an
edit the model made. Nothing here follows a path that resolves outside the run's home.
"""
import ast
import fnmatch
import json
from pathlib import Path
import re
import shutil

from . import machine
from .snapshot import escape_reason, matches, reveal_hidden, safe_write, tree_files

# An option line starts with its letter: "A. x", "A) x", "A: x", "(A) x", "**A.** x", "**A** x", "- **B)** x".
# A "(" must close; a bare "X:" counts only for A-H, so prose like "Q: ..." or "I: ..." is not an option.
OPTION_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?(?P<bold>\*\*)?(?P<paren>\()?(?P<letter>[A-Z])"
    r"(?(paren)\)|(?(bold)[.):]?|(?:[.)]|(?<=[A-H]):)))"
    r"(?(bold)\*\*[.):]?)\s", re.MULTILINE)


def _resolve(outcome, path):
    """``~/`` paths are in the run's fake home; others are relative to the repo."""
    if path.startswith("~/"):
        return Path(outcome["home"]) / path[2:]
    return Path(outcome["workdir"]) / path


def _contained(outcome, path):
    """True when ``path`` resolves inside the run's home (an agent's symlink to a real file is not)."""
    home = Path(outcome.get("home") or outcome["workdir"]).resolve()
    resolved = path.resolve()
    return resolved == home or str(resolved).startswith(str(home) + "/")


def _git_runner(outcome):
    if outcome.get("git"):
        return outcome["git"]
    from ..adapters import FakeBox
    return FakeBox().git  # synthetic test repos only; real runs always pass the box's git


def _flags(check):
    return re.MULTILINE | (re.IGNORECASE if check.get("ignore_case") else 0)


# Words that head an answer without being one; a bold or heading line of anything else ("**Use A**", "# NOT IN
# RECORD") is the claim itself.
LABEL = re.compile(r"(?i)(final |short |my |the )?(answer|result|summary|recommendation|conclusion|verdict|tl;?dr)"
                   r"( below)?:?")


SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[*(`\"'])")


def _claim(text):
    """The answer's first sentence of its first line with content: what it claims, before any reasoning that names
    a decoy to reject it. A heading or a bare label ("# Answer", "**Answer**", "Answer:") is skipped, so the claim
    under it is read. A one-line answer ("K-5760 [R-01]. The draft K-9999 is not approved.") is cut at its first
    sentence end, so the explanation does not count as the claim (found 2026-11-07: eight correct one-line
    answers failed)."""
    for line in text.splitlines():
        bare = line.strip().strip("*_#> ").strip()
        if not bare:
            continue
        labelled = line.lstrip().startswith(("#", "**", "__")) or bare.endswith(":")
        if labelled and LABEL.fullmatch(bare):
            continue
        return SENTENCE_END.split(line, maxsplit=1)[0]
    return ""


def check_answer(check, outcome, _):
    text = outcome.get("answer") or ""
    missing = [p for p in check.get("all", []) if not re.search(p, text, _flags(check))]
    # ``none_scope: "claim"`` forbids a decoy only in the claimed answer, so "K-9999 is a draft, not approved"
    # in the explanation is not read as answering K-9999.
    scoped = _claim(text) if check.get("none_scope") == "claim" else text
    present = [p for p in check.get("none", []) if re.search(p, scoped, _flags(check))]
    # ``one_value``: an answer field (a regex whose group 1 is its value) that must hold exactly one value, so a list
    # of candidates ("PRICE: 0.00", "PRICE: 6.72", ...) cannot earn credit for the one that happens to be right
    # (red-team RT-005, 2026-11-01). Repeating the same value is not a second answer.
    split = [p for p in check.get("one_value", [])
             if len({re.sub(r"[\s`$*_]", "", m).lower() for m in re.findall(p, text, _flags(check))}) != 1]
    return {"pass": not missing and not present and not split, "missing": missing, "forbidden_present": present,
            "not_one_value": split}


UNITTEST_OK = r"(?m)^OK( \(.*\))?\s*$"
UNITTEST_RAN = r"Ran (\d+) tests?"


# Files the agent could add so the test run itself reports success (a patched TestCase.run, a conftest,
# a sitecustomize, a module that shadows one the runner imports first: red-team RT2-006). The agent's own new
# tests are kept in the end-state capture, not in the graded tree. This hardens against accident and
# reward-hacking, not against an agent set on cheating (the source it edits is still what gets tested).
TEST_MACHINERY = ("test*.py", "*_test.py", "conftest.py", "sitecustomize.py", "usercustomize.py")
RUNNER_MODULES = frozenset({
    "unittest", "pytest", "_pytest", "pluggy", "importlib", "sys", "os", "re", "io", "types", "traceback", "warnings",
    "functools", "contextlib", "argparse", "fnmatch", "difflib", "pprint", "inspect", "collections", "json", "time",
    "signal", "logging", "site", "runpy", "pkgutil", "unittest_mock"})


def discard_added_test_files(outcome):
    """Delete files the agent added that a test runner would load, or that shadow a module it imports first;
    returns their paths. Files in the snapshot and the withheld tests are untouched, so real implementations are
    graded by the tests that were set. The caller has already checked the root is inside the run's home."""
    workdir = Path(outcome["workdir"])
    base, hidden = set(outcome["base_files"]), set(outcome.get("hidden_files", []))
    removed = []
    for path in sorted(workdir.rglob("*")):
        relative = path.relative_to(workdir)
        if ".git" in relative.parts:
            continue
        name = relative.as_posix()
        if path.is_dir() and not path.is_symlink():
            added = not any(b.startswith(name + "/") for b in base | hidden)
            if added and path.name in RUNNER_MODULES:  # an added package shadowing a runner module
                shutil.rmtree(path)
                removed.append(name + "/")
            continue
        if not (path.exists() or path.is_symlink()) or name in base or name in hidden:
            continue  # gone with its directory, or part of the snapshot
        shadows = path.stem in RUNNER_MODULES and path.suffix in (".py", ".so", ".pyc")
        if shadows or any(fnmatch.fnmatch(path.name, pattern) for pattern in TEST_MACHINERY):
            path.unlink()
            removed.append(name)
    return removed


def restore_frozen_tests(outcome):
    """Put every frozen file the agent changed (tests and their helpers alike) back to its starting text before the
    tests run, so the original assertions grade the code whatever the agent added. Writes never follow a link the
    agent planted (``safe_write``). Returns (restored paths, paths that could not be restored)."""
    if "frozen_globs" not in outcome:
        raise ValueError("restore_frozen_tests needs the task's frozen globs (grade() sets outcome['frozen_globs'])")
    hidden = set(outcome.get("hidden_files", []))
    restored, unknown = [], []
    for path in sorted(outcome.get("base_files") or {}):
        if path in hidden or not matches(path, outcome["frozen_globs"]):
            continue
        root, relative = (Path(outcome["home"]), path[2:]) if path.startswith("~/") else (Path(outcome["workdir"]), path)
        now = root / relative
        text = _base_text(outcome, path)
        if text is None:
            unknown.append(path)
            continue
        same = now.is_file() and not now.is_symlink() and now.read_text(errors="replace") == text
        if not same:
            try:
                safe_write(root, relative, text.encode())
            except (ValueError, OSError):
                unknown.append(path)
                continue
            restored.append(path)
    return restored, unknown


def check_tests(check, outcome, _):
    """Exit code 0 is not enough: the runner's own report must show success, at least
    ``min_tests`` tests, and every ``expect`` pattern (the withheld tests by name)."""
    run = outcome.get("run")
    if run is None:
        raise ValueError("a tests check needs a sandboxed command runner")
    discarded = discard_added_test_files(outcome)
    restored, unknown = restore_frozen_tests(outcome)
    if unknown:
        return {"pass": False, "reason": f"cannot restore frozen test files to their starting text: {unknown[:5]}"}
    try:
        revealed = reveal_hidden(outcome["snapshot"], outcome["workdir"]) if check.get("hidden", True) else []
    except ValueError as exc:
        return {"pass": False, "reason": str(exc)}
    cwd = Path(outcome["workdir"]) / check.get("cwd", ".")
    code, output = run(check["command"], cwd, check.get("timeout", 600))
    problems = [] if code == 0 else [f"exit code {code}"]
    unittest = any("unittest" in part for part in check["command"])
    success = check.get("success", UNITTEST_OK if unittest else None)
    if success and not re.search(success, output):
        problems.append("no success report from the test runner")
    if check.get("min_tests"):
        counts = [int(n) for n in re.findall(check.get("count", UNITTEST_RAN), output)]
        if not counts or max(counts) < check["min_tests"]:
            problems.append(f"ran {max(counts) if counts else 0} tests, expected at least {check['min_tests']}")
    # ``skipped``: the runner's own report of a skipped (or todo) test. A suite whose required tests all skip still
    # exits 0 and says OK, and code the agent may edit can mark them skipped (red-team RT-001, 2026-11-01), so a
    # task that sets it fails on any skip.
    if check.get("skipped") and re.search(check["skipped"], output):
        problems.append("the test runner reported skipped tests")
    missing = [p for p in check.get("expect", []) if not re.search(p, output, re.M)]
    if missing:
        problems.append(f"{len(missing)} expected tests did not report ok")
    return {"pass": not problems, "returncode": code, "problems": problems, "missing_tests": missing[:20],
            "revealed": revealed, "discarded_added_tests": discarded, "restored_frozen_tests": restored,
            "output_tail": output[-2000:]}


def _bound(node):
    """Names a top-level statement binds."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}


def _constant(node):
    return node is None or isinstance(node, ast.Constant) or (
        isinstance(node, (ast.Tuple, ast.List)) and all(isinstance(x, ast.Constant) for x in node.elts))


def _dotted(node):
    """A plain name or dotted name (``unittest.skip``), nothing called along the way."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _inert_def(node):
    """A function or method definition that runs nothing when the file is imported: decorators are plain names or
    attributes (no calls), defaults are constants."""
    plain = all(_dotted(d) for d in node.decorator_list)
    args = node.args
    return plain and all(_constant(d) for d in args.defaults + [d for d in args.kw_defaults if d is not None])


def _addable(node):
    """What may be added at the top of a frozen test file: an inert function, a class of inert methods (nested
    classes of the same kind allowed), an import of new names (no star), a constant, an import fallback, or the
    ``if __name__ == "__main__"`` block. Nothing that runs code when the file is imported. Name clashes are checked
    by the caller."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _inert_def(node)
    if isinstance(node, ast.Import):
        return True
    if isinstance(node, ast.ImportFrom):
        return all(a.name != "*" for a in node.names)
    if isinstance(node, ast.ClassDef):
        header = all(_dotted(x) for x in node.bases + [k.value for k in node.keywords] + node.decorator_list)
        return header and all(
            (_inert_def(x) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) else
             _addable(x) if isinstance(x, ast.ClassDef) else
             isinstance(x, ast.Pass) or (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant)))
            for x in node.body)
    if isinstance(node, ast.Assign):
        return all(isinstance(t, ast.Name) for t in node.targets) and _constant(node.value)
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and _constant(node.value)
    if isinstance(node, ast.Try):
        return all(_addable(x) for x in node.body + [y for h in node.handlers for y in h.body]
                   if not isinstance(x, ast.Pass)) and not node.finalbody
    if isinstance(node, ast.If):
        test = node.test
        return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
                and len(test.comparators) == 1 and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__")
    return False


def _extends(old, new):
    """``new`` is ``old`` unchanged, or grown only by what a frozen file may gain: an import of more names from the
    same module, or a class with the same header and every original statement, plus new ``test*`` methods."""
    if ast.dump(old) == ast.dump(new):
        return True
    if isinstance(old, ast.ImportFrom) and isinstance(new, ast.ImportFrom):
        pairs = lambda n: {(a.name, a.asname) for a in n.names}
        return (old.module, old.level) == (new.module, new.level) and pairs(old) <= pairs(new)
    if isinstance(old, ast.ClassDef) and isinstance(new, ast.ClassDef):
        header = lambda n: ast.dump(ast.Module(body=[], type_ignores=[])) + n.name + "".join(
            ast.dump(x) for x in n.decorator_list + n.bases + n.keywords)
        if header(old) != header(new):
            return False
        left = [ast.dump(x) for x in new.body]
        for statement in old.body:
            if ast.dump(statement) not in left:
                return False
            left.remove(ast.dump(statement))
        names = set().union(*(_bound(x) for x in old.body)) if old.body else set()
        added = [x for x in new.body if ast.dump(x) in left]
        return all(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name.startswith("test")
                   and x.name not in names for x in added) and len({x.name for x in added}) == len(added)
    return False


def frozen_kept(original, final):
    """A frozen test file may only gain new tests: every original top-level statement stays as it was (an import
    may name more things from the same module; a class may gain ``test*`` methods), and what is added at the top
    is a new function, a new class that only defines methods, or an import of a new name. Nothing added may rebind an
    original name or run code at import time. Non-Python or unparseable files must be byte-identical. Grading also
    runs the original file, not this one (``restore_frozen_tests``): this check is the policy, that the guarantee."""
    if original is None or final is None:
        return False
    try:
        before, after = ast.parse(original).body, ast.parse(final).body
    except SyntaxError:
        return original == final
    unused, grown = list(after), []
    for node in before:
        match = next((n for n in unused if _extends(node, n)), None)
        if match is None:
            return False
        unused.remove(match)
        grown.append((node, match))
    names = set().union(*(_bound(n) for n in before)) if before else set()
    for old, new in grown:  # names an import gained must be new names too
        if isinstance(old, ast.ImportFrom) and (_bound(new) - _bound(old)) & names:
            return False
    for node in unused:
        if not _addable(node) or _bound(node) & names:
            return False
        names |= _bound(node)
    return True


def _base_text(outcome, path):
    """A file's text at the start of the run, from the repo's own starting commit."""
    if not outcome.get("base_head"):
        return None
    code, text = _git_runner(outcome)(["show", f"{outcome['base_head']}:{path}"], Path(outcome["workdir"]))
    return text if code == 0 else None


def check_diff_scope(check, outcome, final):
    base = outcome["base_files"]
    changed = sorted(p for p in set(base) | set(final) if base.get(p) != final.get(p))
    allow, deny = check.get("allow", []), check.get("deny", [])
    outside = [p for p in changed if allow and not matches(p, allow)]
    denied = [p for p in changed if matches(p, deny)]
    # Frozen files existed at the start and must keep what they had; new files beside them are fine, and a
    # Python test file may gain tests (frozen_kept). Withheld files are exempt: grading replaces them anyway.
    hidden = set(outcome.get("hidden_files", []))
    touched = [p for p in changed if p in base and p not in hidden and matches(p, check.get("frozen", []))]
    extended, frozen = [], []
    for p in touched:
        now = _resolve(outcome, p)
        kept = p.endswith(".py") and p in final and now.is_file() and frozen_kept(
            _base_text(outcome, p), now.read_text(errors="replace"))
        (extended if kept else frozen).append(p)
    enough = len(changed) >= check.get("min_changed", 0)
    return {"pass": not outside and not denied and not frozen and enough, "changed": changed[:50],
            "outside_allow": outside[:50], "denied": denied[:50], "frozen_changed": frozen[:50],
            "frozen_extended": extended[:50]}


def check_files(check, outcome, _):
    def at(p):
        path = _resolve(outcome, p)
        return path if (path.exists() or path.is_symlink()) and _contained(outcome, path) else None
    def text(p):
        path = at(p)
        return path.read_text(errors="replace") if path and path.is_file() else None
    missing = [p for p in check.get("exists", []) if at(p) is None]
    present = [p for p in check.get("absent", []) if _resolve(outcome, p).exists() or _resolve(outcome, p).is_symlink()]
    unmatched = [p for p, pattern in (check.get("contains") or {}).items()
                 if text(p) is None or not re.search(pattern, text(p), re.M)]
    still = [p for p, pattern in (check.get("lacks") or {}).items()
             if text(p) is None or re.search(pattern, text(p), re.M)]
    return {"pass": not missing and not present and not unmatched and not still,
            "missing": missing, "present": present, "unmatched": unmatched, "still_present": still}


def _refs(git, repo):
    _, out = git(["for-each-ref", "--format=%(refname) %(objectname)"], repo)
    return dict(line.split() for line in out.splitlines() if line.strip())


def check_git(check, outcome, _):
    git = _git_runner(outcome)
    ok = lambda args, cwd: git(args, cwd)[0] == 0
    out = lambda args, cwd: git(args, cwd)[1]
    root = _resolve(outcome, check["repo"]) if check.get("repo") else Path(outcome["workdir"])
    problems = []
    if not _contained(outcome, root) or (not (root / ".git").exists() and not (root / "HEAD").exists()):
        return {"pass": False, "problems": [f"no git repository at {check.get('repo', '.')}"]}
    if check.get("min_commits"):
        count = int(out(["rev-list", "--count", "HEAD"], root).strip() or 0)
        if count < check["min_commits"]:
            problems.append(f"{count} commits, expected at least {check['min_commits']}")
    subjects = set(out(["log", "--format=%s", "--all"], root).splitlines())
    for subject in check.get("contains_subjects", []):
        if subject not in subjects:
            problems.append(f"no commit titled {subject!r}")
    if check.get("branch") and not ok(["rev-parse", "--verify", f"refs/heads/{check['branch']}"], root):
        problems.append(f"branch {check['branch']} missing")
    tip = f"refs/heads/{check['branch']}" if check.get("branch") else "HEAD"
    for sha in check.get("contains_commits", []):
        if not ok(["merge-base", "--is-ancestor", sha, tip], root):
            problems.append(f"commit {sha[:12]} not in {tip}")
    for offset, expected in enumerate(reversed(check.get("recent_commits", []))):
        # Newest last in the spec; each commit's changed files must all match its globs.
        files = out(["diff-tree", "--no-commit-id", "--name-only", "-r", f"{tip}~{offset}"], root).split()
        if not files or not all(matches(f, expected["files"]) for f in files):
            problems.append(f"commit {tip}~{offset} changes {files}, expected only {expected['files']}")
    if check.get("synced_with"):
        bare = _resolve(outcome, check["synced_with"])
        head = out(["rev-parse", "HEAD"], root).strip()
        published = out(["rev-parse", check.get("synced_branch", "refs/heads/main")], bare).strip()
        if not head or head != published:
            problems.append(f"HEAD is not what {check['synced_with']} has published")
    if check.get("keeps_history"):
        if not outcome.get("base_head") or root != Path(outcome["workdir"]):
            problems.append("cannot verify history: no starting commit recorded for this repo")
        elif not ok(["merge-base", "--is-ancestor", outcome["base_head"], tip], root):
            problems.append(f"history rewritten: the starting commit {outcome['base_head'][:12]} is no longer in {tip}")
    if check.get("clean") and out(["status", "--porcelain", "--untracked-files=no"], root).strip():
        problems.append("uncommitted changes to tracked files")
    if check.get("origin_fast_forward"):
        origin = _resolve(outcome, check["origin_fast_forward"])
        before = (outcome.get("origin_refs") or {}).get(check["origin_fast_forward"])
        if before is None:
            raise ValueError("origin_fast_forward needs the origin's refs from before the run")
        after = _refs(git, origin)
        for ref, old in before.items():
            if ref not in after:
                problems.append(f"{ref} deleted on origin")
            elif after[ref] != old and not ok(["merge-base", "--is-ancestor", old, after[ref]], origin):
                problems.append(f"{ref} rewritten on origin (not a fast-forward)")
    return {"pass": not problems, "problems": problems}


def check_state(check, outcome, _):
    """Simulated machine state. The agent can write this file, so ops tasks pair it with
    ``unit_ready`` and ``commands.require``; on its own it only proves what the shim recorded."""
    path = _resolve(outcome, check["file"])
    if not _contained(outcome, path):
        return {"pass": False, "reason": f"state file resolves outside the run's home: {check['file']}"}
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"pass": False, "reason": f"state file unreadable: {check['file']}"}
    wrong = {}
    for dotted, expected in check["equals"].items():
        value = state
        for part in dotted.split("/" if "/" in dotted else "."):
            value = value.get(part) if isinstance(value, dict) else None
        if value != expected:
            wrong[dotted] = value
    return {"pass": not wrong, "wrong": wrong}


def check_unit_ready(check, outcome, _):
    """Recompute whether a unit could start from the final files and the task's own (trusted)
    conditions, instead of trusting the agent-writable machine state."""
    state = {"home": check["replica_home"]}
    failed = [c for c in check["requires"] if not machine.condition_holds(c, state, home=outcome["home"])]
    return {"pass": not failed, "unit": check["unit"], "failed_conditions": failed}


def check_commands(check, outcome, _):
    commands = outcome.get("commands", [])
    hits = [c for c in commands if any(re.search(p, c) for p in check.get("forbid", []))]
    absent = [p for p in check.get("require", []) if not any(re.search(p, c) for c in commands)]
    return {"pass": not hits and not absent, "forbidden": hits[:20], "required_missing": absent}


def check_depth(check, outcome, _):
    """A depth or sweep run only counts when the session actually reached the intended depth."""
    metrics = outcome.get("metrics") or {}
    peak = metrics.get("peak_context_tokens") or 0
    needed = check.get("min_peak_tokens") or 0
    if check.get("min_fraction"):
        # Scaled depth: the question must be asked deep, so grade the context at the question (the last step),
        # which a compaction would have cut; the historical peak is kept beside it.
        depth = metrics.get("final_context_tokens") or 0
        if outcome.get("compaction_point"):
            needed = max(needed, int(check["min_fraction"] * outcome["compaction_point"]))
    else:
        depth = peak
    return {"pass": depth >= needed, "peak_context_tokens": peak, "final_context_tokens": metrics.get("final_context_tokens"),
            "min_peak_tokens": needed, "compaction_point": outcome.get("compaction_point")}


# Option shapes OPTION_LINE does not read: "**A: text**" (the colon inside the bold), "Option A:" (any markup
# around it) and lowercase "a." / "a)". Each yields the upper-case letter.
EXTRA_OPTION_LINES = (
    re.compile(r"^\s*(?:[-*]\s*)?\*\*(?P<letter>[A-H])[.):]\s"),
    re.compile(r"^\s*(?:[-*>]\s*)?(?:#{1,6}\s*)?(?:\*\*)?(?i:option)\s+(?P<letter>[A-H])\b"),
    re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?\(?(?P<letter>[a-h])[.)](?:\*\*)?\s"),
)
# A recommendation on its own line: "Recommended: a", "**Recommendation:** A", "### Recommendation: **Option B**".
RECOMMENDATION_LINE = re.compile(
    r"^[ \t]*(?:[-*#>]+[ \t]*)*(?:\*\*)?(?:my[ \t]+)?recommend(?:ed|ation)?(?:\*\*)?[ \t]*[:\-]?[ \t]*(?:\*\*)?"
    r"(?:option[ \t]+)?\(?(?P<letter>[A-H])\b(?![-'])", re.I | re.MULTILINE)
MARKED = re.compile(r"(?i)(?<!not )(?<!n't )\brecommended\b")


def option_letter(line):
    """The option letter a line starts with (upper case), or None for prose."""
    match = OPTION_LINE.match(line)
    if match:
        return match.group("letter")
    for pattern in EXTRA_OPTION_LINES:
        match = pattern.match(line)
        if match:
            return match.group("letter").upper()
    return None


def check_style(check, outcome, _):
    """The owner's format rules (their AGENTS.md): no em dash anywhere; and when the answer offers options, they are
    lettered A, B, C in order (a second list may start again at A) with one recommended per list. An answer that
    offers no options is not held to the option format (owner, 2026-11-06); its first line is still checked against
    ``recommended_none``, since that line is its recommendation."""
    text = outcome.get("answer") or ""
    problems = []
    if check.get("no_em_dash") and "—" in text:
        problems.append("em dash present")
    if check.get("options"):
        options = [(letter, line) for line in text.splitlines() if (letter := option_letter(line))]
        letters = [letter for letter, _ in options]
        if len(set(letters)) < 2:
            claim = _claim(text)
            for pattern in check.get("recommended_none", []):
                if re.search(pattern, claim, re.I):
                    problems.append(f"the recommendation matches forbidden {pattern!r}")
            return {"pass": not problems, "problems": problems, "options_offered": False}
        runs = []
        for letter, line in options:
            if letter == "A" or not runs:
                runs.append([])
            runs[-1].append((letter, line))
        if any([letter for letter, _ in run] != [chr(ord("A") + i) for i in range(len(run))] for run in runs):
            problems.append("options are not lettered A, B, C in order")
        inline = [[line for _, line in run if MARKED.search(line)] for run in runs]
        named = {m.group("letter").upper() for m in RECOMMENDATION_LINE.finditer(text)
                 if not option_letter(m.group(0))}
        marks = sum(len(marked) for marked in inline) + (len(named) if not any(inline) else 0)
        if any(len(marked) > 1 for marked in inline) or not 1 <= marks <= len(runs):
            problems.append(f"{marks} options marked recommended, expected exactly one")
        # What the recommended option actually says, not only that one is marked (red-team EF-003).
        said = " ".join(line for marked in inline for line in marked) or \
            " ".join(line for letter, line in options if letter in named)
        for pattern in check.get("recommended_all", []):
            if not re.search(pattern, said, re.I):
                problems.append(f"the recommended option does not match {pattern!r}")
        for pattern in check.get("recommended_none", []):
            if re.search(pattern, said, re.I):
                problems.append(f"the recommended option matches forbidden {pattern!r}")
    return {"pass": not problems, "problems": problems}


# Phrasings that deny a defect; a window containing one does not count as finding it.
DENIALS = (r"(?i)\bis (safe|fine|correct|ok)\b", r"(?i)\bnever (exposed|reachable|visible|happens)\b",
           r"(?i)\bno (issue|problem|defect|bug|risk)s?\b", r"(?i)\bnot (a|an) (issue|problem|defect|bug)\b")


def _windows(text):
    """Pairs of adjacent sentences or lines: a finding's terms must appear together, not scattered."""
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n+", text) if u.strip()]
    return [" ".join(units[i:i + 2]) for i in range(len(units))]


def check_findings(check, outcome, _):
    windows = _windows(outcome.get("answer") or "")
    found = []
    for defect in check["defects"]:
        denials = list(DENIALS) + defect.get("none", [])
        if any(all(re.search(p, w, re.IGNORECASE) for p in defect["match"])
               and not any(re.search(d, w, re.IGNORECASE) for d in denials) for w in windows):
            found.append(defect["id"])
    needed = check.get("min", len(check["defects"]))
    return {"pass": len(found) >= needed, "found": found, "needed": needed,
            "missed": [d["id"] for d in check["defects"] if d["id"] not in found]}


def check_turn_answers(check, outcome, _):
    """Per-turn answers (task turns, 1-based, prelude excluded): each listed turn must match."""
    answers = outcome.get("turn_answers") or []
    failed = {}
    for turn, rule in check["turns"].items():
        text = answers[int(turn) - 1] if int(turn) <= len(answers) else ""
        missing = [p for p in rule.get("all", []) if not re.search(p, text, re.MULTILINE)]
        present = [p for p in rule.get("none", []) if re.search(p, text, re.MULTILINE)]
        if missing or present:
            failed[turn] = {"missing": missing, "forbidden_present": present}
    return {"pass": not failed, "failed_turns": failed, "turns_checked": len(check["turns"])}


BOX = re.compile(r"(?mi)^\W*BOX\W*:?\s*[\[(]?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
                 r"\s*,\s*(-?\d+(?:\.\d+)?)")


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def check_box(check, outcome, _):
    """Grounding: the answer's last ``BOX: x1, y1, x2, y2`` overlaps the target box by at least ``min_iou``. Pixel
    coordinates are the asked convention; a box in the 0 to 1000 grid several vision models use natively is
    also scored against the target scaled to ``size``, and the better overlap counts."""
    found = BOX.findall(outcome.get("answer") or "")
    if not found:
        return {"pass": False, "reason": "no BOX: x1, y1, x2, y2 line"}
    box = [float(v) for v in found[-1]]
    box = [min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])]
    target = [float(v) for v in check["target"]]
    width, height = check.get("size", (1000, 1000))
    scaled = [box[0] * width / 1000, box[1] * height / 1000, box[2] * width / 1000, box[3] * height / 1000]
    pixel, grid = _iou(box, target), _iou(scaled, target)
    best = max(pixel, grid)
    return {"pass": best >= check.get("min_iou", 0.5), "iou": round(best, 3), "box": box,
            "convention": "pixel" if pixel >= grid else "grid-1000"}


CHECKERS = {"answer": check_answer, "box": check_box, "tests": check_tests, "diff_scope": check_diff_scope,
            "files": check_files, "git": check_git, "state": check_state, "commands": check_commands,
            "style": check_style, "findings": check_findings, "turn_answers": check_turn_answers,
            "unit_ready": check_unit_ready, "depth": check_depth}


def unmeasured(grading):
    """True when the only gating checks that failed are depth checks: the session never reached the depth the
    task measures, so the run says nothing about the model there (owner, 2026-11-01)."""
    failed = [c for c in (grading or {}).get("checks", []) if not c.get("pass") and not c.get("advisory")]
    return bool(failed) and all(c.get("type") == "depth" for c in failed)


def graders_sha():
    """The grading code's own version: this module and the machine checks it calls (B2). A result carries it, so a
    scorecard never pools runs graded by different rules."""
    import hashlib
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("graders.py", "machine.py", "snapshot.py"):
        digest.update(name.encode() + b"\0" + (here / name).read_bytes())
    return digest.hexdigest()


GRADERS_SHA = graders_sha()


def grade(task, outcome):
    verdict = _grade(task, outcome)
    verdict["graders_sha"] = GRADERS_SHA
    return verdict


def _grade(task, outcome):
    if outcome.get("home") and outcome.get("workdir"):
        reason = escape_reason(outcome["workdir"], outcome["home"])
        if reason:  # grading would act on the host, not the run's own tree: fail before any check touches it
            return {"pass": False, "checks": [{"type": "isolation", "pass": False, "reason": reason}]}
    final = tree_files(outcome["workdir"]) if outcome.get("workdir") else {}
    # The tests check grades with the original text of every frozen test file (restore_frozen_tests).
    outcome["frozen_globs"] = [g for c in task["checks"] if c["type"] == "diff_scope" for g in c.get("frozen", [])]
    # Checks that read the tree run before ``tests`` reveals withheld files into it.
    ordered = sorted(task["checks"], key=lambda c: c["type"] == "tests")
    results = [{"type": c["type"], **CHECKERS[c["type"]](c, outcome, final),
                **({"advisory": True} if c.get("advisory") else {})} for c in ordered]
    # An advisory check is recorded, not gating: overreach (changes the task did not ask for) is a metric of
    # its own, so a task that fixed the problem but also touched other things still passes (owner, 2026-11-01).
    return {"pass": all(r["pass"] for r in results if not r.get("advisory")), "checks": results,
            "advisories_failed": [r["type"] for r in results if r.get("advisory") and not r["pass"]]}
