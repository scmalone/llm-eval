"""Find bug-fix commits that could become workload tasks, the SWE-bench way: a commit that fixed something and
added or changed a test. The parent is the task's starting point, the commit message is the issue, and the
commit's own tests are the hidden tests.

Read-only: only ``git log`` and ``git show --numstat`` on each repo. Usage:

    python3 tools/mine_fixes.py ~/Workspace/meal-planner ~/Workspace/flight-test ... > candidates.json

A candidate still has to be proven before it is a task (``tools/prove_fix.py``): its tests fail at the parent
and pass at the commit, in a clean checkout.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

TEST = re.compile(r"(^|/)(tests?|__tests__|spec|Tests|tests_integration)/|(^|/)test_[^/]+\.py$|_test\.(py|go)$|"
                  r"\.(test|spec)\.(m?[jt]sx?|cjs)$|Tests?\.swift$|(^|/)test-[^/]+\.(m?[jt]s|py|sh)$")
FIX = re.compile(r"\b(fix(e[sd])?|bug|regress\w*|broken|crash\w*|wrong|incorrect|off[- ]by|leak\w*|race|hang\w*|"
                 r"guard|reject\w*|clamp\w*|handle[sd]?|repair\w*|restore[sd]?)\b", re.I)
NOISE = re.compile(r"\.(md|json|lock|png|jpe?g|webp|glb|gltf|bin|log|pdf|wav|csv|txt|svg|html|css)$|(^|/)package-lock|"
                   r"(^|/)(docs|evidence|archive|artifacts|research)/", re.I)
SOURCE_LINES = (3, 250)  # a task-sized source change: more than a typo, less than a feature
SOURCE_FILES = 6


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, errors="replace").stdout


def commits(repo):
    """(sha, date, subject, body) for every non-merge commit on the current branch."""
    out = git(repo, "log", "--no-merges", "--format=%x1e%H%x1f%cs%x1f%s%x1f%b")
    for record in out.split("\x1e")[1:]:
        sha, date, subject, body = (record.split("\x1f") + ["", "", "", ""])[:4]
        yield sha, date, subject.strip(), body.strip()


def numstat(repo, sha):
    rows = []
    for line in git(repo, "show", "--numstat", "--format=", sha).splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            added, removed, path = parts
            rows.append((path, int(added) if added.isdigit() else 0, int(removed) if removed.isdigit() else 0))
    return rows


def candidate(repo, sha, date, subject, body):
    rows = numstat(repo, sha)
    tests = [r for r in rows if TEST.search(r[0])]
    source = [r for r in rows if not TEST.search(r[0]) and not NOISE.search(r[0])]
    if not tests or not source:
        return None
    lines = sum(a + d for _, a, d in source)
    if not (SOURCE_LINES[0] <= lines <= SOURCE_LINES[1]) or len(source) > SOURCE_FILES:
        return None
    message = f"{subject}\n{body}"
    fix = bool(FIX.search(message))
    # Score: a stated fix, a small focused change, a test written for it, and a message long enough to be an issue.
    score = (3 if fix else 0) + (2 if len(source) <= 2 else 1) + (2 if 10 <= lines <= 120 else 0)
    score += 1 if any(a > 0 and d == 0 for _, a, d in tests) else 0  # a test added, not only edited
    score += 1 if len(message) > 120 else 0
    return {"repo": str(repo), "sha": sha, "date": date, "subject": subject, "fix": fix, "score": score,
            "source": [p for p, _, _ in source], "tests": [p for p, _, _ in tests], "source_lines": lines,
            "message_chars": len(message)}


def mine(repo):
    found = [c for c in (candidate(repo, *row) for row in commits(repo)) if c]
    return sorted(found, key=lambda c: (-c["score"], c["date"]), reverse=False)


def main(argv):
    out = {}
    for path in argv:
        repo = Path(path).expanduser().resolve()
        if not (repo / ".git").exists():
            print(f"skip {repo}: not a repo root", file=sys.stderr)
            continue
        out[repo.name] = mine(repo)
        print(f"{repo.name}: {len(out[repo.name])} candidates", file=sys.stderr)
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
