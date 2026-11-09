"""Re-grade saved workload results with the 2026-11-01 grader fixes, read-only.

Usage: python3 tools/regrade_saved.py <harness dir> <run workload dir>...
Model code is never run: a ``tests`` check keeps its saved result. Checks that read the answer use the full
answer from export.json; diff_scope re-tests only the frozen files it flagged; git keeps_history and the
opencode-port advisories are evaluated on the saved repo.
"""
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from llm_eval.workload import graders  # noqa: E402

FIX = Path(sys.argv[1]) / "fixtures" / "workload"


def git(args, cwd):
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return done.returncode, done.stdout


def full_answer(run):
    try:
        export = json.loads((run / "export.json").read_text())
    except (OSError, ValueError):
        return None
    answer = ""
    for entry in export.get("entries", []):
        message = entry.get("message") or {}
        if message.get("role") == "assistant":
            for block in message.get("content") or []:
                if block.get("type") == "text" and (block.get("text") or "").strip():
                    answer = block["text"]
    return answer.strip()


def task_for(result):
    path = FIX / f"{result['task']}.json"
    return json.loads(path.read_text()) if path.exists() else None


def regrade(run):
    result = json.loads((run / "result.json").read_text())
    task = task_for(result)
    old = result["grading"]["checks"]
    if task is None:  # cohort document tasks: rebuild only their answer check from the saved one
        task = {"checks": []}
    answer = full_answer(run) or result.get("answer") or ""
    repo, home = Path(result["repo"]), Path(result["home"])
    outcome = {"answer": answer, "workdir": repo, "home": home, "base_head": result["base_head"], "git": git,
               "metrics": result.get("metrics") or {}, "compaction_point": (result.get("depth") or {}).get("compaction_point")}
    new_checks = []
    spec_by_type = {}
    for c in task["checks"]:
        spec_by_type.setdefault(c["type"], []).append(c)
    used = {}
    for saved in old:
        kind = saved["type"]
        index = used.get(kind, 0)
        used[kind] = index + 1
        specs = [c for c in spec_by_type.get(kind, []) if not c.get("advisory")]
        spec = specs[index] if index < len(specs) else None
        now = dict(saved)
        if kind in ("answer", "style", "findings") and spec:
            now.update(graders.CHECKERS[kind](spec, outcome, {}))
        elif kind == "answer" and saved.get("forbidden_present"):  # a cohort task: same check with the claim scope
            claim = graders._claim(answer)
            still = [p for p in saved["forbidden_present"] if re.search(p, claim, re.M)]
            now.update({"forbidden_present": still, "pass": not still and not saved.get("missing")})
        elif kind == "commands" and spec:
            hits = [c for c in saved.get("forbidden", []) if any(re.search(p, c) for p in spec.get("forbid", []))]
            now.update({"forbidden": hits, "pass": not hits and not saved.get("required_missing")})
        elif kind == "diff_scope" and saved.get("frozen_changed"):
            kept = [p for p in saved["frozen_changed"] if p.endswith(".py") and (repo / p).is_file() and graders.frozen_kept(
                git(["show", f"{result['base_head']}:{p}"], repo)[1], (repo / p).read_text(errors="replace"))]
            frozen = [p for p in saved["frozen_changed"] if p not in kept]
            now.update({"frozen_changed": frozen, "frozen_extended": kept,
                        "pass": not frozen and not saved.get("outside_allow") and not saved.get("denied")})
        elif kind == "git" and spec and spec.get("keeps_history"):
            now.update(graders.check_git(spec, outcome, {}))
        elif kind == "unit_ready" and spec:
            now.update(graders.check_unit_ready(spec, outcome, {}))
        elif kind == "depth" and spec:
            now.update(graders.check_depth(spec, outcome, {}))
        new_checks.append(now)
    for spec in task["checks"]:  # advisories added since the run
        if spec.get("advisory"):
            if spec["type"] == "files":
                present = [p for p in spec.get("absent", []) if (home / p[2:]).exists()]
                new_checks.append({"type": "files", "advisory": True, "pass": not present, "present": present})
            elif spec["type"] == "diff_scope":
                changed = set(git(["diff", "--name-only", result["base_head"]], repo)[1].split())
                changed |= set(git(["ls-files", "--others", "--exclude-standard"], repo)[1].split())
                commits = int(git(["rev-list", "--count", f"{result['base_head']}..HEAD"], repo)[1].strip() or 0)
                new_checks.append({"type": "diff_scope", "advisory": True, "pass": not changed,
                                   "changed": sorted(changed), "new_commits": commits})
    grading = {"checks": new_checks}
    passed = all(c["pass"] for c in new_checks if not c.get("advisory"))
    status = "pass" if passed else ("unmeasured" if graders.unmeasured(grading) else "fail")
    if (result.get("metrics") or {}).get("length_finishes") and not passed:
        status = "fail (output cap)"
    flipped = [c["type"] for c, s in zip(new_checks, old + [{}] * 9) if c.get("pass") != s.get("pass") and not c.get("advisory")]
    adv = [{k: v for k, v in c.items() if k in ("type", "present", "changed", "new_commits")} for c in new_checks
           if c.get("advisory") and not c["pass"]]
    return result["task"], result["pass"], status, flipped, adv


for root in sys.argv[2:]:
    print(f"## {root}")
    for path in sorted(glob.glob(f"{root}/*/result.json")):
        task, before, after, flipped, adv = regrade(Path(path).parent)
        mark = "" if (after == "pass") == before and not adv else "  <--"
        print(f"{task:45s} {('pass' if before else 'fail'):5s} -> {after:12s} {flipped or ''} {adv or ''}{mark}")
