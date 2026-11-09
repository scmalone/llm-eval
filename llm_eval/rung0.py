"""Rung 0: the pre-pilot gate, enforced by ``prepare --next`` before the first live rung.

Two machine-checkable conditions, no GPU and no agent loop:

(a) Grading isolation: a repo with a planted ``core.fsmonitor`` is graded with nothing executing
    on the host. The sentinel that hook would touch must stay absent.
(b) Output caps: every classic fixture's ``max_output_tokens`` sits between a per-kind floor and
    the agent-turn output budget, and its estimated generation time fits the acceptance request
    cap. The acceptance caps, the hang cap and the worst-case downtime are printed.

``python3 -m llm_eval.acceptance rung0`` runs it standalone; ``prepare --next`` refuses with
``blocked_rung0`` while it fails. Exit code 0 passes, 2 refuses (matches the acceptance module's
"refused before touching anything").
"""
import json
from pathlib import Path
import shutil
import tempfile

from .adapters import FakeBox
from .workload import snapshot
from .workload.graders import grade

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OK, REFUSED = 0, 2
# The output budget is the agent-turn reservation, not an answer size (2026-10-03 decision).
OUTPUT_BUDGET_TOKENS = 16384
# A fixture's output cap must not sit below these. From the measured peaks in RUNS.md with
# headroom: smoke 96, tools 135, recall 1,189 tokens.
OUTPUT_FLOOR_TOKENS = {"smoke": 512, "vision": 1024, "tools": 1024, "recall": 4096}
DECODE_TOK_S = 45  # Muse Q4 decode, the assumption behind the printed time estimate
RESTORE_SECONDS = 120  # the restore wait in LiveAdapter
GATE = "rung0"  # sentinel: prepare --next enforces the gate only when it is passed


def _commit(repo, files, message):
    for name, text in files.items():
        path = Path(repo) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    snapshot.git(["add", "-A"], repo, isolated=True)
    snapshot.git(["commit", "-q", "--no-verify", "-m", message], repo, isolated=True)
    return snapshot.git(["rev-parse", "HEAD"], repo).strip()


def check_grading_isolation(root):
    """Grade a repo whose ``core.fsmonitor`` would touch a sentinel; it must stay absent.

    Uses the fake-backend git seam, so it needs no bwrap and no GPU. The sentinel is the
    observable: if any git the grader runs on the host lacks the safe flags, the hook fires.
    """
    root = Path(root)
    repo = root / "src"
    repo.mkdir(parents=True)
    snapshot.git(["init", "-q", "-b", "main"], repo, isolated=True)
    commit = _commit(repo, {"app.py": "x = 1\n"}, "base")
    sentinel = root / "fsmonitor-ran"
    task = {"id": "rung0-isolation", "kind": "workload", "category": "implementation",
            "source": {"repo": str(repo), "commit": commit},
            "checks": [{"type": "git", "clean": True}], "limits": {"turns": 1, "wall_seconds": 60}}
    manifest = snapshot.build(task, root / "snap")
    run = snapshot.materialize(root / "snap", root / "home/repo")
    snapshot.git(["config", "core.fsmonitor", f"touch {sentinel}; false"], run, isolated=True)
    snapshot.git(["config", "core.hooksPath", str(root / "hooks")], run, isolated=True)
    outcome = {"answer": "", "workdir": run, "home": root / "home", "base_files": manifest["files"],
               "snapshot": root / "snap", "commands": [], "git": FakeBox().git}
    graded = grade(task, outcome)["pass"]
    ran = sentinel.exists()
    return {"passed": not ran, "graded": graded, "sentinel": str(sentinel),
            "detail": "a planted core.fsmonitor executed on the host during grading" if ran
                      else "the planted fsmonitor did not run on the host"}


def check_output_caps(limits, fixtures_dir=FIXTURES):
    """Every fixture's output cap within [floor, budget], and its time estimate inside the request cap."""
    rows, problems = [], []
    for path in sorted(Path(fixtures_dir).glob("*.json")):
        fixture = json.loads(path.read_text())
        cap = fixture.get("max_output_tokens")
        if cap is None:
            continue
        kind = fixture.get("kind")
        floor = OUTPUT_FLOOR_TOKENS.get(kind)
        seconds = round(cap / DECODE_TOK_S, 1)
        rows.append({"fixture": path.stem, "kind": kind, "max_output_tokens": cap, "floor": floor,
                     "budget": OUTPUT_BUDGET_TOKENS, "estimated_seconds": seconds})
        if floor is None:
            problems.append(f"{path.stem}: no output floor is defined for kind {kind!r}")
        elif cap < floor:
            problems.append(f"{path.stem}: output cap {cap} sits below its {kind} floor {floor}")
        if cap > OUTPUT_BUDGET_TOKENS:
            problems.append(f"{path.stem}: output cap {cap} sits above the {OUTPUT_BUDGET_TOKENS}-token budget")
        if seconds > limits["request_seconds"]:
            problems.append(f"{path.stem}: estimated {seconds}s exceeds the {limits['request_seconds']}s request cap")
    return {"passed": not problems, "problems": problems, "fixtures": rows}


def run(limits, root=None, fixtures_dir=FIXTURES):
    """Run both conditions and return (report, exit code). No GPU, no agent loop."""
    temporary = Path(root) if root else Path(tempfile.mkdtemp(prefix="llm-eval-rung0-"))
    try:
        isolation = check_grading_isolation(temporary)
        caps = check_output_caps(limits, fixtures_dir)
    finally:
        if root is None:
            shutil.rmtree(temporary, ignore_errors=True)
    passed = isolation["passed"] and caps["passed"]
    report = {
        "passed": passed,
        "conditions": {"grading_isolation": isolation, "output_caps": caps},
        "limits_seconds": limits,
        "hang_cap_seconds": limits["model_seconds"],
        "worst_case_down_seconds": limits["load_seconds"] + limits["campaign_seconds"] + RESTORE_SECONDS,
        "next": "Rung 0 passed; the pilot can be prepared." if passed else
                "STOP: rung 0 failed. Fix every problem above before any live run.",
    }
    return report, (OK if passed else REFUSED)
