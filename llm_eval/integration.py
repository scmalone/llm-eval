"""Stamp that the integration tier (real opencode, bwrap and Unix sockets against a fake LLM) passed on
this exact source. The first live workload run is refused without a current stamp (``prepare --next``).

The tier cannot run inside the agent's own sandbox, so ``python3 -m llm_eval.acceptance integration`` is
excluded from it like the other runner commands. The stamp is a content hash of every Python file under
``llm_eval/`` and ``tests_integration/``: any edit makes it stale, a commit that changes nothing does not.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
STAMP = Path.home() / ".cache" / "llm-eval" / "integration-pass.json"
TREES = ("llm_eval", "tests_integration")


def source_sha(project=PROJECT):
    digest = hashlib.sha256()
    for tree in TREES:
        for path in sorted((Path(project) / tree).rglob("*.py")):
            digest.update(str(path.relative_to(project)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def blocker(stamp=STAMP, project=PROJECT):
    """Why the first live run must wait, or None when the stamp matches the current source."""
    try:
        recorded = json.loads(Path(stamp).read_text())
    except (OSError, ValueError):
        return "the integration tier has not passed on this machine"
    if recorded.get("source_sha") != source_sha(project):
        return f"the source changed since the integration tier passed ({recorded.get('at')})"
    return None


def run_and_stamp(stamp=STAMP, project=PROJECT, command=None):
    """Run the tier; write the stamp only when every test passed. Returns (result dict, exit code)."""
    command = command or [sys.executable, "-m", "unittest", "discover", "-s", "tests_integration", "-v"]
    out = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=1800)
    tail = (out.stdout + out.stderr)[-3000:]
    ran = re.search(r"Ran (\d+) tests?", tail)
    passed = out.returncode == 0 and bool(ran) and bool(re.search(r"^OK", tail, re.M))
    result = {"passed": passed, "ran": int(ran[1]) if ran else None, "output_tail": tail[-1500:]}
    if passed:
        Path(stamp).parent.mkdir(parents=True, exist_ok=True)
        Path(stamp).write_text(json.dumps({"source_sha": source_sha(project), "ran": result["ran"],
                                           "at": time.strftime("%Y-%m-%d %H:%M:%S")}))
        result["next"] = "Stamped. `prepare --next` will accept the first live run."
    else:
        result["next"] = "The integration tier failed. Report the output to the owner; do not start a live run."
    return result, 0 if passed else 1
