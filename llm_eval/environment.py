"""What a run ran on, written once to ``environment.json`` beside the journal.

Statistics across runs are only honest if each result says which harness, server build, driver and
kernel produced it. Capture is best effort and read-only: a field that cannot be read is None and
never fails a run. The host part (GPU, llama-server version) comes from the adapter's host.
"""
import platform
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args):
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=10)
    if out.returncode:
        raise RuntimeError(out.stderr.strip())
    return out.stdout.strip()


# Untracked files here change what a run does; untracked notes elsewhere in the folder do not.
BEHAVIOUR_DIRS = ("llm_eval", "tools", "fixtures", "suites", "configs")


def dirty_paths():
    """Uncommitted changes that make a run irreproducible from its SHA: any change to a tracked file, and any
    untracked file under the code, task or config folders."""
    tracked = _git("diff", "--name-only", "--relative", "HEAD", "--", ".").splitlines()
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *BEHAVIOUR_DIRS).splitlines()
    return sorted(set(tracked + untracked))


def harness_version():
    """The llm-eval commit and whether its folder has uncommitted changes (a dirty run is not
    reproducible from the SHA alone)."""
    try:
        return {"sha": _git("rev-parse", "HEAD"), "dirty": bool(dirty_paths())}
    except Exception:
        return {"sha": None, "dirty": None}


def capture(adapter):
    environment = {"harness": harness_version(), "kernel": platform.release(), "python": platform.python_version()}
    host = getattr(adapter, "environment", None)
    try:
        environment.update(host() if callable(host) else {})
    except Exception as exc:
        environment["host_error"] = str(exc)[:200]
    return environment
