"""Raw capture for one task-run, kept under ``<task-run dir>/raw/``.

Principle (owner, 2026-10-03): collect widely up front, prune sources nobody uses later. A live run
cannot be replayed, so everything the harness saw is saved untruncated beside the graded result.
Nothing here is read by grading or by the metrics warehouse, and nothing here can fail a task-run:
every writer swallows its own errors and notes them in ``raw/capture-errors.json``.

Files: ``turns.json`` (every turn's full text, command and flags), ``turn-NN.events.jsonl`` (opencode's
JSON stream), ``turn-NN.stderr.log`` (its logs), ``commands.jsonl`` (every test or grader command with
full output), ``end-state/`` (git status, diff, log, file manifest, small untracked files),
``opencode-data/`` (opencode's own logs and database from the fake home).
"""
import hashlib
import json
from pathlib import Path
import shutil
import time

from .snapshot import escape_reason

UNTRACKED_FILE_LIMIT = 256 * 1024
MANIFEST_FILE_LIMIT = 8 * 1024 * 1024
OPENCODE_DATA_LIMIT = 200 * 1024 * 1024
OPENCODE_DATA = (".local/share/opencode", ".local/state/opencode", ".cache/opencode/log")


def _inside(path, root):
    """True when ``path`` really lives under ``root`` (no symlink in between leads elsewhere)."""
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except OSError:
        return False


class RawCapture:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.errors = []

    def _safe(self, label, action):
        try:
            return action()
        except Exception as exc:  # capture never fails a task-run
            self.errors.append({"step": label, "error": f"{type(exc).__name__}: {exc}"[:300]})
            return None

    def _write(self, name, text):
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def turns(self, records):
        """``records``: the driver's per-turn dicts (``events`` list, ``stderr`` text, anything else JSON-safe)."""
        def save():
            summary = []
            for number, record in enumerate(records, 1):
                events, stderr = record.get("events") or [], record.get("stderr") or ""
                self._write(f"turn-{number:02d}.events.jsonl", "".join(json.dumps(e) + "\n" for e in events))
                self._write(f"turn-{number:02d}.stderr.log", stderr)
                summary.append({k: v for k, v in record.items() if k not in ("events", "stderr")}
                               | {"events": len(events), "stderr_bytes": len(stderr)})
            self._write("turns.json", json.dumps(summary, indent=1, default=str))
        self._safe("turns", save)

    def recording(self, run):
        """Wrap a ``run(argv, cwd, timeout) -> (code, output)`` so every call lands in commands.jsonl in full."""
        def recorded(command, cwd, timeout):
            started = time.time()
            code, output = run(command, cwd, timeout)
            def save():
                self.dir.mkdir(parents=True, exist_ok=True)
                with (self.dir / "commands.jsonl").open("a") as stream:
                    stream.write(json.dumps({"t": started, "seconds": round(time.time() - started, 3),
                                             "command": command, "cwd": str(cwd), "code": code,
                                             "output": output}, default=str) + "\n")
            self._safe("commands", save)
            return code, output
        return recorded

    def end_state(self, box, repo, home, base_head):
        """Workspace as the agent left it, taken before grading touches anything."""
        out = self.dir / "end-state"
        def git(name, args):
            code, text = box.git(args, repo)
            self._write(f"end-state/{name}.txt", f"[exit {code}]\n{text}")
            return text
        self._safe("git-status", lambda: git("status", ["status", "--porcelain=v1", "-uall", "--branch"]))
        self._safe("git-diff", lambda: git("diff", ["diff", "--binary", base_head] if base_head else ["diff", "--binary"]))
        self._safe("git-log", lambda: git("log", ["log", "--all", "--decorate", "--stat", "-n", "30"]))
        self._safe("git-branches", lambda: git("branches", ["branch", "-a", "-vv"]))
        untracked = self._safe("git-untracked", lambda: git("untracked", ["ls-files", "--others", "--exclude-standard"]))
        escaped = escape_reason(repo, home)
        if escaped:  # never walk a root the agent pointed outside its home
            self.errors.append({"step": "end-state-files", "error": f"skipped: {escaped}"})
            return
        self._safe("manifest", lambda: self._manifest(repo, out / "files.json"))
        for relative in (untracked or "").splitlines():
            source = Path(repo) / relative
            def keep(source=source, relative=relative):
                if source.is_file() and not source.is_symlink() and source.stat().st_size <= UNTRACKED_FILE_LIMIT:
                    target = out / "untracked" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            self._safe(f"untracked:{relative}", keep)

    @staticmethod
    def _manifest(repo, path):
        rows = {}
        for item in sorted(Path(repo).rglob("*")):
            relative = item.relative_to(repo)
            if ".git" in relative.parts or not item.is_file() or item.is_symlink():
                continue
            size = item.stat().st_size
            rows[str(relative)] = {"size": size, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()
                                   if size <= MANIFEST_FILE_LIMIT else None}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1))

    def opencode_data(self, home):
        """opencode's own log files and session database from inside the box, size-capped."""
        copied = 0
        home = Path(home)
        for relative in OPENCODE_DATA:
            source = home / relative
            # The agent controls this tree: a symlinked root or ancestor would resolve on the host, outside the box.
            if not source.is_dir() or not _inside(source, home):
                continue
            for item in sorted(source.rglob("*")):
                if not item.is_file() or item.is_symlink() or not _inside(item, home):
                    continue
                size = item.stat().st_size
                if copied + size > OPENCODE_DATA_LIMIT:
                    self.errors.append({"step": "opencode-data", "error": f"size cap reached before {item.name}"})
                    return
                def keep(item=item):
                    target = self.dir / "opencode-data" / item.relative_to(home)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(item, target)
                self._safe(f"opencode-data:{item.name}", keep)
                copied += size

    def finish(self):
        if self.errors:
            self._write("capture-errors.json", json.dumps(self.errors, indent=1))
