"""Atomic snapshots, durable journal, exclusive locks, and committed sections."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}", value):
        raise ValueError("invalid identifier")
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def machine_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another evaluation owns the machine lock") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Journal:
    def __init__(self, directory):
        self.root = Path(directory)
        self._lock = threading.RLock()  # the sleep-lock watcher writes events from its own thread
        self.sampler = None  # a telemetry.GpuSampler while a campaign runs
        self.extra_samplers = []  # more samplers (host load and memory) tagged together with it
        self.server_probe = None  # callable(tag, phase) that saves a llama-server snapshot, or None
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "state.json"
        self.state = json.loads(path.read_text()) if path.exists() else {
            "schema_version": 1, "status": "planned", "restoration_required": False,
            "sections": {}, "sequence": 0}
        self._reconcile()

    def _reconcile(self):
        """A section file is the commit record; rebuild missing journal projections."""
        logged = set()
        events = self.root / "events.jsonl"
        if events.exists():
            raw = events.read_bytes()
            valid_end = 0
            for line in raw.splitlines(keepends=True):
                try:
                    item = json.loads(line)
                except ValueError:
                    break
                self.state["sequence"] = max(self.state["sequence"], item["sequence"])
                if item["event"] == "section_complete":
                    logged.add(item["section"])
                valid_end += len(line)
            if valid_end != len(raw):
                # Preserve the torn suffix for diagnosis before truncating the journal.
                (self.root / "events.torn").write_bytes(raw[valid_end:])
                with events.open("r+b") as stream:
                    stream.truncate(valid_end)
                    stream.flush()
                    os.fsync(stream.fileno())
        for path in sorted((self.root / "sections").glob("*.json")):
            result = json.loads(path.read_text())
            key = result["section"]
            self.state["sections"][key] = result
            if key not in logged:
                self.event("section_complete", section=key, recovered=True)
        self.save()

    def save(self):
        with self._lock:
            atomic_json(self.root / "state.json", self.state)

    def event(self, event, **values):
        with self._lock:
            self.state["sequence"] += 1
            entry = {"sequence": self.state["sequence"], "time": time.time(), "event": event, **values}
            with (self.root / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def section(self, key, status, **values):
        safe_id(key)
        if key in self.state["sections"]:
            return self.state["sections"][key]
        result = {"section": key, "status": status, "time": time.time(), **values}
        atomic_json(self.root / "sections" / f"{key}.json", result)
        self.event("section_complete", section=key, status=status)
        self.state["sections"][key] = result
        self.save()
        return result
