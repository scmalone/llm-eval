"""Harness drivers: one live agent per trial, inside the sandbox, over a prepared fake home.

A driver is a context manager and answers four calls:

* ``turn(text, workdir, timeout) -> {"answer", "timed_out", "returncode", "events"}``
* ``export() -> dict`` (the harness's own session record, kept raw)
* ``summarize() -> dict`` (the metrics every driver must produce, in ``metrics.summarize``'s shape)
* ``context_tokens() -> int | None``

``OpencodeDriver`` drives ``opencode run``; ``pi_driver.PiDriver`` drives ``pi --print --mode json``.
The metrics contract is the seam: a harness that can report those numbers can be compared with another,
whatever its own session format looks like.
"""
import contextlib
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid

from . import opencode_config
from .relay import HostRelay
from .sandbox import Box, relay_dir

OPENCODE_HOME = Path.home() / ".opencode"
CONFIG_ASSETS = ("AGENTS.md", "agents", "commands")
LINGER_SECONDS = 20
EXIT_GRACE_SECONDS = 10
STARTUP_SECONDS = 90
# A turn is hung when opencode has emitted no event or log line and the model server's own progress
# fingerprint (slot state and counters) has not moved for this long. One generation can run ~6 min
# (16,384 reasoning tokens at 45 tok/s), so this sits above that, at the classic request cap.
STALL_SECONDS = 600
PROGRESS_POLL_SECONDS = 10
# The task home's disk cap (B1, after a 2026-10-31 trial grepped its own output file to 17 GB). Task homes run 1 to
# 21 MiB; a task whose repo needs more sets ``limits.home_mib``.
HOME_CAP_MIB = 1024
DISK_POLL_SECONDS = 2  # a runaway writer can add GBs a second; a home of a few thousand files walks in milliseconds


def tree_mib(root):
    """Disk use of ``root`` in MiB (allocated blocks, symlinks not followed); files that vanish mid-walk are skipped."""
    total, stack = 0, [str(root)]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                else:
                    total += entry.stat(follow_symlinks=False).st_blocks * 512
            except OSError:
                continue
    return total / 1048576


def _copy_resolved(source, target):
    """Copy files or trees with symlinks resolved, so the box never needs the link targets."""
    if source.is_dir():
        shutil.copytree(source, target, symlinks=False, dirs_exist_ok=True)
    elif source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.resolve(), target)


# Linux refuses any single argument over 128 KiB (MAX_ARG_STRLEN), and the document tasks paste a 32,739-token
# corpus (about 130 KB) or a 100k one (about 400 KB) into the first message.
ARGV_LIMIT = 100_000


def argv_chunks(text, limit=ARGV_LIMIT):
    """Split a message at spaces into arguments opencode rejoins with a space, so a prompt too long for one
    argument arrives unchanged. A word longer than ``limit`` is left whole: it cannot be split and the CLI
    would see it either way."""
    if len(text.encode()) <= limit:
        return [text]
    chunks, current, size = [], [], 0
    for word in text.split(" "):
        width = len(word.encode())
        if current and size + 1 + width > limit:
            chunks.append(" ".join(current))
            current, size = [word], width
        else:
            current.append(word)
            size += width + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def run_streaming(box, command, workdir, timeout, is_begun, is_settled, progress=None,
                  stall_seconds=STALL_SECONDS, home_cap_mib=HOME_CAP_MIB):
    """Run one turn, reading the harness's JSON events and logs as they arrive.

    ``is_begun(event, stderr_line)`` says the agent loop started; ``is_settled(event)`` says the turn
    finished its work (the process may still linger). Three failure shapes are named: ``stalled`` (no
    agent loop within STARTUP_SECONDS, safe to retry), ``lingered`` (settled but never exited) and
    ``hung`` (silence plus an unmoving server fingerprint for ``stall_seconds``) and ``disk_cap`` (the box's home
    passed ``home_cap_mib``, checked every DISK_POLL_SECONDS; the turn is killed there)."""
    process = subprocess.Popen(box.argv(command, box.inside(workdir)), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    stderr, began = [], threading.Event()
    active = [time.monotonic()]  # last sign of life: an event, a log line or the fingerprint moving

    def read_logs():
        for line in process.stderr:
            active[0] = time.monotonic()
            stderr.append(line)
            if is_begun(None, line):
                began.set()

    lines = queue.Queue()
    readers = [threading.Thread(target=read_logs, daemon=True),
               threading.Thread(target=lambda: [lines.put(l) for l in process.stdout] + [lines.put(None)], daemon=True)]
    for reader in readers:
        reader.start()
    events, finished_at, lingered, stalled, hung, disk_cap, home_mib = [], None, False, False, False, False, None
    home = getattr(box, "home", None)
    start = time.monotonic()
    deadline = start + timeout
    deadline_hit = False
    fingerprint, have_signal, next_poll, next_disk, silence = None, False, start, start, 0.0
    try:
        while True:
            now = time.monotonic()
            if home is not None and home_cap_mib and now >= next_disk:
                next_disk = now + DISK_POLL_SECONDS
                home_mib = round(tree_mib(home), 1)
                if home_mib > home_cap_mib and process.poll() is None:
                    disk_cap = True
                    break
            if progress and now >= next_poll:
                next_poll = now + PROGRESS_POLL_SECONDS
                try:
                    seen = progress()
                except Exception:  # an unreachable server shows no progress
                    seen = None
                if seen is not None:
                    have_signal = True
                    if seen != fingerprint:
                        fingerprint, active[0] = seen, now
            silence = max(silence, now - active[0])
            if have_signal and stall_seconds and now - active[0] > stall_seconds and process.poll() is None:
                hung = True
                break
            if finished_at and now - finished_at > LINGER_SECONDS and process.poll() is None:
                lingered = True
                break
            if not began.is_set() and not events and now - start > STARTUP_SECONDS and process.poll() is None:
                stalled = True
                break
            if now > deadline:
                deadline_hit = True
                break
            try:
                line = lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                # stdout closed: the harness is exiting. Give it a moment to be reaped, or a finished turn reads
                # as a timeout (review 2.3: two 20 s trials were scored wall_cap).
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=EXIT_GRACE_SECONDS)
                break
            try:
                event = json.loads(line)
            except ValueError:
                continue
            events.append(event)
            active[0] = time.monotonic()
            if is_begun(event, None):
                began.set()
            if is_settled(event):
                finished_at = time.monotonic()
            else:
                finished_at = None
        timed_out = deadline_hit and process.poll() is None and not lingered and not stalled and not hung and not disk_cap
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        # A child that outlived the harness can hold the pipes open; close only what has been drained.
        for reader, stream in zip(readers, (process.stderr, process.stdout)):
            reader.join(timeout=2)
            if not reader.is_alive():
                stream.close()
    return {"returncode": process.returncode, "events": events, "stderr": "".join(stderr),
            "timed_out": timed_out, "lingered": lingered, "stalled": stalled, "hung": hung,
            "disk_cap": disk_cap, "home_mib": home_mib, "max_silence_seconds": round(silence, 1)}


class OpencodeDriver:
    name = "opencode"

    def __init__(self, home, config, ref, llm_port, production_dir=None, binary_home=OPENCODE_HOME,
                 read_only=(), path=(), relays=(), turn_args=(), env=None, copy_assets=True,
                 progress=None, stall_seconds=STALL_SECONDS):
        opencode_config.check(config)
        self.home = Path(home)
        self.config, self.ref, self.llm_port = config, ref, llm_port
        self.production_dir = Path(production_dir or Path.home() / ".config/opencode")
        self.binary_home = Path(binary_home)
        self.extra_read_only, self.extra_path = list(read_only), list(path)
        self.extra_relays = list(relays)  # (inside host:port, host port) pairs, e.g. canned pages
        self.turn_args = list(turn_args)
        self.extra_env = dict(env or {})
        self.copy_assets = copy_assets  # False when pinned copies arrive as home files
        self.progress = progress  # () -> fingerprint of what the model server is doing, or None
        self.stall_seconds = stall_seconds
        self.home_cap_mib = HOME_CAP_MIB  # the session sets a task's own ``limits.home_mib``
        self.session = None
        self.raw_turns = []  # every turn in full (events, stderr, text) for llm_eval.workload.capture
        self.box = None
        self._relays = []

    def install(self):
        config_dir = self.home / ".config/opencode"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "opencode.json").write_text(json.dumps(self.config, indent=2))
        if self.copy_assets:
            for name in CONFIG_ASSETS:
                _copy_resolved(self.production_dir / name, config_dir / name)
        catalog = Path.home() / ".cache/opencode/models.json"  # avoids a models.dev fetch
        if catalog.exists():
            _copy_resolved(catalog, self.home / ".cache/opencode/models.json")
        (self.home / ".opencode").mkdir(exist_ok=True)

    def __enter__(self):
        self.install()
        self.sockets = relay_dir()
        specs = [(f"127.0.0.1:{opencode_config.SANDBOX_PORT}", "llm.sock", self.llm_port)]
        specs += [(inside, f"r{i}.sock", port) for i, (inside, port) in enumerate(self.extra_relays)]
        for _, name, port in specs:
            relay = HostRelay(self.sockets / name, port)
            relay.__enter__()
            self._relays.append(relay)
        inside_home = Path(os.fspath(Path.home()))
        self.box = Box(self.home, read_only=[(self.binary_home, str(inside_home / ".opencode")), *self.extra_read_only],
                       env={**opencode_config.environment(self.ref), **self.extra_env},
                       path=[str(inside_home / ".opencode/bin"), *self.extra_path],
                       relays=[f"{inside}={name}" for inside, name, _ in specs], sockets=self.sockets)
        return self

    def __exit__(self, *exc):
        for relay in self._relays:
            relay.__exit__(None, None, None)
        shutil.rmtree(self.sockets, ignore_errors=True)

    def turn(self, text, workdir, timeout):
        command = ["opencode", "run", "--format", "json", "--auto", "--print-logs", "--log-level", "INFO",
                   "-m", self.ref, "--dir", self.box.inside(workdir), *self.turn_args]
        if self.session:
            command += ["--session", self.session]
        started = time.monotonic()
        wall_start = time.time()
        result = self._run_streaming(command + argv_chunks(text), workdir, timeout)
        attempts = [result]
        restarts = 0
        if result["stalled"]:
            # Nothing happened yet (no agent loop began), so a retry cannot repeat any work.
            restarts = 1
            result = self._run_streaming(command + argv_chunks(text), workdir, max(1, timeout - (time.monotonic() - started)))
            attempts.append(result)
        for number, attempt in enumerate(attempts, 1):  # a stalled first attempt is kept too
            self.raw_turns.append({"said": text, "attempt": number, "argv": command, "started": wall_start,
                                   "seconds": round(time.monotonic() - started, 3),
                                   "returncode": attempt["returncode"], "timed_out": attempt["timed_out"],
                                   "lingered": attempt["lingered"], "stalled": attempt["stalled"],
                                   "hung": attempt.get("hung", False), "max_silence_seconds": attempt.get("max_silence_seconds"),
                                   "disk_cap": attempt.get("disk_cap", False), "home_mib": attempt.get("home_mib"),
                                   "events": attempt["events"], "stderr": attempt["stderr"]})
        events = result["events"]
        self.session = self.session or next((e.get("sessionID") for e in events if e.get("sessionID")), None)
        texts = [e["part"].get("text", "") for e in events if e.get("type") == "text" and e.get("part")]
        return {"answer": (texts[-1] if texts else "").strip(), "timed_out": result["timed_out"],
                "returncode": result["returncode"], "events": len(events), "lingered": result["lingered"],
                "restarts": restarts, "stderr_tail": result["stderr"][-1500:],
                "hung": result.get("hung", False), "max_silence_seconds": result.get("max_silence_seconds"),
                "disk_cap": result.get("disk_cap", False), "home_mib": result.get("home_mib"),
                "stalled": result.get("stalled", False)}  # still stalled after the one retry: opencode never started

    def _run_streaming(self, command, workdir, timeout):
        """opencode's own turn shape: the loop starts at its ``message=loop`` log line, and a step that
        finished with reason "stop" is the end of the work."""
        return run_streaming(
            self.box, command, workdir, timeout, progress=self.progress, stall_seconds=self.stall_seconds,
            home_cap_mib=getattr(self, "home_cap_mib", HOME_CAP_MIB), is_begun=lambda event, line: bool(line and "message=loop" in line) or event is not None,
            is_settled=lambda event: event.get("type") == "step_finish"
                                     and (event.get("part") or {}).get("reason") == "stop")

    def summarize(self, export=None):
        """Metrics in the shape ``metrics.summarize`` defines, from this harness's own record. The record the
        caller already holds is passed in: exporting twice in a row makes opencode fail the second time."""
        from .metrics import summarize
        return summarize(self.export() if export is None else export)

    def context_tokens(self):
        """The session's context now (the latest step's input in an export, so a compaction shows as a drop, not the
        old peak), or None when it cannot be read."""
        try:
            return self.summarize().get("final_context_tokens")
        except Exception:
            return None

    def export(self):
        """The harness's own session record.

        ``opencode export`` writes to stdout and exits without draining the pipe, so anything past one pipe
        buffer (64 KiB, and a real session is ~500 KB) arrives cut mid-string. Redirect it to a file in the
        box's own home -- a regular file, written synchronously -- and read that from the host instead."""
        if not self.session:
            return {"info": {}, "messages": []}
        stem = f".export-{uuid.uuid4().hex}"
        out_file, err_file = self.home / f"{stem}.json", self.home / f"{stem}.err"
        command = ["sh", "-c", f"opencode export {shlex.quote(self.session)} "
                               f"> {shlex.quote(self.box.inside(out_file))} 2> {shlex.quote(self.box.inside(err_file))}"]
        try:
            code, _, _, timed_out = self.box.run(command, self.home, 300)
            text = out_file.read_text(errors="replace") if out_file.exists() else ""
            said = err_file.read_text(errors="replace").strip() if err_file.exists() else ""
            if timed_out:
                raise RuntimeError("opencode export timed out after 300s")
            if code:
                raise RuntimeError(f"opencode export failed: {said[-500:]}")
            return json.loads(text[text.index("{"):])
        finally:
            for path in (out_file, err_file):
                path.unlink(missing_ok=True)
