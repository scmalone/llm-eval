"""A systemd sleep/idle inhibitor lock held for exactly as long as the runner lives.

The lock is `systemd-inhibit ... tail --pid=<runner pid> -f /dev/null`: `tail` exits about a second after the
runner is gone (even on SIGKILL), and the lock goes with it. The child runs in its own process group so a
terminal Ctrl-C cannot drop it before the restore finishes."""
import contextlib
import os
import signal
import subprocess
import threading
import time

INHIBIT = "systemd-inhibit"
WHO = "llm-eval"
VERIFY_SECONDS = 3.0
RELEASE_WAIT_SECONDS = 5.0
WATCHER_JOIN_SECONDS = 20.0


class InhibitUnavailable(PermissionError):
    """The sleep lock could not be taken. Raised before the campaign touches the machine."""


def _listed(pid):
    """True when `systemd-inhibit --list` has a row for this runner whose PID column is `pid`.
    Rows are WHO UID USER PID COMM WHAT WHY MODE; matching the PID column, not any field, keeps a row whose UID
    happens to equal the pid from passing."""
    out = subprocess.run([INHIBIT, "--list", "--no-pager", "--no-legend"], capture_output=True, text=True, timeout=10)
    for line in out.stdout.splitlines():
        fields = line.split()
        if len(fields) > 3 and fields[0] == WHO and fields[3] == str(pid):
            return True
    return False


def _kill_group(child):
    """Release a lock child: systemd-inhibit forks tail, so signal the whole group, not just the Popen child."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            child.wait(RELEASE_WAIT_SECONDS if sig == signal.SIGTERM else 1)
            return
        except subprocess.TimeoutExpired:
            continue


def _spawn(why):
    """Start the lock child and return it only once logind lists it; otherwise kill it and raise."""
    try:
        child = subprocess.Popen(
            [INHIBIT, "--what=sleep:idle", f"--who={WHO}", f"--why={why}", "--mode=block",
             "tail", f"--pid={os.getpid()}", "-f", "/dev/null"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        raise InhibitUnavailable(f"cannot start {INHIBIT}: {exc}") from exc
    try:
        end = time.monotonic() + VERIFY_SECONDS
        while True:
            if child.poll() is not None:
                raise InhibitUnavailable(f"{INHIBIT} exited with {child.returncode} before taking the lock")
            try:
                if _listed(child.pid):
                    return child
            except (OSError, subprocess.SubprocessError) as exc:
                raise InhibitUnavailable(f"cannot list inhibitors: {exc}") from exc
            if time.monotonic() > end:
                raise InhibitUnavailable("the sleep lock never appeared in `systemd-inhibit --list`")
            time.sleep(0.1)
    except BaseException:  # includes KeyboardInterrupt: never leave a child behind
        _kill_group(child)
        raise


class Handle:
    """The live lock. The current child can be replaced once by the watcher; release kills whichever is current."""

    def __init__(self, why, child, guard):
        self.why, self.child, self.guard = why, child, guard
        self.started_at = time.time()
        self.lost = False  # set when the lock was lost and could not be re-taken
        self._retakes = 1
        self._stop = threading.Event()
        self._thread = None
        self._released = False

    @property
    def pid(self):
        return self.child.pid

    def alive(self):
        return self.child.poll() is None

    def start_watcher(self, on_event, interval=5.0):
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, args=(on_event, interval), daemon=True)
        self._thread.start()

    def watcher_running(self):
        return self._thread is not None and self._thread.is_alive()

    def stop_watcher(self, timeout=None):
        """Stop the watcher and wait for it. Returns False when it was still running after `timeout` seconds,
        so a hung watcher can never hold up the restore."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            return not self._thread.is_alive()
        return True

    def _watch(self, on_event, interval):
        try:
            while not self._stop.wait(interval):
                if self.alive():
                    continue
                on_event("sleep_inhibit_lost", pid=self.child.pid)
                _kill_group(self.child)  # a dead systemd-inhibit can leave its tail behind
                if self._retakes <= 0:
                    self.lost = True
                    return
                self._retakes -= 1
                if self._stop.is_set():  # release has begun: do not spawn a lock nobody will release
                    return
                try:
                    child = _spawn(self.why)
                except InhibitUnavailable as exc:
                    self.lost = True
                    on_event("sleep_inhibit_retake_failed", error=str(exc))
                    return
                if self._stop.is_set():
                    _kill_group(child)
                    return
                self.child = child
                on_event("sleep_inhibit_retaken", pid=child.pid)
        except Exception:  # an unexpected failure must not leave a campaign silently unprotected
            self.lost = True

    def release(self):
        if self._released:
            return
        self._released = True
        with self.guard():
            self.stop_watcher(WATCHER_JOIN_SECONDS)
            _kill_group(self.child)  # re-read after the join: the watcher may have replaced the child


@contextlib.contextmanager
def sleep_inhibit(why, guard=contextlib.nullcontext):
    """Hold the lock for the block. `guard` is a context manager factory that masks signals during release."""
    handle = Handle(why, _spawn(why), guard)
    try:
        yield handle
    finally:
        handle.release()
