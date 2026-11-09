"""bwrap sandbox for the harness and for graders' test commands.

The box sees a read-only ``/usr`` and ``/etc``, a private ``/tmp``, and a fake home bound at the
real home's path (so instructions that name ``~/Workspace/...`` resolve to the task's copy, and
nothing else of the real home exists). It runs in fresh user, pid, ipc and network namespaces;
its only exits are the relays passed in. ``--die-with-parent`` and a process-group kill on timeout
mean a dead or stuck runner never leaves a harness behind.
"""
import os
from pathlib import Path
import signal
import subprocess
import tempfile

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
INSIDE_PACKAGE = "/opt/llm-eval"
# Only the stdlib-only modules that run inside the box are bound, never the project: task files,
# graders and replica builders hold expected answers a curious agent could otherwise read.
INSIDE_MODULES = ("llm_eval/__init__.py", "llm_eval/workload/__init__.py",
                  "llm_eval/workload/relay.py", "llm_eval/workload/machine.py")
INSIDE_RELAY = "/run/relay"
BASE_PATH = "/usr/bin"
TMP_CAP_BYTES = 1 << 30  # the box's /tmp is RAM (tmpfs): capped like the task home (B1)
# Executable git extensions a repo's own config could name; off for every grading call.
SAFE_GIT = ("-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "diff.external=",
            "-c", "core.pager=cat", "-c", "protocol.allow=never", "-c", "core.sshCommand=false")


def relay_dir():
    """A short directory for relay sockets (Unix socket paths are capped at 108 bytes)."""
    base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return Path(tempfile.mkdtemp(prefix="lle-", dir=base))


class Box:
    def __init__(self, home, real_home=None, read_only=(), env=None, path=(), relays=(), sockets=None):
        """``read_only``: (host, inside) pairs; ``relays``: ``host:port=socket-name`` specs whose
        sockets live in ``sockets`` (a host dir bound at /run/relay)."""
        self.home = Path(home)
        self.real_home = str(real_home or Path.home())
        self.read_only = [(str(Path(a).expanduser()), b) for a, b in read_only]
        self.tmpfs = []  # paths inside the box given a fresh, capped tmpfs (a read-only env's cache folder)
        self.env = dict(env or {})
        self.path = ":".join([*path, BASE_PATH])
        self.relays = list(relays)
        self.sockets = sockets

    def argv(self, command, cwd):
        args = ["bwrap", "--die-with-parent", "--unshare-all", "--new-session",
                "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/bin", "/sbin",
                "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
                "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev", "--size", str(TMP_CAP_BYTES), "--tmpfs", "/tmp",
                "--bind", str(self.home), self.real_home]
        for module in INSIDE_MODULES:
            if (PACKAGE_ROOT / module).exists():
                args += ["--ro-bind", str(PACKAGE_ROOT / module), f"{INSIDE_PACKAGE}/{module}"]
        for host, inside in self.read_only:
            args += ["--ro-bind", host, inside]
        for inside in self.tmpfs:
            args += ["--size", str(TMP_CAP_BYTES), "--tmpfs", inside]
        if self.sockets:
            args += ["--bind", str(self.sockets), INSIDE_RELAY]
        args += ["--clearenv", "--setenv", "HOME", self.real_home, "--setenv", "PATH", self.path,
                 "--setenv", "PYTHONPATH", INSIDE_PACKAGE, "--setenv", "TERM", "dumb",
                 "--setenv", "LANG", "C.UTF-8", "--setenv", "USER", Path(self.real_home).name]
        for key, value in self.env.items():
            args += ["--setenv", key, value]
        args += ["--chdir", str(cwd)]
        if self.relays:
            # -P: the task repo can contain its own llm_eval/ (the llm-eval tasks do), and without it the
            # CWD would shadow the package bound at /opt/llm-eval and the relay would not import.
            args += ["python3", "-P", "-m", "llm_eval.workload.relay"]
            for spec in self.relays:
                address, name = spec.split("=", 1)
                args += ["--relay", f"{address}={INSIDE_RELAY}/{name}"]
            args.append("--")
        return args + list(command)

    def inside(self, host_path):
        """Where a path under the fake home appears inside the box."""
        return str(Path(self.real_home) / Path(host_path).resolve().relative_to(self.home.resolve()))

    def run(self, command, cwd, timeout):
        """(returncode, stdout, stderr, timed_out); ``cwd`` is a host path under the fake home."""
        process = subprocess.Popen(self.argv(command, self.inside(cwd)), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            out, err = process.communicate(timeout=timeout)
            return process.returncode, out, err, False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            out, err = process.communicate()
            return process.returncode, out, err, True

    def git(self, args, cwd, timeout=120):
        """Git against an agent-controlled repo, inside the box: whatever the agent put in its
        repo config (fsmonitor, hooks, diff drivers) can only run where the agent already runs."""
        code, out, err, timed_out = self.run(["git", *SAFE_GIT, *args], cwd, timeout)
        return (124 if timed_out else code), out

    def runner(self):
        """The ``run(argv, cwd, timeout) -> (code, output)`` graders use for test commands."""
        def run(command, cwd, timeout):
            code, out, err, timed_out = self.run(command, cwd, timeout)
            return (124 if timed_out else code), out + err + ("\n[timed out]" if timed_out else "")
        return run
