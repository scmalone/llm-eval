"""Byte relays that are the sandbox's only way out of its private network namespace.

Inside the sandbox a TCP listener on loopback forwards to a Unix socket bind-mounted from the
host; on the host, that socket forwards to one fixed loopback port (the scratch llama-server or
the canned-page server). Nothing else is reachable, so the harness cannot leave the machine even
if its config or the model's shell commands try to.
"""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading


def _pipe(source, sink):
    try:
        while data := source.recv(65536):
            sink.sendall(data)
    except OSError:
        pass
    finally:
        for end in (source, sink):
            try:
                end.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _serve(listener, connect):
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            return
        try:
            upstream = connect()
        except OSError:
            client.close()
            continue
        threading.Thread(target=_bridge, args=(client, upstream), daemon=True).start()


def _bridge(a, b):
    """Copy both ways until both directions end, then close both sockets."""
    back = threading.Thread(target=_pipe, args=(b, a), daemon=True)
    back.start()
    _pipe(a, b)
    back.join()
    a.close()
    b.close()


class HostRelay:
    """Host side: Unix socket at ``path`` -> 127.0.0.1:``port`` (loopback only, by construction)."""

    def __init__(self, path, port):
        self.path = Path(path)
        self.port = int(port)
        if self.path.exists():
            self.path.unlink()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.path))
        self.listener.listen(64)
        self.thread = threading.Thread(target=_serve, daemon=True, args=(
            self.listener, lambda: socket.create_connection(("127.0.0.1", self.port), timeout=None)))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.listener.close()
        if self.path.exists():
            self.path.unlink()


def inside(relays, command):
    """Sandbox side: start ``host:port=socket`` listeners, run ``command``, return its exit code."""
    for spec in relays:
        address, path = spec.split("=", 1)
        host, port = address.rsplit(":", 1)
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, int(port)))
        listener.listen(64)
        def connect(path=path):
            unix = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            unix.connect(path)
            return unix
        threading.Thread(target=_serve, args=(listener, connect), daemon=True).start()
    return subprocess.run(command, env=os.environ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m llm_eval.workload.relay")
    parser.add_argument("--relay", action="append", default=[], help="host:port=/unix/socket")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return inside(args.relay, command)


if __name__ == "__main__":
    sys.exit(main())
