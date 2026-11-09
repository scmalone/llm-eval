"""Simulated machine for ops tasks: ``systemctl``, ``journalctl``, ``nvidia-smi`` and ``ps``.

Runs inside the sandbox (stdlib only), behind wrapper scripts on the fake home's PATH. State
lives in ``~/.local/state/llm-eval-machine/machine.json``: user units with start conditions and
journals, GPU processes, and a process list. Starting a unit evaluates its conditions against the
real files in the fake home, so fixing a config and restarting is what makes it come up. Every
mutating command is appended to ``log``. The system scope is unreachable, as in a user session.
"""
import json
import os
from pathlib import Path
import re
import sys

STATE = Path(".local/state/llm-eval-machine/machine.json")


def _home():
    return Path(os.environ.get("HOME", "~")).expanduser()


def load():
    return json.loads((_home() / STATE).read_text())


def save(state):
    path = _home() / STATE
    path.write_text(json.dumps(state, indent=2))


def _path(value, state, home=None):
    """Paths under the replica's home (absolute, as real configs write them) resolve in the run's
    home, so a replica unpacked anywhere never reads the real machine's files. With ``home``
    given (grading on the host), a path that resolves outside it is refused."""
    base = Path(home) if home else _home()
    prefix = state.get("home")
    if value.startswith("~"):
        path = base / value[1:].lstrip("/")
    elif prefix and (value == prefix or value.startswith(prefix.rstrip("/") + "/")):
        path = base / value[len(prefix):].lstrip("/")
    elif home and (value == str(Path.home()) or value.startswith(str(Path.home()).rstrip("/") + "/")):
        # The box binds the run's home at the real home's path, so a config an agent wrote with the absolute
        # path (/home/<user>/models/x.gguf) is right inside the box; read it the same way on the host
        # (found 2026-11-07: correct fixes to depth-ops-ground-rule failed unit_ready).
        path = base / value[len(str(Path.home())):].lstrip("/")
    else:
        path = Path(value)
    if home and not str(path.resolve()).startswith(str(Path(home).resolve()) + "/"):
        raise OSError(f"{value} resolves outside the run's home")
    return path


def _ini_values(path, key, state, home=None):
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$")
    return [m.group(1) for line in _path(path, state, home).read_text().splitlines() if (m := pattern.match(line))]


def condition_holds(condition, state, home=None):
    kind, arg = next(iter(condition.items()))
    try:
        if kind == "file_exists":
            return _path(arg, state, home).exists()
        if kind == "file_matches":
            return re.search(arg[1], _path(arg[0], state, home).read_text(), re.M) is not None
        if kind == "file_lacks":
            return re.search(arg[1], _path(arg[0], state, home).read_text(), re.M) is None
        if kind == "ini_paths_exist":
            values = _ini_values(arg[0], arg[1], state, home)
            return bool(values) and all(_path(v, state, home).exists() for v in values)
        if kind == "unit_inactive":
            return not state["units"].get(arg, {}).get("active")
    except OSError:
        return False
    raise ValueError(f"unknown condition {kind}")


def _unit(state, name):
    name = name if "." in name else f"{name}.service"
    if name not in state["units"]:
        print(f"Unit {name} could not be found.", file=sys.stderr)
        raise SystemExit(4)
    return name, state["units"][name]


def _journal(unit, count=None):
    lines = unit.get("journal", {}).get("active" if unit.get("active") else "failed", [])
    return lines[-count:] if count else lines


def _start(state, name, unit):
    ok = all(condition_holds(c, state) for c in unit.get("requires", []))
    unit["active"] = ok
    unit["result"] = "success" if ok else "exit-code"
    state["processes"] = [p for p in state.get("processes", []) if p.get("unit") != name or ok]
    return ok


def _stop(state, name, unit):
    unit["active"] = False
    unit["result"] = "success"
    state["processes"] = [p for p in state.get("processes", []) if p.get("unit") != name]
    gpu = state.get("gpu", {})
    gpu["processes"] = [p for p in gpu.get("processes", []) if p.get("unit") != name]


def systemctl(args, state):
    if "--user" not in args:
        print("Failed to connect to system scope bus via local transport: Operation not permitted", file=sys.stderr)
        return 1
    words = [a for a in args if not a.startswith("-")]
    now = "--now" in args
    verb, names = (words[0], words[1:]) if words else ("list-units", [])
    if verb in ("list-units", "list-timers"):
        suffix = ".timer" if verb == "list-timers" else ""
        for name, unit in sorted(state["units"].items()):
            if name.endswith(suffix):
                status = "active running" if unit.get("active") else ("failed failed" if unit.get("result") == "exit-code" else "inactive dead")
                print(f"  {name:<40} loaded {status:<16} {unit.get('description', '')}")
        return 0
    if verb == "daemon-reload":
        state["log"].append("daemon-reload")
        return 0
    code = 0
    for raw in names:
        name, unit = _unit(state, raw)
        if verb == "status":
            active = "active (running)" if unit.get("active") else (
                "failed (Result: exit-code)" if unit.get("result") == "exit-code" else "inactive (dead)")
            enabled = "enabled" if unit.get("enabled") else "disabled"
            print(f"{'●' if unit.get('active') else '×'} {name} - {unit.get('description', '')}")
            print(f"     Loaded: loaded ({unit.get('path', '~/.config/systemd/user/' + name)}; {enabled}; preset: enabled)")
            print(f"     Active: {active}")
            print()
            print("\n".join(_journal(unit, 5)))
            code = code or (0 if unit.get("active") else 3)
        elif verb == "is-active":
            print("active" if unit.get("active") else ("failed" if unit.get("result") == "exit-code" else "inactive"))
            code = code or (0 if unit.get("active") else 3)
        elif verb == "is-enabled":
            print("enabled" if unit.get("enabled") else "disabled")
            code = code or (0 if unit.get("enabled") else 1)
        elif verb in ("start", "restart", "reload-or-restart", "try-restart"):
            state["log"].append(f"{verb} {name}")
            if not _start(state, name, unit):
                print(f"Job for {name} failed because the control process exited with error code.\n"
                      f'See "systemctl --user status {name}" and "journalctl --user -xeu {name}" for details.', file=sys.stderr)
                code = 1
        elif verb == "stop":
            state["log"].append(f"stop {name}")
            _stop(state, name, unit)
        elif verb in ("enable", "disable"):
            state["log"].append(f"{verb}{' --now' if now else ''} {name}")
            unit["enabled"] = verb == "enable"
            if now:
                (_start if verb == "enable" else _stop)(state, name, unit)
        elif verb == "reset-failed":
            unit["result"] = "success" if unit.get("active") else "none"
        else:
            print(f"Unknown command verb {verb}.", file=sys.stderr)
            return 1
    return code


def journalctl(args, state):
    names, count, i = [], None, 0
    while i < len(args):
        if args[i] in ("-u", "--unit"):
            names.append(args[i + 1])
            i += 1
        elif args[i].startswith("--unit="):
            names.append(args[i].split("=", 1)[1])
        elif args[i] in ("-n", "--lines"):
            count = int(args[i + 1])
            i += 1
        elif re.fullmatch(r"-xeu|-eu|-xu", args[i]) and i + 1 < len(args):
            names.append(args[i + 1])
            i += 1
        i += 1
    if "--user" not in args and not any(a in args for a in ("-xeu", "-eu", "-xu")):
        print("No journal files were found.", file=sys.stderr)
        return 1
    for raw in names or sorted(state["units"]):
        _, unit = _unit(state, raw)
        print("\n".join(_journal(unit, count)))
    return 0


def nvidia_smi(args, state):
    gpu = state.get("gpu", {})
    processes = gpu.get("processes", [])
    used = sum(p["mib"] for p in processes) + gpu.get("base_mib", 400)
    total = gpu.get("total_mib", 24564)
    util = max([p.get("util", 0) for p in processes] or [0])
    query = next((a.split("=", 1)[1] for a in args if a.startswith("--query-")), None)
    if query and any(a.startswith("--query-compute-apps") for a in args):
        for p in processes:
            row = {"pid": p["pid"], "process_name": p["name"], "used_memory": p["mib"], "used_gpu_memory": p["mib"]}
            print(", ".join(str(row.get(field.strip(), "")) for field in query.split(",")))
        return 0
    if query:
        values = {"memory.used": used, "memory.total": total, "memory.free": total - used,
                  "utilization.gpu": util, "name": gpu.get("name", "NVIDIA GeForce RTX 4090"),
                  "temperature.gpu": gpu.get("temperature", 40 + util // 2), "fan.speed": gpu.get("fan", 30 + util // 2)}
        print(", ".join(str(values.get(field.strip(), "[N/A]")) for field in query.split(",")))
        return 0
    print(f"| {gpu.get('name', 'NVIDIA GeForce RTX 4090'):<28} | Fan {gpu.get('fan', 30 + util // 2)}% "
          f"Temp {gpu.get('temperature', 40 + util // 2)}C | {used}MiB / {total}MiB | Util {util}% |")
    print("| Processes:  PID   Process name                     GPU Memory |")
    for p in processes:
        print(f"|        {p['pid']:>6}   {p['name']:<32} {p['mib']}MiB |")
    return 0


def ps(args, state):
    print("USER         PID %CPU %MEM COMMAND")
    for p in state.get("processes", []):
        print(f"{p.get('user', 'worker'):<10} {p['pid']:>5} {p.get('cpu', 0.0):>4} {p.get('mem', 0.0):>4} {p['command']}")
    return 0


TOOLS = {"systemctl": systemctl, "journalctl": journalctl, "nvidia-smi": nvidia_smi, "ps": ps}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    tool, args = argv[0], argv[1:]
    state = load()
    state.setdefault("log", [])
    try:
        code = TOOLS[tool](args, state)
    except SystemExit as exc:
        code = exc.code
    save(state)
    return code


if __name__ == "__main__":
    sys.exit(main())
