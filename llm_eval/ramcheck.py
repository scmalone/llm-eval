"""Host RAM: the floor a RAM-hungry run needs, and the snapshots that log it.

Strata reads its whole GGUF into RAM. With too little free the load stalls under memory pressure until
systemd-oomd kills it (2026-11-01 22:05: 47,400 MiB available, killed twice, with the owner's terminal). Every
load that passed began with 57,014 MiB or more available, so that is the floor (owner, 2026-11-03).
The floor was measured after the plan stopped the router, but the launch checks run before that, so they credit
what the router gives back (``router_return_mib``). Snapshots are derived data: ``snapshot`` never raises.
"""
import json
import os
from pathlib import Path
import time

RAM_FLOOR_MIB = 57014
FIELDS = {"MemTotal": "mem_total_mib", "MemFree": "mem_free_mib", "MemAvailable": "mem_available_mib",
          "Buffers": "buffers_mib", "Cached": "cached_mib", "Shmem": "shmem_mib", "AnonPages": "anon_mib",
          "SwapTotal": "swap_total_mib", "SwapFree": "swap_free_mib"}


def meminfo(text=None):
    """Memory fields in MiB (``swap_used_mib`` derived) from /proc/meminfo, or from ``text``."""
    text = Path("/proc/meminfo").read_text() if text is None else text
    raw = {k: int(v.split()[0]) / 1024 for k, _, v in (line.partition(":") for line in text.splitlines()) if v.split()}
    row = {name: round(raw[key], 1) for key, name in FIELDS.items() if key in raw}
    if "swap_total_mib" in row and "swap_free_mib" in row:
        row["swap_used_mib"] = round(row["swap_total_mib"] - row["swap_free_mib"], 1)
    return row


ROUTER_UNIT = "llama-router.service"


def router_return_mib(uid=None):
    """MiB the router hands back when a plan stops it: its cgroup's anonymous and shared memory. Its page cache
    (the model file, often most of its RSS) is already counted in MemAvailable, so it is not added again. 0 when
    the router is not running or its cgroup is unreadable. Only the launch checks use this; the load gate in
    ``strata.py`` runs after the stop and reads MemAvailable alone."""
    uid = os.getuid() if uid is None else uid
    stat = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{ROUTER_UNIT}/memory.stat")
    try:
        fields = dict(line.split() for line in stat.read_text().splitlines() if len(line.split()) == 2)
        return round((int(fields.get("anon", 0)) + int(fields.get("shmem", 0))) / 2**20)
    except (OSError, ValueError):
        return 0


def refusal(available_mib, need_mib, what, returning_mib=0):
    back = f" plus {int(returning_mib):,} MiB the router returns" if returning_mib else ""
    return (f"refused: {int(available_mib):,} MiB RAM available{back}, {what} needs {int(need_mib):,} MiB; close large "
            "programs (Steam, games, browsers, other model servers) and run again")


def snapshot(path, tag, **extra):
    """Append one row ``{t, tag, ...memory fields}`` to ``path``; returns the row (None when unreadable)."""
    try:
        row = {"t": time.time(), "tag": tag, **meminfo(), **extra}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        return row
    except (OSError, ValueError, KeyError):
        return None


def floor_blocker(runtime):
    """(refusal or None, MiB available or None) before the router is stopped: a Strata run needs RAM_FLOOR_MIB
    available once the router's own memory is back; other runtimes fit in VRAM and are never blocked here."""
    try:
        available = meminfo()["mem_available_mib"]
    except (OSError, ValueError, KeyError):
        return None, None
    if runtime != "strata":
        return None, available
    returning = router_return_mib()
    if available + returning >= RAM_FLOOR_MIB:
        return None, available
    return refusal(available, RAM_FLOOR_MIB, "a Strata run", returning), available
