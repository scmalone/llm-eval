"""Prebuilt task environments (venvs, node_modules, Blender, local-only assets) under ``<data>/envs``, pinned by content.

Each environment folder holds an ``ENV.json`` that records where it came from and its ``digest``: a sha256 over every
path, symlink target, executable bit and file content in the folder (``ENV.json`` itself left out). A task's ``env``
entry carries the digest it was proven with, and the folder is re-hashed before every task-run that mounts it, so a
dependency updated after a plan was prepared refuses the run instead of quietly changing what is measured (red-team
RT-002, 2026-11-01). The digest sits in the task file, so it is part of the task's identity (``task_hash``) too.

    python3 -m llm_eval.workload.envs [name ...]    # (re)write the digest in each ENV.json; prints name and digest
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

MANIFEST = "ENV.json"


def tree_digest(root):
    """sha256 over the folder's tree: sorted paths, kinds, symlink targets, executable bits and file contents."""
    root = Path(root)
    value = hashlib.sha256()
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        here = Path(directory)
        for name in sorted(dirs + files):
            path = here / name
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST:
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                value.update(f"L {relative} {os.readlink(path)}\n".encode())
            elif stat.S_ISDIR(info.st_mode):
                value.update(f"D {relative}\n".encode())
            elif stat.S_ISREG(info.st_mode):
                content = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        content.update(block)
                executable = "x" if info.st_mode & stat.S_IXUSR else "-"
                value.update(f"F {relative} {executable} {content.hexdigest()}\n".encode())
            else:
                value.update(f"O {relative}\n".encode())
    return value.hexdigest()


def recorded(folder):
    """The digest the folder's ENV.json records, or None."""
    path = Path(folder) / MANIFEST
    return json.loads(path.read_text()).get("digest") if path.is_file() else None


def verify(folder, expected):
    """Raise unless the folder's content still hashes to ``expected``."""
    actual = tree_digest(folder)
    if actual != expected:
        raise ValueError(f"environment {Path(folder).name} changed since its tasks were proven "
                         f"(digest {actual[:12]}, task expects {expected[:12]}); rebuild the tasks or restore it")


def pin(folder):
    """Write the folder's digest into its ENV.json and return it."""
    path = Path(folder) / MANIFEST
    meta = json.loads(path.read_text()) if path.is_file() else {"name": Path(folder).name}
    meta["digest"] = tree_digest(folder)
    path.write_text(json.dumps(meta, indent=1) + "\n")
    return meta["digest"]


def main(argv=None):
    from ..paths import DATA
    base = DATA / "envs"
    names = (argv if argv is not None else sys.argv[1:]) or sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))
    for name in names:
        print(name, pin(base / name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
