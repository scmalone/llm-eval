"""Pinned starting states for workload tasks: built once at prepare time, cloned fresh per run.

Every snapshot ends as a git bundle, so each run starts from an identical repo. Two modes:

* history: a clone pruned to the base commit's ancestors, so the answer commit is unreachable.
  Refused when any denied path appears anywhere in that history.
* tree-only (default): the base commit's files, denied paths removed, in a fresh one-commit repo.

Withheld files come from the answer commit and land in ``hidden/``; they are copied into a run
only for grading. ``reference.diff`` (base to answer) is for the offline judge, never the model.
Nothing here modifies the source repo: it is read with ``archive``, ``show``, ``log`` and a clone.
"""
from fnmatch import fnmatch
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

from ..plan import digest
from ..state import atomic_json

DEFAULT_DENY = ("data/**", "state/**", "**/*.db", "**/*.sqlite", "**/.env", "**/.env.*",
                "**/*.pem", "**/*.key", "**/secrets*", "**/credentials*")
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache"}
SECRETS = {
    "github token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36}"),
    "aws access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "api key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}"),
    "slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
}
SCAN_LIMIT = 2 * 1024 * 1024
BRANCH = "snapshot"
EMBEDDED_ROOT = Path(__file__).resolve().parents[2] / "fixtures/workload"
# Our own temp repos only: fixed identity and dates, and no user-level hooks or config.
GIT_ENV = {"GIT_AUTHOR_NAME": "llm-eval", "GIT_AUTHOR_EMAIL": "llm-eval@localhost",
           "GIT_COMMITTER_NAME": "llm-eval", "GIT_COMMITTER_EMAIL": "llm-eval@localhost",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def matches(path, patterns):
    """Glob match where ``*`` crosses directories and a leading ``**/`` also matches the root."""
    return any(fnmatch(path, p) or (p.startswith("**/") and fnmatch(path, p[3:])) for p in patterns)


def git(args, cwd, isolated=False, binary=False):
    env = {**os.environ, **GIT_ENV} if isolated else None
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, timeout=600, env=env)
    if out.returncode:
        raise ValueError(f"git {args[0]} failed: {out.stderr.decode(errors='replace').strip()}")
    return out.stdout if binary else out.stdout.decode()


def tree_files(root):
    """path -> sha256 for every file under ``root``, skipping VCS and cache directories."""
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_file() and not path.is_symlink() and not IGNORED_DIRS & set(relative.parts):
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def scan(root):
    """(path, rule) for each likely secret in text files; the matched value is never returned."""
    hits = []
    for relative in tree_files(root):
        path = Path(root) / relative
        if path.stat().st_size > SCAN_LIMIT:
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        hits += [(relative, rule) for rule, pattern in SECRETS.items() if pattern.search(text)]
    return hits


def _history_denied(repo, commit, deny):
    specs = [f":(glob){p}" for p in deny]
    names = git(["log", "--name-only", "--format=", commit, "--", *specs], repo)
    return sorted({n for n in names.splitlines() if n.strip()})


def _tree_repo(repo, commit, deny, work):
    raw = git(["archive", "--format=tar", commit], repo, binary=True)
    removed = []
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        members = []
        for member in archive.getmembers():
            if member.isfile() and matches(member.name, deny):
                removed.append(member.name)
            elif member.isfile() or member.isdir():
                members.append(member)
        archive.extractall(work, members=members, filter="data")
    git(["init", "-q", "-b", BRANCH], work, isolated=True)
    git(["add", "-A"], work, isolated=True)
    git(["commit", "-q", "--no-verify", "-m", f"snapshot of {commit[:12]}"], work, isolated=True)
    return sorted(removed)


def _history_repo(repo, commit, deny, work):
    denied = _history_denied(repo, commit, deny)
    if denied:
        raise ValueError(f"history mode refused: denied paths appear in history ({', '.join(denied[:5])})")
    git(["clone", "-q", "--no-local", "--no-checkout", str(repo), str(work)], work.parent, isolated=True)
    git(["update-ref", f"refs/heads/{BRANCH}", commit], work, isolated=True)
    for ref in git(["for-each-ref", "--format=%(refname)"], work).split():
        if ref != f"refs/heads/{BRANCH}":
            git(["update-ref", "-d", ref], work, isolated=True)
    git(["symbolic-ref", "HEAD", f"refs/heads/{BRANCH}"], work, isolated=True)
    git(["remote", "remove", "origin"], work, isolated=True)
    git(["checkout", "-q", BRANCH], work, isolated=True)
    return []


def build(task, out_dir):
    """Build ``out_dir`` (bundle, hidden files, reference diff, manifest) for one repo task."""
    source = task["source"]
    commit, answer = source["commit"], source.get("answer")
    deny = DEFAULT_DENY + tuple(source.get("deny", []))
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    temporary = Path(tempfile.mkdtemp(prefix="llm-eval-snapshot-"))
    try:
        if "bundle" in source:
            asset = (EMBEDDED_ROOT / source["bundle"]).resolve()
            if not asset.is_relative_to(EMBEDDED_ROOT.resolve()) or not asset.is_file():
                raise ValueError(f"embedded bundle is absent: {source['bundle']}")
            repo = temporary / "embedded"
            git(["clone", "-q", "--no-checkout", str(asset), str(repo)], temporary, isolated=True)
            repo_label, place = f"embedded:{source['bundle']}", "Workspace/llm-eval"
        else:
            repo = Path(source["repo"]).expanduser().resolve()
            repo_label, place = str(repo), None
        git(["cat-file", "-e", f"{commit}^{{commit}}"], repo)
        work = temporary / "repo"
        work.mkdir()
        if source.get("history"):
            work.rmdir()
            removed = _history_repo(repo, commit, deny, work)
        else:
            removed = _tree_repo(repo, commit, deny, work)
        hidden = {}
        for relative in source.get("withhold", []):
            target = out / "hidden" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git(["show", f"{answer}:{relative}"], repo, binary=True))
            hidden[relative] = digest(target)
        hits = scan(work)
        if hidden:
            hits += [(f"hidden/{p}", r) for p, r in scan(out / "hidden")]
        if hits:
            raise ValueError("snapshot refused, likely secrets: " +
                             ", ".join(f"{p} ({r})" for p, r in hits[:10]))
        if answer:
            reference = git(["diff", commit, answer], repo, binary=True)
            found = [rule for rule, pattern in SECRETS.items() if pattern.search(reference.decode("utf-8", "ignore"))]
            if found:
                raise ValueError(f"snapshot refused, likely secrets in the reference diff ({', '.join(found)})")
            (out / "reference.diff").write_bytes(reference)
        git(["bundle", "create", "-q", str(out / "repo.bundle"), BRANCH], work, isolated=True)
        manifest = {"task": task["id"], "repo": repo_label, "commit": commit, "answer": answer,
                    "history": bool(source.get("history")), "workdir": source.get("workdir", "."),
                    "removed": removed, "files": tree_files(work), "hidden": hidden,
                    "bundle_sha256": digest(out / "repo.bundle")}
        if place:
            manifest["place"] = place
        if "cwd" in source:  # where the harness starts, home-relative ("" is ~, as in real use)
            manifest["cwd"] = source["cwd"]
        atomic_json(out / "manifest.json", manifest)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def pinned_files(snapshot_dir):
    """Every file a plan must hash so a changed snapshot fails verification."""
    root = Path(snapshot_dir)
    return sorted(str(p) for p in root.rglob("*") if p.is_file())


def materialize(snapshot_dir, dest):
    """A fresh working repo for one run: no remote, a local identity for the model's commits."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    git(["clone", "-q", "-b", BRANCH, str(Path(snapshot_dir) / "repo.bundle"), str(dest)],
        dest.parent, isolated=True)
    git(["remote", "remove", "origin"], dest, isolated=True)
    git(["config", "user.name", "llm-eval"], dest, isolated=True)
    git(["config", "user.email", "llm-eval@localhost"], dest, isolated=True)
    return dest


def escape_reason(path, home):
    """Why ``path`` is not safely inside the run's fake home, or None. The agent controls everything under the
    home, so a symlink anywhere between the home and ``path`` (the repo root included) would make host-side
    grading and capture read, hash or delete real files outside the box (red-team RT2-001)."""
    path, home = Path(path), Path(home)
    try:
        relative = path.relative_to(home)
    except ValueError:
        return f"{path} is not under the run's home"
    current = home
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return f"{current} is a symlink"
    if not path.resolve().is_relative_to(home.resolve()):
        return f"{path} resolves outside the run's home"
    return None


def safe_write(root, relative, data):
    """Write ``root/relative`` without following any symlink an agent may have planted: a
    symlinked parent directory refuses the write, a symlinked or special target is removed (the
    link, never what it points at), and the file is opened with O_NOFOLLOW."""
    root = Path(root)
    target = root / relative
    current = root
    for part in Path(relative).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"refusing to write through a symlinked directory: {current.relative_to(root)}")
        current.mkdir(exist_ok=True)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def reveal_hidden(snapshot_dir, dest):
    """Copy withheld files into a finished run for grading; returns the paths copied."""
    hidden = Path(snapshot_dir) / "hidden"
    copied = []
    for path in sorted(hidden.rglob("*")) if hidden.exists() else []:
        if path.is_file():
            relative = path.relative_to(hidden)
            safe_write(dest, relative, path.read_bytes())
            copied.append(relative.as_posix())
    return copied
