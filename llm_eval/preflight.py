"""Read-only machine observations and retained-profile manifest generation.

This is deliberately not an execution Adapter: there are no load, request,
stop, restore, or service-control methods here.
"""
import configparser
import http.client
import ipaddress
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

from .plan import digest
from .policy import LADDER


DEFAULT_PRESET = Path.home() / ".config/llama-cpp/models-preset.ini"
PIN = Path(__file__).resolve().parents[1] / "configs" / "llama-cpp-pin.json"
RETAINED = (
    ("muse-glimmer-30b-dflash", "draft-dflash"),
    ("muse-glimmer-30b-q4", None),
    ("qwen3.8-27b-unsloth-q4-mtp", "draft-mtp"),
    ("qwen3.6-35b-a3b", None),
)
ARTIFACT_KEYS = ("model", "model-draft", "mmproj", "chat-template-file")  # a template file is part of the model (B2)
# Owner start rungs (STATE.md, 2026-10-01). Anything else starts at min(preset ctx, 98304).
ANCHORS = {"muse-glimmer-30b-dflash": 98304, "muse-glimmer-30b-q4": 98304,
           "qwen3.8-27b-unsloth-q4-mtp": 86016, "qwen3.6-35b-a3b": 98304}
START_CAP = 98304
# Settings that change neither the artifact nor the launch shape.
IGNORED_FOR_DEDUPE = ("ctx-size", "alias", "sleep-idle-seconds")


def read_presets(path=DEFAULT_PRESET):
    path = Path(path).expanduser().resolve()
    parser = configparser.ConfigParser(interpolation=None)
    with path.open() as stream:
        parser.read_file(stream)
    return path, {name: dict(parser[name]) for name in parser.sections()}


def artifact(path, role, base):
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing regular {role} artifact: {resolved}")
    result = {"role": role, "path": str(resolved), "size_bytes": resolved.stat().st_size}
    return result


def pinned_artifacts(settings, base, cache):
    # A shared cache avoids hashing the Muse target twice.
    records = []
    for key in ARTIFACT_KEYS:
        if settings.get(key):
            record = artifact(settings[key], key, base)
            if record["path"] not in cache:
                cache[record["path"]] = digest(record["path"])
            record["sha256"] = cache[record["path"]]
            records.append(record)
    return records


def retained_manifest(preset=DEFAULT_PRESET, unit="llama-router.service",
                      restore_model="muse-glimmer-30b-dflash"):
    """Read config/files only; return a pinned live manifest, never run it."""
    path, presets = read_presets(preset)
    validate_unit(unit)
    if restore_model not in presets:
        raise ValueError(f"restore model absent from presets: {restore_model}")
    fixtures = sorted((Path(__file__).resolve().parents[1] / "fixtures").glob("*.json"))
    models, cache = [], {}
    for name, spec_type in RETAINED:
        if name not in presets:
            raise ValueError(f"required retained profile absent: {name}")
        settings = presets[name]
        if not settings.get("model") or settings.get("spec-type") != spec_type:
            raise ValueError(f"unexpected model/speculation settings for {name}")
        if bool(settings.get("model-draft")) != bool(spec_type):
            raise ValueError(f"unexpected draft artifact for {name}")
        anchor = int(settings["ctx-size"])
        if anchor not in LADDER:
            raise ValueError(f"unsupported context for {name}: {anchor}")
        models.append({"id": name, "anchor": anchor, "contexts": list(LADDER),
                       "settings": settings, "artifacts": pinned_artifacts(settings, path.parent, cache),
                       "fixtures": [str(fixture) for fixture in fixtures]})
    return {"schema_version": 1, "backend": "live",
            "production": {"unit": unit, "scope": "user", "restore_model": restore_model,
                           "preset": {"path": str(path), "sha256": digest(path)}},
            "models": models}


# Preset sampling keys and the request fields they become. A profile's production sampling is what the owner gets,
# so the benchmark pins exactly that, and the capture refuses a client that sends something else.
PRESET_SAMPLING = {"temperature": "temperature", "temp": "temperature", "top-p": "top_p", "top-k": "top_k",
                   "min-p": "min_p", "presence-penalty": "presence_penalty", "repeat-penalty": "repetition_penalty"}


def preset_request(settings):
    """The pinned request for a llama.cpp profile: its preset's sampling, as numbers."""
    request = {}
    for key, field in PRESET_SAMPLING.items():
        if key in settings:
            value = float(settings[key])
            request[field] = int(value) if field == "top_k" else value
    return request


def default_anchor(name, settings):
    if name in ANCHORS:
        return ANCHORS[name]
    wanted = min(int(settings["ctx-size"]), START_CAP)
    fitting = [rung for rung in LADDER if rung <= wanted]
    if not fitting:
        raise ValueError(f"preset context below the ladder for {name}")
    return fitting[-1]


def distinct_profiles(presets):
    """One representative per launch configuration; context-only variants add nothing here.

    Profiles differing only in ctx-size or alias load the same server, and the runner
    qualifies context itself. The highest-context profile represents each group.
    """
    groups = {}
    for name, settings in presets.items():
        key = tuple(sorted((k, v) for k, v in settings.items() if k not in IGNORED_FOR_DEDUPE))
        groups.setdefault(key, []).append(name)
    chosen = []
    for names in groups.values():
        names.sort(key=lambda n: (-int(presets[n]["ctx-size"]), n))
        chosen.append((names[0], names))
    return sorted(chosen)


def eval_manifest(preset=DEFAULT_PRESET, unit="llama-router.service",
                  restore_model="muse-glimmer-30b-q4", profiles=None):
    """Pinned live manifest for every distinct preset profile (or a named subset)."""
    path, presets = read_presets(preset)
    validate_unit(unit)
    # Only the owner's live preset runs on the real llama.cpp; a test preset names no binary.
    pinned = check_pin() if Path(preset).expanduser().resolve() == DEFAULT_PRESET.resolve() else None
    if restore_model not in presets:
        raise ValueError(f"restore model absent from presets: {restore_model}")
    unknown = sorted(set(profiles or ()) - set(presets))
    if unknown:
        raise ValueError(f"profiles absent from presets: {', '.join(unknown)}")
    fixtures = sorted((Path(__file__).resolve().parents[1] / "fixtures").glob("*.json"))
    models, cache = [], {}
    for name, covers in distinct_profiles(presets):
        if profiles and not set(profiles) & set(covers):
            continue
        name = next((n for n in covers if n in (profiles or ())), name)
        settings = presets[name]
        if not settings.get("model") or not settings.get("ctx-size"):
            raise ValueError(f"profile lacks model or ctx-size: {name}")
        models.append({"id": name, "anchor": default_anchor(name, settings), "contexts": list(LADDER),
                       "covers": covers, "settings": settings,
                       "artifacts": pinned_artifacts(settings, path.parent, cache),
                       "fixtures": [str(fixture) for fixture in fixtures]})
    if not models:
        raise ValueError("no profiles selected")
    return {"schema_version": 1, "backend": "live",
            "production": {"unit": unit, "scope": "user", "restore_model": restore_model,
                           "preset": {"path": str(path), "sha256": digest(path)},
                           **({"llama_cpp": pinned} if pinned else {})},
            "models": models}


def llama_cpp_build(binary):
    """(build, commit) the binary reports, or (None, None) when it cannot say."""
    binary = Path(binary).expanduser()
    env = {"LD_LIBRARY_PATH": str(binary.parent), "PATH": "/usr/bin:/bin"}
    try:
        text = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    match = re.search(r"build (\d+), commit ([0-9a-f]+)", text.stdout + text.stderr)
    return (int(match.group(1)), match.group(2)) if match else (None, None)


def check_pin(pin_path=PIN, build_of=llama_cpp_build):
    """The pinned llama.cpp build, refused when the binary is another one: runs on different builds do not
    compare (configs/llama-cpp-pin.json, DECISIONS 2026-11-01)."""
    import json
    pin = json.loads(Path(pin_path).read_text())
    build, commit = build_of(pin["binary"])
    if build != pin["build"] or not (commit or "").startswith(pin["commit"][:7]):
        raise ValueError(f"llama.cpp is build {build} ({commit}), not the pinned build {pin['build']} "
                         f"({pin['commit']}) in {pin_path}: rebuild at the pin, or move the pin with a DECISIONS "
                         f"entry and a fresh speed sweep")
    return {"build": build, "commit": commit, "binary": pin["binary"]}


def validate_unit(unit):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*\.service", unit):
        raise ValueError("expected one literal systemd .service unit")


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def router_health(url):
    """Only GET /health on a literal loopback address; never follow redirects."""
    parsed = urlsplit(url)
    try:
        local = ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        local = False
    if (parsed.scheme != "http" or not local or parsed.path != "/health"
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError("health URL must be http://<loopback>:<port>/health")
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=5)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        body = response.read(65537)
        if len(body) > 65536:
            raise ValueError("health response exceeds 64 KiB")
        return {"status": response.status, "healthy": response.status == 200,
                "body": body.decode("utf-8", errors="replace")}
    finally:
        connection.close()


def inspect_machine(preset=DEFAULT_PRESET, unit="llama-router.service",
                    health_url="http://127.0.0.1:8080/health"):
    """Collect observations/errors. Success never grants live-run authority."""
    validate_unit(unit)
    report = {"read_only": True, "live_execution_ready": False, "observations": {}, "errors": {}}
    probes = {
        "systemd": lambda: command(["systemctl", "--user", "show", unit, "--no-pager",
                                    "--property=LoadState,ActiveState,SubState,MainPID,ExecStart"]),
        "router_health": lambda: router_health(health_url),
        "gpu": lambda: command(["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used,memory.free",
                                "--format=csv,noheader,nounits"]),
        "gpu_processes": lambda: command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                          "--format=csv,noheader,nounits"]),
    }
    def configured_artifacts():
        path, presets = read_presets(preset)
        cache = {}
        return {"path": str(path), "sha256": digest(path), "profiles": {
            name: {"settings": settings, "artifacts": pinned_artifacts(settings, path.parent, cache)}
            for name, settings in presets.items()}}
    probes["presets"] = configured_artifacts
    for name, probe in probes.items():
        try:
            report["observations"][name] = probe()
        except (OSError, ValueError, KeyError, configparser.Error,
                subprocess.SubprocessError, http.client.HTTPException) as exc:
            report["errors"][name] = str(exc)
    health = report["observations"].get("router_health", {})
    report["observations_complete"] = not report["errors"]
    report["router_healthy"] = health.get("healthy", False)
    return report
