"""Read-only plan compiler. Does not contact a server or start a process."""
import hashlib
import json
from pathlib import Path
from .policy import LADDER
from .state import safe_id


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_manifest(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("use JSON (also valid YAML), or install PyYAML for YAML syntax") from exc
        return yaml.safe_load(text)


def compile_plan(manifest_path):
    source = Path(manifest_path).resolve()
    manifest = read_manifest(source)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("expected suite schema_version 1")
    backend = manifest.get("backend", "live")
    if backend not in ("fake", "live"):
        raise ValueError("backend must be fake or live")
    runtime = manifest.get("runtime", "llama.cpp")
    if runtime not in ("llama.cpp", "strata"):
        raise ValueError("unsupported inference runtime")
    production = manifest.get("production", {})
    if not production.get("unit") or not production.get("restore_model"):
        raise ValueError("production unit and explicit restore_model are required")
    models = manifest.get("models", [])
    if not models:
        raise ValueError("suite has no models")
    seen = set()
    files = {}
    if production.get("preset"):
        preset = production["preset"]
        path = (source.parent / preset["path"]).resolve()
        actual = digest(path)
        if actual != preset.get("sha256"):
            raise ValueError(f"preset hash mismatch: {path}")
        preset["path"] = str(path)
        files[str(path)] = actual
    resolved = []
    for model in models:
        model = dict(model)
        name = safe_id(model["id"])
        if name in seen:
            raise ValueError("duplicate model id")
        seen.add(name)
        # Strata declares its own rungs: its anchor may sit above the llama.cpp ladder (262,144 is the model's
        # native window) and it must never be offered a rung above the anchor, because context is paid for in
        # expert-cache VRAM. The llama.cpp contract stays exactly as it was.
        if runtime == "strata":
            rungs = model.get("contexts", [model.get("anchor")])
            if (not all(isinstance(r, int) and r > 0 for r in rungs)
                    or model.get("anchor") not in rungs):
                raise ValueError("contexts must contain the anchor")
        else:
            # The ladder, plus the preset's own ctx-size: the 262K presets sit above the ladder's 131,072 top and
            # are pinned there with --no-step-down (2026-10-31). Nothing else above the ladder is a rung.
            loads = int((model.get("settings") or {}).get("ctx-size") or 0)
            supported = set(LADDER) | ({loads} if loads else set())
            if model.get("anchor") not in supported:
                raise ValueError("anchor must be a supported ladder rung")
            rungs = model.get("contexts", list(LADDER))
            if any(r not in supported for r in rungs) or model["anchor"] not in rungs:
                raise ValueError("contexts must contain the anchor and supported rungs")
        model["contexts"] = sorted(set(rungs))
        if not model.get("settings"):
            raise ValueError("explicit model settings required")
        if runtime == "strata":
            from .strata import pin
            model["settings"] = pin(model["settings"], files, source.parent)
        artifacts = model.get("artifacts", [])
        if backend == "live" and not artifacts:
            raise ValueError("live profiles require pinned model artifacts")
        for artifact in artifacts:
            path = (source.parent / artifact["path"]).resolve()
            actual = digest(path)
            if actual != artifact.get("sha256"):
                raise ValueError(f"artifact hash mismatch: {path}")
            artifact["path"] = str(path)
            files[str(path)] = actual
        fixtures = []
        for fixture_path in model.get("fixtures", []):
            path = (source.parent / fixture_path).resolve()
            fixture = read_manifest(path)
            safe_id(fixture["id"])
            if fixture.get("kind") not in ("recall", "tools", "evidence", "smoke", "vision", "workload"):
                raise ValueError("unsupported fixture kind")
            if fixture["kind"] == "workload":
                from .workload.tasks import validate
                validate(fixture)
            else:
                sized = isinstance(fixture.get("haystack"), dict)
                if (not sized and not isinstance(fixture.get("prompt"), str)) or fixture.get("max_output_tokens", 0) <= 0:
                    raise ValueError("fixture requires prompt or haystack and positive max_output_tokens")
            files[str(path)] = digest(path)
            fixtures.append(fixture)
        if not fixtures:
            raise ValueError("each model needs at least one fixture")
        model["fixtures"] = fixtures
        if any(f["kind"] == "workload" for f in fixtures):
            files.update(_pin_workload(model, fixtures, source.parent))
        resolved.append(model)
    plan = {"schema_version": 1, "backend": backend, "suite": str(source),
            "production": production, "models": resolved, "files": files,
            "scenario": manifest.get("scenario", {}),
            "limits": {"campaign_seconds": 28800, "model_seconds": 5400,
                       "load_seconds": 240, "request_seconds": 900,
                       **manifest.get("limits", {})},
            "live_execution_ready": False,
            "notes": ["Token counts are measured at execution, not during planning.",
                      "Haystack fixtures are sized to each context at execution; live adapters remain gated."]}
    if "runtime" in manifest:
        plan["runtime"] = runtime
    plan["plan_hash"] = fingerprint(plan)
    return plan


def _pin_workload(model, fixtures, base):
    """Validate a model's workload block and return every file it depends on, hashed.

    Snapshots are built before planning (building runs git); planning only reads and hashes.
    """
    from .workload import pi_config
    from .workload.opencode_config import check
    from .workload.snapshot import pinned_files
    workload = model.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("workload fixtures need a model 'workload' block")
    if workload.get("context") not in model["contexts"]:
        raise ValueError("workload context must be one of the model's ladder rungs")
    harness = workload.get("harness", "opencode")
    if harness == "pi":
        pi_config.check(workload.get("config") or {}, workload.get("settings") or {})
    elif harness == "opencode":
        check(workload.get("config") or {})
    else:
        raise ValueError(f"unknown harness {harness!r}")
    if not str(workload.get("ref", "")).startswith("local-llama/"):
        raise ValueError("workload ref must name the local provider's model entry")
    pinned = {}
    for fixture in (f for f in fixtures if f["kind"] == "workload"):
        snapshot = workload.get("snapshots", {}).get(fixture["id"])
        if not snapshot:
            raise ValueError(f"no snapshot for workload task {fixture['id']}")
        snapshot = (base / snapshot).resolve()
        manifest = read_manifest(snapshot / "manifest.json")
        if manifest.get("task") != fixture["id"]:
            raise ValueError(f"snapshot {snapshot} was built for {manifest.get('task')}, not {fixture['id']}")
        workload["snapshots"][fixture["id"]] = str(snapshot)
        pinned.update({path: digest(path) for path in pinned_files(snapshot)})
    binary_key = "binary" if harness == "pi" else "opencode_binary"
    version_key = "binary_version" if harness == "pi" else "opencode_version"
    binary = Path(workload.get(binary_key, "")).expanduser()
    if not binary.is_file() or not workload.get(version_key):
        raise ValueError(f"workload needs the {harness} binary path and its version")
    workload[binary_key] = str(binary.resolve())
    pinned[workload[binary_key]] = digest(binary)
    if workload.get("production_config"):
        config = Path(workload["production_config"]).expanduser().resolve()
        workload["production_config"] = str(config)
        pinned[str(config)] = digest(config)
    for relative, host in workload.get("home_files", {}).items():
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"home file paths must be relative: {relative}")
        path = (base / host).resolve()
        workload["home_files"][relative] = str(path)
        pinned[str(path)] = digest(path)
    return pinned


def verify_plan(plan):
    payload = {k: v for k, v in plan.items() if k != "plan_hash"}
    if fingerprint(payload) != plan.get("plan_hash"):
        raise ValueError("plan integrity mismatch")
    for path, expected in plan["files"].items():
        if digest(path) != expected:
            raise ValueError(f"planned input changed: {path}")
