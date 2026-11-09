"""Strata plans and runtime hooks over the existing managed scratch-service lifecycle.

The production service remains the captured llama router. Only the run-owned scratch server
uses Strata. Planning reads/hashes files; it never loads a model or invokes setup.
"""
import copy
import json
import re
from pathlib import Path

from .adapters import HostMemoryError, InfrastructureError
from .ramcheck import RAM_FLOOR_MIB
from .live import LiveAdapter
from .plan import digest
from .state import atomic_json

PATH_FLAGS = {"--pack", "--native", "--ple-gguf", "--expert-profile", "--mtp", "--model", "--mmproj"}
CONFIG_KEYS = {"exe", "args", "cwd", "tokenizer", "model_name", "lib_dirs", "gpu", "gpus_asked",
               "env", "sampling", "reasoning_effort", "reasoning_budget_tokens", "fit_max_tokens", "answer_room"}
DEFAULT_REQUEST = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                   "presence_penalty": 0.0, "repetition_penalty": 1.0, "reasoning_effort": "high",
                   "reasoning_budget_tokens": 0, "chat_template_kwargs": {"enable_thinking": True,
                                                                          "preserve_thinking": True}}


DEFAULT_ENV = {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_ROUTE_PRIOR": "0", "STRATA_KVA": "0"}


def _path(value, base):
    return (base / Path(value).expanduser()).resolve()


def pin(settings, files, base):
    """Resolve a Strata launch description and pin every declared model/code dependency."""
    settings = copy.deepcopy(settings)
    for key in ("config", "python", "server"):
        if not settings.get(key):
            raise ValueError(f"Strata settings require {key}")
        path = (base / Path(settings[key]).expanduser()).absolute() if key == "python" else _path(settings[key], base)
        if not path.is_file():
            raise ValueError(f"Strata {key} is missing")
        settings[key] = str(path)
        files[str(path)] = digest(path)
    config = json.loads(Path(settings["config"]).read_text())
    unknown = set(config) - CONFIG_KEYS - {"log", "port", "host", "open_browser", "vision"}
    if unknown:
        raise ValueError("Strata config contains unsupported keys: " + ", ".join(sorted(unknown)))
    if not isinstance(config.get("args"), list) or not all(isinstance(a, str) for a in config["args"]):
        raise ValueError("Strata args must be a list of strings")
    cwd = _path(config.get("cwd", "."), Path(settings["config"]).parent)
    config["cwd"] = str(cwd)
    config["exe"] = str(_path(config["exe"], cwd))
    config["tokenizer"] = str(_path(config["tokenizer"], cwd))
    args = config["args"]
    paths = [Path(config["exe"]), Path(config["tokenizer"])]
    engine_dir = Path(config["exe"]).parent
    paths += sorted(engine_dir.glob('*.so*'))
    build_info = engine_dir / 'BUILD.json'
    settings['engine_build'] = json.loads(build_info.read_text()) if build_info.is_file() else None
    if build_info.is_file():
        paths.append(build_info)
    for i, arg in enumerate(args):
        if arg in PATH_FLAGS:
            if i + 1 >= len(args):
                raise ValueError(f"Strata missing value for {arg}")
            args[i + 1] = str(_path(args[i + 1], cwd))
            paths.append(Path(args[i + 1]))
        elif "=" in arg and arg.split("=", 1)[0] in PATH_FLAGS:
            raise ValueError("Strata file flags must use separate flag/value elements")
    # Engine cwd also imports its Python server and tokenizer modules.
    server = Path(settings["server"])
    paths += [server.parent]
    tools = server.parent.parent / "tools"
    if tools.is_dir():
        paths += sorted(tools.glob("*.py"))
    for dependency in settings.get("dependencies", []):
        paths.append(_path(dependency, base))
    import re
    for path in list(paths):
        match = re.search(r"-00001-of-(\d+)\.gguf$", path.name)
        if match:
            for shard in range(2, int(match[1]) + 1):
                paths.append(path.with_name(path.name.replace("-00001-of-", f"-{shard:05}-of-")))
    for path in paths:
        if not path.exists():
            raise ValueError(f"Strata dependency missing: {path}")
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for child in candidates:
            if child.is_file() and "__pycache__" not in child.parts:
                files[str(child.resolve())] = digest(child)
    config = {k: v for k, v in config.items() if k in CONFIG_KEYS}
    settings["pinned_files"] = dict(files)
    settings["resolved_config"] = config
    request = settings.get("request", DEFAULT_REQUEST)
    if not isinstance(request, dict):
        raise ValueError("Strata request settings must be an object")
    if set(request) - set(DEFAULT_REQUEST) - {"seed"}:
        raise ValueError("unsupported Strata experiment request setting")
    settings["request"] = copy.deepcopy(request)
    settings["capabilities"] = {"token_count": "templated Anthropic count_tokens",
                                "reasoning_preservation": "unverified", "live_progress": "unverified"}
    return settings


def run_config(settings, context, port, directory):
    """Generate only run-owned config/log paths; original config is never edited."""
    config = copy.deepcopy(settings["resolved_config"])
    args, clean, i = config["args"], [], 0
    while i < len(args):
        arg = args[i]
        if arg.split("=", 1)[0] in ("--max-context", "--host", "--port", "--batch", "--parallel"):
            i += 1 if "=" in arg else 2
            continue
        if arg == "--vision":
            i += 1
            continue
        clean.append(arg)
        i += 1
    config["args"] = clean + ["--max-context", str(context)]
    config.update(host="127.0.0.1", port=port, open_browser=False, log=str(directory / "engine.log"))
    # Never silently shorten the declared output allowance or reserve answer room from thinking.
    config.update(answer_room=False, fit_max_tokens=False)
    config["env"] = {**config.get("env", {}), **settings.get("env", {})}
    request = settings["request"]
    config["sampling"] = {k: v for k, v in request.items() if k in DEFAULT_REQUEST and k not in
                           ("reasoning_effort", "reasoning_budget_tokens", "chat_template_kwargs")}
    for key in ("reasoning_effort", "reasoning_budget_tokens"):
        if key in request:
            config[key] = request[key]
    atomic_json(directory / "config.json", config)
    return directory / "config.json"


class StrataAdapter(LiveAdapter):
    """Reuse scratch ownership, snapshot, restore, memory gates, telemetry and inhibition."""

    def __init__(self, plan, *args, **kwargs):
        super().__init__(plan, *args, **kwargs)
        self.settings = plan['models'][0]['settings']

    def launch_command(self, model, context, port):
        self.settings = model["settings"]
        directory = self.log_dir / f"strata-{model['id']}-{context}"
        path = run_config(self.settings, context, port, directory)
        self.host.ensure_dir(self.log_dir)
        return [self.settings["python"], self.settings["server"], "--engine", "strata",
                "--config", str(path), "--host", "127.0.0.1", "--port", str(port), "--parallel", "1"]

    def is_ready(self, status, body):
        return status == 200 and isinstance(body, dict) and body.get("loaded") is True

    # The engine prices the cache itself and prints the result, so the log is the forecast's source:
    #   expert cache auto: 14.89 GiB free, 2048 MiB reserved (+218 MiB for the draft head,
    #                      +256 MiB for what is allocated after the cache) -> 5010 slots
    #   expert cache 6542 slots, 12.42 GiB of VRAM
    # A load leaves the reserve minus 35 MiB free on 0.1.54 (upstream cfa197e2, #4474).
    AUTO = re.compile(r"expert cache auto: ([\d.]+) GiB free, (\d+) MiB reserved "
                      r"\(\+(\d+) MiB for the draft head, \+(\d+) MiB for what is allocated after the cache\) "
                      r"-> (\d+) slots")
    CACHE = re.compile(r"expert cache (\d+) slots, ([\d.]+) GiB of VRAM")
    DRAFT = re.compile(r"draft layer loaded, (\d+) MiB of VRAM")
    FREE_AFTER_RESERVE_MIB = 35
    # Strata's /metrics fingerprint counts generated tokens, so a long generation is progress, not silence. The
    # longest silence across every 0.1.54 trial to 2026-10-30 was 116.5 s (a 243K-token prefill is ~60 s); the
    # llama.cpp default stays at driver.STALL_SECONDS until its slot fingerprint is shown to move mid-generation.
    STALL_SECONDS = 300
    KV_MIB_PER_131K = 1946  # measured 1.9 +/- 0.2 GiB per 131,072 tokens on 0.1.54

    def memory_report(self, text):
        """What the engine said it priced: cache slots and size, reserve, draft head and late allocation."""
        report = {}
        if auto := self.AUTO.search(text):
            report.update(free_before_mib=int(float(auto.group(1)) * 1024), reserve_mib=int(auto.group(2)),
                          draft_mib=int(auto.group(3)), post_cache_mib=int(auto.group(4)),
                          planned_slots=int(auto.group(5)))
        if cache := self.CACHE.search(text):
            report.update(cache_slots=int(cache.group(1)), cache_mib=int(float(cache.group(2)) * 1024))
        if draft := self.DRAFT.search(text):
            report.setdefault("draft_mib", int(draft.group(1)))
        return report

    VERSION = re.compile(r"session is up \(engine ([\w.+-]+)\)")

    def _engine_log(self):
        """The engine's own log (run_config points it at ``engine.log`` beside the run-local config). The cache
        and version lines are written there, not to the server log the parent reads."""
        context = getattr(self, "_loading_context", None)
        model = getattr(self, "_loading_model", None)
        if context is None or model is None:
            return ""
        path = self.log_dir / f"strata-{model}-{context}" / "engine.log"
        try:
            return path.read_text(errors="replace")
        except OSError:
            return ""

    def kv_mib(self, text):
        """The KV share of the footprint at the context being loaded, from the measured slope. The engine's
        own cache line is what the section reports; this number is what lets the runner forecast a lower rung."""
        engine = self._engine_log()
        # The last load's lines: a restarted engine appends to the same log.
        start = engine.rfind("expert cache auto:")
        self.memory = self.memory_report((engine[start:] if start >= 0 else engine) + "\n" + text)
        if version := self.VERSION.findall(engine):
            self.memory["engine_version"] = version[-1]
        context = getattr(self, "_loading_context", None)
        return None if context is None else round(self.KV_MIB_PER_131K * context / 131072)

    # The engine reads the whole --native GGUF into RAM. With less available than that plus this margin the load
    # stalls under memory pressure until systemd-oomd kills it (2026-11-01 22:05: 47,400 MiB available for a
    # 52,278 MiB model, killed twice, along with the owner's terminal). Loads that passed began at 57,014 MiB or more,
    # so the need is never below ramcheck.RAM_FLOOR_MIB.
    RAM_MARGIN_MIB = 2048

    def ram_gate(self, model):
        """Refuse a load the host cannot hold; runs after quiesce, so production's RAM is already back."""
        args = (model["settings"].get("resolved_config") or {}).get("args", [])
        if "--native" not in args:
            return
        native = Path(args[args.index("--native") + 1])
        try:
            need = max(native.stat().st_size // 2**20 + self.RAM_MARGIN_MIB, RAM_FLOOR_MIB)
            fields = dict(line.split(":", 1) for line in self.host.read_text("/proc/meminfo").splitlines() if ":" in line)
            available = int(fields["MemAvailable"].split()[0]) // 1024
        except (OSError, KeyError, ValueError) as exc:
            raise InfrastructureError(f"cannot size the RAM gate: {exc}") from exc
        if available < need:
            raise HostMemoryError(
                f"refused: {available:,} MiB RAM available, the load needs {need:,} MiB (the larger of the "
                f"{RAM_FLOOR_MIB:,} MiB floor and {native.name} plus {self.RAM_MARGIN_MIB} MiB); close large programs "
                "(Steam, games, browsers, other model servers) and run again")

    def load(self, model, context):
        """Add the engine's own memory report and a forecast to what the parent measured."""
        self.ram_gate(model)
        self._loading_context, self._loading_model = context, model["id"]
        try:
            result = super().load(model, context)
        finally:
            self._loading_context = self._loading_model = None
        report = getattr(self, "memory", {}) or {}
        if report:
            result["memory"] = report
            built = (self.settings.get("engine_build") or {}).get("version")
            if report.get("engine_version") and built and built != report["engine_version"]:
                # BUILD.json beside the binary is stale (2026-10-30: it said 0.1.45 under a 0.1.54 engine).
                result["engine_build_mismatch"] = {"build_json": built, "engine_log": report["engine_version"]}
            if report.get("reserve_mib") is not None:
                result["predicted_free_mib"] = report["reserve_mib"] - self.FREE_AFTER_RESERVE_MIB
                result["forecast_source"] = "engine cache line and the measured reserve rule"
        return result

    def tokenize(self, fixture):
        status, body = self.host.http("POST", self._server(), "/v1/messages/count_tokens",
            {"messages": [{"role": "user", "content": fixture["prompt"]}], "thinking": {"type": "enabled"}}, timeout=120)
        count = body.get("input_tokens") if isinstance(body, dict) else None
        if status != 200 or type(count) is not int or count <= 0:
            raise InfrastructureError("Strata templated token count unavailable")
        return count

    def request(self, fixture):
        # Base OpenAI request/answer normalization remains shared, with explicit experiment fields.
        return super().request(fixture)

    def progress_signal(self):
        status, body = self.host.http("GET", self._server(), "/metrics", timeout=5)
        if status != 200 or not isinstance(body, dict) or not isinstance(body.get("live"), dict):
            return None
        live = body["live"]
        # Exclude elapsed time and hardware sampling: these move even when inference is wedged.
        return tuple(live.get(k) for k in ("state", "phase", "prompt_tokens", "prompt_read", "prompt_total", "generated"))

    def environment(self):
        info = super().environment() or {}
        # The parent's discovered binary is the production router, not the Strata engine.
        production = info.pop("llama_server", {})
        info["production_server"] = production
        engine = self.settings['resolved_config']['exe']
        info["runtime"] = {"name": "strata", "engine_build": self.settings.get('engine_build'),
                           "engine_version": (getattr(self, "memory", {}) or {}).get("engine_version"),
                           "path": engine, "sha256": self.settings.get('pinned_files', {}).get(engine),
                           "server": self.settings['server'], "properties": "captured after load"}
        return info

    def pinned_request(self, model):
        return dict(model["settings"]["request"])

    def served_model_name(self, model):
        return model["settings"]["resolved_config"].get("model_name")

    def check_alive(self):
        self.check_capture()
        if self._scratch_state() != "active":
            self._request_failed("Strata scratch service exited")
        status, body = self.host.http("GET", self._server(), "/health", timeout=10)
        if not self.is_ready(status, body):
            self._request_failed("Strata engine is not loaded")


def ladder(context):
    """The anchor plus the standard rungs below it, so a marginal fit degrades instead of aborting.

    The Strata pilot of 2026-10-29 declared a single rung, so a capacity failure had nothing to fall back to.
    Rungs above the anchor are deliberately not offered: the anchor is the target, and a Strata engine's
    context is priced in VRAM that the expert cache then loses.
    """
    from .policy import LADDER
    return sorted({int(context), *(int(r) for r in LADDER if int(r) < int(context))})


def manifest(config, python, server, profile, context, preset, restore_model, request=None):
    """Read-only preparation without registering Strata in production presets."""
    from .policy import LADDER
    from .preflight import read_presets
    path, presets = read_presets(preset)
    if restore_model not in presets:
        raise ValueError("restore model absent from production presets")
    config = Path(config).expanduser().resolve()
    # The launch config's own env wins over the accuracy-preserving defaults: settings.env overrides the config
    # at launch (run_config), so hard-coding it here silently turned a STRATA_ROUTE_TAIL_SKIP=1 knob arm back
    # into the baseline.
    env = {**DEFAULT_ENV, **(json.loads(config.read_text()).get("env") or {})}
    settings = {"config": str(config), "python": str(Path(python).expanduser().absolute()),
                "server": str(Path(server).expanduser().resolve()), "env": env,
                "request": copy.deepcopy(request or DEFAULT_REQUEST)}
    return {"schema_version": 1, "backend": "live", "runtime": "strata",
            "production": {"unit": "llama-router.service", "scope": "user", "restore_model": restore_model,
                           "preset": {"path": str(path), "sha256": digest(path)}},
            "models": [{"id": profile, "anchor": context, "contexts": ladder(context), "settings": settings,
                        "artifacts": [{"role": "launch_config", "path": str(config), "sha256": digest(config)}]}]}
