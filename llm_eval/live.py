"""Execution adapter for the real machine: systemd router, scratch llama-server, nvidia-smi.

Nothing here is reachable from the CLI. ``execute`` only accepts a live plan when a caller
constructs this adapter and passes it in, and every external effect goes through ``Host`` so
the recovery behavior is tested against a scripted machine. Safety rules:

* The scratch server runs as a transient user unit named from the run ID, so a resumed run can
  stop a crashed predecessor's server by name; nothing is ever killed by pid or process name.
* The production unit is the one literal unit named in the plan.
* A GPU client outside the production cgroup that holds more than 1 GiB of VRAM blocks the campaign,
  whatever its name; smaller clients (the compositor) are ignored.
* Buffer sizes come from the server's own log; unparsable logs yield ``None``, never a guess.
"""
import contextlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time

from .adapters import CapacityError, InfrastructureError
from .preflight import validate_unit
from .state import safe_id

GPU_CLIENT_LIMIT_MIB = 1024  # a non-production GPU client above this blocks a run; the name does not matter
DROPPED_KEYS = {"sleep-idle-seconds", "alias", "port", "host", "models-max"}
OOM_MARKERS = ("out of memory", "cudamalloc failed", "failed to allocate", "unable to allocate",
               "cudamalloc error", "ggml_backend_cuda_buffer_type_alloc_buffer")
BASELINE_TOLERANCE_MIB = 256
KV_LINE = re.compile(r"KV buffer size\s*=\s*([0-9.]+)\s*MiB")
EXEC_PATH = re.compile(r"path=(\S+)")
EXEC_ARGV = re.compile(r"argv\[\]=(.*?)\s;")


class RealHost:
    """The only place that touches the machine."""

    def run(self, args, timeout=30):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise InfrastructureError(f"{args[0]} timed out after {timeout}s") from exc

    def ensure_dir(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def http(self, method, port, path, body=None, timeout=30):
        import http.client
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            payload = None if body is None else json.dumps(body)
            connection.request(method, path, body=payload, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read()
            try:
                return response.status, json.loads(raw)
            except ValueError:
                return response.status, raw.decode("utf-8", errors="replace")
        except TimeoutError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise ConnectionError(str(exc)) from exc
        finally:
            connection.close()

    def read_text(self, path):
        return Path(path).read_text()

    def free_port(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def gpu_free_mib(self):
        out = self.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], 10)
        if out.returncode:
            raise InfrastructureError("nvidia-smi failed: " + out.stderr.strip())
        return int(out.stdout.splitlines()[0])

    def gpu_sample(self):
        from .telemetry import QUERY, parse_nvidia_smi
        out = self.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"], 5)
        if out.returncode:
            raise InfrastructureError("nvidia-smi failed: " + out.stderr.strip())
        return parse_nvidia_smi(out.stdout.splitlines()[0])

    def host_sample(self):
        """Load, memory, pressure and extra GPU fields for telemetry/host.jsonl; each part fails alone."""
        row, errors = {}, []
        def part(name, action):
            try:
                row.update(action())
            except Exception as exc:
                errors.append(f"{name}: {exc}"[:120])
        def load():
            one, five, fifteen = Path("/proc/loadavg").read_text().split()[:3]
            return {"load1": float(one), "load5": float(five), "load15": float(fifteen)}
        def memory():
            from .ramcheck import meminfo
            return meminfo()  # available, free, cached, anon, shmem, swap used: all MiB
        def pressure():
            out = {}
            for name in ("cpu", "memory", "io"):
                path = Path("/proc/pressure") / name
                if path.exists():
                    some = path.read_text().splitlines()[0].split()
                    out.update({f"psi_{name}_some_{k}": float(v) for k, v in (p.split("=") for p in some[1:4])})
            return out
        def gpu_extra():
            wanted = ("pstate", "clocks_throttle_reasons.active", "fan.speed", "pcie.link.gen.current",
                      "pcie.link.width.current", "power.limit", "temperature.memory", "utilization.encoder",
                      "utilization.decoder")
            if getattr(self, "_gpu_fields", None) is None:  # which names this driver accepts, found once
                found = [f for f in wanted if self.run(
                    ["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits"], 5).returncode == 0]
                if not found:
                    raise InfrastructureError("no extra GPU fields available")  # asked again next sample
                self._gpu_fields = found
            out = self.run(["nvidia-smi", f"--query-gpu={','.join(self._gpu_fields)}",
                            "--format=csv,noheader,nounits"], 5)
            if out.returncode:
                raise InfrastructureError(out.stderr.strip()[:100])
            cells = [c.strip() for c in out.stdout.splitlines()[0].split(",")]
            return {"gpu_" + name.replace(".", "_"): cell for name, cell in zip(self._gpu_fields, cells)}
        def apps():
            return {"gpu_apps": self.gpu_processes()}
        for name, action in (("load", load), ("memory", memory), ("pressure", pressure),
                             ("gpu_extra", gpu_extra), ("gpu_apps", apps)):
            part(name, action)
        if errors:
            row["partial_errors"] = errors
        return row

    def environment(self, binary):
        """GPU model and driver plus the llama-server version string; None for what cannot be read."""
        info = {"gpu": {"name": None, "driver": None, "memory_total_mib": None},
                "llama_server": {"path": binary, "version": None}}
        try:
            out = self.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader,nounits"], 10)
            name, driver, memory = [c.strip() for c in out.stdout.splitlines()[0].split(",")]
            info["gpu"] = {"name": name, "driver": driver, "memory_total_mib": float(memory)}
        except Exception:
            pass
        try:
            out = self.run([binary, "--version"], 20)
            text = (out.stdout + out.stderr).strip().splitlines()
            info["llama_server"]["version"] = " | ".join(l.strip() for l in text if "version" in l.lower() or "build" in l.lower())[:300] or None
        except Exception:
            pass
        return info

    def gpu_processes(self):
        out = self.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                        "--format=csv,noheader,nounits"], 10)
        if out.returncode:
            raise InfrastructureError("nvidia-smi failed: " + out.stderr.strip())
        rows = [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line.strip()]
        return [{"pid": int(r[0]), "name": r[1], "mib": int(r[2])} for r in rows]

    def sleep(self, seconds):
        time.sleep(seconds)

    def monotonic(self):
        return time.monotonic()


def server_args(binary, settings, context, port):
    """ini profile -> llama-server argv. ``true`` is a bare flag; ``false`` is omitted."""
    args = [binary, "--host", "127.0.0.1", "--port", str(port)]
    merged = {**settings, "ctx-size": str(context)}
    for key in sorted(merged):
        value = str(merged[key]).strip()
        if key in DROPPED_KEYS or value.lower() == "false":
            continue
        args.append(f"--{key}")
        if value.lower() != "true":
            args.append(value)
    if "metrics" not in merged:
        args.append("--metrics")  # read-only Prometheus endpoint, snapshotted around every trial and task-run
    return args


def exec_identity(exec_start):
    """The launch shape of a unit's ExecStart (path and argv), without runtime fields like pid."""
    return [(path, argv.strip()) for path, argv in
            zip(EXEC_PATH.findall(exec_start), EXEC_ARGV.findall(exec_start))]


def has_oom(text):
    lowered = text.lower()
    return any(marker in lowered for marker in OOM_MARKERS)


def parse_kv_mib(log_text):
    sizes = [float(m) for m in KV_LINE.findall(log_text)]
    return sum(sizes) if sizes else None


def tool_schema(fixture):
    properties = {name: {"type": "string"} for name in fixture["arguments"]}
    return [{"type": "function", "function": {
        "name": fixture["tool_name"], "description": "Return the status of a request by id.",
        "parameters": {"type": "object", "properties": properties,
                       "required": sorted(properties)}}}]


class LiveAdapter:
    def __init__(self, plan, log_dir, host=None, *, run_id):
        production = plan["production"]
        self.unit = production["unit"]
        validate_unit(self.unit)
        if production.get("scope", "user") != "user":
            raise ValueError("only user-scope units are supported")
        self.restore_model = production["restore_model"]
        self.limits = plan["limits"]
        self.log_dir = Path(log_dir)
        self.host = host or RealHost()
        # Only a campaign on the real machine takes the sleep lock; scripted hosts (tests) must not touch systemd.
        self.holds_machine = isinstance(self.host, RealHost)
        self.gpu_client_limit = production.get("gpu_client_limit_mib", GPU_CLIENT_LIMIT_MIB)
        self.baseline = None
        self.active = None
        self.binary = None
        # One scratch unit per run: a resumed run reuses the name and so can stop a leftover.
        self.run_id = safe_id(run_id)
        self.scratch_unit = f"llm-eval-scratch-{self.run_id}.service"
        # Set at every launch; cleared only when memory is seen back near baseline.
        self.owes_memory = False

    # -- systemd -------------------------------------------------------------------------
    def _systemctl(self, *args, timeout=60, program="systemctl"):
        return self.host.run([program, "--user", *args], timeout)

    def _show(self, prop, unit=None):
        out = self._systemctl("show", unit or self.unit, f"--property={prop}", "--value")
        if out.returncode:
            raise InfrastructureError(f"systemctl show {prop} failed: {out.stderr.strip()}")
        return out.stdout.strip()

    def _wait(self, predicate, seconds, label):
        deadline = self.host.monotonic() + seconds
        while True:
            if predicate():
                return
            if self.host.monotonic() >= deadline:
                raise InfrastructureError(f"timed out waiting for {label}")
            self.host.sleep(1)

    def _production_pids(self):
        group = self._show("ControlGroup")
        if not group:
            return set()
        try:
            text = self.host.read_text(f"/sys/fs/cgroup{group}/cgroup.procs")
        except OSError:
            return set()
        return {int(line) for line in text.split() if line.strip()}

    # -- Adapter protocol ----------------------------------------------------------------
    def snapshot(self):
        state = self._show("ActiveState")
        exec_start = self._show("ExecStart")
        match = re.search(r"path=(\S+)", exec_start)
        if not match:
            raise InfrastructureError("cannot read the llama-server path from the unit")
        self.binary = match.group(1)
        own = self._production_pids()
        unknown = [f"{p['pid']}:{p['name']} ({p['mib']} MiB)" for p in self.host.gpu_processes()
                   if p["pid"] not in own and p["mib"] > self.gpu_client_limit]
        return {"was_active": state == "active", "exec_start": exec_start, "scratch_unit": self.scratch_unit,
                "free_mib": self.host.gpu_free_mib(), "unknown_gpu_clients": unknown,
                "restore_model": self.restore_model}

    def quiesce(self):
        """Stop production, then wait for the GPU to settle; returns the usable baseline."""
        out = self._systemctl("stop", self.unit)
        if out.returncode:
            raise InfrastructureError("could not stop the production unit: " + out.stderr.strip())
        self._wait(lambda: self._show("ActiveState") in ("inactive", "failed"), 60, "unit to stop")
        readings = []
        def settled():
            readings.append(self.host.gpu_free_mib())
            return len(readings) >= 3 and max(readings[-3:]) - min(readings[-3:]) <= 128
        self._wait(settled, 60, "GPU memory to settle")
        self.baseline = readings[-1]
        return {"free_mib": self.baseline, "quiesced": True}

    def adopt(self, snapshot):
        """A resumed process inherits the crashed run's scratch unit and memory obligation."""
        if snapshot.get("scratch_unit", self.scratch_unit) != self.scratch_unit:
            raise ValueError("adapter run ID does not match the run being resumed")
        # With production back up its memory would fail the gate, so only owe while it is down.
        if snapshot.get("quiesced") and self._show("ActiveState") != "active":
            self.baseline = snapshot["free_mib"]
            self.owes_memory = True

    def load(self, model, context):
        self.model_request = self.pinned_request(model)
        if self.baseline is None:
            raise InfrastructureError("quiesce must run before load")
        if self.owes_memory or self._scratch_state() not in ("inactive", "failed"):
            raise InfrastructureError("the previous scratch server is not cleaned up; stop it first")
        port = self.host.free_port()
        log_path = self.log_dir / f"{model['id']}-{context}-{int(self.host.monotonic())}.log"
        args = self.launch_command(model, context, port)
        started = self.host.monotonic()
        self.host.ensure_dir(self.log_dir)
        self.owes_memory = True
        out = self._systemctl(
            "--quiet", "--collect", "--unit", self.scratch_unit.removesuffix(".service"),
            "--property=KillMode=control-group", "--property=TimeoutStopSec=30",
            f"--property=StandardOutput=append:{log_path}", f"--property=StandardError=append:{log_path}",
            "--", *args, program="systemd-run")
        if out.returncode:
            raise InfrastructureError("could not start the scratch server: " + out.stderr.strip())
        self.active = {"port": port, "log": log_path, "offset": 0}
        deadline = started + self.limits["load_seconds"]
        while True:
            if self._scratch_state() not in ("active", "activating"):
                failure = CapacityError(f"server exited during load at {context}: allocation failure") \
                    if has_oom(self._log_text(log_path)) \
                    else InfrastructureError("server exited during load")
                # A failed load still has to give its memory back before anything else runs.
                self._terminate(gate=True)
                raise failure
            try:
                status, _ = self.host.http("GET", port, "/health", timeout=5)
                if self.is_ready(status, _):
                    break
            except (ConnectionError, TimeoutError):
                pass
            if self.host.monotonic() >= deadline:
                self._terminate(gate=True)
                raise InfrastructureError(f"load exceeded {self.limits['load_seconds']}s at {context}")
            self.host.sleep(1)
        free = self.host.gpu_free_mib()
        text = self._log_text(log_path)
        self.active["offset"] = len(text)
        kv = self.kv_mib(text)
        used = self.baseline - free
        return {"free_mib": free, "kv_mib": kv,
                "fixed_mib": None if kv is None else used - kv,
                "load_seconds": self.host.monotonic() - started,
                "buffers_source": "log" if kv is not None else "unparsed"}

    def launch_command(self, model, context, port):
        return server_args(self.binary, model["settings"], context, port)

    def is_ready(self, status, body):
        return status == 200

    def kv_mib(self, text):
        return parse_kv_mib(text)

    def _scratch_state(self):
        return self._show("ActiveState", self.scratch_unit)

    def _log_text(self, path):
        try:
            return self.host.read_text(path)
        except OSError:
            return ""

    def _server(self):
        if not self.active:
            raise InfrastructureError("no scratch server is loaded")
        return self.active["port"]

    def tokenize(self, fixture):
        status, body = self.host.http("POST", self._server(), "/tokenize",
                                      {"content": fixture["prompt"]}, timeout=120)
        if status != 200 or not isinstance(body, dict) or "tokens" not in body:
            raise InfrastructureError(f"tokenize failed with HTTP {status}")
        return len(body["tokens"])

    def request(self, fixture):
        content = fixture["prompt"]
        if fixture["kind"] == "vision":
            content = [{"type": "image_url", "image_url": {"url": fixture["image_uri"]}},
                       {"type": "text", "text": fixture["prompt"]}]
        payload = {"messages": [{"role": "user", "content": content}],
                   "max_tokens": fixture["max_output_tokens"], "seed": fixture.get("seed", 0),
                   "stream": False}
        payload.update(getattr(self, "settings", {}).get("request", {}) or getattr(self, "model_request", {}))
        if fixture["kind"] == "tools":
            payload["tools"] = tool_schema(fixture)
            payload["tool_choice"] = "auto"
        started = self.host.monotonic()
        try:
            status, body = self.host.http("POST", self._server(), "/v1/chat/completions", payload,
                                          timeout=self.limits["request_seconds"])
        except ConnectionError as exc:
            self._request_failed(str(exc))
        except TimeoutError as exc:
            self._request_failed(str(exc), TimeoutError)
        if status >= 500:
            self._request_failed(f"server returned HTTP {status}")
        if status != 200 or not isinstance(body, dict) or not body.get("choices"):
            raise ValueError(f"unexpected response (HTTP {status})")
        message = body["choices"][0]["message"]
        # Graders read only the answer channel; reasoning and finish_reason are kept so an
        # empty answer can be told apart from a budget spent on reasoning.
        result = {"content": message.get("content") or "",
                  "reasoning": message.get("reasoning_content") or "",
                  "finish_reason": body["choices"][0].get("finish_reason"),
                  "output_tokens": (body.get("usage") or {}).get("completion_tokens"),
                  # Kept for the metrics corpus: the server's own timings (prefill/decode speed, prompt
                  # cache hits) and usage, plus the wall time of this request as the runner saw it.
                  "usage": body.get("usage"), "timings": body.get("timings"),
                  "request_seconds": round(self.host.monotonic() - started, 3)}
        if fixture["kind"] == "tools":
            result["tool_calls"] = message.get("tool_calls") or []
        return result

    def environment(self):
        """Host details for environment.json; test hosts that cannot report give nothing."""
        report = getattr(self.host, "environment", None)
        return report(self.binary) if report else {}

    def sample_gpu(self):
        """One telemetry sample, or None when the host cannot take one (scripted test hosts)."""
        sample = getattr(self.host, "gpu_sample", None)
        return sample() if sample else None

    def sample_host(self):
        """Host load, memory, pressure, extra GPU fields and the scratch unit's memory and CPU; None for scripted hosts."""
        sample = getattr(self.host, "host_sample", None)
        if not sample:
            return None
        row = sample()
        try:
            show = self._systemctl("show", self.scratch_unit, "--property=ActiveState,MemoryCurrent,CPUUsageNSec", timeout=10)
            row.update({k.lower(): v for k, _, v in (l.partition("=") for l in show.stdout.splitlines())})
        except Exception as exc:
            row.setdefault("partial_errors", []).append(f"unit: {exc}"[:120])
        return row

    def server_snapshot(self):
        """The scratch llama-server's own view: /props, /slots, /metrics and /health with the log position."""
        if not getattr(self.host, "host_sample", None):
            return None  # scripted hosts have nothing to snapshot
        if not self.active:
            return {"error": "no scratch server is loaded"}
        port, report = self.active["port"], {"log": str(self.active["log"])}
        report["log_bytes"] = len(self._log_text(self.active["log"]))
        for name in ("health", "props", "slots", "metrics"):
            try:
                status, body = self.host.http("GET", port, f"/{name}", timeout=10)
                report[name] = {"status": status, "body": body if len(json.dumps(body, default=str)) < 2_000_000 else "[over 2 MB]"}
            except Exception as exc:
                report[name] = {"error": str(exc)[:200]}
        return report

    def _request_failed(self, detail, otherwise=InfrastructureError):
        """An allocation failure after load is a capacity limit, not a transient fault to retry."""
        if self.active and has_oom(self._log_text(self.active["log"])[self.active["offset"]:]):
            raise CapacityError(f"allocation failure during request: {detail}")
        raise otherwise(detail)

    # -- workload seam: real opencode against the scratch server ---------------------------
    def endpoint(self):
        """The scratch server's loopback port (RealHost.http and server_args bind 127.0.0.1 only)."""
        return self._server()

    def check_capture(self):
        """No successful completion at all, or a 5xx/transport failure, means the machine is not serving: stop
        the block. Drift the client caused is the trial's own result (an ops task can ask the agent to rewrite
        its own client config), so it is recorded and the trial is graded."""
        capture = getattr(self, "inference_capture", None)
        if capture is not None and capture.errors:
            if capture.server_errors or not capture.ok:
                raise InfrastructureError("inference request contract/transport failed; see raw/inference")
            self.client_drift = capture.client_errors[:5]

    def check_alive(self):
        """After a task-run: the scratch server must still be up; an allocation failure that killed
        it is a capacity limit, anything else is infrastructure."""
        self.check_capture()
        port = self._server()
        try:
            alive = self._scratch_state() == "active" and self.host.http("GET", port, "/health", timeout=10)[0] == 200
        except (ConnectionError, TimeoutError):
            alive = False
        if not alive:
            self._request_failed("scratch server is gone after a workload task")

    # The llama.cpp slot fingerprint is not yet shown to move mid-generation, so a whole generation can read as
    # silence: keep the watchdog above the longest one (driver.STALL_SECONDS).
    STALL_SECONDS = None

    def pinned_request(self, model):
        """The request settings this run pins (sampling, effort): Strata keeps them in ``settings.request``, a
        llama.cpp profile at the model level (``settings`` there is the preset, which becomes server flags)."""
        return dict(model.get("request") or {})

    def served_model_name(self, model):
        return None

    @contextlib.contextmanager
    def workload_driver(self, home, manifest, model, relays=(), env=None):
        """The harness (opencode or pi) behind a run-local capture: every request is recorded, the pinned
        settings are supplied where the client leaves them out, and a conflicting client value is refused.
        The same boundary for every runtime, so a llama.cpp profile and Strata are measured alike."""
        from .workload.inference import InferenceCapture
        from .workload.driver import OpencodeDriver, STALL_SECONDS
        from .workload.pi_driver import PI_HOME, PiDriver
        from .workload import pi_config
        workload = model["workload"]
        harness = workload.get("harness", "opencode")
        request = self.pinned_request(model)
        stall = self.STALL_SECONDS or STALL_SECONDS
        expected = {**request, "max_tokens": workload["mode"]["output_limit"]}
        if harness == "pi":
            # pi sends the same sampling the pinned request names, but maps max_tokens to max_completion_tokens
            # and reasoning_effort through its thinking level, so the contract is built from its own config.
            expected = pi_config.expected_request(model["id"], workload["mode"]["context"],
                                                 workload["mode"]["output_limit"], request,
                                                 workload.get("thinking", "high"), workload.get("production_entry"))
        with InferenceCapture(self.endpoint(), Path(home).parent / "raw" / "inference", expected,
                              self.served_model_name(model), self.limits["request_seconds"]) as capture:
            self.inference_capture = capture
            path = [str(Path.home() / p) for p in manifest.get("path", [])] + list(workload.get("path", []))
            read_only = [tuple(p) for p in workload.get("read_only", [])]
            if harness == "pi":
                driver = PiDriver(home, model["id"], workload["ref"], capture.port, request,
                                  effort=workload.get("thinking", "high"),
                                  context=workload["mode"]["context"],
                                  output=workload["mode"]["output_limit"],
                                  binary_home=Path(workload.get("binary_home") or PI_HOME),
                                  read_only=read_only, path=path, relays=relays, env=env,
                                  progress=self.progress_signal, stall_seconds=stall,
                                  version=workload.get("binary_version", ""),
                                  config=workload.get("config"), settings=workload.get("settings"))
            else:
                driver = OpencodeDriver(home, workload["config"], workload["ref"], capture.port,
                                        binary_home=Path(workload["opencode_binary"]).parents[1],
                                        read_only=read_only, path=path, relays=relays, env=env, copy_assets=False,
                                        progress=self.progress_signal, stall_seconds=stall)
            with driver:
                yield driver

    def progress_signal(self):
        """A fingerprint of what the scratch server is doing: slot state and counters change while a
        prompt is processed or tokens are generated and stay put when it is wedged. None when neither
        endpoint answers (so the driver has no signal and only measures silence). Which of the two
        moves mid-generation is unverified until a GPU run; any change counts as progress."""
        port, seen = self._server(), []
        for name in ("slots", "metrics"):
            status, body = self.host.http("GET", port, f"/{name}", timeout=5)
            if status == 200:
                seen.append((name, body if isinstance(body, str) else json.dumps(body, sort_keys=True)))
        return tuple(seen) or None

    def workload_box(self, home, model):
        from .workload.sandbox import Box
        workload = model["workload"]
        return Box(home, read_only=[tuple(pair) for pair in workload.get("read_only", [])],
                   path=list(workload.get("path", [])))

    def stop(self):
        self._terminate(gate=False)

    def _terminate(self, gate):
        """Stop the scratch unit by name (also a crashed run's leftover), then require memory
        back near baseline whenever something ran since the last passing check."""
        self.active = None
        state = self._scratch_state()
        running = state not in ("inactive", "failed")
        if running:
            self._unit_report("show", ["show", self.scratch_unit])
            out = self._systemctl("stop", self.scratch_unit)
            if out.returncode:
                raise InfrastructureError("could not stop the scratch server: " + out.stderr.strip())
            self._wait(lambda: self._scratch_state() in ("inactive", "failed"), 60, "scratch server to stop")
            self._unit_report("journal", None)
        if state == "failed" or self._scratch_state() == "failed":
            self._systemctl("reset-failed", self.scratch_unit)
        if (running or gate or self.owes_memory) and self.baseline is not None:
            self._wait(lambda: self.baseline - self.host.gpu_free_mib() <= BASELINE_TOLERANCE_MIB,
                       60, "GPU memory to return to baseline")
            self.owes_memory = False

    def _unit_report(self, label, show_args):
        """Keep systemd's view of the scratch unit (properties before the stop, its journal after) beside the
        server log. Best effort and real hosts only: it never changes what the run does."""
        if not getattr(self.host, "host_sample", None):
            return
        try:
            if show_args:
                out = self._systemctl(*show_args, timeout=20)
            else:
                out = self.host.run(["journalctl", "--user", "-u", self.scratch_unit, "--no-pager",
                                     "-o", "short-iso-precise"], 30)
            name = f"{self.scratch_unit.removesuffix('.service')}-{int(self.host.monotonic())}.{label}.txt"
            self.host.ensure_dir(self.log_dir)
            (self.log_dir / name).write_text(out.stdout + (f"\n[stderr]\n{out.stderr}" if out.stderr else ""))
        except Exception:
            pass

    def restore(self, snapshot):
        if not snapshot["was_active"]:
            self._systemctl("stop", self.unit)
            self._wait(lambda: self._show("ActiveState") in ("inactive", "failed"), 60, "unit to stay stopped")
            return {"service_matches": self._show("ActiveState") != "active", "health": None,
                    "four_token_probe": None, "restored_inactive": True}
        out = self._systemctl("start", self.unit, timeout=120)
        if out.returncode:
            raise InfrastructureError("could not start the production unit: " + out.stderr.strip())
        self._wait(lambda: self._show("ActiveState") == "active", 60, "unit to start")
        matches = exec_identity(self._show("ExecStart")) == exec_identity(snapshot["exec_start"])
        return {"service_matches": matches, "health": self._router_health(),
                "four_token_probe": self._probe(), "restored_inactive": False}

    def _router_port(self):
        match = re.search(r"--port\s+(\d+)", self._show("ExecStart"))
        return int(match.group(1)) if match else 8080

    def _router_health(self):
        port = self._router_port()
        def healthy():
            try:
                return self.host.http("GET", port, "/health", timeout=5)[0] == 200
            except (ConnectionError, TimeoutError):
                return False
        self._wait(healthy, 120, "router health")
        return True

    def _probe(self):
        """Four tokens through the production restore model, proving it loads and answers."""
        status, body = self.host.http(
            "POST", self._router_port(), "/v1/chat/completions",
            {"model": self.restore_model, "max_tokens": 4, "stream": False,
             "messages": [{"role": "user", "content": "Say hello."}]},
            timeout=self.limits["load_seconds"] + 60)
        return status == 200 and isinstance(body, dict) and bool(body.get("choices"))
