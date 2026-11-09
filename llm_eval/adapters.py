"""Fake interfaces and read-only machine discovery. No credential access."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Protocol
from .state import atomic_json


class CapacityError(RuntimeError):
    pass


class InfrastructureError(RuntimeError):
    pass


class HostMemoryError(InfrastructureError):
    """The machine lacks the RAM a load needs; refused before starting, and not retried."""


class Adapter(Protocol):
    def snapshot(self): ...
    def quiesce(self): ...
    def load(self, model, context): ...
    def tokenize(self, fixture): ...
    def request(self, fixture): ...
    def stop(self): ...
    def restore(self, snapshot): ...
    def adopt(self, snapshot): ...
    def endpoint(self): ...
    def check_alive(self): ...
    def workload_driver(self, home, manifest, model, relays=(), env=None): ...
    def workload_box(self, home, model): ...
    def progress_signal(self): ...
    def environment(self): ...
    def sample_gpu(self): ...
    def sample_host(self): ...
    def server_snapshot(self): ...


class FakeAdapter:
    """Persist fake machine state to test a new process recovering an old run."""
    def __init__(self, directory, scenario=None):
        self.path = Path(directory) / "fake-machine.json"
        self.scenario = scenario or {}
        self.calls = []
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "active": self.scenario.get("production_active", True), "loaded": None,
            "free_mib": self.scenario.get("free_mib", 22000), "restores": 0,
            "loads": {}, "requests": {}}
        self.baseline = self.scenario.get("free_mib", 22000)

    def _save(self):
        atomic_json(self.path, self.state)

    def snapshot(self):
        self.calls.append("snapshot")
        return {"was_active": self.state["active"], "free_mib": self.baseline,
                "unknown_gpu_clients": self.scenario.get("unknown_gpu_clients", []),
                "restore_model": "fake-production"}

    def quiesce(self):
        self.calls.append("quiesce")
        self.state["active"] = False
        self._save()

    def load(self, model, context):
        key = f"{model['id']}:{context}"
        self.calls.append(f"load:{key}")
        self.state["loads"][key] = self.state["loads"].get(key, 0) + 1
        self._save()
        behavior = self.scenario.get("loads", {}).get(key, {})
        if behavior.get("oom"):
            raise CapacityError("simulated allocation failure")
        if behavior.get("transient") and self.state["loads"][key] == 1:
            raise InfrastructureError("simulated load reset")
        if behavior.get("timeout"):
            raise TimeoutError("simulated load timeout")
        memory = model.get("memory", {"fixed_mib": 12000, "kv_anchor_mib": 3000})
        free = behavior.get("free_mib", self.baseline - memory["fixed_mib"]
                            - memory["kv_anchor_mib"] * context / model["anchor"])
        self.state.update(loaded=key, free_mib=free)
        self._save()
        return {"free_mib": free, "fixed_mib": memory["fixed_mib"],
                "kv_mib": memory["kv_anchor_mib"] * context / model["anchor"],
                "load_seconds": 0.01, "buffers_source": "fake"}

    def tokenize(self, fixture):
        return fixture.get("fake_token_count", len(fixture["prompt"].split()))

    def request(self, fixture):
        name = fixture["id"]
        self.calls.append(f"request:{name}")
        self.state["requests"][name] = self.state["requests"].get(name, 0) + 1
        self._save()
        behavior = self.scenario.get("requests", {}).get(name, {})
        if behavior.get("transient") and self.state["requests"][name] == 1:
            raise InfrastructureError("simulated HTTP 503")
        if behavior.get("timeout"):
            raise TimeoutError("simulated request timeout")
        if "response" in behavior:
            return behavior["response"]
        loaded = int((self.state["loaded"] or ":0").rsplit(":", 1)[1])
        if fixture["kind"] == "smoke":
            return {"content": "OK"}
        if fixture["kind"] == "vision":
            return {"content": "\n".join(fixture["expected"])}
        if fixture["kind"] == "recall":
            if loaded > self.scenario.get("recall_fails_above", float("inf")):
                return {"content": "\n".join(fixture["expected"][:-1])}
            return {"content": "\n".join(fixture["expected"])}
        if fixture["kind"] == "tools":
            return {"tool_calls": [{"function": {"name": fixture["tool_name"],
                    "arguments": json.dumps(fixture["arguments"])}}]}
        return {"content": json.dumps(fixture.get("expected", []))}

    def adopt(self, snapshot):
        self.calls.append("adopt")

    def sample_gpu(self):
        free = self.state["free_mib"]
        return {"mem_used_mib": self.baseline - free, "mem_free_mib": free, "util_gpu_pct": 0.0,
                "util_mem_pct": 0.0, "temp_c": 40.0, "power_w": 20.0, "clock_sm_mhz": 210.0, "clock_mem_mhz": 405.0}

    # -- workload seam: scripted agents instead of a harness ---------------------------------
    def endpoint(self):
        return 0

    def check_alive(self):
        """After a task-run: a scenario can make the server have died of an allocation failure."""
        if self.state.pop("pending_capacity", False):
            self._save()
            raise CapacityError("simulated allocation failure during a workload task")

    def workload_driver(self, home, manifest, model, relays=(), env=None):
        behavior = self.scenario.get("workload", {}).get(manifest["task"], {})
        if behavior.get("capacity"):
            self.state["pending_capacity"] = True
            self._save()
        self.calls.append(f"workload:{manifest['task']}")
        return FakeWorkloadDriver(behavior.get("turns", []))

    def workload_box(self, home, model):
        return FakeBox()

    def stop(self):
        self.calls.append("stop")
        self.state.update(loaded=None, free_mib=self.baseline)
        self._save()
        if self.scenario.get("leak_mib", 0) > 256:
            raise InfrastructureError("memory failed to return to baseline")

    def restore(self, snapshot):
        self.calls.append("restore")
        self.state["restores"] += 1
        self._save()
        if self.state["restores"] <= self.scenario.get("restore_failures", 0):
            raise InfrastructureError("simulated restore failure")
        self.state.update(active=snapshot["was_active"], loaded=None, free_mib=self.baseline)
        self._save()
        return {"service_matches": True, "health": True if snapshot["was_active"] else None,
                "four_token_probe": True if snapshot["was_active"] else None,
                "restored_inactive": not snapshot["was_active"]}


class FakeWorkloadDriver:
    """A scripted agent: each turn is ``[{path: new text}, answer]`` applied to the working dir."""
    def __init__(self, turns):
        self.turns = [list(t) for t in turns]
        self.last = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def turn(self, text, workdir, timeout):
        edits, answer = self.turns.pop(0) if self.turns else ({}, "Nothing more to do.")
        for name, content in edits.items():
            path = Path(workdir) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.last = answer
        return {"answer": answer, "timed_out": False, "returncode": 0, "events": 1}

    def export(self):
        return {"messages": [{"info": {"role": "user", "time": {"created": 0}}, "parts": []},
                             {"info": {"role": "assistant", "time": {"created": 0, "completed": 1000}},
                              "parts": [{"type": "step-finish", "tokens": {"input": 100, "output": 10}},
                                        {"type": "text", "text": self.last}]}]}

    def summarize(self, export=None):
        from .workload.metrics import summarize
        return summarize(self.export() if export is None else export)


class FakeBox:
    """Fake-backend test commands run directly; only synthetic test repos ever reach here."""
    def git(self, args, cwd, timeout=120):
        from .workload.sandbox import SAFE_GIT
        env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
        out = subprocess.run(["git", *SAFE_GIT, *args], cwd=cwd, capture_output=True, text=True,
                             timeout=timeout, env=env)
        return out.returncode, out.stdout

    def runner(self):
        def run(command, cwd, timeout):
            out = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
            return out.returncode, out.stdout + out.stderr
        return run


def read_gpu_snapshot():
    """Explicit read-only discovery; never called by plan or fake execution."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used,memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True,
                            text=True, check=True, timeout=10)
    rows = [line.split(",") for line in result.stdout.splitlines() if line.strip()]
    return [{"uuid": row[0].strip(), "total_mib": int(row[1]),
             "used_mib": int(row[2]), "free_mib": int(row[3])} for row in rows]
