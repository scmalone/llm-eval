import json
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from llm_eval.adapters import CapacityError, InfrastructureError
from llm_eval.live import LiveAdapter, RealHost, exec_identity, parse_kv_mib, server_args
from llm_eval.plan import compile_plan
from llm_eval.preflight import eval_manifest
from llm_eval.runner import execute

EXEC = ("{ path=/opt/llama/llama-server ; argv[]=/opt/llama/llama-server --host 127.0.0.1 --port 8080 "
        "--models-max 1 ; ignore_errors=no ; start_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }")
LOG = "load_tensors: CUDA0 model buffer size = 15000.00 MiB\nllama_kv_cache: CUDA0 KV buffer size = 2000.00 MiB\n"


class ScriptedHost:
    """A tiny machine: one production unit, one GPU, scratch servers as transient units."""

    def __init__(self, oom_above=None, leak_mib=0, procs=(), cgroup_pids=(), refuse_probe=False,
                 oom_text="ggml_cuda: cudaMalloc failed: out of memory\n", request_oom_above=None,
                 stuck_after_failed_load=False, request_oom_mode="503", stuck_after_request_oom=False):
        self.clock, self.calls, self.stops = 0.0, [], []
        self.unit_active, self.exec_start = True, EXEC
        self.total, self.server, self.starts = 23000, None, 0
        self.oom_above, self.leak_mib, self.procs = oom_above, leak_mib, list(procs)
        self.cgroup_pids, self.refuse_probe, self.logs, self.probed = list(cgroup_pids), refuse_probe, {}, 0
        self.oom_text, self.request_oom_above = oom_text, request_oom_above
        self.stuck_after_failed_load, self.stuck = stuck_after_failed_load, False
        self.request_oom_mode, self.stuck_after_request_oom = request_oom_mode, stuck_after_request_oom

    def run(self, args, timeout=30):
        self.calls.append(args)
        if args[0] == "systemd-run":
            return self._launch(args)
        verb = args[2]
        unit = args[3]
        scratch = unit.startswith("llm-eval-scratch-")
        if verb == "show":
            prop = args[4].split("=")[1]
            if scratch:
                value = "active" if self.server else "inactive"
            else:
                value = {"ActiveState": "active" if self.unit_active else "inactive", "ExecStart": self.exec_start,
                         "ControlGroup": "/user.slice/llama-router.service"}[prop]
            return subprocess.CompletedProcess(args, 0, value + "\n", "")
        if scratch:
            if verb == "stop":
                self.stops.append(unit)
                self.server = None
            return subprocess.CompletedProcess(args, 0, "", "")
        self.unit_active = verb == "start"
        if verb == "start":
            self.starts += 1
            self.exec_start = re.sub(r"pid=\d+", f"pid={self.starts}", self.exec_start)
            self.exec_start = re.sub(r"start_time=\[[^\]]*\]", f"start_time=[t{self.starts}]", self.exec_start)
        return subprocess.CompletedProcess(args, 0, "", "")

    def _launch(self, argv):
        command = argv[argv.index("--") + 1:]
        log = next(a.split("append:")[1] for a in argv if a.startswith("--property=StandardOutput="))
        context = int(command[command.index("--ctx-size") + 1])
        self.ensure_dir(Path(log).parent)
        if self.oom_above and context > self.oom_above:
            self.logs[log] = self.oom_text
            self.stuck = self.stuck_after_failed_load
            return subprocess.CompletedProcess(argv, 0, "", "")
        self.logs[log] = LOG
        self.server = {"ctx": context, "port": int(command[command.index("--port") + 1]), "log": log}
        return subprocess.CompletedProcess(argv, 0, "", "")

    def ensure_dir(self, path):
        pass

    def read_text(self, path):
        if "cgroup.procs" in str(path):
            return "\n".join(map(str, self.cgroup_pids))
        if str(path) in self.logs:
            return self.logs[str(path)]
        raise FileNotFoundError(path)

    def free_port(self):
        return 40123

    def gpu_free_mib(self):
        if self.server:
            return self.total - 17000
        if self.stuck:
            return self.total - 9000
        leak = self.leak_mib * bool(self.stops)
        return self.total - 500 - leak if self.unit_active else self.total - leak

    def gpu_processes(self):
        return self.procs

    def gpu_sample(self):
        free = float(self.gpu_free_mib())
        return {"mem_used_mib": self.total - free, "mem_free_mib": free, "util_gpu_pct": 7.0, "util_mem_pct": 3.0,
                "temp_c": 51.0, "power_w": 120.0, "clock_sm_mhz": 1800.0, "clock_mem_mhz": 10000.0}

    def sleep(self, seconds):
        self.clock += seconds

    def monotonic(self):
        self.clock += 0.01
        return self.clock

    def http(self, method, port, path, body=None, timeout=30):
        if port == 8080:
            if path == "/health":
                return 200, {"status": "ok"}
            self.probed += 1
            return (503, "loading") if self.refuse_probe else (200, {"choices": [{"message": {"content": "hi"}}]})
        if not self.server or self.server["port"] != port:
            raise ConnectionError("refused")
        if (self.request_oom_above and path == "/v1/chat/completions"
                and self.server["ctx"] > self.request_oom_above
                and len(str(body["messages"][0]["content"])) > 1000):  # only deep-context prompts
            self.logs[self.server["log"]] += self.oom_text
            self.server = None
            self.stuck = self.stuck_after_request_oom
            if self.request_oom_mode == "timeout":
                raise TimeoutError("timed out")
            return 503, "gone"
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/tokenize":
            return 200, {"tokens": [0] * len(body["content"].split())}
        prompt = body["messages"][0]["content"]
        if isinstance(prompt, list):
            self.saw_image = prompt[0]["image_url"]["url"].startswith("data:image/png;base64,")
            return 200, {"choices": [{"message": {"content": "red\ngreen\nblue\nyellow"}}]}
        if prompt.startswith("Reply with the single word OK"):
            return 200, {"choices": [{"message": {"content": "OK", "reasoning_content": "thinking"},
                                      "finish_reason": "stop"}]}
        if "tools" in body:
            wanted = re.search(r"id (\S+?)\.", prompt).group(1)
            call = {"function": {"name": body["tools"][0]["function"]["name"], "arguments": json.dumps({"id": wanted})}}
            return 200, {"choices": [{"message": {"content": "", "tool_calls": [call]}}]}
        lines = [f"{a}:{b}" for a, b in re.findall(r"RECORD: site=(\S+) code=(\d+)", prompt)]
        return 200, {"choices": [{"message": {"content": "\n".join(lines)}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 90, "completion_tokens": 40 + len(lines)},
                     "timings": {"prompt_n": 90, "cache_n": 0, "prompt_ms": 31.4, "predicted_n": 40 + len(lines),
                                 "predicted_ms": 888.0, "predicted_per_second": 45.0}}


def plan_for(adapter_root, vision=False, **production):
    """A live plan over fake gguf files, built through the real manifest generator."""
    root = Path(adapter_root)
    root.mkdir(parents=True, exist_ok=True)
    model = root / "m.gguf"
    model.write_bytes(b"not a model")
    projector = root / "mmproj.gguf"
    projector.write_bytes(b"not a projector")
    preset = root / "models-preset.ini"
    preset.write_text(f"[tiny]\nmodel = {model}\nctx-size = 131072\njinja = true\nparallel = 1\n"
                      + (f"mmproj = {projector}\n" if vision else ""))
    manifest = eval_manifest(preset, restore_model="tiny")
    manifest["production"].update(production)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return compile_plan(root / "manifest.json")


class PureTests(unittest.TestCase):
    def test_quadrant_png_is_a_valid_image_with_the_right_colors(self):
        from llm_eval.images import COLORS, quadrants_png
        import struct
        import zlib
        png = quadrants_png(["red", "green", "blue", "yellow"], 8)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (8, 8))
        data = zlib.decompress(png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8])
        pixel = lambda x, y: tuple(data[y * 25 + 1 + x * 3: y * 25 + 4 + x * 3])
        self.assertEqual([pixel(0, 0), pixel(7, 0), pixel(0, 7), pixel(7, 7)],
                         [COLORS["red"], COLORS["green"], COLORS["blue"], COLORS["yellow"]])

    def test_server_args_translate_the_preset(self):
        args = server_args("/bin/ls", {"jinja": "true", "mmproj-offload": "false", "flash-attn": "on",
                                       "ctx-size": "131072", "sleep-idle-seconds": "600"}, 98304, 41000)
        self.assertEqual(args[:5], ["/bin/ls", "--host", "127.0.0.1", "--port", "41000"])
        self.assertIn("--jinja", args)
        self.assertNotIn("--mmproj-offload", args)
        self.assertEqual(args[args.index("--ctx-size") + 1], "98304")
        self.assertEqual(args[args.index("--flash-attn") + 1], "on")
        self.assertNotIn("--sleep-idle-seconds", args)

    def test_kv_parsing_never_guesses(self):
        self.assertEqual(parse_kv_mib(LOG + "KV buffer size = 500.5 MiB"), 2500.5)
        self.assertIsNone(parse_kv_mib("nothing useful here"))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = plan_for(self.root)

    def adapter(self, **host):
        self.host = ScriptedHost(**host)
        return LiveAdapter(self.plan, self.root / "logs", self.host, run_id="t")

    def test_snapshot_blocks_only_unknown_gpu_clients(self):
        procs = [{"pid": 1, "name": "/usr/bin/kwin_wayland", "mib": 200},
                 {"pid": 7, "name": "llama-server", "mib": 9000},
                 {"pid": 9, "name": "/home/u/ComfyUI/python", "mib": 20000}]
        snap = self.adapter(procs=procs, cgroup_pids=[7]).snapshot()
        self.assertEqual(snap["unknown_gpu_clients"], ["9:/home/u/ComfyUI/python (20000 MiB)"])
        self.assertTrue(snap["was_active"])

    def test_the_gpu_check_is_by_usage_not_by_name(self):
        procs = [{"pid": 1, "name": "anything-small", "mib": 1024},          # exactly 1 GiB is allowed
                 {"pid": 2, "name": "/usr/bin/kwin_wayland", "mib": 2000},   # a formerly allow-listed name is judged by size
                 {"pid": 3, "name": "Grim Dawn.exe", "mib": 1025},
                 {"pid": 4, "name": "tiny-helper", "mib": 300},
                 {"pid": 7, "name": "llama-server", "mib": 17000}]            # production's own, by cgroup
        snap = self.adapter(procs=procs, cgroup_pids=[7]).snapshot()
        self.assertEqual(snap["unknown_gpu_clients"], ["2:/usr/bin/kwin_wayland (2000 MiB)", "3:Grim Dawn.exe (1025 MiB)"])

    def test_load_measures_memory_from_the_server_log(self):
        adapter = self.adapter()
        adapter.snapshot()
        self.assertEqual(adapter.quiesce(), {"free_mib": 23000, "quiesced": True})
        result = adapter.load(self.plan["models"][0], 98304)
        self.assertEqual((result["free_mib"], result["kv_mib"], result["fixed_mib"]), (6000, 2000.0, 15000.0))
        self.assertEqual(result["buffers_source"], "log")
        self.assertEqual(adapter.tokenize({"prompt": "a b c"}), 3)
        adapter.stop()
        self.assertEqual(self.host.stops, [adapter.scratch_unit])

    def test_allocation_failure_is_a_capacity_error_and_other_exits_are_not(self):
        adapter = self.adapter(oom_above=98304)
        adapter.snapshot()
        adapter.quiesce()
        with self.assertRaises(CapacityError):
            adapter.load(self.plan["models"][0], 131072)
        self.assertEqual(self.host.stops, [])

    def test_stop_requires_memory_to_return_to_baseline(self):
        adapter = self.adapter(leak_mib=4000)
        adapter.snapshot()
        adapter.quiesce()
        adapter.load(self.plan["models"][0], 65536)
        with self.assertRaisesRegex(InfrastructureError, "return to baseline"):
            adapter.stop()

    def test_server_errors_are_infrastructure_and_only_the_scratch_unit_is_stopped(self):
        adapter = self.adapter()
        adapter.snapshot()
        adapter.quiesce()
        adapter.load(self.plan["models"][0], 65536)
        original = self.host.http
        self.host.http = lambda *a, **k: (503, "busy") if a[2] == "/v1/chat/completions" else original(*a, **k)
        with self.assertRaises(InfrastructureError):
            adapter.request({"kind": "recall", "prompt": "x", "max_output_tokens": 8})
        self.host.http = original
        adapter.stop()
        self.assertEqual(self.host.stops, [adapter.scratch_unit])

    def test_restore_verifies_unit_health_and_probe(self):
        adapter = self.adapter()
        snap = adapter.snapshot()
        adapter.quiesce()
        self.assertFalse(self.host.unit_active)
        verified = adapter.restore(snap)
        self.assertEqual((verified["service_matches"], verified["health"], verified["four_token_probe"]), (True, True, True))
        self.assertNotEqual(self.host.exec_start, snap["exec_start"])  # pid and start time moved
        self.host.exec_start = EXEC.replace("--models-max 1", "--models-max 2")
        self.assertFalse(adapter.restore(snap)["service_matches"])

    def test_restore_leaves_an_inactive_unit_inactive(self):
        adapter = self.adapter()
        self.host.unit_active = False
        snap = adapter.snapshot()
        verified = adapter.restore(snap)
        self.assertTrue(verified["restored_inactive"])
        self.assertEqual(self.host.probed, 0)


class ReviewRegressionTests(unittest.TestCase):
    """One test per Codex finding (LLE-001 to LLE-007) plus the ExecStart hardening."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = plan_for(self.root)

    def adapter(self, **host):
        self.host = ScriptedHost(**host)
        return LiveAdapter(self.plan, self.root / "logs", self.host, run_id="t")

    def test_command_timeout_is_infrastructure_not_an_escape(self):  # LLE-001
        timeout = subprocess.TimeoutExpired(["systemctl"], 5)
        with patch("llm_eval.live.subprocess.run", side_effect=timeout):
            with self.assertRaises(InfrastructureError):
                RealHost().run(["systemctl", "--user", "start", "x"], 5)

    def test_a_restore_that_raises_unexpectedly_is_retried_and_recorded(self):  # LLE-001
        from llm_eval.adapters import FakeAdapter
        from test_runner import plan_with

        class Flaky(FakeAdapter):
            tripped = False

            def restore(self, snapshot):
                if not Flaky.tripped:
                    Flaky.tripped = True
                    raise subprocess.TimeoutExpired(["systemctl"], 5)
                return super().restore(snapshot)

        result = execute(plan_with(), self.root / "r", "flaky", adapter=Flaky(self.root / "r" / "flaky"))
        self.assertFalse(result["restoration_required"])
        self.assertEqual(result["status"], "complete")

    def test_a_fresh_adapter_stops_a_crashed_runs_scratch_server(self):  # LLE-002
        crashed = self.adapter()
        crashed.snapshot()
        crashed.quiesce()
        crashed.load(self.plan["models"][0], 65536)
        self.assertIsNotNone(self.host.server)
        resumed = LiveAdapter(self.plan, self.root / "logs", self.host, run_id="t")  # new process, no memory
        resumed.stop()
        self.assertIsNone(self.host.server)
        self.assertEqual(self.host.stops, [crashed.scratch_unit])
        self.assertEqual(resumed.scratch_unit, crashed.scratch_unit)

    def test_unit_names_follow_the_run_id(self):  # LLE-002
        self.assertEqual(LiveAdapter(self.plan, self.root, ScriptedHost(), run_id="r1").scratch_unit,
                         "llm-eval-scratch-r1.service")
        with self.assertRaises(ValueError):
            LiveAdapter(self.plan, self.root, ScriptedHost(), run_id="../x")

    def test_mixed_case_allocation_marker_is_recognised(self):  # LLE-003
        adapter = self.adapter(oom_above=98304, oom_text="ggml_cuda: cudaMalloc failed\n")
        adapter.snapshot()
        adapter.quiesce()
        with self.assertRaises(CapacityError):
            adapter.load(self.plan["models"][0], 131072)

    def test_request_time_oom_is_capacity_not_a_retry(self):  # LLE-003
        adapter = self.adapter(request_oom_above=98304)
        adapter.snapshot()
        adapter.quiesce()
        adapter.load(self.plan["models"][0], 131072)
        with self.assertRaises(CapacityError):
            adapter.request({"kind": "recall", "prompt": "x " * 600, "max_output_tokens": 8})

    def test_campaign_steps_down_after_a_request_time_oom(self):  # LLE-003
        self.host = ScriptedHost(request_oom_above=114688)
        adapter = LiveAdapter(self.plan, self.root / "runs/live/server-logs", self.host, run_id="live")
        result = execute(self.plan, self.root / "runs", "live", adapter=adapter, lock_path=self.root / "lock")
        recall = result["sections"]["tiny__long_context_recall"]
        self.assertEqual(recall["verified_context"], 114688)
        self.assertIn("capacity_fail", recall["descent"][0])
        self.assertEqual(result["status"], "complete")

    def test_resume_rejects_a_different_plan(self):  # LLE-004
        from test_runner import plan_with
        plan = plan_with()
        execute(plan, self.root / "r", "bound")
        other = plan_with({"free_mib": 20000})
        with self.assertRaisesRegex(ValueError, "original plan"):
            execute(other, self.root / "r", "bound", resume=True)

    def test_failed_load_must_still_return_memory(self):  # LLE-005
        adapter = self.adapter(oom_above=98304, stuck_after_failed_load=True)
        adapter.snapshot()
        adapter.quiesce()
        with self.assertRaisesRegex(InfrastructureError, "return to baseline"):
            adapter.load(self.plan["models"][0], 131072)

    def test_resume_before_smoke_loads_the_scratch_server(self):  # LLE-006
        self.host = ScriptedHost()
        adapter = LiveAdapter(self.plan, self.root / "runs/live/server-logs", self.host, run_id="live")
        execute(self.plan, self.root / "runs", "live", adapter=adapter, lock_path=self.root / "lock")
        directory = self.root / "runs/live"
        state = json.loads((directory / "state.json").read_text())
        for name in [k for k in state["sections"] if "smoke" in k or k.endswith("__teardown")]:
            (directory / "sections" / f"{name}.json").unlink()
            del state["sections"][name]
        state["status"] = "partial"
        (directory / "state.json").write_text(json.dumps(state))
        self.host = ScriptedHost()
        adapter = LiveAdapter(self.plan, directory / "server-logs", self.host, run_id="live")
        result = execute(self.plan, self.root / "runs", "live", resume=True, adapter=adapter,
                         lock_path=self.root / "lock")
        self.assertEqual(result["sections"]["tiny__smoke"]["status"], "pass")
        self.assertEqual(result["status"], "complete")

    def test_smoke_wants_the_word_ok_not_a_substring(self):  # LLE-007
        from llm_eval.runner import grade
        smoke = {"kind": "smoke"}
        self.assertTrue(grade(smoke, {"content": " OK. "})["pass"])
        self.assertFalse(grade(smoke, {"content": "book a table"})["pass"])
        self.assertFalse(grade(smoke, {"content": "not ok"})["pass"])

    def big(self):
        return {"kind": "recall", "prompt": "x " * 600, "max_output_tokens": 8}

    def test_a_dead_server_still_owes_its_memory(self):  # RT-001
        adapter = self.adapter(request_oom_above=98304, stuck_after_request_oom=True)
        adapter.snapshot()
        adapter.quiesce()
        adapter.load(self.plan["models"][0], 131072)
        with self.assertRaises(CapacityError):
            adapter.request(self.big())
        for _ in range(2):  # a failed check is not forgotten by the next cleanup
            with self.assertRaisesRegex(InfrastructureError, "return to baseline"):
                adapter.stop()
        with self.assertRaisesRegex(InfrastructureError, "not cleaned up"):
            adapter.load(self.plan["models"][0], 65536)

    def test_restore_does_not_pass_while_memory_is_held(self):  # RT-001
        self.host = ScriptedHost(request_oom_above=114688, stuck_after_request_oom=True)
        adapter = LiveAdapter(self.plan, self.root / "runs/live/server-logs", self.host, run_id="live")
        result = execute(self.plan, self.root / "runs", "live", adapter=adapter, lock_path=self.root / "lock")
        self.assertEqual(result["status"], "restore_failed")
        self.assertTrue(result["restoration_required"])

    def test_a_resumed_adapter_adopts_the_memory_obligation(self):  # RT-001
        crashed = self.adapter(request_oom_above=98304, stuck_after_request_oom=True)
        snap = crashed.snapshot()
        snap.update(crashed.quiesce())
        crashed.load(self.plan["models"][0], 131072)
        with self.assertRaises(CapacityError):
            crashed.request(self.big())
        resumed = LiveAdapter(self.plan, self.root / "logs", self.host, run_id="t")
        resumed.adopt(snap)
        with self.assertRaisesRegex(InfrastructureError, "return to baseline"):
            resumed.stop()

    def test_adopt_skips_the_gate_once_production_is_back(self):  # RT-001
        adapter = self.adapter()
        snap = adapter.snapshot()
        snap.update(adapter.quiesce())
        adapter.restore(snap)
        resumed = LiveAdapter(self.plan, self.root / "logs", self.host, run_id="t")
        resumed.adopt(snap)
        self.assertFalse(resumed.owes_memory)

    def test_run_id_binds_adapter_and_snapshot(self):  # RT-003
        with self.assertRaises(TypeError):
            LiveAdapter(self.plan, self.root, ScriptedHost())
        adapter = LiveAdapter(self.plan, self.root / "l", ScriptedHost(), run_id="other")
        with self.assertRaisesRegex(ValueError, "different run ID"):
            execute(self.plan, self.root / "runs", "live", adapter=adapter, lock_path=self.root / "lock")
        snap = self.adapter().snapshot()
        self.assertEqual(snap["scratch_unit"], "llm-eval-scratch-t.service")
        with self.assertRaisesRegex(ValueError, "does not match"):
            LiveAdapter(self.plan, self.root, ScriptedHost(), run_id="u").adopt(snap)

    def test_oom_then_timeout_is_capacity_and_a_plain_timeout_is_not(self):  # RT-004
        adapter = self.adapter(request_oom_above=98304, request_oom_mode="timeout")
        adapter.snapshot()
        adapter.quiesce()
        adapter.load(self.plan["models"][0], 131072)
        with self.assertRaises(CapacityError):
            adapter.request(self.big())
        adapter.stop()
        adapter.load(self.plan["models"][0], 65536)
        original = self.host.http
        def hang(*a, **k):
            if a[2] == "/v1/chat/completions":
                raise TimeoutError("slow")
            return original(*a, **k)
        self.host.http = hang
        with self.assertRaises(TimeoutError):
            adapter.request(self.big())

    def test_exec_identity_ignores_runtime_fields(self):  # hardening
        moved = EXEC.replace("pid=0", "pid=99").replace("[n/a]", "[Tue 2026]")
        self.assertEqual(exec_identity(moved), exec_identity(EXEC))
        self.assertNotEqual(exec_identity(EXEC.replace("--models-max 1", "--models-max 2")), exec_identity(EXEC))


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = plan_for(self.root)

    def campaign(self, **host):
        self.host = ScriptedHost(**host)
        adapter = LiveAdapter(self.plan, self.root / "runs/live/server-logs", self.host, run_id="live")
        return execute(self.plan, self.root / "runs", "live", adapter=adapter, lock_path=self.root / "lock")

    def test_workload_seam_uses_only_the_loopback_scratch_server(self):
        self.host = ScriptedHost()
        adapter = LiveAdapter(self.plan, self.root / "logs", self.host, run_id="live")
        adapter.snapshot()
        adapter.quiesce()
        adapter.load(self.plan["models"][0], 65536)
        port = adapter.endpoint()
        self.assertEqual(port, adapter.active["port"])
        adapter.check_alive()
        self.host.server = None  # the scratch server died during a task
        with self.assertRaises(InfrastructureError):
            adapter.check_alive()

    def test_full_live_campaign_against_a_scripted_machine(self):
        result = self.campaign()
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["restoration_required"])
        self.assertEqual(result["sections"]["tiny__long_context_recall"]["verified_context"], 131072)
        self.assertEqual(result["sections"]["tiny__structured_tools"]["passed"], 10)
        self.assertEqual(result["last_restore"]["four_token_probe"], True)
        self.assertTrue(self.host.unit_active)

    def test_smoke_runs_and_text_only_profiles_skip_vision(self):
        result = self.campaign()
        saved = json.loads((self.root / "runs/live/responses/tiny__trial-smoke-1-at-131072.json").read_text())
        self.assertEqual((saved["reasoning"], saved["finish_reason"]), ("thinking", "stop"))
        self.assertEqual(result["sections"]["tiny__smoke"]["status"], "pass")
        self.assertEqual(result["sections"]["tiny__vision"]["status"], "skipped")
        self.assertFalse(getattr(self.host, "saw_image", False))

    def test_vision_profile_sends_a_png_and_is_graded(self):
        self.plan = plan_for(self.root / "v", vision=True)
        result = self.campaign()
        self.assertEqual(result["sections"]["tiny__vision"]["status"], "pass")
        self.assertTrue(self.host.saw_image)

    def test_oom_at_the_top_rung_falls_back_and_still_restores(self):
        result = self.campaign(oom_above=114688)
        qualification = result["sections"]["tiny__context_qualification"]
        self.assertEqual(qualification["selected_context"], 114688)
        self.assertEqual(qualification["status"], "capacity_fallback")
        self.assertTrue(self.host.unit_active)

    def test_unknown_gpu_workload_blocks_before_production_is_touched(self):
        result = self.campaign(procs=[{"pid": 9, "name": "ComfyUI", "mib": 20000}])
        self.assertEqual(result["status"], "blocked")
        self.assertTrue(self.host.unit_active)
        self.assertFalse([c for c in self.host.calls if c[2] in ("stop", "start")])

    def test_failed_probe_leaves_the_restore_marker_set(self):
        result = self.campaign(refuse_probe=True)
        self.assertEqual(result["status"], "restore_failed")
        self.assertTrue(result["restoration_required"])


if __name__ == "__main__":
    unittest.main()
