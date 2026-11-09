"""speed_sweep: the search over one preset against a scripted machine (memory and speed are formulas here)."""
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

TOOL = Path(__file__).resolve().parents[1] / "tools" / "speed_sweep.py"
spec = importlib.util.spec_from_file_location("speed_sweep", TOOL)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)

DENSE = {"model": "/m/dense.gguf", "ctx-size": "131072", "ubatch-size": "512", "batch-size": "2048",
         "cache-type-k": "q4_0", "cache-type-v": "q4_0", "spec-type": "draft-mtp", "spec-draft-n-max": "2"}
MOE = {"model": "/m/moe.gguf", "ctx-size": "262144", "ubatch-size": "512", "batch-size": "2048",
       "cache-type-k": "q8_0", "cache-type-v": "q8_0", "n-cpu-moe": "10"}


class Machine:
    """Free memory = budget - KV - compute - expert layers on the GPU; speed from ``speed(settings)`` seconds."""

    def __init__(self, budget=22000, fixed=16000, kv_per_ctx=0.004, per_layer=400, layers=40, speed=None,
                 accept=0.8, trained=262144, experts=0):
        self.budget, self.fixed, self.kv_per_ctx, self.per_layer, self.layers = budget, fixed, kv_per_ctx, per_layer, layers
        self.speed = speed or (lambda s: 10.0)
        self.accept, self.trained, self.experts, self.seen, self.quick = accept, trained, experts, [], []

    def __call__(self, name, s, quick=False):
        self.seen.append(dict(s))
        self.quick.append(quick)
        bits = sweep.KV_BITS[s.get("cache-type-k", "f16")] / 16
        kv = int(s["ctx-size"]) * self.kv_per_ctx * bits * 4
        compute = int(s.get("ubatch-size", 512)) * 1.0
        moe = int(s.get("n-cpu-moe", 0))
        free = int(self.budget - self.fixed - kv - compute - (self.layers - moe) * self.per_layer * (1 if self.experts else 0))
        result = {"preset": name, "settings": s, "infra": False, "min_free_mib": free,
                  "headroom_mib": free - sweep.FLOOR_MIB, "score": self.speed(s), "quick_score": self.speed(s),
                  "quick": quick,
                  "accept": self.accept if "spec-type" in s else None,
                  "buffers": {"kv_mib": kv, "compute_mib": compute, "cpu_model_mib": moe * self.per_layer},
                  "info": {"n_ctx_train": self.trained, "n_expert": self.experts}}
        result["fits"] = free >= sweep.FLOOR_MIB
        result["why"] = f"free {free}"
        return result


def run(machine, settings, **kw):
    return sweep.Sweep(machine, say=lambda m: None).preset("p", dict(settings), **kw)


def rows(outcome, knob):
    return [(r["value"], r["outcome"]) for r in outcome["trail"] if r["knob"] == knob]


class SearchTests(unittest.TestCase):
    def test_ubatch_climbs_one_step_at_a_time_while_each_step_is_faster(self):
        speed = {512: 10.0, 1024: 9.5, 2048: 9.0, 4096: 8.9}  # 4096 is only 1% faster than 2048
        m = Machine(budget=30000, speed=lambda s: speed[int(s["ubatch-size"])])
        out = run(m, {**DENSE, "ctx-size": "262144"}, speed_first=True)
        self.assertEqual(rows(out, "ubatch"), [(1024, "kept"), (2048, "kept"), (4096, "no gain")])
        self.assertEqual(out["after"]["ubatch-size"], "2048")

    def test_a_step_that_does_not_fit_ends_the_climb(self):
        speed = {512: 10.0, 1024: 9.0, 2048: 8.0, 4096: 7.0}
        m = Machine(budget=21000, speed=lambda s: speed[int(s["ubatch-size"])], trained=131072)
        out = run(m, DENSE)
        self.assertEqual(rows(out, "ubatch"), [(1024, "kept"), (2048, "kept"), (4096, "failed")])
        self.assertEqual(out["after"]["ubatch-size"], "2048")

    def test_cpu_moe_moves_layers_back_one_at_a_time_until_one_fails(self):
        m = Machine(budget=20600, fixed=2336, per_layer=400, layers=40, experts=128,
                    speed=lambda s: 5 + int(s["n-cpu-moe"]) * 0.5)
        real = m.__call__
        def measure(name, s, quick=False):
            r = real(name, s, quick)
            if int(s.get("n-cpu-moe", 0)) < 8:
                r.update(fits=False, why="out of memory")
            return r
        out = sweep.Sweep(measure, say=lambda x: None).preset("p", dict(MOE), speed_first=True)
        self.assertEqual(rows(out, "cpu moe"), [(9, "kept"), (8, "kept"), (7, "failed")])
        self.assertEqual(out["after"]["n-cpu-moe"], "8")

    def test_context_climbs_to_the_trained_maximum_as_a_proposal_and_kv_never_climbs(self):
        m = Machine(budget=24000, fixed=14000, trained=262144)
        out = run(m, DENSE)
        self.assertEqual(rows(out, "context"), [(163840, "kept"), (196608, "kept"), (262144, "kept")])
        self.assertEqual(rows(out, "kv"), [])
        self.assertTrue(all(s["cache-type-k"] == DENSE["cache-type-k"] for s in m.seen))
        speed, proposals = sweep.changes(out)
        self.assertIn("ctx-size", proposals)
        self.assertNotIn("ctx-size", speed)

    def test_speed_first_spends_headroom_on_speed_before_context(self):
        speed = {512: 10.0, 1024: 8.0, 2048: 7.0, 4096: 6.0}
        m = Machine(budget=21500, fixed=14000, speed=lambda s: speed[int(s["ubatch-size"])])
        first = run(m, DENSE, speed_first=True)
        tried = [r["knob"] for r in first["trail"] if r["outcome"] != "skipped" or r["value"] != "-"]
        self.assertEqual(tried[1], "drafter")
        self.assertEqual(first["after"]["ubatch-size"], "4096")

    def test_a_preset_that_does_not_fit_steps_context_down_first_and_holds_it(self):
        m = Machine(budget=19600, fixed=4336, per_layer=400, layers=40, experts=128)  # 500 MiB short at 262K
        out = run(m, MOE)
        self.assertEqual(rows(out, "rescue"), [("ctx-size 196608", "fits")])
        self.assertTrue(out["final"]["fits"])
        self.assertEqual(rows(out, "context"), [("-", "skipped")])  # not climbed back into the memory it freed
        self.assertTrue(all(s["cache-type-k"] == "q8_0" for s in m.seen))

    def test_context_stops_at_131072_then_cpu_moe_layers(self):
        m = Machine(budget=18200, fixed=4336, per_layer=400, layers=40, experts=128)
        out = run(m, MOE)
        self.assertEqual(rows(out, "rescue"), [("ctx-size 196608", "failed"), ("ctx-size 163840", "failed"),
                                               ("ctx-size 131072", "failed"), ("n-cpu-moe 11", "failed"),
                                               ("n-cpu-moe 12", "fits")])
        self.assertTrue(all(int(s["ctx-size"]) >= sweep.CONTEXT_FLOOR for s in m.seen))
        self.assertEqual(out["after"]["ctx-size"], "131072")

    def test_the_step_down_order_ends_with_kv_and_q4_only_last(self):
        cfg = {**MOE, "cache-type-k": "f16", "cache-type-v": "f16", "n-cpu-moe": "6"}
        self.assertEqual([label for label, _ in sweep.rescues(cfg, {})],
                         ["ctx-size 196608", "ctx-size 163840", "ctx-size 131072", "n-cpu-moe 7", "n-cpu-moe 8",
                          "n-cpu-moe 9", "n-cpu-moe 10", "ubatch-size 256", "cache-type q8_0", "cache-type q4_0"])
        self.assertEqual([label for label, _ in sweep.rescues({**DENSE, "ctx-size": "65536"}, {})],
                         ["ubatch-size 256"])  # already q4_0 and under the context floor

    def test_the_model_file_header_fills_facts_the_server_log_left_out(self):
        facts = lambda path: {"n_expert": 128, "n_ctx_train": 262144}
        info = sweep.model_info({"model": "/m/x.gguf"}, {"info": {"n_expert": None, "n_ctx_train": None}}, facts)
        self.assertEqual(info, {"n_expert": 128, "n_ctx_train": 262144})
        no_moe_key = {k: v for k, v in MOE.items() if k != "n-cpu-moe"}
        self.assertIn("n-cpu-moe 1", [label for label, _ in sweep.rescues(no_moe_key, info)])
        self.assertEqual(sweep.model_info({"model": "/missing.gguf"}, None), {})

    def test_kv_drops_to_q4_only_after_ubatch_and_never_shrinks_gemma_ubatch(self):
        q8 = {**DENSE, "cache-type-k": "q8_0", "cache-type-v": "q8_0"}
        out = run(Machine(budget=18000), q8)
        self.assertEqual(rows(out, "rescue"), [("ubatch-size 256", "failed"), ("cache-type q4_0", "fits")])
        m = Machine(budget=18200)
        gemma = sweep.Sweep(m, say=lambda s: None).preset("gemma-4-26b-a4b-it", dict(q8))
        self.assertEqual(rows(gemma, "rescue"), [("cache-type q4_0", "fits")])  # no ubatch step on Gemma

    def test_nothing_fits_is_reported_for_the_owner(self):
        out = run(Machine(budget=12000), DENSE)
        self.assertEqual(out["status"], "does not fit")
        self.assertEqual(rows(out, "rescue")[-1][1], "gave up")

    def test_the_drafter_off_arm_runs_only_on_low_acceptance_and_wins_ties(self):
        self.assertEqual(rows(run(Machine(budget=30000, accept=0.9), DENSE, speed_first=True), "drafter"),
                         [("off", "skipped")])
        low = Machine(budget=30000, accept=0.3, speed=lambda s: 10.0 if "spec-type" in s else 10.2)
        out = run(low, DENSE, speed_first=True)
        self.assertEqual(rows(out, "drafter"), [("off", "kept")])  # 2% slower: inside the tie band
        self.assertNotIn("spec-type", out["after"])
        self.assertEqual(rows(out, "draft depth"), [("-", "skipped")])

    def test_mtp_depth_starts_at_three_and_climbs_while_faster(self):
        speed = {2: 10.0, 3: 9.0, 4: 8.5, 5: 8.4}
        m = Machine(budget=30000, accept=0.9, speed=lambda s: speed.get(int(s.get("spec-draft-n-max", 0)), 12.0))
        out = run(m, DENSE, speed_first=True)
        self.assertEqual(rows(out, "draft depth"), [(3, "kept"), (4, "kept"), (5, "no gain")])
        self.assertEqual(out["after"]["spec-draft-n-max"], "4")

    def test_a_start_that_ties_keeps_the_preset_depth_and_tries_lower(self):
        speed = {1: 8.0, 2: 10.0, 3: 10.1}
        m = Machine(budget=30000, accept=0.9, speed=lambda s: speed.get(int(s.get("spec-draft-n-max", 0)), 12.0))
        out = run(m, DENSE, speed_first=True)
        self.assertEqual(rows(out, "draft depth"), [(3, "no gain"), (1, "kept")])
        self.assertEqual(out["after"]["spec-draft-n-max"], "1")

    def test_decode_knobs_are_chosen_on_quick_measurements_and_the_winner_gets_the_full_ladder(self):
        speed = {2: 10.0, 3: 9.0, 4: 8.5, 5: 8.4}
        m = Machine(budget=30000, accept=0.9, speed=lambda s: speed.get(int(s.get("spec-draft-n-max", 0)), 12.0))
        out = run(m, DENSE, speed_first=True)
        depth = [(s.get("spec-draft-n-max"), q) for s, q in zip(m.seen, m.quick)]
        self.assertEqual(depth[:5], [("2", False), ("3", True), ("4", True), ("5", True), ("4", False)])
        self.assertEqual(rows(out, "verify"), [("full ladder", "fits")])
        self.assertFalse(out["final"]["quick"])

    def test_a_winner_that_fails_the_full_ladder_falls_back_to_the_last_verified_setting(self):
        speed = {2: 10.0, 3: 9.0}
        m = Machine(budget=30000, accept=0.9, speed=lambda s: speed.get(int(s.get("spec-draft-n-max", 0)), 12.0))
        real = m.__call__
        def measure(name, s, quick=False):
            r = real(name, s, quick)
            if s.get("spec-draft-n-max") == "3" and not quick:
                r.update(fits=False, why="out of memory deep")
            return r
        out = sweep.Sweep(measure, say=lambda x: None).preset("p", dict(DENSE), speed_first=True)
        self.assertEqual(rows(out, "verify"), [("full ladder", "failed")])
        self.assertEqual(out["after"]["spec-draft-n-max"], "2")

    def test_the_noise_check_measures_the_same_setting_again_under_its_own_key(self):
        seen = []
        def measure(name, s, quick=False, key=None):
            seen.append(key)
            score = 10.0 + len(seen) * 0.1
            return {"fits": True, "why": "ok", "score": score, "quick_score": score, "headroom_mib": 500,
                    "min_free_mib": 1524, "prefill_tps": 1000.0, "decode_tps": 50.0, "short_decode_tps": 55.0,
                    "deep_decode_tps": 45.0, "accept": None}
        with tempfile.TemporaryDirectory() as tmp:
            cached = sweep.Cached(measure, Path(tmp) / "results.jsonl", lambda m: None)
            out = sweep.Sweep(cached, say=lambda m: None).preset("p", dict(DENSE), baseline_only=True, repeats=3)
            self.assertEqual(len(out["repeats"]), 3)
            self.assertEqual(seen[0], None)
            self.assertTrue(seen[1].endswith("-r1") and seen[2].endswith("-r2"))
            sweep.report([out], Path(tmp), 22600)
            text = (Path(tmp) / "report.md").read_text()
            self.assertIn("Run-to-run noise", text)
            self.assertIn("| p | 3 | 10.1 to 10.3 | 2.0% |", text)
        with self.assertRaises(SystemExit):
            sweep.main(["--repeat", "3"])

    def test_baseline_only_measures_once(self):
        m = Machine(budget=30000)
        out = run(m, DENSE, baseline_only=True)
        self.assertEqual((out["status"], len(m.seen)), ("fits", 1))


class RecordTests(unittest.TestCase):
    def test_results_are_reused_and_infrastructure_errors_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, calls = Path(tmp) / "results.jsonl", []
            def measure(name, s, quick=False):
                calls.append(s)
                return {"fits": True, "infra": len(calls) == 1, "why": "x"}
            cached = sweep.Cached(measure, path, lambda m: None)
            for _ in range(3):
                cached("p", DENSE)
            self.assertEqual(len(calls), 2)
            sweep.Cached(measure, path, lambda m: None)("p", DENSE)
            self.assertEqual(len(calls), 2)

    def test_buffers_and_model_facts_come_from_the_server_log(self):
        log = ("print_info: arch                  = qwen35moe\nprint_info: n_ctx_train           = 262144\n"
               "print_info: n_layer               = 40\nprint_info: n_expert              = 256\n"
               "load_tensors:   CPU_Mapped model buffer size =  4100.50 MiB\n"
               "load_tensors:        CUDA0 model buffer size = 15000.00 MiB\n"
               "llama_kv_cache:      CUDA0 KV buffer size =  2720.00 MiB\n"
               "llama_kv_cache:      CUDA0 KV buffer size =    80.00 MiB\n"
               "llama_memory_recurrent:      CUDA0 RS buffer size =   62.81 MiB\n"
               "llama_context:      CUDA0 compute buffer size =   980.25 MiB\n"
               "llama_context:  CUDA_Host compute buffer size =   520.00 MiB\n")
        buffers, info = sweep.parse_buffers(log)
        self.assertEqual((buffers["kv_mib"], buffers["compute_mib"], buffers["cpu_model_mib"], buffers["rs_mib"]),
                         (2800.0, 980.2, 4100.5, 62.8))
        self.assertEqual(buffers["compute_by_device"], {"CUDA0": 980.2, "CUDA_Host": 520.0})
        self.assertEqual((info["n_ctx_train"], info["n_expert"], info["n_layer"], info["arch"], info["oom"]),
                         (262144, 256, 40, "qwen35moe", False))

    def test_the_report_gives_preset_lines_for_speed_and_names_proposals(self):
        speed = {512: 10.0, 1024: 8.0, 2048: 7.9, 4096: 7.8}
        m = Machine(budget=24000, fixed=14000, speed=lambda s: speed[int(s["ubatch-size"])])
        out = run(m, DENSE)
        for r in (out["baseline"], out["final"]):
            r.update(prefill_tps=1000.0, decode_tps=50.0, short_decode_tps=55.0, deep_decode_tps=45.0)
        with tempfile.TemporaryDirectory() as tmp:
            sweep.report([out], Path(tmp), 22600)
            text = (Path(tmp) / "report.md").read_text()
            self.assertIn("ubatch-size = ", text)
            self.assertIn("Proposal (owner): ctx-size 131072 → 262144", text)
            self.assertTrue(json.loads((Path(tmp) / "summary.json").read_text())["outcomes"])


class FakeAdapter:
    """What LiveMeasure uses of LiveAdapter. ``reuse``: the prompt cache carries a prefix (False: hybrid reprocess)."""

    def __init__(self, logs, free=3000, oom=False, reuse=True):
        self.logs, self.baseline, self.oom, self.reuse, self.stopped = logs, 22600, oom, reuse, 0
        self.binary, self.cached, self.prompts = "/bin/llama-server", 0, []
        adapter = self
        class Host:
            @staticmethod
            def gpu_free_mib():
                return free
            @staticmethod
            def http(method, port, path, body=None, timeout=30):
                return adapter.http(path, body)
        self.host = Host()

    def http(self, path, body):
        if path == "/tokenize":
            return 200, {"tokens": list(range(len(body["content"]) // 4))}
        if path == "/apply-template":
            kept = "\n".join(m.get("reasoning_content") or "" for m in body["messages"])
            return 200, {"prompt": f"system\nReasoning strength: xhigh.\n{kept}\nuser"}
        n = len(body["prompt"])
        self.prompts.append(n)
        cache = self.cached if self.reuse else 0
        self.cached = n
        return 200, {"content": "x", "timings": {"cache_n": cache, "prompt_n": n - cache,
                                                  "prompt_ms": (n - cache) / 2.0, "prompt_per_second": 2000.0,
                                                  "predicted_n": 128, "predicted_per_second": 40.0,
                                                  "draft_n": 100, "draft_n_accepted": 75}}

    def endpoint(self):
        return 1

    def load(self, model, ctx):
        self.logs.mkdir(parents=True, exist_ok=True)
        (self.logs / f"{model['id']}-{ctx}-1.log").write_text(
            "n_ctx_train = 131072\nCUDA0 KV buffer size = 1000.00 MiB\nCUDA0 compute buffer size = 500.00 MiB\n")
        if self.oom:
            raise sweep.CapacityError("allocation failure")
        return {"free_mib": 5000, "load_seconds": 12.0}

    def request(self, fixture):
        return {"content": "ok", "timings": {"prompt_per_second": 0.0, "predicted_per_second": 60.0,
                                             "draft_n": 100, "draft_n_accepted": 75}}

    def sample_gpu(self):
        return {"mem_free_mib": self.host.gpu_free_mib(), "power_w": 300.0}

    def sample_host(self):
        return {"load1": 1.0}

    def server_snapshot(self):
        return {"props": {}}

    def _request_failed(self, detail):
        raise sweep.InfrastructureError(detail)

    def stop(self):
        self.stopped += 1


class LiveMeasureTests(unittest.TestCase):
    def measure(self, adapter, tmp, settings=None):
        (Path(tmp) / "corpus.txt").write_text("int x = 1;\n" * 200_000)  # 2.2M chars: 550K fake tokens
        return sweep.LiveMeasure(adapter, Path(tmp))("p", settings or DENSE)

    def test_a_measurement_reads_speed_memory_buffers_and_keeps_the_raw_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = FakeAdapter(Path(tmp) / "logs")
            result = self.measure(adapter, tmp)
            raw = Path(tmp) / "raw" / result["key"]
            self.assertTrue(all((raw / f).exists() for f in ("gpu.jsonl", "host.jsonl", "requests.jsonl",
                                                             "server.loaded.json", "server.end.json")))
            requests = [json.loads(l) for l in (raw / "requests.jsonl").read_text().splitlines()]
            template = json.loads((raw / "template.json").read_text())
        self.assertTrue(template["earlier_reasoning_kept"])
        self.assertEqual(template["reasoning_line"], "Reasoning strength: xhigh.")
        self.assertTrue(result["fits"], result["why"])
        self.assertEqual(adapter.prompts, [32768, 65536, 98304, 117964])  # 25/50/75/90% of 131,072
        self.assertEqual([p["cache_carried"] for p in result["ladder"]], [None, True, True, True])
        self.assertEqual((result["prefill_tps"], result["decode_tps"], result["accept"]), (2000.0, 48.0, 0.75))
        self.assertEqual(result["headroom_mib"], 3000 - sweep.FLOOR_MIB)
        self.assertEqual(result["info"]["n_ctx_train"], 131072)
        self.assertEqual(len(requests), 6)
        self.assertIn("--ctx-size", result["argv"])
        self.assertEqual(adapter.stopped, 1)

    def test_a_ladder_the_cache_does_not_carry_goes_straight_to_the_deepest_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = FakeAdapter(Path(tmp) / "logs", reuse=False)
            result = self.measure(adapter, tmp)
        self.assertEqual(adapter.prompts, [32768, 65536, 117964])
        self.assertFalse(result["cache_carried"])
        self.assertTrue(result["fits"])

    def test_a_quick_measurement_runs_only_the_first_depth_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = FakeAdapter(Path(tmp) / "logs")
            (Path(tmp) / "corpus.txt").write_text("int x = 1;\n" * 200_000)
            result = sweep.LiveMeasure(adapter, Path(tmp))("p", DENSE, quick=True)
        self.assertEqual(adapter.prompts, [32768])
        self.assertTrue(result["quick"] and result["fits"])
        self.assertEqual(result["quick_score"], result["score"])

    def test_an_allocation_failure_is_a_result_and_the_server_is_still_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = FakeAdapter(Path(tmp) / "logs", oom=True)
            result = self.measure(adapter, tmp)
        self.assertFalse(result["fits"] or result["infra"])
        self.assertIn("out of memory", result["why"])
        self.assertEqual(result["info"]["n_ctx_train"], 131072)
        self.assertEqual(adapter.stopped, 1)

    def test_a_floor_breach_stops_the_probe_before_the_ladder(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = FakeAdapter(Path(tmp) / "logs", free=900)
            adapter.sample_gpu = lambda: {"mem_free_mib": 900, "power_w": 300.0}
            with mock.patch.object(sweep, "GPU_SAMPLE_SECONDS", 0.01):
                original = adapter.load
                adapter.load = lambda m, c: (time.sleep(0.1), original(m, c))[1]
                result = self.measure(adapter, tmp)
        self.assertFalse(result["fits"] or result["infra"])
        self.assertIn("under the 1,024 MiB floor", result["why"])
        self.assertEqual(result["stopped_early"], "load")
        self.assertEqual(result["headroom_mib"], 900 - sweep.FLOOR_MIB)
        self.assertEqual(adapter.prompts, [])  # no depth request was sent
        self.assertEqual(adapter.stopped, 1)

    def test_low_free_memory_during_the_run_fails_the_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.measure(FakeAdapter(Path(tmp) / "logs", free=900), tmp)
        self.assertFalse(result["fits"])
        self.assertIn("under the 1,024 MiB floor", result["why"])


class Lifecycle:
    """LiveAdapter's lifecycle calls, recorded."""

    def __init__(self, clients=()):
        self.calls, self.baseline, self.clients = [], None, list(clients)

    def snapshot(self):
        self.calls.append("snapshot")
        return {"unknown_gpu_clients": self.clients, "was_active": True}

    def quiesce(self):
        self.calls.append("quiesce")
        self.baseline = 22600
        return {"free_mib": 22600}

    def environment(self):
        return {"gpu": {}}

    def stop(self):
        self.calls.append("stop")

    def restore(self, snapshot):
        self.calls.append("restore")
        return {"service_matches": True}


class MainTests(unittest.TestCase):
    def main(self, adapter, measure):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(sweep, "LiveAdapter", lambda *a, **k: adapter), \
                mock.patch.object(sweep.inhibit, "sleep_inhibit", lambda *a, **k: contextlib.nullcontext()), \
                mock.patch.object(sweep, "LiveMeasure", lambda *a, **k: measure), \
                mock.patch.object(sweep.signal, "signal"):
            ini = Path(tmp) / "presets.ini"
            ini.write_text("[dense]\n" + "".join(f"{k} = {v}\n" for k, v in DENSE.items()))
            code = sweep.main(["--preset-file", str(ini), "--baseline-only", "--out", tmp])
            report = Path(tmp) / "report.md"
            return code, report.read_text() if report.exists() else ""

    def test_the_default_list_leaves_out_skipped_presets_and_an_explicit_list_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "presets.ini"
            ini.write_text("".join(f"[{name}]\n" + "".join(f"{k} = {v}\n" for k, v in DENSE.items())
                                   for name in ("dense", "gemma-4-31b-it")))
            for argv, expected in (([], ["dense"]), (["--skip", ""], ["dense", "gemma-4-31b-it"]),
                                   (["--presets", "gemma-4-31b-it"], ["gemma-4-31b-it"])):
                with mock.patch.object(sweep, "dry_run") as dry:
                    self.assertEqual(sweep.main(["--preset-file", str(ini), "--dry-run", *argv]), 0)
                self.assertEqual(dry.call_args.args[0], expected, argv)

    def test_an_interrupt_mid_sweep_still_restores_the_router(self):
        adapter = Lifecycle()
        def measure(name, settings, quick=False):
            raise KeyboardInterrupt("SIGTERM")
        code, text = self.main(adapter, measure)
        self.assertEqual(adapter.calls, ["snapshot", "quiesce", "stop", "restore"])
        self.assertIn("Router restore", text)

    def test_another_gpu_program_refuses_before_the_router_is_touched(self):
        adapter = Lifecycle(clients=["123:game (8000 MiB)"])
        code, _ = self.main(adapter, None)
        self.assertEqual((code, adapter.calls), (2, ["snapshot"]))

    def test_a_crash_mid_sweep_restores_the_router_and_then_raises(self):
        adapter = Lifecycle()
        def measure(name, settings, quick=False):
            raise RuntimeError("memory did not come back")
        with self.assertRaises(RuntimeError):
            self.main(adapter, measure)
        self.assertEqual(adapter.calls[-1], "restore")


if __name__ == "__main__":
    unittest.main()
