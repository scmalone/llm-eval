"""The fixes the 2026-10-30 corpus review asked for, each with the run or code path that found it."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from llm_eval import runner
from llm_eval.metrics import scoring
from llm_eval.strata import StrataAdapter, manifest as strata_manifest
from llm_eval.workload import graders, replicas

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "workload"


def _task(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _answer_check(task):
    return next(c for c in task["checks"] if c["type"] == "answer")


class PiContextTests(unittest.TestCase):
    """2026-11-01-pi-262144 crashed with KeyError 'provider': the runner assumed opencode's config shape."""

    def test_the_pi_models_json_gets_the_loaded_context(self):
        config = {"providers": {"local-llama": {"models": [{"id": "strata-iq3s", "contextWindow": 262144}]}}}
        out = runner._with_context(config, "local-llama/strata-iq3s", 131072)
        self.assertEqual(out["providers"]["local-llama"]["models"][0]["contextWindow"], 131072)
        self.assertEqual(config["providers"]["local-llama"]["models"][0]["contextWindow"], 262144)

    def test_the_opencode_config_still_gets_limit_context(self):
        config = {"provider": {"local-llama": {"models": {"m": {"limit": {"context": 262144}}}}}}
        out = runner._with_context(config, "local-llama/m", 131072)
        self.assertEqual(out["provider"]["local-llama"]["models"]["m"]["limit"]["context"], 131072)


class FailedCheckDetailTests(unittest.TestCase):
    """Failed checks reached sections and the DB as ``passed=0, missing: []``: the reason was stripped."""

    def test_a_failed_check_keeps_its_whole_payload(self):
        result = {"grading": {"checks": [
            {"type": "answer", "pass": False, "missing": [], "forbidden_present": ["(?i)not overbuilt"]},
            {"type": "diff_scope", "pass": True, "denied": [], "outside_allow": [], "changed": ["a.py"]}]},
            "stop": "done", "turns": [1], "clarifications": 0, "tests_pasted": 0, "metrics": {}}
        checks = runner._trial_summary(result, "/out")["checks"]
        self.assertEqual(checks[0]["forbidden_present"], ["(?i)not overbuilt"])
        self.assertNotIn("changed", checks[1])


class TruncationScoringTests(unittest.TestCase):
    """An output_cap or wall_cap trial is ingested as ``incomplete`` under a quality_fail status; scoring used
    to raise on it, and the scorecard view dropped it from the denominator."""

    def test_incomplete_is_a_failure_class_of_quality_fail(self):
        self.assertIn("incomplete", scoring._failure_classes("quality_fail"))
        self.assertNotIn("incomplete", scoring._failure_classes("pass"))

    def test_the_scorecard_counts_incomplete_in_its_denominator(self):
        from llm_eval.metrics import schema
        source = Path(schema.__file__).read_text()
        self.assertIn("IN ('model', 'timeout', 'incomplete')", source)


class ReproducibleReplicaTests(unittest.TestCase):
    """12 of 18 tasks changed identity between the two cohort arms with byte-identical files: tar mtimes."""

    def test_two_builds_of_one_replica_are_the_same_bytes(self):
        task = _task("ops-fans-gpu-hog")
        digests = []
        with tempfile.TemporaryDirectory() as tmp:
            for n in (1, 2):
                time.sleep(1.1)  # whole-second tar mtimes: the second build must land in a later second
                replicas.build(task, Path(tmp) / str(n))
                digests.append(hashlib.sha256((Path(tmp) / str(n) / "home.tar").read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])


class StrataEnvTests(unittest.TestCase):
    """manifest() hard-coded STRATA_ROUTE_TAIL_SKIP=0, which overrides the config at launch: a tail-skip knob
    arm would have run the baseline."""

    def test_the_launch_configs_env_reaches_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "c.json"
            config.write_text(json.dumps({"exe": "x", "args": [], "env": {"STRATA_ROUTE_TAIL_SKIP": "1"}}))
            preset = Path(tmp) / "p.ini"
            preset.write_text("[restore-me]\nmodel = /m.gguf\nctx-size = 65536\n")
            out = strata_manifest(config, "/py", "/srv.py", "strata-iq3s", 262144, preset, "restore-me")
        env = out["models"][0]["settings"]["env"]
        self.assertEqual(env["STRATA_ROUTE_TAIL_SKIP"], "1")
        self.assertEqual(env["STRATA_KVA"], "0")  # defaults still fill what the config leaves out


class EngineLogTests(unittest.TestCase):
    """The cache and version lines are in the engine's own log; the parent read only the server log, so every
    0.1.54 section lacked the cache report and the DB recorded the stale BUILD.json version (0.1.45)."""

    def test_memory_and_version_come_from_the_engine_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = StrataAdapter.__new__(StrataAdapter)
            adapter.log_dir = Path(tmp)
            adapter._loading_context, adapter._loading_model = 262144, "strata-iq3s"
            log = Path(tmp) / "strata-strata-iq3s-262144" / "engine.log"
            log.parent.mkdir()
            log.write_text("strata generate: expert cache auto: 12.30 GiB free, 2048 MiB reserved (+218 MiB for the "
                           "draft head, +256 MiB for what is allocated after the cache) -> 3966 slots\n"
                           "strata generate: expert cache 5169 slots, 9.83 GiB of VRAM; policy is\n"
                           "strata generate: session is up (engine 0.1.54)\n")
            adapter.kv_mib("server log without the cache line")
        self.assertEqual(adapter.memory["cache_slots"], 5169)
        self.assertEqual(adapter.memory["reserve_mib"], 2048)
        self.assertEqual(adapter.memory["engine_version"], "0.1.54")

    def test_ingest_prefers_the_engine_version_and_flags_a_stale_build_file(self):
        from llm_eval.metrics.ingest import _runtime
        sections = {"strata-iq3s__anchor_load": {"attempts": [{"memory": {"engine_version": "0.1.54"}}]}}
        runtime = _runtime({"runtime": {"engine_build": {"version": "0.1.45"}}}, sections)
        self.assertEqual(runtime["engine_version"], "0.1.54")
        self.assertTrue(runtime["engine_build_stale"])


class GraderFalseNegativeTests(unittest.TestCase):
    """Three coverage answers at 262K were right and graded wrong (2026-11-01-coverage-262144)."""

    def answer(self, name, text):
        return graders.check_answer(_answer_check(_task(name)), {"answer": text}, None)["pass"]

    def test_a_rejected_option_does_not_fail_models_to_keep(self):
        text = ("A. Keep only `muse-glimmer-30b-Q4_K_XL.gguf` and `qwen3.8-27b-Q4_K_XL.gguf`. Recommended\n"
                "C. Keep the smallest quants (Q3_K_S, Q3_K_M) for faster CPU-only inference.")
        self.assertTrue(self.answer("advisory-models-to-keep", text))
        # the recommendation itself is still policed, by the style check's recommended_none
        style = next(c for c in _task("advisory-models-to-keep")["checks"] if c["type"] == "style")
        self.assertTrue(style["recommended_none"])

    def test_a_part_called_not_overbuilt_does_not_fail_the_verdict(self):
        text = ("Yes, in one place. Evidence from the tree:\n- `live.py`: speculative.\n"
                "- `preflight.py`: on the critical path this week. Not overbuilt.\n"
                "A. Park `live.py` and keep the runner (Recommended)")
        self.assertTrue(self.answer("advisory-overbuilt", text))
        self.assertFalse(self.answer("advisory-overbuilt", "Not overbuilt. The runner is fine as is."))

    def test_the_open_range_names_lle_002(self):
        text = ("the 2026-10-02 `/redteam` left LLE-001 to LLE-007 open. Your turn is the one decision that "
                "gates the fixes")
        self.assertTrue(self.answer("resume-llm-eval-owner-decision", text))
        self.assertFalse(self.answer("resume-llm-eval-owner-decision", "You were writing the README."))


class AlignmentTests(unittest.TestCase):
    """2026-10-30: llama.cpp profiles and Strata are measured through the same pi harness and the same capture."""

    def test_a_preset_becomes_the_pinned_request(self):
        from llm_eval.preflight import preset_request
        request = preset_request({"temperature": "0.6", "top-p": "0.95", "top-k": "20", "min-p": "0.00",
                                  "ctx-size": "131072", "jinja": "true"})
        self.assertEqual(request, {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0})

    def test_each_runtime_names_where_its_request_lives(self):
        from llm_eval.live import LiveAdapter
        live = LiveAdapter.__new__(LiveAdapter)
        strata = StrataAdapter.__new__(StrataAdapter)
        self.assertEqual(live.pinned_request({"request": {"temperature": 1.0}, "settings": {"temp": "1.0"}}),
                         {"temperature": 1.0})
        self.assertEqual(strata.pinned_request({"settings": {"request": {"top_k": 20}}}), {"top_k": 20})
        self.assertIsNone(live.served_model_name({}))

    def test_both_adapters_share_one_capture_backed_driver(self):
        from llm_eval.live import LiveAdapter
        self.assertIs(StrataAdapter.workload_driver, LiveAdapter.workload_driver)
        self.assertEqual(StrataAdapter.STALL_SECONDS, 300)
        self.assertIsNone(LiveAdapter.STALL_SECONDS)  # driver.STALL_SECONDS until llama.cpp progress is proven

    def test_pi_sends_the_pinned_sampling_for_either_runtime(self):
        from llm_eval.workload import pi_config
        for model in ({"request": {"temperature": 0.6}}, {"settings": {"request": {"temperature": 0.6}}}):
            request = model.get("request") or (model.get("settings") or {}).get("request") or {}
            models, _, _ = pi_config.build("m", 131072, 16384, request)
            self.assertEqual(models["providers"]["local-llama"]["models"][0]["samplingParams"]["temperature"], 0.6)


class PiBinaryTests(unittest.TestCase):
    """The first real pi trial exited at once: 'pi' was not found inside the box."""

    def test_the_pinned_binary_lives_under_the_home_the_driver_mounts(self):
        from llm_eval.workload.prepare import PI_BINARY
        from llm_eval.workload.pi_driver import PI_HOME
        self.assertEqual(PI_BINARY.parents[1], PI_HOME)
        self.assertEqual(PI_BINARY.relative_to(PI_HOME).as_posix(), "bin/pi")

    def test_the_block_carries_the_mount_point_that_plan_compilation_would_lose(self):
        # plan.compile resolves "binary" to the bundle's cli.js; its grandparent is not the npm prefix
        from llm_eval.workload.pi_driver import PI_HOME
        resolved = PI_HOME / "lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
        self.assertNotEqual(resolved.parents[1], PI_HOME)
        source = (Path(__file__).resolve().parents[1] / "llm_eval/live.py").read_text()
        self.assertIn('binary_home=Path(workload.get("binary_home") or PI_HOME)', source)


if __name__ == "__main__":
    unittest.main()
