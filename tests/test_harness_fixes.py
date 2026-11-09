"""The fixes the two cohort runs asked for, each with the case that found it."""
import json
import tempfile
import unittest
from pathlib import Path

from llm_eval.strata import StrataAdapter
from llm_eval.workload import session


class TurnCapTests(unittest.TestCase):
    """The six document tasks are allowed one turn and were being recorded as having hit a cap."""

    def test_a_one_turn_task_that_took_its_turn_is_done(self):
        self.assertEqual(session.turn_stop({"turns": 1}), "done")

    def test_a_multi_turn_task_that_runs_out_is_capped(self):
        self.assertEqual(session.turn_stop({"turns": 3}), "turn_cap")


class ReasoningFromCaptureTests(unittest.TestCase):
    """Every trial reported zero reasoning tokens while the server streamed a thinking block."""

    def test_the_capture_counts_what_the_server_streamed(self):
        from llm_eval.workload.inference import InferenceCapture
        capture = InferenceCapture.__new__(InferenceCapture)
        capture.reasoning_chars, capture.reasoning_blocks, capture.records = 954, 3, []
        self.assertEqual(capture.reasoning_summary(),
                         {"blocks": 3, "chars": 954, "tokens_estimate": 238, "cap_hits": 0})


class MemoryReportTests(unittest.TestCase):
    """The engine prints what it priced; parsing it turns the reserve into a forecast."""

    LOG = ("strata generate: expert cache auto: 14.89 GiB free, 2048 MiB reserved (+218 MiB for the draft head, "
           "+256 MiB for what is allocated after the cache) -> 5010 slots\n"
           "strata generate: expert cache 6542 slots, 12.42 GiB of VRAM; policy is\n")

    def adapter(self):
        return StrataAdapter.__new__(StrataAdapter)

    def test_the_cache_line_is_parsed(self):
        report = self.adapter().memory_report(self.LOG)
        self.assertEqual(report["reserve_mib"], 2048)
        self.assertEqual(report["draft_mib"], 218)
        self.assertEqual(report["post_cache_mib"], 256)
        self.assertEqual(report["planned_slots"], 5010)
        self.assertEqual(report["cache_slots"], 6542)
        self.assertEqual(report["cache_mib"], 12718)
        self.assertEqual(report["free_before_mib"], 15247)

    def test_the_forecast_is_the_reserve_rule(self):
        adapter = self.adapter()
        adapter._loading_context = 131072
        self.assertEqual(adapter.kv_mib(self.LOG), 1946)  # the measured slope at the anchor
        self.assertEqual(2048 - adapter.FREE_AFTER_RESERVE_MIB, 2013)  # the prediction; measurements vary ±2 MiB

    def test_a_log_without_the_line_is_not_guessed(self):
        self.assertEqual(self.adapter().memory_report("nothing here"), {})
        adapter = self.adapter()
        adapter._loading_context = None
        self.assertIsNone(adapter.kv_mib("nothing here"))


if __name__ == "__main__":
    unittest.main()


class ExtraRosterTests(unittest.TestCase):
    """The measurements the frozen 18-task cohort cannot make: a long session and a real image."""

    def test_the_depth_task_states_a_fact_then_fills_before_asking_for_it(self):
        from llm_eval import cohort
        from llm_eval.workload.graders import check_answer
        task = cohort.depth_task({"id": "depth-x", "seed": 111, "files": 2, "blocks": 2}, lambda text: 1)
        self.assertIn("Kestrel dispatch code", task["prelude"][0])
        self.assertEqual(len(task["fill"]["prompts"]), 2)
        self.assertGreater(task["fill"]["fraction"], 0.5)
        self.assertIn("depth", [c["type"] for c in task["checks"]])
        check = task["checks"][0]
        self.assertTrue(check_answer(check, {"answer": task["solution"]["answer"]}, None)["pass"])
        self.assertFalse(check_answer(check, {"answer": "NOT IN RECORD"}, None)["pass"])
