"""Rung 0: the pre-pilot gate. No GPU, no bwrap, no agent loop."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from llm_eval import rung0

LIMITS = {"campaign_seconds": 4500, "model_seconds": 3600, "load_seconds": 240, "request_seconds": 600}


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_planted_fsmonitor_does_not_run_when_graded(self):
        result = rung0.check_grading_isolation(self.root)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["graded"], result)
        self.assertFalse(Path(result["sentinel"]).exists())

    def test_the_sentinel_is_a_real_detector(self):
        """The same plant, run through an unsafe git, does execute: the pass above is not vacuous."""
        rung0.check_grading_isolation(self.root)
        run = self.root / "home/repo"
        self.assertTrue((run / ".git").exists())
        subprocess.run(["git", "status"], cwd=run, capture_output=True, text=True, timeout=60)
        self.assertTrue((self.root / "fsmonitor-ran").exists(),
                        "the planted fsmonitor should have fired on an unsafe git status")


class OutputCapTests(unittest.TestCase):
    def test_the_shipped_fixtures_pass(self):
        result = rung0.check_output_caps(LIMITS)
        self.assertTrue(result["passed"], result)
        self.assertTrue({r["kind"] for r in result["fixtures"]} >= {"smoke", "recall", "tools", "vision"})

    def caps(self, fixture, limits=LIMITS):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "x.json").write_text(json.dumps(fixture))
            return rung0.check_output_caps(limits, fixtures_dir=tmp)

    def test_a_cap_below_its_floor_fails(self):
        result = self.caps({"kind": "recall", "max_output_tokens": 768})
        self.assertFalse(result["passed"])
        self.assertTrue(any("below its recall floor" in p for p in result["problems"]), result["problems"])

    def test_a_cap_above_the_budget_fails(self):
        result = self.caps({"kind": "recall", "max_output_tokens": rung0.OUTPUT_BUDGET_TOKENS + 1})
        self.assertFalse(result["passed"])
        self.assertTrue(any("above the" in p for p in result["problems"]), result["problems"])

    def test_a_cap_that_cannot_finish_inside_the_request_cap_fails(self):
        result = self.caps({"kind": "recall", "max_output_tokens": rung0.OUTPUT_BUDGET_TOKENS},
                           LIMITS | {"request_seconds": 100})
        self.assertFalse(result["passed"])
        self.assertTrue(any("request cap" in p for p in result["problems"]), result["problems"])

    def test_a_fixture_without_an_output_cap_is_skipped(self):
        result = self.caps({"kind": "workload", "limits": {"turns": 1, "wall_seconds": 60}})
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["fixtures"], [])


class ReportTests(unittest.TestCase):
    def test_run_reports_both_conditions_and_the_down_time(self):
        report, code = rung0.run(LIMITS)
        self.assertEqual(code, rung0.OK, report)
        self.assertTrue(report["passed"])
        self.assertIn("grading_isolation", report["conditions"])
        self.assertIn("output_caps", report["conditions"])
        self.assertEqual(report["hang_cap_seconds"], 3600)
        self.assertEqual(report["worst_case_down_seconds"], 240 + 4500 + 120)

    def test_a_failed_condition_refuses(self):
        report, code = rung0.run(LIMITS | {"request_seconds": 1})
        self.assertEqual(code, rung0.REFUSED)
        self.assertFalse(report["passed"])
        self.assertIn("STOP", report["next"])


if __name__ == "__main__":
    unittest.main()
