"""Regression tests for the 2026-10-04 efficiency red-team (EF-001 to EF-010)."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import acceptance, ladder
from llm_eval.metrics.schema import connect
from llm_eval.workload import session
from llm_eval.workload.driver import OpencodeDriver
from llm_eval.workload.metrics import summarize
from llm_eval.workload.session import run_task
import test_workload_runner
import test_workload_session
from test_workload_session import FakeBox, FakeDriver


class StartupFailureTests(unittest.TestCase):
    """EF-002: a harness that never started is not a model that failed."""
    def driver(self, attempts):
        d = OpencodeDriver.__new__(OpencodeDriver)
        d.session, d.ref, d.turn_args, d.raw_turns = None, "p/m", [], []
        d.box = type("B", (), {"inside": lambda self, p: str(p)})()
        it = iter(attempts)
        d._run_streaming = lambda *a, **k: next(it)
        return d

    stalled = {"stalled": True, "events": [], "stderr": "", "returncode": -9, "timed_out": False, "lingered": False}

    def test_two_startup_stalls_surface_as_stalled(self):
        result = self.driver([self.stalled, self.stalled]).turn("go", "/w", 600)
        self.assertTrue(result["stalled"])
        self.assertEqual(result["restarts"], 1)

    def test_a_retry_that_works_is_not_stalled(self):
        ok = {"stalled": False, "events": [{"type": "text", "part": {"text": "hi"}}], "stderr": "", "returncode": 0,
              "timed_out": False, "lingered": False}
        self.assertFalse(self.driver([self.stalled, ok]).turn("go", "/w", 600)["stalled"])


class SessionStopTests(unittest.TestCase):
    def setUp(self):  # reuse the base fixtures without inheriting (and rerunning) its tests
        self.base = test_workload_session.RunTaskTests("test_a_noop_run_fails_and_turn_cap_stops_it")
        self.base.setUp()
        self.addCleanup(self.base.tearDown)
        self.task, self.root = self.base.task, self.base.root

    def run_stop(self, turn_result, name):
        class Odd(FakeDriver):
            def turn(self, text, workdir, timeout):
                self.last = ""
                return {"answer": "", "timed_out": False, "events": 0, **turn_result}
        return run_task(self.task, self.root / name, self.root / "snap", lambda home, manifest: Odd(home, []),
                        lambda home: FakeBox())

    def test_a_startup_stall_is_a_harness_stall_not_a_wrong_answer(self):
        result = self.run_stop({"stalled": True, "returncode": -9}, "stall")
        self.assertEqual(result["stop"], "harness_stall")
        self.assertFalse(result["pass"])
        self.assertEqual(result["tests_pasted"], 0)  # no test-repair follow-up for a run that never started

    def test_a_crash_before_any_event_is_a_harness_exit(self):
        self.assertEqual(self.run_stop({"returncode": 1}, "crash")["stop"], "harness_exit")

    def test_a_clean_empty_exit_is_still_the_models_answer(self):
        self.assertNotIn(self.run_stop({"returncode": 0}, "empty")["stop"], ("harness_stall", "harness_exit"))


class StopClassTests(unittest.TestCase):
    def setUp(self):
        self.base = test_workload_runner.WorkloadCampaignTests("test_plan_pins_snapshots_binary_and_home_files")
        self.base.setUp()
        self.addCleanup(self.base.tearDown)
        self.suite, self.root, self.plan, self.sections = self.base.suite, self.base.root, self.base.plan, self.base.sections

    def run_with_stop(self, stop, keep_going=False):
        self.suite["models"][0]["workload"]["stop_on_infra_fail"] = not keep_going
        real = session.run_task
        def stopped(*args, **kwargs):
            result = real(*args, **kwargs)
            return {**result, "stop": stop, "pass": False}
        with patch.object(session, "run_task", stopped):
            from llm_eval.runner import execute
            return self.sections(execute(self.plan(), self.root / "runs", "wl"))

    def test_a_wall_cap_is_a_timeout_and_the_block_goes_on(self):
        sections = self.run_with_stop("wall_cap")
        self.assertEqual(sections["trial-wl-impl-1-r1"]["status"], "timeout")
        self.assertIn("trial-wl-ops-1-r2", sections)

    def test_harness_stops_are_infrastructure_and_end_the_block(self):
        for stop in ("harness_stall", "harness_exit", "stalled", "budget_cap"):
            with self.subTest(stop=stop):
                self.base.tearDown(); self.base.setUp()
                self.suite, self.root, self.plan, self.sections = self.base.suite, self.base.root, self.base.plan, self.base.sections
                sections = self.run_with_stop(stop)
                self.assertEqual(sections["trial-wl-impl-1-r1"]["status"], "infra_fail")
                self.assertNotIn("trial-wl-impl-1-r2", sections)


class OutputMeasurementTests(unittest.TestCase):
    """EF-004: the output limit has to cover one generation, so record the largest one and any truncation."""
    def summarize(self, export=None):
        from llm_eval.workload.metrics import summarize
        return summarize(self.export() if export is None else export)

    def export(self):
        steps = [{"type": "step-finish", "reason": "tool-calls", "tokens": {"input": 100, "output": 300}},
                 {"type": "step-finish", "reason": "length", "tokens": {"input": 400, "output": 16384}},
                 {"type": "step-finish", "reason": "stop", "tokens": {"input": 900, "output": 50}}]
        return {"messages": [{"info": {"role": "user", "time": {"created": 0}}, "parts": []},
                             {"info": {"role": "assistant", "time": {"created": 0, "completed": 1000}}, "parts": steps}]}

    def test_the_peak_step_and_length_cutoffs_are_recorded_beside_the_sum(self):
        m = summarize(self.export())
        self.assertEqual((m["output_tokens"], m["peak_step_output_tokens"], m["length_finishes"]), (16734, 16384, 1))

    def test_the_acceptance_report_names_the_runs_that_were_cut_off(self):
        sections = {"trial-wl-a-r1": {"section": "m__trial-wl-a-r1", "metrics": {"peak_step_output_tokens": 900, "length_finishes": 0}},
                    "trial-wl-b-r1": {"section": "m__trial-wl-b-r1", "metrics": {"peak_step_output_tokens": 16384, "length_finishes": 1}}}
        self.assertEqual(acceptance._workload_output(sections),
                         {"peak_step_output_tokens": 16384, "length_finishes": ["trial-wl-b-r1"]})


class TimeoutCountsTests(unittest.TestCase):
    def test_a_timeout_counts_as_done_for_the_ladder(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        conn = connect(Path(tmp.name) / "metrics.sqlite")
        with conn:
            conn.execute("INSERT INTO runs (run_id, valid_for_analysis) VALUES ('r', 1)")
            conn.execute("INSERT INTO run_models (run_id, profile, mode_entry, mode_context_overridden) VALUES ('r', 'p', 'e', 0)")
            conn.execute("INSERT INTO task_runs (run_id, profile, section, task_id, task_spec_sha, failure_class, passed) "
                         "VALUES ('r', 'p', 's', 't', 'x', 'timeout', 0)")
        conn.close()
        self.assertEqual(ladder._counts(tmp.name, "p", "e", {"t": "x"}), {"t": 1})


class ScorecardViewTests(unittest.TestCase):
    """EF-009: core and depth tasks, and timeouts, are not pooled into one rate."""
    def test_groups_stay_apart_and_timeouts_are_counted_as_failures(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        conn = connect(Path(tmp.name) / "metrics.sqlite")
        with conn:
            conn.execute("INSERT INTO runs (run_id, valid_for_analysis) VALUES ('r', 1)")
            conn.execute("INSERT INTO run_models (run_id, profile, mode_entry, mode_context, mode_output_limit) VALUES ('r','p','e',131072,16384)")
            for section, group, passed, failure in (("a", "lookup", 1, None), ("b", "depth", 0, "model"), ("c", "lookup", 0, "timeout"),
                                                    ("d", "lookup", 0, "infra")):
                conn.execute("INSERT INTO task_runs (run_id, profile, section, task_id, task_hash, category, group_name, passed, failure_class, context) "
                             "VALUES ('r','p',?,?,'h','lookup',?,?,?,114688)", (section, section, group, passed, failure))
        rows = {r["group_name"]: (r["runs"], r["passes"], r["timeouts"], r["mode_context"]) for r in
                conn.execute("SELECT * FROM v_scorecard")}
        conn.close()
        self.assertEqual(rows, {"lookup": (2, 1, 1, 114688), "depth": (1, 0, 0, 114688)})  # infra left out; context actually run at


class GraderTighteningTests(unittest.TestCase):
    """EF-003: answers that carry the right words but not the right claim must not pass."""
    FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/workload"

    def grades(self, name):
        from llm_eval.workload.graders import CHECKERS
        task = json.loads((self.FIXTURES / f"{name}.json").read_text())
        checks = [c for c in task["checks"] if c["type"] in ("answer", "style", "findings")]
        def grade(text):
            outcome = {"answer": text}
            return all(CHECKERS[c["type"]](c, outcome, {})["pass"] for c in checks)
        return grade, task["solution"]["answer"]

    def test_each_tightened_task_still_passes_its_own_solution(self):
        for name in ("advisory-models-to-keep", "advisory-overbuilt", "advisory-output-limit", "review-llm-eval-live-recovery"):
            grade, solution = self.grades(name)
            with self.subTest(task=name):
                self.assertTrue(grade(solution))

    def test_overbuilt_rejects_keep_everything_as_the_recommendation(self):
        grade, _ = self.grades("advisory-overbuilt")
        self.assertFalse(grade("A. **(Recommended)** Keep everything.\nB. Cut the runner.\nC. Pause."))

    def test_output_limit_rejects_never_compact_and_needs_the_measured_numbers(self):
        grade, _ = self.grades("advisory-output-limit")
        self.assertFalse(grade("A. **(Recommended)** Never compact; use output 1.\nB. 16,384."))
        self.assertFalse(grade("A. **(Recommended)** 16,384, it will compact later.\nB. 32,768."))  # no measured evidence cited

    def test_models_to_keep_rejects_keeping_the_q3_files(self):
        grade, _ = self.grades("advisory-models-to-keep")
        self.assertFalse(grade("A. **(Recommended)** Keep the Q3_K_M muse and Q3_K_S qwen files and the Q4_K_XL ones.\nB. Delete all."))

    def test_review_rejects_scattered_keywords_that_state_no_defect(self):
        grade, _ = self.grades("review-llm-eval-live-recovery")
        self.assertFalse(grade("restore time out. scratch resume memory. cudaMalloc lower. load memory fail."))


class ScaledDepthTests(unittest.TestCase):
    """Plan 2026-10-04: depth scaled to the mode's compaction point, not a fixed floor."""
    def setUp(self):
        self.base = test_workload_session.RunTaskTests("test_a_noop_run_fails_and_turn_cap_stops_it")
        self.base.setUp()
        self.addCleanup(self.base.tearDown)

    def go(self, meter, fill=None, name="d", compaction_point=114688, driver_cls=None):
        task = {**self.base.task, "fill": fill or {"prompts": ["read a", "read b"], "fraction": 0.9, "max_turns": 6},
                "limits": {"turns": 2, "wall_seconds": 900}}
        readings = iter(meter)
        class Metered(driver_cls or FakeDriver):
            def context_tokens(self):
                return next(readings, None)
        return run_task(task, self.base.root / name, self.base.root / "snap",
                        lambda home, manifest: Metered(home, [({}, "ok")] * 12), lambda home: FakeBox(),
                        compaction_point=compaction_point), task

    def test_the_session_is_grown_until_it_reaches_the_fraction_then_the_task_is_asked(self):
        result, _ = self.go([30000, 60000, 95000, 104000])
        self.assertEqual(result["depth"], {"target_tokens": int(0.9 * 114688), "reached_tokens": 104000, "fill_turns": 4,
                                           "compaction_point": 114688})
        sources = [t["source"] for t in result["turns"]]
        self.assertEqual(sources[:5], ["fill"] * 4 + ["prompt"])  # then the base task's own test follow-up

    def test_a_session_that_will_not_grow_stops_at_max_turns(self):
        result, _ = self.go([1000] * 20)
        self.assertEqual(result["depth"]["fill_turns"], 6)
        self.assertEqual(result["depth"]["reached_tokens"], 1000)

    def test_no_compaction_point_means_no_fill(self):
        result, _ = self.go([50000], compaction_point=None)
        self.assertIsNone(result["depth"])
        self.assertEqual(result["turns"][0]["source"], "prompt")

    def test_a_hung_fill_turn_stops_the_task(self):
        class Hung(FakeDriver):
            def turn(self, text, workdir, timeout):
                self.last = ""
                return {"answer": "", "timed_out": False, "events": 0, "hung": True, "returncode": -9}
        result, _ = self.go([], driver_cls=Hung)
        self.assertEqual(result["stop"], "stalled")

    def test_the_depth_check_scales_to_the_mode_and_keeps_its_floor(self):
        from llm_eval.workload.graders import check_depth
        check = {"min_peak_tokens": 40000, "min_fraction": 0.85}
        outcome = lambda peak, cp: {"metrics": {"peak_context_tokens": peak, "final_context_tokens": peak}, "compaction_point": cp}
        self.assertFalse(check_depth(check, outcome(60000, 114688), None)["pass"])   # needs 97,484
        self.assertTrue(check_depth(check, outcome(100000, 114688), None)["pass"])
        self.assertTrue(check_depth(check, outcome(45000, 40000), None)["pass"])      # a small mode keeps the 40K floor
        self.assertTrue(check_depth(check, outcome(60000, None), None)["pass"])       # no mode known: the floor alone

    def test_the_shipped_scaled_task_validates_and_its_prompts_name_files_that_exist_in_the_pinned_commit(self):
        import re
        from llm_eval.workload import prepare as workload
        from llm_eval.workload.snapshot import build, materialize
        task = next(t for t in workload.load_tasks() if t["id"] == "depth-scaled-early-instruction")
        self.assertTrue(task["depth"] and task["fill"]["fraction"] == 0.9)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build(task, root / "snapshot")
            repo = materialize(root / "snapshot", root / "repo")
            for prompt in task["fill"]["prompts"]:
                for name in re.findall(r"[\w./-]+\.(?:py|md)", prompt):
                    self.assertTrue((repo / name).is_file(), name)

    def test_validation_refuses_a_bad_fill_and_a_depth_check_with_no_floor_or_fraction(self):
        from llm_eval.workload import tasks as task_module
        base = json.loads((Path(__file__).resolve().parents[1] / "fixtures/workload/depth-scaled-early-instruction.json").read_text())
        for change in ({"fill": {"prompts": [], "fraction": 0.9, "max_turns": 3}}, {"fill": {"prompts": ["x"], "fraction": 1.5, "max_turns": 3}},
                       {"fill": {"prompts": ["x"], "fraction": 0.9, "max_turns": 0}}):
            with self.assertRaises(ValueError):
                task_module.validate({**base, **change})
        checks = [c if c["type"] != "depth" else {"type": "depth"} for c in base["checks"]]
        with self.assertRaises(ValueError):
            task_module.validate({**base, "checks": checks})


if __name__ == "__main__":
    unittest.main()
