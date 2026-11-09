"""Regression tests for the 2026-10-04 live-path red team (RT2-001 to RT2-010)."""
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import ladder, screen
from llm_eval.metrics.schema import connect
from llm_eval.runner import restore, terminate_as_interrupt
from llm_eval.workload.capture import RawCapture
from llm_eval.workload.graders import CHECKERS, check_depth, discard_added_test_files, grade
from llm_eval.workload.metrics import summarize
from llm_eval.workload.snapshot import escape_reason
import test_screen
import test_workload_runner

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/workload"


class Dir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


class SymlinkedRootTests(Dir):
    """RT2-001: the agent owns the home, so a symlinked repo root must never be walked, hashed or cleaned on the host."""
    def setUp(self):
        super().setUp()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        (self.outside / "test_outside.py").write_text("host file")
        (self.outside / "keep.txt").write_text("host file")
        self.home = self.root / "home"
        (self.home / "Workspace").mkdir(parents=True)

    def test_escape_reason_names_a_symlink_anywhere_between_home_and_path(self):
        (self.home / "Workspace/repo").symlink_to(self.outside)
        self.assertIn("symlink", escape_reason(self.home / "Workspace/repo", self.home))
        self.assertIn("symlink", escape_reason(self.home / "Workspace/repo/sub", self.home))
        (self.home / "real").mkdir()
        self.assertIsNone(escape_reason(self.home / "real", self.home))
        self.assertIn("not under", escape_reason(self.outside, self.home))

    def test_grading_a_symlinked_root_fails_before_anything_touches_the_host(self):
        (self.home / "Workspace/repo").symlink_to(self.outside)
        outcome = {"home": self.home, "workdir": self.home / "Workspace/repo", "answer": "x", "base_files": {}}
        verdict = grade({"checks": [{"type": "answer", "all": ["x"]}]}, outcome)
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["checks"][0]["type"], "isolation")
        self.assertTrue((self.outside / "test_outside.py").exists() and (self.outside / "keep.txt").exists())

    def test_capture_skips_a_symlinked_root_and_says_so(self):
        from llm_eval.adapters import FakeBox
        (self.home / "Workspace/repo").symlink_to(self.outside)
        capture = RawCapture(self.root / "raw")
        capture.end_state(FakeBox(), self.home / "Workspace/repo", self.home, None)
        self.assertFalse((self.root / "raw/end-state/files.json").exists())
        self.assertTrue(any("skipped" in e["error"] for e in capture.errors))


class ShadowTests(Dir):
    def test_files_that_shadow_a_runner_module_are_removed_and_ordinary_new_code_is_kept(self):
        work = self.root / "repo"
        for name in ("unittest.py", "tests/unittest/__init__.py", "os.py", "src/html.py", "src/feature.py", "json.py"):
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            (work / name).write_text("x")
        outcome = {"workdir": work, "base_files": {"json.py": "h"}, "hidden_files": []}
        removed = discard_added_test_files(outcome)
        self.assertEqual(set(removed), {"unittest.py", "tests/unittest/", "os.py"})
        self.assertTrue((work / "src/html.py").exists() and (work / "src/feature.py").exists() and (work / "json.py").exists())


class SignalDuringRestoreTests(Dir):
    """RT2-002: a first termination signal that arrives while production is being restored must not abort it."""
    def adapter(self, signal_number):
        calls = []
        class A:
            def stop(self):
                calls.append("stop")
                os.kill(os.getpid(), signal_number)
            def restore(self, snapshot):
                calls.append("restore")
                return {"service_matches": True, "health": True, "four_token_probe": True}
        return A(), calls

    def test_each_signal_during_the_restore_still_finishes_it(self):
        class J:
            state = {"restoration_required": True, "snapshot": {"was_active": True}}
            def save(self): pass
            def event(self, *a, **k): pass
        for number in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            with self.subTest(signal=number.name):
                adapter, calls = self.adapter(number)
                journal = J()
                journal.state = {"restoration_required": True, "snapshot": {"was_active": True}}
                before = signal.getsignal(number)
                with terminate_as_interrupt():
                    self.assertTrue(restore(journal, adapter))
                self.assertEqual(calls, ["stop", "restore"])
                self.assertEqual(signal.getsignal(number), before)  # handlers are back


class DepthAtTheQuestionTests(unittest.TestCase):
    """RT2-005: a compaction must not let an old peak certify a shallow question as deep."""
    def export(self, inputs):
        parts = [{"type": "step-finish", "tokens": {"input": n, "output": 10}} for n in inputs]
        return {"messages": [{"info": {"role": "user", "time": {"created": 0}}, "parts": []},
                             {"info": {"role": "assistant", "time": {"created": 0, "completed": 1000}}, "parts": parts}]}

    def test_the_last_step_is_the_context_at_the_question(self):
        m = summarize(self.export([110000, 2000]))
        self.assertEqual((m["peak_context_tokens"], m["final_context_tokens"]), (110000, 2000))

    def test_the_scaled_depth_check_grades_the_final_context_not_the_peak(self):
        check = {"min_peak_tokens": 40000, "min_fraction": 0.85}
        shallow = {"metrics": summarize(self.export([110000, 2000])), "compaction_point": 114688}
        self.assertFalse(check_depth(check, shallow, None)["pass"])
        deep = {"metrics": summarize(self.export([60000, 100000])), "compaction_point": 114688}
        self.assertTrue(check_depth(check, deep, None)["pass"])

    def test_the_fixed_floor_checks_still_use_the_peak(self):
        self.assertTrue(check_depth({"min_peak_tokens": 40000}, {"metrics": summarize(self.export([60000, 2000]))}, None)["pass"])

    def test_the_fill_meter_reads_the_latest_step(self):
        from llm_eval.workload.driver import OpencodeDriver
        d = OpencodeDriver.__new__(OpencodeDriver)
        d.export = lambda: self.export([110000, 2000])
        self.assertEqual(d.context_tokens(), 2000)


class ModelLevelFailureTests(test_screen.Corpus):
    """RT2-007: a profile that cannot load must block a blind repeat and show up as a capacity result."""
    def attempt(self, sections):
        directory = Path(self.tmp.name) / "attempt"
        (directory / "runs/acceptance-x").mkdir(parents=True)
        (directory / "runs/acceptance-x/state.json").write_text(json.dumps({"status": "complete", "sections": sections}))
        return directory

    def test_a_model_level_capacity_failure_blocks_the_next_attempt(self):
        directory = self.attempt({"tiny__failure": {"status": "capacity_fail", "reason": "no context loads"}})
        self.assertIn("capacity_fail in failure: no context loads", ladder.blocker(directory))

    def test_a_clean_attempt_does_not_block(self):
        self.assertIsNone(ladder.blocker(self.attempt({"tiny__smoke": {"status": "pass"}})))

    def test_the_screen_lists_a_profile_that_cannot_load_instead_of_owing_it_forever(self):
        with self.conn:
            self.conn.execute("INSERT INTO runs (run_id, valid_for_analysis) VALUES ('r', 1)")
            self.conn.execute("INSERT INTO sections (run_id, key, status, time, reason) VALUES ('r', 'a__failure', 'capacity_fail', 1, 'no context loads')")
        presets = {"a": {"model": "a.gguf", "ctx-size": "131072"}, "b": {"model": "b.gguf", "ctx-size": "131072"}}
        with patch("llm_eval.preflight.read_presets", lambda path=None: (Path("x"), presets)):
            shown = screen.report(self.root)
        self.assertEqual(shown["cannot_run"], {"a": "no context loads"})
        self.assertEqual(shown["not_started"], ["b"])
        self.assertIn("could not load", shown["next"])


class CohortAgreementTests(test_screen.Corpus):
    """RT2-008: the ladder and the screen count the same runs; unsupported overrides are refused up front."""
    def test_a_run_at_another_output_limit_does_not_complete_a_screen_rung(self):
        self.add("p", test_screen.TASKS[0], "pass", output=32768)
        self.assertEqual(ladder._counts(self.root, "p", "p", {test_screen.TASKS[0]: test_screen.SPEC[test_screen.TASKS[0]]}, 16384),
                         {test_screen.TASKS[0]: 0})
        self.assertEqual(ladder._counts(self.root, "p", "p", {test_screen.TASKS[0]: test_screen.SPEC[test_screen.TASKS[0]]}, None),
                         {test_screen.TASKS[0]: 1})  # a full-stage rung takes the output limit it was prepared with

    def test_next_refuses_context_and_a_screen_output_override(self):
        from llm_eval import acceptance
        for argv in (["prepare", "--next", "--context", "98304"], ["prepare", "--next", "--output", "8192"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                acceptance.main(argv)


class UnreachableDepthTests(unittest.TestCase):
    """RT2-009: a mode whose compaction point is below the owner's 64K floor gets a capacity result, not a quality failure."""
    def setUp(self):
        self.base = test_workload_runner.WorkloadCampaignTests("test_plan_pins_snapshots_binary_and_home_files")
        self.base.setUp()
        self.addCleanup(self.base.tearDown)

    def run_with(self, context, output):
        from llm_eval.runner import execute
        b = self.base
        task = {**b.tasks["impl-1"], "fill": {"prompts": ["read a"], "fraction": 0.9, "max_turns": 2}, "runs": 1}
        (b.root / "impl-1.json").write_text(json.dumps(task))
        b.suite["models"][0]["workload"].update(context=context, mode={"output_limit": output})
        scenario = {"workload": {"impl-1": {"turns": [[{"src/app.py": test_workload_runner.FIX}, "Fixed add()."]]}}}
        return b.sections(execute(b.plan(scenario), b.root / "runs", "wl"))

    def test_a_low_compaction_point_is_a_capacity_fail_and_nothing_is_run(self):
        from llm_eval.runner import COMPACTION_MARGIN
        trial = self.run_with(65536, 32768)["trial-wl-impl-1-r1"]  # compaction point 33,536 less the margin
        self.assertEqual(trial["status"], "capacity_fail")
        self.assertIn("below the 64000", trial["stop"])
        self.assertEqual(trial["compaction_point"], 33536 - COMPACTION_MARGIN)

    def test_a_roomy_mode_runs_the_task(self):
        trial = self.run_with(131072, 16384)["trial-wl-impl-1-r1"]
        self.assertNotEqual(trial["status"], "capacity_fail")


class AdvisoryPolarityTests(unittest.TestCase):
    """RT2-010: say what to keep, and a careful negation is not a recommendation to keep it."""
    def grade(self, text):
        task = json.loads((FIXTURES / "advisory-models-to-keep.json").read_text())
        outcome = {"answer": text}
        return all(CHECKERS[c["type"]](c, outcome, {})["pass"] for c in task["checks"] if c["type"] in ("answer", "style"))

    def test_a_negated_mention_of_q3_is_fine(self):
        self.assertTrue(self.grade("A. **(Recommended)** Keep Q4_K_XL for both muse and qwen; do not keep Q3_K_M or Q3_K_S.\nB. Keep Q8 too."))
        self.assertTrue(self.grade("A. **(Recommended)** Keep Q4_K_XL for both muse and qwen, not Q3_K_M.\nB. Keep Q8 too."))
        self.assertTrue(self.grade("A. **(Recommended)** Delete the Q3 files and keep the Q4_K_XL ones for muse and qwen.\nB. Keep Q8 too."))

    def test_keeping_q3_or_deleting_q4_is_not(self):
        self.assertFalse(self.grade("A. **(Recommended)** Keep the Q3_K_M and the Q4_K_XL files for muse and qwen.\nB. Delete all."))
        self.assertFalse(self.grade("A. **(Recommended)** Delete Q4_K_XL for both muse and qwen.\nB. Keep everything."))

    def test_the_screen_audit_lists_failures_as_well_as_passes(self):
        corpus = test_screen.RankTests("test_a_timeout_is_a_failure_that_stays_visible")
        corpus.setUp()
        self.addCleanup(corpus.tearDown)
        corpus.profile("a", "pfpppff")  # task 1 (advisory is index 2) passes; index 1 resume fails
        entry = screen.rank(corpus.cells("a"), test_screen.SPECS)["ranking"][0]
        self.assertEqual({k: v["outcome"] for k, v in entry["audit"].items()},
                         {"advisory-models-to-keep": "pass", "review-llm-eval-live-recovery": "fail"})


class TiebreakTests(test_screen.Corpus):
    """RT2-004: the extra tasks the runbook asks for must change the third place."""
    TB = screen.tiebreak_specs()

    def tb(self, profile, task_index, status):
        task, _, sha = self.TB[task_index]
        self.add(profile, task, status, spec=sha)

    def board(self, *rows):
        for name, outcomes in rows:
            self.profile(name, outcomes)
        presets = {n: {"model": f"{n}.gguf", "ctx-size": "131072"} for n, _ in rows}
        with patch("llm_eval.preflight.read_presets", lambda path=None: (Path("x"), presets)):
            return screen.report(self.root)

    def test_the_tiebreak_tasks_come_from_the_ladder_file(self):
        self.assertEqual([t for t, _, _ in self.TB], ["impl-llm-eval-output-budget", "review-llm-eval-memory-obligation"])

    def test_with_no_tiebreak_run_the_first_one_is_owed_for_every_boundary_candidate(self):
        shown = self.board(("a", "ppppppp"), ("b", "pppppff"), ("c", "pppppff"), ("d", "ppppfff"))
        self.assertEqual(sorted(shown["boundary_candidates"]), ["c", "d"])
        self.assertEqual(shown["tiebreak_owed"]["task"], "impl-llm-eval-output-budget")
        self.assertEqual(shown["tiebreak_owed"]["profiles"], ["c", "d"])
        self.assertTrue(shown["third_place_provisional"])
        self.assertIn("impl-llm-eval-output-budget", shown["next"])

    def test_a_decisive_tiebreak_changes_who_takes_the_third_slot(self):
        self.profile("a", "ppppppp"); self.profile("b", "ppppppf"); self.profile("c", "pppppff"); self.profile("d", "pppppff")
        presets = {n: {"model": f"{n}.gguf", "ctx-size": "131072"} for n in "abcd"}
        # c is first by the tie order (name) when nothing else separates them; d wins the extra task
        with patch("llm_eval.preflight.read_presets", lambda path=None: (Path("x"), presets)):
            before = screen.report(self.root)
            self.tb("c", 0, "quality_fail"); self.tb("d", 0, "pass")
            after = screen.report(self.root)
        self.assertEqual(before["top3"][2], "c")
        self.assertEqual(after["top3"], ["a", "b", "d"])
        self.assertFalse(after["third_place_provisional"])
        self.assertIsNone(after["tiebreak_owed"])
        self.assertEqual(after["boundary_totals"], {"c": 5, "d": 6})

    def test_a_tie_that_survives_every_tiebreak_stays_provisional_and_owes_nothing(self):
        self.profile("a", "ppppppp"); self.profile("b", "ppppppf"); self.profile("c", "pppppff"); self.profile("d", "pppppff")
        for i in (0, 1):
            self.tb("c", i, "pass"); self.tb("d", i, "pass")
        presets = {n: {"model": f"{n}.gguf", "ctx-size": "131072"} for n in "abcd"}
        with patch("llm_eval.preflight.read_presets", lambda path=None: (Path("x"), presets)):
            shown = screen.report(self.root)
        self.assertIsNone(shown["tiebreak_owed"])
        self.assertTrue(shown["third_place_provisional"])
        self.assertEqual(shown["tiebreak_counted"], [t for t, _, _ in self.TB])
        self.assertIn("provisional", shown["next"])


if __name__ == "__main__":
    unittest.main()
