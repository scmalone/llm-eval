"""Regression tests for the 2026-10-04 red-team findings (RT-001 to RT-005, RT-008)."""
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import ladder, runner
from llm_eval.metrics.schema import connect
from llm_eval.runner import execute, restore, terminate_as_interrupt
from llm_eval.workload.capture import RawCapture
from llm_eval.workload.graders import discard_added_test_files
from test_runner import plan_with


class Dir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


class RestoreSurvivesBookkeepingTests(Dir):
    def test_a_full_disk_does_not_stop_the_restore(self):
        calls = []
        class Adapter:
            def stop(self): calls.append("stop")
            def restore(self, snapshot):
                calls.append("restore")
                return {"service_matches": True, "health": True, "four_token_probe": True}
        class Journal:
            state = {"restoration_required": True, "snapshot": {"was_active": True}}
            def save(self): raise OSError(28, "No space left on device")
            def event(self, *a, **k): raise OSError(28, "No space left on device")
        self.assertTrue(restore(Journal(), Adapter()))
        self.assertEqual(calls, ["stop", "restore"])

    def test_a_failed_restore_with_a_full_disk_reports_failure_instead_of_raising(self):
        class Adapter:
            def stop(self): raise RuntimeError("bus down")
        class Journal:
            state = {"restoration_required": True, "snapshot": {}}
            def save(self): raise OSError(28, "full")
            def event(self, *a, **k): raise OSError(28, "full")
        self.assertFalse(restore(Journal(), Adapter()))


class TerminationTests(Dir):
    def test_sigterm_becomes_an_interrupt_once_and_the_handlers_come_back(self):
        before = signal.getsignal(signal.SIGTERM)
        with terminate_as_interrupt():
            with self.assertRaises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGTERM)  # while restoring, a second one is ignored
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_a_terminated_campaign_still_restores_production(self):
        real = runner.run_trial
        def terminated(fixture, *args, **kwargs):
            if fixture["kind"] == "tools":
                os.kill(os.getpid(), signal.SIGTERM)
            return real(fixture, *args, **kwargs)
        with patch.object(runner, "run_trial", terminated):
            state = execute(plan_with(), self.root, "test")
        self.assertEqual(state["status"], "partial")
        self.assertFalse(state["restoration_required"])


class CaptureStaysInsideTheHomeTests(Dir):
    def setUp(self):
        super().setUp()
        self.outside = self.root / "outside"
        (self.outside / "state/opencode").mkdir(parents=True)
        (self.outside / "state/opencode/secret.txt").write_text("host-only")
        self.home = self.root / "home"
        (self.home / ".local/share/opencode").mkdir(parents=True)
        (self.home / ".local/share/opencode/log.txt").write_text("inside")

    def copied(self):
        capture = RawCapture(self.root / "raw")
        capture.opencode_data(self.home)
        return sorted(p.name for p in (self.root / "raw").rglob("*") if p.is_file())

    def test_a_real_tree_is_copied(self):
        self.assertEqual(self.copied(), ["log.txt"])

    def test_a_symlinked_root_is_not_followed(self):
        (self.home / ".local/state").mkdir()
        (self.home / ".local/state/opencode").symlink_to(self.outside / "state/opencode")
        self.assertNotIn("secret.txt", self.copied())

    def test_a_symlinked_ancestor_is_not_followed(self):
        (self.home / ".local/state").symlink_to(self.outside / "state")
        self.assertNotIn("secret.txt", self.copied())


class AddedTestFilesTests(Dir):
    def test_files_that_could_fake_a_pass_are_dropped_and_set_tests_are_kept(self):
        work = self.root / "repo"
        for name in ("tests/test_a.py", "tests/test_h.py", "src/new.py", "tests/test_000_disable.py", "conftest.py", "sitecustomize.py"):
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            (work / name).write_text("x")
        outcome = {"workdir": work, "base_files": {"tests/test_a.py": "h"}, "hidden_files": ["tests/test_h.py"]}
        self.assertEqual(discard_added_test_files(outcome), ["conftest.py", "sitecustomize.py", "tests/test_000_disable.py"])
        self.assertTrue((work / "tests/test_a.py").exists() and (work / "tests/test_h.py").exists() and (work / "src/new.py").exists())


class CohortTests(Dir):
    def seed(self, **model):
        conn = connect(self.root / "metrics.sqlite")
        with conn:
            conn.execute("INSERT INTO runs (run_id, valid_for_analysis) VALUES ('r', 1)")
            conn.execute("INSERT INTO run_models (run_id, profile, mode_entry, mode_context, mode_output_limit, mode_context_overridden)"
                         " VALUES ('r', 'p', 'e', 65536, ?, ?)", (model.get("limit", 12288), model.get("overridden", 0)))
            conn.execute("INSERT INTO task_runs (run_id, profile, section, task_id, task_hash, task_spec_sha, category, passed) "
                         "VALUES ('r', 'p', 's', 't1', 'h1', 'spec1', 'ops', 1)")
        return conn

    def test_runs_of_an_edited_task_do_not_count(self):
        self.seed().close()
        self.assertEqual(ladder._counts(self.root, "p", "e", {"t1": "spec1"}), {"t1": 1})
        self.assertEqual(ladder._counts(self.root, "p", "e", {"t1": "spec2"}), {"t1": 0})

    def test_a_context_override_does_not_satisfy_the_default_rung(self):
        self.seed(overridden=1).close()
        self.assertEqual(ladder._counts(self.root, "p", "e", {"t1": "spec1"}), {"t1": 0})

    def test_the_scorecard_keeps_output_limits_apart_and_flags_mixed_versions(self):
        conn = self.seed(limit=1024)
        with conn:
            conn.execute("INSERT INTO runs (run_id, valid_for_analysis) VALUES ('r2', 1)")
            conn.execute("INSERT INTO run_models (run_id, profile, mode_entry, mode_context, mode_output_limit, mode_context_overridden)"
                         " VALUES ('r2', 'p', 'e', 65536, 32768, 0)")
            conn.execute("INSERT INTO task_runs (run_id, profile, section, task_id, task_hash, task_spec_sha, category, passed) "
                         "VALUES ('r2', 'p', 's', 't1', 'h2', 'spec2', 'ops', 0)")
        rows = conn.execute("SELECT mode_output_limit, runs, passes, task_versions, tasks FROM v_scorecard ORDER BY mode_output_limit").fetchall()
        conn.close()
        self.assertEqual([tuple(r) for r in rows], [(1024, 1, 1, 1, 1), (32768, 1, 0, 1, 1)])


    def test_the_scorecard_never_pools_grader_harness_engine_settings_or_effort_versions(self):
        conn = self.seed()
        base = {"harness_sha": "a", "harness_version": "1.4.0", "llama_server_version": "12017", "graders_sha": "g1",
                "settings_json": '{"cache-type-k": "q8_0"}', "effort": "high"}
        with conn:
            conn.execute("UPDATE runs SET harness_sha='a', harness_name='pi', harness_version='1.4.0', "
                         "llama_server_version='12017' WHERE run_id='r'")
            conn.execute("UPDATE run_models SET settings_json=?, effort='high' WHERE run_id='r'", (base["settings_json"],))
            conn.execute("UPDATE task_runs SET graders_sha='g1' WHERE run_id='r'")
            for n, (field, value) in enumerate([("harness_sha", "b"), ("harness_version", "1.5.0"),
                                                ("llama_server_version", "12100"), ("graders_sha", "g2"),
                                                ("settings_json", '{"cache-type-k": "q4_0"}'), ("effort", "max")]):
                cfg = {**base, field: value}
                run = f"v{n}"
                conn.execute("INSERT INTO runs (run_id, valid_for_analysis, harness_sha, harness_name, harness_version, "
                             "llama_server_version) VALUES (?, 1, ?, 'pi', ?, ?)",
                             (run, cfg["harness_sha"], cfg["harness_version"], cfg["llama_server_version"]))
                conn.execute("INSERT INTO run_models (run_id, profile, mode_entry, mode_context, mode_output_limit, "
                             "settings_json, effort) VALUES (?, 'p', 'e', 65536, 12288, ?, ?)",
                             (run, cfg["settings_json"], cfg["effort"]))
                conn.execute("INSERT INTO task_runs (run_id, profile, section, task_id, task_hash, category, passed, "
                             "graders_sha) VALUES (?, 'p', 's', 't1', 'h1', 'ops', 1, ?)", (run, cfg["graders_sha"]))
            conn.execute("INSERT INTO runs (run_id, valid_for_analysis, harness_sha, harness_name, harness_version, "
                         "llama_server_version) VALUES ('same', 1, 'a', 'pi', '1.4.0', '12017')")
            conn.execute("INSERT INTO run_models (run_id, profile, mode_entry, mode_context, mode_output_limit, "
                         "settings_json, effort) VALUES ('same', 'p', 'e', 65536, 12288, ?, 'high')", (base["settings_json"],))
            conn.execute("INSERT INTO task_runs (run_id, profile, section, task_id, task_hash, category, passed, "
                         "graders_sha) VALUES ('same', 'p', 's', 't1', 'h1', 'ops', 0, 'g1')")
        rows = conn.execute("SELECT runs FROM v_scorecard ORDER BY runs DESC").fetchall()
        conn.close()
        self.assertEqual([r[0] for r in rows], [2, 1, 1, 1, 1, 1, 1])  # only the identical run pools

    def test_every_verdict_carries_the_graders_hash(self):
        from llm_eval.workload import graders
        self.assertEqual(len(graders.GRADERS_SHA), 64)
        verdict = graders.grade({"checks": []}, {})
        self.assertEqual(verdict["graders_sha"], graders.GRADERS_SHA)


class LookupControlTests(unittest.TestCase):
    """RT-007: a lookup whose keywords are present but whose claim is false must fail."""
    def answer_check(self, name):
        from llm_eval.workload.graders import check_answer
        task = json.loads((Path(__file__).resolve().parents[1] / "fixtures/workload" / f"{name}.json").read_text())
        check = next(c for c in task["checks"] if c["type"] == "answer")
        return lambda text: check_answer(check, {"answer": text}, None)["pass"]

    def test_loopback_guard_rejects_the_reviewers_counterexample_and_accepts_the_truth(self):
        grade = self.answer_check("lookup-loopback-guard")
        self.assertFalse(grade("preflight.py router_health accepts remote hosts; loopback is optional, and /health redirects are followed"))
        self.assertFalse(grade("preflight.py router_health: it allows any host, loopback optional, GET /health"))
        self.assertTrue(grade("In preflight.py, router_health raises ValueError unless the URL is http on a literal loopback address "
                              "with the path /health, and never follows redirects."))

    def test_uncommitted_work_rejects_a_clean_tree_claim(self):
        grade = self.answer_check("lookup-uncommitted-work")
        self.assertFalse(grade("llm_eval/policy.py, README.md and notes/scratch.md are fine; the working tree is clean."))
        self.assertTrue(grade("Uncommitted: llm_eval/policy.py (modified), README.md (modified), notes/scratch.md (untracked)."))


if __name__ == "__main__":
    unittest.main()
