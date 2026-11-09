import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import acceptance, integration, ladder
from llm_eval.adapters import FakeBox, FakeWorkloadDriver
from llm_eval.live import LiveAdapter
from llm_eval.metrics.ingest import spec_sha
from llm_eval.metrics.schema import connect
from llm_eval.workload import prepare as workload, replicas
from test_live import ScriptedHost
from test_workload_acceptance import OPS, PRODUCTION

PROFILE = "tiny"


def write_corpus(root, profile=PROFILE, entry="tiny", task="ops-router-1", passes=1, failure=None, valid=1, run="r1"):
    """Rows a finished run would leave, so the ladder can read them."""
    conn = connect(Path(root) / "metrics.sqlite")
    spec = next((spec_sha(t) for t in workload.load_tasks(Path(root) / "tasks") if t["id"] == task), None) if (Path(root) / "tasks").exists() else None
    with conn:
        conn.execute("INSERT OR IGNORE INTO runs (run_id, valid_for_analysis) VALUES (?, ?)", (run, valid))
        conn.execute("INSERT OR IGNORE INTO run_models (run_id, profile, mode_entry, mode_output_limit) VALUES (?, ?, ?, 16384)", (run, profile, entry))
        for n in range(passes):
            conn.execute("INSERT INTO task_runs (run_id, profile, section, task_id, task_spec_sha, passed, failure_class) VALUES (?,?,?,?,?,?,?)",
                         (run, profile, f"{profile}__trial-wl-{task}-r{n}", task, spec, int(failure is None), failure))
    conn.close()


class LadderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tasks = self.root / "tasks"
        self.tasks.mkdir()
        (self.tasks / "ops-router-1.json").write_text(json.dumps(OPS))
        (self.tasks / "ops-router-2.json").write_text(json.dumps(OPS | {"id": "ops-router-2"}))
        self.records = {"RUNS.md": self.root / "RUNS.md", "STATE.md": self.root / "STATE.md"}
        self.file = self.root / "ladder.json"
        self.file.write_text(json.dumps({"rungs": [
            {"id": "one", "suite": "workload:ops", "tasks": ["ops-router-1"], "runs": 1, "adds": "first"},
            {"id": "block", "suite": "workload:ops", "adds": "whole category"}]}))

    def step(self, **kw):
        return ladder.next_step(self.root, PROFILE, kw.pop("modes", ("bare",)), task_dir=self.tasks, ladder_file=self.file, **kw)

    def test_the_shipped_ladder_matches_the_real_task_fixtures(self):
        rungs = ladder.load()
        self.assertEqual([r["id"] for r in rungs], ["r01-pilot", "r02-screen", "r03-core", "r04-depth"])
        self.assertEqual(ladder.requirements(rungs[3]), {"depth-scaled-early-instruction": 1})
        self.assertEqual([r.get("stage", "screen") for r in rungs], ["screen", "screen", "full", "full"])
        self.assertEqual(ladder.requirements(rungs[0]), {"lookup-smoke-test": 1})
        self.assertEqual(list(ladder.requirements(rungs[1])), [
            "resume-llm-eval-owner-decision", "advisory-models-to-keep", "ops-fans-gpu-hog", "surgery-split-commit",
            "review-llm-eval-live-recovery", "impl-llm-eval-html-report"])  # the order the owner's screen names them
        screen = ladder.screen_tasks()
        self.assertEqual(len(screen), 7)
        self.assertEqual({t["category"] for t in workload.load_tasks() if t["id"] in screen},
                         {"lookup", "resume", "advisory", "ops", "surgery", "review", "implementation"})  # one per category
        rest = ladder.requirements(rungs[2])
        self.assertEqual(len(rest), 35 - 7)  # the core suite minus the screen
        self.assertFalse(set(rest) & set(screen))
        self.assertTrue(all(n == 1 for n in rest.values()))  # n=1

    def test_a_bad_ladder_is_refused(self):
        self.file.write_text(json.dumps({"rungs": [{"id": "x", "suite": "workload:ops", "tasks": ["nope"]}]}))
        with self.assertRaisesRegex(ValueError, "not in suite"):
            ladder.load(self.file, self.tasks)
        self.file.write_text(json.dumps({"rungs": [{"id": "Bad Id", "suite": "workload:ops"}]}))
        with self.assertRaisesRegex(ValueError, "unique slugs"):
            ladder.load(self.file, self.tasks)

    def test_an_empty_corpus_starts_at_the_first_rung(self):
        step = self.step()
        self.assertEqual((step["status"], step["rung"], step["mode"], step["needs"]), ("next", "one", "bare", {"ops-router-1": 1}))
        self.assertEqual(step["dir"], str(self.root / "bench-tiny-bare-one-1"))

    def test_a_rung_is_done_by_passes_or_model_failures_only(self):
        write_corpus(self.root, failure="model")  # a wrong answer is data
        self.assertEqual(self.step()["rung"], "block")
        self.assertEqual(self.step()["needs"], {"ops-router-1": 1, "ops-router-2": 1})

    def test_infra_capacity_invalid_runs_and_other_modes_do_not_count(self):
        write_corpus(self.root, failure="infra", run="a")
        write_corpus(self.root, failure="capacity", run="b")
        write_corpus(self.root, valid=0, run="c")
        write_corpus(self.root, entry="tiny-agent", run="d")
        write_corpus(self.root, profile="other", run="e")
        self.assertEqual(self.step()["rung"], "one")  # none of these count, the -agent entry's run included

    def test_the_screen_stage_never_runs_a_full_rung_unless_asked(self):
        self.file.write_text(json.dumps({"rungs": [
            {"id": "one", "stage": "screen", "suite": "workload:ops", "tasks": ["ops-router-1"], "runs": 1},
            {"id": "block", "stage": "full", "suite": "workload:ops", "adds": "the rest"}]}))
        write_corpus(self.root)
        self.assertEqual(self.step()["status"], "ladder_complete")  # the default stage is the screen
        self.assertEqual(self.step(stage="full")["rung"], "block")

    def test_a_rung_can_be_everything_else_in_its_suite(self):
        self.file.write_text(json.dumps({"rungs": [
            {"id": "one", "suite": "workload:ops", "tasks": ["ops-router-1"], "runs": 1},
            {"id": "rest", "suite": "workload:ops", "except_rungs": ["one"], "runs": 1}]}))
        self.assertEqual(self.step()["needs"], {"ops-router-1": 1})
        write_corpus(self.root)
        self.assertEqual(self.step()["needs"], {"ops-router-2": 1})
        self.file.write_text(json.dumps({"rungs": [{"id": "rest", "suite": "workload:ops", "except_rungs": ["later"]}]}))
        with self.assertRaisesRegex(ValueError, "do not come before"):
            ladder.load(self.file, self.tasks)

    def test_a_finished_ladder_says_so(self):
        write_corpus(self.root, task="ops-router-1", passes=2)
        write_corpus(self.root, task="ops-router-2", passes=2)
        self.assertEqual(self.step()["status"], "ladder_complete")

    def attempt(self, n, status="complete", sections=None, started=True):
        directory = ladder.attempt_dir(self.root, PROFILE, "bare", "one", n)
        directory.mkdir()
        if started:
            (directory / "runs" / "acceptance-x").mkdir(parents=True)
            (directory / "runs/acceptance-x/state.json").write_text(json.dumps({"status": status, "sections": sections or {}}))
        return directory

    def test_an_infrastructure_stop_blocks_until_the_owner_says_retry(self):
        self.attempt(1, sections={"tiny__trial-wl-ops-router-1-r1": {"status": "infra_fail", "stop": "opencode export failed"}})
        step = self.step()
        self.assertEqual((step["status"], step["rung"]), ("blocked", "one"))
        self.assertIn("infra_fail in trial-wl-ops-router-1-r1: opencode export failed", step["why"])
        self.assertTrue(step["next"].startswith("STOP"))
        retried = self.step(retry=True)
        self.assertEqual((retried["status"], retried["dir"]), ("next", str(self.root / "bench-tiny-bare-one-2")))

    def test_an_unfinished_run_blocks(self):
        self.attempt(1, status="partial")
        self.assertIn("did not finish", self.step()["why"])

    def test_a_prepared_but_unstarted_attempt_is_reused(self):
        self.attempt(1, started=False)
        self.assertEqual(self.step()["dir"], str(self.root / "bench-tiny-bare-one-1"))

    def test_a_completed_attempt_without_a_stop_does_not_block_a_new_attempt(self):
        self.attempt(1)
        self.assertEqual(self.step()["dir"], str(self.root / "bench-tiny-bare-one-2"))


    def finished(self, token="tokentoken12"):
        directory = ladder.attempt_dir(self.root, PROFILE, "bare", "one", 1)
        (directory / "runs/acceptance-x").mkdir(parents=True)
        (directory / "runs/acceptance-x/state.json").write_text(json.dumps({"status": "complete", "sections": {}}))
        (directory / "plan.json").write_text(json.dumps({"plan_hash": token + "0" * 52}))
        return directory

    def test_the_shipped_ladder_asks_for_the_report_and_capture_audit_after_rung_one(self):
        after = ladder.load()[0]["after"]
        self.assertEqual(len(after), 2)
        self.assertIn("report", after[0])
        self.assertIn("host.jsonl", after[1])

    def test_a_record_draft_is_pre_filled_with_facts_and_blanks_for_judgment(self):
        directory = self.finished()
        summary = {"status": "complete", "failing_sections": [], "scorecard": {"ops": {"passed": "1/1"}},
                   "production_restore": {"health": True, "service_matches": True}, "report_html": "/x/report.html"}
        draft = ladder.record_draft(directory, summary, 0, "one", "bare", PROFILE, "2026-10-04")
        self.assertIn("token `tokentoken12`", draft["runs_md_row"])
        self.assertIn("ops 1/1", draft["runs_md_row"])
        self.assertIn("production restored and verified", draft["state_md_row"])
        self.assertIn("<what this means", draft["state_md_row"])
        self.assertTrue((directory / "record-draft.md").exists())
        self.assertTrue(draft["copy_report"].startswith("cp /x/report.html "))


class PrepareRunNextTests(unittest.TestCase):
    """The owner-facing path: prepare --next, then run --next, over the scripted machine."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = root = Path(self.tmp.name) / "runs"
        root.mkdir()
        base = Path(self.tmp.name)
        model = base / "m.gguf"
        model.write_bytes(b"not a model")
        self.preset = base / "models-preset.ini"
        self.preset.write_text(f"[tiny]\nmodel = {model}\nctx-size = 131072\njinja = true\n\n"
                               f"[tiny-agent]\nmodel = {model}\nctx-size = 65536\njinja = true\n")
        self.tasks = base / "tasks"
        self.tasks.mkdir()
        (self.tasks / "ops-router-1.json").write_text(json.dumps(OPS))
        self.config = base / "opencode.jsonc"
        self.config.write_text(PRODUCTION)
        self.binary = base / "bin/opencode"
        self.binary.parent.mkdir()
        self.binary.write_text("#!/bin/sh\necho 1.18.34\n")
        self.binary.chmod(0o755)
        self.file = base / "ladder.json"
        self.file.write_text(json.dumps({"rungs": [
            {"id": "one", "suite": "workload:ops", "tasks": ["ops-router-1"], "runs": 1, "adds": "the ops shims"}]}))
        self.inputs = dict(task_dir=self.tasks, production_config=self.config, binary=self.binary)

    def prepare(self, **kw):
        return acceptance.prepare_next(PROFILE, root=self.root, restore_model="tiny", preset=self.preset,
                                       host=ScriptedHost(**kw.pop("host", {})), ladder_file=self.file, **self.inputs, **kw)

    def test_prepare_next_forwards_an_output_limit_for_the_full_stage(self):
        result, code = self.prepare(output=8192)
        self.assertEqual(code, acceptance.OK, result)
        self.assertEqual(result["card"]["workload"]["mode"]["output_limit"], 8192)

    def test_prepare_next_picks_everything_and_records_the_pending_run(self):
        result, code = self.prepare()
        self.assertEqual(code, acceptance.OK, result)
        self.assertEqual((result["status"], result["rung"], result["mode"]), ("prepared", "one", "bare"))
        self.assertEqual(result["card"]["workload"]["mode"]["entry"], "tiny")
        self.assertEqual(result["card"]["workload"]["tasks"], ["ops-router-1"])
        self.assertIn(f"run --next --approve {result['card']['approval_token']}", result["next"])
        self.assertEqual(ladder.read_pending(self.root)["token"], result["card"]["approval_token"])
        self.assertTrue((self.root / "bench-tiny-bare-one-1/plan.json").exists())

    def test_a_busy_gpu_refuses_and_leaves_nothing_pending(self):
        result, code = self.prepare(host={"procs": [{"pid": 9, "name": "Grim Dawn.exe", "mib": 9000}]})
        self.assertEqual((code, result["status"]), (acceptance.REFUSED, "refused"))
        self.assertTrue(result["next"].startswith("STOP"))
        with self.assertRaises(PermissionError):
            ladder.read_pending(self.root)

    def test_run_next_needs_a_prepared_plan_and_the_right_token(self):
        with self.assertRaisesRegex(PermissionError, "nothing is prepared"):
            acceptance.run_next("x", root=self.root, lock=self.root / "lock")
        self.prepare()
        with self.assertRaisesRegex(PermissionError, "approval token"):
            acceptance.run_next("000000000000", root=self.root, host=ScriptedHost(), lock=self.root / "lock",
                                ladder_file=self.file, task_dir=self.tasks)

    def test_run_next_runs_ingests_clears_pending_and_advances_the_ladder(self):
        result, _ = self.prepare()
        fix = replicas.facts(OPS)
        def driver(adapter, home, manifest, model, relays=(), env=None):
            preset = home / ".config/llama-cpp/models-preset.ini"
            preset.write_text(preset.read_text().replace(fix["wrong"], fix["actual"]))
            state = json.loads((home / replicas.MACHINE).read_text())
            state["units"]["llama-router.service"]["active"] = True
            (home / replicas.MACHINE).write_text(json.dumps(state))
            return FakeWorkloadDriver([[{}, "Fixed the model path and restarted the router."]])
        with patch.object(LiveAdapter, "workload_driver", driver), \
                patch.object(LiveAdapter, "workload_box", lambda self, home, model: FakeBox()):
            summary, code = acceptance.run_next(result["card"]["approval_token"], root=self.root, host=ScriptedHost(),
                                                lock=self.root / "lock", ladder_file=self.file, task_dir=self.tasks)
        self.assertEqual(code, acceptance.OK, summary)
        self.assertEqual((summary["rung"], summary["mode"]), ("one", "bare"))
        self.assertEqual(summary["scorecard"]["ops"]["passed"], "1/1")
        self.assertEqual(summary["after_this"]["status"], "ladder_complete")
        with self.assertRaises(PermissionError):
            ladder.read_pending(self.root)
        done = ladder.table(self.root, PROFILE, ("bare",), task_dir=self.tasks, ladder_file=self.file)
        self.assertEqual([(r["rung"], r["complete"]) for r in done], [("one", True)])

    def test_the_command_line_has_next_flags_and_a_ladder_view(self):
        with patch.object(acceptance, "RUNS_ROOT", self.root), contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(acceptance.main(["ladder", "--profile", "tiny"]), acceptance.OK)
        shown = json.loads(out.getvalue())
        self.assertEqual(shown["rungs"][0]["rung"], "r01-pilot")
        self.assertEqual(shown["next"], "r01-pilot")
        with self.assertRaises(SystemExit):
            acceptance.main(["run", "--approve", "x"])  # no --dir and no --next

    def test_the_first_live_run_waits_for_an_integration_stamp(self):
        stamp = self.root / "stamp.json"
        result, code = self.prepare(integration_stamp=stamp)
        self.assertEqual((code, result["status"]), (acceptance.REFUSED, "blocked_integration"))
        self.assertIn("acceptance integration", result["next"])
        with self.assertRaises(PermissionError):
            ladder.read_pending(self.root)
        stamp.write_text(json.dumps({"source_sha": integration.source_sha(), "at": "now"}))
        result, code = self.prepare(integration_stamp=stamp)
        self.assertEqual((code, result["status"]), (acceptance.OK, "prepared"))
        self.assertIn("run --next", result["watch"] + result["next"])
        self.assertIn("wl_task", result["watch"])

    def test_the_first_live_run_waits_for_rung0(self):
        with patch.object(acceptance.rung0, "run",
                          lambda limits: ({"passed": False, "next": "STOP: caps too low"}, acceptance.REFUSED)):
            result, code = self.prepare(rung0_gate=acceptance.rung0.GATE)
        self.assertEqual((code, result["status"]), (acceptance.REFUSED, "blocked_rung0"))
        self.assertIn("rung 0", result["next"])
        with self.assertRaises(PermissionError):
            ladder.read_pending(self.root)

    def test_rung0_passes_and_prepare_continues(self):
        card = {"approval_token": "tok", "estimated_minutes": 1, "next": "x",
                "workload": {"mode": {"entry": "tiny", "context": 1, "output_limit": 16384}}}
        with patch.object(acceptance, "prepare", lambda *a, **k: (card, acceptance.OK)):
            result, code = self.prepare(rung0_gate=acceptance.rung0.GATE)
        self.assertEqual(code, acceptance.OK, result)
        self.assertEqual(result["status"], "prepared")

    def test_after_a_corpus_has_a_workload_run_the_stamp_is_not_asked_for(self):
        write_corpus(self.root, task="other-task", run="earlier")
        result, code = self.prepare(integration_stamp=self.root / "missing.json")
        self.assertEqual(code, acceptance.OK, result)

    def test_run_next_hands_back_draft_rows_and_nothing_waits_on_recording_them(self):
        result, _ = self.prepare()
        fix = replicas.facts(OPS)
        def driver(adapter, home, manifest, model, relays=(), env=None):
            preset = home / ".config/llama-cpp/models-preset.ini"
            preset.write_text(preset.read_text().replace(fix["wrong"], fix["actual"]))
            state = json.loads((home / replicas.MACHINE).read_text())
            state["units"]["llama-router.service"]["active"] = True
            (home / replicas.MACHINE).write_text(json.dumps(state))
            return FakeWorkloadDriver([[{}, "Fixed the model path and restarted the router."]])
        with patch.object(LiveAdapter, "workload_driver", driver), \
                patch.object(LiveAdapter, "workload_box", lambda self, home, model: FakeBox()):
            summary, code = acceptance.run_next(result["card"]["approval_token"], root=self.root, host=ScriptedHost(),
                                                lock=self.root / "lock", ladder_file=self.file, task_dir=self.tasks,
                                                today="2026-10-04")
        self.assertEqual(code, acceptance.OK, summary)
        token = result["card"]["approval_token"]
        self.assertIn(f"token `{token}`", summary["record"]["runs_md_row"])
        self.assertEqual(summary["after_this"]["status"], "ladder_complete")  # no record gate: the corpus is the record
        again, code = self.prepare()
        self.assertEqual(again["status"], "ladder_complete")


class IntegrationStampTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for tree in integration.TREES:
            (self.root / tree).mkdir()
            (self.root / tree / "a.py").write_text("x = 1\n")
        self.stamp = self.root / "stamp.json"

    def test_no_stamp_or_a_stale_stamp_blocks_and_a_current_one_does_not(self):
        self.assertIn("has not passed", integration.blocker(self.stamp, self.root))
        result, code = integration.run_and_stamp(self.stamp, self.root, [
            "python3", "-c", "import sys; print('Ran 8 tests in 1s', file=sys.stderr); print('OK', file=sys.stderr)"])
        self.assertEqual((code, result["passed"], result["ran"]), (0, True, 8))
        self.assertIsNone(integration.blocker(self.stamp, self.root))
        (self.root / "llm_eval/a.py").write_text("x = 2\n")
        self.assertIn("source changed", integration.blocker(self.stamp, self.root))

    def test_a_failing_tier_writes_no_stamp(self):
        result, code = integration.run_and_stamp(self.stamp, self.root, [
            "python3", "-c", "import sys; print('Ran 8 tests in 1s\\nFAILED (failures=1)', file=sys.stderr); sys.exit(1)"])
        self.assertEqual((code, result["passed"]), (1, False))
        self.assertFalse(self.stamp.exists())
        self.assertTrue(result["next"].startswith("The integration tier failed"))


if __name__ == "__main__":
    unittest.main()
