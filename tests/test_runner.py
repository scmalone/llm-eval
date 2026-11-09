import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_eval.adapters import FakeAdapter
from llm_eval.cli import main
from llm_eval.graders import recall, tool_call, evidence
from llm_eval.plan import compile_plan, fingerprint
from llm_eval.policy import classification, fits, forecast, context_label
from llm_eval.runner import execute
from llm_eval.state import Journal, atomic_json, machine_lock

ROOT = Path(__file__).resolve().parents[1]


def plan_with(scenario=None):
    value = compile_plan(ROOT / "suites/fake.json")
    value["scenario"] = scenario or {}
    value["plan_hash"] = fingerprint({k: v for k, v in value.items() if k != "plan_hash"})
    return value


class AboveLadderRungTests(unittest.TestCase):
    """A llama.cpp preset's own ctx-size is a rung (the 262K presets); any other context above the ladder is not."""
    def compile(self, anchor, ctx_size):
        manifest = json.loads((ROOT / "suites/fake.json").read_text())
        model = manifest["models"][0]
        model.update(anchor=anchor, contexts=[anchor], settings={**model["settings"], "ctx-size": str(ctx_size)})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.json"
            path.write_text(json.dumps(manifest))
            for fixture in model.get("fixtures", []):
                source = (ROOT / "suites" / fixture).resolve()
                (path.parent / fixture).parent.mkdir(parents=True, exist_ok=True)
                (path.parent / fixture).write_bytes(source.read_bytes())
            return compile_plan(path)

    def test_the_preset_s_own_context_compiles_above_the_ladder(self):
        self.assertEqual(self.compile(262144, 262144)["models"][0]["contexts"], [262144])

    def test_a_context_above_the_ladder_that_the_preset_does_not_load_is_refused(self):
        with self.assertRaisesRegex(ValueError, "supported ladder rung"):
            self.compile(262144, 196608)


class PolicyTests(unittest.TestCase):
    def test_memory_boundaries(self):
        self.assertEqual([classification(x) for x in (1023, 1024, 1536, 2048)],
                         ["unsafe", "marginal", "usable", "comfortable"])

    def test_budget_and_forecast(self):
        self.assertTrue(fits(100000, 8192, 114688))
        self.assertFalse(fits(105000, 8192, 114688))
        self.assertEqual(forecast(22000, 12000, 3000, 98304, 131072), 5744)
        self.assertEqual(context_label(65536), "specialist_only")
        self.assertEqual(context_label(131072), "above_64k_fallback")

    def test_recall_rejects_extra_text_duplicate_wrong_order(self):
        for answer in ("a:1\nb:2\nextra", "a:1\na:1\nb:2", "b:2\na:1"):
            self.assertFalse(recall(answer, ["a:1", "b:2"])["pass"])
        self.assertTrue(recall("a:1\nb:2", ["a:1", "b:2"])["pass"])

    def test_tools_reject_extra_argument_and_calls(self):
        call = {"function": {"name": "get", "arguments": '{"id": "1", "extra": 2}'}}
        self.assertFalse(tool_call({"tool_calls": [call]}, "get", {"id": "1"})["pass"])
        self.assertFalse(tool_call({"tool_calls": [call, call]}, "get", {})["pass"])

    def test_evidence_requires_known_fact_source_pair(self):
        self.assertFalse(evidence('[{"fact_id":"a","source":"wrong"}]', ["a"], {"a": ["doc:5"]})["pass"])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_plan(self, scenario=None):
        return execute(plan_with(scenario), self.root, "test")

    def test_plan_never_calls_process_or_adapter(self):
        with patch("subprocess.run", side_effect=AssertionError("spawn")), patch.object(FakeAdapter, "snapshot", side_effect=AssertionError("adapter")):
            self.assertEqual(main(["plan", str(ROOT / "suites/fake.json"), "--out", str(self.root / "plan.json")]), 0)

    def test_full_fake_campaign_and_resume_are_idempotent(self):
        plan = plan_with()
        result = execute(plan, self.root, "test")
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["restoration_required"])
        self.assertEqual(result["sections"]["fake-muse__structured_tools"]["passed"], 10)
        before = (self.root / "test/events.jsonl").read_bytes()
        execute(plan, self.root, "test", resume=True)
        self.assertEqual(before, (self.root / "test/events.jsonl").read_bytes())

    def test_oom_falls_back_without_retrying_oom(self):
        result = self.run_plan({"loads": {"fake-muse:114688": {"oom": True}}})
        section = result["sections"]["fake-muse__context_qualification"]
        self.assertEqual(section["selected_context"], 102400)
        self.assertEqual(section["status"], "capacity_fallback")
        machine = json.loads((self.root / "test/fake-machine.json").read_text())
        self.assertEqual(machine["loads"]["fake-muse:114688"], 1)

    def test_actual_memory_floor_falls_back(self):
        result = self.run_plan({"loads": {"fake-muse:98304": {"free_mib": 100}}})
        self.assertEqual(result["sections"]["fake-muse__context_qualification"]["selected_context"], 86016)

    def test_forecast_rejects_before_load(self):
        result = self.run_plan({"free_mib": 16500})
        self.assertEqual(result["sections"]["fake-muse__context_qualification"]["selected_context"], 102400)
        machine = json.loads((self.root / "test/fake-machine.json").read_text())
        self.assertNotIn("fake-muse:114688", machine["loads"])

    def test_wrong_answer_is_not_retried(self):
        result = self.run_plan({"requests": {"recall-1": {"response": {"content": "wrong"}}}})
        for context in (131072, 65536):
            self.assertEqual(result["sections"][f"fake-muse__trial-recall-1-at-{context}"]["status"], "quality_fail")
        self.assertEqual(result["sections"]["fake-muse__long_context_recall"]["status"], "quality_fail")
        self.assertIsNone(result["sections"]["fake-muse__long_context_recall"]["verified_context"])
        machine = json.loads((self.root / "test/fake-machine.json").read_text())
        # One request per rung from 131072 down to 65536: a wrong answer is never retried in place.
        self.assertEqual(machine["requests"]["recall-1"], 6)

    def test_transient_request_gets_one_retry(self):
        result = self.run_plan({"requests": {"recall-1": {"transient": True}}})
        self.assertEqual(result["sections"]["fake-muse__trial-recall-1-at-131072"]["status"], "pass")
        machine = json.loads((self.root / "test/fake-machine.json").read_text())
        self.assertEqual(machine["requests"]["recall-1"], 2)

    def test_quality_failure_steps_down_to_the_highest_passing_context(self):
        result = self.run_plan({"recall_fails_above": 102400})
        recall = result["sections"]["fake-muse__long_context_recall"]
        self.assertEqual(recall["status"], "pass")
        self.assertEqual(recall["verified_context"], 102400)
        self.assertEqual([d["context"] for d in recall["descent"]], [131072, 114688, 102400])
        self.assertEqual(result["sections"]["fake-muse__structured_tools"]["context"], 102400)
        self.assertEqual(recall["context_class"], "above_64k_fallback")  # 102,400 - 16,384 - 2,048 < 100K

    def test_failed_smoke_stops_the_model_before_the_ladder(self):
        plan = plan_with({"requests": {"smoke-1": {"response": {"content": ""}}}})
        plan["models"][0]["fixtures"].append(json.loads((ROOT / "fixtures/smoke-1.json").read_text()))
        plan["plan_hash"] = fingerprint({k: v for k, v in plan.items() if k != "plan_hash"})
        result = execute(plan, self.root, "test")
        sections = result["sections"]
        self.assertEqual(sections["fake-muse__smoke"]["status"], "quality_fail")
        self.assertEqual(sections["fake-muse__long_context_recall"]["reason"], "smoke failed")
        self.assertFalse([k for k in sections if "trial-recall" in k])
        self.assertFalse(result["restoration_required"])

    def test_an_answer_cut_off_at_the_cap_fails(self):
        from llm_eval.runner import grade
        cut = {"content": "ironwood:8993", "finish_reason": "length"}
        self.assertFalse(grade({"kind": "recall", "expected": ["ironwood:8993"]}, cut)["pass"])

    def test_recall_prompts_are_sized_to_each_context(self):
        result = self.run_plan()
        trial = result["sections"]["fake-muse__trial-recall-1-at-131072"]
        budget = 131072 - 16384 - 2048  # the window opencode has left before compacting, minus reserve
        self.assertLessEqual(trial["prompt_tokens"], budget)
        self.assertGreater(trial["prompt_tokens"], budget * 0.99)
        self.assertEqual(trial["max_output_tokens"], 16384)
        self.assertEqual(result["sections"]["fake-muse__long_context_recall"]["context_class"], "100k_prompt_measured")

    def test_resume_after_partial_descent_reloads_only_for_unfinished_rungs(self):
        plan = plan_with({"recall_fails_above": 98304})
        first = execute(plan, self.root, "test")
        before = (self.root / "test/events.jsonl").read_bytes()
        execute(plan, self.root, "test", resume=True)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(before, (self.root / "test/events.jsonl").read_bytes())

    def test_timeout_restores_and_is_reported(self):
        result = self.run_plan({"requests": {"recall-1": {"timeout": True}}})
        self.assertEqual(result["sections"]["fake-muse__failure"]["status"], "timeout")
        self.assertFalse(result["restoration_required"])

    def test_restore_failure_retains_marker(self):
        result = self.run_plan({"restore_failures": 2})
        self.assertEqual(result["status"], "restore_failed")
        self.assertTrue(result["restoration_required"])

    def test_inactive_production_remains_inactive(self):
        result = self.run_plan({"production_active": False})
        self.assertTrue(result["last_restore"]["restored_inactive"])
        self.assertIsNone(result["last_restore"]["four_token_probe"])

    def test_unknown_workload_blocks_before_quiesce(self):
        result = self.run_plan({"unknown_gpu_clients": ["user-job"]})
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["restoration_required"])
        self.assertNotIn("04_production_quiesce", result["sections"])

    def test_stale_restore_marker_is_recovered_before_continuing(self):
        plan = plan_with({"restore_failures": 2})
        execute(plan, self.root, "test")
        result = execute(plan, self.root, "test", resume=True)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["restoration_required"])
        events = [json.loads(x)["event"] for x in (self.root / "test/events.jsonl").read_text().splitlines()]
        self.assertIn("restoration_complete", events)

    def test_live_run_refuses_before_output_directory_creation(self):
        plan = plan_with()
        plan["backend"] = "live"
        with self.assertRaisesRegex(RuntimeError, "live execution disabled"):
            execute(plan, self.root / "new", "test")
        self.assertFalse((self.root / "new").exists())

    def test_plan_tamper_fails(self):
        plan = plan_with()
        plan["models"][0]["anchor"] = 131072
        with self.assertRaisesRegex(ValueError, "integrity"):
            execute(plan, self.root, "test")

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            execute(plan_with(), self.root, "../bad")

    def test_machine_lock_excludes_second_runner(self):
        with machine_lock(self.root / ".fake-machine.lock"):
            with self.assertRaisesRegex(RuntimeError, "lock"):
                execute(plan_with(), self.root, "test")

    def test_torn_event_and_orphan_section_are_reconciled(self):
        journal = Journal(self.root / "journal")
        journal.section("first", "pass")
        with (journal.root / "events.jsonl").open("ab") as stream:
            stream.write(b'{"partial":')
        atomic_json(journal.root / "sections/second.json", {"section": "second", "status": "pass"})
        recovered = Journal(journal.root)
        self.assertIn("second", recovered.state["sections"])
        self.assertTrue((journal.root / "events.torn").exists())
        events = [json.loads(x) for x in (journal.root / "events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(x.get("section") == "second" for x in events), 1)

    def test_an_interrupt_mid_model_resumes_its_unfinished_sections(self):
        import llm_eval.runner as runner
        real, calls = runner.run_trial, []
        def interrupted(fixture, *args, **kwargs):
            calls.append(fixture["id"])
            if fixture["kind"] == "tools" and len(calls) < 30 and not any(c == "x" for c in calls):
                calls.append("x")
                raise KeyboardInterrupt("ctrl-c")
            return real(fixture, *args, **kwargs)
        plan = plan_with()
        with patch.object(runner, "run_trial", interrupted):
            state = execute(plan, self.root, "test")
        self.assertEqual(state["status"], "partial")
        self.assertFalse(state["restoration_required"])
        self.assertNotIn("fake-muse__teardown", state["sections"])
        state = execute(plan, self.root, "test", resume=True)
        self.assertEqual(state["sections"]["fake-muse__structured_tools"]["passed"], 10)
        self.assertIn("fake-muse__teardown", state["sections"])


if __name__ == "__main__":
    unittest.main()
