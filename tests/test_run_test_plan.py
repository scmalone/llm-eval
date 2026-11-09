"""The unattended plan runner's verdicts and stop paths, each with the 2026-10-30 run that showed the gap."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import time
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_test_plan.py"
spec = importlib.util.spec_from_file_location("run_test_plan", TOOL)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)

TARGET = plan.TARGET_CONTEXT
QUALIFIED = {"m__context_qualification": {"status": "pass", "selected_context": TARGET}}


def trial(status, **extra):
    return {"status": status, "metrics": {"model_seconds": 10}, **extra}


class VerdictTests(unittest.TestCase):
    def test_a_crashed_run_is_never_a_pass(self):
        # 2026-11-01-pi-262144: KeyError mid-campaign, status partial, logged as "complete (all pass)"
        sections = {**QUALIFIED, "m__preflight": {"status": "pass"}, "m__smoke": {"status": "skipped"}}
        status, detail = plan.verdict({"status": "partial", "failure": None}, sections)
        self.assertEqual(status, "infra_fail")
        self.assertIn("partial", detail)

    def test_an_infra_section_is_an_infra_failure_not_a_quality_one(self):
        # the cohort runs carried __failure: infra_fail under a run status of complete
        sections = {**QUALIFIED, "m__failure": {"status": "infra_fail"}, "m__trial-wl-a-r1": trial("pass")}
        self.assertEqual(plan.verdict({"status": "complete"}, sections)[0], "infra_fail")

    def test_a_fallback_context_fails_the_phase(self):
        sections = {"m__context_qualification": {"status": "capacity_fallback", "selected_context": 131072},
                    "m__trial-wl-a-r1": trial("pass")}
        status, detail = plan.verdict({"status": "complete"}, sections, {"graded": True}, {"trial-wl-a-r1"})
        self.assertEqual(status, "assert_fail")
        self.assertIn("131072", detail)

    def test_a_phase_that_graded_nothing_fails(self):
        sections = {**QUALIFIED, "m__workload_implementation": {"status": "skipped"}}
        status, _ = plan.verdict({"status": "complete"}, sections, {"graded": True}, {"trial-wl-a-r1"})
        self.assertEqual(status, "assert_fail")

    def test_an_unmeasured_task_is_named_as_unmeasured(self):
        sections = {**QUALIFIED, "m__trial-wl-a-r1": {"status": "unmeasured"}, "m__trial-wl-b-r1": {"status": "pass"}}
        status, line = plan.verdict({"status": "complete"}, sections, {"graded": True}, {"trial-wl-a-r1", "trial-wl-b-r1"})
        self.assertEqual(status, "assert_fail")
        self.assertIn("1 task-runs unmeasured (stopped short of the depth floor): trial-wl-a-r1", line)
        self.assertNotIn("no workload task was graded", line)
        self.assertNotIn("have no verdict", line)

    def test_a_silently_narrowed_phase_fails(self):
        sections = {**QUALIFIED, "m__trial-wl-a-r1": trial("pass")}
        status, detail = plan.verdict({"status": "complete"}, sections, {"graded": True},
                                      {"trial-wl-a-r1", "trial-wl-b-r1"})
        self.assertEqual(status, "assert_fail")
        self.assertIn("trial-wl-b-r1", detail)

    def test_recall_must_be_verified_at_the_target(self):
        sections = {**QUALIFIED, "m__long_context_recall": {"status": "pass", "verified_context": 65536},
                    "m__structured_tools": {"status": "pass", "passed": 10, "required": 10}}
        status, detail = plan.verdict({"status": "complete"}, sections, {"recall": True, "tools": True})
        self.assertEqual(status, "assert_fail")
        self.assertIn("65536", detail)

    def test_a_recall_or_tools_miss_at_the_target_is_a_result(self):
        sections = {**QUALIFIED,
                    "m__long_context_recall": {"status": "quality_fail", "verified_context": None,
                                               "descent": [{"context": TARGET, "passed": 2, "total": 3}]},
                    "m__trial-recall-1-at-131072": {"status": "quality_fail",
                                                    "grading": {"correct_positions": 8, "expected_count": 12}},
                    "m__structured_tools": {"status": "quality_fail", "passed": 9, "required": 10, "context": TARGET}}
        status, detail = plan.verdict({"status": "complete"}, sections, {"recall": True, "tools": True})
        self.assertEqual(status, "quality_fail")
        line = plan.describe(sections)
        self.assertIn(f"2/3 at {TARGET}", line)
        self.assertIn("8/12 records", line)

    def test_failed_tasks_are_a_result(self):
        sections = {**QUALIFIED, "m__trial-wl-a-r1": trial("pass"), "m__trial-wl-b-r1": trial("quality_fail")}
        status, detail = plan.verdict({"status": "complete"}, sections, {"graded": True},
                                      {"trial-wl-a-r1", "trial-wl-b-r1"})
        self.assertEqual(status, "quality_fail")
        self.assertIn("trial-wl-b-r1", detail)

    def test_everything_measured_and_passed_is_complete(self):
        sections = {**QUALIFIED, "m__trial-wl-a-r1": trial("pass"),
                    "m__long_context_recall": {"status": "pass", "verified_context": TARGET},
                    "m__structured_tools": {"status": "pass", "passed": 10, "required": 10}}
        self.assertEqual(plan.verdict({"status": "complete"}, sections, {"graded": True, "recall": True,
                                                                         "tools": True}, {"trial-wl-a-r1"})[0],
                         "complete")


class KnobTests(unittest.TestCase):
    def test_no_mtp_drops_the_flag_and_its_value(self):
        config = {"args": ["--spec", "4", "--mtp", "/d", "--kv", "int8"]}
        self.assertEqual(plan.KNOBS["no-mtp"](config)["args"], ["--spec", "4", "--kv", "int8"])

    def test_tail_skip_changes_one_variable(self):
        config = {"env": {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_KVA": "0"}}
        self.assertEqual(plan.KNOBS["tail-skip"](config)["env"], {"STRATA_ROUTE_TAIL_SKIP": "1", "STRATA_KVA": "0"})

    def test_an_arm_that_changes_the_request_is_refused(self):
        import llm_eval.paired as paired
        original = paired._contract
        paired._contract = lambda p: p
        try:
            with self.assertRaises(ValueError):
                plan.paired_changes({"settings": {"request": {"temperature": 1.0}}},
                                    {"settings": {"request": {"temperature": 0.6}}})
            self.assertEqual(plan.paired_changes({"settings": {"config": "a"}}, {"settings": {"config": "b"}}),
                             ["settings.config"])
        finally:
            paired._contract = original


class StopPathTests(unittest.TestCase):
    """The old runner's subprocess timeout SIGKILLed the campaign, the one way it cannot restore."""

    def test_an_over_budget_campaign_gets_sigint_and_time_to_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "restored"
            child = Path(tmp) / "child.py"
            child.write_text(textwrap.dedent(f"""
                import time
                try:
                    time.sleep(60)
                except KeyboardInterrupt:
                    time.sleep(0.5)  # the restore
                    open({str(marker)!r}, "w").write("ok")
                    print('{{"status": "partial"}}')
            """))
            original_log, original_wait = plan.log, plan.RESTORE_WAIT_SECONDS
            plan.log, plan.RESTORE_WAIT_SECONDS = (lambda message: None), 10
            try:
                started = time.monotonic()
                out, _, ended = plan._call([sys.executable, str(child)], timeout=1)
            finally:
                plan.log, plan.RESTORE_WAIT_SECONDS = original_log, original_wait
            self.assertEqual(ended, "interrupted")
            self.assertTrue(marker.exists())
            self.assertEqual(json.loads(out)["status"], "partial")
            self.assertLess(time.monotonic() - started, 10)

    def test_a_re_prepare_never_reuses_or_moves_an_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "2026-10-30-core"
            self.assertEqual(plan.fresh(first), first)
            first.mkdir()
            self.assertEqual(plan.fresh(first).name, "2026-10-30-core-attempt2")
            self.assertTrue(first.exists())


class RamRefusalTests(unittest.TestCase):
    """2026-11-01 22:05: a Strata load short of RAM was killed by systemd-oomd; the gate now refuses it instead."""

    def test_a_ram_refusal_stops_the_phase_without_a_re_prepare(self):
        refused = {"m__failure": {"status": "infra_fail", "reason": "refused: 47,400 MiB RAM available, the load "
                                  "needs 54,326 MiB"}}
        names = ("wait_for_gpu", "prepare", "run", "sections_of", "production_ok", "log", "planned_trials")
        original = {n: getattr(plan, n) for n in names}
        prepared, logged = [], []
        plan.wait_for_gpu = lambda: True
        plan.prepare = lambda *a, **k: (prepared.append(a) or Path("/nonexistent"), "token")
        plan.run = lambda *a: {"status": "complete"}
        plan.sections_of = lambda d: refused
        plan.production_ok = lambda: True
        plan.log = logged.append
        plan.planned_trials = lambda d: ()
        try:
            self.assertEqual(plan.phase(False, next(iter(plan.PHASES))), "refused")
        finally:
            for n, f in original.items():
                setattr(plan, n, f)
        self.assertEqual(len(prepared), 1)
        self.assertTrue(any("47,400 MiB" in line for line in logged), logged)


class RamPreflightTests(unittest.TestCase):
    """The plan start refuses a Strata run below the 57,014 MiB floor and logs RAM for every runtime."""

    def run_blocker(self, runtime, available, returning=0):
        from llm_eval import ramcheck
        saved = (plan.RUNTIME, plan.PLAN_DIR, plan.SUMMARY, ramcheck.meminfo, ramcheck.router_return_mib)
        with tempfile.TemporaryDirectory() as tmp:
            plan.RUNTIME, plan.PLAN_DIR, plan.SUMMARY = runtime, Path(tmp), Path(tmp) / "SUMMARY.md"
            ramcheck.meminfo = lambda text=None: {"mem_available_mib": available}
            ramcheck.router_return_mib = lambda uid=None: returning
            try:
                why = plan.ram_blocker()
                rows = [json.loads(l) for l in (Path(tmp) / "ram.jsonl").read_text().splitlines()]
            finally:
                plan.RUNTIME, plan.PLAN_DIR, plan.SUMMARY, ramcheck.meminfo, ramcheck.router_return_mib = saved
        return why, rows

    def test_the_routers_own_memory_counts_as_coming_back(self):
        why, _ = self.run_blocker("strata", 56500, returning=600)
        self.assertIsNone(why)
        why, _ = self.run_blocker("strata", 56000, returning=600)
        self.assertRegex(why, r"^refused: 56,000 MiB RAM available plus 600 MiB the router returns, a Strata run needs")

    def test_router_return_reads_anon_and_shmem_and_is_zero_when_unreadable(self):
        from unittest import mock
        from llm_eval import ramcheck
        stat = "anon 524288000\nfile 17000000000\nshmem 104857600\n"
        with mock.patch.object(ramcheck.Path, "read_text", return_value=stat):
            self.assertEqual(ramcheck.router_return_mib(1000), 600)
        with mock.patch.object(ramcheck.Path, "read_text", side_effect=FileNotFoundError):
            self.assertEqual(ramcheck.router_return_mib(1000), 0)

    def test_strata_below_the_floor_is_refused_and_logged(self):
        why, rows = self.run_blocker("strata", 57013)
        self.assertRegex(why, r"^refused: 57,013 MiB RAM available, a Strata run needs 57,014 MiB")
        self.assertEqual((rows[0]["tag"], rows[0]["mem_available_mib"]), ("plan_start", 57013))

    def test_strata_at_the_floor_passes(self):
        self.assertIsNone(self.run_blocker("strata", 57014)[0])

    def test_llama_cpp_is_logged_not_gated(self):
        why, rows = self.run_blocker("llama.cpp", 100)
        self.assertIsNone(why)
        self.assertEqual(len(rows), 1)


class DescribeTests(unittest.TestCase):
    def test_the_summary_line_carries_cache_categories_and_incomplete(self):
        sections = {"m__context_qualification": {"status": "pass", "selected_context": TARGET,
                                                 "memory": {"cache_slots": 5169}},
                    "m__trial-wl-a-r1": trial("pass"), "m__trial-wl-b-r1": trial("quality_fail", incomplete="output_cap"),
                    "m__workload_implementation": {"status": "quality_fail", "passed": 1, "total": 2}}
        line = plan.describe(sections)
        for part in ("1/2 pass", "1 incomplete", "20 s model time", "implementation 1/2", "5169 slots"):
            self.assertIn(part, line)


class TargetTests(unittest.TestCase):
    def tearDown(self):
        plan.configure()

    def test_the_default_target_is_strata_at_262k_through_pi(self):
        plan.configure()
        self.assertEqual((plan.RUNTIME, plan.PROFILE, plan.TARGET_CONTEXT, plan.HARNESS),
                         ("strata", "strata-iq3s", 262144, "pi"))
        self.assertTrue(plan.label("core").endswith("-strata-iq3s-262144-core"))

    def test_strata_at_131k_uses_its_frozen_config(self):
        plan.configure(context=131072)
        self.assertEqual(plan.TARGET.name, "iq3s-131072.json")

    def test_a_context_without_a_frozen_strata_config_is_refused(self):
        with self.assertRaises(ValueError):
            plan.configure(context=65536)

    def test_a_llama_target_needs_a_profile(self):
        with self.assertRaises(ValueError):
            plan.configure(runtime="llama.cpp")


    def test_a_profile_whose_pi_window_differs_from_its_preset_is_refused(self):
        from unittest import mock
        with mock.patch("llm_eval.workload.pi_config.production_entry", return_value={"contextWindow": 262144}):
            with self.assertRaisesRegex(ValueError, "sync_pi_context"):
                plan.pi_matches_preset("m", {"ctx-size": "131072"})
            plan.pi_matches_preset("m", {"ctx-size": "262144"})
        with mock.patch("llm_eval.workload.pi_config.production_entry", return_value=None):
            plan.pi_matches_preset("m", {"ctx-size": "131072"})  # no pi entry: nothing to compare


class CheckSuiteTests(unittest.TestCase):
    def test_the_check_suite_is_a_fixed_list_every_runtime_can_load(self):
        from llm_eval.ladder import check_tasks
        from llm_eval.workload import prepare as workload
        ids = check_tasks()
        self.assertEqual(len(ids), 9)  # until the three slots fill
        _, tasks = workload.select("check", task_ids=ids)
        self.assertEqual([t["id"] for t in tasks], ids)
        self.assertTrue(any(t.get("depth") for t in tasks))
        self.assertTrue(set(ids) >= set(__import__("llm_eval.ladder", fromlist=["x"]).screen_tasks()))

    def test_a_filled_slot_joins_the_list(self):
        from llm_eval.ladder import check_tasks
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "check.json"
            spec.write_text(json.dumps({"tasks": ["a"], "slots": [{"kind": "x", "task": "b"}, {"kind": "y", "task": None}]}))
            self.assertEqual(check_tasks(spec), ["a", "b"])

    def test_variants_come_from_ini_sections_and_unset_drops_a_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "v.ini"
            ini.write_text("[kv-q4]\ncache-type-k = q4_0\ncache-type-v = q4_0\n\n[no-draft]\nspec-type = unset\n"
                           "\n[template]\nchat-template-file = /t/new.jinja\n")
            found = plan.load_variants(ini)
            self.assertEqual(found["kv-q4"], {"cache-type-k": "q4_0", "cache-type-v": "q4_0"})
            self.assertEqual(found["no-draft"], {"spec-type": None})
            self.assertEqual(found["template"], {"chat-template-file": "/t/new.jinja"})
            ini.write_text("[empty]\n")
            with self.assertRaisesRegex(ValueError, "changes nothing"):
                plan.load_variants(ini)
        plan.VARIANTS.update({"kv-q4": {"cache-type-k": "q4_0"}})
        try:
            self.assertIn("kv-q4", plan.llama_knobs())
            self.assertIn("no-spec", plan.llama_knobs())
        finally:
            plan.VARIANTS.clear()

    def test_the_gate_rule(self):
        self.assertIn("gate passed", plan.gate_verdict([]))
        self.assertIn("three-seed", plan.gate_verdict(["t-p1"]))
        self.assertIn("regression: 2", plan.gate_verdict(["t-p1", "u-p1"]))


class SyncPiContextTests(unittest.TestCase):
    def test_only_preset_models_windows_change_and_nothing_else_is_touched(self):
        from unittest import mock
        spec = importlib.util.spec_from_file_location("sync_pi_context",
                                                      Path(__file__).resolve().parents[1] / "tools/sync_pi_context.py")
        sync = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sync)
        models = {"providers": {"local-llama": {"baseUrl": "http://x", "apiKey": "K", "models": [
            {"id": "a", "contextWindow": 262144, "maxTokens": 32768}, {"id": "b", "contextWindow": 131072},
            {"id": "other", "contextWindow": 1}]}, "strata": {"models": [{"id": "strata", "contextWindow": 262144}]}}}
        changes = sync.plan(models, {"a": 131072, "b": 131072})
        self.assertEqual(changes, [("local-llama", "a", 262144, 131072)])
        out = sync.apply(json.loads(json.dumps(models)), changes)
        self.assertEqual(out["providers"]["local-llama"]["models"][0], {"id": "a", "contextWindow": 131072, "maxTokens": 32768})
        self.assertEqual(out["providers"]["local-llama"]["apiKey"], "K")
        self.assertEqual(out["providers"]["strata"], models["providers"]["strata"])
        with tempfile.TemporaryDirectory() as tmp:
            pi = Path(tmp) / "models.json"
            pi.write_text(json.dumps(models))
            copy = Path(tmp) / "copy.json"
            copy.write_text(json.dumps({"models": {"a": {"contextWindow": 262144}, "strata": {"contextWindow": 262144}}}))
            self.assertEqual(sync.sync_copy({"a": 131072}, copy), ["a"])
            self.assertEqual(json.loads(copy.read_text())["models"]["strata"]["contextWindow"], 262144)
            with mock.patch.object(sync, "contexts", return_value={"a": 131072}):
                sync.main(["--models", str(pi)])  # dry run
                self.assertEqual(json.loads(pi.read_text()), models)
                copy.write_text(json.dumps({"models": {"a": {"contextWindow": 262144}}}))
                with mock.patch.object(sync, "COPY", copy):
                    sync.main(["--models", str(pi), "--write"])
                written = json.loads(pi.read_text())
                self.assertEqual(written["providers"]["local-llama"]["models"][0]["contextWindow"], 131072)
                self.assertEqual(json.loads(copy.read_text())["models"]["a"]["contextWindow"], 131072)
                backups = list(Path(tmp).glob("models.json.*.bak"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(json.loads(backups[0].read_text()), models)


class VerdictModuleTests(unittest.TestCase):
    def test_the_verdict_module_is_outside_restore_code_and_acts_on_nothing(self):
        # Outside RESTORE_CODE so a wording change needs no drill; that holds only while it only reads records.
        self.assertNotIn("llm_eval/phase_verdict.py", plan.RESTORE_CODE)
        import ast
        tree = ast.parse((Path(TOOL).resolve().parents[1] / "llm_eval" / "phase_verdict.py").read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertEqual(imports, [], "phase_verdict imports nothing, so it cannot start, stop or signal")
        self.assertFalse(calls & {"open", "exec", "eval", "__import__"}, calls)

    def test_the_runner_judges_at_its_own_context(self):
        sections = {"m__context_qualification": {"status": "pass", "selected_context": 131072}}
        self.assertEqual(plan.verdict({"status": "complete"}, sections)[0], "assert_fail")


class NarrowTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: dict(v) for k, v in plan.PHASES.items()}

    def tearDown(self):
        plan.PHASES.clear()
        plan.PHASES.update(self.saved)

    def test_a_named_task_narrows_only_its_phase(self):
        named = plan.narrow(["4:depth-scaled-early-instruction", "2:impl-llm-eval-haystack",
                             "2:impl-llm-eval-output-budget"])
        self.assertEqual(named, ["2", "4"])
        self.assertEqual(plan.PHASES["2"]["tasks"], ["impl-llm-eval-haystack", "impl-llm-eval-output-budget"])
        self.assertEqual(plan.PHASES["4"]["tasks"], ["depth-scaled-early-instruction"])
        self.assertEqual(plan.PHASES["2"]["suite"], "core")
        self.assertEqual(plan.PHASES["2"]["name"], "core-subset")
        self.assertNotIn("tasks", plan.PHASES["1"])

    def test_a_task_the_phase_does_not_have_is_refused(self):
        with self.assertRaisesRegex(ValueError, "names no task"):
            plan.narrow(["2:impl-llm-eval-haystak"])
        with self.assertRaisesRegex(ValueError, "names no task"):
            plan.narrow(["3:doc-retrieve-64k"])

    def test_a_malformed_pick_is_refused(self):
        for pick in ("impl-llm-eval-haystack", "9:x", "2:"):
            with self.assertRaises(ValueError):
                plan.narrow([pick])


class DefaultKnobTests(unittest.TestCase):
    """Owner, 2026-11-02: no speculation knob campaign unless one is named."""

    def tearDown(self):
        plan.VARIANTS.clear()

    def test_no_knob_runs_by_default(self):
        self.assertEqual(plan.default_knobs(), [])

    def test_a_variant_file_still_supplies_its_arms(self):
        plan.VARIANTS.update({"ub256": {"ubatch-size": "256"}})
        self.assertEqual(plan.default_knobs(), ["ub256"])


class AccuracySuiteTests(unittest.TestCase):
    """--suite accuracy (plans/2026-11-01-accuracy-suite.md): one phase per part, its own plan folder."""

    def tearDown(self):
        plan.configure()

    def test_the_accuracy_suite_swaps_the_phases_and_the_plan_folder(self):
        plan.configure(suite="accuracy")
        self.assertEqual(list(plan.PHASES), ["core", "depth", "documents", "mined", "game", "vision"])
        self.assertEqual(plan.PHASES["mined"]["suite"], "mined")
        self.assertTrue(plan.PLAN_DIR.name.endswith("-strata-iq3s-262144-accuracy"))
        plan.configure()
        self.assertEqual(list(plan.PHASES), ["0", "1", "2", "3", "4"])
        self.assertTrue(plan.PLAN_DIR.name.endswith("-strata-iq3s-262144-plan"))

    def test_an_unknown_suite_is_refused(self):
        with self.assertRaises(ValueError):
            plan.configure(suite="check")

    def test_a_part_narrows_by_name(self):
        plan.configure(suite="accuracy")
        self.assertEqual(plan.narrow(["documents:doc-retrieve-32k"]), ["documents"])
        self.assertEqual(plan.PHASES["documents"]["tasks"], ["doc-retrieve-32k"])
        with self.assertRaisesRegex(ValueError, "names no task"):
            plan.narrow(["mined:doc-retrieve-32k"])

    def test_vision_is_skipped_for_a_profile_without_image_input(self):
        plan.configure(suite="accuracy")
        original = plan.sees_images, plan.prepare, plan.log
        prepared = []
        plan.sees_images, plan.prepare, plan.log = (lambda: False), (lambda *a, **k: prepared.append(a)), (lambda m: None)
        try:
            self.assertEqual(plan.phase(False, "vision"), "skipped")
        finally:
            plan.sees_images, plan.prepare, plan.log = original
        self.assertEqual(prepared, [])

    def test_strata_iq3s_takes_no_images(self):
        plan.configure(suite="accuracy")
        self.assertFalse(plan.sees_images())


if __name__ == "__main__":
    unittest.main()
