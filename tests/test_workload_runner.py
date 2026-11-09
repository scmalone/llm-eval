import copy
import contextlib
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from llm_eval.plan import compile_plan, verify_plan
from llm_eval.runner import execute
from llm_eval.workload import opencode_config as oc, replicas
from llm_eval.workload.snapshot import build
from tests.test_workload import make_repo
from tests.test_workload_session import PRODUCTION

ROOT = Path(__file__).resolve().parents[1]
FIX = "def add(a, b):\n    return a + b\n"


class WorkloadCampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        repo, base, answer = make_repo(self.root, with_data=False)
        impl = {"id": "impl-1", "kind": "workload", "category": "implementation", "prompt": "Fix add.",
                "source": {"repo": str(repo), "commit": base, "answer": answer, "withhold": ["tests/test_add.py"]},
                "checks": [{"type": "files", "contains": {"src/app.py": r"a \+ b"}},
                           {"type": "diff_scope", "allow": ["src/**"], "min_changed": 1},
                           {"type": "answer", "all": ["(?i)fixed"]}],
                "limits": {"turns": 3, "wall_seconds": 600}, "runs": 2}
        ops = {"id": "ops-1", "kind": "workload", "category": "ops", "prompt": "The router is down.",
               "source": {"replica": "router-bad-model-path", "seed": 3},
               "checks": [{"type": "state", "file": "~/" + replicas.MACHINE,
                           "equals": {"units/llama-router.service/active": True}}],
               "limits": {"turns": 2, "wall_seconds": 600}, "runs": 2}
        self.tasks = {"impl-1": impl, "ops-1": ops}
        for task in self.tasks.values():
            (self.root / f"{task['id']}.json").write_text(json.dumps(task))
        build(impl, self.root / "snapshots/impl-1")
        replicas.build(ops, self.root / "snapshots/ops-1")
        (self.root / "opencode").write_text("not really opencode")
        (self.root / "PROJECTS.md").write_text("| sandbox | ~/Workspace/sandbox |\n")
        config, ref, _ = oc.build(oc.parse_jsonc(PRODUCTION), "muse-q4", ["muse-q4-agent"])
        suite = json.loads((ROOT / "suites/fake.json").read_text())
        model = suite["models"][0]
        model["fixtures"] = [str(ROOT / "fixtures/smoke-1.json"), "impl-1.json", "ops-1.json"]
        model["workload"] = {"context": 65536, "config": config, "ref": ref,
                             "snapshots": {"impl-1": "snapshots/impl-1", "ops-1": "snapshots/ops-1"},
                             "opencode_binary": str(self.root / "opencode"), "opencode_version": "1.18.34",
                             "home_files": {"Workspace/PROJECTS.md": "PROJECTS.md"}}
        self.suite = suite
        self.write_suite()

    def tearDown(self):
        self.tmp.cleanup()

    def write_suite(self, suite=None):
        (self.root / "suite.json").write_text(json.dumps(suite or self.suite))
        return self.root / "suite.json"

    def plan(self, scenario=None):
        plan = compile_plan(self.write_suite())
        if scenario:
            from llm_eval.plan import fingerprint
            plan["scenario"] = scenario
            plan["plan_hash"] = fingerprint({k: v for k, v in plan.items() if k != "plan_hash"})
        return plan

    def sections(self, state, prefix="fake-muse__"):
        return {k[len(prefix):]: v for k, v in state["sections"].items() if k.startswith(prefix)}

    def test_workload_runs_grade_and_land_in_their_sections(self):
        scenario = {"workload": {"impl-1": {"turns": [[{"src/app.py": FIX}, "Fixed add()."]]}}}
        state = execute(self.plan(scenario), self.root / "runs", "wl")
        sections = self.sections(state)
        self.assertEqual(state["status"], "complete")
        self.assertFalse(state["restoration_required"])
        self.assertEqual((sections["workload_implementation"]["status"], sections["workload_implementation"]["passed"]), ("pass", 2))
        self.assertEqual((sections["workload_ops"]["status"], sections["workload_ops"]["passed"]), ("quality_fail", 0))
        self.assertEqual(sections["workload_review"]["status"], "skipped")
        self.assertEqual(sections["long_context_recall"]["status"], "skipped")
        trial = sections["trial-wl-impl-1-r1"]
        self.assertEqual(trial["context"], 65536)
        self.assertTrue(all(c["pass"] for c in trial["checks"]))
        failed = sections["trial-wl-ops-1-r2"]
        self.assertEqual([c["type"] for c in failed["checks"] if not c["pass"]], ["state"])
        events = (self.root / "runs/wl/events.jsonl").read_text()
        self.assertEqual(events.count('"wl_task"'), 4)
        home = Path(trial["output"]) / "home"
        self.assertTrue((home / "Workspace/PROJECTS.md").exists())
        self.assertTrue((Path(failed["output"]) / "home/.config/llama-cpp/models-preset.ini").exists())
        # Exercise the actual producer -> saved run -> ingestion -> CLI path,
        # rather than fabricating core-labelled task rows for the scorer.
        from llm_eval.metrics.ingest import ingest_run
        from llm_eval.metrics.schema import connect
        db = self.root / "metrics.sqlite"
        with contextlib.closing(connect(db)) as conn:
            ingest_run(conn, self.root / "runs/wl")
            tasks = [dict(r) for r in conn.execute(
                "SELECT DISTINCT task_id AS id, category, task_spec_sha, task_identity_sha FROM task_runs ORDER BY task_id")]
        request = {"version": 2, "suite": {"id": "fake-runner-core", "group": "core", "tasks": tasks},
                   "candidate": {"label": "fake", "selections": [{"run_id": "wl", "profile": "fake-muse"}]}}
        request_path = self.root / "score.json"
        request_path.write_text(json.dumps(request))
        result = subprocess.run([sys.executable, "-m", "llm_eval.metrics", "--db", str(db),
                                 "score", "--request", str(request_path)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        scored = json.loads(result.stdout)["cohorts"]["candidate"]
        self.assertEqual((scored["percentage"], scored["attempts"], scored["passes"]), (50, 4, 2))
        self.assertFalse(scored["selection_coverage"][0]["missing_attempts"])

    def test_the_compaction_point_leaves_room_for_a_full_output_plus_a_margin(self):
        """A full-output request that lands a few hundred tokens over the ceiling is refused, not truncated."""
        from llm_eval.runner import COMPACTION_MARGIN
        self.suite["models"][0]["workload"]["mode"] = {"context": 65536, "output_limit": 8192}
        scenario = {"workload": {"impl-1": {"turns": [[{"src/app.py": FIX}, "Fixed add()."]]}}}
        state = execute(self.plan(scenario), self.root / "runs", "wl")
        trial = self.sections(state)["trial-wl-impl-1-r1"]
        self.assertEqual(trial["compaction_point"], 65536 - 8192 - COMPACTION_MARGIN)
        self.assertGreater(COMPACTION_MARGIN, 0)

    def test_a_crash_mid_workload_resumes_without_redoing_committed_runs(self):
        from llm_eval.workload import session
        real = session.run_task
        calls = []
        def flaky(*args, **kwargs):
            calls.append(args[0]["id"])
            if len(calls) == 3:
                (Path(args[1]) / "home").mkdir(parents=True)  # partial output, as a mid-task crash leaves
                raise KeyboardInterrupt("owner pressed ctrl-c")
            return real(*args, **kwargs)
        plan = self.plan({"workload": {"impl-1": {"turns": [[{"src/app.py": FIX}, "Fixed."]]}}})
        with patch.object(session, "run_task", flaky):
            state = execute(plan, self.root / "runs", "wl")
        self.assertEqual(state["status"], "partial")
        self.assertFalse(state["restoration_required"])
        before = self.sections(state)
        self.assertEqual(sorted(k for k in before if k.startswith("trial-wl")), ["trial-wl-impl-1-r1", "trial-wl-impl-1-r2"])
        state = execute(plan, self.root / "runs", "wl", resume=True)
        after = self.sections(state)
        self.assertEqual(after["trial-wl-impl-1-r1"]["time"], before["trial-wl-impl-1-r1"]["time"])
        self.assertEqual(sorted(k for k in after if k.startswith("trial-wl")),
                         ["trial-wl-impl-1-r1", "trial-wl-impl-1-r2", "trial-wl-ops-1-r1", "trial-wl-ops-1-r2"])
        self.assertTrue(any(".aborted-" in p.name for p in (self.root / "runs/wl/workload").iterdir()))

    def test_a_server_death_fails_that_run_and_the_campaign_continues(self):
        scenario = {"workload": {"impl-1": {"capacity": True, "turns": [[{"src/app.py": FIX}, "Fixed."]]}}}
        state = execute(self.plan(scenario), self.root / "runs", "wl")
        sections = self.sections(state)
        self.assertEqual(sections["trial-wl-impl-1-r1"]["status"], "capacity_fail")
        self.assertIn("server died", sections["trial-wl-impl-1-r1"]["stop"])
        self.assertIn("trial-wl-ops-1-r2", sections)
        self.assertEqual(state["status"], "complete")

    def test_the_first_infrastructure_failure_stops_the_block_and_restores_production(self):
        from llm_eval.adapters import FakeWorkloadDriver
        def broken(self):
            raise RuntimeError("opencode export failed: database is locked")
        with patch.object(FakeWorkloadDriver, "export", broken):
            state = execute(self.plan(), self.root / "runs", "wl")
        sections = self.sections(state)
        self.assertEqual(sections["trial-wl-impl-1-r1"]["status"], "infra_fail")
        self.assertNotIn("trial-wl-impl-1-r2", sections)
        self.assertNotIn("trial-wl-ops-1-r1", sections)
        self.assertEqual(sections["failure"]["status"], "infra_fail")
        self.assertIn("first infrastructure failure", sections["failure"]["reason"])
        self.assertEqual(sections["workload_implementation"]["status"], "skipped")
        self.assertFalse(state["restoration_required"])
        events = [json.loads(l)["event"] for l in (self.root / "runs/wl/events.jsonl").read_text().splitlines()]
        self.assertIn("wl_stop_on_infra_fail", events)

    def test_wlr007_an_export_failure_fails_that_run_and_the_campaign_continues(self):
        from llm_eval.adapters import FakeWorkloadDriver
        self.suite["models"][0]["workload"]["stop_on_infra_fail"] = False  # --keep-going
        def broken(self):
            raise RuntimeError("opencode export failed: database is locked")
        with patch.object(FakeWorkloadDriver, "export", broken):
            state = execute(self.plan(), self.root / "runs", "wl")
        sections = self.sections(state)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(sections["trial-wl-impl-1-r1"]["status"], "infra_fail")
        self.assertIn("harness failure", sections["trial-wl-impl-1-r1"]["stop"])
        self.assertIn("trial-wl-ops-1-r2", sections)
        self.assertFalse(state["restoration_required"])

    def test_a_target_context_above_the_qualified_rung_runs_at_the_qualified_rung(self):
        self.suite["models"][0]["workload"]["context"] = 131072
        self.suite["models"][0]["workload"]["mode"] = {"output_limit": 16384}
        scenario = {"loads": {"fake-muse:131072": {"oom": True}},
                    "workload": {"impl-1": {"turns": [[{"src/app.py": FIX}, "Fixed add()."]]}}}
        sections = self.sections(execute(self.plan(scenario), self.root / "runs", "wl"))
        trial = sections["trial-wl-impl-1-r1"]
        self.assertEqual((trial["context"], trial["target_context"]), (114688, 131072))
        from llm_eval.runner import COMPACTION_MARGIN
        self.assertEqual(trial["compaction_point"], 114688 - 16384 - COMPACTION_MARGIN)
        self.assertEqual(trial["status"], "pass")
        self.assertEqual(sections["workload_implementation"]["status"], "pass")

    def test_the_driver_gets_a_config_whose_context_is_the_one_actually_loaded(self):
        from llm_eval.adapters import FakeAdapter
        self.suite["models"][0]["workload"]["context"] = 131072
        seen = []
        real = FakeAdapter.workload_driver
        def spy(adapter, home, manifest, model, relays=(), env=None):
            entry = model["workload"]["ref"].split("/", 1)[1]
            seen.append(model["workload"]["config"]["provider"]["local-llama"]["models"][entry]["limit"]["context"])
            return real(adapter, home, manifest, model, relays, env)
        with patch.object(FakeAdapter, "workload_driver", spy):
            execute(self.plan({"loads": {"fake-muse:131072": {"oom": True}}}), self.root / "runs", "wl")
        self.assertEqual(set(seen), {114688})
        entry = self.suite["models"][0]["workload"]["ref"].split("/", 1)[1]
        self.assertEqual(self.suite["models"][0]["workload"]["config"]["provider"]["local-llama"]["models"][entry]["limit"]["context"], 131072)

    def test_plan_pins_snapshots_binary_and_home_files(self):
        plan = self.plan()
        bundle = str((self.root / "snapshots/impl-1/repo.bundle").resolve())
        self.assertIn(bundle, plan["files"])
        self.assertIn(str((self.root / "opencode").resolve()), plan["files"])
        self.assertIn(str((self.root / "PROJECTS.md").resolve()), plan["files"])
        verify_plan(plan)
        (self.root / "snapshots/impl-1/hidden/tests/test_add.py").write_text("# swapped\n")
        with self.assertRaisesRegex(ValueError, "planned input changed"):
            verify_plan(plan)

    def test_plan_refuses_bad_workload_blocks(self):
        for change in ({"context": 12345}, {"ref": "deepinfra/x"}, {"snapshots": {"impl-1": "snapshots/impl-1"}},
                       {"snapshots": {"impl-1": "snapshots/ops-1", "ops-1": "snapshots/ops-1"}},
                       {"opencode_version": ""}, {"home_files": {"../escape": "PROJECTS.md"}}):
            suite = copy.deepcopy(self.suite)
            suite["models"][0]["workload"].update(change)
            with self.assertRaises(ValueError, msg=str(change)):
                compile_plan(self.write_suite(suite))
        suite = copy.deepcopy(self.suite)
        suite["models"][0]["workload"]["config"]["provider"]["deepinfra"] = {"options": {"baseURL": "https://x"}}
        with self.assertRaises(ValueError):
            compile_plan(self.write_suite(suite))


if __name__ == "__main__":
    unittest.main()
