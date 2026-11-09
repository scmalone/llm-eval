import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import acceptance
from llm_eval.adapters import FakeBox, FakeWorkloadDriver
from llm_eval.live import LiveAdapter
from llm_eval.workload import prepare as workload, replicas
from test_live import ScriptedHost

PRODUCTION = json.dumps({"share": "disabled", "provider": {"local-llama": {
    "options": {"baseURL": "{env:OC_LLAMA_BASEURL}"},
    "models": {"tiny": {"limit": {"context": 131072, "output": 32768}},
               "tiny-agent": {"limit": {"context": 65536, "output": 12288}}}}},
    "agent": {"build": {"model": "{env:OC_MODEL_BALANCED}"}}})
OPS = {"id": "ops-router-1", "kind": "workload", "category": "ops", "prompt": "The router is down; fix it.",
       "source": {"replica": "router-bad-model-path", "seed": 3},
       "checks": [{"type": "state", "file": "~/" + replicas.MACHINE, "equals": {"units/llama-router.service/active": True}}],
       "limits": {"turns": 2, "wall_seconds": 600}}


class WorkloadAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model = self.root / "m.gguf"
        model.write_bytes(b"not a model")
        self.preset = self.root / "models-preset.ini"
        self.preset.write_text(f"[tiny]\nmodel = {model}\nctx-size = 131072\njinja = true\n\n"
                               f"[tiny-agent]\nmodel = {model}\nctx-size = 65536\njinja = true\n")
        self.tasks = self.root / "tasks"
        self.tasks.mkdir()
        (self.tasks / "ops-router-1.json").write_text(json.dumps(OPS))
        self.config = self.root / "opencode.jsonc"
        self.config.write_text(PRODUCTION)
        self.binary = self.root / "bin/opencode"
        self.binary.parent.mkdir()
        self.binary.write_text("#!/bin/sh\necho 1.18.34\n")
        self.binary.chmod(0o755)
        self.dir = self.root / "acceptance"

    def prepare(self, suite="workload:ops", **host):
        self.host = ScriptedHost(**host)
        return acceptance.prepare(self.dir, "tiny", "tiny", self.preset, self.host, suite=suite,
                                  task_dir=self.tasks, production_config=self.config, binary=self.binary)

    def test_prepare_pins_a_workload_suite_without_touching_the_machine(self):
        card, code = self.prepare()
        self.assertEqual(code, acceptance.OK, card)
        # The bare entry at the 131,072 target and the 16,384 screen output limit, whatever production's own limits say.
        self.assertEqual(card["workload"], {"harness": "opencode", "client": "1.18.34", "model_entry": "local-llama/tiny",
                                            "context": 131072, "tasks": ["ops-router-1"], "runs": {"ops-router-1": 1},
                                            "stop_on_infra_fail": True,
                                            "mode": {"entry": "tiny", "context": 131072, "output_limit": 16384,
                                                     "compaction_point": 114688, "entry_explicit": False,
                                                     "context_overridden": False, "output_overridden": False}})
        self.assertEqual(card["fixtures"], ["workload"])  # no smoke trial: the first task proves the harness
        self.assertEqual(card["estimated_minutes"], round((9.4 + 1) * 1 + 5))
        # the task's 600 s wall cap plus the per-task overhead for its one run, plus a 300 s load
        self.assertEqual(card["limits_seconds"]["model_seconds"], 600 + workload.TASK_OVERHEAD_SECONDS + 300)
        self.assertEqual({c[2] for c in self.host.calls}, {"show"})
        plan = json.loads((self.dir / "plan.json").read_text())
        self.assertIn(str(self.config.resolve()), plan["files"])
        self.assertTrue((self.dir / "snapshots/ops-router-1/home.tar").exists())

    def prepare_with(self, **narrowing):
        self.host = ScriptedHost()
        return acceptance.prepare(self.dir, "tiny", "tiny", self.preset, self.host, suite="workload:ops",
                                  task_dir=self.tasks, production_config=self.config, binary=self.binary, **narrowing)

    def test_one_task_one_run_is_a_hashed_copy_with_a_short_estimate(self):
        card, code = self.prepare_with(tasks=["ops-router-1"], runs=1)
        self.assertEqual(code, acceptance.OK, card)
        self.assertEqual(card["workload"]["runs"], {"ops-router-1": 1})
        self.assertEqual(card["estimated_minutes"], round((9.4 + 1) * 1 + 5))
        copy = self.dir / "tasks/ops-router-1.json"
        self.assertEqual(json.loads(copy.read_text())["runs"], 1)
        plan = json.loads((self.dir / "plan.json").read_text())
        self.assertIn(str(copy.resolve()), plan["files"])
        self.assertEqual(json.loads((self.tasks / "ops-router-1.json").read_text()).get("runs", 2), 2)

    def test_an_unknown_task_or_a_bad_run_count_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not in suite"):
            self.prepare_with(tasks=["nope"])
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.prepare_with(runs=0)

    def test_the_mode_is_explicit_and_recorded(self):
        card, code = self.prepare_with(entry="tiny")
        self.assertEqual(code, acceptance.OK, card)
        self.assertEqual(card["workload"]["model_entry"], "local-llama/tiny")
        self.assertEqual(card["workload"]["context"], 131072)
        self.assertEqual(card["workload"]["mode"], {"entry": "tiny", "context": 131072, "output_limit": 16384,
                                                    "compaction_point": 131072 - 16384, "entry_explicit": True,
                                                    "context_overridden": False, "output_overridden": False})
        plan = json.loads((self.dir / "plan.json").read_text())
        self.assertEqual(plan["models"][0]["workload"]["mode"]["entry"], "tiny")
        with self.assertRaisesRegex(ValueError, "not a local-llama model entry"):
            self.prepare_with(entry="other")

    def test_a_context_override_changes_the_sandbox_limit_not_production(self):
        card, code = self.prepare_with(entry="tiny", context=98304)
        self.assertEqual(code, acceptance.OK, card)
        self.assertEqual(card["workload"]["mode"]["context"], 98304)
        self.assertTrue(card["workload"]["mode"]["context_overridden"])
        plan = json.loads((self.dir / "plan.json").read_text())
        limit = plan["models"][0]["workload"]["config"]["provider"]["local-llama"]["models"]["tiny"]["limit"]
        self.assertEqual(limit["context"], 98304)
        self.assertEqual(json.loads(self.config.read_text())["provider"]["local-llama"]["models"]["tiny"]["limit"]["context"], 131072)

    def test_an_output_override_changes_the_limit_and_the_compaction_point_only_in_the_sandbox(self):
        card, code = self.prepare_with(output=8192)
        self.assertEqual(code, acceptance.OK, card)
        mode = card["workload"]["mode"]
        self.assertEqual((mode["output_limit"], mode["compaction_point"], mode["output_overridden"]), (8192, 131072 - 8192, True))
        plan = json.loads((self.dir / "plan.json").read_text())
        self.assertEqual(plan["models"][0]["workload"]["config"]["provider"]["local-llama"]["models"]["tiny"]["limit"]["output"], 8192)
        self.assertEqual(json.loads(self.config.read_text())["provider"]["local-llama"]["models"]["tiny"]["limit"]["output"], 32768)

    def test_keep_going_is_pinned_in_the_plan(self):
        card, _ = self.prepare_with(keep_going=True)
        self.assertFalse(card["workload"]["stop_on_infra_fail"])

    def test_narrowing_flags_are_refused_on_the_classic_suite(self):
        self.host = ScriptedHost()
        with self.assertRaisesRegex(ValueError, "workload suites only"):
            acceptance.prepare(self.dir, "tiny", "tiny", self.preset, self.host, suite="classic", runs=1)

    def test_public_opencode_instructions_are_pinned_without_reading_live_home(self):
        (self.root / "AGENTS.md").write_text("# working agreement\n")
        (self.root / "commands").mkdir()
        (self.root / "commands/checkpoint.md").write_text("checkpoint\n")
        card, code = self.prepare()
        self.assertEqual(code, acceptance.OK, card)
        plan = json.loads((self.dir / "plan.json").read_text())
        home_files = plan["models"][0]["workload"]["home_files"]
        self.assertIn(".config/opencode/AGENTS.md", home_files)
        self.assertIn(".config/opencode/commands/checkpoint.md", home_files)
        self.assertIn(home_files[".config/opencode/AGENTS.md"], plan["files"])
        (self.root / "AGENTS.md").write_text("token " + "ghp_" + "a" * 36 + "\n")
        second, code = self.prepare()
        self.assertEqual(code, acceptance.OK, second)
        pinned = Path(home_files[".config/opencode/AGENTS.md"]).read_text()
        self.assertNotIn("working agreement", pinned)

    def test_unknown_suite_or_empty_selection_is_refused(self):
        with self.assertRaises(ValueError):
            self.prepare(suite="vibes")
        with self.assertRaisesRegex(ValueError, "no workload tasks"):
            self.prepare(suite="workload:review")

    def test_a_workload_run_restores_production_and_reports_a_scorecard(self):
        card, _ = self.prepare()
        fix = replicas.facts(OPS)
        def driver(adapter, home, manifest, model, relays=(), env=None):
            # A scripted agent that fixes the replica the way a model would, through the shim.
            preset = home / ".config/llama-cpp/models-preset.ini"
            preset.write_text(preset.read_text().replace(fix["wrong"], fix["actual"]))
            state = json.loads((home / replicas.MACHINE).read_text())
            state["units"]["llama-router.service"]["active"] = True
            (home / replicas.MACHINE).write_text(json.dumps(state))
            return FakeWorkloadDriver([[{}, "Fixed the model path and restarted the router."]])
        with patch.object(LiveAdapter, "workload_driver", driver), \
                patch.object(LiveAdapter, "workload_box", lambda self, home, model: FakeBox()):
            self.host = ScriptedHost()
            summary, code = acceptance.run(self.dir, card["approval_token"], False, self.host, self.root / "lock")
        self.assertEqual(code, acceptance.OK, summary)
        self.assertFalse(summary["restoration_required"])
        self.assertTrue(self.host.unit_active)
        self.assertEqual(summary["scorecard"]["ops"]["passed"], "1/1")
        self.assertNotIn("implementation", summary["scorecard"])
        report = Path(summary["report_html"]).read_text()
        self.assertIn("Workload scorecard", report)
        self.assertIn("1/1", report)

    def test_a_workload_run_lands_in_the_corpus_with_its_mode_checks_and_hashes(self):
        import sqlite3
        card, _ = self.prepare_with(tasks=["ops-router-1"], runs=1)
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
            self.host = ScriptedHost()
            summary, code = acceptance.run(self.dir, card["approval_token"], False, self.host, self.root / "lock")
        self.assertEqual(code, acceptance.OK, summary)
        conn = sqlite3.connect(self.root / "metrics.sqlite")
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        run = conn.execute("SELECT suite_kind, suite_label, opencode_version, valid_for_analysis FROM runs").fetchone()
        self.assertEqual(tuple(run), ("workload", "workload:ops", "1.18.34", 1))
        model = conn.execute("SELECT mode_entry, mode_context, mode_output_limit, mode_compaction_point, "
                             "mode_entry_explicit FROM run_models").fetchone()
        self.assertEqual(tuple(model), ("tiny", 131072, 16384, 114688, 0))
        task = conn.execute("SELECT * FROM task_runs").fetchone()
        self.assertEqual((task["task_id"], task["category"], task["run_n"], task["passed"], task["failure_class"]),
                         ("ops-router-1", "ops", 1, 1, None))
        self.assertRegex(task["task_hash"], "^[0-9a-f]{64}$")
        from llm_eval.metrics.ingest import spec_sha  # the ladder recomputes this from the fixture file
        self.assertEqual(task["task_spec_sha"], spec_sha(workload.load_tasks(self.tasks)[0]))
        checks = conn.execute("SELECT type, passed FROM checks").fetchall()
        self.assertEqual([tuple(c) for c in checks], [("state", 1)])
        card_row = conn.execute("SELECT mode_entry, category, runs, passes, pass_rate FROM v_scorecard").fetchone()
        self.assertEqual(tuple(card_row), ("tiny", "ops", 1, 1, 1.0))

    def test_the_manual_run_form_is_held_by_the_integration_stamp_before_anything_changes(self):
        card, _ = self.prepare_with(tasks=["ops-router-1"], runs=1)
        self.host = ScriptedHost()
        with self.assertRaisesRegex(PermissionError, "integration tier has not passed"):
            acceptance.run(self.dir, card["approval_token"], False, self.host, self.root / "lock",
                           integration_stamp=self.root / "no-stamp.json")
        self.assertEqual(self.host.calls, [])

    def test_an_infrastructure_stop_is_classified_apart_from_a_model_failure(self):
        import sqlite3
        card, _ = self.prepare_with(tasks=["ops-router-1"])
        def broken(self):
            raise RuntimeError("opencode export failed: database is locked")
        with patch.object(FakeWorkloadDriver, "export", broken), \
                patch.object(LiveAdapter, "workload_box", lambda self, home, model: FakeBox()), \
                patch.object(LiveAdapter, "workload_driver", lambda a, home, manifest, model, relays=(), env=None: FakeWorkloadDriver([[{}, "x"]])):
            self.host = ScriptedHost()
            summary, code = acceptance.run(self.dir, card["approval_token"], False, self.host, self.root / "lock")
        self.assertEqual(code, acceptance.EVAL_FAILURES, summary)
        self.assertFalse(summary["restoration_required"])
        conn = sqlite3.connect(self.root / "metrics.sqlite")
        self.addCleanup(conn.close)
        rows = conn.execute("SELECT section, failure_class, passed FROM task_runs").fetchall()
        self.assertEqual([(r[1], r[2]) for r in rows], [("infra", 0)])  # run 2 never started
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM events WHERE event = 'wl_stop_on_infra_fail'").fetchone()[0], 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM v_scorecard").fetchone()[0], 0)  # not a model verdict


if __name__ == "__main__":
    unittest.main()
