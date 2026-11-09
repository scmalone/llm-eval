"""Synthetic cohorts only: no local model, service, or real run data."""
import contextlib
import copy
import io
import json
import random
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from llm_eval.metrics.__main__ import main
from llm_eval.metrics.ingest import ingest_run, put, task_identity_sha
from llm_eval.metrics.schema import MIGRATIONS, connect
from llm_eval.metrics.scoring import analyze, campaign_bound, score


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "metrics.sqlite"
        self.conn = connect(self.db)
        self.addCleanup(self.conn.close)
        self.tasks = [{"id": "lookup-a", "category": "lookup", "task_spec_sha": "a" * 64,
                       "task_identity_sha": "b" * 64, "family": "utility"},
                      {"id": "review-b", "category": "review", "task_spec_sha": "c" * 64,
                       "task_identity_sha": "d" * 64, "family": "utility"}]
        self.request = {"version": 2, "suite": {"id": "synthetic-v1", "group": "core", "tasks": self.tasks},
                        "candidate": {"label": "new", "selections": []},
                        "inference": {"independent_blocks": True, "fixed_design": True,
                                      "note": "Synthetic independent campaigns, counts fixed before outcomes."}}

    def add(self, run_id, outcomes, *, profile="tiny", block=None, arm="candidate", valid=1):
        put(self.conn, "runs", run_id=run_id, valid_for_analysis=valid, invalid_reason="synthetic outage" if not valid else None,
            backend="fake", harness_sha="e" * 40, harness_dirty=0, opencode_version="synthetic-1",
            llama_server_version="synthetic-1", gpu_name="fake-gpu", gpu_driver="fake-driver",
            gpu_memory_mib=24000, kernel="fake-kernel", limits_json=json.dumps({
                "campaign_seconds": 1000, "model_seconds": 800, "load_seconds": 100, "request_seconds": 100}))
        workload_plan = {"version": 1, "tasks": [
            {"id": t["id"], "category": t["category"], "group": t["category"],
             "task_spec_sha": t["task_spec_sha"], "task_identity_sha": t["task_identity_sha"],
             "runs": max(1, len(outcomes[n]))} for n, t in enumerate(self.tasks)]}
        put(self.conn, "run_models", run_id=run_id, profile=profile, model_path="synthetic.gguf", quant="Q4",
            artifacts_json=json.dumps([{"path": "synthetic.gguf", "role": "model", "sha256": "f" * 64}]),
            settings_json='{"model": "synthetic.gguf", "temperature": 0.6}',
            mode_entry="tiny", mode_output_limit=16384, workload_plan_json=json.dumps(workload_plan))
        for index, values in enumerate(outcomes):
            task = self.tasks[index]
            for repeat, value in enumerate(values, 1):
                status = "pass" if value == 1 else "quality_fail" if value == 0 else value
                failure = {"pass": None, "quality_fail": "model", "timeout": "timeout",
                           "infra_fail": "infra", "capacity_fail": "capacity"}.get(status, status)
                put(self.conn, "task_runs", run_id=run_id, profile=profile,
                    section=f"{profile}__{task['id']}__{repeat}", task_id=task["id"],
                    task_spec_sha=task["task_spec_sha"], task_hash=task["task_identity_sha"],
                    task_identity_sha=task["task_identity_sha"], category=task["category"],
                    group_name=task["category"], run_n=repeat, context=65536, target_context=65536,
                    compaction_point=49152, status=status, passed=int(status == "pass"), failure_class=failure)
        selection = {"run_id": run_id, "profile": profile}
        if block is not None:
            selection["block"] = block
        self.request.setdefault(arm, {"label": "old", "selections": []})["selections"].append(selection)
        self.conn.commit()

    def result(self):
        return analyze(self.conn, self.request)

    def test_equal_tasks_not_equal_attempts(self):
        self.add("r1", [[1] * 9, [0]], block="one")
        c = self.result()["cohorts"]["candidate"]
        self.assertEqual((c["success_rate"], c["passes"], c["attempts"]), (0.5, 9, 10))
        self.assertEqual(c["repeat_distribution"], {1: 1, 9: 1})
        self.assertEqual(c["categories"]["review"]["observed_success_rate"], 0)
        self.assertEqual(c["uncertainty"]["status"], "insufficient_evidence")

    def test_ls001_producer_category_and_chat_sweep_groups_are_recognized(self):
        from llm_eval.runner import _group
        self.add("r1", [[1], [1]], block="one")
        with self.conn:
            for task in self.tasks:
                self.conn.execute("UPDATE task_runs SET group_name=? WHERE task_id=?",
                                  (_group(task), task["id"]))
        self.assertEqual(self.result()["cohorts"]["candidate"]["percentage"], 100)
        self.request["suite"]["group"] = "diagnostic"
        with self.conn:
            self.conn.execute("UPDATE task_runs SET group_name=?", (_group({"diagnostic": True}),))
            plan = json.loads(self.conn.execute("SELECT workload_plan_json FROM run_models").fetchone()[0])
            for task in plan["tasks"]:
                task["group"] = "chat_sweep"
            self.conn.execute("UPDATE run_models SET workload_plan_json=?", (json.dumps(plan),))
        self.assertEqual(self.result()["cohorts"]["candidate"]["percentage"], 100)

    def test_ls002_empty_selected_campaign_is_visible_and_cannot_support_bounds(self):
        for n in range(3):
            self.add(f"r{n}", [[1], [1]], block=f"b{n}")
        self.add("missing", [[], []], block="missing")
        c = self.result()["cohorts"]["candidate"]
        self.assertEqual(c["uncertainty"]["status"], "insufficient_evidence")
        self.assertTrue(any(e["run_id"] == "missing" for e in c["excluded"]))

    def test_ls003_bad_json_provenance_cannot_support_bounds(self):
        for n in range(3):
            self.add(f"r{n}", [[1], [1]], block=f"b{n}")
        for table, field, value in (("run_models", "artifacts_json", "[]"),
                                    ("run_models", "settings_json", "[]"),
                                    ("runs", "limits_json", "false")):
            with self.subTest(field=field):
                before = self.conn.execute(f"SELECT {field} FROM {table} LIMIT 1").fetchone()[0]
                with self.conn:
                    self.conn.execute(f"UPDATE {table} SET {field}=?", (value,))
                try:
                    status = self.result()["cohorts"]["candidate"]["uncertainty"]["status"]
                except ValueError:
                    status = "insufficient_evidence"
                self.assertEqual(status, "insufficient_evidence")
                with self.conn:
                    self.conn.execute(f"UPDATE {table} SET {field}=?", (before,))

    def test_ls004_normalized_identity_ignores_repeat_count_and_transient_path(self):
        from llm_eval.metrics.ingest import task_identity_sha
        a = {"id": "synthetic", "prompt": "dummy", "runs": 1, "_path": "/synthetic/one.json"}
        b = {**a, "runs": 2, "_path": "/synthetic/two.json"}
        self.assertEqual(task_identity_sha(a, {}), task_identity_sha(b, {}))
        self.assertNotEqual(task_identity_sha(a, {}), task_identity_sha({**b, "prompt": "changed"}, {}))
        files = {"/one/snapshots/synthetic/module.py": "a" * 64}
        moved = {"/two/snapshots/synthetic/module.py": "a" * 64}
        changed = {"/two/snapshots/synthetic/module.py": "b" * 64}
        self.assertEqual(task_identity_sha(a, files), task_identity_sha(b, moved))
        self.assertNotEqual(task_identity_sha(a, files), task_identity_sha(b, changed))

    def test_missing_one_task_or_repeat_in_a_populated_campaign_is_not_hidden(self):
        self.add("r1", [[1, 1], [1]], block="one")
        self.add("r2", [[1, 1], [1]], block="two")
        self.add("r3", [[1, 1], [1]], block="three")
        with self.conn:
            self.conn.execute("DELETE FROM task_runs WHERE run_id='r2' AND task_id='review-b'")
            self.conn.execute("DELETE FROM task_runs WHERE run_id='r3' AND task_id='lookup-a' AND run_n=2")
        c = self.result()["cohorts"]["candidate"]
        self.assertEqual(c["observed_tasks"], 2)
        self.assertEqual(c["observed_success_rate"], 1)
        self.assertFalse(c["complete"])
        self.assertIsNone(c["percentage"])
        self.assertEqual([(e["run_id"], e["task_id"], e["repeats"]) for e in c["excluded"]],
                         [("r2", "review-b", [1]), ("r3", "lookup-a", [2])])
        self.assertIn("missing_planned_attempts", c["uncertainty"]["reasons"])

    def test_profile_failure_with_no_rows_is_preserved(self):
        for n in range(3):
            self.add(f"r{n}", [[1], [1]], block=f"b{n}")
        self.add("capacity", [[], []], block="capacity")
        with self.conn:
            put(self.conn, "sections", run_id="capacity", key="tiny__failure", status="capacity_fail", reason="synthetic OOM")
        c = self.result()["cohorts"]["candidate"]
        failure = next(e for e in c["excluded"] if e["reason"] == "profile_failure")
        self.assertEqual(failure["detail"]["status"], "capacity_fail")
        self.assertEqual(c["uncertainty"]["status"], "insufficient_evidence")

    def test_saved_plan_counts_cannot_be_redeclared_to_hide_missing_work(self):
        self.add("r1", [[1, 1], [1]], block="one")
        selection = self.request["candidate"]["selections"][0]
        selection["expected_attempts"] = {"lookup-a": 1, "review-b": 1}
        with self.assertRaisesRegex(ValueError, "disagrees with saved plan"):
            self.result()

    def test_malformed_saved_plan_cannot_hide_a_selected_task(self):
        self.add("r1", [[1], [1]], block="one")
        self.add("r2", [[1], [1]], block="two")
        original = self.conn.execute("SELECT workload_plan_json FROM run_models LIMIT 1").fetchone()[0]
        for change in ({"group": []}, {"group": "unknown"}, {"runs": True}, {"task_identity_sha": "bad"}):
            with self.subTest(change=change):
                plan = json.loads(original)
                plan["tasks"][0].update(change)
                with self.conn:
                    self.conn.execute("UPDATE run_models SET workload_plan_json=?", (json.dumps(plan),))
                with self.assertRaises(ValueError):
                    self.result()

    def test_known_partial_plans_can_cover_a_complete_suite_without_inventing_debt(self):
        for n in range(2):
            for index, task in enumerate(self.tasks):
                run_id = f"partial-{n}-{index}"
                outcomes = [[], []]
                outcomes[index] = [1]
                self.add(run_id, outcomes, block=f"block-{n}")
                with self.conn:
                    plan = json.loads(self.conn.execute("SELECT workload_plan_json FROM run_models WHERE run_id=?", (run_id,)).fetchone()[0])
                    plan["tasks"] = [t for t in plan["tasks"] if t["id"] == task["id"]]
                    self.conn.execute("UPDATE run_models SET workload_plan_json=? WHERE run_id=?", (json.dumps(plan), run_id))
        c = self.result()["cohorts"]["candidate"]
        self.assertTrue(c["complete"])
        self.assertEqual(c["percentage"], 100)
        self.assertEqual(c["uncertainty"]["status"], "available")

    def test_missing_plan_metadata_and_legacy_identity_are_explicit(self):
        self.add("r1", [[1], [1]], block="one")
        self.add("r2", [[1], [1]], block="two")
        with self.conn:
            self.conn.execute("UPDATE run_models SET workload_plan_json=NULL")
        c = self.result()["cohorts"]["candidate"]
        self.assertIn("missing_planned_coverage", c["uncertainty"]["reasons"])
        self.request["version"] = 1
        for task in self.tasks:
            task["task_hash"] = task.pop("task_identity_sha")
        c = self.result()["cohorts"]["candidate"]
        self.assertIn("legacy_task_identity_reingest_and_use_request_v2", c["uncertainty"]["reasons"])
        self.assertIsNone(c["percentage"])
        self.assertEqual(c["observed_success_rate"], 1)

    def test_additive_migration_preserves_legacy_rows_and_scoring_does_not_migrate(self):
        path = self.root / "legacy.sqlite"
        legacy = sqlite3.connect(path)
        try:
            for script, _ in MIGRATIONS[:3]:
                legacy.executescript(script)
            legacy.execute("INSERT INTO schema_meta VALUES ('schema_version', '3')")
            legacy.execute("INSERT INTO runs (run_id, valid_for_analysis) VALUES ('legacy', 1)")
            legacy.execute("INSERT INTO run_models (run_id, profile) VALUES ('legacy', 'tiny')")
            legacy.execute("INSERT INTO task_runs (run_id, profile, section, task_id, task_hash, task_spec_sha, category, group_name, run_n, status, passed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           ("legacy", "tiny", "trial", "lookup-a", "b" * 64, "a" * 64, "lookup", "lookup", 1, "pass", 1))
            legacy.commit()
        finally:
            legacy.close()
        request = copy.deepcopy(self.request)
        request["version"] = 1
        request["suite"]["tasks"] = [request["suite"]["tasks"][0]]
        task = request["suite"]["tasks"][0]
        task["task_hash"] = task.pop("task_identity_sha")
        request["candidate"]["selections"] = [{"run_id": "legacy", "profile": "tiny"}]
        before = path.read_bytes()
        result = score(path, request)
        self.assertEqual(result["cohorts"]["candidate"]["observed_success_rate"], 1)
        self.assertEqual(path.read_bytes(), before)
        migrated = connect(path)
        try:
            row = migrated.execute("SELECT task_hash, task_identity_sha FROM task_runs").fetchone()
            self.assertEqual(tuple(row), ("b" * 64, None))
            self.assertIn("workload_plan_json", migrated.execute("SELECT * FROM run_models").fetchone().keys())
        finally:
            migrated.close()

    def test_empty_objects_and_incomplete_artifact_contents_never_permit_bounds(self):
        self.add("r1", [[1], [1]], block="one")
        self.add("r2", [[1], [1]], block="two")
        for table, key, value in (("runs", "limits_json", "{}"), ("run_models", "settings_json", "{}"),
                                  ("run_models", "artifacts_json", "[]"),
                                  ("run_models", "artifacts_json", '[{}]'),
                                  ("run_models", "artifacts_json", json.dumps([
                                      {"path": "synthetic.gguf", "role": "model", "sha256": 10 ** 63}])),
                                  ("runs", "limits_json", '{"request_seconds": false}')):
            with self.subTest(key=key, value=value):
                before = self.conn.execute(f"SELECT {key} FROM {table} LIMIT 1").fetchone()[0]
                with self.conn:
                    self.conn.execute(f"UPDATE {table} SET {key}=?", (value,))
                try:
                    self.assertEqual(self.result()["cohorts"]["candidate"]["uncertainty"]["status"], "insufficient_evidence")
                except ValueError:
                    pass
                finally:
                    with self.conn:
                        self.conn.execute(f"UPDATE {table} SET {key}=?", (before,))

    def test_attempts_inside_a_campaign_do_not_shrink_bounds(self):
        self.add("r1", [[1], [0]], block="one")
        self.add("r2", [[1], [0]], block="two")
        before = self.result()["cohorts"]["candidate"]["uncertainty"]["interval"]
        # Repeating identical work ten times within each same campaign is not
        # ten new independent campaigns, even though the raw count increases.
        self.request["candidate"]["selections"] = []
        self.add("r3", [[1] * 10, [0] * 10], block="one")
        self.add("r4", [[1] * 10, [0] * 10], block="two")
        after = self.result()["cohorts"]["candidate"]["uncertainty"]["interval"]
        self.assertAlmostEqual(before["half_width"], after["half_width"])
        self.assertEqual(before["blocks"], after["blocks"])

    def test_all_pass_and_all_fail_do_not_claim_certainty(self):
        for outcome in (0, 1):
            self.request["candidate"]["selections"] = []
            for n in range(40):
                self.add(f"edge-{outcome}-{n}", [[outcome], [outcome]], block=f"block-{n}")
            interval = self.result()["cohorts"]["candidate"]["uncertainty"]["interval"]
            self.assertGreater(interval["high"] - interval["low"], 0.1)
            self.assertEqual(interval["low"] if outcome == 0 else interval["high"], outcome)

    def test_paired_task_delta_and_shared_campaign_dependence(self):
        for n in range(30):
            self.add(f"new-{n}", [[1], [1]], block=f"pair-{n}")
            self.add(f"old-{n}", [[0], [0]], arm="baseline", block=f"pair-{n}")
        comparison = self.result()["comparison"]
        self.assertEqual(comparison["delta_percentage_points"], 100)
        self.assertEqual(comparison["direction"], "increase")
        paired = comparison["uncertainty"]["interval"]["half_width"]
        for selection in self.request["baseline"]["selections"]:
            selection["block"] = "independent-" + selection["block"]
        independent = self.result()["comparison"]["uncertainty"]["interval"]["half_width"]
        self.assertGreater(paired, independent)
        self.request["candidate"], self.request["baseline"] = self.request["baseline"], self.request["candidate"]
        self.assertEqual(self.result()["comparison"]["direction"], "decrease")

    def test_zero_delta_is_unresolved_not_equivalent(self):
        for n in range(5):
            self.add(f"new-{n}", [[1], [0]], block=f"pair-{n}")
            self.add(f"old-{n}", [[1], [0]], arm="baseline", block=f"pair-{n}")
        comparison = self.result()["comparison"]
        self.assertEqual(comparison["delta_percentage_points"], 0)
        self.assertEqual(comparison["direction"], "unresolved")
        self.assertLess(comparison["uncertainty"]["interval"]["low"], 0)

    def test_missing_tasks_never_get_a_complete_suite_headline(self):
        self.add("r1", [[1], []], block="one")
        self.add("r2", [[1], []], block="two", arm="baseline")
        result = self.result()
        c = result["cohorts"]["candidate"]
        self.assertEqual(c["missing_tasks"], ["review-b"])
        self.assertEqual(c["observed_success_rate"], 1)
        self.assertIsNone(c["percentage"])
        self.assertIsNone(result["comparison"]["delta_percentage_points"])
        self.assertIn("incomplete_suite", c["uncertainty"]["reasons"])

    def test_invalid_infra_capacity_and_unknown_are_visible_exclusions(self):
        self.add("invalid", [[1], [1]], valid=0, block="zero")
        self.add("infra", [["infra_fail"], ["capacity_fail"]], block="one")
        self.add("unknown", [["unexpected"], [0]], block="two")
        self.add("valid", [[1], ["timeout"]], block="three")
        c = self.result()["cohorts"]["candidate"]
        self.assertEqual((c["attempts"], c["passes"]), (3, 1))
        self.assertEqual(c["success_rate"], 0.5)
        self.assertEqual({e["reason"] for e in c["excluded"]},
                         {"invalid_run", "infra_fail", "capacity_fail", "unexpected"})
        self.assertIn("filtered_attempts", c["uncertainty"]["reasons"])

    def test_changed_versions_or_mixed_configurations_fail_closed(self):
        self.add("r1", [[1], [0]], block="one")
        self.add("r2", [[1], [0]], block="two")
        with self.conn:
            self.conn.execute("UPDATE task_runs SET task_identity_sha=? WHERE run_id='r2'", ("f" * 64,))
        with self.assertRaisesRegex(ValueError, "version/category mismatch"):
            self.result()
        with self.conn:
            self.conn.execute("UPDATE task_runs SET task_identity_sha=CASE task_id WHEN 'lookup-a' THEN ? ELSE ? END",
                              ("b" * 64, "d" * 64))
            self.conn.execute("UPDATE task_runs SET context=98304 WHERE run_id='r2'")
        with self.assertRaisesRegex(ValueError, "mixed configurations"):
            self.result()

    def test_cross_cohort_changes_must_be_declared_but_harness_changes_are_rejected(self):
        self.add("new", [[1], [0]], block="one")
        self.add("old", [[1], [0]], block="two", arm="baseline")
        with self.conn:
            self.conn.execute("UPDATE run_models SET settings_json=? WHERE run_id='new'", ('{"temperature": 0.2}',))
        with self.assertRaisesRegex(ValueError, "undeclared cohort differences"):
            self.result()
        self.request["allowed_changes"] = ["settings_json"]
        self.assertIn("settings_json", self.result()["comparison"]["declared_differences"])
        with self.conn:
            self.conn.execute("UPDATE runs SET harness_sha=? WHERE run_id='new'", ("f" * 40,))
        with self.assertRaisesRegex(ValueError, "harness_sha"):
            self.result()

    def test_provenance_blocks_and_fixed_design_are_not_assumed(self):
        self.add("r1", [[1], [0]])
        self.add("r2", [[1], [0]])
        self.request.pop("inference")
        with self.conn:
            self.conn.execute("UPDATE runs SET gpu_driver=NULL, harness_dirty=1")
        reasons = self.result()["cohorts"]["candidate"]["uncertainty"]["reasons"]
        self.assertTrue({"missing_campaign_blocks", "independent_fixed_design_not_confirmed",
                         "missing_provenance:gpu_driver", "unverified_clean_harness"} <= set(reasons))

    def test_no_inference_of_new_families_or_generalization(self):
        self.add("r1", [[1], [0]], block="one")
        transfer = self.result()["transfer"]
        self.assertEqual(transfer["families"], {"utility": 2})
        self.assertEqual(transfer["status"], "unavailable")

    def test_outside_suite_does_not_change_core_or_its_interval(self):
        self.add("r1", [[1], [0]], block="one")
        self.add("r2", [[0], [1]], block="two")
        before = self.result()["cohorts"]["candidate"]
        put(self.conn, "task_runs", run_id="r1", profile="tiny", section="depth", task_id="depth-new",
            group_name="depth", passed=0, status="quality_fail")
        after = self.result()["cohorts"]["candidate"]
        self.assertEqual((before["success_rate"], before["uncertainty"]),
                         (after["success_rate"], after["uncertainty"]))
        self.assertEqual(after["excluded"][-1]["reason"], "outside_suite")

    def test_duplicate_outcomes_and_overlapping_selections_are_rejected(self):
        self.add("r1", [[1], [0]], block="one")
        selection = copy.deepcopy(self.request["candidate"])
        self.request["baseline"] = selection
        with self.assertRaisesRegex(ValueError, "overlapping selection"):
            self.result()
        self.request.pop("baseline")
        with self.conn:
            self.conn.execute("UPDATE task_runs SET passed=0 WHERE status='pass'")
        with self.assertRaisesRegex(ValueError, "inconsistent outcome"):
            self.result()

    def test_cli_replay_is_deterministic_and_database_is_read_only(self):
        self.add("r1", [[1], [0]], block="one")
        self.add("r2", [[0], [1]], block="two")
        path = self.root / "request.json"
        path.write_text(json.dumps(self.request))
        db_before = self.db.read_bytes()
        results = []
        for _ in range(2):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["--db", str(self.db), "score", "--request", str(path)]), 0)
            results.append(json.loads(output.getvalue()))
        self.assertEqual(results[0], results[1])
        self.assertEqual(self.db.read_bytes(), db_before)
        missing = self.root / "missing.sqlite"
        with self.assertRaises(sqlite3.OperationalError):
            score(missing, self.request)
        self.assertFalse(missing.exists())

    def test_bad_input_is_a_cli_error_without_a_traceback(self):
        path = self.root / "bad.json"
        path.write_text('{"version": 1, "suite": []}')
        with contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as caught:
                main(["--db", str(self.db), "score", "--request", str(path)])
        self.assertEqual(caught.exception.code, 2)
        self.assertNotIn("Traceback", error.getvalue())

    def test_raw_run_ingest_to_real_cli_uses_recorded_versions(self):
        from llm_eval.runner import _group
        fixtures = [{"id": t["id"], "kind": "workload", "category": t["category"],
                     "prompt": "Synthetic task", "checks": [{"type": "answer", "all": ["OK"]}]}
                    for t in self.tasks]
        run_dir = self.root / "raw-run"
        run_dir.mkdir()
        plan = {"plan_hash": "a" * 64, "backend": "fake", "models": [
            {"id": "tiny", "fixtures": fixtures, "workload": {"mode": {"entry": "tiny", "output_limit": 16384}}}]}
        state = {"status": "complete", "sections": {
            f"tiny__trial-wl-{task['id']}-r1": {"task": task["id"], "category": task["category"],
                "group": _group(task), "run": 1, "status": "pass", "time": 1,
                "context": 65536, "target_context": 65536, "compaction_point": 49152}
            for task in self.tasks}}
        (run_dir / "plan.json").write_text(json.dumps(plan))
        (run_dir / "state.json").write_text(json.dumps(state))
        ingest_run(self.conn, run_dir, now=1)
        rows = self.conn.execute("SELECT task_id, task_identity_sha, task_spec_sha FROM task_runs ORDER BY task_id").fetchall()
        for task, row in zip(self.tasks, rows):
            task.update(task_identity_sha=row["task_identity_sha"], task_spec_sha=row["task_spec_sha"])
        self.request["candidate"]["selections"] = [{"run_id": "raw-run", "profile": "tiny"}]
        request_path = self.root / "raw-request.json"
        request_path.write_text(json.dumps(self.request))
        result = subprocess.run([sys.executable, "-m", "llm_eval.metrics", "--db", str(self.db),
                                 "score", "--request", str(request_path)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        scored = json.loads(result.stdout)["cohorts"]["candidate"]
        self.assertEqual(scored["percentage"], 100)
        self.assertEqual(scored["attempts"], 2)
        self.assertEqual(scored["uncertainty"]["status"], "insufficient_evidence")
        # Repeat allocation and manifest location change, while fixture content
        # and snapshot stay pinned. Legacy task hashes must remain untouched.
        old_hashes = {r["task_id"]: r["task_hash"] for r in self.conn.execute("SELECT task_id, task_hash FROM task_runs")}
        second = self.root / "raw-repeated"
        second.mkdir()
        second_plan = copy.deepcopy(plan)
        for fixture in second_plan["models"][0]["fixtures"]:
            fixture.update(runs=2, _path="/synthetic/moved.json")
        second_state = copy.deepcopy(state)
        second_state["sections"] = {}
        for fixture in fixtures:
            for repeat in (1, 2):
                second_state["sections"][f"tiny__trial-wl-{fixture['id']}-r{repeat}"] = {
                    **state["sections"][f"tiny__trial-wl-{fixture['id']}-r1"], "run": repeat, "status": "quality_fail"}
        (second / "plan.json").write_text(json.dumps(second_plan))
        (second / "state.json").write_text(json.dumps(second_state))
        ingest_run(self.conn, second, now=2)
        for row in self.conn.execute("SELECT task_id, task_hash FROM task_runs WHERE run_id='raw-repeated'"):
            self.assertNotEqual(row["task_hash"], old_hashes[row["task_id"]])
        self.request["baseline"] = self.request["candidate"]
        self.request["candidate"] = {"label": "repeated", "selections": [{"run_id": "raw-repeated", "profile": "tiny"}]}
        request_path.write_text(json.dumps(self.request))
        result = subprocess.run([sys.executable, "-m", "llm_eval.metrics", "--db", str(self.db),
                                 "score", "--request", str(request_path)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        comparison = json.loads(result.stdout)["comparison"]
        self.assertEqual(comparison["delta_percentage_points"], -100)

    def test_raw_depth_and_diagnostic_flags_ingest_to_real_cli(self):
        from llm_eval.runner import _group
        for group, flag in (("depth", "depth"), ("diagnostic", "diagnostic")):
            with self.subTest(group=group):
                fixture = {"id": "synthetic", "kind": "workload", "category": "lookup", flag: True,
                           "prompt": "Dummy", "checks": [{"type": "answer", "all": ["OK"]}]}
                run_dir = self.root / group
                run_dir.mkdir()
                plan = {"backend": "fake", "models": [{"id": "tiny", "fixtures": [fixture]}]}
                state = {"status": "complete", "sections": {"tiny__trial-wl-synthetic-r1": {
                    "task": "synthetic", "category": "lookup", "group": _group(fixture), "run": 1, "status": "pass"}}}
                (run_dir / "plan.json").write_text(json.dumps(plan))
                (run_dir / "state.json").write_text(json.dumps(state))
                ingest_run(self.conn, run_dir)
                row = self.conn.execute("SELECT * FROM task_runs WHERE run_id=?", (group,)).fetchone()
                request = {"version": 2, "suite": {"id": group, "group": group, "tasks": [{
                    "id": row["task_id"], "category": row["category"], "task_spec_sha": row["task_spec_sha"],
                    "task_identity_sha": row["task_identity_sha"]}]},
                    "candidate": {"label": group, "selections": [{"run_id": group, "profile": "tiny"}]}}
                request_path = self.root / f"{group}.json"
                request_path.write_text(json.dumps(request))
                result = subprocess.run([sys.executable, "-m", "llm_eval.metrics", "--db", str(self.db),
                                         "score", "--request", str(request_path)], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["cohorts"]["candidate"]["percentage"], 100)

    def test_request_rejects_bad_types_unknown_keys_and_duplicate_tasks(self):
        self.add("r1", [[1], [0]], block="one")
        original = copy.deepcopy(self.request)
        cases = [lambda r: r.update(version=True), lambda r: r.update(inference={"confidence": True}),
                 lambda r: r.update(inference={"fixed_design": "yes"}),
                 lambda r: r.update(allowed_changes=[[]]),
                 lambda r: r.update(extra="typo"),
                 lambda r: r["suite"]["tasks"].append(r["suite"]["tasks"][0])]
        for mutate in cases:
            self.request = copy.deepcopy(original)
            mutate(self.request)
            with self.assertRaises(ValueError):
                self.result()


class CoverageSimulationTests(unittest.TestCase):
    def test_known_probability_coverage_with_perfect_within_campaign_dependence(self):
        # Each block shares one Bernoulli outcome across every task. Treating
        # those task outcomes as independent would give severe undercoverage.
        rng = random.Random(20261005)
        campaigns, simulations = 120, 1500
        for probability in (0.02, 0.5, 0.98):
            covered = 0
            for _ in range(simulations):
                successes = sum(rng.random() < probability for _ in range(campaigns))
                interval = campaign_bound(successes / campaigns, [1 / campaigns] * campaigns)
                covered += interval["low"] <= probability <= interval["high"]
            self.assertGreaterEqual(covered / simulations, 0.95)
            self.assertLess(interval["high"] - interval["low"], 0.3)

    def test_paired_anti_correlated_arms_and_unequal_task_repeat_counts(self):
        rng = random.Random(7)
        campaigns, simulations = 100, 1500
        covered = 0
        # Tasks have 100 and 50 repeats; first 50 blocks contain both tasks.
        # Baseline is 1-X, candidate X, so both arms are fully dependent.
        widths = [0.03] * 50 + [0.01] * 50
        truth = 2 * 0.7 - 1
        for _ in range(simulations):
            outcomes = [int(rng.random() < 0.7) for _ in range(campaigns)]
            delta = (sum(outcomes) / 100 + sum(outcomes[:50]) / 50) - 1
            interval = campaign_bound(delta, widths, limits=(-1, 1))
            covered += interval["low"] <= truth <= interval["high"]
        self.assertGreaterEqual(covered / simulations, 0.95)
        self.assertLess(interval["high"] - interval["low"], 0.8)

    def test_more_independent_campaigns_narrow_bounds(self):
        small = campaign_bound(0.5, [0.1] * 10)
        large = campaign_bound(0.5, [0.01] * 100)
        self.assertLess(large["half_width"], small["half_width"])
        with self.assertRaises(ValueError):
            campaign_bound(0.5, [float("nan")])


if __name__ == "__main__":
    unittest.main()
