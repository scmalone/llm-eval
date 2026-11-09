import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import acceptance
from llm_eval.metrics import __main__ as metrics_cli
from llm_eval.metrics.ingest import KNOWN_INVALID, find_runs, ingest_run
from llm_eval.metrics.logparse import last_stamp, parse_server_log
from llm_eval.metrics.schema import MIGRATIONS, TABLES, connect
from llm_eval.metrics.stats import median, paired_bootstrap, wilson_interval
from test_live import ScriptedHost

SAMPLE = Path(__file__).parent / "data" / "llama-server-sample.log"


class LogParseTests(unittest.TestCase):
    def test_real_server_log_lines_become_one_row_per_request(self):
        requests = parse_server_log(SAMPLE.read_text())
        self.assertEqual([r["task"] for r in requests], [0, 624, 2484])
        first = requests[0]
        self.assertEqual((first["prompt_n"], first["predicted_n"]), (46974, 598))
        self.assertEqual((first["prompt_tps"], first["predicted_tps"]), (3186.05, 47.07))
        self.assertEqual((first["prompt_ms"], first["predicted_ms"], first["total_ms"]), (14743.64, 12683.67, 27427.3))
        self.assertEqual((first["ctx_tokens_after"], first["graphs_reused"], first["truncated"]), (47571, 595, 0))
        self.assertAlmostEqual(first["t_log_s"], 31.770795, places=5)
        self.assertAlmostEqual(requests[1]["t_log_s"], 64.858938, places=5)  # 1.04.858.938: 1 min 4.858938 s
        self.assertIsNone(first["slot_similarity"])
        self.assertEqual(first["cache_n_est"], 0)

    def test_a_cached_prefix_shows_in_similarity_and_the_cache_estimate(self):
        cached = parse_server_log(SAMPLE.read_text())[2]
        self.assertEqual(cached["slot_similarity"], 0.988)
        self.assertEqual((cached["prompt_n"], cached["predicted_n"]), (5, 91))
        self.assertEqual(cached["cache_n_est"], 497 + 1 - 5 - 91)

    def test_an_unfinished_request_keeps_none_and_inf_is_not_a_rate(self):
        text = ("0.01.000.000 I slot launch_slot_: id  0 | task 7 | processing task, is_child = 0\n"
                "0.02.000.000 I slot print_timing: id  0 | task 7 | prompt eval time =       0.00 ms /     0 tokens "
                "(    -nan ms per token,      inf tokens per second)\n")
        (request,) = parse_server_log(text)
        self.assertEqual(request["task"], 7)
        self.assertIsNone(request["prompt_tps"])
        self.assertIsNone(request["predicted_n"])
        self.assertIsNone(request["t_log_s"])
        self.assertEqual(last_stamp(text), 2.0)  # the last stamped line: 0 min 2 s

    def test_startup_lines_and_progress_lines_are_not_requests(self):
        self.assertEqual(parse_server_log("0.00.016.135 I srv  llama_server: initializing ...\n"), [])


class StatsTests(unittest.TestCase):
    def test_wilson_interval_behaves_at_the_edges(self):
        self.assertEqual(wilson_interval(0, 0), (None, None))
        low, high = wilson_interval(3, 3)
        self.assertAlmostEqual(low, 0.4385, places=3)
        self.assertEqual(high, 1.0)
        low, high = wilson_interval(0, 3)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.5615, places=3)
        low, high = wilson_interval(50, 100)
        self.assertAlmostEqual((low + high) / 2, 0.5, places=2)

    def test_paired_bootstrap_is_deterministic_and_checks_its_input(self):
        result = paired_bootstrap([1, 1, 1], [0, 0, 0])
        self.assertEqual((result["mean_diff"], result["low"], result["high"], result["n"]), (1, 1, 1, 3))
        a, b = [1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 0, 0]
        self.assertEqual(paired_bootstrap(a, b, seed=3), paired_bootstrap(a, b, seed=3))
        self.assertLess(paired_bootstrap(a, b)["low"], paired_bootstrap(a, b)["mean_diff"])
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1, 2])

    def test_median(self):
        self.assertEqual((median([3, 1, 2]), median([4, 1, 3, 2]), median([None])), (2, 2.5, None))


class SchemaTests(unittest.TestCase):
    def test_connect_migrates_once_and_creates_every_table_and_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.sqlite"
            connect(path).close()
            conn = connect(path)
            self.addCleanup(conn.close)
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
            self.assertTrue(set(TABLES) <= names)
            self.assertTrue({"v_scorecard", "v_recall_ladder", "v_gpu_by_tag", "v_runs_valid"} <= names)
            self.assertEqual(conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0],
                             str(len(MIGRATIONS)))


def count_all(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}


class IngestClassicRunTests(unittest.TestCase):
    """A real acceptance flow over the scripted machine, then its folder ingested."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model = self.root / "Tiny-Q4_K_M.gguf"
        model.write_bytes(b"not a model")
        preset = self.root / "models-preset.ini"
        preset.write_text(f"[tiny]\nmodel = {model}\nctx-size = 131072\njinja = true\n")
        self.dir = self.root / "acceptance"
        card, _ = acceptance.prepare(self.dir, "tiny", "tiny", preset, ScriptedHost())
        self.summary, self.code = acceptance.run(self.dir, card["approval_token"], False, ScriptedHost(), self.root / "lock")
        self.run_dir = self.dir / "runs" / acceptance.run_id(json.loads((self.dir / "plan.json").read_text()))
        self.db = self.root / "metrics.sqlite"

    def conn(self):
        conn = connect(self.db)
        self.addCleanup(conn.close)
        return conn

    def test_a_finished_run_is_ingested_automatically_beside_the_profile_folder(self):
        self.assertEqual(self.code, acceptance.OK, self.summary)
        self.assertEqual(self.summary["metrics_db"], str(self.db))
        conn = self.conn()
        run = conn.execute("SELECT * FROM runs").fetchone()
        self.assertEqual((run["run_id"], run["status"], run["suite_label"], run["valid_for_analysis"],
                          run["restore_verified"], run["restoration_required"]),
                         (self.run_dir.name, "complete", "classic", 1, 1, 0))
        self.assertGreater(run["wall_seconds"], 0)
        self.assertIn("harness", json.loads(run["environment_json"]))
        model = conn.execute("SELECT * FROM run_models").fetchone()
        self.assertEqual((model["profile"], model["quant"], model["selected_context"], model["verified_recall_context"]),
                         ("tiny", "Q4_K_M", 131072, 131072))
        self.assertIsNone(model["mode_entry"])  # classic runs have no opencode mode

    def test_trial_rows_match_the_section_files_and_keep_timings(self):
        conn = self.conn()
        files = sorted((self.run_dir / "sections").glob("*__trial-*.json"))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0], len(files))
        recall = conn.execute("SELECT * FROM trials WHERE fixture_id = 'recall-1'").fetchone()
        self.assertEqual((recall["fixture_kind"], recall["status"], recall["predicted_tps"], recall["prompt_n"]),
                         ("recall", "pass", 45.0, 90))
        self.assertEqual(len(recall["fixture_hash"]), 64)
        kinds = dict(conn.execute("SELECT fixture_kind, COUNT(*) FROM trials GROUP BY 1").fetchall())
        self.assertEqual(kinds, {"smoke": 1, "recall": 3, "tools": 10})
        ladder = conn.execute("SELECT context, trials, passes FROM v_recall_ladder").fetchall()
        self.assertEqual([tuple(r) for r in ladder], [(131072, 3, 3)])

    def test_gpu_samples_join_to_trials_by_tag(self):
        conn = self.conn()
        tagged = conn.execute("SELECT COUNT(*) FROM v_gpu_by_tag WHERE tag LIKE '%trial-recall-1-at-131072'").fetchone()[0]
        self.assertEqual(tagged, 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM gpu_samples WHERE error IS NOT NULL").fetchone()[0], 0)

    def test_ingest_twice_is_the_same_as_once(self):
        conn = self.conn()
        before = count_all(conn)
        ingest_run(conn, self.run_dir)
        ingest_run(conn, self.run_dir)
        self.assertEqual(count_all(conn), before)
        self.assertGreater(before["trials"], 0)

    def test_events_and_sections_are_kept(self):
        conn = self.conn()
        events = {r[0] for r in conn.execute("SELECT event FROM events")}
        self.assertIn("section_complete", events)
        self.assertEqual(conn.execute("SELECT status FROM sections WHERE key = '06_restore_and_verify'").fetchone()[0], "pass")

    def test_an_incomplete_run_is_not_valid_for_analysis_but_is_kept(self):
        state = json.loads((self.run_dir / "state.json").read_text())
        state["status"] = "partial"
        (self.run_dir / "state.json").write_text(json.dumps(state))
        conn = self.conn()
        ingest_run(conn, self.run_dir)
        row = conn.execute("SELECT valid_for_analysis, invalid_reason FROM runs").fetchone()
        self.assertEqual(row[0], 0)
        self.assertIn("partial", row[1])
        self.assertGreater(conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0], 0)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM v_recall_ladder").fetchone()[0], 0)

    def test_a_run_can_mark_itself_invalid_or_be_known_invalid(self):
        conn = self.conn()
        (self.run_dir / "analysis-notes.json").write_text(json.dumps({"valid_for_analysis": False, "reason": "fan failed"}))
        ingest_run(conn, self.run_dir)
        self.assertEqual(tuple(conn.execute("SELECT valid_for_analysis, invalid_reason FROM runs").fetchone()), (0, "fan failed"))
        (self.run_dir / "analysis-notes.json").unlink()
        plan = json.loads((self.run_dir / "plan.json").read_text())
        with patch.dict(KNOWN_INVALID, {plan["plan_hash"][:12]: "old caps"}):
            ingest_run(conn, self.run_dir)
        self.assertEqual(tuple(conn.execute("SELECT valid_for_analysis, invalid_reason FROM runs").fetchone()), (0, "old caps"))

    def test_server_log_requests_are_ingested_with_an_epoch_estimate(self):
        logs = self.run_dir / "server-logs"
        logs.mkdir(exist_ok=True)
        (logs / "tiny-65536-1.log").write_text(SAMPLE.read_text())
        conn = self.conn()
        ingest_run(conn, self.run_dir)
        rows = conn.execute("SELECT * FROM server_requests ORDER BY task").fetchall()
        self.assertEqual([r["task"] for r in rows], [0, 624, 2484])
        self.assertEqual(rows[0]["load_context"], 65536)
        self.assertEqual(rows[2]["cache_n_est"], 402)
        self.assertLess(rows[0]["t_epoch_est"], rows[2]["t_epoch_est"])
        speed = conn.execute("SELECT * FROM v_speed_by_load_context").fetchone()
        self.assertEqual((speed["load_context"], speed["requests"]), (65536, 3))

    def test_judge_output_lands_beside_the_gates(self):
        (self.run_dir / "judge.json").write_text(json.dumps({
            "endpoint": {"base_url": "https://judge.example/v1", "model": "judge-1"}, "against": "acceptance-other",
            "rubric_sha": "abc", "judged_at": 5.0,
            "results": [{"task": "t1", "verdict": {"correct": "first", "overall": "first"}, "orders": []},
                        {"task": "t2", "scores": {"clear": 4, "tidy": None}, "reason": "ok"}], "skipped": []}))
        conn = self.conn()
        ingest_run(conn, self.run_dir)
        rows = conn.execute("SELECT task_id, mode, criterion, value, overall, judge_model, against_run_id FROM judge_scores "
                            "ORDER BY task_id, criterion").fetchall()
        self.assertEqual([tuple(r) for r in rows], [
            ("t1", "pairwise", "correct", "first", "first", "judge-1", "acceptance-other"),
            ("t2", "reference", "clear", "4", None, "judge-1", "acceptance-other"),
            ("t2", "reference", "tidy", None, None, "judge-1", "acceptance-other")])

    def test_export_writes_one_jsonl_per_table(self):
        out = self.root / "export"
        written = metrics_cli.export(self.db, out)
        self.assertEqual(set(written), set(TABLES))
        lines = (out / "trials.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), written["trials"])
        self.assertIn("fixture_hash", json.loads(lines[0]))

    def test_the_command_line_ingests_a_folder_and_summarises(self):
        db = self.root / "cli.sqlite"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(metrics_cli.main(["--db", str(db), "ingest", "--run", str(self.run_dir)]), 0)
        self.assertEqual(json.loads(out.getvalue())["ingested"][0]["run_id"], self.run_dir.name)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            metrics_cli.main(["--db", str(db), "summary"])
        self.assertEqual(json.loads(out.getvalue())["tables"]["runs"], 1)
        self.assertEqual(find_runs(self.dir), [self.run_dir])

    def test_a_broken_ingest_never_changes_the_run_outcome(self):
        card, _ = acceptance.prepare(self.root / "second", "tiny", "tiny", self.root / "models-preset.ini", ScriptedHost())
        with patch("llm_eval.metrics.ingest.ingest_all", side_effect=RuntimeError("disk full")):
            summary, code = acceptance.run(self.root / "second", card["approval_token"], False, ScriptedHost(), self.root / "lock2")
        self.assertEqual(code, acceptance.OK, summary)
        self.assertIn("disk full", summary["metrics_ingest_error"])
        self.assertNotIn("metrics_db", summary)

    def test_a_folder_that_is_not_a_run_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a run folder"):
            ingest_run(self.conn(), self.root)


if __name__ == "__main__":
    unittest.main()


class RewrittenShaTests(unittest.TestCase):
    """2026-11-06: the trailer strip rewrote main; runs from before it name commits only on the backup branch."""

    def test_a_rewritten_sha_maps_to_main_and_others_pass_through(self):
        from llm_eval.metrics.ingest import main_sha
        self.assertEqual(main_sha("old", {"old": "new"}), "new")
        self.assertEqual(main_sha("other", {"old": "new"}), "other")
        self.assertIsNone(main_sha(None, {}))

    def test_every_mapped_commit_has_the_same_tree_on_main(self):
        import json, subprocess
        from llm_eval.metrics.ingest import REWRITTEN
        if not REWRITTEN.exists():
            self.skipTest("the rewritten-commit map belongs to the private development history")
        repo = REWRITTEN.parents[1]
        tree = lambda sha: subprocess.run(["git", "rev-parse", f"{sha}^{{tree}}"], cwd=repo, capture_output=True,
                                          text=True).stdout.strip()
        mapping = json.loads(REWRITTEN.read_text())["map"]
        self.assertTrue(mapping)
        for old, new in mapping.items():
            if not tree(old):  # the backup branch is absent in this clone: nothing to compare
                continue
            self.assertEqual(tree(old), tree(new), old)
            self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", new, "HEAD"], cwd=repo).returncode, 0)
