import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from llm_eval import acceptance
from llm_eval.state import machine_lock
from test_live import ScriptedHost


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model = self.root / "m.gguf"
        model.write_bytes(b"not a model")
        self.preset = self.root / "models-preset.ini"
        self.preset.write_text(f"[tiny]\nmodel = {model}\nctx-size = 131072\njinja = true\n")
        self.dir = self.root / "acceptance"
        self.lock = self.root / "machine.lock"

    def prepare(self, **host):
        self.host = ScriptedHost(**host)
        return acceptance.prepare(self.dir, "tiny", "tiny", self.preset, self.host)

    def run_it(self, approve=None, resume=False, **host):
        self.host = ScriptedHost(**host)
        return acceptance.run(self.dir, approve or self.card["approval_token"], resume, self.host, self.lock)

    def test_prepare_is_read_only_and_hands_back_a_token(self):
        self.card, code = self.prepare()
        self.assertEqual(code, acceptance.OK)
        self.assertEqual(self.card["profile"], "tiny")
        self.assertEqual(len(self.card["approval_token"]), 12)
        self.assertTrue((self.dir / "plan.json").exists())
        self.assertEqual(self.card["limits_seconds"]["model_seconds"], 3600)
        self.assertEqual(self.card["recall_prompt_tokens_by_context"][131072], 131072 - 16384 - 2048)
        self.assertEqual({c[2] for c in self.host.calls}, {"show"})

    def test_prepare_stops_on_an_unknown_gpu_client(self):
        self.card, code = self.prepare(procs=[{"pid": 9, "name": "ComfyUI", "mib": 20000}])
        self.assertEqual(code, acceptance.REFUSED)
        self.assertTrue(self.card["next"].startswith("STOP"))

    def test_a_run_keeps_server_timings_and_gpu_samples_tagged_by_trial(self):
        self.card, _ = self.prepare()
        summary, code = self.run_it()
        self.assertEqual(code, acceptance.OK, summary)
        run = self.dir / "runs" / acceptance.run_id(json.loads((self.dir / "plan.json").read_text()))
        recall = json.loads(next((run / "sections").glob("*trial-recall-1-at-131072.json")).read_text())
        self.assertEqual(recall["timings"]["predicted_per_second"], 45.0)
        self.assertEqual(recall["usage"]["prompt_tokens"], 90)
        self.assertGreater(recall["request_seconds"], 0)
        saved = json.loads(next((run / "responses").glob("*trial-recall-1-at-131072.json")).read_text())
        self.assertEqual(saved["timings"]["cache_n"], 0)
        samples = [json.loads(l) for l in (run / "telemetry/gpu.jsonl").read_text().splitlines()]
        self.assertTrue(any(r["tag"] and "trial-recall-1-at-131072" in r["tag"] for r in samples))
        self.assertFalse([r for r in samples if "error" in r], samples)
        self.assertTrue(all("mem_free_mib" in r for r in samples))

    def test_run_refuses_a_wrong_token_before_touching_the_machine(self):
        self.card, _ = self.prepare()
        with self.assertRaises(PermissionError):
            self.run_it(approve="000000000000")
        self.assertEqual(self.host.calls, [])

    def test_approved_run_completes_restores_and_cannot_be_started_twice(self):
        self.card, _ = self.prepare()
        summary, code = self.run_it()
        self.assertEqual(code, acceptance.OK, summary)
        self.assertEqual(summary["verified_recall_context"], 131072)
        self.assertEqual(summary["tools_passed"], "10/10")
        self.assertFalse(summary["restoration_required"])
        self.assertTrue(self.host.unit_active)
        self.assertEqual(summary["peak_output_tokens"]["recall"], 52)
        self.assertEqual(summary["largest_passing_recall_prompt_tokens"] > 100000, True)
        self.assertEqual((summary["failing_sections"], summary["hit_output_cap"]), ([], []))
        with self.assertRaisesRegex(PermissionError, "already started"):
            self.run_it()
        self.assertEqual(acceptance.status(self.dir, self.lock)[1], acceptance.OK)

    def test_run_writes_an_html_report_and_embeds_analysis_on_status(self):
        self.card, _ = self.prepare()
        summary, _ = self.run_it()
        report = Path(summary["report_html"])
        self.assertEqual(report.name, "report.html")
        text = report.read_text()
        for expected in ("Recall by context", "131,072", "10/10", "<!doctype html>"):
            self.assertIn(expected, text)
        self.assertNotIn("Analysis</h2>", text)
        (report.parent / "analysis.html").write_text("<p>read me &amp; <b>weep</b></p>")
        summary, _ = acceptance.status(self.dir, self.lock)
        self.assertIn("<p>read me &amp; <b>weep</b></p>", Path(summary["report_html"]).read_text())

    def test_failed_smoke_is_reported_as_an_evaluation_failure(self):
        self.card, _ = self.prepare()
        self.host = ScriptedHost()
        original = self.host.http
        def silent(method, port, path, body=None, timeout=30):
            if path == "/v1/chat/completions" and port != 8080 and body["messages"][0]["content"].startswith("Reply"):
                return 200, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
            return original(method, port, path, body, timeout)
        self.host.http = silent
        summary, code = acceptance.run(self.dir, self.card["approval_token"], False, self.host, self.lock)
        self.assertEqual(code, acceptance.EVAL_FAILURES)
        self.assertIn("smoke", summary["failing_sections"])
        self.assertEqual(summary["hit_output_cap"], ["trial-smoke-1-at-131072"])

    def test_failed_restore_is_a_stop(self):
        self.card, _ = self.prepare()
        summary, code = self.run_it(refuse_probe=True)
        self.assertEqual(code, acceptance.STOP)
        self.assertTrue(summary["next"].startswith("STOP"))

    def test_status_tells_running_from_crashed(self):
        self.card, _ = self.prepare()
        self.assertEqual(acceptance.status(self.dir, self.lock)[0]["status"], "not_started")
        state_dir = self.dir / "runs" / acceptance.run_id(json.loads((self.dir / "plan.json").read_text()))
        state_dir.mkdir(parents=True)
        (state_dir / "state.json").write_text(json.dumps(
            {"status": "running", "restoration_required": True, "sections": {"tiny__smoke": {}}}))
        with machine_lock(self.lock):
            result, code = acceptance.status(self.dir, self.lock)
        self.assertEqual((result["status"], code), ("running", acceptance.OK))
        result, code = acceptance.status(self.dir, self.lock)
        self.assertEqual((result["status"], code), ("crashed", acceptance.STOP))

    def test_cli_refuses_without_the_owner_token(self):
        self.card, _ = self.prepare()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = acceptance.main(["run", "--dir", str(self.dir), "--approve", "nope"])
        self.assertEqual(code, acceptance.REFUSED)
        self.assertIn("refused", json.loads(out.getvalue()))


if __name__ == "__main__":
    unittest.main()
