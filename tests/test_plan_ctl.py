"""plan_ctl: the launcher, watcher and report an unattended agent drives (RUNCARD.md)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

TOOL = Path(__file__).resolve().parents[1] / "tools" / "plan_ctl.py"
spec = importlib.util.spec_from_file_location("plan_ctl", TOOL)
ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctl)


class LaunchTests(unittest.TestCase):
    def test_the_target_options_are_read_from_the_runner_arguments(self):
        args = ctl.target_args(["--runtime", "llama.cpp", "--profile", "occamy-1.0", "--context", "131072",
                                "--compare", "x", "--only", "0,1"])
        self.assertEqual((args.runtime, args.profile, args.context, args.only), ("llama.cpp", "occamy-1.0", 131072, "0,1"))

    def test_a_refused_plan_never_starts_the_unit(self):
        with mock.patch.object(ctl, "refusals", return_value=["uncommitted harness changes"]), \
                mock.patch.object(ctl.plan, "configure"), mock.patch.object(ctl.subprocess, "run") as run:
            result = ctl.launch([])
        self.assertFalse(result["launched"])
        run.assert_not_called()

    def test_the_unit_stops_the_runner_alone_and_waits_for_the_restore(self):
        calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            return mock.Mock(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(ctl, "refusals", return_value=[]), \
                mock.patch.object(ctl.plan, "configure"), mock.patch.object(ctl.plan, "gpu_holders", return_value=[]), \
                mock.patch.object(ctl.plan, "RUNS", Path(tmp)), mock.patch.object(ctl.plan, "PLAN_DIR", Path(tmp) / "p"), \
                mock.patch.object(ctl.plan, "SUMMARY", Path(tmp) / "p/SUMMARY.md"), \
                mock.patch.object(ctl, "STATE", Path(tmp) / "state.json"), mock.patch.object(ctl.subprocess, "run", fake_run), \
                mock.patch.object(ctl.time, "sleep"), mock.patch("llm_eval.integration.blocker", return_value=None):
            ctl.launch(["--only", "0"])
        start = next(c for c in calls if c[0] == "systemd-run")
        self.assertIn("KillMode=mixed", start)  # SIGTERM to the runner only; it forwards SIGINT to the campaign
        self.assertIn(f"TimeoutStopSec={ctl.STOP_TIMEOUT_SECONDS}", start)
        self.assertGreater(ctl.STOP_TIMEOUT_SECONDS, ctl.plan.RESTORE_WAIT_SECONDS)
        script = start[start.index("-c") + 1]
        self.assertIn("tail --pid=$$$$", script)  # systemd turns $$ into $; the shell must see $$
        self.assertEqual(start[-2:], ["--only", "0"])


class PreflightTests(unittest.TestCase):
    QUEUE = """| Row | Run | Arguments or command | Status | Record |
|---|---|---|---|---|
| Q1 | drill | `--only D` | blocked: ram | x |
| Q5 | ref | `--suite accuracy --only documents` | todo | |
| Q2 | noise | `python3 tools/speed_sweep.py --out /nonexistent/x` | partial (left: 2) | |
| Q4 | adopt | - | owner: adopt | |
| Q6 | screen | `--runtime llama.cpp --profile a` | todo | |
"""

    def test_open_rows_stop_at_the_first_owner_row(self):
        rows = ctl.queue_rows(self.QUEUE)
        self.assertEqual([r[0] for r in rows], ["Q5", "Q2"])
        self.assertEqual(rows[0][2], "--suite accuracy --only documents")

    def test_a_low_ram_strata_row_and_a_missing_drill_are_reported_without_launching(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("llm_eval.environment.dirty_paths", return_value=[]), \
                mock.patch.object(ctl, "running", return_value=False), mock.patch.object(ctl.plan, "production_ok", return_value=True), \
                mock.patch.object(ctl.plan, "gpu_holders", return_value=[]), mock.patch.object(ctl.plan, "drill_passed", return_value=False), \
                mock.patch("llm_eval.ramcheck.router_return_mib", return_value=0), \
                mock.patch("llm_eval.ramcheck.meminfo", return_value={"mem_available_mib": 50000}), \
                mock.patch.object(ctl.subprocess, "run") as run:
            queue = Path(tmp) / "QUEUE.md"
            queue.write_text(self.QUEUE)
            result = ctl.preflight(queue)
        self.assertFalse(result["ok"])
        self.assertTrue(any("50,000 MiB RAM available" in p and p.startswith("Q5") for p in result["problems"]), result)
        self.assertTrue(any("no passing signal drill" in p for p in result["problems"]), result)
        run.assert_not_called()

    def test_a_drill_first_queue_with_enough_ram_is_clear(self):
        queue_text = "| Q1 | drill | `--only D` | todo | |\n| Q5 | ref | `--suite accuracy --only documents` | todo | |\n"
        with tempfile.TemporaryDirectory() as tmp, mock.patch("llm_eval.environment.dirty_paths", return_value=[]), \
                mock.patch.object(ctl, "running", return_value=False), mock.patch.object(ctl.plan, "production_ok", return_value=True), \
                mock.patch.object(ctl.plan, "gpu_holders", return_value=[]), mock.patch.object(ctl.plan, "drill_passed", return_value=False), \
                mock.patch("llm_eval.ramcheck.router_return_mib", return_value=0), \
                mock.patch("llm_eval.ramcheck.meminfo", return_value={"mem_available_mib": 60000}), \
                mock.patch.object(ctl.shutil, "disk_usage", return_value=mock.Mock(free=500 * 2**30)):
            queue = Path(tmp) / "QUEUE.md"
            queue.write_text(queue_text)
            result = ctl.preflight(queue)
        self.assertTrue(result["ok"], result)


class ReportTests(unittest.TestCase):
    def record(self, tmp, summary, offset=0, log=""):
        (Path(tmp) / "SUMMARY.md").write_text(summary)
        (Path(tmp) / "runner.log").write_text(log)
        return {"summary": str(Path(tmp) / "SUMMARY.md"), "summary_offset": offset, "log": str(Path(tmp) / "runner.log"),
                "argv": [], "plan_dir": tmp, "started": "now", "started_epoch": 0}

    def test_only_lines_since_this_launch_and_only_the_ones_a_reader_acts_on(self):
        old = "- 2026-10-30 20:23 plan finished: {'D': 'complete'}\n"
        new = ("- 2026-10-31 01:00 core: preparing (core)\n"
               "- 2026-10-31 02:30 phase 2 finished: quality_fail\n")
        with tempfile.TemporaryDirectory() as tmp:
            record = self.record(tmp, old + new, offset=len(old))
            self.assertEqual(ctl.summary_lines(record), ["- 2026-10-31 02:30 phase 2 finished: quality_fail"])
            self.assertEqual(len(ctl.summary_lines(record, full=True)), 2)

    def test_the_next_step_follows_the_stop_rule(self):
        cases = [(("active", "", ""), True, "", "keep waiting"),
                 (("inactive", "success", "0"), True, "- x plan finished: {}\n", "RUNCARD.md section 4"),
                 (("inactive", "success", "0"), True, "- x phase 2 finished: infra_fail\n", "RUNCARD.md section 5"),
                 (("failed", "exit-code", "1"), False, "- x plan finished: {}\n", "start the router")]
        for state, healthy, summary, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                ctl_state = Path(tmp) / "state.json"
                ctl_state.write_text(json.dumps(self.record(tmp, summary)))
                with mock.patch.object(ctl, "STATE", ctl_state), mock.patch.object(ctl, "unit_state", return_value=state), \
                        mock.patch.object(ctl.plan, "production_ok", return_value=healthy):
                    self.assertIn(expected, ctl.report()["next"])

    def test_watch_returns_once_the_unit_stops(self):
        states = iter([("active", "", ""), ("active", "", ""), ("inactive", "success", "0")])
        with mock.patch.object(ctl, "unit_state", side_effect=lambda: next(states, ("inactive", "success", "0"))), \
                mock.patch.object(ctl, "_record", return_value=None), mock.patch.object(ctl.time, "sleep") as sleep, \
                mock.patch.object(ctl, "report", return_value={"next": "x"}):
            self.assertEqual(ctl.watch(1), {"next": "x"})
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
