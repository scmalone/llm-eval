import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from llm_eval.state import Journal
from llm_eval.telemetry import GpuSampler, sampling
from llm_eval.workload.capture import RawCapture
from llm_eval.workload.driver import OpencodeDriver
from llm_eval.workload.session import run_task
from llm_eval.workload.snapshot import build
from tests.test_live import ScriptedHost, plan_for
from tests.test_workload import make_repo, repo_task
from tests.test_workload_session import FakeBox, FakeDriver


class LocalBox:
    """A box that runs git on the host, for tests that need a real repository."""
    def git(self, args, cwd, timeout=120):
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
        return out.returncode, out.stdout

    def runner(self):
        return FakeBox().runner()


class RawCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "home/Workspace/r"
        self.repo.mkdir(parents=True)
        git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=self.repo,
                                        check=True, capture_output=True)
        git("init", "-q")
        (self.repo / "a.txt").write_text("one\n")
        git("add", "a.txt")
        git("commit", "-q", "-m", "base")
        self.base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True).stdout.strip()
        self.raw = RawCapture(self.root / "raw")

    def test_end_state_keeps_status_diff_log_manifest_and_small_untracked_files(self):
        (self.repo / "a.txt").write_text("two\n")
        (self.repo / "new.txt").write_text("fresh\n")
        (self.repo / "big.bin").write_bytes(b"x" * 300_000)
        self.raw.end_state(LocalBox(), self.repo, self.root / "home", self.base)
        out = self.root / "raw/end-state"
        self.assertIn(" M a.txt", (out / "status.txt").read_text())
        self.assertIn("+two", (out / "diff.txt").read_text())
        self.assertIn("base", (out / "log.txt").read_text())
        self.assertEqual((out / "untracked/new.txt").read_text(), "fresh\n")
        self.assertFalse((out / "untracked/big.bin").exists())  # over the small-file limit: listed, not copied
        manifest = json.loads((out / "files.json").read_text())
        self.assertEqual(manifest["new.txt"]["size"], 6)
        self.assertIn("big.bin", manifest)
        self.assertNotIn(".git/HEAD", manifest)

    def test_a_broken_box_is_noted_and_never_raises(self):
        self.raw.end_state(object(), self.repo, self.root / "home", self.base)
        self.raw.finish()
        errors = json.loads((self.root / "raw/capture-errors.json").read_text())
        self.assertTrue(any(e["step"] == "git-status" for e in errors))
        self.assertTrue((self.root / "raw/end-state/files.json").exists())  # the rest still ran

    def test_turns_are_saved_in_full_and_summarised(self):
        self.raw.turns([{"said": "long prompt " * 100, "events": [{"type": "text"}, {"type": "step_finish"}],
                         "stderr": "INFO message=loop\n", "attempt": 1}])
        out = self.root / "raw"
        self.assertEqual(len((out / "turn-01.events.jsonl").read_text().splitlines()), 2)
        self.assertIn("message=loop", (out / "turn-01.stderr.log").read_text())
        summary = json.loads((out / "turns.json").read_text())
        self.assertEqual((summary[0]["events"], summary[0]["stderr_bytes"]), (2, 18))
        self.assertEqual(len(summary[0]["said"]), len("long prompt " * 100))

    def test_commands_are_recorded_with_their_full_output(self):
        run = self.raw.recording(lambda command, cwd, timeout: (1, "x" * 5000))
        self.assertEqual(run(["pytest"], self.repo, 60), (1, "x" * 5000))
        row = json.loads((self.root / "raw/commands.jsonl").read_text())
        self.assertEqual((row["command"], row["code"], len(row["output"])), (["pytest"], 1, 5000))

    def test_opencode_data_is_copied_up_to_the_cap(self):
        data = self.root / "home/.local/share/opencode/log"
        data.mkdir(parents=True)
        (data / "run.log").write_text("hello")
        self.raw.opencode_data(self.root / "home")
        self.assertEqual((self.root / "raw/opencode-data/.local/share/opencode/log/run.log").read_text(), "hello")


class SessionCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        repo, base, answer = make_repo(self.root, with_data=False)
        self.task = repo_task(repo, base, answer)
        self.task.update(checks=[{"type": "tests", "command": ["python3", "-m", "unittest"]}],
                         followups={"tests": {"command": ["python3", "-m", "unittest"], "max": 1}})
        build(self.task, self.root / "snap")

    def test_a_task_run_leaves_raw_turns_and_every_command(self):
        class Raw(FakeDriver):
            raw_turns = [{"said": "full text", "events": [{"type": "text"}], "stderr": "log line\n"}]
        run_task(self.task, self.root / "run", self.root / "snap", lambda h, m: Raw(h, [({}, "done")]),
                 lambda home: FakeBox())
        raw = self.root / "run/raw"
        self.assertEqual(json.loads((raw / "turns.json").read_text())[0]["said"], "full text")
        commands = [json.loads(l) for l in (raw / "commands.jsonl").read_text().splitlines()]
        self.assertTrue(any("AssertionError" in c["output"] for c in commands))  # follow-up and grader runs

    def test_raw_is_kept_when_the_driver_fails_midway(self):
        class Boom(FakeDriver):
            raw_turns = [{"said": "before the crash", "events": [], "stderr": "boom\n"}]
            def turn(self, text, workdir, timeout):
                raise RuntimeError("opencode died")
        with self.assertRaises(RuntimeError):
            run_task(self.task, self.root / "boom", self.root / "snap", lambda h, m: Boom(h, []),
                     lambda home: FakeBox())
        self.assertEqual(json.loads((self.root / "boom/raw/turns.json").read_text())[0]["said"], "before the crash")


class DriverCaptureTests(unittest.TestCase):
    def test_a_stalled_first_attempt_is_kept_beside_the_retry(self):
        driver = OpencodeDriver.__new__(OpencodeDriver)
        driver.session, driver.ref, driver.turn_args, driver.raw_turns = None, "p/m", [], []
        driver.box = type("B", (), {"inside": lambda self, p: str(p)})()
        attempts = iter([{"stalled": True, "events": [], "stderr": "stuck\n", "returncode": -9, "timed_out": False, "lingered": False},
                         {"stalled": False, "events": [{"type": "text", "sessionID": "s1", "part": {"text": "hi"}}],
                          "stderr": "ok\n", "returncode": 0, "timed_out": False, "lingered": False}])
        driver._run_streaming = lambda *a, **k: next(attempts)
        result = driver.turn("do it", "/w", 600)
        self.assertEqual((result["answer"], result["restarts"]), ("hi", 1))
        self.assertEqual([(t["attempt"], t["stalled"], t["stderr"]) for t in driver.raw_turns],
                         [(1, True, "stuck\n"), (2, False, "ok\n")])
        self.assertTrue(all(t["said"] == "do it" for t in driver.raw_turns))

    def test_a_pi_turn_keeps_its_events_for_the_capture(self):
        """Review 2.7: pi stored the event count, so capture.turns raised and no pi turn was ever saved."""
        from unittest.mock import patch
        from llm_eval.workload import pi_driver
        from llm_eval.workload.capture import RawCapture
        driver = pi_driver.PiDriver.__new__(pi_driver.PiDriver)
        driver.events, driver.raw_turns, driver.session, driver.progress, driver.stall_seconds = [], [], None, None, 0
        driver.box = None
        events = [{"type": "session", "id": "s1"}]
        streamed = {"events": events, "stderr": "log\n", "returncode": 0, "timed_out": False, "lingered": False,
                    "stalled": False, "hung": False, "disk_cap": False, "home_mib": 0.1, "max_silence_seconds": 0.0}
        with patch.object(pi_driver.PiDriver, "command", lambda self, workdir: ["pi"]), \
                patch.object(pi_driver, "run_streaming", lambda *a, **k: streamed):
            driver.turn("hi", "/w", 60)
        self.assertEqual(driver.raw_turns[0]["events"], events)
        streamed["events"] = [{"type": "tool_execution_start", "toolName": "bash"}]
        with patch.object(pi_driver.PiDriver, "command", lambda self, workdir: ["pi"]), \
                patch.object(pi_driver, "run_streaming", lambda *a, **k: streamed):
            self.assertTrue(driver.turn("go", "/w", 60)["tool_open"])
        with tempfile.TemporaryDirectory() as tmp:
            raw = RawCapture(Path(tmp))
            raw.turns(driver.raw_turns)
            self.assertEqual(raw.errors, [])
            self.assertTrue((Path(tmp) / "turn-01.events.jsonl").exists())


class TelemetryExtrasTests(unittest.TestCase):
    def test_every_sampler_is_tagged_and_the_probe_runs_at_both_ends(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        journal = Journal(Path(tmp.name) / "run")
        journal.sampler = GpuSampler(lambda: {"a": 1}, Path(tmp.name) / "gpu.jsonl", interval=None)
        journal.extra_samplers = [GpuSampler(lambda: {"b": 2}, Path(tmp.name) / "host.jsonl", interval=None)]
        calls = []
        journal.server_probe = lambda tag, phase: calls.append((tag, phase))
        with sampling(journal, "trial-x"):
            self.assertEqual(journal.extra_samplers[0].tag, "trial-x")
        rows = [json.loads(l) for l in (Path(tmp.name) / "host.jsonl").read_text().splitlines()]
        self.assertEqual([r["tag"] for r in rows], ["trial-x", "trial-x"])
        self.assertEqual(calls, [("trial-x", "start"), ("trial-x", "end")])

    def test_a_failing_probe_never_fails_the_work(self):
        journal = Journal(Path(tempfile.mkdtemp()) / "run")
        journal.server_probe = lambda tag, phase: 1 / 0
        with sampling(journal, "t"):
            pass


class LiveExtrasTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = plan_for(self.root)

    def adapter(self, real=True):
        from llm_eval.live import LiveAdapter

        class Real(ScriptedHost):
            def host_sample(self):
                return {"load1": 0.5, "mem_available_mib": 1000.0}

            def ensure_dir(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)

            def run(self, args, timeout=30):
                if args[:3] == ["systemctl", "--user", "show"] and len(args) == 4:  # a full property dump
                    return subprocess.CompletedProcess(args, 0, "ActiveState=active\nMemoryPeak=1\n", "")
                return super().run(args, timeout)

            def http(self, method, port, path, body=None, timeout=30):
                if path in ("/props", "/slots", "/metrics") and self.server:
                    return (404, "not enabled") if path == "/metrics" else (200, {"path": path})
                return super().http(method, port, path, body, timeout)
        self.host = (Real if real else ScriptedHost)()
        return LiveAdapter(self.plan, self.root / "logs", self.host, run_id="t")

    def loaded(self):
        adapter = self.adapter()
        adapter.snapshot()
        adapter.quiesce()
        adapter.load(self.plan["models"][0], 65536)
        return adapter

    def test_scripted_hosts_have_nothing_to_sample_or_snapshot(self):
        adapter = self.adapter(real=False)
        self.assertIsNone(adapter.sample_host())
        self.assertIsNone(adapter.server_snapshot())

    def test_server_snapshot_keeps_each_endpoint_even_when_one_is_missing(self):
        adapter = self.loaded()
        report = adapter.server_snapshot()
        self.assertEqual(report["props"], {"status": 200, "body": {"path": "/props"}})
        self.assertEqual(report["metrics"]["status"], 404)
        self.assertEqual(report["health"]["status"], 200)
        self.assertGreater(report["log_bytes"], 0)
        self.assertEqual(adapter.sample_host()["load1"], 0.5)

    def test_the_scratch_unit_report_is_saved_beside_the_server_log(self):
        adapter = self.loaded()
        adapter.stop()
        names = sorted(p.name for p in (self.root / "logs").glob("llm-eval-scratch-t-*"))
        self.assertTrue(any(n.endswith(".show.txt") for n in names), names)


if __name__ == "__main__":
    unittest.main()
