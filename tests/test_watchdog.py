import itertools
import unittest
import subprocess
from unittest.mock import patch

from llm_eval.workload import driver as driver_module
from llm_eval.workload.driver import OpencodeDriver
from llm_eval.workload.prepare import TASK_OVERHEAD_SECONDS as O, limits


class SleepBox:
    """A box whose 'opencode' is a silent process: it never writes an event or a log line."""
    def argv(self, command, cwd):
        return ["sleep", "30"]

    def inside(self, path):
        return str(path)


def driver(progress, stall_seconds=1):
    d = OpencodeDriver.__new__(OpencodeDriver)
    d.box, d.progress, d.stall_seconds = SleepBox(), progress, stall_seconds
    return d


class LimitsTests(unittest.TestCase):
    def task(self, wall, runs=1):
        return {"limits": {"turns": 2, "wall_seconds": wall}, "runs": runs}

    def test_the_model_cap_is_the_sum_of_task_caps_plus_overhead_so_it_never_cuts_a_task_inside_its_cap(self):
        caps = limits(5, [self.task(900), self.task(1500)])
        self.assertEqual(caps["model_seconds"], (900 + O) + (1500 + O) + 300)
        self.assertEqual(caps["campaign_seconds"], caps["model_seconds"] + 900)

    def test_a_short_task_gets_a_short_cap_with_no_fixed_floor(self):
        self.assertEqual(limits(10, [self.task(300)])["model_seconds"], 300 + O + 300)

    def test_runs_scale_the_cap(self):
        self.assertEqual(limits(600, [self.task(900, runs=2)])["model_seconds"], 2 * (900 + O) + 300)

    def test_the_request_timeout_covers_a_cold_prefill_of_the_whole_context(self):
        # a streamed request is silent through its prefill: 262,144 tokens at the 250 t/s floor is ~1,049 s
        self.assertEqual(limits(5, [self.task(900)], 262144)["request_seconds"], 262144 // 250 + 120)
        self.assertEqual(limits(5, [self.task(900)], 65536)["request_seconds"], 600)
        self.assertEqual(limits(5, [self.task(900)])["request_seconds"], 600)

    def test_an_empty_suite_falls_back_to_the_estimate(self):
        self.assertEqual(limits(10)["model_seconds"], 900)


class WriterBox:
    """A box whose harness writes ``mib`` MiB into its home, then stays silent."""
    def __init__(self, home, mib):
        self.home, self.mib = home, mib

    def argv(self, command, cwd):
        return ["sh", "-c", f"head -c {self.mib}M /dev/zero > '{self.home}/runaway'; sleep 30"]

    def inside(self, path):
        return str(path)


class DiskCapTests(unittest.TestCase):
    def test_a_home_past_its_cap_kills_the_turn_and_says_so(self):
        import tempfile, time
        from llm_eval.workload import session
        with tempfile.TemporaryDirectory() as home:
            d = driver(None)
            d.box, d.home_cap_mib = WriterBox(home, 8), 4
            started = time.monotonic()
            result = d._run_streaming(["x"], home, 20)
            self.assertTrue(result["disk_cap"])
            self.assertGreater(result["home_mib"], 4)
            self.assertFalse(result["timed_out"])
            self.assertLess(time.monotonic() - started, 10)
            self.assertEqual(session._stop_for(result, "wall_cap"), "disk_cap")
            self.assertIn("disk_cap", session.UNGRADED)

    def test_a_home_under_its_cap_runs_to_its_timeout(self):
        import tempfile
        with tempfile.TemporaryDirectory() as home:
            d = driver(None)
            d.box = WriterBox(home, 1)
            result = d._run_streaming(["x"], home, 3)
            self.assertFalse(result["disk_cap"])
            self.assertTrue(result["timed_out"])

    def test_a_harness_that_closes_stdout_before_it_exits_has_not_timed_out(self):
        """Review 2.3: EOF arrived before the exit was reaped, and two finished turns were scored wall_cap."""
        class ClosingBox(SleepBox):
            def argv(self, command, cwd):
                return ["sh", "-c", "echo '{}'; exec 1>&-; sleep 1"]
        d = driver(None)
        d.box = ClosingBox()
        result = d._run_streaming(["x"], "/tmp", 20)
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["returncode"], 0)

    def test_the_box_caps_its_tmpfs(self):
        from llm_eval.workload.sandbox import Box, TMP_CAP_BYTES
        argv = Box("/tmp/h").argv(["true"], "/tmp")
        at = argv.index("/tmp")
        self.assertEqual(argv[at - 3:at], ["--size", str(TMP_CAP_BYTES), "--tmpfs"])


class WatchdogTests(unittest.TestCase):
    def test_cancel_reaps_the_running_harness_process(self):
        processes = []
        popen = subprocess.Popen
        def launch(*args, **kwargs):
            process = popen(*args, **kwargs)
            processes.append(process)
            return process
        def cancel():
            raise KeyboardInterrupt('owner canceled the turn')
        try:
            with patch.object(driver_module.subprocess, 'Popen', side_effect=launch):
                with self.assertRaises(KeyboardInterrupt):
                    driver(cancel)._run_streaming(['x'], '/tmp', 20)
            self.assertIsNotNone(processes[0].poll(), 'canceled harness process was left running')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_a_silent_process_with_a_frozen_fingerprint_is_hung(self):
        result = driver(lambda: ("frozen",))._run_streaming(["x"], "/tmp", 20)
        self.assertTrue(result["hung"])
        self.assertFalse(result["timed_out"])
        self.assertGreaterEqual(result["max_silence_seconds"], 1)

    def test_a_moving_fingerprint_is_progress_so_only_the_turn_timeout_applies(self):
        counter = itertools.count()
        original = driver_module.PROGRESS_POLL_SECONDS
        driver_module.PROGRESS_POLL_SECONDS = 0.1
        self.addCleanup(setattr, driver_module, "PROGRESS_POLL_SECONDS", original)
        result = driver(lambda: next(counter))._run_streaming(["x"], "/tmp", 2.5)
        self.assertFalse(result["hung"])
        self.assertTrue(result["timed_out"])

    def test_no_signal_means_silence_is_measured_but_never_enforced(self):
        result = driver(lambda: None)._run_streaming(["x"], "/tmp", 2.5)
        self.assertFalse(result["hung"])
        self.assertTrue(result["timed_out"])
        self.assertGreaterEqual(result["max_silence_seconds"], 2)

    def test_an_unreachable_server_after_a_good_first_poll_counts_as_no_progress(self):
        calls = itertools.count()
        def progress():
            if next(calls) == 0:
                return ("up",)
            raise ConnectionError("refused")
        original = driver_module.PROGRESS_POLL_SECONDS
        driver_module.PROGRESS_POLL_SECONDS = 0.1
        self.addCleanup(setattr, driver_module, "PROGRESS_POLL_SECONDS", original)
        self.assertTrue(driver(progress)._run_streaming(["x"], "/tmp", 20)["hung"])


if __name__ == "__main__":
    unittest.main()
