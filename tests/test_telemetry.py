import json
from pathlib import Path
import tempfile
import time
import unittest

from llm_eval import acceptance
from llm_eval.live import RealHost
from llm_eval.telemetry import GpuSampler, parse_nvidia_smi, sampling


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "telemetry/gpu.jsonl"

    def test_snap_writes_a_tagged_row(self):
        sampler = GpuSampler(lambda: {"mem_free_mib": 5.0}, self.path, interval=None, clock=lambda: 12.5)
        sampler.tag = "trial-1"
        sampler.snap()
        self.assertEqual(rows(self.path), [{"mem_free_mib": 5.0, "t": 12.5, "tag": "trial-1"}])

    def test_a_failing_sample_is_an_error_row_not_an_exception(self):
        def broken():
            raise RuntimeError("nvidia-smi failed")
        sampler = GpuSampler(broken, self.path, interval=None)
        sampler.snap()
        self.assertEqual(sampler.errors, 1)
        self.assertIn("nvidia-smi failed", rows(self.path)[0]["error"])

    def test_a_host_that_cannot_sample_writes_nothing(self):
        GpuSampler(lambda: None, self.path, interval=None).snap()
        self.assertFalse(self.path.exists())

    def test_sampling_tags_the_window_and_samples_at_both_ends(self):
        class Journal:
            pass
        journal = Journal()
        journal.sampler = GpuSampler(lambda: {"mem_free_mib": 1.0}, self.path, interval=None)
        with sampling(journal, "trial-x"):
            self.assertEqual(journal.sampler.tag, "trial-x")
        self.assertIsNone(journal.sampler.tag)
        self.assertEqual([r["tag"] for r in rows(self.path)], ["trial-x", "trial-x"])
        with sampling(Journal(), "no sampler"):  # a journal without a sampler is a no-op
            pass

    def test_the_thread_samples_until_stopped(self):
        sampler = GpuSampler(lambda: {"mem_free_mib": 1.0}, self.path, interval=0.01).start()
        time.sleep(0.15)
        sampler.stop()
        count = len(rows(self.path))
        self.assertGreaterEqual(count, 3)
        time.sleep(0.05)
        self.assertEqual(len(rows(self.path)), count)

    def test_nvidia_smi_row_parses_and_na_is_none(self):
        row = parse_nvidia_smi("20480, 3500, 97, 41, 66, 301.25, 2520, [N/A]")
        self.assertEqual(row["mem_free_mib"], 3500.0)
        self.assertEqual(row["power_w"], 301.25)
        self.assertIsNone(row["clock_mem_mhz"])
        with self.assertRaises(ValueError):
            parse_nvidia_smi("1, 2")

    def test_real_host_asks_nvidia_smi_for_one_row(self):
        host = RealHost()
        calls = []
        class Out:
            returncode, stdout, stderr = 0, "1, 2, 3, 4, 5, 6.5, 7, 8\n", ""
        host.run = lambda args, timeout=30: calls.append(args) or Out()
        self.assertEqual(host.gpu_sample()["power_w"], 6.5)
        self.assertIn("--query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,"
                      "power.draw,clocks.sm,clocks.mem", calls[0])


if __name__ == "__main__":
    unittest.main()
