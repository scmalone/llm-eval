import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("flip_rate", Path(__file__).resolve().parents[1] / "tools" / "flip_rate.py")
flip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flip)


def sec(task, n, status, **extra):
    return f"m__trial-wl-{task}-r{n}", {"status": status, **extra}


class FlipRateTests(unittest.TestCase):
    def test_flips_rate_and_spread(self):
        sections = dict([sec("a", 1, "pass"), sec("a", 2, "pass"), sec("a", 3, "pass"),
                         sec("b", 1, "pass"), sec("b", 2, "fail"), sec("b", 3, "pass"),
                         sec("c", 1, "fail"), sec("c", 2, "fail"), sec("c", 3, "fail")])
        result = flip.analyse(sections)
        self.assertEqual((result["tasks_repeated"], result["flipped"], result["flip_rate"]), (3, ["b"], 0.333))
        self.assertEqual((result["passes_per_repeat"], result["spread"]), ([2, 1, 2], 1))

    def test_incomplete_and_single_runs_are_left_out(self):
        sections = dict([sec("a", 1, "pass"), sec("a", 2, "fail", incomplete=True), sec("b", 1, "pass"), sec("b", 2, "pass")])
        result = flip.analyse(sections)
        self.assertEqual((result["tasks_repeated"], result["flipped"]), (1, []))


if __name__ == "__main__":
    unittest.main()
