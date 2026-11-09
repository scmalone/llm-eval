import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import screen
from llm_eval.metrics.schema import connect

SPECS = screen.screen_specs()
TASKS = [t for t, _, _ in SPECS]
SPEC = {t: s for t, _, s in SPECS}


class Corpus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.conn = connect(self.root / "metrics.sqlite")
        self.clock = 0

    def add(self, profile, task, status, seconds=60.0, spec=None, valid=1, overridden=0, context=131072, output=16384):
        self.clock += 1
        run = f"{profile}-{task}-{self.clock}"
        with self.conn:
            self.conn.execute("INSERT INTO runs (run_id, valid_for_analysis) VALUES (?, ?)", (run, valid))
            self.conn.execute("INSERT INTO run_models (run_id, profile, mode_entry, mode_context_overridden, mode_output_limit) VALUES (?,?,?,?,?)",
                              (run, profile, profile, overridden, output))
            self.conn.execute("INSERT INTO task_runs (run_id, profile, section, task_id, task_spec_sha, status, wall_seconds, context, recorded_at) "
                              "VALUES (?,?,?,?,?,?,?,?,?)", (run, profile, "s", task, spec or SPEC[task], status, seconds, context, self.clock))

    def profile(self, name, outcomes, seconds=60.0):
        """outcomes: status per screen task in order, e.g. 'ppfptpf' (p pass, f fail, t timeout, i infra, c capacity)."""
        status = {"p": "pass", "f": "quality_fail", "t": "timeout", "i": "infra_fail", "c": "capacity_fail"}
        for task, letter in zip(TASKS, outcomes):
            self.add(name, task, status[letter], seconds)

    def cells(self, *profiles):
        return screen.matrix(self.root, list(profiles), SPECS)


class MatrixTests(Corpus):
    def test_the_screen_is_the_seven_tasks_one_per_category_in_order(self):
        self.assertEqual(TASKS[0], "lookup-smoke-test")
        self.assertEqual(len(TASKS), 7)
        self.assertEqual(len({c for _, c, _ in SPECS}), 7)

    def test_only_exact_version_default_mode_valid_runs_count(self):
        self.add("a", TASKS[0], "pass", spec="old-version")
        self.add("a", TASKS[1], "pass", valid=0)
        self.add("a", TASKS[2], "pass", overridden=1)
        self.add("a", TASKS[3], "pass")
        self.add("a", TASKS[4], "pass", output=32768)  # another output limit is another cohort, not the screen
        cells = self.cells("a")["a"]
        self.assertEqual(list(cells), [TASKS[3]])

    def test_the_newest_run_of_a_task_is_the_one_credited(self):
        self.add("a", TASKS[0], "infra_fail")
        self.add("a", TASKS[0], "pass")
        self.assertEqual(self.cells("a")["a"][TASKS[0]]["outcome"], "pass")


class RankTests(Corpus):
    def rank(self, *profiles):
        return screen.rank(self.cells(*profiles), SPECS)

    def test_profiles_rank_by_passes_and_unfinished_ones_are_listed_not_ranked(self):
        self.profile("a", "ppppppp")
        self.profile("b", "pppppfp")
        self.profile("c", "ppppiff")  # an infrastructure failure: owed, not a verdict
        self.profile("d", "ppppppf")
        result = self.rank("a", "b", "c", "d", "never-run")
        self.assertEqual([(e["profile"], e["passes"]) for e in result["ranking"]], [("a", 7), ("b", 6), ("d", 6)])
        self.assertEqual(set(result["owes_results"]), {"c", "never-run"})
        self.assertEqual(result["owes_results"]["c"], {TASKS[4]: "infra"})

    def test_a_timeout_is_a_failure_that_stays_visible(self):
        self.profile("a", "ppppppt")
        entry = self.rank("a")["ranking"][0]
        self.assertEqual((entry["passes"], entry["timeouts"]), (6, [TASKS[6]]))

    def test_a_tie_is_broken_by_time_on_the_tasks_both_passed(self):
        self.profile("slow", "pppppff", seconds=120.0)
        self.profile("fast", "ppppfpf", seconds=30.0)  # same passes and coverage, different tasks passed
        self.profile("also", "pppppff", seconds=60.0)
        result = self.rank("slow", "fast", "also")
        # all three pass five tasks in five categories; on the four they all passed, fast < also < slow
        self.assertEqual([e["profile"] for e in result["ranking"]], ["fast", "also", "slow"])

    def test_tied_profiles_with_no_common_pass_stay_tied(self):
        self.profile("x", "pppffff", seconds=10.0)
        self.profile("y", "ffffppp", seconds=99.0)
        result = self.rank("x", "y")
        self.assertTrue(all(e["tied_with"] for e in result["ranking"]))
        self.assertTrue(all(e["elapsed_common_seconds"] is None for e in result["ranking"]))

    def test_a_contested_third_place_names_the_boundary_and_is_provisional(self):
        for name, outcomes in (("a", "ppppppp"), ("b", "pppppp" "f"), ("c", "ppppp" "ff"), ("d", "pppp" "fff"), ("e", "pp" "fffff")):
            self.profile(name, outcomes)
        result = self.rank("a", "b", "c", "d", "e")
        self.assertEqual(result["top3"], ["a", "b", "c"])
        self.assertEqual(set(result["boundary_candidates"]), {"c", "d"})  # the third place and the profile within one pass below it
        self.assertTrue(result["third_place_provisional"])

    def test_a_clear_third_place_is_not_provisional(self):
        for name, outcomes in (("a", "ppppppp"), ("b", "pppppp" "f"), ("c", "pppppff"), ("d", "ppffff" "f")):
            self.profile(name, outcomes)
        result = self.rank("a", "b", "c", "d")
        self.assertEqual(result["boundary_candidates"], [])
        self.assertFalse(result["third_place_provisional"])


class ReportTests(Corpus):
    def test_the_report_says_what_to_do_next(self):
        presets = {"a": {"model": "a.gguf", "ctx-size": "131072"}, "b": {"model": "b.gguf", "ctx-size": "131072"}}
        with patch("llm_eval.preflight.read_presets", lambda path=None: (Path("x"), presets)):
            first = screen.report(self.root)
            self.assertEqual(first["not_started"], ["a", "b"])
            self.assertIn("no screen results yet", first["next"])
            self.profile("a", "ppppppp")
            self.profile("b", "pppppff")
            done = screen.report(self.root)
        self.assertEqual(done["screen_tasks"], TASKS)
        self.assertEqual(done["top3"], ["a", "b"])
        self.assertEqual(done["matrix"]["a"][TASKS[0]], "pass")
        self.assertIn("audit", done["next"])
        self.assertEqual(set(done["ranking"][0]["audit"]), {"advisory-models-to-keep", "review-llm-eval-live-recovery"})  # passes to read


class CommandLineTests(Corpus):
    def test_the_screen_command_prints_the_matrix_and_exits_ok(self):
        import contextlib, io
        from llm_eval import acceptance
        presets = {"a": {"model": "a.gguf", "ctx-size": "131072"}}
        self.profile("a", "ppppppp")
        with patch.object(acceptance, "RUNS_ROOT", self.root), patch("llm_eval.preflight.read_presets", lambda path=None: (Path("x"), presets)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(acceptance.main(["screen"]), acceptance.OK)
        shown = json.loads(out.getvalue())
        self.assertEqual(shown["top3"], ["a"])
        self.assertEqual(shown["screen_tasks"], TASKS)

    def test_the_prepare_stage_flag_defaults_to_the_screen(self):
        import contextlib, io
        from llm_eval import acceptance
        with patch.object(acceptance, "prepare_next", lambda *a, **k: ({"stage": k["stage"]}, 0)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            acceptance.main(["prepare", "--next"])
            acceptance.main(["prepare", "--next", "--stage", "full"])
        self.assertEqual([json.loads(chunk)["stage"] for chunk in out.getvalue().replace("}\n{", "}\n\n{").split("\n\n")], ["screen", "full"])


if __name__ == "__main__":
    unittest.main()
