"""Task and snapshot fixes from 2026-10-30, each with the run or check that found it."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time
import unittest

from llm_eval.workload import graders, pi_config, replicas
from llm_eval.workload import prepare as workload

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "workload"
TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_test_plan.py"


def _task(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


class HaystackPacketTests(unittest.TestCase):
    """impl-llm-eval-haystack failed all six runs. Three hit the 16,384 output cap before writing a file; the three
    that finished failed hidden tests on section names and fields the packet never stated (each named its rung
    sections its own way, e.g. ``trial-recall-1__102400``)."""

    def test_the_packet_states_every_section_name_and_field_the_hidden_tests_read(self):
        prompt = _task("impl-llm-eval-haystack")["prompt"]
        for contract in ("`<model>__trial-<fixture id>-at-<context>`", "fake-muse__trial-recall-1-at-131072",
                         "`prompt_tokens`", "`structured_tools`", "`context`", "`verified_context`", "`descent`",
                         "`context_class`"):
            self.assertIn(contract, prompt)


class ToolCallAtDepthTests(unittest.TestCase):
    """The one tool call the depth suite never asked for: a command and an edit at about 200K on Strata's 262K."""

    def setUp(self):
        self.task = next(t for t in workload.load_tasks() if t["id"] == "depth-tool-call-late")

    def test_it_is_a_depth_task_filled_to_about_200k_at_262k(self):
        self.assertTrue(self.task["depth"])
        point = 262144 - pi_config.production_compaction()[0]  # the owner's reserve
        self.assertGreater(self.task["fill"]["fraction"] * point, 195000)
        self.assertIn(self.task["id"], [t["id"] for t in workload.select("depth")[1]])

    def test_its_prompts_name_files_in_the_pinned_commit_and_the_hash_is_that_file_s(self):
        from llm_eval.workload.snapshot import build, materialize
        with tempfile.TemporaryDirectory() as tmp:
            build(self.task, Path(tmp) / "snapshot")
            repo = materialize(Path(tmp) / "snapshot", Path(tmp) / "repo")
            for prompt in self.task["fill"]["prompts"]:
                for name in re.findall(r"[\w./-]+\.(?:py|md)", prompt):
                    self.assertTrue((repo / name).is_file(), name)
            blob = subprocess.run(["git", "hash-object", "llm_eval/ladder.py"], cwd=repo, capture_output=True,
                                  text=True).stdout.strip()
            self.assertIn(blob, self.task["checks"][0]["all"])
            # the reference edit passes the file check, and the untouched file does not
            files = next(c for c in self.task["checks"] if c["type"] == "files")
            outcome = {"workdir": repo, "home": Path(tmp)}
            self.assertFalse(graders.check_files(files, outcome, None)["pass"])
            policy = repo / "llm_eval/policy.py"
            policy.write_text(policy.read_text().replace("RESERVE = 2048\n", "RESERVE = 2048\nRECALL_RESERVE_TOKENS = 2048\n"))
            self.assertTrue(graders.check_files(files, outcome, None)["pass"])

    def test_the_solution_answer_and_the_command_pass(self):
        answer = next(c for c in self.task["checks"] if c["type"] == "answer")
        self.assertTrue(graders.check_answer(answer, {"answer": self.task["solution"]["answer"]}, None)["pass"])
        commands = next(c for c in self.task["checks"] if c["type"] == "commands")
        self.assertTrue(graders.check_commands(commands, {"commands": ["git hash-object llm_eval/ladder.py"]}, None)["pass"])
        self.assertFalse(graders.check_commands(commands, {"commands": ["cat llm_eval/ladder.py"]}, None)["pass"])


class ReplicaIndexTests(unittest.TestCase):
    """A model comparison was refused: surgery-split-commit's identity differed between arms prepared a minute
    apart, because the replica's git index (inode, ctime) and a clone's reflog (a temporary path) went into home.tar."""

    def build_twice(self, name):
        digests = []
        with tempfile.TemporaryDirectory() as tmp:
            for n in (1, 2):
                time.sleep(1.1)
                replicas.build(_task(name), Path(tmp) / str(n))
                digests.append(hashlib.sha256((Path(tmp) / str(n) / "home.tar").read_bytes()).hexdigest())
            home = Path(tmp) / "home"
            replicas.unpack_home(Path(tmp) / "2", home)
            dirty = {str(g.parent): subprocess.run(["git", "status", "--porcelain"], cwd=g.parent, capture_output=True,
                                                   text=True).stdout for g in home.rglob(".git") if g.is_dir()
                     and (g / "HEAD").exists() and (g.parent / ".git") == g}
        return digests, dirty

    def test_git_replicas_are_the_same_bytes_and_unpack_clean(self):
        for name in ("surgery-split-commit", "surgery-reconcile-diverged"):
            digests, dirty = self.build_twice(name)
            self.assertEqual(digests[0], digests[1], name)
            self.assertTrue(dirty, name)
            self.assertFalse(any(dirty.values()), (name, dirty))


class RunnerTargetTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("run_test_plan", TOOL)
        self.plan = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.plan)

    def test_a_llama_profile_runs_at_the_context_the_owner_s_pi_gives_it(self):
        rungs = [65536, 98304, 131072]
        self.assertEqual(self.plan.production_context("muse-glimmer-30b-q4", {"ctx-size": "131072"}, rungs, 98304), 131072)
        # pi's window above what the preset loads (gemma-4-31b-it: pi 131,072, preset 65,536): the preset wins
        self.assertEqual(self.plan.production_context("gemma-4-31b-it", {"ctx-size": "65536"}, rungs, 65536), 65536)
        self.assertEqual(self.plan.production_context("unknown", {"ctx-size": "131072"}, rungs, 98304), 98304)
        # pi's window above the ladder top, and exactly what the preset loads (occamy-1.0 at 196,608 since the
        # 2026-11-05 adoption): it runs there; a preset loading more than pi's window, off the ladder, falls back
        self.assertEqual(self.plan.production_context("occamy-1.0", {"ctx-size": "196608"}, rungs, 98304), 196608)
        self.assertEqual(self.plan.production_context("occamy-1.0", {"ctx-size": "262144"}, rungs, 98304), 98304)

    def test_llama_knobs_apply_only_where_the_preset_has_the_setting(self):
        overrides, applies = self.plan.LLAMA_KNOBS["no-spec"]
        self.assertTrue(applies({"spec-type": "draft-mtp"}))
        self.assertFalse(applies({"cache-type-k": "f16"}))
        self.assertIsNone(overrides["spec-type"])
        self.assertFalse(self.plan.LLAMA_KNOBS["kv-q8"][1]({"cache-type-k": "f16", "cache-type-v": "f16"}))

    def test_preset_overrides_become_set_and_unset_flags(self):
        from llm_eval.acceptance import preset_overrides
        self.assertEqual(preset_overrides(["cache-type-k=q8_0"], ["spec-type"]),
                         {"cache-type-k": "q8_0", "spec-type": None})
        with self.assertRaises(ValueError):
            preset_overrides(["cache-type-k"])


if __name__ == "__main__":
    unittest.main()


class OutputCapTests(unittest.TestCase):
    """qwen3.8-27b-q4-agent's smoke on 2026-10-30 spent its whole 12,288 tokens in one response and wrote nothing;
    on llama.cpp that scored as an ordinary wrong answer, because only Strata runs marked output_cap."""

    def test_a_llama_cpp_trial_cut_at_the_output_limit_is_an_output_cap(self):
        import test_workload_runner as fixtures
        from unittest import mock
        from llm_eval.adapters import FakeWorkloadDriver
        from llm_eval.runner import execute
        helper = fixtures.WorkloadCampaignTests()
        helper.setUp()
        self.addCleanup(helper.tearDown)
        plan = helper.plan({"workload": {"impl-1": {"turns": [[{"src/app.py": fixtures.FIX}, "Fixed add()."]]}}})
        self.assertNotEqual(plan.get("runtime"), "strata")
        original = FakeWorkloadDriver.summarize
        with mock.patch.object(FakeWorkloadDriver, "summarize",
                               lambda self, export=None: {**original(self, export), "length_finishes": 1}):
            state = execute(plan, helper.root / "runs", "cap")
        trial = state["sections"]["fake-muse__trial-wl-impl-1-r1"]
        self.assertEqual((trial["status"], trial["stop"]), ("quality_fail", "output_cap"))
