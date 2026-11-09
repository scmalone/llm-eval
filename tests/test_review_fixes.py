"""One regression per finding of the 2026-10-03 blind review (WLR-001 to WLR-009)."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import judge
from llm_eval.adapters import FakeBox
from llm_eval.workload import replicas
from llm_eval.workload.graders import grade
from llm_eval.workload.snapshot import build, git, materialize, reveal_hidden
from tests.test_workload import make_repo, repo_task


class GradingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        repo, base, answer = make_repo(self.root, with_data=False)
        self.manifest = build(repo_task(repo, base, answer), self.root / "snap")
        self.home = self.root / "home"
        self.run = materialize(self.root / "snap", self.home / "Workspace/repo")
        self.outside = self.root / "outside.txt"  # stands in for a real file outside the run's home
        self.outside.write_text("real content\n")

    def tearDown(self):
        self.tmp.cleanup()

    def outcome(self, **extra):
        return {"answer": "", "workdir": self.run, "home": self.home, "base_files": self.manifest["files"],
                "snapshot": self.root / "snap", "commands": [], "git": FakeBox().git, **extra}

    def test_wlr001_repo_config_cannot_run_a_program_during_grading(self):
        sentinel = self.root / "fsmonitor-ran"
        git(["config", "core.fsmonitor", f"touch {sentinel}; false"], self.run, isolated=True)
        git(["config", "core.hooksPath", str(self.root / "hooks")], self.run, isolated=True)
        grade({"checks": [{"type": "git", "clean": True}]}, self.outcome())
        self.assertFalse(sentinel.exists())

    def test_wlr002_revealing_tests_never_writes_through_a_planted_symlink(self):
        (self.run / "tests/test_add.py").symlink_to(self.outside)
        reveal_hidden(self.root / "snap", self.run)
        self.assertEqual(self.outside.read_text(), "real content\n")
        self.assertFalse((self.run / "tests/test_add.py").is_symlink())
        os.symlink(self.root, self.run / "linked")  # a symlinked directory refuses the reveal
        (self.root / "snap/hidden/linked").mkdir()
        (self.root / "snap/hidden/linked/x.py").write_text("# x\n")
        result = grade({"checks": [{"type": "tests", "command": ["true"]}]}, self.outcome(run=lambda *a: (0, "OK")))
        self.assertFalse(result["pass"])
        self.assertFalse((self.root / "x.py").exists())

    def test_wlr002_file_checks_do_not_follow_symlinks_out_of_the_home(self):
        (self.run / "leak.txt").symlink_to(self.outside)
        result = grade({"checks": [{"type": "files", "exists": ["leak.txt"], "contains": {"leak.txt": "real"}}]}, self.outcome())
        self.assertFalse(result["pass"])

    def test_wlr005_forged_machine_state_does_not_pass_unit_ready(self):
        task = {"id": "ops", "source": {"replica": "renamed-flag", "seed": 5}}
        spec = replicas.REPLICAS["renamed-flag"](5)
        machine = json.loads(spec["home_files"][replicas.MACHINE])
        unit = machine["units"]["llama-router.service"]
        flag = replicas.facts(task)["flag"]
        preset = self.home / ".config/llama-cpp/models-preset.ini"
        preset.parent.mkdir(parents=True)
        preset.write_text(f"[m]\nmodel = x\n{flag} = 0.1\n")
        state = self.home / replicas.MACHINE
        state.parent.mkdir(parents=True)
        unit["active"] = True  # the agent edits the state file instead of fixing the preset
        state.write_text(json.dumps(machine))
        checks = [{"type": "state", "file": "~/" + replicas.MACHINE, "equals": {"units/llama-router.service/active": True}},
                  {"type": "unit_ready", "unit": "llama-router.service", "requires": unit["requires"], "replica_home": machine["home"]}]
        self.assertFalse(grade({"checks": checks}, self.outcome())["pass"])
        preset.write_text("[m]\nmodel = x\n")
        self.assertTrue(grade({"checks": checks}, self.outcome())["pass"])
        # The box binds the home at the real home's path, so an absolute path an agent wrote is read in the run's home.
        (self.home / "models").mkdir()
        (self.home / "models/a.gguf").write_text("x")
        ready = {"type": "unit_ready", "unit": "u", "replica_home": "~", "requires": [
            {"ini_paths_exist": ["~/.config/llama-cpp/models-preset.ini", "model"]}]}
        preset.write_text(f"[m]\nmodel = {Path.home()}/models/a.gguf\n")
        self.assertTrue(grade({"checks": [ready]}, self.outcome())["pass"])
        preset.write_text(f"[m]\nmodel = {Path.home()}/models/missing.gguf\n")
        self.assertFalse(grade({"checks": [ready]}, self.outcome())["pass"])

    def test_wlr006_a_denial_does_not_count_as_a_finding(self):
        check = [{"type": "findings", "defects": [{"id": "project-bound",
                  "match": [r"PACKAGE_ROOT|whole project", r"task|fixture|answer|test|grader|expected"]}]}]
        self.assertFalse(grade({"checks": check}, self.outcome(answer="PACKAGE_ROOT is safe. Task answers are never exposed."))["pass"])
        self.assertFalse(grade({"checks": check}, self.outcome(answer="PACKAGE_ROOT is fine.\n\nUnrelated: tests are slow."))["pass"])
        self.assertTrue(grade({"checks": check}, self.outcome(
            answer="sandbox.py binds PACKAGE_ROOT read-only, so the agent can read task fixtures and graders."))["pass"])

    def test_wlr008_a_shallow_session_fails_the_depth_floor(self):
        check = [{"type": "depth", "min_peak_tokens": 40000}]
        self.assertFalse(grade({"checks": check}, self.outcome(metrics={"peak_context_tokens": 100}))["pass"])
        self.assertTrue(grade({"checks": check}, self.outcome(metrics={"peak_context_tokens": 47000}))["pass"])


class JudgeCoverageTests(unittest.TestCase):
    def test_wlr009_answer_tasks_are_judged_against_their_solution_and_gaps_are_reported(self):
        tasks = {"lookup-1": {"id": "lookup-1", "category": "lookup", "prompt": "Where?", "solution": {"answer": "docs/x.md"}},
                 "odd-1": {"id": "odd-1", "category": "lookup", "prompt": "?"}}
        seen = []
        def fake_score(endpoint, task, run, reference):
            seen.append((task["id"], reference))
            return {"task": task["id"], "scores": {}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(judge, "score", fake_score), \
                patch.object(judge, "evidence", lambda d, box_factory=None: {"answer": "", "diff": ""}):
            results, skipped = judge.judge_runs({}, tasks, {"lookup-1": [Path(tmp)], "odd-1": [Path(tmp)]}, {}, Path(tmp))
        self.assertEqual(seen, [("lookup-1", "docs/x.md")])
        self.assertEqual(skipped, [{"task": "odd-1", "reason": "no reference and no second profile"}])


if __name__ == "__main__":
    unittest.main()
