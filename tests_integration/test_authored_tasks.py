"""Every authored workload task can fail and can pass: a no-op agent fails it, and an agent
that replays the task's solution (an embedded reference diff, or competent shell steps)
passes it. All source states are bundled or generated, so it runs from a standalone checkout:
    python3 -m unittest discover -s tests_integration -p test_authored_tasks.py -v
"""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from llm_eval.adapters import FakeWorkloadDriver
from llm_eval.workload import prepare as workload, replicas, snapshot
from llm_eval.workload.sandbox import Box
from llm_eval.workload.session import run_task

TASKS = sorted(workload.TASK_DIR.glob("*.json"))


class SolutionDriver(FakeWorkloadDriver):
    """Plays the task's solution once: apply the reference diff and/or run shell steps in the box."""
    def __init__(self, home, manifest, snapshot_dir, task, box):
        super().__init__([])
        place = manifest.get("place") or f"Workspace/{Path(manifest['repo']).name}"
        self.repo, self.snapshot_dir, self.solution, self.box = Path(home) / place, Path(snapshot_dir), task["solution"], box
        self.commands, self.prelude, self.turn_number = [], len(task.get("prelude", [])), 0
        # A solved depth task reached its depth; the no-op driver reports a shallow session.
        self.peak = 60000 if task.get("depth") or task.get("diagnostic") else 100

    def turn(self, text, workdir, timeout):
        self.turn_number += 1
        if self.turn_number <= self.prelude:  # prelude turns: read and summarize, nothing to solve
            return {"answer": "Summary.", "timed_out": False, "returncode": 0, "events": 1}
        scripted = self.solution.get("turn_answers")
        task_turn = self.turn_number - self.prelude
        if scripted and task_turn > 1:
            self.last = scripted[task_turn - 1]
            return {"answer": self.last, "timed_out": False, "returncode": 0, "events": 1}
        if self.solution.get("apply_reference"):
            out = subprocess.run(["git", "apply", "--whitespace=nowarn", str(self.snapshot_dir / "reference.diff")],
                                 cwd=self.repo, capture_output=True, text=True)
            if out.returncode:
                raise AssertionError(f"reference diff did not apply: {out.stderr[-500:]}")
        for step in self.solution.get("steps", []):
            code, out, err, _ = self.box.run(["bash", "-c", step], self.repo, 120)
            if code:
                raise AssertionError(f"solution step failed ({code}): {step}\n{out[-800:]}{err[-800:]}")
            self.commands.append(step)
        self.last = scripted[0] if scripted else self.solution.get("answer", "")
        return {"answer": self.last, "timed_out": False, "returncode": 0, "events": 1}

    def export(self):
        export = super().export()
        export["messages"][1]["parts"][0]["tokens"]["input"] = self.peak
        export["messages"][1]["parts"] += [{"type": "tool", "tool": "bash", "state": {"status": "completed",
                                            "input": {"command": c}}} for c in self.commands]
        return export


def home_files(tasks, root):
    return workload.bundled_home_files() | {".gitconfig": workload.GITCONFIG}


def box_for(home, manifest):
    path = [str(Path.home() / p) for p in manifest.get("path", [])]
    node = workload.NODE_HOME
    return Box(home, read_only=[(node, str(node))] if node.exists() else [],
               path=path + ([str(node / "aliases/default/bin")] if node.exists() else []))


class AuthoredTaskTests(unittest.TestCase):
    def test_there_are_five_tasks_per_category(self):
        tasks = workload.load_tasks()
        counts = {c: sum(t["category"] == c and not t.get("depth") and not t.get("diagnostic") for t in tasks)
                  for c in workload.CATEGORIES}
        self.assertEqual(counts, {c: 5 for c in workload.CATEGORIES})
        self.assertEqual(sum(bool(t.get("depth")) for t in tasks), 5)  # three authored, the scaled one, a tool call late
        self.assertEqual(sum(bool(t.get("diagnostic")) for t in tasks), 5)

    def test_every_task_fails_on_a_noop_and_passes_when_solved(self):
        for path in TASKS:
            task = json.loads(path.read_text())
            with self.subTest(task=task["id"]), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                snap = root / "snap"
                (replicas.build if "replica" in task["source"] else snapshot.build)(task, snap)
                manifest = json.loads((snap / "manifest.json").read_text())
                files = home_files([task], root)
                noop = run_task(task, root / "noop", snap, lambda home, m: FakeWorkloadDriver([[{}, ""]]),
                                lambda home: box_for(home, manifest), files)
                self.assertFalse(noop["pass"], f"{task['id']}: a no-op run passed; its checks cannot fail")
                solved = run_task(task, root / "solved", snap,
                                  lambda home, m: SolutionDriver(home, m, snap, task, box_for(home, m)),
                                  lambda home: box_for(home, manifest), files)
                failing = [c for c in solved["grading"]["checks"] if not c["pass"]]
                self.assertTrue(solved["pass"], f"{task['id']}: the solution failed {json.dumps(failing)[:1500]}")


if __name__ == "__main__":
    unittest.main()
