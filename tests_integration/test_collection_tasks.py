"""Every collection task (fixtures/workload/{mined,game,vision}) can fail and can pass, inside the real sandbox
with its prebuilt environment mounted: a no-op agent fails it, the reference fix passes it (E4's first two proofs).
    python3 -m unittest discover -s tests_integration -p test_collection_tasks.py -v
Needs the environments under <data>/envs and the source repos the tasks name.
"""
import json
import os
from pathlib import Path
import tempfile
import unittest

from llm_eval.adapters import FakeWorkloadDriver
from llm_eval.workload import prepare as workload, snapshot
from llm_eval.workload.session import run_task
from llm_eval.workload.tasks import COLLECTIONS
from test_authored_tasks import SolutionDriver, box_for, home_files

ONLY = set(filter(None, os.environ.get("LLM_EVAL_ONLY", "").split(",")))  # comma-separated task ids, for a rerun
TASKS = sorted(p for c in COLLECTIONS for p in (workload.TASK_DIR / c).glob("*.json") if not ONLY or p.stem in ONLY)


class CollectionTaskTests(unittest.TestCase):
    def test_every_collection_task_fails_on_a_noop_and_passes_when_solved(self):
        for path in TASKS:
            task = json.loads(path.read_text())
            with self.subTest(task=task["id"]), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                snap = root / "snap"
                snapshot.build(task, snap)
                manifest = json.loads((snap / "manifest.json").read_text())
                files = home_files([task], root)
                noop = run_task(task, root / "noop", snap, lambda home, m: FakeWorkloadDriver([[{}, ""]]),
                                lambda home: box_for(home, manifest), files)
                self.assertFalse(noop["pass"], f"{task['id']}: a no-op run passed; its checks cannot fail")
                solved = run_task(task, root / "solved", snap,
                                  lambda home, m: SolutionDriver(home, m, snap, task, box_for(home, m)),
                                  lambda home: box_for(home, manifest), files)
                failing = [c for c in solved["grading"]["checks"] if not c["pass"]]
                self.assertTrue(solved["pass"], f"{task['id']}: the solution failed {json.dumps(failing)[:2500]}")

    def test_marking_every_test_skipped_from_editable_source_fails(self):
        """Red-team RT-001's reproduction, replayed through run_task in the real sandbox: the bug left intact and
        code the agent may edit marking every test class skipped. unittest says OK (skipped=N) and exits 0; the
        task must still fail."""
        path = workload.TASK_DIR / "mined" / "mined-strata-live-rate.json"
        if ONLY and path.stem not in ONLY:
            self.skipTest("not selected")
        task = json.loads(path.read_text())
        patch = ("cat >> serve/server.py <<'PY'\nimport unittest as _u\n"
                 "_u.TestCase.__init_subclass__ = classmethod(lambda cls, **kw: setattr(cls, '__unittest_skip__', True))\nPY")
        task["solution"] = {"steps": [patch], "answer": "Implemented; the tests pass."}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = root / "snap"
            snapshot.build(task, snap)
            manifest = json.loads((snap / "manifest.json").read_text())
            result = run_task(task, root / "skip", snap, lambda home, m: SolutionDriver(home, m, snap, task, box_for(home, m)),
                              lambda home: box_for(home, manifest), home_files([task], root))
            tests = next(c for c in result["grading"]["checks"] if c["type"] == "tests")
            self.assertRegex(tests["output_tail"], r"OK \(skipped=\d+\)")  # the attack took effect
            self.assertEqual(tests["returncode"], 0)
            self.assertFalse(result["pass"])
            self.assertIn("the test runner reported skipped tests", tests["problems"])


if __name__ == "__main__":
    unittest.main()
