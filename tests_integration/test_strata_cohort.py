"""Reference and negative grading for the separate roster through the real bwrap sandbox."""
import json
from pathlib import Path
import tempfile
import unittest

from llm_eval import cohort
from llm_eval.adapters import FakeWorkloadDriver
from llm_eval.workload import replicas, snapshot
from llm_eval.workload.session import run_task
from test_authored_tasks import SolutionDriver, box_for, home_files


class StrataCohortTests(unittest.TestCase):
    def test_all_18_tasks_accept_reference_and_reject_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Test sizing with an explicit surrogate. Installed-tokenizer measurements are
            # separate machine evidence, not fabricated by this portable grading test.
            tasks = cohort.prepare(root/'frozen', lambda text: len(text.split()))
            for task in tasks:
                with self.subTest(task=task['id']):
                    base = root/task['id']
                    snap = base/'snapshot'
                    (replicas.build if 'replica' in task['source'] else snapshot.build)(task, snap)
                    manifest = json.loads((snap/'manifest.json').read_text())
                    files = home_files(tasks, root)
                    noop = run_task(task, base/'noop', snap, lambda home, m: FakeWorkloadDriver([[{}, '']]),
                                    lambda home: box_for(home, manifest), files)
                    self.assertFalse(noop['pass'], task['id'])
                    solved = run_task(task, base/'solved', snap,
                        lambda home, m: SolutionDriver(home, m, snap, task, box_for(home, m)),
                        lambda home: box_for(home, manifest), files)
                    self.assertTrue(solved['pass'], solved['grading'])
