from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from llm_eval import cohort
from llm_eval.workload.graders import check_answer
from llm_eval.workload.tasks import validate


class CohortTests(unittest.TestCase):
    def test_roster_is_balanced_and_document_grading_rejects_incomplete_and_wrong_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = cohort.prepare(tmp, lambda text: len(text.split()))
            self.assertEqual(Counter(t['cohort_category'] for t in tasks), {'coding': 6, 'document': 6, 'tool': 6})
            for task in tasks:
                validate(task)
                self.assertEqual(task['runs'], 1)
                if task['cohort_category'] != 'document':
                    continue
                check = task['checks'][0]
                answer = task['solution']['answer']
                self.assertTrue(check_answer(check, {'answer': answer}, None)['pass'])
                for wrong in ('', 'Rowan [R-05]', 'K-9999 [R-01]', answer.split('[')[0]):
                    self.assertFalse(check_answer(check, {'answer': wrong}, None)['pass'])
                measured = task['prompt_measurement']
                self.assertGreaterEqual(measured['tokens'], measured['target'] * .995)
                self.assertLessEqual(measured['tokens'], measured['target'])
            again = cohort.prepare(tmp, lambda text: len(text.split()))
            self.assertEqual([t['prompt'] for t in tasks], [t['prompt'] for t in again])
            path = Path(tasks[-1]['_path'])
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'frozen cohort'):
                cohort.prepare(tmp, lambda text: len(text.split()))
