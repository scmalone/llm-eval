import copy
import json
from pathlib import Path
import unittest

import test_workload_runner as fixtures
from llm_eval import paired
from llm_eval.plan import fingerprint
from llm_eval.runner import execute


def seal(plan):
    plan['plan_hash'] = fingerprint({k: v for k, v in plan.items() if k != 'plan_hash'})
    return plan


class PairedTests(unittest.TestCase):
    def setUp(self):
        helper = fixtures.WorkloadCampaignTests()
        helper.setUp()
        self.addCleanup(helper.tearDown)
        self.root = helper.root/'paired'
        plan = helper.plan({'workload': {'impl-1': {'turns': [[{'src/app.py': fixtures.FIX}, 'Fixed.']]}}})
        plan['runtime'] = 'strata'
        model = plan['models'][0]
        model['fixtures'] = [f for f in model['fixtures'] if f['id'] == 'impl-1']
        model.update(contexts=[65536], anchor=65536, settings={'request': {'reasoning_effort': 'high'}})
        a = seal(plan)
        b = copy.deepcopy(a)
        b['models'][0]['settings']['request']['reasoning_effort'] = 'medium'
        self.arms = {'A': a, 'B': seal(b)}

    def freeze(self):
        return paired.freeze(self.arms, self.root, ['settings.request.reasoning_effort'])

    def test_freezes_seed_order_and_rejects_undeclared_client_budget_and_request_drift(self):
        manifest = self.freeze()
        self.assertEqual([a['arm'] for a in manifest['attempts']], ['A', 'B', 'B', 'A', 'A', 'B'])
        self.assertEqual([a['seed'] for a in manifest['attempts']], [17, 17, 29, 29, 43, 43])
        for mutate in (lambda p: p['models'][0]['settings']['request'].update(temperature=0),
                       lambda p: p['limits'].update(request_seconds=1),
                       lambda p: p['models'][0]['workload'].update(opencode_version='2.0.0')):
            changed = copy.deepcopy(self.arms)
            mutate(changed['B']); seal(changed['B'])
            with self.assertRaisesRegex(ValueError, 'undeclared arm changes'):
                paired.freeze(changed, self.root/'changed', ['settings.request.reasoning_effort'])

    def test_fake_campaign_completes_and_resume_does_not_repeat_completed_attempts(self):
        manifest = self.freeze()
        calls = []
        def interrupted(plan, *args, **kwargs):
            if len(calls) == 1:
                raise KeyboardInterrupt('between arms')
            calls.append(plan['paired_attempt'])
            return execute(plan, *args, **kwargs)
        with self.assertRaises(KeyboardInterrupt):
            paired.run(self.root, manifest['campaign_hash'][:12], executor=interrupted)
        saved = json.loads((self.root/'paired-state.json').read_text())
        self.assertEqual(len(saved['completed']), 1)
        def continued(plan, *args, **kwargs):
            calls.append(plan['paired_attempt'])
            return execute(plan, *args, **kwargs)
        result = paired.run(self.root, manifest['campaign_hash'][:12], resume=True, executor=continued)
        self.assertTrue(result['complete'])
        self.assertEqual(len(calls), 6)
        self.assertEqual(result['arms']['A']['passed'], 3)
        self.assertEqual(result['arms']['B']['passed'], 3)
        self.assertTrue(all(p['common_success'] for p in result['pairs']))
        again = paired.run(self.root, manifest['campaign_hash'][:12], resume=True, executor=continued)
        self.assertTrue(again['complete'])
        self.assertEqual(len(calls), 6)

    def test_the_gate_freezes_one_seed_and_the_report_names_lost_and_gained_pairs(self):
        manifest = paired.freeze(self.arms, self.root, ['settings.request.reasoning_effort'], seeds=paired.SEEDS[:1])
        self.assertEqual([(a['arm'], a['seed']) for a in manifest['attempts']], [('A', 17), ('B', 17)])
        self.assertEqual(manifest['seeds'], [17])
        saved = {'completed': {'impl-1-p1-a': {'arm': 'A', 'status': 'pass', 'elapsed_seconds': 5},
                               'impl-1-p1-b': {'arm': 'B', 'status': 'quality_fail', 'elapsed_seconds': 6}}}
        result = paired.report(manifest, saved)
        self.assertEqual((result['lost_by_b'], result['gained_by_b']), (['impl-1-p1'], []))
        self.assertIn('1 paired seed', result['uncertainty'])

    def test_frozen_campaign_cannot_be_overwritten_or_tampered(self):
        manifest = self.freeze()
        self.arms['A']['models'][0]['settings']['request']['seed'] = 99
        seal(self.arms['A'])
        with self.assertRaises(ValueError):
            self.freeze()
        manifest['seeds'][0] = 99
        (self.root/'campaign.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            paired.run(self.root, manifest['campaign_hash'][:12])

    def test_restore_failure_stops_before_the_other_arm(self):
        manifest = self.freeze()
        calls = []
        def failed(*args, **kwargs):
            calls.append(1)
            return {'sections': {}, 'status': 'restore_failed', 'restoration_required': True}
        with self.assertRaisesRegex(RuntimeError, 'STOP'):
            paired.run(self.root, manifest['campaign_hash'][:12], executor=failed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads((self.root/'paired-state.json').read_text())['completed'], {})
