import json
from pathlib import Path
import unittest

import test_strata as fixtures
from llm_eval.runner import execute
from llm_eval.metrics.ingest import ingest_run
from llm_eval.metrics.schema import connect
from llm_eval.state import atomic_json


class StrataMetricsTests(unittest.TestCase):
    def test_strata_ingestion_uses_capture_not_llama_log_parser(self):
        helper = fixtures.StrataTests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        plan, host, adapter = helper.adapter()
        root = helper.root/'runs'/'strata-test'
        execute(plan, root.parent, root.name, adapter=adapter, lock_path=helper.root/'lock')
        atomic_json(root/'workload/trial/raw/inference/0001/request.json', {
            'path': '/v1/chat/completions', 'request': 1, 'turn': 2, 'status': 200,
            'started': 100, 'ended': 110, 'effective_body': {'seed': 17}, 'conflicts': [],
            'response': {'first_token_seconds': 1.5, 'first_answer_seconds': 8, 'finish_reason': 'stop',
                         'usage': {'completion_tokens': 100}, 'timings': {'prompt_ms': 500}}})
        # A Strata log resembling llama syntax must not be interpreted as llama measurements.
        (root/'server-logs').mkdir(exist_ok=True)
        (root/'server-logs/fake.log').write_text('slot launch_slot_: id  0 | task 7 | processing task\n')
        conn = connect(helper.root/'metrics.sqlite')
        self.addCleanup(conn.close)
        ingest_run(conn, root)
        runtime = conn.execute('SELECT * FROM runs').fetchone()
        self.assertEqual(runtime['runtime_name'], 'strata')
        self.assertIsNone(runtime['llama_server_path'])
        self.assertIsNone(runtime['llama_server_version'])
        self.assertEqual(conn.execute('SELECT COUNT(*) FROM server_requests').fetchone()[0], 0)
        request = conn.execute('SELECT * FROM inference_requests').fetchone()
        self.assertEqual(request['ttft_seconds'], 1.5)
        self.assertEqual(request['first_answer_seconds'], 8)
        self.assertEqual(json.loads(request['usage_json'])['completion_tokens'], 100)
        ingest_run(conn, root)
        self.assertEqual(conn.execute('SELECT COUNT(*) FROM inference_requests').fetchone()[0], 1)
