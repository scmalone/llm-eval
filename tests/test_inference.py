import http.client
import json
from pathlib import Path
import tempfile
import unittest
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_eval.workload.fakellm import FakeLLM
from llm_eval.workload.inference import InferenceCapture, ResponseEvidence, resolved_request


class InferenceTests(unittest.TestCase):
    def test_cancel_closes_a_stream_without_waiting_for_request_timeout(self):
        ready, release = threading.Event(), threading.Event()
        class Slow(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: {"choices":[]}\n\n')
                self.wfile.flush()
                ready.set()
                release.wait(5)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Slow)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                before = time.monotonic()
                with InferenceCapture(server.server_port, tmp, {}, timeout=30) as capture:
                    client = http.client.HTTPConnection('127.0.0.1', capture.port, timeout=5)
                    client.request('POST', '/v1/chat/completions', '{}')
                    response = client.getresponse()
                    self.assertTrue(ready.wait(2))
                self.assertLess(time.monotonic() - before, 3)
                client.close()
                self.assertFalse(json.loads((Path(tmp)/'contract.json').read_text())['valid'])
        finally:
            release.set()
            server.shutdown()
            server.server_close()

    def test_stream_parser_retains_reasoning_and_timings_separately(self):
        events = [{'choices': [{'delta': {'reasoning_content': 'private thought'}}]},
                  {'choices': [{'delta': {'tool_calls': [{'index': 0}]}}]},
                  {'choices': [{'delta': {'content': 'answer'}, 'finish_reason': 'length'}],
                   'usage': {'completion_tokens': 42}, 'timings': {'prompt_ms': 12}}]
        raw = ''.join('data: ' + json.dumps(e) + '\n\n' for e in events).encode()
        evidence = ResponseEvidence(0, True)
        for byte in raw:
            evidence.feed(bytes([byte]))
        data = evidence.finish()
        self.assertEqual(data['answer'], 'answer')
        self.assertEqual(data['reasoning'], 'private thought')
        self.assertEqual(data['finish_reason'], 'length')
        self.assertEqual(data['usage']['completion_tokens'], 42)
        self.assertIsNotNone(data['first_tool_seconds'])

    def test_nested_contract_is_copied_and_conflicts_are_visible(self):
        client = {'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
        expected = {'temperature': 1, 'seed': 17, 'chat_template_kwargs': {'enable_thinking': True}}
        effective, supplied, conflicts = resolved_request(client, expected)
        self.assertEqual(supplied, ['seed'])
        self.assertEqual(conflicts, ['temperature', 'chat_template_kwargs.enable_thinking'])
        self.assertNotIn('seed', client)
        self.assertEqual(effective['seed'], 17)

    def test_a_client_may_ask_for_less_output_but_not_more(self):
        """pi caps max_tokens at the window minus the context minus 4,096 and derives its thinking budget from
        that cap, so near compaction it asks for less than the pinned limits. That used to be a 409."""
        from llm_eval.workload.inference import lowered_limits
        expected = {'max_tokens': 32768, 'thinking_budget_tokens': 16384, 'temperature': 1.0}
        client = {'max_tokens': 9000, 'thinking_budget_tokens': 7976, 'temperature': 1.0}
        effective, supplied, conflicts = resolved_request(client, expected)
        self.assertEqual((supplied, conflicts), ([], []))
        self.assertEqual(lowered_limits(client, expected), ['max_tokens', 'thinking_budget_tokens'])
        # pi omits a budget it computes as zero; the pinned one must not come back under a lowered cap
        effective, supplied, _ = resolved_request({'max_tokens': 900}, expected)
        self.assertNotIn('thinking_budget_tokens', effective)
        # the alias is mirrored into max_tokens rather than overridden by the pinned value
        effective, _, conflicts = resolved_request({'max_completion_tokens': 9000}, expected)
        self.assertEqual((effective['max_tokens'], conflicts), (9000, []))
        _, _, conflicts = resolved_request({'max_tokens': 40000}, expected)
        self.assertEqual(conflicts, ['max_tokens'])

    def test_raw_response_and_effective_request_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp, FakeLLM([{'text': 'OK'}]) as llm:
            with InferenceCapture(llm.port, tmp, {'seed': 17}, model='strata') as capture:
                connection = http.client.HTTPConnection('127.0.0.1', capture.port, timeout=5)
                connection.request('POST', '/v1/chat/completions', json.dumps({
                    'messages': [{'role': 'user', 'content': 'hello'}], 'stream': True}),
                    {'Content-Type': 'application/json'})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                raw = response.read()
                connection.close()
            self.assertEqual((Path(tmp)/'0001/response.bin').read_bytes(), raw)
            self.assertEqual(llm.requests[0]['seed'], 17)
            self.assertEqual(llm.requests[0]['model'], 'strata')
            self.assertTrue(json.loads((Path(tmp)/'contract.json').read_text())['valid'])

    def test_override_is_refused_before_upstream(self):
        with tempfile.TemporaryDirectory() as tmp, FakeLLM([]) as llm:
            with InferenceCapture(llm.port, tmp, {'temperature': 1}) as capture:
                connection = http.client.HTTPConnection('127.0.0.1', capture.port, timeout=5)
                connection.request('POST', '/v1/chat/completions', json.dumps({'temperature': 0}))
                response = connection.getresponse()
                self.assertEqual(response.status, 409)
                response.read()
                connection.close()
            self.assertFalse(llm.requests)
            self.assertFalse(json.loads((Path(tmp)/'contract.json').read_text())['valid'])


class ErrorClassificationTests(unittest.TestCase):
    """Drift the client caused fails the trial; a server that is not serving stops the block."""

    def capture(self, errors, ok):
        from llm_eval.workload.inference import InferenceCapture
        capture = InferenceCapture.__new__(InferenceCapture)
        capture.errors, capture.ok = errors, ok
        return capture

    def test_a_client_that_edits_its_own_config_is_the_trials_result(self):
        capture = self.capture([{"request": 11, "conflicts": ["max_tokens"]},
                                {"request": 19, "http_status": 400}], ok=18)
        self.assertEqual(len(capture.client_errors), 2)
        self.assertEqual(capture.server_errors, [])

    def test_a_server_that_stops_serving_is_infrastructure(self):
        capture = self.capture([{"request": 26, "transport_error": "[Errno 32] Broken pipe"},
                                {"request": 27, "http_status": 503}], ok=18)
        self.assertEqual(len(capture.server_errors), 2)
        self.assertEqual(capture.client_errors, [])

    def test_a_client_that_hangs_up_is_not_a_transport_failure(self):
        """A broken pipe writing back to the client is the client's doing, not the server's."""
        import socket
        from llm_eval.workload.inference import InferenceCapture
        capture = InferenceCapture.__new__(InferenceCapture)
        capture.errors, capture.ok = [], 3
        self.assertEqual(capture.server_errors, [])
        self.assertEqual(capture.client_errors, [])
