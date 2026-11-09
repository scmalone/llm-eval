"""Run-local HTTP capture and experiment contract for OpenAI-compatible inference.

The upstream is a fixed host-loopback port. Missing experiment fields are supplied by
this benchmark boundary and recorded. Conflicting client values are refused and recorded;
they never become a valid result from the declared arm. Bodies and streaming bytes stay raw.
"""
import copy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import socket

from ..state import atomic_json


class ResponseEvidence:
    """Parse verified OpenAI JSON/SSE fields without grading the reasoning channel."""
    def __init__(self, started, streaming):
        self.started, self.streaming = started, streaming
        self.pending = b''
        self.data = {'answer': '', 'reasoning': '', 'usage': None, 'timings': None,
                     'finish_reason': None, 'first_answer_seconds': None, 'first_tool_seconds': None,
                     'first_reasoning_seconds': None, 'first_token_seconds': None}

    def event(self, body):
        for key in ('usage', 'timings'):
            if body.get(key):
                self.data[key] = body[key]
        for choice in body.get('choices', []):
            delta = choice.get('delta', choice.get('message', {}))
            for source, target in (('content', 'answer'), ('reasoning_content', 'reasoning')):
                if isinstance(delta.get(source), str):
                    self.data[target] += delta[source]
            elapsed = time.time() - self.started
            if (delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls')) and self.data['first_token_seconds'] is None:
                self.data['first_token_seconds'] = elapsed
            if delta.get('reasoning_content') and self.data['first_reasoning_seconds'] is None:
                self.data['first_reasoning_seconds'] = elapsed
            if delta.get('content') and self.data['first_answer_seconds'] is None:
                self.data['first_answer_seconds'] = elapsed
            if delta.get('tool_calls') and self.data['first_tool_seconds'] is None:
                self.data['first_tool_seconds'] = elapsed
            if choice.get('finish_reason'):
                self.data['finish_reason'] = choice['finish_reason']

    def feed(self, chunk):
        self.pending += chunk
        if self.streaming:
            while b'\n' in self.pending:
                line, self.pending = self.pending.split(b'\n', 1)
                if line.startswith(b'data:') and line[5:].strip() != b'[DONE]':
                    self.event(json.loads(line[5:]))

    def finish(self):
        if not self.streaming:
            self.event(json.loads(self.pending))
        if self.data['finish_reason'] is None:
            raise ValueError('incomplete inference response: missing finish reason')
        return self.data


# Top-level limits a client may lower on its own: pi caps max_tokens at the window minus the context minus 4,096
# and derives its thinking budget from that cap, so the last turns before compaction ask for less. Asking for less
# is recorded (lowered_limits), never refused; asking for more is a conflict.
CEILINGS = ('max_tokens', 'max_completion_tokens', 'thinking_budget_tokens', 'reasoning_budget_tokens',
            'thinking_token_budget')


def _lowered(client, pinned):
    return (isinstance(client, int) and isinstance(pinned, int) and not isinstance(client, bool)
            and 0 < client < pinned)


def lowered_limits(body, expected):
    """The CEILINGS the client asked below the pinned value for (max_completion_tokens against max_tokens)."""
    pinned = {**{'max_completion_tokens': expected.get('max_tokens')}, **expected}
    return [k for k in CEILINGS if k in (body or {}) and _lowered(body[k], pinned.get(k))]


def resolved_request(body, expected):
    result = copy.deepcopy(body)
    supplied, conflicts = [], []
    def merge(target, wanted, path=''):
        for key, value in wanted.items():
            name = path + key
            if key not in target:
                target[key] = copy.deepcopy(value)
                supplied.append(name)
            elif isinstance(value, dict) and isinstance(target[key], dict):
                merge(target[key], value, name + '.')
            elif target[key] != value and not (not path and key in CEILINGS and _lowered(target[key], value)):
                conflicts.append(name)
    wanted = dict(expected)
    if 'max_completion_tokens' in result and 'max_tokens' not in result and \
            _lowered(result['max_completion_tokens'], expected.get('max_tokens')):
        wanted['max_tokens'] = result['max_completion_tokens']  # the pinned value would undo the client's lower cap
    if any(k in result and _lowered(result[k], expected.get('max_tokens')) for k in ('max_tokens', 'max_completion_tokens')):
        # a lowered cap near the window's end: pi omits a thinking budget it computes as zero; don't restore it
        for key in CEILINGS[2:]:
            if key not in result:
                wanted.pop(key, None)
    merge(result, wanted)
    for alias, name in (('max_completion_tokens', 'max_tokens'),):
        if alias in result and name in expected and result[alias] != expected[name] \
                and not _lowered(result[alias], expected[name]):
            conflicts.append(alias)
    effort = (result.get('reasoning') or {}).get('effort') if isinstance(result.get('reasoning'), dict) else None
    if effort is not None and 'reasoning_effort' in expected and effort != expected['reasoning_effort']:
        conflicts.append('reasoning.effort')
    template_effort = (result.get('chat_template_kwargs') or {}).get('reasoning_effort')
    if template_effort is not None and 'reasoning_effort' in expected and template_effort != expected['reasoning_effort']:
        conflicts.append('chat_template_kwargs.reasoning_effort')
    return result, supplied, conflicts


def thinking_cap_hits(records):
    """Chat turns whose thinking ran into the request's thinking budget, where llama.cpp forces the block closed.

    The server says nothing when it forces the close, so this is inferred: the turn generated at least the
    budget, and its reasoning (characters / 4, within about 10% of the token count on captured runs) is at
    least 90% of it."""
    hits = 0
    for record in records:
        body, response = record.get('effective_body') or {}, record.get('response') or {}
        budget = body.get('thinking_budget_tokens') or body.get('reasoning_budget_tokens')
        generated = (response.get('usage') or {}).get('completion_tokens') or 0
        if budget and budget > 0 and generated >= budget and len(response.get('reasoning') or '') // 4 >= 0.9 * budget:
            hits += 1
    return hits


class InferenceCapture:
    def __init__(self, upstream, directory, expected, model=None, timeout=600):
        self.upstream = upstream
        self.directory = Path(directory)
        self.expected = copy.deepcopy(expected)
        self.model = model
        self.timeout = timeout
        self.errors = []
        self.ok = 0  # chat completions the server answered with 200
        self.client_disconnects = 0  # streams the client stopped reading before they ended
        self.reasoning_chars = 0  # thinking the server streamed, which no client reliably counts
        self.reasoning_blocks = 0
        self.records = []
        self.lock = threading.Lock()
        self.connections = set()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.forward(None)

            def do_POST(self):
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 32 * 1024 * 1024:
                        raise ValueError('invalid inference body length')
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError('inference body must be an object')
                except (ValueError, json.JSONDecodeError):
                    self.send_error(400, 'invalid inference body')
                    return
                self.forward(body)

            def forward(self, body):
                if self.path not in ('/v1/chat/completions', '/v1/models', '/models', '/health'):
                    self.send_error(404)
                    return
                with owner.lock:
                    number = len(owner.records) + 1
                    record = {'request': number, 'path': self.path, 'started': time.time(), 'client_body': body,
                              'turn': sum(m.get('role') == 'user' for m in (body or {}).get('messages', []))}
                    owner.records.append(record)
                folder = owner.directory / f'{number:04}'
                response_file = folder / 'response.bin'
                status, connection, upstream_socket = None, None, None
                try:
                    if body is not None:
                        lowered = lowered_limits(body, owner.expected)
                        body, supplied, conflicts = resolved_request(body, owner.expected)
                        record.update(effective_body=body, supplied=supplied, conflicts=conflicts,
                                      lowered_limits=lowered)
                        if owner.model:
                            body['model'] = owner.model
                        if conflicts:
                            with owner.lock:
                                owner.errors.append({'request': number, 'conflicts': conflicts})
                            atomic_json(folder / 'request.json', record)
                            self.send_error(409, 'benchmark request contract conflict: ' + ', '.join(conflicts))
                            return
                    atomic_json(folder / 'request.json', record)
                    connection = http.client.HTTPConnection('127.0.0.1', owner.upstream, timeout=owner.timeout)
                    connection.connect()
                    upstream_socket = connection.sock
                    with owner.lock:
                        owner.connections.update((upstream_socket, self.connection))
                    connection.request('POST' if body is not None else 'GET', self.path,
                        body=json.dumps(body) if body is not None else None, headers={'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    evidence = ResponseEvidence(record['started'], 'text/event-stream' in response.getheader('Content-Type', ''))
                    status = response.status
                    record['status'] = status
                    self.send_response(status)
                    self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.close_connection = True
                    with response_file.open('wb') as output:
                        while chunk := response.read1(65536):
                            output.write(chunk)
                            if 'first_byte' not in record:
                                record['first_byte'] = time.time()
                            if self.path == '/v1/chat/completions' and status == 200:
                                evidence.feed(chunk)
                            try:
                                self.wfile.write(chunk)
                                self.wfile.flush()
                            except OSError as exc:  # the client hung up: its business, not the server's
                                record['client_disconnected'] = str(exc)
                                break
                    if 'client_disconnected' in record:
                        # The client hung up mid-stream (an abort, a cancelled turn): the stream was not read to its
                        # end, so it has no finish reason, and that is not a server or transport failure.
                        with owner.lock:
                            owner.client_disconnects += 1
                    elif self.path == '/v1/chat/completions' and status == 200:
                        record['response'] = evidence.finish()
                        thinking = record['response'].get('reasoning') or ''
                        with owner.lock:
                            owner.ok += 1
                            if thinking:
                                owner.reasoning_chars += len(thinking)
                                owner.reasoning_blocks += 1
                    if status >= 400:
                        with owner.lock:
                            owner.errors.append({'request': number, 'http_status': status})
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    record['error'] = str(exc)
                    with owner.lock:
                        owner.errors.append({'request': number, 'transport_error': str(exc)})
                    if status is None:
                        self.send_error(502)
                finally:
                    if connection:
                        connection.close()
                        with owner.lock:
                            owner.connections.discard(upstream_socket)
                            owner.connections.discard(self.connection)
                    record['ended'] = time.time()
                    atomic_json(folder / 'request.json', record)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.port = self.server.server_address[1]

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def reasoning_summary(self):
        """What the server streamed as thinking. The client's own count is the export's business and can be
        zero for a provider it does not model; this is the measurement."""
        return {"blocks": self.reasoning_blocks, "chars": self.reasoning_chars,
                "tokens_estimate": self.reasoning_chars // 4, "cap_hits": thinking_cap_hits(self.records)}

    @property
    def client_errors(self):
        """Errors the client caused: it asked for something other than the pinned experiment, or sent a
        request the server refused. An ops task may ask the agent to rewrite the client's own config, and the
        drift is then the trial's result, not a broken machine."""
        return [e for e in self.errors if 'conflicts' in e or 400 <= (e.get('http_status') or 0) < 500]

    @property
    def server_errors(self):
        """Errors the server or the machine caused."""
        return [e for e in self.errors if e.get('transport_error') or (e.get('http_status') or 0) >= 500]

    def __exit__(self, *args):
        self.server.shutdown()
        # A canceled harness must not leave a handler waiting a full generation timeout.
        with self.lock:
            active = list(self.connections)
        for connection in active:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        self.server.server_close()
        self.thread.join(timeout=5)
        chats = [r for r in self.records if r['path'] == '/v1/chat/completions']
        reasoning = [r['response']['reasoning'] for r in chats if r.get('response', {}).get('reasoning')]
        histories = [m.get('reasoning_content', '') for r in chats for m in
                     (r.get('client_body') or {}).get('messages', []) if m.get('role') == 'assistant']
        preservation = ('present_in_client_history_template_unverified' if reasoning and any(t in histories for t in reasoning)
                        else 'not_observed_in_client_history' if reasoning else 'no_reasoning_observed')
        atomic_json(self.directory / 'contract.json', {'expected': self.expected, 'requests': len(chats),
            # A run whose every stream the client abandoned has no completed exchange to show for itself.
            'valid': bool(chats) and not self.errors and (self.ok > 0 or not self.client_disconnects), 'errors': self.errors, 'successful': self.ok, 'client_disconnects': self.client_disconnects,
            'client_errors': len(self.client_errors), 'server_errors': len(self.server_errors),
            'reasoning': self.reasoning_summary(),
            'reasoning_preservation': preservation, 'server_effective_settings': 'requires retained props/template evidence'})
