"""Scripted OpenAI-compatible chat server on loopback, for driving a real harness without a GPU.

Each agent request (one that offers tools) consumes the next scripted reply: ``{"text": ...}`` or ``{"tool_calls": [{"name",
"arguments"}]}``. Streaming (SSE chunks) and non-streaming responses are both served, every
request body is recorded, and a reply can echo the last tool result back with ``"{last_tool}"``.
Used by the integration tests and the sandbox spike; never on the live path.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading


class FakeLLM:
    def __init__(self, script, model="fake", tool_less_too=False):
        self.script = list(script)
        self.tool_less_too = tool_less_too  # a judge's requests carry no tools
        self.model = model
        self.requests = []
        self.lock = threading.Lock()
        handler = type("Handler", (_Handler,), {"owner": self})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def next_reply(self, body):
        with self.lock:
            self.requests.append(body)
            # Harness side calls (titles, summaries) carry no tools and never consume the script.
            if not body.get("tools") and not self.tool_less_too:
                return {"text": "Scripted session"}
            reply = self.script.pop(0) if self.script else {"text": "done"}
        last_tool = next((m.get("content") for m in reversed(body.get("messages", []))
                          if m.get("role") == "tool"), "")
        if isinstance(last_tool, list):
            last_tool = " ".join(p.get("text", "") for p in last_tool if isinstance(p, dict))
        if "text" in reply:
            return {**reply, "text": reply["text"].replace("{last_tool}", str(last_tool or ""))}
        return reply


class _Handler(BaseHTTPRequestHandler):
    owner = None

    def log_message(self, *args):
        pass

    def _json(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list", "data": [{"id": self.owner.model, "object": "model"}]})
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": "not found"})
        reply = self.owner.next_reply(body)
        calls = [{"index": i, "id": f"call_{len(self.owner.requests)}_{i}", "type": "function",
                  "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                 for i, c in enumerate(reply.get("tool_calls", []))]
        finish = "tool_calls" if calls else "stop"
        usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
        if not body.get("stream"):
            message = {"role": "assistant", "content": reply.get("text")}
            if calls:
                message["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
            return self._json(200, {"id": "fake", "object": "chat.completion", "model": self.owner.model,
                                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                    "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        def chunk(delta, finish_reason=None, **extra):
            payload = {"id": "fake", "object": "chat.completion.chunk", "model": self.owner.model,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}], **extra}
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        chunk({"role": "assistant", "content": ""})
        if reply.get("text"):
            chunk({"content": reply["text"]})
        if reply.get("reasoning"):
            chunk({"reasoning_content": reply["reasoning"]})
        if calls:
            chunk({"tool_calls": calls})
        chunk({}, finish, usage=usage)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
