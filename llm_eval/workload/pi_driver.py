"""The pi harness driver: one ``pi --print --mode json`` per user turn on a single session.

pi is a second target harness beside opencode, so a result can say which agent produced it. It differs
from opencode in two ways that matter for measurement:

* its JSON event stream already carries what the metrics need (per-response ``usage``, tool names,
  durations, errors, compaction results), so nothing is parsed out of a session export;
* the session is a JSONL file in ``--session-dir``, read straight off the disk, so the 64 KiB pipe
  truncation that ``opencode export`` hit cannot happen here.

``summarize()`` reports the same keys as ``metrics.summarize`` so the two harnesses compare directly.
"""
import json
from pathlib import Path

from . import pi_config
from .driver import (HOME_CAP_MIB, LINGER_SECONDS, PROGRESS_POLL_SECONDS, STALL_SECONDS,  # noqa: F401
                     STARTUP_SECONDS, argv_chunks, run_streaming)
from .relay import HostRelay
from .sandbox import Box, relay_dir

PI_HOME = Path.home() / ".npm-global"  # where the npm-installed pi and its bundle live
PI_SESSION_DIR = ".pi/sessions"


class PiDriver:
    name = "pi"

    def __init__(self, home, model_id, ref, llm_port, request, effort="high", binary_home=PI_HOME,
                 context=131072, output=16384, compaction_at=None, session_dir=PI_SESSION_DIR,
                 relays=(), read_only=(), path=(), env=None, progress=None, stall_seconds=STALL_SECONDS,
                 turn_args=(), version="", config=None, settings=None):
        self.home = Path(home)
        self.model_id, self.ref, self.llm_port = model_id, ref, llm_port
        self.request, self.effort = dict(request), effort
        self.context, self.output, self.compaction_at = context, output, compaction_at
        self.binary_home = Path(binary_home)
        self.session_dir = Path(session_dir)
        self.extra_relays, self.extra_read_only, self.extra_path = list(relays), list(read_only), list(path)
        self.extra_env = dict(env or {})
        self.turn_args = list(turn_args)
        self.progress, self.stall_seconds = progress, stall_seconds
        self.home_cap_mib = HOME_CAP_MIB  # the session sets a task's own ``limits.home_mib``
        self.version = version
        # The plan's pinned models.json and settings.json; without them the files are rebuilt from the arguments
        self.config, self.settings = config, settings
        self.session = None
        self.events = []  # every JSON event, every turn, in order: the metrics source
        self.raw_turns = []
        self.box = None
        self._relays = []

    def install(self):
        if self.config is not None:
            models, settings = self.config, self.settings or {}
            pi_config.check(models, settings)
        else:
            models, settings, self.ref = pi_config.build(self.model_id, self.context, self.output, self.request,
                                                         self.effort, self.compaction_at)
        pi_config.write(self.home, models, settings)
        (self.home / self.session_dir).mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self.install()
        self.sockets = relay_dir()
        specs = [(f"127.0.0.1:{pi_config.SANDBOX_PORT}", "llm.sock", self.llm_port)]
        specs += [(inside, f"r{i}.sock", port) for i, (inside, port) in enumerate(self.extra_relays)]
        for _, name, port in specs:
            relay = HostRelay(self.sockets / name, port)
            relay.__enter__()
            self._relays.append(relay)
        inside_home = Path(self.box_home())
        self.box = Box(self.home, read_only=[(self.binary_home, str(inside_home / ".npm-global")),
                                             *self.extra_read_only],
                       env={**self.extra_env},
                       path=[str(inside_home / ".npm-global/bin"), *self.extra_path],
                       relays=[f"{inside}={name}" for inside, name, _ in specs], sockets=self.sockets)
        return self

    @staticmethod
    def box_home():
        return Path.home()

    def __exit__(self, *exc):
        for relay in self._relays:
            relay.__exit__(None, None, None)
        self._relays = []
        import shutil
        shutil.rmtree(self.sockets, ignore_errors=True)

    def command(self, workdir):
        command = ["pi", "--print", "--mode", "json",
                   "--provider", pi_config.LOCAL_PROVIDER, "--model", self.model_id,
                   "--session-dir", self.box.inside(self.home / self.session_dir),
                   "--thinking", self.effort,
                   "--no-extensions", "--no-mcp", "--no-skills", "--no-prompt-templates", "--no-themes",
                   "--no-approve"]  # project-local .pi/ config must not change the experiment
        if self.session:
            command += ["--session-id", self.session]
        return command + self.turn_args

    def turn(self, text, workdir, timeout):
        command = self.command(workdir)
        result = run_streaming(self.box, command + argv_chunks(text), workdir, timeout,
                               progress=self.progress, stall_seconds=self.stall_seconds,
                               home_cap_mib=getattr(self, "home_cap_mib", HOME_CAP_MIB), is_begun=lambda event, line: event is not None,
                               is_settled=lambda event: event.get("type") == "agent_settled")
        self.events.extend(result["events"])
        self.session = self.session or next((e["id"] for e in result["events"] if e.get("type") == "session"), None)
        self.raw_turns.append({"said": text, "argv": command, "returncode": result["returncode"],
                               "timed_out": result["timed_out"], "lingered": result["lingered"],
                               "stalled": result["stalled"], "hung": result["hung"],
                               "disk_cap": result["disk_cap"], "home_mib": result["home_mib"],
                               "events": result["events"], "stderr": result["stderr"]})
        kinds = [e.get("type") for e in result["events"]]
        # A tool the model started that never ended: a hang in that command, not in pi or the server (review 4).
        tool_open = kinds.count("tool_execution_start") > kinds.count("tool_execution_end")
        return {"answer": self._answer(result["events"]), "timed_out": result["timed_out"], "tool_open": tool_open,
                "returncode": result["returncode"], "events": len(result["events"]),
                "lingered": result["lingered"], "restarts": 0, "stderr_tail": result["stderr"][-1500:],
                "hung": result["hung"], "max_silence_seconds": result["max_silence_seconds"],
                "disk_cap": result["disk_cap"], "home_mib": result["home_mib"], "stalled": result["stalled"]}

    @staticmethod
    def _answer(events):
        answer = ""
        for event in events:
            message = event.get("message") or {}
            if event.get("type") != "message_end" or message.get("role") != "assistant":
                continue
            for block in message.get("content") or []:
                if block.get("type") == "text" and (block.get("text") or "").strip():
                    answer = block["text"]
        return answer.strip()

    def session_file(self):
        """The session JSONL pi wrote, found by the id it was given (the name carries a timestamp prefix)."""
        if not self.session:
            return None
        matches = sorted((self.home / self.session_dir).glob(f"*_{self.session}.jsonl"))
        return matches[-1] if matches else None

    def export(self):
        """The harness's own session record, read from disk rather than exported through a pipe."""
        path = self.session_file()
        if path is None:
            return {"session": {}, "entries": []}
        entries = []
        for line in path.read_text(errors="replace").splitlines():
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
        header = next((e for e in entries if e.get("type") == "session"), {})
        return {"session": header, "entries": entries}

    def summarize(self, export=None):
        """The metrics contract, computed from pi's own event stream (the record is not re-read)."""
        answer, tools, commands = "", {}, []
        steps = user_turns = tool_calls = tool_errors = compactions = 0
        length_finishes = output_tokens = reasoning_chars = peak = final = peak_step = 0
        tool_ms = model_ms = 0.0
        for event in self.events:
            kind = event.get("type")
            if kind == "message_end":
                message = event.get("message") or {}
                role = message.get("role")
                if role == "user":
                    user_turns += 1
                elif role == "assistant":
                    steps += 1
                    usage = message.get("usage") or {}
                    context = (usage.get("input") or 0) + (usage.get("cacheRead") or 0) + (usage.get("cacheWrite") or 0)
                    peak, final = max(peak, context), context
                    step_output = usage.get("output") or 0
                    output_tokens += step_output
                    peak_step = max(peak_step, step_output)
                    model_ms += message.get("durationMs") or 0
                    length_finishes += message.get("stopReason") == "length"
                    for block in message.get("content") or []:
                        if block.get("type") == "text" and (block.get("text") or "").strip():
                            answer = block["text"]
                        elif block.get("type") == "thinking":
                            reasoning_chars += len(block.get("thinking") or block.get("text") or "")
            elif kind == "tool_execution_start":
                tool_calls += 1
                name = event.get("toolName") or "unknown"
                tools[name] = tools.get(name, 0) + 1
                args = event.get("args") or {}
                if name == "bash" and isinstance(args.get("command"), str):
                    commands.append(args["command"])
            elif kind == "tool_execution_end":
                tool_errors += bool(event.get("isError"))
                tool_ms += event.get("durationMs") or 0
            elif kind == "compaction_end" and event.get("result"):
                compactions += 1
        return {"answer": answer.strip(), "commands": commands, "user_turns": user_turns, "steps": steps,
                "tool_calls": tool_calls, "tools": tools, "invalid_calls": 0, "tool_errors": tool_errors,
                "compactions": compactions, "peak_context_tokens": peak, "final_context_tokens": final,
                "output_tokens": output_tokens,
                # pi does not report reasoning tokens; the character count is exact, the token count is an
                # estimate at four characters per token, and the capture's own count supersedes both.
                "reasoning_tokens": reasoning_chars // 4, "reasoning_chars": reasoning_chars,
                "peak_step_output_tokens": peak_step, "length_finishes": length_finishes,
                "model_seconds": round(max(0.0, model_ms) / 1000, 1), "tool_seconds": round(tool_ms / 1000, 1),
                "ended_with_question": answer.strip().endswith("?")}

    def context_tokens(self):
        try:
            return self.summarize().get("final_context_tokens") or None
        except Exception:
            return None
