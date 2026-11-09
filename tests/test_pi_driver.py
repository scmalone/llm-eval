import json
import tempfile
import unittest
from pathlib import Path

from llm_eval.workload import pi_config
from llm_eval.workload.pi_driver import PiDriver

REQUEST = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0.0,
           "repetition_penalty": 1.0, "reasoning_effort": "high", "reasoning_budget_tokens": 0,
           "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}


class ConfigTests(unittest.TestCase):
    """One loopback provider, one model, sampling exactly as pinned, compaction where the runner compacts."""

    def test_the_model_carries_the_pinned_request_and_the_context(self):
        models, settings, ref = pi_config.build("strata-iq3s", 262144, 16384, REQUEST, "high")
        entry = models["providers"]["local-llama"]["models"][0]
        self.assertEqual((ref, entry["contextWindow"], entry["maxTokens"]), ("local-llama/strata-iq3s", 262144, 16384))
        for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
            self.assertEqual(entry["samplingParams"][key], REQUEST[key], key)
        self.assertEqual(entry["samplingParams"]["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(entry["thinkingLevelMap"]["high"], "high")
        # reasoning_effort reaches the server through the thinking level, not as a sampling field
        self.assertNotIn("reasoning_effort", entry["samplingParams"])

    def test_compaction_starts_one_output_plus_the_margin_under_the_ceiling(self):
        _, settings, _ = pi_config.build("m", 131072, 16384, REQUEST)
        self.assertEqual(settings["compaction"]["reserveTokens"], 16384 + 1024)
        _, settings, _ = pi_config.build("m", 262144, 16384, REQUEST, compaction_at=200000)
        self.assertEqual(settings["compaction"]["reserveTokens"], 62144)  # the owner's 200k autocompact point

    def test_a_non_loopback_endpoint_is_refused(self):
        models, settings, _ = pi_config.build("m", 131072, 16384, REQUEST)
        models["providers"]["local-llama"]["baseUrl"] = "http://10.0.0.5:8390/v1"
        with self.assertRaises(ValueError):
            pi_config.check(models, settings)

    def test_a_second_provider_is_refused(self):
        models, settings, _ = pi_config.build("m", 131072, 16384, REQUEST)
        models["providers"]["anthropic"] = {"baseUrl": "http://127.0.0.1:1/v1"}
        with self.assertRaises(ValueError):
            pi_config.check(models, settings)

    def test_the_expected_contract_uses_pi_s_field_names(self):
        expected = pi_config.expected_request("m", 131072, 16384, REQUEST, "high")
        self.assertEqual(expected["max_tokens"], 16384)
        self.assertEqual(expected["reasoning_effort"], "high")
        self.assertNotIn("reasoning_budget_tokens", expected)  # pi has no such field


def event_stream():
    """One assistant response with a bash call, then a final answer: pi's real event shapes."""
    return [
        {"type": "session", "id": "01a18d28-abc"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "fix it"}]}},
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "toolUse", "durationMs": 1200,
                                            "usage": {"input": 1000, "output": 100, "cacheRead": 0, "cacheWrite": 0},
                                            "content": [{"type": "thinking", "thinking": "think " * 50},
                                                        {"type": "toolCall", "toolName": "bash",
                                                         "args": {"command": "ls -la"}}]}},
        {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "bash", "args": {"command": "ls -la"}},
        {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "bash", "isError": False, "durationMs": 30},
        {"type": "tool_execution_start", "toolCallId": "c2", "toolName": "read", "args": {"path": "a.py"}},
        {"type": "tool_execution_end", "toolCallId": "c2", "toolName": "read", "isError": True, "durationMs": 5},
        {"type": "message_end", "message": {"role": "assistant", "stopReason": "stop", "durationMs": 800,
                                            "usage": {"input": 1200, "output": 50, "cacheRead": 300, "cacheWrite": 0},
                                            "content": [{"type": "text", "text": "Fixed it."}]}},
        {"type": "turn_end", "message": {"role": "assistant"}, "toolResults": []},
        {"type": "agent_settled", "aborted": False},
    ]


class MetricsTests(unittest.TestCase):
    """pi's own events carry every number the harness needs, so nothing is parsed out of an export."""

    def driver(self):
        driver = PiDriver.__new__(PiDriver)  # only the metrics path is under test
        driver.events, driver.session = event_stream(), "01a18d28-abc"
        return driver

    def test_the_contract_is_filled_from_the_event_stream(self):
        metrics = self.driver().summarize()
        self.assertEqual(metrics["answer"], "Fixed it.")
        self.assertEqual(metrics["steps"], 2)
        self.assertEqual(metrics["user_turns"], 1)
        self.assertEqual(metrics["tool_calls"], 2)
        self.assertEqual(metrics["tools"], {"bash": 1, "read": 1})
        self.assertEqual(metrics["tool_errors"], 1)
        self.assertEqual(metrics["commands"], ["ls -la"])
        self.assertEqual(metrics["output_tokens"], 150)
        self.assertEqual(metrics["peak_context_tokens"], 1500)   # 1200 + 300 cacheRead
        self.assertEqual(metrics["final_context_tokens"], 1500)
        self.assertEqual(metrics["peak_step_output_tokens"], 100)
        self.assertEqual(metrics["length_finishes"], 0)
        self.assertEqual(metrics["model_seconds"], 2.0)
        self.assertEqual(metrics["tool_seconds"], 0.0)  # 35 ms rounds to zero
        self.assertEqual(metrics["reasoning_tokens"], 300 // 4)
        self.assertEqual(metrics["reasoning_chars"], 300)

    def test_a_length_finish_is_recorded(self):
        driver = self.driver()
        driver.events[-3]["message"]["stopReason"] = "length"
        self.assertEqual(driver.summarize()["length_finishes"], 1)


class CommandTests(unittest.TestCase):
    """The turn command pins the model, the thinking level and the session, and trusts nothing local."""

    def driver(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        driver = PiDriver(Path(tmp.name), "strata-iq3s", "local-llama/strata-iq3s", 8391, REQUEST, effort="high")
        driver.session = "01a18d28-abc"
        driver.box = type("B", (), {"inside": staticmethod(lambda p: "/home/tester/.pi/sessions")})()
        return driver

    def test_the_command_pins_the_experiment(self):
        command = self.driver().command(Path("/tmp/repo"))
        for flag in ("--print", "--mode", "json", "--provider", "local-llama", "--model", "strata-iq3s",
                     "--thinking", "high", "--no-approve", "--no-extensions", "--no-mcp"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("--session-id") + 1], "01a18d28-abc")

    def test_install_writes_both_agent_files(self):
        driver = self.driver()
        driver.install()
        agent = Path(driver.home) / ".pi" / "agent"
        self.assertIn("local-llama", json.loads((agent / "models.json").read_text())["providers"])
        self.assertIn("compaction", json.loads((agent / "settings.json").read_text()))


class SessionFileTests(unittest.TestCase):
    """The session is read off the disk, so the 64 KiB pipe that cut opencode's export cannot apply."""

    def test_the_session_file_is_found_by_its_id_and_parsed(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name)
        (home / ".pi/sessions").mkdir(parents=True)
        path = home / ".pi/sessions" / "2026-10-31T00-00-00-000Z_01a18d28-abc.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in event_stream()) + "\nnot json\n")
        driver = PiDriver(home, "m", "local-llama/m", 8391, REQUEST)
        driver.session = "01a18d28-abc"
        export = driver.export()
        self.assertEqual(export["session"]["id"], "01a18d28-abc")
        self.assertEqual(len(export["entries"]), len(event_stream()))  # the unparsable line is skipped

    def test_no_session_yet_is_an_empty_record(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        driver = PiDriver(Path(tmp.name), "m", "local-llama/m", 8391, REQUEST)
        self.assertEqual(driver.export(), {"session": {}, "entries": []})


if __name__ == "__main__":
    unittest.main()
