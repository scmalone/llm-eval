"""Real pi in the real sandbox against a scripted fake LLM. No GPU, no network.

The point is the driver, not the model: it proves pi starts inside the bwrap box with its own agent
directory, that a turn's events carry the metrics the harness grades on, and that the session record
is read off the disk rather than exported through a pipe.

Needs bwrap and the pi binary, and Unix sockets, so run it outside an agent's own sandbox:
    python3 -m unittest discover -s tests_integration -v
"""
import json
import shutil
from pathlib import Path
import tempfile
import unittest

from llm_eval.workload import pi_config
from llm_eval.workload.fakellm import FakeLLM
from llm_eval.workload.inference import InferenceCapture
from llm_eval.workload.pi_driver import PI_HOME, PiDriver
from llm_eval.workload.sandbox import Box
from llm_eval.workload.session import run_task
from llm_eval.workload.snapshot import build, git
from tests.test_workload import commit

APP_TEST = ("import unittest\nfrom src.app import add\n\nclass T(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(add(2, 2), 4)\n")
HIDDEN_TEST = ("import unittest\nfrom src.app import add\n\nclass H(unittest.TestCase):\n"
               "    def test_negative(self):\n        self.assertEqual(add(-1, 1), 0)\n")
REQUEST = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "presence_penalty": 0.0,
           "repetition_penalty": 1.0, "reasoning_effort": "high", "reasoning_budget_tokens": 0,
           "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}


class PiHarnessTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(shutil.which("bwrap"), "bwrap is required")
        self.assertTrue((PI_HOME / "bin/pi").exists(), "the pi binary is required")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_pi_fixes_a_bug_and_reports_the_metrics_contract(self):
        repo = self.root / "src-repo"
        repo.mkdir()
        git(["init", "-q", "-b", "main"], repo, isolated=True)
        base = commit(repo, {"src/__init__.py": "", "src/app.py": "def add(a, b):\n    return a - b\n",
                             "tests/__init__.py": "", "tests/test_app.py": APP_TEST}, "base")
        answer = commit(repo, {"src/app.py": "def add(a, b):\n    return a + b\n",
                               "tests/test_hidden.py": HIDDEN_TEST}, "fix")
        task = {"id": "impl-tiny", "kind": "workload", "category": "implementation",
                "prompt": "add() is broken; fix it.",
                "source": {"repo": str(repo), "commit": base, "answer": answer, "withhold": ["tests/test_hidden.py"]},
                "checks": [{"type": "tests", "command": ["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."]},
                           {"type": "diff_scope", "allow": ["src/**"], "min_changed": 1}],
                "limits": {"turns": 2, "wall_seconds": 300}}
        build(task, self.root / "snap")
        script = [{"tool_calls": [{"name": "bash", "arguments": {
                      "command": "sed -i 's/a - b/a + b/' src/app.py && python3 -m unittest discover -s tests -t .",
                      "description": "fix and test"}}]},
                  {"text": "Fixed add()."}]
        expected = pi_config.expected_request("tiny", 65536, 12288, REQUEST, "high")
        with FakeLLM(script) as llm, InferenceCapture(llm.port, self.root / "inference", expected) as capture:
            result = run_task(task, self.root / "run", self.root / "snap",
                              lambda home, manifest: PiDriver(home, "tiny", "local-llama/tiny", capture.port,
                                                              REQUEST, context=65536, output=12288),
                              lambda home: Box(home))
        self.assertTrue(result["pass"], json.dumps(result.get("grading"), indent=1)[:800])
        metrics = result["metrics"]
        self.assertEqual(metrics["tools"].get("bash"), 1)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertGreater(metrics["output_tokens"], 0)
        self.assertGreater(metrics["peak_context_tokens"], 0)
        self.assertEqual(metrics["steps"], 2)
        self.assertIn("Fixed add().", result["answer"])
        # the contract the capture checked is the one pi was configured to send
        self.assertTrue(json.loads((self.root / "inference/contract.json").read_text())["valid"], )
        # the session record came off the disk
        export = json.loads((self.root / "run/export.json").read_text())
        self.assertEqual(export["session"]["id"], result["turns"][0].get("session") or export["session"]["id"])
        self.assertTrue(export["entries"])

    def test_the_owner_s_entry_and_agents_md_reach_the_request(self):
        # production declares no reasoning flag: pi must still run with --thinking high, send no effort of its
        # own (the capture supplies the pinned one), use the system role and carry the owner's AGENTS.md
        repo = self.root / "src-repo"
        repo.mkdir()
        git(["init", "-q", "-b", "main"], repo, isolated=True)
        base = commit(repo, {"NOTES.md": "The answer is 42.\n"}, "notes")
        task = {"id": "lookup-tiny", "kind": "workload", "category": "lookup", "prompt": "What is the answer?",
                "source": {"repo": str(repo), "commit": base}, "checks": [{"type": "answer", "all": ["42"]}],
                "limits": {"turns": 1, "wall_seconds": 300}}
        build(task, self.root / "snap")
        production = {"contextWindow": 65536, "maxTokens": 12288}
        models, settings, ref = pi_config.build("tiny", 65536, 12288, REQUEST, production=production)
        agents = (pi_config.AGENT_DIR / "AGENTS.md").as_posix()
        expected = pi_config.expected_request("tiny", 65536, 12288, REQUEST, "high")
        with FakeLLM([{"text": "42."}]) as llm, InferenceCapture(llm.port, self.root / "inference", expected) as capture:
            result = run_task(task, self.root / "run", self.root / "snap",
                              lambda home, manifest: PiDriver(home, "tiny", ref, capture.port, REQUEST, context=65536,
                                                              output=12288, config=models, settings=settings),
                              lambda home: Box(home), home_files={agents: "# Working agreement\n\nSay MARKER-7.\n"})
        self.assertTrue(result["pass"], json.dumps(result.get("grading"), indent=1)[:600])
        record = json.loads(next((self.root / "inference").glob("*/request.json")).read_text())
        sent = record["client_body"]
        self.assertNotIn("reasoning_effort", sent)
        self.assertEqual(sent["messages"][0]["role"], "system")
        self.assertIn("MARKER-7", sent["messages"][0]["content"])
        self.assertEqual(record["effective_body"]["reasoning_effort"], "high")
        self.assertEqual(sent["max_completion_tokens"], 12288)

    def test_pi_sends_the_thinking_fields_the_contract_replays(self):
        # 2026-10-31: production entries declare reasoning with chat-template kwargs and a budget field, so pi
        # sends thinking itself; the capture must see exactly what thinking_request predicts, supply none of it,
        # and refuse nothing (Swift's template raises on the "high" the capture used to supply)
        repo = self.root / "src-repo"
        repo.mkdir()
        git(["init", "-q", "-b", "main"], repo, isolated=True)
        base = commit(repo, {"NOTES.md": "The answer is 42.\n"}, "notes")
        task = {"id": "lookup-tiny", "kind": "workload", "category": "lookup", "prompt": "What is the answer?",
                "source": {"repo": str(repo), "commit": base}, "checks": [{"type": "answer", "all": ["42"]}],
                "limits": {"turns": 1, "wall_seconds": 300}}
        build(task, self.root / "snap")
        request = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}  # a llama.cpp preset's sampling
        for name, level in (("swift-qwen3.8-27b-vision", "high"), ("muse-glimmer-30b-dflash", "max"),
                            ("qwen3.8-27b-q4", "off"), ("gemma-4-26b-a4b-it", "max")):
            with self.subTest(model=name, level=level):
                run = self.root / f"{name}-{level}"
                production = pi_config.production_entry(name)
                models, settings, ref = pi_config.build(name, 65536, 32768, request, level, production=production)
                expected = pi_config.expected_request(name, 65536, 32768, request, level, production)
                with FakeLLM([{"text": "42."}]) as llm, \
                        InferenceCapture(llm.port, run / "inference", expected) as capture:
                    result = run_task(task, run / "run", self.root / "snap",
                                      lambda home, manifest: PiDriver(home, name, ref, capture.port, request,
                                                                      effort=level, context=65536, output=32768,
                                                                      config=models, settings=settings),
                                      lambda home: Box(home))
                self.assertTrue(result["pass"], json.dumps(result.get("grading"), indent=1)[:600])
                record = json.loads(next((run / "inference").glob("*/request.json")).read_text())
                sent = record["client_body"]
                thinking = pi_config.thinking_request(production, level, sent.get("max_completion_tokens",
                                                                                   sent.get("max_tokens")))
                self.assertEqual({k: sent.get(k) for k in thinking}, thinking)
                self.assertNotIn("reasoning_effort", sent)
                self.assertEqual(record["conflicts"], [])
                self.assertFalse([s for s in record["supplied"] if "thinking" in s or "reasoning" in s], record["supplied"])

    def test_a_two_turn_session_keeps_its_history(self):
        repo = self.root / "src-repo"
        repo.mkdir()
        git(["init", "-q", "-b", "main"], repo, isolated=True)
        base = commit(repo, {"NOTES.md": "The answer is 42.\n"}, "notes")
        task = {"id": "depth-tiny", "kind": "workload", "category": "lookup",
                "prelude": ["Remember the codename TEAL-9. Say OK."],
                "prompt": "What codename did I give you?", "source": {"repo": str(repo), "commit": base},
                "checks": [{"type": "answer", "all": ["TEAL-9"]}], "limits": {"turns": 1, "wall_seconds": 300}}
        build(task, self.root / "snap")
        script = [{"text": "OK."}, {"text": "You gave me TEAL-9."}]
        with FakeLLM(script) as llm:
            result = run_task(task, self.root / "run", self.root / "snap",
                              lambda home, manifest: PiDriver(home, "tiny", "local-llama/tiny", llm.port, REQUEST,
                                                              context=65536, output=12288),
                              lambda home: Box(home))
        self.assertTrue(result["pass"], json.dumps(result.get("grading"), indent=1)[:600])
        self.assertEqual([t["source"] for t in result["turns"]], ["prelude", "prompt"])
        self.assertEqual(result["metrics"]["user_turns"], 2)
        last = [r for r in llm.requests if r.get("tools")][-1]
        self.assertIn("TEAL-9", str(last["messages"]))  # the prelude is in the history the task sees


if __name__ == "__main__":
    unittest.main()
