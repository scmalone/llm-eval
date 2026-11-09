"""Real opencode in the real sandbox against a scripted fake LLM. No GPU, no network.

Needs bwrap and the opencode binary, and Unix sockets, so run it outside an agent's own sandbox:
    python3 -m unittest discover -s tests_integration -v
Missing prerequisites fail loudly; nothing here is skipped.
"""
import shutil
import json
from pathlib import Path
import tempfile
import unittest

from llm_eval.workload import opencode_config as oc
from llm_eval.workload.driver import OPENCODE_HOME, OpencodeDriver
from llm_eval.workload.fakellm import FakeLLM
from llm_eval.workload.inference import InferenceCapture
from llm_eval.workload.sandbox import Box
from llm_eval.workload.session import run_task
from llm_eval.workload.snapshot import build, git
from tests.test_workload import commit

APP_TEST = "import unittest\nfrom src.app import add\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 2), 4)\n"
HIDDEN_TEST = "import unittest\nfrom src.app import add\n\nclass H(unittest.TestCase):\n    def test_negative(self):\n        self.assertEqual(add(-1, 1), 0)\n"
CONFIG = {"share": "disabled", "provider": {"local-llama": {"npm": "@ai-sdk/openai-compatible", "name": "Local",
          "options": {"baseURL": "{env:OC_LLAMA_BASEURL}"},
          "models": {"tiny-agent": {"name": "tiny", "supportsToolCalls": True, "limit": {"context": 65536, "output": 12288}}}}},
          "agent": {"build": {"model": "{env:OC_MODEL_BALANCED}"}}}


class OpencodeSandboxTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(shutil.which("bwrap"), "bwrap is required")
        self.assertTrue((OPENCODE_HOME / "bin/opencode").exists(), "the opencode binary is required")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_box_has_no_egress_and_no_real_home(self):
        home = self.root / "home"
        home.mkdir()
        code, out, err, _ = Box(home).run(["python3", "-c", (
            "import os, urllib.request\n"
            "print('ssh' if os.path.exists(os.path.expanduser('~/.ssh')) else 'no-ssh')\n"
            "try:\n urllib.request.urlopen('http://1.1.1.1', timeout=3); print('EGRESS')\n"
            "except Exception: print('no-egress')")], home, 60)
        self.assertEqual(code, 0, err)
        self.assertEqual(out.split(), ["no-ssh", "no-egress"])

    def test_real_opencode_fixes_a_bug_over_two_turns_and_is_graded(self):
        repo = self.root / "src-repo"
        repo.mkdir()
        git(["init", "-q", "-b", "main"], repo, isolated=True)
        base = commit(repo, {"src/__init__.py": "", "src/app.py": "def add(a, b):\n    return a - b\n",
                             "tests/__init__.py": "", "tests/test_app.py": APP_TEST}, "base")
        answer = commit(repo, {"src/app.py": "def add(a, b):\n    return a + b\n",
                               "tests/test_hidden.py": HIDDEN_TEST}, "fix")
        task = {"id": "impl-tiny", "kind": "workload", "category": "implementation", "prompt": "add() is broken; fix it.",
                "source": {"repo": str(repo), "commit": base, "answer": answer, "withhold": ["tests/test_hidden.py"]},
                "checks": [{"type": "tests", "command": ["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."]},
                           {"type": "diff_scope", "allow": ["src/**"], "min_changed": 1}],
                "followups": {"scripted": [{"after_turn": 1, "say": "Did the tests pass?"}]},
                "limits": {"turns": 3, "wall_seconds": 300}}
        build(task, self.root / "snap")
        config, ref, _ = oc.build(CONFIG, "tiny", ["tiny-agent"])
        script = [{"tool_calls": [{"name": "bash", "arguments": {
                      "command": "sed -i 's/a - b/a + b/' src/app.py && python3 -m unittest discover -s tests -t .",
                      "description": "fix and test"}}]},
                  {"text": "Fixed add()."},
                  {"text": "Yes, the tests pass."}]
        expected = {'seed': 17, 'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'max_tokens': 12288,
                    'min_p': 0, 'reasoning_effort': 'high',
                    'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True}}
        script[0]['reasoning'] = 'Inspect the subtraction and repair it.'
        with FakeLLM(script) as llm, InferenceCapture(llm.port, self.root/'inference', expected) as capture:
            result = run_task(task, self.root / "run", self.root / "snap",
                              lambda home, manifest: OpencodeDriver(home, config, ref, capture.port),
                              lambda home: Box(home))
        self.assertTrue(result["pass"], result)
        self.assertEqual([t["source"] for t in result["turns"]], ["prompt", "scripted"])
        self.assertEqual(result["metrics"]["tools"].get("bash"), 1)
        self.assertEqual(result["metrics"]["user_turns"], 2)
        agent = [r for r in llm.requests if r.get("tools")]
        self.assertGreater(len(agent[-1]["messages"]), len(agent[0]["messages"]))  # history grew
        self.assertIn("Yes, the tests pass.", result["answer"])
        self.assertTrue(json.loads((self.root/'inference/contract.json').read_text())['valid'])
        for request in agent:
            for key, value in expected.items():
                self.assertEqual(request[key], value)
        self.assertTrue(any(m.get('role') == 'tool' for r in agent for m in r['messages']))

    def test_prelude_turns_grow_the_session_before_the_task(self):
        repo = self.root / "src-repo"
        repo.mkdir()
        git(["init", "-q", "-b", "main"], repo, isolated=True)
        base = commit(repo, {"NOTES.md": "The answer is 42.\n"}, "notes")
        task = {"id": "depth-tiny", "kind": "workload", "category": "lookup", "depth": True,
                "prelude": ["Remember the codename TEAL-9. Say OK.", "Read NOTES.md."],
                "prompt": "What codename did I give you?", "source": {"repo": str(repo), "commit": base},
                "checks": [{"type": "answer", "all": ["TEAL-9"]}], "limits": {"turns": 1, "wall_seconds": 300}}
        build(task, self.root / "snap")
        config, ref, _ = oc.build(CONFIG, "tiny", ["tiny-agent"])
        script = [{"text": "OK."}, {"tool_calls": [{"name": "read", "arguments": {"filePath": "NOTES.md"}}]},
                  {"text": "It says 42."}, {"text": "You gave me TEAL-9."}]
        with FakeLLM(script) as llm:
            result = run_task(task, self.root / "run", self.root / "snap",
                              lambda home, manifest: OpencodeDriver(home, config, ref, llm.port), lambda home: Box(home))
        self.assertTrue(result["pass"], result)
        self.assertEqual([t["source"] for t in result["turns"]], ["prelude", "prelude", "prompt"])
        self.assertEqual(result["metrics"]["user_turns"], 3)
        last = [r for r in llm.requests if r.get("tools")][-1]
        self.assertIn("TEAL-9", str(last["messages"]))  # the prelude is in the history the task sees


if __name__ == "__main__":
    unittest.main()
