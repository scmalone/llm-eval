import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from llm_eval import judge
from llm_eval.workload.session import run_task
from llm_eval.workload.snapshot import build
from tests.test_workload import FAKE_TOKEN, make_repo, repo_task
from tests.test_workload_session import FakeBox, FakeDriver

ROOT = Path(__file__).resolve().parents[1]
TASK = {"id": "impl-1", "category": "implementation", "prompt": "Fix add."}
ENDPOINT = {"base_url": "https://judge.invalid/v1", "model": "pinned-judge", "key_env": "LLM_EVAL_TEST_JUDGE_KEY"}


class FakeJudge:
    """Answers like a judge; ``prefer`` picks the run whose answer contains that word."""
    def __init__(self, prefer=None, always=None):
        self.prefer, self.always, self.requests = prefer, always, []

    def __call__(self, request, timeout):
        body = json.loads(request.data)
        self.requests.append({"body": body, "headers": dict(request.headers)})
        user = body["messages"][1]["content"]
        if "## Run under review" in user:
            reply = {"scores": {c: 4 for c in judge.RUBRICS["implementation"]}, "reason": "fine"}
        else:
            a, b = user.split("## Run B")
            pick = self.always or ("A" if self.prefer in a else "B")
            reply = {"criteria": {c: pick for c in judge.RUBRICS["implementation"]}, "overall": pick, "reason": "x"}
        content = json.dumps({"choices": [{"message": {"content": "Verdict:\n" + json.dumps(reply)}}]})
        return io.BytesIO(content.encode())


def run(answer, diff=""):
    return {"task": "impl-1", "category": "implementation", "answer": answer, "diff": diff, "passed_checks": True}


class JudgeTests(unittest.TestCase):
    def ask(self, fake, fn, *args):
        with patch.object(judge.urllib.request, "urlopen", fake):
            return fn(ENDPOINT, *args)

    def test_a_consistent_preference_wins_and_position_bias_ties(self):
        fake = FakeJudge(prefer="good")
        result = self.ask(fake, judge.compare, TASK, run("good fix"), run("weak fix"))
        self.assertEqual(result["verdict"]["overall"], "first")
        self.assertEqual(len(fake.requests), 2)  # shown in both orders
        result = self.ask(FakeJudge(prefer="good"), judge.compare, TASK, run("weak fix"), run("good fix"))
        self.assertEqual(result["verdict"]["overall"], "second")
        biased = self.ask(FakeJudge(always="A"), judge.compare, TASK, run("good fix"), run("weak fix"))
        self.assertEqual(biased["verdict"]["overall"], "tie")

    def test_reference_scoring(self):
        result = self.ask(FakeJudge(), judge.score, TASK, run("fixed"), "diff --git a/src/app.py")
        self.assertEqual(set(result["scores"].values()), {4})

    def test_secrets_are_refused_before_any_request(self):
        fake = FakeJudge(prefer="x")
        with self.assertRaisesRegex(ValueError, "likely secrets"):
            self.ask(fake, judge.compare, TASK, run(f"token {FAKE_TOKEN}"), run("x"))
        self.assertEqual(fake.requests, [])

    def test_the_key_is_read_by_name_and_sent_only_as_a_header(self):
        fake = FakeJudge(prefer="good")
        with patch.dict(os.environ, {"LLM_EVAL_TEST_JUDGE_KEY": "k-123"}):
            self.ask(fake, judge.compare, TASK, run("good"), run("weak"))
        self.assertEqual(fake.requests[0]["headers"]["Authorization"], "Bearer k-123")
        self.assertNotIn("k-123", json.dumps([r["body"] for r in fake.requests]))

    def test_calibration_sheet_and_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            sheet = Path(tmp) / "calibration.md"
            key = judge.calibration_sheet([(TASK, run("good"), run("weak")), (TASK | {"id": "impl-2"}, run("a"), run("b"))], sheet)
            self.assertIn("Pair 2", sheet.read_text())
            self.assertTrue(sheet.with_suffix(".key.json").exists())
            first_pick = "A" if key["1"]["A"] == "first" else "B"
            result = judge.agreement({"1": first_pick, "2": "tie"}, {"impl-1": "first", "impl-2": "second"}, key)
            self.assertEqual(result, {"pairs": 2, "agree": 1, "rate": 0.5})

    def test_evidence_reconstructs_the_diff_a_run_made(self):
        class LocalBox:
            def __init__(self, home):
                self.home = home

            def git(self, args, cwd, timeout=120):
                result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
                return result.returncode, result.stdout

            def run(self, command, cwd, timeout):
                result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
                return result.returncode, result.stdout, result.stderr, False

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, base, answer = make_repo(root, with_data=False)
            task = repo_task(repo, base, answer) | {"checks": [{"type": "answer", "all": ["Fixed"]}]}
            build(task, root / "snap")
            run_task(task, root / "out", root / "snap",
                     lambda home, manifest: FakeDriver(home, [({"src/app.py": "def add(a, b):\n    return a + b\n",
                                                                "NOTES.md": "new file\n"}, "Fixed.")]),
                     lambda home: FakeBox())
            found = judge.evidence(root / "out", LocalBox)
        self.assertIn("+    return a + b", found["diff"])
        self.assertIn("new file NOTES.md", found["diff"])
        self.assertEqual(found["answer"], "Fixed.")

    def test_the_runner_never_imports_the_judge(self):
        import re
        for module in ("runner.py", "acceptance.py", "live.py", "workload/session.py", "workload/driver.py"):
            text = (ROOT / "llm_eval" / module).read_text()
            self.assertIsNone(re.search(r"^\s*(from|import)\s+[.\w]*judge", text, re.M), module)


if __name__ == "__main__":
    unittest.main()
