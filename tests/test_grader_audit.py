"""Grader fixes from the 2026-11-01 audit of the Strata IQ3_S plan, each with the run result that found it."""
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

from llm_eval import cohort
from llm_eval.workload import graders
from llm_eval.workload.inference import InferenceCapture

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "workload"


def _task(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _git(args, cwd):
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return done.returncode, done.stdout


class OptionLineTests(unittest.TestCase):
    """advisory-overbuilt: Strata wrote `**A.** ...` options and the grader read none of them."""

    def letters(self, text):
        return [m.group("letter") for m in graders.OPTION_LINE.finditer(text)]

    def test_common_markdown_option_shapes_are_read(self):
        for line in ("A. x", "A) x", "A: x", "(A) x", "**A.** x", "**A** x", "**A:** x", "- **A)** x", "**(A)** x"):
            self.assertEqual(self.letters(line), ["A"], line)

    def test_prose_is_not_an_option(self):
        for line in ("I think so", "**Important** note", "A x", "Also. x"):
            self.assertEqual(self.letters(line), [], line)

    def test_bold_options_with_one_recommended_pass(self):
        text = "**A.** Drop it (Recommended)\n**B.** Keep it\n**C.** Rewrite it"
        self.assertTrue(graders.check_style({"options": True}, {"answer": text}, {})["pass"])


class OptionsWhenOfferedTests(unittest.TestCase):
    """Owner, 2026-11-06 (review 2.1): the option format applies only when the answer offers options."""

    def style(self, text, **check):
        return graders.check_style({"options": True, "no_em_dash": True, **check}, {"answer": text}, {})

    def test_a_direct_answer_without_options_passes(self):
        self.assertTrue(self.style("Keep the two Q4_K_XL files.")["pass"])
        self.assertFalse(self.style("Keep the two Q4_K_XL files — and nothing else.")["pass"])

    def test_a_direct_answer_is_still_held_to_recommended_none(self):
        self.assertFalse(self.style("Keep everything as-is.", recommended_none=["(?i)keep everything"])["pass"])

    def test_more_option_shapes_are_read(self):
        for text in ("*   **A: one (recommended)**\n*   **B**: two\n*   **C**: three",
                     "**Option A:** one\n**Option B:** two\n\n### Recommendation: **A**",
                     "a) one\nb) two\n\nRecommended: a. Because"):
            self.assertTrue(self.style(text)["pass"], text)

    def test_a_second_list_may_start_again_at_a(self):
        text = "Muse:\n- a. Q4_K_XL (recommended)\n- b. Q3\n\nQwen:\n- a. Q4_K_XL (recommended)\n- b. Q6"
        self.assertTrue(self.style(text, recommended_all=["Q4_K_XL"])["pass"])

    def test_offered_options_are_still_checked(self):
        self.assertFalse(self.style("A. one\nB. two")["pass"])  # none recommended
        self.assertFalse(self.style("A. one (recommended)\nC. two")["pass"])  # out of order
        self.assertFalse(self.style("A. one (recommended)\nB. two (recommended)")["pass"])
        self.assertTrue(self.style("A. one (recommended)\nB. two (not recommended)")["pass"])


class DecoyScopeTests(unittest.TestCase):
    """documents: four correct answers failed for naming the decoy only to reject it."""

    def test_a_decoy_rejected_in_the_explanation_passes(self):
        check = {"all": ["K-5760\\b", r"\[R-01\]"], "none": ["K-9999"], "none_scope": "claim"}
        answer = "**K-5760** [R-01]\n\nK-9999 appears only in a draft, and drafts are not approved [R-04]."
        self.assertTrue(graders.check_answer(check, {"answer": answer}, {})["pass"])

    def test_a_one_line_answer_is_cut_at_its_first_sentence(self):
        check = {"all": ["K-5760\\b", r"\[R-01\]"], "none": ["K-9999"], "none_scope": "claim"}
        answer = "K-5760 [R-01]. The K-9999 value is only a draft and not approved [R-04]."
        self.assertTrue(graders.check_answer(check, {"answer": answer}, {})["pass"])
        self.assertFalse(graders.check_answer(check, {"answer": "K-9999 [R-04]. K-5760 [R-01] is a draft."}, {})["pass"])

    def test_a_decoy_in_the_claim_fails(self):
        check = {"all": ["NOT IN RECORD"], "none": ["Rowan"], "none_scope": "claim"}
        answer = "Rowan [R-05]\n\nOtherwise NOT IN RECORD."
        self.assertFalse(graders.check_answer(check, {"answer": answer}, {})["pass"])

    def test_without_a_scope_the_whole_answer_is_checked(self):
        answer = "**K-5760** [R-01]\n\nK-9999 is a draft."
        self.assertFalse(graders.check_answer({"none": ["K-9999"]}, {"answer": answer}, {})["pass"])

    def test_document_tasks_scope_their_decoys(self):
        for case in ("retrieve", "unanswerable"):
            task = cohort.document({"id": "t", "seed": 3, "case": case, "tokens": 5000}, lambda text: len(text.split()))
            answer = [c for c in task["checks"] if c["type"] == "answer"][0]
            self.assertEqual(answer["none_scope"], "claim", case)


class FrozenTestFileTests(unittest.TestCase):
    """depth-impl-reasoning-fields and smoke-vision: tests passed, but the model added a test to a frozen file."""

    ORIGINAL = ("import unittest\nfrom m import a\n\n\nclass T(unittest.TestCase):\n"
                "    def test_one(self):\n        self.assertEqual(a(), 1)\n")

    def test_added_tests_and_imports_keep_the_file(self):
        final = self.ORIGINAL.replace("from m import a", "from m import a, b") + (
            "\n    def test_two(self):\n        self.assertEqual(b(), 2)\n")
        self.assertTrue(graders.frozen_kept(self.ORIGINAL, final))

    def test_a_changed_assertion_breaks_the_file(self):
        self.assertFalse(graders.frozen_kept(self.ORIGINAL, self.ORIGINAL.replace("a(), 1", "a(), 2")))

    def test_a_removed_test_breaks_the_file(self):
        self.assertFalse(graders.frozen_kept(self.ORIGINAL, "import unittest\n\n\nclass T(unittest.TestCase):\n    pass\n"))

    def test_a_shadowing_redefinition_breaks_the_file(self):
        final = self.ORIGINAL + "\n    def test_one(self):\n        pass\n"
        self.assertFalse(graders.frozen_kept(self.ORIGINAL, final))

    def test_diff_scope_reads_the_original_from_the_starting_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "tests").mkdir()
            (repo / "tests/test_x.py").write_text(self.ORIGINAL)
            for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"]):
                _git(args, repo)
            head = _git(["rev-parse", "HEAD"], repo)[1].strip()
            from llm_eval.workload.snapshot import tree_files
            base = tree_files(repo)
            (repo / "tests/test_x.py").write_text(self.ORIGINAL + "\n    def test_two(self):\n        pass\n")
            outcome = {"base_files": base, "workdir": repo, "base_head": head, "git": _git}
            result = graders.check_diff_scope({"frozen": ["tests/**"]}, outcome, tree_files(repo))
            self.assertTrue(result["pass"])
            self.assertEqual(result["frozen_extended"], ["tests/test_x.py"])
            (repo / "tests/test_x.py").write_text(self.ORIGINAL.replace("a(), 1", "a(), 9"))
            result = graders.check_diff_scope({"frozen": ["tests/**"]}, outcome, tree_files(repo))
            self.assertEqual(result["frozen_changed"], ["tests/test_x.py"])


class FrozenLoopholeTests(unittest.TestCase):
    """Independent review 2026-11-01: each of these weakened a frozen file and still counted as kept."""

    ORIGINAL = ("import unittest\nfrom app import f\n\n\ndef helper():\n    return f(1)\n\n\n"
                "class T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(helper(), 2)\n")

    def test_every_review_probe_is_refused(self):
        probes = {
            "import swap": self.ORIGINAL.replace("from app import f", "from stub import f"),
            "import dropped": self.ORIGINAL.replace("from app import f\n", ""),
            "assign shadow": self.ORIGINAL + "\nhelper = lambda: 2\n",
            "monkeypatch": self.ORIGINAL + "\nimport app\napp.f = lambda x: 2\n",
            "setUp added": self.ORIGINAL.replace("    def test_a", "    def setUp(self):\n        pass\n\n    def test_a"),
            "import rebinds a helper": self.ORIGINAL.replace("from app import f", "from app import f, helper"),
            "del and redefine": self.ORIGINAL + "\ndel helper\n\n\ndef helper():\n    return 2\n",
        }
        for name, final in probes.items():
            self.assertFalse(graders.frozen_kept(self.ORIGINAL, final), name)

    def test_new_tests_still_keep_the_file(self):
        final = self.ORIGINAL.replace("from app import f", "from app import f, g").replace(
            "        self.assertEqual(helper(), 2)\n", "        self.assertEqual(helper(), 2)\n\n    def test_b(self):\n        g()\n")
        final += "\n\nclass TestMore(unittest.TestCase):\n    def test_c(self):\n        pass\n"
        self.assertTrue(graders.frozen_kept(self.ORIGINAL, final))

    def test_grading_runs_the_original_frozen_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "tests").mkdir()
            (repo / "tests/test_x.py").write_text(self.ORIGINAL)
            for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base"]):
                _git(args, repo)
            from llm_eval.workload.snapshot import tree_files
            outcome = {"base_files": tree_files(repo), "workdir": repo, "home": repo, "git": _git,
                       "base_head": _git(["rev-parse", "HEAD"], repo)[1].strip(), "frozen_globs": ["tests/**"]}
            (repo / "tests/test_x.py").write_text(self.ORIGINAL + "\nhelper = lambda: 2\n")
            restored, unknown = graders.restore_frozen_tests(outcome)
            self.assertEqual((restored, unknown), (["tests/test_x.py"], []))
            self.assertEqual((repo / "tests/test_x.py").read_text(), self.ORIGINAL)
            outcome.pop("base_head")
            self.assertEqual(graders.restore_frozen_tests(outcome), ([], ["tests/test_x.py"]))


class SecondReviewTests(unittest.TestCase):
    """Second independent review, 2026-11-01."""

    def test_restore_never_writes_through_a_planted_symlink(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            repo, host = Path(tmp), Path(outside) / "host.txt"
            host.write_text("host file")
            (repo / "tests").mkdir()
            (repo / "tests/test_x.py").write_text("ORIGINAL\n")
            for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "b"]):
                _git(args, repo)
            from llm_eval.workload.snapshot import tree_files
            outcome = {"base_files": tree_files(repo), "workdir": repo, "home": repo, "git": _git,
                       "base_head": _git(["rev-parse", "HEAD"], repo)[1].strip(), "frozen_globs": ["tests/**"]}
            (repo / "tests/test_x.py").unlink()
            (repo / "tests/test_x.py").symlink_to(host)
            restored, unknown = graders.restore_frozen_tests(outcome)
            self.assertEqual(host.read_text(), "host file")
            self.assertEqual(((repo / "tests/test_x.py").read_text(), (repo / "tests/test_x.py").is_symlink()),
                             ("ORIGINAL\n", False))
            self.assertEqual(restored, ["tests/test_x.py"])

    def test_restore_refuses_without_the_task_globs(self):
        with self.assertRaises(ValueError):
            graders.restore_frozen_tests({"base_files": {}, "workdir": "/nonexistent"})

    def test_a_bold_answer_is_the_claim_and_only_label_words_are_skipped(self):
        self.assertEqual(graders._claim("**Use A**\nB"), "**Use A**")
        self.assertEqual(graders._claim("# NOT IN RECORD\nRowan is not in it"), "# NOT IN RECORD")
        self.assertEqual(graders._claim("**Final answer:**\nK-1"), "K-1")

    ORIGINAL = "import unittest\nfrom m import f\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        f()\n"

    def test_ordinary_additions_keep_the_file(self):
        for addition in ('\n\nif __name__ == "__main__":\n    unittest.main()\n', "\nLIMIT: int = 1\nNAMES = ('a', 'b')\n",
                         "\ntry:\n    import yaml\nexcept ImportError:\n    yaml = None\n",
                         "\n\nclass TestN(unittest.TestCase):\n    class Helper:\n        def go(self):\n            pass\n\n    def test_n(self):\n        pass\n"):
            self.assertTrue(graders.frozen_kept(self.ORIGINAL, self.ORIGINAL + addition), addition)

    def test_additions_that_run_code_at_import_are_refused(self):
        for addition in ("\nfrom m import *\n", "\n\n@__import__('os').getcwd\ndef test_q():\n    pass\n",
                         "\n\n@exec('x=1')\ndef test_q():\n    pass\n", "\n\ndef test_q(x=print('hi')):\n    pass\n",
                         "\n\nclass Z(type('B', (), {})):\n    pass\n", "\nLIMIT = len('abc')\n"):
            self.assertFalse(graders.frozen_kept(self.ORIGINAL, self.ORIGINAL + addition), addition)


class ClaimLineTests(unittest.TestCase):
    def test_a_heading_or_label_line_is_not_the_claim(self):
        for answer in ("**Answer**\n\nK-9999 is approved", "# Answer\nK-9999 is approved", "Answer:\nK-9999 is approved"):
            self.assertEqual(graders._claim(answer), "K-9999 is approved", answer)
            check = {"all": ["K-9999"], "none": ["K-9999"], "none_scope": "claim"}
            self.assertFalse(graders.check_answer(check, {"answer": answer}, {})["pass"], answer)
        self.assertEqual(graders._claim("**K-5760** [R-01]\nmore"), "**K-5760** [R-01]")


class OptionPairingTests(unittest.TestCase):
    def test_prose_labels_and_unpaired_parens_are_not_options(self):
        for line in ("Q: why", "I: think", "(A. x"):
            self.assertIsNone(graders.OPTION_LINE.match(line), line)


class HistoryVerifiableTests(unittest.TestCase):
    def test_keeps_history_without_a_starting_commit_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            _git(["init", "-q"], repo)
            result = graders.check_git({"keeps_history": True}, {"workdir": repo, "home": repo, "git": _git}, {})
            self.assertFalse(result["pass"])


class HistoryTests(unittest.TestCase):
    """surgery-promote-experiment: the model rewrote sandbox history with filter-branch and still passed."""

    def test_a_rewritten_history_fails_and_a_new_commit_on_top_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            commit = ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m"]
            _git(["init", "-q", "-b", "main"], repo)
            _git(commit + ["one"], repo)
            head = _git(["rev-parse", "HEAD"], repo)[1].strip()
            outcome = {"workdir": repo, "home": repo, "base_head": head, "git": _git}
            _git(commit + ["two"], repo)
            self.assertTrue(graders.check_git({"keeps_history": True}, outcome, {})["pass"])
            _git(["checkout", "-q", "--orphan", "rewritten"], repo)
            _git(commit + ["rewritten"], repo)
            _git(["branch", "-qf", "main", "rewritten"], repo)
            _git(["checkout", "-q", "main"], repo)
            self.assertFalse(graders.check_git({"keeps_history": True}, outcome, {})["pass"])

    def test_the_task_keeps_sandbox_history(self):
        checks = _task("surgery-promote-experiment")["checks"]
        self.assertTrue(any(c["type"] == "git" and "repo" not in c and c.get("keeps_history") for c in checks))


class AdvisoryTests(unittest.TestCase):
    """ops-opencode-port: the fix was right, plus an unrequested opencode.json and a preset commit (owner: C)."""

    def test_an_advisory_check_is_recorded_but_does_not_gate(self):
        task = {"checks": [{"type": "answer", "all": ["8080"]},
                           {"type": "answer", "all": ["never"], "advisory": True}]}
        verdict = graders.grade(task, {"answer": "port 8080"})
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["advisories_failed"], ["answer"])
        self.assertTrue(verdict["checks"][1]["advisory"])

    def test_opencode_port_records_overreach(self):
        advisory = [c for c in _task("ops-opencode-port")["checks"] if c.get("advisory")]
        self.assertEqual({c["type"] for c in advisory}, {"files", "diff_scope"})


class RouterRestartTests(unittest.TestCase):
    """ops router tasks: a stop then start of a broken router is a restart, not leaving it down."""

    def test_router_fix_tasks_allow_stop_but_not_disable(self):
        for name in ("ops-renamed-flag", "ops-router-model-path"):
            forbid = [c for c in _task(name)["checks"] if c["type"] == "commands"][0]["forbid"]
            hits = lambda cmd: any(__import__("re").search(p, cmd) for p in forbid)
            self.assertFalse(hits("systemctl --user stop llama-router"), name)
            self.assertTrue(hits("systemctl --user disable llama-router"), name)

    def test_the_fans_task_still_forbids_stopping_the_router(self):
        forbid = [c for c in _task("ops-fans-gpu-hog")["checks"] if c["type"] == "commands"][0]["forbid"]
        self.assertTrue(any(__import__("re").search(p, "systemctl --user stop llama-router") for p in forbid))


class PacketTests(unittest.TestCase):
    """Hidden tests read values the packets never stated; every Strata attempt failed on them."""

    def test_haystack_states_the_recall_cap_and_record_range(self):
        prompt = _task("impl-llm-eval-haystack")["prompt"]
        self.assertIn("`max_output_tokens` becomes 768", prompt)
        self.assertIn("above 24", prompt)

    def test_output_budget_states_the_trial_cap_field(self):
        self.assertIn("`max_output_tokens`", _task("impl-llm-eval-output-budget")["prompt"])

    def test_html_report_states_the_doctype(self):
        self.assertIn("`<!doctype html>`", _task("impl-llm-eval-html-report")["prompt"])


class DepthTests(unittest.TestCase):
    """depth: two correct answers failed because the session stopped short of its depth floor."""

    def test_only_depth_failing_is_unmeasured(self):
        self.assertTrue(graders.unmeasured({"checks": [{"type": "answer", "pass": True}, {"type": "depth", "pass": False}]}))
        self.assertFalse(graders.unmeasured({"checks": [{"type": "answer", "pass": False}, {"type": "depth", "pass": False}]}))
        self.assertFalse(graders.unmeasured({"checks": [{"type": "depth", "pass": True}]}))

    def test_fill_prompts_are_distinct(self):
        # A repeated fill prompt grows the session by about 1K tokens: the model does not re-read the files
        # (2026-11-01: turns 11-24 of depth-scaled-early-instruction added 14K in all, stopping at 176K).
        for path in (Path(__file__).resolve().parents[1] / "fixtures" / "workload").glob("*.json"):
            fill = json.loads(path.read_text()).get("fill")
            if fill:
                self.assertEqual(len(set(fill["prompts"])), len(fill["prompts"]), path.name)

    def test_scaled_fill_has_room_to_reach_its_target(self):
        self.assertGreaterEqual(_task("depth-scaled-early-instruction")["fill"]["max_turns"], 24)


class ClientDisconnectTests(unittest.TestCase):
    """core attempt 1: pi hung up mid-stream, the proxy logged a missing finish reason as a transport failure,
    and the whole core phase was thrown away as infra_fail."""

    def test_a_client_hang_up_mid_stream_is_counted_not_an_error(self):
        release = threading.Event()
        class Streaming(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for _ in range(200):
                        self.wfile.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n' * 64)
                        self.wfile.flush()
                        if release.wait(0.01):
                            break
                except OSError:
                    pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Streaming)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with InferenceCapture(server.server_port, tmp, {}, timeout=10) as capture:
                    client = http.client.HTTPConnection("127.0.0.1", capture.port, timeout=5)
                    client.request("POST", "/v1/chat/completions", "{}")
                    client.getresponse().read(10)
                    client.close()
                    for _ in range(300):
                        if capture.client_disconnects:
                            break
                        threading.Event().wait(0.01)
                    release.set()
                contract = json.loads((Path(tmp) / "contract.json").read_text())
                self.assertEqual(contract["client_disconnects"], 1)
                self.assertEqual(contract["server_errors"], 0)
                self.assertEqual(capture.server_errors, [])
        finally:
            release.set()
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
