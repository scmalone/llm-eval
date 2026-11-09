import json
from pathlib import Path
import tempfile
import unittest

from llm_eval.workload import opencode_config as oc
from llm_eval.workload.metrics import summarize
from llm_eval.workload.sandbox import Box
from llm_eval.workload.session import FollowUps, run_task
from llm_eval.workload.snapshot import build
from tests.test_workload import make_repo, repo_task

PRODUCTION = """{
  // comment with a URL inside a string below
  "model": "{env:OC_MODEL_BALANCED}", "share": "disabled", "autoupdate": "notify",
  "mcp": {"vault": {"command": ["{env:OC_OBSIDIAN_MCP}"]}},
  "provider": {
    "deepinfra": {"options": {"baseURL": "https://api.deepinfra.com/v1/openai"}, "models": {}},
    /* block comment */
    "local-llama": {"options": {"baseURL": "{env:OC_LLAMA_BASEURL}"}, "models": {
      "muse-q4": {"limit": {"context": 131072, "output": 32768}},
      "muse-q4-agent": {"limit": {"context": 65536, "output": 12288},},
    }},
  },
  "agent": {"build": {"model": "{env:OC_MODEL_BALANCED}", "top_p": 0.95},
            "flash": {"model": "deepinfra/x"}, "plain": {"description": "no model"}},
}"""


class OpencodeConfigTests(unittest.TestCase):
    def test_jsonc_comments_and_trailing_commas(self):
        parsed = oc.parse_jsonc(PRODUCTION)
        self.assertEqual(parsed["provider"]["deepinfra"]["options"]["baseURL"], "https://api.deepinfra.com/v1/openai")

    def test_the_bare_entry_is_the_default_and_the_agent_entry_needs_an_explicit_name(self):
        production = oc.parse_jsonc(PRODUCTION)
        _, ref, limit = oc.build(production, "muse-q4", ["muse-q4-agent"])
        self.assertEqual((ref, limit), ("local-llama/muse-q4", {"context": 131072, "output": 32768}))
        _, ref, limit = oc.build(production, "muse-q4", ["muse-q4-agent"], entry="muse-q4-agent")
        self.assertEqual((ref, limit), ("local-llama/muse-q4-agent", {"context": 65536, "output": 12288}))

    def test_context_and_output_overrides_touch_only_the_sandbox_copy(self):
        production = oc.parse_jsonc(PRODUCTION)
        _, _, limit = oc.build(production, "muse-q4", [], None, 114688, 16384)
        self.assertEqual(limit, {"context": 114688, "output": 16384})
        self.assertEqual(production["provider"]["local-llama"]["models"]["muse-q4"]["limit"], {"context": 131072, "output": 32768})

    def test_build_keeps_one_loopback_provider_and_the_agent_entry(self):
        config, ref, limit = oc.build(oc.parse_jsonc(PRODUCTION), "muse-q4", ["muse-q4-agent"], entry="muse-q4-agent")
        self.assertEqual(ref, "local-llama/muse-q4-agent")
        self.assertEqual(limit, {"context": 65536, "output": 12288})
        self.assertEqual(list(config["provider"]), ["local-llama"])
        self.assertEqual(config["provider"]["local-llama"]["options"]["baseURL"], oc.SANDBOX_BASE_URL)
        self.assertNotIn("mcp", config)
        self.assertEqual({a.get("model") for a in config["agent"].values()}, {ref, None})
        self.assertEqual(config["agent"]["build"]["top_p"], 0.95)
        self.assertFalse(config["autoupdate"])

    def test_check_refuses_anything_but_loopback(self):
        config, _, _ = oc.build(oc.parse_jsonc(PRODUCTION), "muse-q4")
        for change in ({"provider": {**config["provider"], "deepinfra": {}}},
                       {"provider": {"local-llama": {"options": {"baseURL": "http://10.0.0.5:8080/v1"}}}},
                       {"provider": {"local-llama": {"options": {"baseURL": "http://localhost:8080/v1"}}}},
                       {"mcp": {"x": {}}}, {"share": "auto"},
                       {"agent": {"build": {"model": "deepinfra/x"}}}):
            with self.assertRaises(ValueError):
                oc.check({**config, **change})
        with self.assertRaises(ValueError):
            oc.build(oc.parse_jsonc(PRODUCTION), "unknown")


class BoxTests(unittest.TestCase):
    def test_box_is_isolated_and_relays_are_the_only_exit(self):
        box = Box("/tmp/fake-home", real_home="/home/u", read_only=[("/opt/x", "/home/u/.x")],
                  env={"A": "1"}, relays=["127.0.0.1:8080=llm.sock"], sockets="/run/user/1/lle-1")
        argv = box.argv(["opencode", "run"], "/home/u/Workspace/r")
        for flag in ("--die-with-parent", "--unshare-all", "--clearenv", "--new-session"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("/tmp/fake-home") - 1], "--bind")
        self.assertEqual(argv[argv.index("/tmp/fake-home") + 1], "/home/u")
        self.assertIn("127.0.0.1:8080=/run/relay/llm.sock", argv)
        self.assertEqual(argv[-3:], ["--", "opencode", "run"])
        self.assertNotIn("--share-net", argv)


def export_with(answer, tools=(), compaction=False):
    parts = [{"type": "step-finish", "tokens": {"input": 900, "output": 40, "reasoning": 10, "cache": {"read": 100}}}]
    parts += [{"type": "tool", "tool": t, "state": {"status": "completed", "input": {"command": f"run {t}"},
                                                     "time": {"start": 1000, "end": 3000}}} for t in tools]
    parts.append({"type": "text", "text": answer})
    if compaction:
        parts.append({"type": "compaction"})
    return {"messages": [{"info": {"role": "user", "time": {"created": 0}}, "parts": []},
                         {"info": {"role": "assistant", "time": {"created": 0, "completed": 10000}}, "parts": parts}]}


class MetricsTests(unittest.TestCase):
    def test_summary_separates_model_and_tool_time(self):
        metrics = summarize(export_with("Done?", tools=("bash", "read", "invalid"), compaction=True))
        self.assertEqual(metrics["tool_calls"], 3)
        self.assertEqual(metrics["invalid_calls"], 1)
        self.assertEqual(metrics["commands"], ["run bash"])
        self.assertEqual(metrics["peak_context_tokens"], 1000)
        self.assertEqual(metrics["compactions"], 1)
        self.assertEqual(metrics["model_seconds"], 4.0)
        self.assertEqual(metrics["tool_seconds"], 6.0)
        self.assertTrue(metrics["ended_with_question"])


class FollowUpTests(unittest.TestCase):
    def test_order_caps_and_default_reply(self):
        task = {"followups": {"scripted": [{"after_turn": 1, "say": "Also handle zero."}],
                              "tests": {"command": ["t"], "max": 1},
                              "faq": [{"match": "which file", "say": "src/app.py"}]}}
        results = iter([(1, "FAILED x"), (1, "FAILED again"), (0, "ok")])
        followups = FollowUps(task, lambda *a: next(results))
        self.assertEqual(followups.next(1, "done"), ("Also handle zero.", "scripted"))
        said, source = followups.next(2, "done")
        self.assertEqual(source, "tests")
        self.assertIn("FAILED x", said)
        self.assertEqual(followups.next(3, "Which file should I edit?"), ("src/app.py", "faq"))
        followups = FollowUps({"followups": {}}, None)
        self.assertEqual(followups.next(1, "Shall I proceed?")[1], "default")
        self.assertEqual(followups.next(2, "Shall I proceed?"), (None, None))
        self.assertEqual(followups.clarifications, 2)

    def test_a_hang_inside_a_model_command_is_the_models_timeout(self):
        """north-mini started a fake server that sleeps forever; the stall was scored as a server failure."""
        from llm_eval.workload.session import UNGRADED, _stop_for
        self.assertEqual(_stop_for({"hung": True, "tool_open": True}, "wall_cap"), "tool_hang")
        self.assertEqual(_stop_for({"hung": True}, "wall_cap"), "stalled")
        self.assertIn("tool_hang", UNGRADED)

    def test_question_tasks_never_reply_with_a_go_ahead(self):
        """Review 2.4: every question-only core task answers a question with the no-change reply."""
        from llm_eval.workload.prepare import load_tasks
        from llm_eval.workload.session import QUESTION_REPLY
        core = [t for t in load_tasks() if t.get("category") in ("advisory", "lookup", "resume", "review")
                and not t["id"].startswith(("sweep-", "depth-"))]
        self.assertEqual(len(core), 20)
        for task in core:
            self.assertEqual(FollowUps(task, None).next(1, "Shall I delete the rest?"), (QUESTION_REPLY, "default"),
                             task["id"])


class FakeDriver:
    """Plays a scripted agent: each turn applies file edits to the repo and returns text."""
    def __init__(self, home, turns):
        self.turns, self.said = list(turns), []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def turn(self, text, workdir, timeout):
        self.said.append(text)
        edits, answer = self.turns.pop(0) if self.turns else ({}, "nothing more")
        for name, content in edits.items():
            (Path(workdir) / name).write_text(content)
        self.last = answer
        return {"answer": answer, "timed_out": timeout <= 0, "returncode": 0, "events": 3}

    def export(self):
        return export_with(self.last, tools=("edit",))

    def summarize(self, export=None):
        from llm_eval.workload.metrics import summarize
        return summarize(self.export() if export is None else export)


class FakeBox:
    """Test commands pass when src/app.py adds."""
    def runner(self):
        def run(command, cwd, timeout):
            text = (Path(cwd) / "src/app.py").read_text()
            return (0, "Ran 2 tests in 0.01s\n\nOK\n") if "a + b" in text else (1, "AssertionError: add(2, 2) != 4")
        return run


class RunTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        repo, base, answer = make_repo(self.root, with_data=False)
        self.task = repo_task(repo, base, answer)
        self.task.update(checks=[{"type": "tests", "command": ["python3", "-m", "unittest"]},
                                 {"type": "diff_scope", "allow": ["src/**"], "min_changed": 1}],
                         followups={"tests": {"command": ["python3", "-m", "unittest"], "max": 2}})
        build(self.task, self.root / "snap")

    def tearDown(self):
        self.tmp.cleanup()

    def run_with(self, turns, name="run", **task_changes):
        drivers = []
        def factory(home, manifest):
            drivers.append(FakeDriver(home, turns))
            return drivers[-1]
        result = run_task({**self.task, **task_changes}, self.root / name, self.root / "snap",
                          factory, lambda home: FakeBox(), home_files={"Workspace/PROJECTS.md": "| r |\n"})
        return result, drivers[0]

    def test_failing_tests_are_pasted_back_and_a_fix_passes(self):
        result, driver = self.run_with([({"src/app.py": "def add(a, b):\n    return a * b\n"}, "Fixed."),
                                        ({"src/app.py": "def add(a, b):\n    return a + b\n"}, "Fixed for real.")])
        self.assertTrue(result["pass"], result["grading"])
        self.assertEqual(result["tests_pasted"], 1)
        self.assertIn("AssertionError", driver.said[1])
        self.assertTrue((self.root / "run/home/Workspace/PROJECTS.md").exists())
        self.assertTrue((self.root / "run/result.json").exists())

    def test_a_noop_run_fails_and_turn_cap_stops_it(self):
        result, _ = self.run_with([({}, "Looks fine to me.")] * 5, limits={"turns": 2, "wall_seconds": 900})
        self.assertFalse(result["pass"])
        self.assertEqual(result["stop"], "turn_cap")
        self.assertEqual(len(result["turns"]), 2)

    def test_wall_cap_fails_without_grading(self):
        ticks = iter([0, 0, 2000, 2000, 2000])
        drivers = []
        def factory(home, manifest):
            drivers.append(FakeDriver(home, [({}, "working")] * 3))
            return drivers[-1]
        result = run_task({**self.task, "limits": {"turns": 5, "wall_seconds": 900}}, self.root / "wall",
                          self.root / "snap", factory, lambda home: FakeBox(), clock=lambda: next(ticks))
        self.assertFalse(result["pass"])
        self.assertEqual(result["stop"], "wall_cap")

    def test_a_budget_shorter_than_the_wall_cap_ends_the_task_as_a_budget_cap_without_grading(self):
        ticks = iter([0, 0, 200, 200, 200])
        result = run_task({**self.task, "limits": {"turns": 5, "wall_seconds": 900}}, self.root / "budget",
                          self.root / "snap", lambda home, manifest: FakeDriver(home, [({}, "working")] * 3),
                          lambda home: FakeBox(), clock=lambda: next(ticks), budget=100)
        self.assertEqual(result["stop"], "budget_cap")
        self.assertFalse(result["pass"])
        self.assertIn("budget", result["grading"]["checks"][0]["reason"])

    def test_a_budget_longer_than_the_wall_cap_changes_nothing(self):
        ticks = iter([0, 0, 2000, 2000, 2000])
        result = run_task({**self.task, "limits": {"turns": 5, "wall_seconds": 900}}, self.root / "wall2",
                          self.root / "snap", lambda home, manifest: FakeDriver(home, [({}, "working")] * 3),
                          lambda home: FakeBox(), clock=lambda: next(ticks), budget=5000)
        self.assertEqual(result["stop"], "wall_cap")

    def test_a_hung_turn_stops_the_task_as_stalled_and_keeps_the_silence_reading(self):
        class Hung(FakeDriver):
            def turn(self, text, workdir, timeout):
                self.last = ""
                return {"answer": "", "timed_out": False, "returncode": -9, "events": 0, "hung": True,
                        "max_silence_seconds": 601.0}
        result = run_task(self.task, self.root / "hung", self.root / "snap", lambda home, manifest: Hung(home, []),
                          lambda home: FakeBox())
        self.assertEqual(result["stop"], "stalled")
        self.assertFalse(result["pass"])
        self.assertEqual(result["turns"][0]["max_silence_seconds"], 601.0)
        self.assertTrue(result["turns"][0]["hung"])


if __name__ == "__main__":
    unittest.main()
