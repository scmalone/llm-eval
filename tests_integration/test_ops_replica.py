"""Real opencode fixes a synthetic broken machine through the shims, inside the sandbox."""
from pathlib import Path
import tempfile
import unittest

from llm_eval.workload import opencode_config as oc, replicas
from llm_eval.workload.driver import OpencodeDriver
from llm_eval.workload.fakellm import FakeLLM
from llm_eval.workload.sandbox import Box
from llm_eval.workload.session import run_task
from test_opencode_sandbox import CONFIG


def bash(command):
    return {"tool_calls": [{"name": "bash", "arguments": {"command": command, "description": "run"}}]}


class OpsReplicaTests(unittest.TestCase):
    def test_box_exposes_only_the_inside_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            code, out, err, _ = Box(home).run(["find", "/opt/llm-eval", "-type", "f"], home, 30)
            self.assertEqual(code, 0, err)
            self.assertEqual(sorted(Path(p).name for p in out.split()), ["__init__.py", "__init__.py", "machine.py", "relay.py"])

    def test_real_opencode_diagnoses_and_fixes_the_router(self):
        task = {"id": "ops-router-1", "kind": "workload", "category": "ops",
                "prompt": "opencode can't reach my local models. Find out why and fix it.",
                "source": {"replica": "router-bad-model-path", "seed": 3}, "limits": {"turns": 2, "wall_seconds": 300},
                "checks": [{"type": "state", "file": "~/" + replicas.MACHINE,
                            "equals": {"units/llama-router.service/active": True}},
                           {"type": "commands", "forbid": [r"(^|\s)sudo\s"]}]}
        facts = replicas.facts(task)
        task["checks"].append({"type": "answer", "all": [facts["wrong"].replace(".", r"\.")]})
        preset = "~/.config/llama-cpp/models-preset.ini"
        script = [bash("systemctl --user status llama-router; journalctl --user -u llama-router -n 20"),
                  bash(f"sed -i 's/{facts['wrong']}/{facts['actual']}/' {preset} && "
                       "systemctl --user restart llama-router && systemctl --user is-active llama-router"),
                  {"text": f"The preset pointed at {facts['wrong']}, which is not on disk; fixed and restarted."}]
        with tempfile.TemporaryDirectory() as tmp:
            replicas.build(task, Path(tmp) / "snap")
            config, ref, _ = oc.build(CONFIG, "tiny", ["tiny-agent"])
            with FakeLLM(script) as llm:
                result = run_task(task, Path(tmp) / "run", Path(tmp) / "snap",
                                  lambda home, manifest: OpencodeDriver(
                                      home, config, ref, llm.port,
                                      path=[str(Path.home() / p) for p in manifest["path"]]),
                                  lambda home: Box(home))
            outputs = [m for m in llm.requests if m.get("tools")][1]["messages"][-1]
        self.assertTrue(result["pass"], result["grading"])
        self.assertIn(facts["wrong"], str(outputs))  # the journal reached the model through the shim
        self.assertEqual(result["metrics"]["tools"].get("bash"), 2)


if __name__ == "__main__":
    unittest.main()
