import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_eval.workload import machine, replicas
from llm_eval.workload.graders import grade
from llm_eval.workload.sandbox import PACKAGE_ROOT, Box
from llm_eval.workload.snapshot import materialize

TASK = {"id": "ops-router-1", "kind": "workload", "category": "ops", "prompt": "The router is down; fix it.",
        "source": {"replica": "router-bad-model-path", "seed": 3},
        "checks": [{"type": "answer", "all": ["x"]}], "limits": {"turns": 4, "wall_seconds": 900}}


class ReplicaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = replicas.build(TASK, self.root / "snap")
        self.home = self.root / "home"
        self.repo = materialize(self.root / "snap", self.home / self.manifest["place"])
        replicas.unpack_home(self.root / "snap", self.home)
        self.facts = replicas.facts(TASK)
        self.preset = self.repo / "models-preset.ini"

    def tearDown(self):
        self.tmp.cleanup()

    def shim(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, {"HOME": str(self.home)}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = machine.main(list(argv))
        return code, out.getvalue() + err.getvalue()

    def fix(self):
        self.preset.write_text(self.preset.read_text().replace(self.facts["wrong"], self.facts["actual"]))

    def test_replica_is_deterministic_and_seeded(self):
        again = replicas.build(TASK, self.root / "snap2")
        self.assertEqual(again["files"], self.manifest["files"])
        self.assertEqual(again["home_files"], self.manifest["home_files"])
        seen = {json.dumps(replicas.facts({**TASK, "source": {"replica": "router-bad-model-path", "seed": s}}))
                for s in range(10)}
        self.assertGreater(len(seen), 1)  # the seed varies the fault
        self.assertEqual(self.manifest["place"], ".config/llama-cpp")
        self.assertTrue(os.access(self.home / ".local/bin/systemctl", os.X_OK))
        # The ini points at the wrong file, which is not on disk; the right one is.
        self.assertIn(self.facts["wrong"], self.preset.read_text())
        self.assertFalse((self.home / "models" / self.facts["wrong"]).exists())
        self.assertTrue((self.home / "models" / self.facts["actual"]).exists())

    def test_shims_show_the_fault_until_the_config_is_fixed(self):
        self.assertEqual(self.shim("systemctl", "--user", "status", "llama-router")[0], 3)
        code, journal = self.shim("journalctl", "--user", "-u", "llama-router.service", "-n", "20")
        self.assertIn(self.facts["wrong"], journal)
        self.assertEqual(self.shim("systemctl", "--user", "restart", "llama-router")[0], 1)
        self.assertIn("Operation not permitted", self.shim("systemctl", "restart", "llama-router")[1])
        self.fix()
        self.assertEqual(self.shim("systemctl", "--user", "restart", "llama-router.service")[0], 0)
        self.assertEqual(self.shim("systemctl", "--user", "is-active", "llama-router")[1].strip(), "active")
        state = json.loads((self.home / replicas.MACHINE).read_text())
        self.assertEqual(state["log"], ["restart llama-router.service", "restart llama-router.service"])
        self.assertIn("listening", self.shim("journalctl", "--user", "-u", "llama-router")[1])
        self.assertIn("USER", self.shim("ps", "aux")[1])
        self.assertEqual(self.shim("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader")[1].strip(), "24564")

    def test_state_check_passes_only_after_fix_and_restart(self):
        checks = [{"type": "state", "file": "~/" + replicas.MACHINE, "equals": {"units/llama-router.service/active": True}},
                  {"type": "files", "contains": {"~/.config/llama-cpp/models-preset.ini": self.facts["actual"]}},
                  {"type": "commands", "forbid": [r"(^|\s)sudo\s", r"\brm\s+-rf\b"]}]
        outcome = lambda commands=(): {"answer": "", "commands": list(commands), "workdir": self.repo,
                                       "home": self.home, "base_files": self.manifest["files"]}
        self.assertFalse(grade({"checks": checks}, outcome())["pass"])
        self.fix()
        self.assertFalse(grade({"checks": checks}, outcome())["pass"])  # fixed but not restarted
        self.shim("systemctl", "--user", "restart", "llama-router")
        self.assertTrue(grade({"checks": checks}, outcome())["pass"])
        self.assertFalse(grade({"checks": checks}, outcome(["sudo systemctl restart llama-router"]))["pass"])


class BoxBindTests(unittest.TestCase):
    def test_only_inside_modules_are_visible_to_the_agent(self):
        argv = Box("/tmp/h", real_home="/home/u").argv(["true"], "/home/u")
        sources = [argv[i + 1] for i, a in enumerate(argv) if a == "--ro-bind"]
        self.assertNotIn(str(PACKAGE_ROOT), sources)
        bound = {Path(s).name for s in sources if s.startswith(str(PACKAGE_ROOT))}
        self.assertEqual(bound, {"__init__.py", "relay.py", "machine.py"})
        for secret in ("graders.py", "replicas.py", "tasks.py", "fixtures", "tests"):
            self.assertFalse(any(secret in s for s in sources), secret)


if __name__ == "__main__":
    unittest.main()
