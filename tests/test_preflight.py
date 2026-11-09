import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from llm_eval.cli import main
from llm_eval.plan import compile_plan, digest, verify_plan
from llm_eval.preflight import RETAINED, eval_manifest, inspect_machine, retained_manifest, router_health
from llm_eval.runner import execute


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.target = self.root / "target.gguf"
        self.target.write_bytes(b"fake target, never loaded")
        self.draft = self.root / "draft.gguf"
        self.draft.write_bytes(b"fake draft, never loaded")
        self.preset = self.root / "models-preset.ini"
        sections = []
        for name, speculation in RETAINED:
            section = f"[{name}]\nmodel = {self.target}\nctx-size = 131072\nparallel = 1\n"
            if speculation:
                section += f"model-draft = {self.draft}\nspec-type = {speculation}\n"
            sections.append(section)
        self.preset.write_text("\n".join(sections))

    def test_generator_and_plan_only_read_files_and_pin_artifacts(self):
        with patch("subprocess.run", side_effect=AssertionError("process")), patch(
                "http.client.HTTPConnection", side_effect=AssertionError("network")):
            manifest = retained_manifest(self.preset)
            manifest_path = self.root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            plan = compile_plan(manifest_path)
        self.assertEqual([m["id"] for m in plan["models"]], [name for name, _ in RETAINED])
        self.assertEqual([len(m["artifacts"]) for m in plan["models"]], [2, 1, 2, 1])
        self.assertEqual(plan["files"][str(self.target)], digest(self.target))
        self.assertEqual(plan["files"][str(self.preset)], digest(self.preset))
        self.assertEqual(plan["backend"], "live")
        self.assertFalse(plan["live_execution_ready"])
        for resume in (False, True):
            with self.assertRaisesRegex(RuntimeError, "live execution disabled"):
                execute(plan, self.root / "runs", "test", resume=resume)
        self.assertFalse((self.root / "runs").exists())

    def test_generator_hashes_shared_artifacts_once(self):
        with patch("llm_eval.preflight.digest", wraps=digest) as hashed:
            retained_manifest(self.preset)
        self.assertEqual(sum(str(call.args[0]) == str(self.target) for call in hashed.call_args_list), 1)
        self.assertEqual(sum(str(call.args[0]) == str(self.draft) for call in hashed.call_args_list), 1)

    def test_plan_rejects_preset_changed_after_manifest_generation(self):
        manifest = retained_manifest(self.preset)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        self.preset.write_text(self.preset.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "preset hash mismatch"):
            compile_plan(path)

    def test_plan_rejects_artifact_changed_after_planning(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(retained_manifest(self.preset)))
        plan = compile_plan(path)
        self.draft.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "planned input changed"):
            verify_plan(plan)

    def test_missing_artifact_or_profile_fails_generation(self):
        self.draft.unlink()
        with self.assertRaisesRegex(ValueError, "missing regular model-draft"):
            retained_manifest(self.preset)
        self.preset.write_text("[other]\nmodel=missing\n")
        with self.assertRaisesRegex(ValueError, "restore model absent"):
            retained_manifest(self.preset)

    def test_unexpected_speculation_fails_generation(self):
        self.preset.write_text(self.preset.read_text().replace("draft-dflash", "draft-mtp"))
        with self.assertRaisesRegex(ValueError, "unexpected model/speculation"):
            retained_manifest(self.preset)

    def test_preflight_uses_only_read_commands(self):
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, "observed\n")) as run, patch(
                "llm_eval.preflight.router_health", return_value={"status": 200, "healthy": True}):
            report = inspect_machine(self.preset)
        self.assertTrue(report["observations_complete"])
        self.assertFalse(report["live_execution_ready"])
        self.assertEqual(len(run.call_args_list), 3)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:4], ["systemctl", "--user", "show", "llama-router.service"])
        self.assertTrue(all(args[0] == "nvidia-smi" for args in commands[1:]))
        self.assertEqual(report["observations"]["presets"]["profiles"][RETAINED[0][0]]["artifacts"][1]["sha256"], digest(self.draft))

    def test_failed_probes_are_recorded_and_never_enable_execution(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("missing command")), patch(
                "llm_eval.preflight.router_health", side_effect=TimeoutError("offline")):
            report = inspect_machine(self.preset)
        self.assertFalse(report["observations_complete"])
        self.assertEqual(set(report["errors"]), {"systemd", "gpu", "gpu_processes", "router_health"})
        self.assertIn("presets", report["observations"])
        self.assertFalse(report["live_execution_ready"])

    def test_health_only_gets_health_and_does_not_follow_redirects(self):
        connection = MagicMock()
        connection.getresponse.return_value.status = 302
        connection.getresponse.return_value.read.return_value = b"redirect"
        with patch("http.client.HTTPConnection", return_value=connection):
            result = router_health("http://127.0.0.1:8080/health")
        connection.request.assert_called_once_with("GET", "/health")
        connection.close.assert_called_once()
        self.assertFalse(result["healthy"])

    def test_health_rejects_remote_generation_and_auth_urls_before_network(self):
        urls = ("http://example.com/health", "http://127.0.0.1/completion",
                "http://127.0.0.1/health?model=foo", "http://user:pass@127.0.0.1/health",
                "https://127.0.0.1/health")
        with patch("http.client.HTTPConnection", side_effect=AssertionError("network")):
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    router_health(url)

    def test_unit_options_and_wildcards_rejected_before_process(self):
        with patch("subprocess.run", side_effect=AssertionError("process")):
            for unit in ("--all.service", "*.service", "foo.service bar.service"):
                with self.subTest(unit=unit), self.assertRaises(ValueError):
                    inspect_machine(self.preset, unit=unit)

    def test_cli_preparation_writes_only_requested_files(self):
        output = self.root / "manifest.json"
        with contextlib.redirect_stdout(io.StringIO()), patch("subprocess.run", side_effect=AssertionError("process")):
            self.assertEqual(main(["retained-manifest", "--preset", str(self.preset),
                                   "--restore-model", RETAINED[0][0], "--out", str(output)]), 0)
        self.assertEqual(json.loads(output.read_text())["backend"], "live")
        with contextlib.redirect_stdout(io.StringIO()), patch("llm_eval.cli.inspect_machine", return_value={
                "errors": {}, "router_healthy": False, "live_execution_ready": False}):
            self.assertEqual(main(["preflight", "--out", str(self.root / "preflight.json")]), 1)


class PinTests(unittest.TestCase):
    def test_the_pinned_build_passes_and_another_is_refused(self):
        from llm_eval.preflight import check_pin, PIN
        pin = json.loads(PIN.read_text())
        self.assertEqual(check_pin(build_of=lambda b: (pin["build"], pin["commit"] + "abc"))["build"], pin["build"])
        with self.assertRaisesRegex(ValueError, "not the pinned build"):
            check_pin(build_of=lambda b: (pin["build"] + 1, pin["commit"]))
        with self.assertRaisesRegex(ValueError, "not the pinned build"):
            check_pin(build_of=lambda b: (None, None))


if __name__ == "__main__":
    unittest.main()


class EvalManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("a.gguf", "b.gguf", "draft.gguf"):
            (self.root / name).write_bytes(name.encode())
        self.preset = self.root / "models-preset.ini"
        self.preset.write_text(
            f"[big]\nmodel = {self.root}/a.gguf\nctx-size = 131072\nparallel = 1\n\n"
            f"[big-agent]\nmodel = {self.root}/a.gguf\nctx-size = 65536\nparallel = 1\nsleep-idle-seconds = 600\n\n"
            f"[q8]\nmodel = {self.root}/a.gguf\nctx-size = 131072\ncache-type-k = q8_0\n\n"
            f"[small]\nmodel = {self.root}/b.gguf\nctx-size = 65536\n\n"
            f"[muse-glimmer-30b-q4]\nmodel = {self.root}/b.gguf\nmodel-draft = {self.root}/draft.gguf\nctx-size = 131072\n")

    def test_context_only_variants_collapse_to_the_highest_context_profile(self):
        manifest = eval_manifest(self.preset, restore_model="big-agent")
        by_id = {m["id"]: m for m in manifest["models"]}
        self.assertEqual(sorted(by_id), ["big", "muse-glimmer-30b-q4", "q8", "small"])
        self.assertEqual(by_id["big"]["covers"], ["big", "big-agent"])

    def test_anchor_follows_owner_policy_then_caps_at_98k(self):
        by_id = {m["id"]: m for m in eval_manifest(self.preset, restore_model="big")["models"]}
        self.assertEqual(by_id["muse-glimmer-30b-q4"]["anchor"], 98304)
        self.assertEqual(by_id["big"]["anchor"], 98304)
        self.assertEqual(by_id["small"]["anchor"], 65536)

    def test_profile_subset_and_unknown_names(self):
        manifest = eval_manifest(self.preset, restore_model="big", profiles=["small", "big-agent"])
        self.assertEqual(sorted(m["id"] for m in manifest["models"]), ["big-agent", "small"])
        with self.assertRaisesRegex(ValueError, "absent from presets"):
            eval_manifest(self.preset, restore_model="big", profiles=["nope"])
        with self.assertRaisesRegex(ValueError, "restore model absent"):
            eval_manifest(self.preset, restore_model="nope")

    def test_manifest_compiles_to_a_plan_without_processes(self):
        path = self.root / "manifest.json"
        with patch("subprocess.run", side_effect=AssertionError("process")):
            path.write_text(json.dumps(eval_manifest(self.preset, restore_model="big")))
            plan = compile_plan(path)
        self.assertEqual(len(plan["models"]), 4)
        self.assertFalse(plan["live_execution_ready"])
