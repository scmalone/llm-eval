"""The accuracy suite's collections: mined fixes, game and vision tasks, their environments and checks."""
import json
from pathlib import Path
import tempfile
import unittest

from llm_eval.workload import graders, prepare as workload, session
from llm_eval.workload.sandbox import Box
from llm_eval.workload.tasks import COLLECTIONS, validate


class CollectionTests(unittest.TestCase):
    def test_every_runnable_task_has_a_workload_group(self):
        from llm_eval import runner
        _, everything = workload.select("accuracy")
        # Every workload task must land in a group or the run grades nothing. The vision tasks are workload tasks too
        # (the classic `vision` section runs only the kind-"vision" picture fixtures); exempting them let the first
        # Muse vision phase grade nothing (2026-11-06).
        stray = {t["id"] for t in everything if t.get("kind", "workload") == "workload"
                 and runner._group(t) not in runner.WORKLOAD_GROUPS}
        self.assertEqual(stray, set())

    def test_collections_stay_out_of_core_and_join_the_accuracy_suite(self):
        _, core = workload.select("core")
        self.assertEqual(len(core), 35)
        self.assertFalse({t["category"] for t in core} & set(COLLECTIONS))
        _, everything = workload.select("accuracy")
        counts = {c: sum(t["category"] == c for t in everything) for c in COLLECTIONS}
        self.assertEqual(counts, {"mined": 20, "game": 6, "vision": 10})
        _, mined = workload.select("mined")
        self.assertEqual(len(mined), 20)
        repos = {}
        for task in mined:
            repos[task["mined"]["repo"]] = repos.get(task["mined"]["repo"], 0) + 1
        self.assertLessEqual(max(repos.values()), 4)  # plan rule: at most 4 per repo
        self.assertLessEqual(repos.get("Strata", 0), 2)

    def test_env_entries_are_checked(self):
        task = json.loads(next((workload.TASK_DIR / "mined").glob("mined-thermalright-*.json")).read_text())
        validate(dict(task))
        for bad in ({"from": "../x", "to": ".venv"}, {"from": "x", "to": "/abs"}, {"from": "x"},
                    {"from": "x", "to": "y", "writable": ["/abs"]}, {"from": "x", "to": "y"},
                    {"from": "x", "to": "y", "digest": "abc"},
                    {"from": "x", "to": "y", "digest": "0" * 64, "files": ["../up"]}):
            with self.assertRaises(ValueError):
                validate({**task, "env": [bad]})

    def test_env_mounts_reach_both_boxes_read_only_with_scratch_folders(self):
        from llm_eval.workload import envs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "envs" / "kit").mkdir(parents=True)
            (root / "envs" / "kit" / "lib.js").write_text("one")
            digest = envs.pin(root / "envs" / "kit")
            self.assertEqual(envs.recorded(root / "envs" / "kit"), digest)
            home, repo = root / "home", root / "home" / "Workspace" / "r"
            repo.mkdir(parents=True)
            task = {"id": "t", "env": [{"from": "kit", "to": "node_modules", "writable": [".vite-temp"], "digest": digest}]}
            mounts = session.env_mounts(task, repo, root / "envs")
            box = Box(home)
            session._mount(box, mounts)
            argv = box.argv(["true"], box.inside(repo))
            inside = box.inside(repo / "node_modules")
            self.assertIn(["--ro-bind", str(root / "envs" / "kit"), inside],
                          [argv[i:i + 3] for i in range(len(argv) - 2)])
            self.assertIn(f"{inside}/.vite-temp", box.tmpfs)
            with self.assertRaisesRegex(ValueError, "absent"):
                session.env_mounts({"id": "t", "env": [{"from": "missing", "to": "x", "digest": digest}]}, repo, root / "envs")
            (root / "envs" / "kit" / "lib.js").write_text("two")  # updated after the tasks were proven (RT-002)
            with self.assertRaisesRegex(ValueError, "changed since"):
                session.env_mounts(task, repo, root / "envs")
            with self.assertRaisesRegex(ValueError, "no digest"):
                session.env_mounts({"id": "t", "env": [{"from": "kit", "to": "x"}]}, repo, root / "envs")

    def test_a_files_mount_binds_one_asset_beside_tracked_files(self):
        from llm_eval.workload import envs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "envs" / "assets").mkdir(parents=True)
            (root / "envs" / "assets" / "heights.bin").write_bytes(b"\x01\x02")
            digest = envs.pin(root / "envs" / "assets")
            home, repo = root / "home", root / "home" / "Workspace" / "r"
            (repo / "public").mkdir(parents=True)
            (repo / "public" / "record.json").write_text("{}")
            task = {"id": "t", "env": [{"from": "assets", "to": "public", "files": ["heights.bin"], "digest": digest}]}
            mounts = session.env_mounts(task, repo, root / "envs")
            box = Box(home)
            session._mount(box, mounts)
            self.assertEqual((repo / "public" / "heights.bin").read_bytes(), b"")  # the empty mount point
            self.assertIn((str(root / "envs" / "assets" / "heights.bin"), box.inside(repo / "public" / "heights.bin")),
                          box.read_only)
            self.assertTrue((repo / "public" / "record.json").exists())  # the tracked file beside it is not hidden
            with self.assertRaisesRegex(ValueError, "absent"):
                session.env_mounts({"id": "t", "env": [{**task["env"][0], "files": ["nope.bin"]}]}, repo, root / "envs")

    def test_a_required_skip_fails_and_an_answer_field_holds_one_value(self):
        runs = {"OK (skipped=28)": 0}
        check = {"type": "tests", "command": ["python3", "-m", "unittest", "-v"], "expect": [r"(?m)^test_a \("],
                 "skipped": r"(?m)^(?:OK|FAILED) \(.*\b(?:skipped|expected failures)="}
        with tempfile.TemporaryDirectory() as tmp:
            outcome = {"workdir": tmp, "home": tmp, "base_files": {}, "frozen_globs": [], "snapshot": tmp,
                       "run": lambda argv, cwd, timeout: (0, "test_a (m.T.test_a) ... skipped 'x'\n\nRan 1 test\n\nOK (skipped=1)\n")}
            result = graders.check_tests({**check, "hidden": False}, outcome, None)
            self.assertFalse(result["pass"])
            self.assertIn("the test runner reported skipped tests", result["problems"])
            outcome["run"] = lambda argv, cwd, timeout: (0, "test_a (m.T.test_a) ... ok\n\nRan 1 test\n\nOK\n")
            self.assertTrue(graders.check_tests({**check, "hidden": False}, outcome, None)["pass"])
        task = json.loads((workload.TASK_DIR / "vision" / "vision-ocr-receipt-line.json").read_text())
        answer = task["checks"][0]
        self.assertTrue(graders.check_answer(answer, {"answer": "The bolts are 6.72.\nPRICE: 6.72"}, None)["pass"])
        self.assertTrue(graders.check_answer(answer, {"answer": "**PRICE:** $6.72\nPRICE: 6.72"}, None)["pass"])
        listed = "PRICE: 0.00\nPRICE: 6.72\nPRICE: 999.00"  # the red team's candidate list (RT-005)
        self.assertFalse(graders.check_answer(answer, {"answer": listed}, None)["pass"])
        mined = [json.loads(p.read_text()) for p in (workload.TASK_DIR / "mined").glob("*.json")]
        unit = [t for t in mined if "unittest" in " ".join(t["checks"][0]["command"])]
        self.assertTrue(unit and all(t["checks"][0].get("skipped") for t in unit))

    def test_box_grounding_takes_pixels_or_the_1000_grid_and_needs_overlap(self):
        check = {"type": "box", "target": [100, 100, 300, 200], "size": [1000, 500], "min_iou": 0.5}
        self.assertTrue(graders.check_box(check, {"answer": "It is there.\nBOX: 105, 98, 298, 205"}, {})["pass"])
        grid = graders.check_box(check, {"answer": "BOX: [100, 200, 300, 400]"}, {})  # 0-1000 grid, y halves
        self.assertTrue(grid["pass"])
        self.assertEqual(grid["convention"], "grid-1000")
        self.assertFalse(graders.check_box(check, {"answer": "BOX: 400, 300, 600, 450"}, {})["pass"])
        self.assertFalse(graders.check_box(check, {"answer": "around the top left"}, {})["pass"])

    def test_vision_tasks_need_image_input(self):
        _, vision = workload.select("vision")
        self.assertTrue(all(t.get("requires") == ["image"] for t in vision))
        from llm_eval.acceptance import _sees_images
        from unittest import mock
        with mock.patch("llm_eval.workload.pi_config.production_entry", return_value={"input": ["text"]}):
            self.assertFalse(_sees_images("llama.cpp", "p", {"settings": {"mmproj": "/m.gguf"}}))
        with mock.patch("llm_eval.workload.pi_config.production_entry", return_value={"input": ["text", "image"]}):
            self.assertTrue(_sees_images("llama.cpp", "p", {"settings": {}}))
        with mock.patch("llm_eval.workload.pi_config.production_entry", return_value=None):
            self.assertTrue(_sees_images("llama.cpp", "p", {"settings": {"mmproj": "/m.gguf"}}))

    def test_mined_packets_rebuild_unchanged_and_quote_no_fix(self):
        import subprocess, sys
        done = subprocess.run([sys.executable, "tools/build_mined_tasks.py", "suites/mined-specs.json", "--check"],
                              cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout[-1500:] + done.stderr[-1500:])


if __name__ == "__main__":
    unittest.main()
