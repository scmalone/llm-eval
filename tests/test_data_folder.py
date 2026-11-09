import importlib
import os
from pathlib import Path
import tempfile
import unittest

from llm_eval.workload.snapshot import escape_reason


class DataFolderTests(unittest.TestCase):
    """The data folder moved to ~/Workspace/llm-eval-data on 2026-11-01, with a symlink at the old name."""

    def test_the_environment_variable_names_the_data_folder(self):
        import llm_eval.paths as paths
        old = os.environ.get("LLM_EVAL_DATA")
        try:
            os.environ["LLM_EVAL_DATA"] = "/tmp/elsewhere"
            self.assertEqual(importlib.reload(paths).DATA, Path("/tmp/elsewhere"))
        finally:
            if old is None:
                os.environ.pop("LLM_EVAL_DATA", None)
            else:
                os.environ["LLM_EVAL_DATA"] = old
            importlib.reload(paths)

    def test_a_run_reached_through_the_old_symlinked_name_is_still_inside_its_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "llm-eval-data"
            (real / "run" / "home" / "repo").mkdir(parents=True)
            link = Path(tmp) / "strata-eval-runs"
            link.symlink_to(real)
            home = link / "run" / "home"
            self.assertIsNone(escape_reason(home / "repo", home))
            (home / "repo" / "out").symlink_to("/etc")
            self.assertIsNotNone(escape_reason(home / "repo" / "out", home))


if __name__ == "__main__":
    unittest.main()
