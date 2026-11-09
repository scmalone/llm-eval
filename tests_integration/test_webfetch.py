"""opencode's real webfetch through the canned-page proxy: canned pages served, all else refused."""
import shutil
from pathlib import Path
import tempfile
import unittest

from llm_eval.workload import opencode_config as oc
from llm_eval.workload.driver import OpencodeDriver
from llm_eval.workload.fakellm import FakeLLM
from llm_eval.workload.webfix import PROXY_INSIDE, CannedPages, sandbox_env
from llm_eval.workload.metrics import summarize
from test_opencode_sandbox import CONFIG

PAGES = {"https://docs.example.org/guide": {"body": "<html><body><p>The retry limit is 7.</p></body></html>"},
         "http://plain.example.org/notes": {"body": "Plain notes: timeout is 45 seconds.", "content_type": "text/plain"}}


def fetch(url):
    return {"tool_calls": [{"name": "webfetch", "arguments": {"url": url, "format": "text"}}]}


class WebfetchTests(unittest.TestCase):
    def test_canned_pages_are_served_and_everything_else_fails(self):
        self.assertTrue(shutil.which("openssl"), "openssl is required")
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            (home / "work").mkdir(parents=True)
            config, ref, _ = oc.build(CONFIG, "tiny", ["tiny-agent"])
            script = [fetch("https://docs.example.org/guide"), fetch("http://plain.example.org/notes"),
                      fetch("https://elsewhere.example.com/"), {"text": "Fetched three."}]
            with CannedPages(PAGES, Path(tmp) / "ca") as pages, FakeLLM(script) as llm:
                shutil.copy(pages.ca, home / "canned-ca.pem")
                env = sandbox_env(str(Path.home() / "canned-ca.pem"))
                with OpencodeDriver(home, config, ref, llm.port, relays=[(PROXY_INSIDE, pages.port)], env=env) as driver:
                    driver.turn("Read these pages.", home / "work", 180)
                    export = driver.export()
            outputs = [p["state"].get("output", "") + str(p["state"].get("error", ""))
                       for m in export["messages"] for p in m["parts"] if p.get("type") == "tool"]
            self.assertEqual(len(outputs), 3, outputs)
            self.assertIn("retry limit is 7", outputs[0])
            self.assertIn("timeout is 45", outputs[1])
            self.assertNotIn("retry limit", outputs[2])
            self.assertEqual(sorted(pages.hits), sorted(PAGES))
            self.assertTrue(any("elsewhere.example.com" in m for m in pages.misses))
            self.assertEqual(summarize(export)["tools"].get("webfetch"), 3)


if __name__ == "__main__":
    unittest.main()
