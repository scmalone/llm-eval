import json
import re
import tempfile
import unittest
from pathlib import Path

from llm_eval.workload.driver import ARGV_LIMIT, OpencodeDriver, argv_chunks
from llm_eval.workload.sandbox import Box


class RelayCommandTests(unittest.TestCase):
    """The relay and the shims run with the task repo as CWD, and that repo can carry its own llm_eval/."""

    def box(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        home = Path(tmp.name) / "home"
        home.mkdir()
        return Box(home, real_home="/home/tester", relays=["127.0.0.1:8080=r0.sock"],
                   sockets=Path(tmp.name) / "sockets")

    def test_relay_does_not_put_the_task_cwd_first_on_sys_path(self):
        argv = self.box().argv(["true"], self.box().home)
        self.assertIn("llm_eval.workload.relay", argv)
        relay = argv.index("llm_eval.workload.relay")
        self.assertEqual(argv[relay - 2:relay], ["-P", "-m"])

    def test_the_bound_package_is_on_pythonpath(self):
        argv = self.box().argv(["true"], self.box().home)
        self.assertIn("PYTHONPATH", argv)
        self.assertEqual(argv[argv.index("PYTHONPATH") + 1], "/opt/llm-eval")



class ArgvChunkTests(unittest.TestCase):
    """Linux refuses any single argument over 128 KiB; the document tasks paste 130 KB and 400 KB prompts."""

    def test_a_short_message_stays_one_argument(self):
        self.assertEqual(argv_chunks("fix the bug"), ["fix the bug"])

    def test_a_long_message_splits_at_spaces_and_rejoins_exactly(self):
        text = " ".join(f"word{n}" for n in range(30000))
        chunks = argv_chunks(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.encode()) <= ARGV_LIMIT for chunk in chunks))
        self.assertEqual(" ".join(chunks), text)  # opencode joins its positional message arguments with a space

    def test_no_chunk_reaches_the_kernel_limit(self):
        self.assertLess(ARGV_LIMIT, 131072)

class ExportTests(unittest.TestCase):
    """``opencode export`` exits without draining its stdout pipe, so a session past 64 KiB arrives cut."""

    class PipeBox:
        """A box that writes what the command would have printed to its redirect targets."""

        def __init__(self, home, payload, code=0, said="Exporting session: ses_1"):
            self.home, self.payload, self.code, self.said = Path(home), payload, code, said
            self.commands = []

        def inside(self, host_path):
            return str(Path("/home/tester") / Path(host_path).relative_to(self.home))

        def host(self, inside):
            return self.home / Path(inside).relative_to("/home/tester")

        def run(self, command, cwd, timeout):
            self.commands.append(command)
            out, err = re.search(r"> (\S+) 2> (\S+)$", command[2]).groups()
            self.host(out).write_text(self.payload)
            self.host(err).write_text(self.said)
            return self.code, "", "", False

    def driver(self, box):
        driver = OpencodeDriver.__new__(OpencodeDriver)  # only the export path is under test
        driver.home, driver.session, driver.box = box.home, "ses_1", box
        return driver

    def test_export_reads_a_file_instead_of_the_pipe(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        record = {"info": {"id": "ses_1"}, "messages": [{"role": "user"}]}
        box = self.PipeBox(Path(tmp.name), json.dumps(record))
        self.assertEqual(self.driver(box).export(), record)
        self.assertIn("> ", box.commands[0][2])
        self.assertEqual(sorted(p.name for p in Path(tmp.name).iterdir()), [])  # the temporary file is gone

    def test_a_failed_export_still_says_so(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        box = self.PipeBox(Path(tmp.name), "", code=1, said="session not found")
        with self.assertRaises(RuntimeError) as caught:
            self.driver(box).export()
        self.assertIn("session not found", str(caught.exception))

    def test_an_unparsable_export_is_not_silently_empty(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        box = self.PipeBox(Path(tmp.name), '{"info": {"id": "ses_1"}, "mess')  # what the pipe used to deliver
        with self.assertRaises(ValueError):
            self.driver(box).export()


if __name__ == "__main__":
    unittest.main()
