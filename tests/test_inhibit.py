"""The live runner holds a systemd sleep lock for the armed campaign (plans/2026-10-11-runner-sleep-inhibit.md)."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from llm_eval import acceptance, inhibit
from llm_eval.adapters import FakeAdapter
from llm_eval.inhibit import InhibitUnavailable, sleep_inhibit
from llm_eval.live import LiveAdapter, RealHost
from llm_eval.runner import execute
from llm_eval.state import Journal
from test_live import ScriptedHost
from test_runner import plan_with

ROOT = Path(__file__).resolve().parents[1]

# A systemd-inhibit stand-in that behaves like the real one: it forks the trailing command and waits for it,
# and `--list` prints WHO UID USER PID COMM WHAT WHY MODE rows.
FAKE_INHIBIT = """#!/bin/sh
D="$FAKE_INHIBIT_DIR"
if [ "$1" = "--list" ]; then
  case "$FAKE_INHIBIT_MODE" in
    nomatch|die) exit 0 ;;
    decoy) printf 'llm-eval %s user 4242 systemd-inhibit sleep:idle why block\\n' "$(cat "$D/pid")"; exit 0 ;;
  esac
  printf 'other 1000 user 1 foo sleep why block\\n'
  printf 'llm-eval 1000 user %s systemd-inhibit sleep:idle why block\\n' "$(cat "$D/pid")"
  exit 0
fi
echo $$ > "$D/pid"
echo "$@" > "$D/args"
[ "$FAKE_INHIBIT_MODE" = die ] && exit 1
while [ "${1#--}" != "$1" ]; do shift; done
"$@" &
wait $!
"""


def group_members(pgid):
    """Pids whose process group is pgid (Linux /proc)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        if int(stat.rsplit(")", 1)[1].split()[2]) == pgid:
            found.append(int(entry.name))
    return found


def wait_until(check, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.05)
    return check()


class FakeInhibitCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        script = self.bin / "systemd-inhibit"
        script.write_text(FAKE_INHIBIT)
        script.chmod(0o755)
        self.fake = self.root / "fake"
        self.fake.mkdir()
        env = patch.dict(os.environ, {"PATH": f"{self.bin}:{os.environ['PATH']}", "FAKE_INHIBIT_DIR": str(self.fake),
                                      "FAKE_INHIBIT_MODE": ""})
        env.start()
        self.addCleanup(env.stop)
        verify = patch.object(inhibit, "VERIFY_SECONDS", 0.6)
        verify.start()
        self.addCleanup(verify.stop)


class LockProcessTests(FakeInhibitCase):
    def test_spawn_arguments_and_the_whole_group_is_released(self):
        with sleep_inhibit("llm-eval r1") as handle:
            self.assertEqual((self.fake / "args").read_text().strip(),
                             f"--what=sleep:idle --who=llm-eval --why=llm-eval r1 --mode=block tail --pid={os.getpid()} -f /dev/null")
            self.assertTrue(handle.alive())
            pgid = handle.pid
            self.assertGreaterEqual(len(group_members(pgid)), 2)  # systemd-inhibit and its tail child
        self.assertTrue(wait_until(lambda: not group_members(pgid), 3), group_members(pgid))

    def test_a_killed_runner_does_not_leak_the_lock(self):
        code = ("import sys, time\n"
                "from llm_eval.inhibit import sleep_inhibit\n"
                "with sleep_inhibit('x') as h:\n"
                "    print(h.pid, flush=True)\n"
                "    time.sleep(60)\n")
        child = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                                 env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.addCleanup(child.kill)
        self.addCleanup(child.stdout.close)
        pgid = int(child.stdout.readline())
        self.assertGreaterEqual(len(group_members(pgid)), 2)
        child.send_signal(signal.SIGKILL)
        child.wait()
        self.assertTrue(wait_until(lambda: not group_members(pgid), 5), group_members(pgid))

    def test_unavailable_is_raised_and_nothing_is_left_running(self):
        cases = {"die": "exited", "nomatch": "never appeared", "decoy": "never appeared"}
        for mode, _ in cases.items():
            with self.subTest(mode=mode):
                os.environ["FAKE_INHIBIT_MODE"] = mode
                with self.assertRaises(InhibitUnavailable):
                    with sleep_inhibit("x"):
                        self.fail("must not enter")
                pid = int((self.fake / "pid").read_text())
                self.assertTrue(wait_until(lambda: not group_members(pid), 3))

    def test_a_missing_binary_is_unavailable(self):
        empty = self.root / "empty"
        empty.mkdir()
        with patch.dict(os.environ, {"PATH": str(empty)}):
            with self.assertRaises(InhibitUnavailable):
                with sleep_inhibit("x"):
                    self.fail("must not enter")

    def test_an_interrupt_during_verification_leaves_nothing_behind(self):
        made, real = [], subprocess.Popen

        def spy(*args, **kwargs):
            made.append(real(*args, **kwargs))
            return made[-1]
        with patch.object(inhibit.subprocess, "Popen", spy), patch.object(inhibit, "_listed", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                with sleep_inhibit("x"):
                    self.fail("must not enter")
        self.assertEqual(len(made), 1)
        self.assertTrue(wait_until(lambda: not group_members(made[0].pid), 3))


class WatcherTests(FakeInhibitCase):
    def test_a_lost_lock_is_retaken_once_and_recorded(self):
        events = []
        with sleep_inhibit("x") as handle:
            first = handle.pid
            handle.start_watcher(lambda name, **v: events.append(name), interval=0.1)
            os.killpg(first, signal.SIGKILL)
            self.assertTrue(wait_until(lambda: "sleep_inhibit_retaken" in events, 5), events)
            self.assertNotEqual(handle.pid, first)
            self.assertTrue(handle.alive())
            self.assertFalse(handle.lost)
            second = handle.pid
        self.assertEqual(events[:2], ["sleep_inhibit_lost", "sleep_inhibit_retaken"])
        self.assertTrue(wait_until(lambda: not group_members(second) and not group_members(first), 3))

    def test_a_failed_retake_sets_the_flag(self):
        events = []
        with sleep_inhibit("x") as handle:
            handle.start_watcher(lambda name, **v: events.append(name), interval=0.1)
            os.environ["FAKE_INHIBIT_MODE"] = "die"
            os.killpg(handle.pid, signal.SIGKILL)
            self.assertTrue(wait_until(lambda: handle.lost, 5), events)
        self.assertIn("sleep_inhibit_lost", events)
        self.assertIn("sleep_inhibit_retake_failed", events)

    def test_stop_joins_the_watcher_and_is_idempotent(self):
        with sleep_inhibit("x") as handle:
            handle.start_watcher(lambda name, **v: None, interval=0.05)
            self.assertTrue(handle.stop_watcher())
            self.assertFalse(handle.watcher_running())
            self.assertTrue(handle.stop_watcher())

    def test_a_second_loss_is_not_retaken(self):
        events = []
        with sleep_inhibit("x") as handle:
            handle.start_watcher(lambda name, **v: events.append(name), interval=0.1)
            os.killpg(handle.pid, signal.SIGKILL)
            self.assertTrue(wait_until(lambda: "sleep_inhibit_retaken" in events, 5), events)
            os.killpg(handle.pid, signal.SIGKILL)
            self.assertTrue(wait_until(lambda: handle.lost, 5), events)
        self.assertEqual(events, ["sleep_inhibit_lost", "sleep_inhibit_retaken", "sleep_inhibit_lost"])

    def test_an_unexpected_watcher_failure_marks_the_lock_lost(self):  # L4
        def boom(name, **values):
            raise ValueError("journal write failed")
        with sleep_inhibit("x") as handle:
            handle.start_watcher(boom, interval=0.1)
            os.killpg(handle.pid, signal.SIGKILL)
            self.assertTrue(wait_until(lambda: handle.lost, 5))

    def test_a_hung_watcher_cannot_hold_up_the_stop(self):  # M2
        gate = threading.Event()
        self.addCleanup(gate.set)
        with sleep_inhibit("x") as handle:
            handle.start_watcher(lambda name, **v: gate.wait(30), interval=0.05)
            os.killpg(handle.pid, signal.SIGKILL)  # the watcher notices and blocks inside on_event
            time.sleep(0.4)
            started = time.monotonic()
            self.assertFalse(handle.stop_watcher(0.3))
            self.assertLess(time.monotonic() - started, 2)
            gate.set()
            self.assertTrue(handle.stop_watcher(5))

    def test_a_stopped_watcher_does_not_retake(self):  # M1
        events = []
        with sleep_inhibit("x") as handle:
            handle.start_watcher(lambda name, **v: events.append(name), interval=5)
            handle.stop_watcher()
            os.killpg(handle.pid, signal.SIGKILL)
            time.sleep(0.3)
        self.assertEqual(events, [])


class JournalThreadTests(unittest.TestCase):
    def test_concurrent_events_keep_unique_sequence_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(Path(tmp) / "run")

            def hammer(label):
                for n in range(150):
                    journal.event("tick", who=label, n=n)
                    if n % 25 == 0:
                        journal.save()
            threads = [threading.Thread(target=hammer, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            sequences = [json.loads(line)["sequence"] for line in (Path(tmp) / "run/events.jsonl").read_text().splitlines()]
            self.assertEqual(len(sequences), 450)
            self.assertEqual(len(set(sequences)), 450)


class Recording(FakeAdapter):
    """A fake adapter that claims a real machine and logs the calls the lock must bracket."""
    holds_machine = True
    log = None
    on_quiesce = None

    def quiesce(self):
        self.log.append("quiesce")
        if self.on_quiesce:
            self.on_quiesce()
        return super().quiesce()

    def restore(self, snapshot):
        self.log.append("restore")
        return super().restore(snapshot)


class StubHandle:
    pid = 4242
    lost = False
    stuck = False

    def __init__(self, log):
        self.log, self.on_event = log, None
        self.sigint_at_stop = None

    def alive(self):
        return True

    def start_watcher(self, on_event, interval=5.0):
        self.log.append("start_watcher")
        self.on_event = on_event

    def stop_watcher(self, timeout=None):
        self.log.append("stop_watcher")
        self.sigint_at_stop = signal.getsignal(signal.SIGINT)
        return not self.stuck


def make_stub(log, directory, fail=False, handles=None):
    @contextlib.contextmanager
    def stub(why, guard=contextlib.nullcontext):
        if fail:
            log.append("refuse")
            raise InhibitUnavailable("no systemd-inhibit")
        state = json.loads((directory / "state.json").read_text())
        log.append(("acquired_with_restoration_required", state["restoration_required"]))
        handle = StubHandle(log)
        if handles is not None:
            handles.append(handle)
        try:
            yield handle
        finally:
            log.append("release")
    return stub


class ExecutePlacementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = plan_with()
        self.log = []
        self.directory = self.root / "runs" / "r1"

    def adapter(self, on_quiesce=None):
        adapter = Recording(self.directory)
        adapter.log, adapter.on_quiesce = self.log, on_quiesce
        return adapter

    def go(self, adapter, resume=False, fail=False, handles=None):
        with patch.object(inhibit, "sleep_inhibit", make_stub(self.log, self.directory, fail, handles)):
            return execute(self.plan, self.root / "runs", "r1", resume=resume, adapter=adapter)

    def state(self):
        return json.loads((self.directory / "state.json").read_text())

    def events(self):
        return [json.loads(l) for l in (self.directory / "events.jsonl").read_text().splitlines()]

    def test_lock_is_taken_before_the_journal_is_armed_and_released_after_the_restore(self):
        result = self.go(self.adapter())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(self.log[0], ("acquired_with_restoration_required", False))
        names = [x for x in self.log if isinstance(x, str)]
        self.assertLess(names.index("quiesce"), names.index("restore"))
        self.assertLess(names.index("stop_watcher"), names.index("restore"))  # the watcher is joined before the restore
        self.assertLess(names.index("restore"), names.index("release"))
        self.assertEqual(names[-1], "release")

    def test_held_then_released_are_journaled_after_the_restore(self):
        self.go(self.adapter())
        names = [e["event"] for e in self.events()]
        self.assertIn("sleep_inhibit_held", names)
        self.assertLess(names.index("sleep_inhibit_held"), names.index("sleep_inhibit_released"))
        self.assertLess(names.index("restoration_complete"), names.index("sleep_inhibit_released"))
        self.assertTrue(next(e for e in self.events() if e["event"] == "sleep_inhibit_released")["alive"])

    def test_a_refused_start_is_clean_and_nothing_is_swallowed(self):
        adapter = self.adapter()
        with self.assertRaises(InhibitUnavailable):
            self.go(adapter, fail=True)
        state = self.state()
        self.assertEqual(state["status"], "refused_no_inhibit")
        self.assertFalse(state["restoration_required"])
        self.assertNotIn("quiesce", self.log)
        self.assertNotIn("restore", self.log)
        self.assertFalse(list((self.directory / "telemetry").glob("*"))) if (self.directory / "telemetry").exists() else None

    def test_resume_after_a_refusal_runs_the_campaign(self):
        with self.assertRaises(InhibitUnavailable):
            self.go(self.adapter(), fail=True)
        result = self.go(self.adapter(), resume=True)
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["restoration_required"])

    def test_a_pending_restore_is_never_blocked_by_a_missing_inhibitor(self):
        self.go(self.adapter())
        state = self.state()
        state.update(status="running", restoration_required=True)
        (self.directory / "state.json").write_text(json.dumps(state))
        del self.log[:]
        adapter = self.adapter()
        with self.assertRaises(InhibitUnavailable):
            self.go(adapter, resume=True, fail=True)
        self.assertIn("restore", self.log)  # production was brought back before the refusal
        self.assertLess(self.log.index("restore"), self.log.index("refuse"))
        self.assertFalse(self.state()["restoration_required"])

    def test_a_lost_lock_stops_the_campaign_at_the_model_boundary_and_restores(self):
        handles = []

        def lose():
            handles[0].lost = True
        result = self.go(self.adapter(on_quiesce=lose), handles=handles)
        self.assertEqual(result["status"], "partial")
        self.assertIn("restore", self.log)
        self.assertIn("sleep lock lost", " ".join(str(e.get("error", "")) for e in self.events()))
        self.assertEqual([x for x in self.log if isinstance(x, str)][-1], "release")

    def test_the_watcher_writes_through_the_journal(self):
        handles = []

        def lose():
            handles[0].on_event("sleep_inhibit_lost", pid=1)
        self.go(self.adapter(on_quiesce=lose), handles=handles)
        self.assertIn("sleep_inhibit_lost", [e["event"] for e in self.events()])

    def test_the_watcher_is_stopped_with_signals_ignored_so_a_second_ctrl_c_cannot_skip_the_restore(self):  # M1
        handles = []
        self.go(self.adapter(), handles=handles)
        self.assertEqual(handles[0].sigint_at_stop, signal.SIG_IGN)

    def test_a_stuck_watcher_does_not_block_the_restore(self):  # M2
        handles = []

        def stick():
            handles[0].stuck = True
        result = self.go(self.adapter(on_quiesce=stick), handles=handles)
        self.assertIn("restore", self.log)
        self.assertIn("sleep_inhibit_watcher_stuck", [e["event"] for e in self.events()])
        self.assertEqual(result["status"], "complete")

    def test_a_failed_write_during_a_refusal_does_not_replace_it(self):  # L3
        real_save = Journal.save

        def flaky(journal):
            if journal.state["status"] == "refused_no_inhibit":
                raise OSError("disk full")
            return real_save(journal)
        with patch.object(Journal, "save", flaky), self.assertRaises(InhibitUnavailable):
            self.go(self.adapter(), fail=True)

    def test_a_failed_held_event_does_not_fail_the_run(self):  # L3
        real_event = Journal.event

        def flaky(journal, event, **values):
            if event == "sleep_inhibit_held":
                raise OSError("disk full")
            return real_event(journal, event, **values)
        with patch.object(Journal, "event", flaky):
            result = self.go(self.adapter())
        self.assertEqual(result["status"], "complete")

    def test_adapters_without_holds_machine_take_no_lock(self):
        adapter = FakeAdapter(self.directory)
        with patch.object(inhibit, "sleep_inhibit", side_effect=AssertionError("must not be called")):
            result = execute(self.plan, self.root / "runs", "r1", adapter=adapter)
        self.assertEqual(result["status"], "complete")

    def test_only_a_real_host_claims_the_machine(self):
        from test_live import plan_for
        plan = plan_for(self.root)
        self.assertFalse(LiveAdapter(plan, self.root, ScriptedHost(), run_id="r1").holds_machine)
        self.assertTrue(LiveAdapter(plan, self.root, RealHost(), run_id="r1").holds_machine)


class RealLockExecuteTests(FakeInhibitCase):
    """execute with the real sleep_inhibit and the forking fake systemd-inhibit: no stub between them."""

    def setUp(self):
        super().setUp()
        self.plan = plan_with()
        self.directory = self.root / "runs" / "r1"

    def adapter(self):
        adapter = Recording(self.directory)
        adapter.log = []
        return adapter

    def test_the_campaign_runs_under_the_real_lock_and_leaves_no_process_behind(self):
        result = execute(self.plan, self.root / "runs", "r1", adapter=self.adapter())
        self.assertEqual(result["status"], "complete")
        pid = int((self.fake / "pid").read_text())
        self.assertTrue(wait_until(lambda: not group_members(pid), 3), group_members(pid))
        names = [json.loads(l)["event"] for l in (self.directory / "events.jsonl").read_text().splitlines()]
        self.assertLess(names.index("sleep_inhibit_held"), names.index("sleep_inhibit_released"))

    def test_an_unavailable_inhibitor_refuses_before_the_journal_is_armed(self):
        os.environ["FAKE_INHIBIT_MODE"] = "nomatch"
        adapter = self.adapter()
        with self.assertRaises(InhibitUnavailable):
            execute(self.plan, self.root / "runs", "r1", adapter=adapter)
        state = json.loads((self.directory / "state.json").read_text())
        self.assertEqual((state["status"], state["restoration_required"]), ("refused_no_inhibit", False))
        self.assertNotIn("quiesce", adapter.log)


class LadderRefusalTests(unittest.TestCase):
    def test_a_refused_attempt_points_at_resume_not_retry(self):  # L1
        from llm_eval import ladder
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "runs" / "acceptance-abc"
            run.mkdir(parents=True)
            (run / "state.json").write_text(json.dumps({"status": "refused_no_inhibit", "sections": {}}))
            self.assertEqual(ladder.blocker(tmp), ladder.REFUSED_NO_LOCK)
            (run / "state.json").write_text(json.dumps({"status": "partial", "sections": {}}))
            self.assertNotEqual(ladder.blocker(tmp), ladder.REFUSED_NO_LOCK)


class MainRefusalTests(unittest.TestCase):
    """`main` maps the refusal to REFUSED with an accurate message and the exact recovery command."""

    def setUp(self):
        from test_acceptance import AcceptanceTests
        self.base = AcceptanceTests("test_prepare_is_read_only_and_hands_back_a_token")
        self.base.setUp()
        self.addCleanup(self.base.tmp.cleanup)
        self.card, _ = self.base.prepare()
        self.dir, self.lock = self.base.dir, self.base.lock
        self.log = []
        real_run = acceptance.run

        def run_with_tmp_lock(*args, **kwargs):
            return real_run(*args, lock=self.lock, **kwargs)

        def fake_live(plan, log_dir, host=None, *, run_id):
            adapter = Recording(Path(log_dir).parent)
            adapter.log = self.log
            return adapter

        def no_real_systemctl(*args, **kwargs):
            raise AssertionError("a real systemctl call was made")

        for target in (patch.object(acceptance, "run", run_with_tmp_lock),
                       patch.object(acceptance, "LiveAdapter", fake_live),
                       patch.object(RealHost, "run", no_real_systemctl),
                       patch.object(inhibit, "sleep_inhibit", make_stub(self.log, self.runs_dir(), fail=True))):
            target.start()
            self.addCleanup(target.stop)

    def runs_dir(self):
        plan = json.loads((self.base.dir / "plan.json").read_text())
        return self.base.dir / "runs" / acceptance.run_id(plan)

    def main(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = acceptance.main(list(argv))
        return code, json.loads(out.getvalue())

    def test_run_is_refused_with_the_resume_command(self):
        code, result = self.main("run", "--dir", str(self.dir), "--approve", self.card["approval_token"])
        self.assertEqual(code, acceptance.REFUSED)
        text = result["refused"] + " " + result["next"]
        self.assertIn("The campaign was not started and production was not touched", text)
        self.assertIn(f"resume --dir {self.dir} --approve {self.card['approval_token']}", text)
        self.assertNotIn("Nothing was changed", text)
        self.assertEqual(self.log.count("quiesce"), 0)

    def test_a_second_plain_run_says_already_started_and_status_is_not_a_crash(self):
        self.main("run", "--dir", str(self.dir), "--approve", self.card["approval_token"])
        code, result = self.main("run", "--dir", str(self.dir), "--approve", self.card["approval_token"])
        self.assertEqual(code, acceptance.REFUSED)
        self.assertIn("already started", result["refused"])
        status, status_code = acceptance.status(self.dir, self.lock)
        self.assertEqual(status["status"], "refused_no_inhibit")
        self.assertEqual(status_code, acceptance.REFUSED)
        self.assertIn("resume", status["next"])

    def test_status_says_production_was_restored_when_a_restore_ran(self):  # L2
        self.main("run", "--dir", str(self.dir), "--approve", self.card["approval_token"])
        state_path = self.runs_dir() / "state.json"
        state = json.loads(state_path.read_text())
        self.assertIn("not touched", acceptance.status(self.dir, self.lock)[0]["next"])
        state["last_restore"] = {"service_matches": True}
        state_path.write_text(json.dumps(state))
        self.assertIn("restored first", acceptance.status(self.dir, self.lock)[0]["next"])

    def test_the_resume_path_says_production_was_restored_first(self):
        self.main("run", "--dir", str(self.dir), "--approve", self.card["approval_token"])
        code, result = self.main("resume", "--dir", str(self.dir), "--approve", self.card["approval_token"])
        self.assertEqual(code, acceptance.REFUSED)
        text = result["refused"] + " " + result["next"]
        self.assertIn("Production was restored first", text)
        self.assertIn("resume --dir", text)


if __name__ == "__main__":
    unittest.main()
