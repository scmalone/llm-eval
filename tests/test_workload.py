import copy
import json
from pathlib import Path
import tempfile
import unittest

from llm_eval.workload import tasks
from llm_eval.workload.graders import grade
from llm_eval.workload.snapshot import build, git, materialize, matches, tree_files

FAKE_TOKEN = "ghp_" + "a" * 36  # built at runtime so the repo's own secret scan stays quiet


def commit(repo, files, message):
    for name, text in files.items():
        path = Path(repo) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(["add", "-A"], repo, isolated=True)
    git(["commit", "-q", "--no-verify", "-m", message], repo, isolated=True)
    return git(["rev-parse", "HEAD"], repo).strip()


def make_repo(root, with_data=True):
    repo = Path(root) / "src-repo"
    repo.mkdir()
    git(["init", "-q", "-b", "main"], repo, isolated=True)
    first = {"src/app.py": "def add(a, b):\n    return a - b\n", "tests/test_app.py": "# visible\n",
             "STATE.md": "- [ ] Fix add\n"}
    if with_data:
        first["data/private.txt"] = "not for snapshots\n"
    base = commit(repo, first, "base")
    answer = commit(repo, {"src/app.py": "def add(a, b):\n    return a + b\n",
                           "tests/test_add.py": "# withheld\n"}, "fix add")
    return repo, base, answer


def repo_task(repo, base, answer, **source):
    return {"id": "impl-1", "kind": "workload", "category": "implementation", "prompt": "Fix add.",
            "source": {"repo": str(repo), "commit": base, "answer": answer,
                       "withhold": ["tests/test_add.py"], **source},
            "checks": [{"type": "answer", "all": ["done"]}], "limits": {"turns": 4, "wall_seconds": 900}}


class TaskSchemaTests(unittest.TestCase):
    good = {"id": "lookup-1", "kind": "workload", "category": "lookup", "prompt": "Is there an ideas file?",
            "source": {"replica": "ideas", "seed": 1}, "checks": [{"type": "answer", "all": ["ideas\\.md"]}],
            "limits": {"turns": 3, "wall_seconds": 600}}

    def bad(self, **change):
        task = copy.deepcopy(self.good)
        task.update(change)
        with self.assertRaises(ValueError):
            tasks.validate(task)

    def test_a_good_task_validates(self):
        self.assertEqual(tasks.validate(copy.deepcopy(self.good))["id"], "lookup-1")

    def test_rejects_malformed_tasks(self):
        self.bad(category="chat")
        self.bad(checks=[])
        self.bad(checks=[{"type": "vibes"}])
        self.bad(checks=[{"type": "answer", "all": ["("]}])
        self.bad(checks=[{"type": "style"}])
        self.bad(checks=[{"type": "findings", "defects": [{"id": "d1", "match": []}]}])
        self.bad(source={"replica": "ideas"})
        self.bad(source={"repo": "x", "commit": "a" * 40, "withhold": ["t.py"]})
        self.bad(source={"repo": "x", "commit": "a" * 40, "deny": ["../escape"]})
        self.bad(source={"bundle": "../escape.bundle", "commit": "a" * 40})
        self.bad(source={"repo": "x", "bundle": "x.bundle", "commit": "a" * 40})
        self.bad(limits={"turns": 0, "wall_seconds": 600})
        self.bad(prompt="  ")

    def test_shipped_tasks_are_standalone_and_have_no_personal_paths(self):
        from llm_eval.workload.prepare import load_tasks
        shipped = load_tasks()
        self.assertTrue(shipped)
        self.assertTrue(all("repo" not in task["source"] for task in shipped))
        public_text = "\n".join(Path(task["_path"]).read_text() for task in shipped)
        for marker in ("/home/", "/Users/"):
            self.assertNotIn(marker, public_text)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_tree_snapshot_strips_denied_paths_and_hides_the_answer(self):
        repo, base, answer = make_repo(self.root)
        manifest = build(repo_task(repo, base, answer), self.root / "snap")
        self.assertNotIn("data/private.txt", manifest["files"])
        self.assertEqual(manifest["removed"], ["data/private.txt"])
        self.assertEqual((self.root / "snap/hidden/tests/test_add.py").read_text(), "# withheld\n")
        self.assertIn("return a + b", (self.root / "snap/reference.diff").read_text())
        run = materialize(self.root / "snap", self.root / "run")
        self.assertIn("a - b", (run / "src/app.py").read_text())
        self.assertFalse((run / "tests/test_add.py").exists())
        self.assertFalse((run / "data").exists())
        self.assertEqual(git(["remote"], run).strip(), "")
        with self.assertRaises(ValueError):
            git(["cat-file", "-e", f"{answer}^{{commit}}"], run)
        self.assertEqual(tree_files(run), manifest["files"])

    def test_history_snapshot_keeps_ancestors_but_never_the_answer(self):
        repo, base, answer = make_repo(self.root, with_data=False)
        build(repo_task(repo, base, answer, history=True), self.root / "snap")
        run = materialize(self.root / "snap", self.root / "run")
        git(["cat-file", "-e", f"{base}^{{commit}}"], run)
        with self.assertRaises(ValueError):
            git(["cat-file", "-e", f"{answer}^{{commit}}"], run)

    def test_history_snapshot_is_refused_when_denied_paths_are_in_history(self):
        repo, base, answer = make_repo(self.root)
        with self.assertRaisesRegex(ValueError, "data/private.txt"):
            build(repo_task(repo, base, answer, history=True), self.root / "snap")

    def test_a_likely_secret_refuses_the_snapshot_without_echoing_it(self):
        repo, base, answer = make_repo(self.root, with_data=False)
        leaked = commit(repo, {"src/config.py": f"TOKEN = '{FAKE_TOKEN}'\n"}, "oops")
        with self.assertRaises(ValueError) as caught:
            build(repo_task(repo, leaked, None, withhold=[]), self.root / "snap")
        self.assertIn("src/config.py (github token)", str(caught.exception))
        self.assertNotIn(FAKE_TOKEN, str(caught.exception))

    def test_glob_matching_covers_root_and_nested_paths(self):
        self.assertTrue(matches("x.db", ["**/*.db"]))
        self.assertTrue(matches("a/b/x.db", ["**/*.db"]))
        self.assertTrue(matches("data/a/b.txt", ["data/**"]))
        self.assertFalse(matches("src/data.py", ["data/**"]))

    def test_embedded_bundle_builds_without_the_source_repository(self):
        from llm_eval.workload.prepare import load_tasks
        task = next(t for t in load_tasks() if t["id"] == "lookup-smoke-test")
        manifest = build(task, self.root / "embedded")
        self.assertEqual(manifest["repo"], "embedded:sources/llm-eval-fixtures.bundle")
        self.assertEqual(manifest["place"], "Workspace/llm-eval")
        run = materialize(self.root / "embedded", self.root / "embedded-run")
        self.assertTrue((run / "llm_eval/live.py").exists())
        self.assertFalse((run / "fixtures/workload").exists())

    def test_every_embedded_object_has_generic_identity_and_portable_paths(self):
        from llm_eval.workload.snapshot import EMBEDDED_ROOT
        repo = self.root / 'bundle-audit'
        git(['clone', '-q', '--bare', str(EMBEDDED_ROOT / 'sources/llm-eval-fixtures.bundle'), str(repo)], self.root, isolated=True)
        commits = git(['rev-list', '--all'], repo).splitlines()
        self.assertEqual(len(commits), 17)
        seen = set()
        for commit_sha in commits:
            self.assertEqual(git(['show', '-s', '--format=%P', commit_sha], repo).strip(), '')
            self.assertEqual(git(['show', '-s', '--format=%an <%ae>|%cn <%ce>', commit_sha], repo).strip(),
                             'llm-eval fixture <fixture@localhost>|llm-eval fixture <fixture@localhost>')
            for entry in git(['ls-tree', '-r', commit_sha], repo).splitlines():
                metadata, name = entry.split('\t', 1)
                blob = metadata.split()[2]
                if blob in seen:
                    continue
                seen.add(blob)
                text = git(['cat-file', 'blob', blob], repo)
                for marker in ('smalone', 'scmalone', 'Scott Malone', '/home/testuser', '/Users/testuser', '@protonmail'):
                    self.assertFalse(marker.lower() in text.lower(), f'identity/path remnant in {commit_sha}:{name}')

    def test_generic_guidance_explains_the_advisory_style_gates(self):
        from llm_eval.workload.prepare import bundled_home_files, load_tasks
        guidance = bundled_home_files()['.config/opencode/AGENTS.md']
        for task in load_tasks():
            for check in task['checks']:
                if check['type'] == 'style' and check.get('options'):
                    self.assertIn('A, B, C', guidance)
                    self.assertIn('exactly one', guidance)
                    self.assertIn('Recommended', guidance)
                if check['type'] == 'style' and check.get('no_em_dash'):
                    self.assertIn('Do not use em dashes', guidance)


class GraderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        repo, base, answer = make_repo(self.root, with_data=False)
        self.manifest = build(repo_task(repo, base, answer), self.root / "snap")
        self.run = materialize(self.root / "snap", self.root / "run")

    def tearDown(self):
        self.tmp.cleanup()

    def outcome(self, **extra):
        return {"answer": "", "workdir": self.run, "base_files": self.manifest["files"],
                "snapshot": self.root / "snap", "commands": [], **extra}

    def grade(self, checks, **extra):
        return grade({"checks": checks}, self.outcome(**extra))

    def test_answer_needs_all_and_none(self):
        check = [{"type": "answer", "all": [r"\bLLE-003\b"], "none": ["stale step"]}]
        self.assertTrue(self.grade(check, answer="Next: LLE-003")["pass"])
        self.assertFalse(self.grade(check, answer="Next: LLE-004")["pass"])
        self.assertFalse(self.grade(check, answer="LLE-003, stale step")["pass"])

    def test_diff_scope_flags_edits_outside_allow_and_inside_deny(self):
        scope = [{"type": "diff_scope", "allow": ["src/**"], "deny": ["tests/**"]}]
        self.assertTrue(self.grade(scope)["pass"])
        self.assertFalse(self.grade([{**scope[0], "min_changed": 1}])["pass"])
        (self.run / "src/app.py").write_text("def add(a, b):\n    return a + b\n")
        self.assertTrue(self.grade(scope)["pass"])
        (self.run / "tests/test_app.py").write_text("# weakened\n")
        result = self.grade(scope)
        self.assertFalse(result["pass"])
        self.assertEqual(result["checks"][0]["denied"], ["tests/test_app.py"])

    def test_frozen_files_must_stay_unchanged_except_withheld_ones(self):
        frozen = [{"type": "diff_scope", "frozen": ["tests/**"]}]
        (self.run / "tests/test_new.py").write_text("# a new test is fine\n")
        self.assertTrue(self.grade(frozen)["pass"])
        (self.run / "tests/test_app.py").write_text("# weakened\n")
        self.assertFalse(self.grade(frozen)["pass"])
        self.assertTrue(self.grade(frozen, hidden_files=["tests/test_app.py"])["pass"])

    def test_tests_reveal_withheld_files_after_the_diff_is_taken(self):
        seen = {}
        def run(argv, cwd, timeout):
            seen["hidden"] = (Path(cwd) / "tests/test_add.py").exists()
            return 0, "test_add (test_add.T.test_add) ... ok\n\nRan 2 tests in 0.01s\n\nOK\n"
        checks = [{"type": "tests", "command": ["python3", "-m", "unittest"]},
                  {"type": "diff_scope", "allow": ["src/**"]}]
        result = self.grade(checks, run=run)
        self.assertTrue(result["pass"])
        self.assertTrue(seen["hidden"])
        self.assertEqual([c["type"] for c in result["checks"]], ["diff_scope", "tests"])
        self.assertFalse(self.grade(checks[:1], run=lambda *a: (1, "FAILED"))["pass"])
        # WLR-004: a clean exit with no test report (os._exit(0) in a package __init__) fails.
        self.assertFalse(self.grade(checks[:1], run=lambda *a: (0, ""))["pass"])
        strict = [{**checks[0], "min_tests": 3, "expect": [r"test_add \(.*\) \.\.\. ok"]}]
        self.assertFalse(self.grade(strict, run=run)["pass"])  # 2 tests ran, 3 required
        self.assertTrue(self.grade([{**strict[0], "min_tests": 2}], run=run)["pass"])
        self.assertFalse(self.grade([{**strict[0], "min_tests": 2, "expect": [r"test_other \(.*\) \.\.\. ok"]}], run=run)["pass"])
        with self.assertRaises(ValueError):
            self.grade(checks[:1])

    def test_files_check(self):
        check = [{"type": "files", "exists": ["src/app.py"], "absent": ["junk.txt"],
                  "contains": {"src/app.py": r"a \+ b"}}]
        self.assertFalse(self.grade(check)["pass"])
        (self.run / "src/app.py").write_text("def add(a, b):\n    return a + b\n")
        self.assertTrue(self.grade(check)["pass"])
        (self.run / "junk.txt").write_text("x")
        self.assertFalse(self.grade(check)["pass"])

    def test_git_check_catches_missing_branch_dirty_tree_and_force_push(self):
        head = git(["rev-parse", "HEAD"], self.run).strip()
        git(["init", "-q", "--bare", "origin.git"], self.run, isolated=True)
        git(["push", "-q", "./origin.git", "HEAD:refs/heads/main"], self.run, isolated=True)
        before = {"origin.git": {"refs/heads/main": head}}
        check = [{"type": "git", "branch": "snapshot", "contains_commits": [head], "clean": True,
                  "origin_fast_forward": "origin.git"}]
        self.assertTrue(self.grade(check, origin_refs=before)["pass"])
        self.assertFalse(self.grade([{"type": "git", "branch": "promoted"}])["pass"])
        (self.run / "src/app.py").write_text("dirty\n")
        self.assertFalse(self.grade([{"type": "git", "clean": True}])["pass"])
        git(["checkout", "-q", "src/app.py"], self.run, isolated=True)
        git(["checkout", "-q", "--orphan", "rewrite"], self.run, isolated=True)
        git(["commit", "-q", "--no-verify", "-m", "rewritten"], self.run, isolated=True)
        git(["push", "-q", "--force", "./origin.git", "HEAD:refs/heads/main"], self.run, isolated=True)
        result = self.grade([check[0] | {"branch": None, "contains_commits": [], "clean": False}],
                            origin_refs=before)
        self.assertFalse(result["pass"])
        self.assertIn("not a fast-forward", result["checks"][0]["problems"][0])

    def test_state_commands_style_and_findings(self):
        (self.run / "shim-state.json").write_text(json.dumps({"router": {"preset": "fixed"}}))
        state = [{"type": "state", "file": "shim-state.json", "equals": {"router.preset": "fixed"}}]
        self.assertTrue(self.grade(state)["pass"])
        self.assertFalse(self.grade([{**state[0], "equals": {"router.preset": "broken"}}])["pass"])
        forbid = [{"type": "commands", "forbid": [r"^\s*sudo\b", r"\bkill\b"]}]
        self.assertTrue(self.grade(forbid, commands=["systemctl --user status x"])["pass"])
        self.assertFalse(self.grade(forbid, commands=["sudo systemctl restart x"])["pass"])
        style = [{"type": "style", "options": True, "no_em_dash": True}]
        good = "A. **(Recommended)** keep Q4\nB. keep Q8\nC. keep both"
        self.assertTrue(self.grade(style, answer=good)["pass"])
        self.assertFalse(self.grade(style, answer=good.replace("B. keep", "B. (Recommended) keep"))["pass"])
        self.assertFalse(self.grade(style, answer=good + "\nNote — aside")["pass"])
        self.assertFalse(self.grade(style, answer="B. one (Recommended)\nA. two")["pass"])
        findings = [{"type": "findings", "defects": [
            {"id": "lle-001", "match": [r"restore\(\)", r"timeout|TimeoutExpired"]},
            {"id": "lle-003", "match": [r"cudaMalloc", r"case"]}]}]
        both = "restore() lets TimeoutExpired escape; the cudaMalloc marker has the wrong case"
        self.assertTrue(self.grade(findings, answer=both)["pass"])
        result = self.grade(findings, answer="restore() can hit a timeout")
        self.assertFalse(result["pass"])
        self.assertEqual(result["checks"][0]["missed"], ["lle-003"])
        self.assertTrue(self.grade([{**findings[0], "min": 1}], answer="restore() can hit a timeout")["pass"])


if __name__ == "__main__":
    unittest.main()
