"""Offline judge for what code checks cannot score: run after production is restored, never by
the runner (nothing in the runner imports this module, so the loopback path is untouched).

* Pairwise and blind: two runs of the same task (two profiles, or two runs) are shown as A and B
  in a seeded order, then again swapped; a winner counts only when both orders agree.
* Before a second profile exists, a run is scored against the task's reference (the real diff or
  a reference answer) on a 1-5 scale per criterion.
* The endpoint is any OpenAI-compatible chat API the owner names: ``--base-url``, ``--model``
  and ``--key-env`` (the name of the env var holding the credential, never its value).
* Every payload is scanned for likely secrets first and refused on a hit. Judge scores are
  written beside the gates in ``judge.json`` and never change a pass or fail.
* ``calibration`` writes a blind rating sheet for the owner; ``agreement`` compares the owner's
  ratings with the judge's.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.request

from .workload.snapshot import SECRETS

RUBRICS = {
    "implementation": ["correct beyond the tests", "fits the repo's style and idiom", "smallest change that does the job",
                       "final report is accurate about what changed"],
    "review": ["findings are real defects", "no invented or trivial findings", "each finding is actionable"],
    "advisory": ["reasoning grounded in the facts given", "recommendation is clear and justified",
                 "follows the owner's working agreement (lettered options, one recommended, concise)"],
    "resume": ["identifies where work stands", "next step is concrete and correct", "concise"],
    "lookup": ["answers the question directly", "accurate", "concise"],
    "ops": ["root cause stated correctly", "fix is safe and minimal", "report says what was verified"],
    "surgery": ["history and remotes handled safely", "end state matches the request", "report is accurate"],
}
DIFF_LIMIT = 30000


def evidence(output_dir, box_factory=None):
    """What one task-run left: the task prompt, final answer, the diff it made, and its metrics.

    The repo is the agent's, so git and file reads run inside the sandbox over the run's own home
    (repo config cannot run anything on the host, and a symlink cannot reach a real file)."""
    out = Path(output_dir)
    result = json.loads((out / "result.json").read_text())
    diff = ""
    repo = Path(result.get("repo", ""))
    if result.get("base_head") and result.get("home") and repo.exists():
        if box_factory is None:
            from .workload.sandbox import Box
            box_factory = Box
        box = box_factory(Path(result["home"]))
        _, diff = box.git(["diff", "--no-ext-diff", "--no-textconv", result["base_head"], "--", "."], repo)
        _, untracked = box.git(["ls-files", "--others", "--exclude-standard", "-z"], repo)
        for name in [n for n in untracked.split("\0") if n][:50]:
            code, text, _, _ = box.run(["head", "-c", "20000", "--", name], repo, 30)
            if code == 0:
                diff += f"\n--- new file {name}\n{text}"
    return {"task": result["task"], "category": result["category"], "answer": result.get("answer", ""),
            "diff": diff[:DIFF_LIMIT], "passed_checks": result["pass"]}


def refuse_secrets(text):
    hits = [rule for rule, pattern in SECRETS.items() if pattern.search(text)]
    if hits:
        raise ValueError(f"judge payload refused, likely secrets: {', '.join(hits)}")


def _ask(endpoint, system, user):
    refuse_secrets(user)
    body = json.dumps({"model": endpoint["model"], "temperature": 0, "stream": False,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(endpoint["key_env"]) if endpoint.get("key_env") else None
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(endpoint["base_url"].rstrip("/") + "/chat/completions", body, headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.loads(response.read())
    text = reply["choices"][0]["message"].get("content") or ""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("judge reply has no JSON object")
    return json.loads(text[start:end + 1])


def _describe(run, label):
    return (f"## Run {label}\n### Final answer\n{run['answer'] or '(none)'}\n"
            f"### Changes made\n```diff\n{run['diff'] or '(no file changes)'}\n```\n")


PAIR_SYSTEM = ("You compare two runs of the same task by local coding agents. Judge only against the "
               "criteria. Reply with JSON only: {\"criteria\": {criterion: \"A\"|\"B\"|\"tie\"}, "
               "\"overall\": \"A\"|\"B\"|\"tie\", \"reason\": short string}.")
REFERENCE_SYSTEM = ("You score one run of a task by a local coding agent against a reference. Judge only "
                    "against the criteria. Reply with JSON only: {\"scores\": {criterion: 1-5}, \"reason\": short string}.")


def compare(endpoint, task, first, second, seed=0):
    """Blind pairwise verdict for ``first`` vs ``second``; a win needs both orders to agree."""
    criteria = task.get("judge", {}).get("rubric") or RUBRICS[task["category"]]
    head = f"# Task\n{task['prompt']}\n\n# Criteria\n" + "\n".join(f"- {c}" for c in criteria) + "\n\n"
    order = random.Random(f"{seed}:{task['id']}").random() < 0.5
    verdicts = []
    for swapped in (order, not order):
        a, b = (second, first) if swapped else (first, second)
        reply = _ask(endpoint, PAIR_SYSTEM, head + _describe(a, "A") + _describe(b, "B"))
        mapping = {"A": "second" if swapped else "first", "B": "first" if swapped else "second", "tie": "tie"}
        verdicts.append({k: mapping.get(v, "tie") for k, v in reply.get("criteria", {}).items()}
                        | {"overall": mapping.get(reply.get("overall"), "tie")})
    agreed = {k: (verdicts[0][k] if verdicts[0].get(k) == verdicts[1].get(k) else "tie") for k in verdicts[0]}
    return {"task": task["id"], "verdict": agreed, "orders": verdicts}


def score(endpoint, task, run, reference):
    criteria = task.get("judge", {}).get("rubric") or RUBRICS[task["category"]]
    user = (f"# Task\n{task['prompt']}\n\n# Criteria\n" + "\n".join(f"- {c}" for c in criteria)
            + f"\n\n# Reference\n{reference[:DIFF_LIMIT]}\n\n" + _describe(run, "under review"))
    reply = _ask(endpoint, REFERENCE_SYSTEM, user)
    return {"task": task["id"], "scores": {c: reply.get("scores", {}).get(c) for c in criteria},
            "reason": reply.get("reason", "")}


def _runs(run_dir):
    """task id -> [output dirs], from one acceptance run's workload folder."""
    found = {}
    for result in sorted(Path(run_dir).glob("workload/*/result.json")):
        if ".aborted-" in result.parent.name:
            continue
        found.setdefault(json.loads(result.read_text())["task"], []).append(result.parent)
    return found


def calibration_sheet(pairs, path, seed=0):
    """A blind sheet for the owner: each pair shown as A/B; the key is kept in a separate file."""
    rng = random.Random(seed)
    sheet, key = ["# Calibration: pick the better run of each pair (A, B or tie)\n"], {}
    for index, (task, first, second) in enumerate(pairs, 1):
        swapped = rng.random() < 0.5
        a, b = (second, first) if swapped else (first, second)
        key[str(index)] = {"task": task["id"], "A": "second" if swapped else "first"}
        sheet.append(f"\n---\n\n## Pair {index}: {task['id']}\n\n{task['prompt']}\n\n"
                     + _describe(a, "A") + _describe(b, "B") + f"\nYour pick for pair {index}: \n")
    Path(path).write_text("".join(sheet))
    Path(path).with_suffix(".key.json").write_text(json.dumps(key, indent=2))
    return key


def agreement(owner, judge, key):
    """Share of pairs where the owner's A/B/tie pick matches the judge's agreed overall verdict."""
    hits = total = 0
    for index, pick in owner.items():
        entry = key[str(index)]
        mapped = {"A": entry["A"], "B": "second" if entry["A"] == "first" else "first", "tie": "tie"}[pick]
        verdict = judge.get(entry["task"])
        if verdict is not None:
            total += 1
            hits += mapped == verdict
    return {"pairs": total, "agree": hits, "rate": round(hits / total, 2) if total else None}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m llm_eval.judge")
    parser.add_argument("--run", required=True, help="an acceptance run folder (runs/acceptance-<token>)")
    parser.add_argument("--against", help="another profile's run folder, for pairwise judging")
    parser.add_argument("--tasks", required=True, help="folder of workload task files")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", help="NAME of the env var holding the API key")
    args = parser.parse_args(argv)
    endpoint = {"base_url": args.base_url, "model": args.model, "key_env": args.key_env}
    tasks = {json.loads(p.read_text())["id"]: json.loads(p.read_text()) for p in Path(args.tasks).glob("*.json")}
    mine, theirs = _runs(args.run), _runs(args.against) if args.against else {}
    results, skipped = judge_runs(endpoint, tasks, mine, theirs, Path(args.run).parents[1] / "snapshots")
    out = Path(args.run) / "judge.json"
    rubrics = {"base": RUBRICS, "tasks": {i: t["judge"]["rubric"] for i, t in tasks.items() if t.get("judge", {}).get("rubric")}}
    out.write_text(json.dumps({"endpoint": {"base_url": args.base_url, "model": args.model},
                               "against": Path(args.against).name if args.against else None, "judged_at": time.time(),
                               "rubric_sha": hashlib.sha256(json.dumps(rubrics, sort_keys=True).encode()).hexdigest(),
                               "results": results, "skipped": skipped}, indent=2))
    print(json.dumps({"judged": len(results), "skipped": len(skipped), "out": str(out)}))
    return 0


def reference_for(task, snapshot_dir):
    """The real diff when the task replays a commit; else an explicit judge reference; else the
    task's authored solution answer. Empty only when a task has none of these."""
    diff = Path(snapshot_dir) / "reference.diff"
    if diff.exists():
        return diff.read_text()
    return task.get("judge", {}).get("reference") or (task.get("solution") or {}).get("answer", "")


def judge_runs(endpoint, tasks, mine, theirs, snapshots, box_factory=None):
    """Judge every task-run found; return (results, skipped) so missing coverage is visible."""
    results, skipped = [], []
    for task_id, dirs in sorted(mine.items()):
        task = tasks.get(task_id)
        if task is None:
            skipped.append({"task": task_id, "reason": "task file not found"})
        elif theirs.get(task_id):
            results.append(compare(endpoint, task, evidence(dirs[0], box_factory), evidence(theirs[task_id][0], box_factory)))
        elif reference := reference_for(task, Path(snapshots) / task_id):
            results.append(score(endpoint, task, evidence(dirs[0], box_factory), reference))
        else:
            skipped.append({"task": task_id, "reason": "no reference and no second profile"})
    return results, skipped


if __name__ == "__main__":
    sys.exit(main())
