#!/usr/bin/env python3
"""Per-task repeat consistency for a run made with --runs N: the noise margin for comparing screens.

    python3 tools/flip_rate.py <data>/<plan folder>/runs/<id>   [--json]

Reads state.json. A task "flips" when its repeats disagree (some pass, some fail). Incomplete task-runs are left
out. Prints tasks with repeats, flipped tasks, the flip rate, and the pass count of each repeat number (r1, r2, ...)
whose range is the spread to use as a tie margin.
"""
import argparse
import json
from pathlib import Path
import re
import statistics

TRIAL = re.compile(r"__trial-wl-(?P<task>.+)-r(?P<n>\d+)$")


def analyse(sections):
    tasks, rounds = {}, {}
    for key, value in sections.items():
        match = TRIAL.search(key)
        if not match or value.get("incomplete") or value.get("status") not in ("pass", "fail", "quality_fail"):
            continue
        ok = value["status"] == "pass"
        tasks.setdefault(match["task"], {})[int(match["n"])] = ok
        rounds[int(match["n"])] = rounds.get(int(match["n"]), 0) + ok
    repeated = {t: r for t, r in tasks.items() if len(r) > 1}
    flipped = sorted(t for t, r in repeated.items() if len(set(r.values())) > 1)
    counts = [rounds[n] for n in sorted(rounds)]
    return {"tasks_repeated": len(repeated), "flipped": flipped,
            "flip_rate": round(len(flipped) / len(repeated), 3) if repeated else None,
            "passes_per_repeat": counts,
            "spread": max(counts) - min(counts) if counts else None,
            "stdev": round(statistics.pstdev(counts), 2) if len(counts) > 1 else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = analyse(json.loads((Path(args.run_dir) / "state.json").read_text())["sections"])
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    print(f"{result['tasks_repeated']} tasks with repeats; {len(result['flipped'])} flipped "
          f"(flip rate {result['flip_rate']})")
    print(f"passes per repeat: {result['passes_per_repeat']}; spread {result['spread']}, stdev {result['stdev']}")
    for task in result["flipped"]:
        print(f"  flipped: {task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
