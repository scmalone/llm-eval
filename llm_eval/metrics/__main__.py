"""python3 -m llm_eval.metrics ingest|export|summary|score [--db PATH]

The default database is ``~/llm-eval-runs/metrics.sqlite``; acceptance runs ingest into
``metrics.sqlite`` beside their profile folder. Everything is derived from the run folders and
safe to rerun.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

from .ingest import find_runs, ingest_all
from .schema import TABLES, connect
from .scoring import score

from ..paths import DATA as DEFAULT_ROOT  # noqa: E402  (the data folder)


def default_db(root=DEFAULT_ROOT):
    return Path(root) / "metrics.sqlite"


def export(db_path, out_dir):
    """One JSONL file per table, for pandas, R or DuckDB (``COPY ... TO 'x.parquet'`` converts)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    written = {}
    try:
        for table in TABLES:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            with (out / f"{table}.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(dict(row), sort_keys=True) + "\n")
            written[table] = len(rows)
    finally:
        conn.close()
    return written


def summary(db_path):
    conn = connect(db_path)
    try:
        return {"tables": {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES},
                "runs": [dict(r) for r in conn.execute(
                    "SELECT run_id, suite_label, status, valid_for_analysis, invalid_reason FROM runs ORDER BY started_at")]}
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m llm_eval.metrics")
    parser.add_argument("--db", help="warehouse path (default: ~/llm-eval-runs/metrics.sqlite)")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--run", action="append", help="a run folder (runs/<id>); repeatable")
    ingest.add_argument("--root", default=str(DEFAULT_ROOT), help="ingest every run folder under this root")
    dump = commands.add_parser("export")
    dump.add_argument("--out", required=True)
    commands.add_parser("summary")
    scoring = commands.add_parser("score", help="version-pinned equal-task scores and baseline comparison")
    scoring.add_argument("--request", required=True, help="JSON cohort/suite/inference contract")
    args = parser.parse_args(argv)
    db = Path(args.db) if args.db else default_db(getattr(args, "root", DEFAULT_ROOT))
    if args.command == "ingest":
        runs = [Path(r) for r in args.run] if args.run else find_runs(args.root)
        print(json.dumps({"db": str(db), "ingested": ingest_all(db, runs)}, indent=2))
    elif args.command == "export":
        print(json.dumps({"db": str(db), "written": export(db, args.out)}, indent=2))
    elif args.command == "score":
        try:
            result = score(db, json.loads(Path(args.request).read_text()))
        except (OSError, ValueError, sqlite3.Error) as exc:
            parser.error(str(exc))
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print(json.dumps(summary(db), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
