import argparse
import json
from pathlib import Path
import sys
from .htmlreport import write as write_html
from .plan import compile_plan
from .runner import execute
from .state import atomic_json
from .preflight import DEFAULT_PRESET, eval_manifest, inspect_machine, retained_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="llm-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("manifest")
    plan.add_argument("--out", required=True)
    run = commands.add_parser("run")
    run.add_argument("plan")
    run.add_argument("--run-id", required=True)
    run.add_argument("--run-root", required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("--run-root", required=True)
    for name in ("preflight", "retained-manifest", "eval-manifest"):
        command = commands.add_parser(name)
        command.add_argument("--preset", default=str(DEFAULT_PRESET))
        command.add_argument("--unit", default="llama-router.service")
        command.add_argument("--out", required=True)
        if name == "preflight":
            command.add_argument("--health-url", default="http://127.0.0.1:8080/health")
        else:
            command.add_argument("--restore-model", required=True)
            if name == "eval-manifest":
                command.add_argument("--profiles", help="comma-separated subset; default is every distinct profile")
    args = parser.parse_args(argv)
    try:
        if args.command in ("preflight", "retained-manifest", "eval-manifest"):
            if args.command == "preflight":
                result = inspect_machine(args.preset, args.unit, args.health_url)
            elif args.command == "eval-manifest":
                subset = [p for p in (args.profiles or "").split(",") if p]
                result = eval_manifest(args.preset, args.unit, args.restore_model, subset or None)
            else:
                result = retained_manifest(args.preset, args.unit, args.restore_model)
            atomic_json(args.out, result)
            print(json.dumps({"output": str(Path(args.out).resolve()),
                              "live_execution_ready": False}))
            return 1 if args.command == "preflight" and (result["errors"] or not result["router_healthy"]) else 0
        if args.command == "plan":
            result = compile_plan(args.manifest)
            atomic_json(args.out, result)
            print(json.dumps({"plan": str(Path(args.out).resolve()), "hash": result["plan_hash"],
                              "backend": result["backend"], "models": len(result["models"])}))
            return 0
        path = Path(args.plan) if args.command == "run" else Path(args.run_root) / args.run_id / "plan.json"
        loaded = json.loads(path.read_text())
        result = execute(loaded, args.run_root, args.run_id, args.command == "resume")
        try:
            write_html(Path(args.run_root) / args.run_id, loaded, result)
        except OSError:
            pass
        failures = [k for k, v in result["sections"].items() if v["status"] not in ("pass", "skipped", "capacity_fallback")]
        print(json.dumps({"status": result["status"], "restoration_required": result["restoration_required"],
                          "failed_sections": failures}))
        return 0 if result["status"] == "complete" and not failures else 1
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"llm-eval: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
