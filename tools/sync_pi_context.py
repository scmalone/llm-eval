#!/usr/bin/env python3
"""Set pi's contextWindow for every local model to the context its llama.cpp preset loads.

    python3 tools/sync_pi_context.py           # show what would change; writes nothing
    python3 tools/sync_pi_context.py --write   # back up and update pi's models.json, then the repo's copy

The owner runs this (an agent's shell cannot read pi's agent directory, which holds pi's credentials). A model
whose window is larger than what the server loads compacts too late and its long requests fail, so after the speed
sweep shortens a preset's context, pi must follow. Only ``contextWindow`` changes, only for model ids that are
preset sections; nothing else in the file is read out or printed. The repo's copy (``configs/pi-production.json``)
gets the same numbers, so the benchmark's pre-run check (``run_test_plan.configure``) passes.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from llm_eval.preflight import DEFAULT_PRESET, read_presets  # noqa: E402

COPY = REPO / "configs" / "pi-production.json"


def agent_models(env=os.environ):
    return Path(env.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent") / "models.json"


def contexts(preset=DEFAULT_PRESET):
    _, presets = read_presets(preset)
    return {name: int(settings["ctx-size"]) for name, settings in presets.items() if settings.get("ctx-size")}


def plan(models, wanted):
    """[(provider, id, old, new)] for every model whose window differs from its preset's context."""
    changes = []
    for provider, block in (models.get("providers") or {}).items():
        for model in block.get("models") or []:
            new = wanted.get(model.get("id"))
            if new is not None and model.get("contextWindow") != new:
                changes.append((provider, model["id"], model.get("contextWindow"), new))
    return changes


def apply(models, changes):
    by_id = {(provider, model_id): new for provider, model_id, _, new in changes}
    for provider, block in (models.get("providers") or {}).items():
        for model in block.get("models") or []:
            if (provider, model.get("id")) in by_id:
                model["contextWindow"] = by_id[(provider, model["id"])]
    return models


def sync_copy(wanted, path=COPY):
    """The repo's copy, keyed by model id: the same windows. Returns the ids it changed."""
    copy = json.loads(path.read_text())
    changed = []
    for model_id, entry in copy["models"].items():
        new = wanted.get(model_id)
        if new is not None and entry.get("contextWindow") != new:
            entry["contextWindow"] = new
            changed.append(model_id)
    if changed:
        path.write_text(json.dumps(copy, indent=1) + "\n")
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="back up and write; without it, only show the changes")
    parser.add_argument("--models", type=Path, default=None, help="pi's models.json (default: the agent directory's)")
    args = parser.parse_args(argv)
    wanted = contexts()
    path = (args.models or agent_models()).resolve()
    text = path.read_text()
    models = json.loads(text)
    changes = plan(models, wanted)
    for provider, model_id, old, new in changes:
        print(f"{provider}/{model_id}: contextWindow {old} -> {new}")
    if not changes:
        print("pi already matches every preset's context")
    if not args.write:
        if changes:
            print("dry run: nothing written (add --write)")
        return 0
    if changes:
        backup = path.with_name(f"{path.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
        backup.write_text(text)
        os.chmod(backup, path.stat().st_mode & 0o777)
        path.write_text(json.dumps(apply(models, changes), indent=2) + "\n")
        print(f"wrote {path} (backup {backup.name}); commit it in its own repo if it is tracked there")
    changed = sync_copy(wanted, COPY)
    print(f"repo copy: {', '.join(changed)} updated; commit configs/pi-production.json" if changed
          else "repo copy already matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
