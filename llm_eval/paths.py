"""Where benchmark data lives: one folder outside git (runs, sweeps, mining, task sources and environments,
metrics.sqlite, drills). ``LLM_EVAL_DATA`` overrides it. ``~/Workspace/strata-eval-runs`` is a symlink to the
default, so absolute paths recorded before 2026-11-01 still resolve."""
import os
from pathlib import Path

DATA = Path(os.environ.get("LLM_EVAL_DATA") or Path.home() / "Workspace" / "llm-eval-data").expanduser()
