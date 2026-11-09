# AGENTS.md

Project rules for any agent. Global rules: `~/Workspace/dotfiles/agents/AGENTS.md`.

## Start here
- Resume from `STATE.md` when present; why things are this way: `DECISIONS.md`.
- Running, continuing or checking benchmarks ("continue", "run X", "status"): follow `RUNCARD.md`. The queue
  is `QUEUE.md`; what each run did is in `docs/operator-log/`.

## Rules
- Mode is an explicit, recorded axis (opencode entry, loaded context, output
  limit, compaction point): pin it in the plan, print it on the acceptance
  card, label results with it, never compare across modes. Never default to
  the `-agent` (65K) entry; ask with lettered options instead.
- Capture wide: keep raw artifacts (untruncated streams, stderr, workspace end
  state, diffs, server logs, system samples) beside derived metrics. Runs
  cannot be replayed; prune a source only on evidence it goes unused. Run
  folders are the source of truth; the SQLite warehouse is derived.
