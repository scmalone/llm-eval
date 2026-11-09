# devlog

The story of building this benchmark: what changed, what the owner challenged, what the agent got wrong, what was
learned, and the numbers. One entry per milestone, newest last. `DECISIONS.md` holds the rules; this folder is the
history the rules leave out.

Each entry covers what happened, the back and forth (owner pushback, agent mistakes), what was learned, the numbers,
and the sources (commits, plans, transcripts). Commit IDs refer to the private development repo, and run folders,
transcripts and the metrics database stay private; this public copy has one commit.

| File | Date | Summary |
|---|---|---|
| [2026-10-02-to-10-31-timeline.md](2026-10-02-to-10-31-timeline.md) | 2026-11-01 | Backfilled from git: the month from first runner to the unattended plan, by phase |
| [2026-11-01-strata-plan-and-grader-audit.md](2026-11-01-strata-plan-and-grader-audit.md) | 2026-11-01 | First full Strata plan (10.5 h), the grader audit it forced, the re-run, and Strata's corrected scores |
| [2026-11-01-owner-runs-config-pass.md](2026-11-01-owner-runs-config-pass.md) | 2026-11-01 | Drill, config sweep and Strata changed-task re-run; adopted settings; 1 GiB free stays the pass/fail mark |
| [2026-11-01-sizing-policy-and-harness-fixes.md](2026-11-01-sizing-policy-and-harness-fixes.md) | 2026-11-01 | KV q8_0 default and the step-down order, the 1 GiB task-home cap, version-split scorecard, pi context sync, a draft check suite |
| [2026-11-01-accuracy-suite-tasks.md](2026-11-01-accuracy-suite-tasks.md) | 2026-11-01 | 20 mined fixes, 6 game tasks (flight-test, Blender), 10 vision tasks, documents sized per tokenizer, read-only task environments; red-team fixes (skips fail, envs pinned by digest) |
| [2026-11-02-strata-ram-gate.md](2026-11-02-strata-ram-gate.md) | 2026-11-02 | Drill killed by systemd-oomd (Strata load short of RAM); Strata loads now refuse below model size plus 2 GiB available |
| [2026-11-02-accuracy-entry-point.md](2026-11-02-accuracy-entry-point.md) | 2026-11-02 | `run_test_plan.py --suite accuracy`, one phase per part; documents identity label; Strata reference (strata-reference) queued after the drill |
| [2026-11-04-like-for-like-settings.md](2026-11-04-like-for-like-settings.md) | 2026-11-04 | Qwen3.6 screened on coding sampling; every model set to its vendor general-thinking default; 16K thinking cap kept and counted |
| [2026-11-04-ram-gate-and-mined-game-groups.md](2026-11-04-ram-gate-and-mined-game-groups.md) | 2026-11-04 | Strata launch gate credits the router memory; mined and game phases graded nothing until added to the workload groups; strata-reference 6/6, 18/20, 6/6 |
| [2026-11-05-sweep-resize-and-adoption.md](2026-11-05-sweep-resize-and-adoption.md) | 2026-11-05 | Resize sweep redone on a clean desktop; probes stop at the floor; MoE offload probe; eight preset edits adopted |
| [2026-11-06-screens-and-swift-step-downs.md](2026-11-06-screens-and-swift-step-downs.md) | 2026-11-06 | Six screens done (core 12 to 24 of 35), Swift and north-mini blocked; run card follows STATE.md |
| [2026-11-04-transcripts-and-devlog-tooling.md](2026-11-04-transcripts-and-devlog-tooling.md) | 2026-11-04 | Transcripts copied before the 30-day cleanup, condensed and kept in a local-only repo; global /devlog skill, Codex capture, optional owner notes |
| [2026-11-06-swift-fits-and-scores.md](2026-11-06-swift-fits-and-scores.md) | 2026-11-06 | Swift loads with MTP off and the projector in RAM; screen 23/35, recall passed; Opus review prompt written |
| [2026-11-06-full-review-and-fixes.md](2026-11-06-full-review-and-fixes.md) | 2026-11-06 | Full review: advisory grader, hidden-test shapes, a timeout race and the default reply behind many failures; fixes, re-grade (top 3 unchanged), Muse recall 17/36 |
| [2026-11-06-muse-full-suite-and-vision-group.md](2026-11-06-muse-full-suite-and-vision-group.md) | 2026-11-06 | Muse full accuracy suite (depth 3/4, documents 4/6, mined 16/20, game 5/6, vision 9/10); vision dropped by a missing workload group a second time, fixed with the drill re-run |
