# 2026-11-02: `--suite accuracy` in the runner, and Strata's reference queued

## What happened

The owner asked for the accuracy suite's runner entry point, for Strata first, run unattended. `run_test_plan.py
--suite accuracy` swaps the test plan's phases 0 to 6 for one phase per part: `core`, `depth`, `documents`, `mined`,
`game`, `vision`. Each part is its own campaign, so a failed part is re-run alone with `--only`, the same way a plan
phase is. Knobs and comparisons are refused in this mode. `vision` is skipped, not failed, for a profile without
image input. The plan folder is `<date>-<profile>-<context>-accuracy`. `plan_ctl.py launch` passes `--suite` through
so it knows the folder before the unit starts.

`--check` prepared every part on Strata at 262,144 without refusal; vision skipped (Strata's pi entry has no image
input).

## Back and forth

- The owner asked whether Strata's earlier run still stood after the task changes. Yes for core, depth and
  documents: every suite or fixture commit after the 2026-11-01 17:56 re-run adds files and edits none of those tasks.
- Documents needed a closer look. The accuracy suite sizes them with each profile's tokenizer and labels the
  measurement `profile_tokenizer_raw_user_text`; Strata's phase 3 labelled it `templated_user_only`. The prompts and
  checks are byte-identical (all six compared), but `spec_sha` hashes the label, so the scorecard would treat them as
  different tasks. strata-reference re-runs documents (about 10 minutes) rather than special-casing the hash.
- strata-reference moved up to run straight after the drill, ahead of the sweeps and the owner's owner-adopt-resize stop, so the Strata work can
  run unattended in one sitting.

## Numbers

Harness estimates for strata-reference (opencode-derived band minutes, an upper guide): documents 18 min, mined 297 min, game
93 min. Strata's own pace on the 2026-10-31 core run (implementation tasks 8.4 min model and tool time on average)
puts mined and game nearer 3 to 4 h together.

## Sources

`tools/run_test_plan.py`, `tools/plan_ctl.py`, `tests/test_run_test_plan.py`; `QUEUE.md` row strata-reference;
`plans/2026-11-01-accuracy-suite.md` item 4.
