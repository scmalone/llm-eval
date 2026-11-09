# 2026-11-04: Strata would not launch on a normal desktop, and its mined and game phases graded nothing

- Covers: 2026-11-04 18:09 to 2026-11-05 04:06 UTC, sessions 37c105ae

## What happened
- The grader for `advisory-lle002-option` required the literal text `LLE-002`, so a correct answer failed. It now checks for
  transient or systemd plus option A. The qwen3.8-27b screen re-graded from 22/35 to 23/35. Strata attempt 2 still fails that
  task on style (`fixtures/workload/advisory-lle002-option.json`).
- `strata-reference` could not launch. Available RAM was about 1,000 MiB under the 57,014 MiB Strata floor. The launch check
  ran before the plan stopped the router, but the floor was measured after the stop, so the router's own memory was counted
  against it. The launch checks now add what the router returns when stopped (`llm_eval/ramcheck.py`).
- That edit, and a later one to `runner.py`, are restore code, so each voided the signal drill. The drill ran three times in
  all and passed the last two (first, second).
- `strata-reference` then ran. `documents` passed 6/6. `mined` and `game` graded nothing ("no workload task was graded"):
  they were in no workload group, so the runner dropped every task. After the fix, `mined` ended 18/20 and `game` 6/6
  (log).
- The first scored `mined` pass stopped at a Strata server stall on `mined-thermalright-frame-sender` (57.3 GB peak memory,
  5.5 GB swap). The runner's one fresh re-prepare passed it. The cause is not established and is logged for Opus.
- The owner asked for an update after every phase. `watch --each-phase` is now in STATE.md Gotchas.

## What the owner challenged
- "Reassess the ram situation and provide solutions." The first answer was to quit Brave and stop the router; the
  reassessment found the launch check was the problem.
- "Ok - since strata is the only problem, we don't need to change anything. I'll keep the desktop empty for the benchmarks."
  The owner declined a desktop allowance for the benchmarks. llama.cpp has RAM to spare and is limited by VRAM; Strata
  needs a near-empty desktop, which stays a finding about Strata.

## What the agent got wrong
- It estimated the router returns 1.6 to 1.9 GiB. Measured from the router's cgroup it returns about 715 MiB (anonymous plus
  shared); the rest is page cache that available RAM already counts. The first fix still left the drill about 420 MiB short
  (per session).
- It launched `strata-reference` with `mined` and `game` in no workload group, so two phases ran and graded nothing. The
  `test_every_runnable_task_has_a_workload_group` test now catches this.

## Changed
- `c6b3cd1` advisory-lle002-option check and STATE.
- `39205d4` launch gate credits the router's memory (`ramcheck.py`, two tests).
- `62b7cdb` STATE: update after every phase.
- `c5f54f6` mined and game added to `WORKLOAD_GROUPS`, with the test.
- Operator logs: `ace903c`, `7aba303`, `d6138f9`, `5b11348`.

## Learned
- A gate floor is only valid at the moment it was measured. This floor was taken after the router stop, so it has to be
  checked there, or the earlier check has to credit the stop.
- A phase that grades zero tasks is a harness fault, and the verdict says so. It is not a model result.

## Data
- Stands: `documents` 6/6, `mined` 18/20 (fails: `mined-strata-cut-tool-call`, `mined-thermalright-source-isolation`), `game` 6/6.
- Superseded: the first `mined` and `game` phases (0 of 0 graded).
- The RAM floor of 57,014 MiB was not lowered. Past telemetry shows one run dipping to 2.5 GB and surviving, and one
  systemd-oomd kill at 2.3 GB (per session).
