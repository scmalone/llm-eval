# 2026-11-06: Muse's full suite ran, and the vision phase graded nothing until the same missing-group bug was fixed a second time

- Covers: 2026-11-06 21:34 to 2026-11-07 01:35 UTC, sessions 37c105ae, 7e85e259

## What happened
- The owner asked how long `noise-accuracy` would take and whether the three top models could be told apart.
  - Estimates (per session): about 3 h on Muse, 5.5 h on Qwen3.8, 6 h on Swift.
  - The agent said accuracy alone could not split them: single runs, and 22 of the 35 core tasks had changed that
    day, so old and new scores do not compare. It offered A (core twice more on each of the three, about 9.5 h),
    B (`noise-accuracy` on Muse only) and C (go straight to the full suites).
  - The owner chose C, asked for a time estimate first (12 to 15 h for all three, per session), then said
    "Just do muse for now".
- `full-muse-30b` was queued and launched (`f251443`, `d71d26d`). Results in
  `docs/operator-log/2026-11-06-full-muse-30b.md`:
  - depth 3 of 4 measured (the fifth task reached 32,574 of its 36,000-token floor, so it is unmeasured)
  - documents 4 of 6 (Strata 6 of 6), mined 16 of 20 (Strata 18 of 20), game 5 of 6 (Strata 6 of 6)
  - vision graded nothing: "no workload task was graded"
- The vision fault was the one mined and game had on 2026-11-04. The 10 vision tasks had no workload group, so
  the runner dropped them. `9b4d3fb` adds the group. `runner.py` is restore code, so the signal drill ran again
  and passed (`dde6c24`), then vision ran alone: 9 of 10, the miss a misread chart peak
  (`vision-ocr-chart`, "May 583" for "May 563"). Muse's suite is complete (`1fae686`).
- Then the owner asked for the pi models commit and a push. The push went out (20 commits, all llm-eval). The
  pi commit was blocked by the credential-path hook, so the owner made it; dotfiles then needed a rebase onto 10
  upstream commits. The owner also removed firstmate (deleted on GitHub) from `PROJECTS.md` and the disk.
  None of that is in this repo.

## What the owner challenged
- "I'll do C but before doing that estimate how much time it will take." The estimate scaled Strata's phase
  times by each model's core time, with vision a guess because no model had run it.
- "Just do muse for now": the three-model plan shrank to one.
- "what is night 1?" The agent had written "Night 1: Muse then Qwen3.8" in its own estimate (per session) and
  then answered that the term appeared nowhere in the repo. Its search covered the repo, not its own reply.

## What the agent got wrong
- The `test_every_runnable_task_has_a_workload_group` test exempted category `vision` on the belief that vision
  runs in its own section. That section runs only the picture fixtures, so a whole phase was dropped for a second
  time (operator log, `Problems`). The test now covers every workload task and fails without the fix.
- It told the owner "night 1" was not a term in the repo when it had used the phrase itself (per session).

## Changed
- `9b4d3fb`: `vision` added to `WORKLOAD_GROUPS`; the group test no longer exempts it. 669 tests pass.
- `f251443`, `d71d26d`, `a48a35f`: `full-muse-30b` row added, running, then partial.
- `dde6c24`: drill passed after `9b4d3fb`.
- `1fae686`: `full-muse-30b` done, operator log and `STATE.md` updated.

## What worked
- The per-phase watch on a long plan: the vision phase failed in 1 minute, so the fault showed up as soon as the
  phase started, not in a morning summary.
- Re-running only `--only vision` after the drill, rather than the whole suite.

## Process note
- The same fault class hit twice in three days because the guard test carried a hand-written exemption. A guard
  that enumerates what it covers (here, every `kind == "workload"` task) cannot go stale the way an exemption list
  does. Writing the test that way on 2026-11-04 would have kept vision from being dropped.

## Learned
- Muse's long-context weakness shows beyond the recall test: both document misses are retrieval of a fact that
  was in the prompt (K-5760 at 32K, K-8863 at 100K).
- Two of Muse's four mined misses (`mined-strata-cut-tool-call`, `mined-thermalright-source-isolation`) are
  tasks Strata also failed.

## Data
- Standing: Muse full suite, depth 3/4, documents 4/6, mined 16/20, game 5/6, vision 9/10, from the `-depth`,
  `-documents`, `-mined`, `-game` and `-vision-attempt2` folders beside
  `~/Workspace/llm-eval-data/2026-11-06-muse-glimmer-30b-131072-accuracy`.
- Superseded: the first vision result (`assert_fail`, nothing graded) is a harness fault, not a measurement.
- Not done: Qwen3.8 and Swift full suites, `noise-accuracy` and `screen-strata-rerun` wait on the owner.
