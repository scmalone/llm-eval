# 2026-11-06: A full review found the harness behind a third of the failures, and the fixes moved the scores

- Covers: 2026-11-06 20:09 to 21:34 UTC, sessions 37c105ae

## What happened
- After a `/compact` the owner switched to Opus at high effort and asked for "everything about results of the
  tests". The agent ran the review brief itself:
  - It read sections, answers, end states, server logs and telemetry.
  - It re-graded every saved core run.
  - It wrote `docs/reviews/2026-11-06-opus-full-review.md` (`5d9b281`).
- Main findings:
  - The advisory format check failed 19 of 50 trials whose content was correct.
  - The `impl-llm-eval-output-budget` hidden tests read data shapes its task text never stated.
  - A race in `driver.py` scored two finished 11 to 20 s trials as wall-clock timeouts.
  - Every model that acted on a question did so right after the reply "Use your judgment and continue".
  - North-mini's "infra_fail" was mostly the model. It replaced the real `llama-server` with a fake script that
    prints "Server started successfully" and sleeps forever, then ran it and hung for 600 s.
- The owner chose A for both decisions:
  - The format applies only when options are offered.
  - Question tasks reply "finish with your answer and change nothing".
- Then asked, in a message sent mid-turn: fix the driver and the other harness issues, fix the output-budget test,
  rescore the timeouts, fix the backup-branch provenance, queue but not run a Strata screen, run the Muse recall
  check, and re-run the devlog.
- The signal drill passed after the restore-code changes. Muse's recall re-check got 17 of 36 records, against 32 of
  36 on the same prompts the day before.

## What the owner challenged
- "I want to know everything about results of the tests, findings, any changes/notes in the tests etc."
- "Rescore the timeouts if they haven't been done already."
- "Fix the issue with the data being saved to a backup branch and move items to the main branch/repo." The data was
  never only on the backup branch. Main had every change, with different commit IDs after the trailer strip. The fix
  maps the 34 old IDs to main.
- "I'm also interested in more details on North and it's issues." Answered in a review addendum.

## What the agent got wrong
- Its review draft miscounted three things, and it corrected them before the commit:
  - Strata's changed task specs: 9, not the real 10.
  - Advisory em-dash failures: 23, not the real 21.
  - Tasks every model passes: 5, not the real 6.
- The operator's earlier 2026-11-06 log, written by an agent, said north-mini's failure cause was "not proven" and
  might be the parser. The review found the model ran a self-written fake server.
- The 2026-11-05 entry said 20 of 51 first-pass probes failed. The run file has 26 (corrected there).

## Changed
- `04bc946`:
  - The advisory option format is required only when options are offered.
  - The option parser is wider.
  - The output-budget task text now states the two shapes its hidden tests read.
- `fc80b36`:
  - The EOF race is fixed.
  - pi turn logs are captured.
  - A command the model ran that never returns is the model's timeout (`tool_hang`); the server is left alone.
  - The 20 question tasks get the no-change reply.
  - A recall miss at the target is a measured `quality_fail`.
  - lle-006 matcher widened.
- `0f05585`: a tools miss at the target is a measured result too.
- `0a5897f`: `configs/rewritten-shas.json` maps the rewritten commits; ingest stores main's IDs. 39 runs were
  re-ingested, with 99 runs and 628 task-runs before and after.
- `3ac5232`: model pages regenerated.
- `1e07360`: queued `recall-muse-recheck` and a held `screen-strata-rerun`.
- `d5a8fe4`, `8bf1fe3`: drill and Muse logs.
- `2ed8847`: STATE.

## Learned
- When a check fails almost every model, read the answers before reading the scores. Six of the 35 core tasks passed
  on no screened model, and five of those were the test or the grader.
- A harness reply is part of the experiment. "Continue" read as permission to delete files.
- Muse's 11-05 recall near-pass was a good draw. A single recall run says little at 112K tokens.

## Data
- Re-graded and rescored core:

  | Model | Before | After |
  |---|---|---|
  | Muse | 23 | 28 |
  | Swift | 23 | 26 |
  | Qwen3.8 | 24 | 25 |
  | Gemma | 17 | 20 |
  | Ornith | 13 | 14 |
  | Qwen3.6 | 14 | 14 |
  | Occamy | 12 | 12 |

  The top 3 is unchanged.
- Superseded: every core score as first recorded, and "Swift is not comparable".
- The 20 question tasks, output-budget and runner-resume have new identities, so core runs from here on do not compare
  with the screens on those tasks.
- Strata's 10-31 core (27 re-graded) used 10 older task specs; `screen-strata-rerun` waits for the owner.
- Output-budget could not be re-graded: its hidden tests run code the models wrote to the old task text.
- Ornith's rescored trial covers its first turn only; the second never ran.
