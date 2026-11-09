# 2026-11-01: The first full Strata plan, and the grader audit it forced

## What happened

The unattended plan ran Strata (Qwen3.8 Flash Next IQ3_S on the Strata engine, 262,144 context, through the
owner's pi) from 2026-10-31 18:54 to 11-01 04:22, 10.5 h, and restored production healthy. The raw scores looked
poor in places (advisory 0/5, documents 2/6, depth 2/5), so every failure was read against the model's full
answer and end state before believing it. Most low scores were the graders, not the model.

## Back and forth

- Owner decisions this round: drop the like-for-like 131K table and compare on per-model metrics; treat Strata as
  one more model; accept `**A.**`-style options (A); state the hidden requirements in the packets and re-run (A);
  record overreach as its own metric rather than a failure (C).
- Agent mistakes, caught and corrected:
  - Reported "production is down" from a `plan_ctl report` run inside the sandbox, which cannot reach the user
    bus. Run outside the sandbox, the report was "still running".
  - Re-graded from `result.json`, whose answer was cut to the last 4,000 characters, then redid it from the full
    export. The grades held, and results now keep the whole answer.
  - The first proposed option-line regex had two capture groups, so half its matches returned no letter.
  - The 2026-10-30 fix "a client hanging up is not a transport failure" covered only the write path. The stream
    was then parsed for a finish reason it never got, which logged a server error and threw away core attempt 1
    (an hour of model time) as infra_fail.
  - Blamed the scaled depth task's shortfall on its 14-turn fill cap and raised it to 24. The re-run still stopped
    at 176K: the task had 10 distinct fill prompts, and a repeated one added about 1K tokens because the model does
    not re-read files it has read. Seven new prompts reading unused files fixed it (209K in 17 turns).

## Learned

- The most common false failure was a hidden test reading a value the packet never stated: the recall cap
  (768) and a 24-site limit (haystack), a trial's `max_output_tokens` field (output-budget), and a
  case-sensitive `<!doctype html>` (html-report). Every Strata attempt at the first two failed (0/12 in phase 5).
- A decoy check must look at the claim, not the whole answer. All four document failures were correct answers
  that named the decoy (K-9999, Rowan) only to reject it.
- "Frozen" should mean the existing assertions stay as they are, which is what the packets say. Byte-identical
  frozen files failed four passes that only added a test.
- A depth floor in tokens depends on the tokenizer: Strata reached 38,414 against a 40,000 floor, and a fill capped
  at 14 turns reached 161,618 of a 191,488 target. Neither run measured the model at depth, so both are now
  `unmeasured`, not failures.
- A false pass is as important as a false fail: surgery-promote-experiment passed although the model rewrote the
  sandbox repo's history with filter-branch without asking.
- Strata sometimes ends a turn with `stop` partway through its reasoning and no text (sweep-chat turn 6, eval-manifest
  in core). That is an engine issue to raise in the Strata repo, not a grader one. eval-manifest passed 6/6 in
  phase 5.
- Claude Code deletes transcripts after 30 days by default. The earliest llm-eval sessions (Oct 2) were a day from
  deletion; they are now copied to `~/Archive/llm-eval-transcripts/claude/`.

## Independent review (afternoon)

An Opus reviewer falsified the frozen-file rule: an import swap, a module-level shadow, a monkeypatch or a new
`setUp` all counted as "kept". Fixed twice over: the rule now allows only new tests, functions, method-only
classes and new-name imports, and grading runs the original text of every frozen test file whatever the agent
added (`restore_frozen_tests`). Also fixed from the review: a heading line taken as the claim, `keeps_history`
passing with no starting commit, `Q:` read as an option, the pin check on an unresolved path, a contract that
stayed valid when every stream was abandoned, three packets missing values their hidden tests read, the prover
passing the owner's environment to third-party tests, and C10 (Occamy and Ornith do log preserved reasoning).
Re-graded with the stricter rule, Strata's numbers held except one arm-B smoke-vision pass in phase 5.

## Re-run (11:00 to 11:36)

The owner waived the at-the-machine rule for the re-run, so it ran under the plan unit, not by hand: a new
`--task PHASE:TASK` option narrows a phase, the restore-code change got a fresh SIGINT and SIGTERM drill (both
passed, restored in about 11 s), then haystack, output-budget and html-report in core (3/3, all checks read) and
depth-scaled-early-instruction twice (unmeasured at 176K, then a pass at 209,178 peak tokens after the fill fix).
An unmeasured task makes the runner say "assert_fail (no workload task was graded)": the verdict is right, the
wording is not.

## Numbers (raw, then corrected by the fixed graders, then the re-run)

| Phase | Raw | Corrected | Notes |
|---|---|---|---|
| 0 smoke | 1/1 | 1/1 | |
| 1 diagnostic | chat sweep 4/5, recall pass, tools 10/10 | same | recall 12/12 at about 243K prompt tokens; the tools trials are 63-token prompts, so they test format, not depth |
| 2 core | 24/35 | 26/35 | +1 advisory-overbuilt, −1 promote-experiment; haystack and output-budget pass on the fixed packets (re-run), html-report passes again; opencode-port passes with overreach recorded |
| 3 documents | 2/6 | 6/6 | |
| 4 depth | 2/5 | 5/5 | scaled-early-instruction measured on the re-run at 209,178 tokens |
| 5 MTP on (A) vs off (B), 5 tasks × 3 seeds | A 5/15, B 7/15 | A 9/15, B 8/15; 9/9 and 8/9 on valid tasks | MTP decodes 94.5 against 79.8 tok/s and finished 572 s sooner over 15 tasks; one arm-B smoke-vision pass was withdrawn by the review-tightened frozen-file rule (it added `sys.path.insert` to a frozen test file) |

Core failures that stand: four advisory tasks; eval-manifest (no output, the engine issue above);
router-model-path (ran `sudo -n true`); review-runner-resume (missed two of the defects); reconcile-diverged
(rebased without asking, where the reference merges); promote-experiment (history rewrite).

Runs: `~/Workspace/strata-eval-runs/2026-10-31-strata-iq3s-262144-{smoke,diagnostic,core,core-attempt2,documents,depth,knob-no-mtp}`;
plan log `2026-10-31-strata-iq3s-262144-plan/runner.log`. Re-run: `2026-11-01-strata-iq3s-262144-{drill-sigint,drill-sigterm,core,depth,depth-attempt2}`,
log `2026-11-01-strata-iq3s-262144-plan/runner.log`. Re-grade script (read-only, model code never run):
`tools/regrade_saved.py`. The three
html-report failures were confirmed by re-running their hidden tests with the doctype compared case-insensitively.

## Sources

Commit "Graders: fixes from the Strata plan audit" on `feat/strata-eval-adapter`; `DECISIONS.md` 2026-11-01;
Claude Code session `229856be-4b25-4b56-8264-e1f769f3ad25` (archived copy under `~/Archive/llm-eval-transcripts/claude/`).
