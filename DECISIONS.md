# Decisions

The entries below preserve decisions that would be easy to reverse without their context. They are grouped by date; each separates the rule from the reason for it.

## Dates

[2026-10-30](#2026-10-30) · [2026-10-31](#2026-10-31) · [2026-11-01](#2026-11-01) · [2026-11-02](#2026-11-02) · [2026-11-04](#2026-11-04) · [2026-11-06](#2026-11-06) · [2026-11-07](#2026-11-07) · [2026-11-08](#2026-11-08)


## 2026-10-30

- **Decision:** A trial cut off by `output_cap` or `wall_cap` counts as a failure in the quality denominator, labelled `incomplete`.
  **Why:** the adapter spec ("record any truncation as a failure") and the scoring spec ("task timeouts count as failures") say so; `e50999a` had dropped them from the scorecard, which lets a model score better by running out of budget.

- **Decision:** The 0.1.54 runs before 2026-10-31 stay on disk as a pre-fix record and are not a baseline.
  **Why:** they span 13 harness commits and changed verdict rules, their task identities moved on tar mtimes, their engine version is recorded wrong, and three graders misgraded correct answers.

- **Decision:** Engine comparisons are paired campaigns (`llm_eval.paired`, both arms in one campaign), never a new arm against an old run's numbers.
  **Why:** two 262K runs of the same five tasks differed by 2 of 5 verdicts and 14.6% model time, larger than most knob effects.

- **Decision:** An advisory task's answer check does not repeat the style check's `recommended_none` patterns over the whole answer.
  **Why:** the owner's required A/B/C format names the wrong options on purpose, and `advisory-models-to-keep` failed a correct answer on its rejected option C; `recommended_none` already polices the recommended option.

- **Decision:** The benchmark's pi describes each model as the owner's pi `models.json` does (`configs/pi-production.json`: no reasoning flag, the owner's `maxTokens`), loads the owner's AGENTS.md and compacts at pi's default.
  **Why:** the synthetic entry made pi send `reasoning_effort` and the developer role, which the owner's pi never sends, capped output at 16,384 where the owner allows 32,768 (`impl-llm-eval-haystack` ran out in three of six runs before writing a file), and gave pi no AGENTS.md at all. The capture still supplies the pinned effort and sampling, so the server sees the experiment's request either way.

- **Decision:** A replica's `home.tar` leaves out each `.git/index`, and unpacking rebuilds it from HEAD; a replica that stages changes is refused.
  **Why:** the index holds inode and ctime, so a task's identity changed with every prepare and a paired model comparison could never freeze (`surgery-split-commit`).

- **Decision:** The owner removed the five 65K `-agent` presets (router, pi, OpenCode); the restore model is now `muse-glimmer-30b-q4`.
  **Why:** each was its full preset's model file and settings with a smaller `ctx-size`, which buys no speed on llama.cpp; earlier compaction is a pi `contextWindow` setting the harness tests as a mode. Do not re-add them as benchmark targets.


## 2026-10-31

- **Decision:** pi, not the capture, sends thinking for local models: each entry declares reasoning with chat-template kwargs and `thinking_budget_tokens`, and `expected_request` replays pi's mapping.
  **Why:** the capture supplied `reasoning_effort: high`, which llama.cpp hands to the template; Swift's template raises on it, Qwen3.8 ran at a level production never sends, and Gemma never thought. Do not add a top-level reasoning_effort back for these entries.

- **Decision:** A client may lower max_tokens and its thinking budget below the pinned values (recorded as `lowered_limits`).
  **Why:** pi caps both at the window minus the context minus 4,096, so the last turns before compaction would otherwise get a 409.

- **Decision:** Fixture `wall_seconds` caps stay as they are although they run 3-70x the observed durations.
  **Why:** a task's `limits` are part of its identity (`spec_sha`), so tightening them breaks comparison with every earlier run, and a silent hang is already caught by the stall watchdog; the plan-level backstops (overhead, request timeout) changed.

- **Decision:** `request_seconds` is context/250 + 120 s (at least 600).
  **Why:** it is a socket idle timeout and a streamed request sends nothing during its prefill, so 600 s cut a cold full-context prefill on a slow llama.cpp profile.

- **Decision:** A llama.cpp preset's own `ctx-size` is a rung even above the 131,072 ladder top (plan compiler, `prepare --no-step-down`, the runner's default context); no other above-ladder context is.
  **Why:** the 262K presets could not be prepared at all ("not a rung this profile can load"), and the runner silently fell back to their anchor (occamy-1.0 ran at 98,304), so the plan's 262K rows and phase 6 against them never measured 262K.


## 2026-11-01

- **Decision:** A check marked `"advisory": true` is recorded but does not gate (`advisories_failed` on the grading); ops-opencode-port records unrequested changes this way.
  **Why:** owner's choice C, overreach is its own metric, not a task failure.

- **Decision:** A task whose only failing checks are depth checks is `unmeasured`, outside the quality denominator.
  **Why:** two correct Strata answers failed because the session stopped short of its depth floor; that says nothing about the model at that depth.

- **Decision:** Decoy `none` patterns in the document and depth cohort apply to the claimed answer only (first non-empty line, `none_scope: "claim"`).
  **Why:** all four Strata document failures named the decoy only to reject it.

- **Decision:** haystack, output-budget and html-report packets now state what their hidden tests read (recall cap 768 and 24 sites; trial `max_output_tokens`; lowercase `<!doctype html>`); the router-fix ops tasks allow a stop before the start.
  **Why:** every Strata attempt failed on unstated values (0/12 and 3/6 in phase 5). These task identities changed, so their earlier rows do not compare with new ones.

- **Decision:** A phase's verdict and summary line live in `llm_eval/phase_verdict.py`, outside `RESTORE_CODE`.
  **Why:** the drill hash covers whole files, so a wording fix in `tools/run_test_plan.py` cost a drill. The module only reads finished records; keep anything that starts, stops, signals or waits out of it, or add it to `RESTORE_CODE`.

- **Decision:** llama.cpp is pinned at build 12017 (`5047748f1`, `configs/llama-cpp-pin.json`); a llama.cpp prepare refuses another build and the plan records the pin.
  **Why:** owner, every preset loaded on it in the 2026-10-31 speed sweep and results across builds do not compare. Move it only with a fresh speed sweep.

- **Decision:** (supersedes the first 2026-11-01 frozen-file rule, archived): a frozen test file may gain only new tests, functions, method-only classes and imports of new names, and the tests check always runs the original text of every frozen test file (`restore_frozen_tests`).
  **Why:** an independent review showed the first rule accepted an import swap, a shadowing assignment, a monkeypatch and an added `setUp`; restoring the original makes the policy check a record, not the only guard.

- **Decision:** 1,024 MiB free GPU memory is the pass/fail mark, measured beside the working desktop: Brave open (many tabs), Steam and Slack closed, about 1.7 GB. The runner's 1,024 MiB gate is unchanged; the speed sweep's floor dropped from 1,424 to 1,024 (no 400 MiB margin) because it now runs beside that same desktop. A remote session over Tailscale (no Brave) uses less; a Chromium the model starts inside a task sandbox gets no GPU device (bwrap `--dev /dev`), so it renders in software and costs no VRAM.
  **Why:** owner, after briefly making 1 GiB a tracked target (reverted the same day, `47e0764`): under about 500 MiB a new Brave tab or kwin can fail to allocate.

- **Decision:** KV cache q8_0 is the default on every preset (six moved up from q4_0), and a preset short of the floor steps down in this order, each step on top of the last: context to 131,072 at the lowest, more CPU MoE layers, a smaller ubatch (never on Gemma: it caps image tokens), KV f16 -> q8_0, and q4_0 only as the last resort. Context never goes below 131,072.
  **Why:** owner, who prefers less context over a q4_0 cache, whose accuracy loss grows with context (mostly from K); a model that fails all of it is reported for the owner to decide.

- **Decision:** the speed sweep moves every knob one value at a time and never raises KV (q8_0 is the default, f16 is not worth its memory).
  **Why:** its jump-to-the-estimate logic depended on buffer sizes the server log has not printed on build 12017 (null in every result since 10-31), so it never ran; and a KV climb spent headroom on a proposal the owner does not want. Re-add jumps only with a working buffer source.

- **Decision:** a task home over 1 GiB (or the task's `limits.home_mib`) ends the trial as `capacity_fail` (`disk_cap`), and the sandbox's `/tmp` tmpfs is capped at 1 GiB.
  **Why:** normal homes are 1 to 21 MiB; a 10-31 trial grepped its own output to 17 GB. A not-a-model verdict, like the other capacity stops.

- **Decision:** a task environment (`<data>/envs/<name>`) is pinned by a content digest in its `ENV.json` and in every task `env` entry that mounts it; the folder is re-hashed at prepare and before each task-run, and a mismatch refuses the run.
  **Why:** red-team RT-002, a read-only mount does not freeze the host folder, so an updated venv or `node_modules` would change later runs under an unchanged task and plan. After a deliberate rebuild, run `python3 -m llm_eval.workload.envs <name>` and rebuild the tasks (a new task version), never drop the check.

- **Decision:** unittest and node mined tasks fail on any skipped or todo test (`skipped` in the tests check); pytest and vitest tasks rely on their per-test PASSED and check-mark lines instead, because flight-test's own suites skip tests for assets that are not in git.
  **Why:** red-team RT-001, code the agent may edit marked every test skipped and unittest still printed OK with exit 0. Core tasks keep the old unittest rule for now.


## 2026-11-02

- **Decision:** the operator agent (Sonnet, `RUNCARD.md`) works the queue in `QUEUE.md` on "continue" or "run X", may start, stop and restart `llama-router.service` while nothing of ours is running, and may fix operational harness defects (never tasks, graders, verdicts, frozen configs, presets, limits or gates), each fix its own `ops fix:` commit with tests and an operator-log entry.
  **Why:** the owner wants runs to resume without remembering where they were, and Opus kept finding operational defects mid-run; fixes stay reviewable instead of forbidden.

- **Decision:** the knob phase (5) runs only when `--knobs` or `--variant` names arms; the default is none.
  **Why:** owner; speculation (MTP, DFlash) is checked by the target model, so it changes speed, which the sweep measures, and not answers: Strata's no-mtp campaign spent 6 h for 9/9 against 8/9.

- **Decision:** llama.cpp profiles are screened one per base model (`--only 0,1,2`), and only the top 3 run the rest of the accuracy suite. Muse is screened as `muse-glimmer-30b` (the variant with image input); Qwen3.8 27B as `qwen3.8-27b-unsloth-q4-mtp` (owner; IQ4_XS was the first pick, for 262K, but at the q8_0 KV default it no longer fits 262K by estimate).
  **Why:** owner, option A; other quants of one model differ by less than 35 core tasks can resolve (about 15 points), so speed and fit choose between them, not accuracy runs.


## 2026-11-04

- **Decision:** A sampler the vendor does not name is switched off in the preset (`min-p = 0`, and `top-k = 0` for North), never left to llama.cpp's defaults (min-p 0.05, top-k 40).
  **Why:** owner, for like-for-like screens; those defaults are nobody's recommendation and narrow sampling for Gemma, Muse and North only (their GGUFs embed no min-p, and North's no top-k).

- **Decision:** pi's 16,384-token thinking budget stays on for every model, though no vendor publishes it, and the report counts turns that reach it ("thinking-cap turns").
  **Why:** owner, option A; it is what production pi sends, and on the first screens it fired on 12 of 2,471 Qwen3.6 turns (mostly loops) and 2 of 498 Qwen3.8 turns, so it acts as a loop breaker. Raising it changes production, not just the benchmark.


## 2026-11-06

- **Decision:** The advisory option format (lettered A, B, C, one recommended) is required only when an answer offers options; a direct answer is held to the em-dash rule and its first line to `recommended_none`. The parser also reads `**A: x**`, `Option A:`, lowercase letters, a second list restarting at A and a separate `Recommended: X` line.
  **Why:** owner (review 2.1); the owner's AGENTS.md asks for the format when options are offered, and 19 of 50 screen trials failed on format alone with correct content. Answer-only, so saved runs re-grade (`tools/regrade_saved.py`).

- **Decision:** The 20 question-only core tasks (advisory, lookup, resume, review) reply to a question with "finish with your answer and change nothing" (`followups.default`, `session.QUESTION_REPLY`), not "Use your judgment and continue".
  **Why:** owner (review 2.4); every model that acted on a question did so right after the old reply. Their identities changed, so core rows before 2026-11-06 do not compare with later ones on those tasks; depth tasks are unchanged.

- **Decision:** A pi turn that hangs while a tool call it started is still open stops as `tool_hang`, a model timeout (counted, incomplete), and the server is not torn down.
  **Why:** north-mini ran a fake server that sleeps forever; the stall was scored infra_fail and the healthy server was reported as failed.

- **Decision:** A recall or tool-call section that ran at the phase's target and missed is `quality_fail` (recall with its rungs and record counts), not `assert_fail`.
  **Why:** the phase measured what it promised; "verified at None" hid that Muse got 32 of 36 records.

- **Decision:** (consolidated 2026-11-08; originals in `archive/2026-11-08-DECISIONS-superseded.md`): Candidates are compared only on runs made at the git tag `bench-v2`, now at `c19f776`, which includes the 11-07 test fixes (`7eac04a`); the earlier target is kept as `bench-v2-pre-7eac04a` (`1979d33`) and the grader identity is `04468edc`. Until the comparison runs are done, nothing changes under `llm_eval/`, `fixtures/`, `tools/`, `configs/` or `suites/`: `git diff --quiet <tag> <run harness_sha> -- llm_eval fixtures tools configs suites` must pass. The inputs outside git must match across the runs too: the pi AGENTS.md is the repo copy `configs/pi-agents-md-bench.md` (hash 69cbbcde, not the live dotfiles file), pi 1.4.0, llama.cpp 12017, each candidate's preset section (`settings_json`), the task env digests and Strata's pinned files. During the comparison the operator blocks a row instead of fixing frozen code; a bug found later is written down and fixed after the runs. Owner-approved design (review decisions 1A to 6A): Qwen3.8 KV sweep at q8_0 with MTP off, core x3 on every bench row and no noise-accuracy, full suites for all four candidates, a common-131072 table for recall, depth and documents, Strata's thinking request stated and not changed. Rows run on the old grader (11-07 chain accuracy suites) compare only through their regrades in the chain report, section 7.
  **Why:** the 11-06 fixes changed 22 of the 35 core tasks, so one more edit mid-comparison would split the table again; `suites/ladder.json` sets the context rungs; the live dotfiles AGENTS.md changed on 11-06 with no refusal, so a run could drift with no harness diff; one core run per model cannot rank candidates inside 3 points; and every remaining bench row must share one grader (on the old tag's grader Muse scored 79/105 against 81/105).


## 2026-11-07

- **Decision:** Fixed three test faults at the owner's request, after the `bench-v2` rows had started but before any bench row ran; the tag was not moved. (1) `_claim` cuts a one-line answer at its first sentence, so naming the decoy after the claim no longer fails (eight correct documents answers failed, Qwen3.8 and Swift). (2) `machine._path` reads an absolute path under the real home as the run's home when grading on the host, because the box binds the run's home there (correct `depth-ops-ground-rule` and core `ops-router-model-path` fixes failed `unit_ready`). (3) Depth floors `depth-lookup-early-instruction` 36,000 to 30,000 and `depth-ops-ground-rule` 40,000 to 35,000 (the smallest peaks of models that read the files in full were 32,574 and 37,505). `tools/regrade_saved.py` now re-evaluates `unit_ready`. Saved 11-06 and 11-07 runs were regraded; `bench-v2` rows and every score before this entry that touches these tasks do not compare with later ones.
  **Why:** the review found these scores measured the test, not the model.


## 2026-11-08

- **Decision:** Ranking reports each bench row twice: raw, and with style-only failures (the em-dash check) and confirmed grader or task faults removed (owner, agreeing with the Opus review of bench-qwen3.8-27b). The em-dash rule is not treated as substance when ranking, and neither number replaces the other.
  **Why:** 6 of Qwen3.8's 14 core failures were the em-dash check alone and 4 more were grader or task faults, so the raw score mostly measures formatting; dropping the raw number would hide that the model ignores a rule the owner set. How to apply: label rows "raw" and "adjusted", name every removed trial, never compare across modes, and only remove a fault once it is confirmed from the answer and end state.

- **Decision:** Qwen3.8 bench row adjusted score is 98/105 (raw 91): the 6 style-only trials and overbuilt r3 removed. Haystack r1 r2 r3 stay in as disputed; removing them gives 101/105 as an upper bound, not the headline. Supersedes the "4 more were grader or task faults" count in the entry above (that counted all three haystack trials).
  **Why:** haystack r2 has a real context-label error beside the task faults, and a fault is removed only once confirmed from the answer and end state. How to apply: recompute Muse and later rows on this basis before ranking.
