---
title: Like-for-like review of the bench-v1 comparison plan
date: 2026-11-06
type: review
project: llm-eval
status: done
summary: The pinned tag does not pin the system prompt (the owner's AGENTS.md changed on 11-06 and is read live), "continue" would have run noise-accuracy first, one core run cannot rank four candidates, and Qwen3.8's q4_0 KV is a choice, not a fit; six owner decisions, queue and runcard fixed.
tags: [review, bench-v1, like-for-like, noise, context]
---

# Like-for-like review of the bench-v1 comparison plan

Review of the plan in `STATE.md`, `QUEUE.md` and the 2026-11-06 `DECISIONS.md` freeze entry, against the code at
`bench-v1` (`ff6b302`), the run folders in `~/Workspace/llm-eval-data` and `metrics.sqlite` (opened read-only).
Nothing ran on the GPU; no code, fixture, config or preset changed. One read-only tool ran:
`tools/regrade_saved.py` on two saved Strata runs, to check the regrade note.

## Headline

1. **bench-v1 does not freeze the system prompt.** Every pi run pins a copy of the owner's live
   `~/Workspace/dotfiles/agents/AGENTS.md` (`PI_AGENTS_MD`, `llm_eval/workload/prepare.py` line 25). That file changed
   on 11-06 (dotfiles `6f0b06b`) and the new text is on disk now. The bench rows would run on a prompt no earlier
   result used. The new rules touch exactly what the advisory and question tasks grade.
2. **"continue" would have started with the wrong row.** `noise-accuracy` (`todo`, Qwen3.6, core x3, 6 to 12 h) sat
   above the bench rows, so it was the first open row. Fixed: moved below the owner row.
3. **One core run per model cannot order four candidates that sit within 3 points.** At the measured flip rate, the
   95% band on a gap between two single runs is about ±4.4 tasks.
4. **Qwen3.8's q4_0 KV is a choice, not a fit limit.** Swift is the same base model and fits at q8_0 with MTP off.
   Qwen3.8 keeps MTP and spends q4_0 headroom on 163840 context, which the owner's step-down order puts last.

The rest are smaller: the freeze scope misses `suites/`, the operator could still commit fixes mid-comparison, context
changes what recall, depth and the 100K documents measure, Strata's request differs in thinking, and the Strata
regrade note overstates two things.

## Findings, by severity

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | Critical | AGENTS.md read live, changed 11-06 | Owner decision 1 (code change, re-tag) |
| 2 | High | `noise-accuracy` was the first open row | Fixed (`QUEUE.md`) |
| 3 | High | One core run per model is inside the noise | Owner decision 3 |
| 4 | High | The top-3 gate saves at most one night and is decided inside the noise | Owner decision 4 |
| 5 | High | Qwen3.8 at q4_0 KV with MTP on, against the owner's step-down order | Owner decision 2 |
| 6 | Medium | Freeze scope misses `suites/` and every input outside git; RUNCARD still allowed fixes | Fixed (`DECISIONS.md`, `RUNCARD.md`) |
| 7 | Medium | Context changes recall, depth and 100K documents | Owner decision 5 |
| 8 | Medium | Strata's request differs in thinking, beyond fit | Owner decision 6 (recommend: state it) |
| 9 | Medium | Strata's earlier results: not bench code; regrade note overstated | Note corrected; part of decision 4 |
| 10 | Low | RAM margin about 700 MiB; preflight would stop the whole night on a Strata shortfall | Fixed (`RUNCARD.md`) |
| 11 | Low | RUNCARD RAM figure stale (54,326 against the code's 57,014) | Fixed (`RUNCARD.md`) |
| 12 | Low | No run will ever record SHA `ff6b302`; proof needs a diff | Fixed (provenance check below) |
| 13 | Low | Plan-folder collision if bench-muse starts on 11-06 | Note only |
| 14 | Low | `flip_rate.py` drops incomplete trials; `regrade_saved.py` misses collection specs | Owner decision 1 (same re-tag) |
| 15 | Low | `owner-top3-full-suite` named the screen rows, not the bench rows; a stray cell in `screen-strata-rerun` | Fixed (`QUEUE.md`) |

### 1. The system prompt is not frozen (critical)
- `prepare.py` reads `PI_AGENTS_MD = ~/Workspace/dotfiles/agents/AGENTS.md` at every prepare. Each plan pins a copy
  (`home-files/.pi/agent/AGENTS.md`) and hashes it in `plan.json` `files`. Nothing refuses a change.
- Hashes pinned so far (sha256, first 12):

  | AGENTS.md | Runs that used it |
  |---|---|
  | `c30865d4e6a7` (dotfiles `b4f1576`, 10-23) | Strata 10-31 and 11-01: core, depth, documents |
  | `3c227ed2db20`, `c3ee8aa97d68` | Drills 11-02/03; Qwen3.6 and Qwen3.8 screens 11-03 |
  | `69cbbcdef4ad` (dotfiles `d96b6ec`, 11-03) | Every run from 11-04 to 11-06 19:13: Strata mined, game and documents, every screen since, the Muse full suite |
  | `bd56127290f8` (dotfiles `6f0b06b`, 11-06 13:27) | On disk now; no run yet. The bench rows would pin it |

- The new text adds:
  - "Offer options only for material choices"
  - "When two of my files disagree, take the more permissive reading ... and keep going"
  - "Progress updates never pause work"
  - a diverge-first ideation rule

  The advisory format check, the question-reply tasks and the acting-on-a-question failures all depend on how a model
  reads these rules.
- The freeze check (`git diff --stat bench-v1 -- llm_eval fixtures tools configs`) cannot see this.

### 2. The queue would have run noise-accuracy first (high, fixed)
`QUEUE.md` says the operator "runs the first open row and keeps going down". `noise-accuracy` was `todo` and sat above
every bench row. On "continue" it would have run first: Qwen3.6, a model out of contention, 35 tasks x 3 runs, 6 to 12 h.
The plan's "about 10 h chain" assumed the chain starts at `bench-muse-30b`.
- Fixed: the row now sits below `owner-top3-full-suite` with an `owner:` status (profile and decision 3).

### 3. One core run per model is inside the noise (high)
- Measured: the two Qwen3.8 screens differ on 5 of 35 tasks, a flip rate of 0.14.
- That gives a single-run score SD of about 1.6 and a gap SD of about 2.2 between two single runs; the 95% band on a
  gap is about ±4.4.
- The four candidates sit at Muse 28, Strata 27 (old specs), Swift 26 and Qwen3.8 25.
- One run each can confirm that all four lead the field (Gemma is 20, on old specs). It cannot order them, and it
  cannot pick a top 3.
- With 3 runs each, the SD of a mean is about 0.9 and the gap's 95% band about ±2.6. Each model also gets its own flip
  rate. A single Muse noise run assumes every model is as steady as Muse, which has no loops; Qwen3.8 and Occamy have
  looped.

### 4. The top-3 gate (high)
- Muse's full suite is done (`full-muse-30b`), and Strata's mined, game, documents and depth exist.
- Picking 3 of 4 therefore skips at most one full suite (Qwen3.8 or Swift, about 4 to 5 h). That saving rests on a
  core ranking made inside the noise (finding 3).
- Running the full suite on all four costs one extra night. The like-for-like table then covers every candidate.

### 5. Qwen3.8 KV q4_0 is a choice (high)
- Current preset for each profile:

  | Profile | Context | KV | Speculative decoding | Quant |
  |---|---|---|---|---|
  | muse-glimmer-30b | 131072 (its maximum) | f16 | DFlash on | UD-Q4_K_XL |
  | swift-qwen3.8-27b-vision | 131072 | q8_0 | MTP off, projector in RAM | Q4_K_M |
  | qwen3.8-27b-unsloth-q4-mtp | 163840 | **q4_0** | MTP on | UD-Q4_K_M |
  | Strata (frozen `iq3s-262144.json`) | 262144 | int8 | MTP on | IQ3_S |

- `DECISIONS.md` (2026-11-01) and `RUNCARD.md` section 7 both treat q4_0 as the last resort. They order it after
  dropping speculative decoding and after cutting the context to 131,072.
- The sweeps show Qwen3.8 was not at that last resort:
  - At q8_0 and 131072 with the MTP draft, Qwen3.8 left 573 MiB free (`sweep-qwen3.8-builtin-mtp` row).
  - Swift is the same 27B base model at the same Q4_K_M size. With MTP off and the projector in RAM, it leaves
    1,527 MiB free at q8_0 and 131072.
- So Qwen3.8 at q8_0 with MTP off very likely fits at 131072. At 163840 it probably does not.
- The q4_0 cache on Qwen3.8 is the one setting difference that is not a fit limit. It also sits on the closest pair in
  the table: Swift against its own base model.

### 6. Freeze scope and outside inputs (medium, fixed)
- `suites/` is a behaviour folder (`environment.BEHAVIOUR_DIRS`; `ladder.json` sets the context rungs) but is not in
  the freeze check.
- Inputs outside git that change a run and that nothing refuses:
  - the AGENTS.md copy (finding 1);
  - each candidate's section of `~/.config/llama-cpp/models-preset.ini` (recorded in `settings_json`);
  - the pi binary (1.4.0 on every run, recorded but never refused);
  - node through fnm (`NODE_HOME`, mounted read-only into tasks, not digested);
  - the system python, git and bwrap on a rolling distribution.
- `RUNCARD.md` section 7 still let the operator commit an `ops fix:` to restore code in the middle of the comparison.
- Fixed:
  - A `DECISIONS.md` entry adds `suites/` to the check and lists the outside inputs that must match.
  - `RUNCARD.md` section 7: while the comparison runs, a defect in frozen code blocks the row instead of being fixed.
  - The provenance check below.

### 7. Context changes three measurements (medium)
- **Core: barely affected.** On the last screens, 1 Muse trial compacted. Qwen3.8 (1 trial) and Strata (1 to 2) went
  past 94,208 tokens, where a 131072 profile compacts, without compacting. That is noise-sized, but state it per model.
- **Recall:** the prompt fills the window. Muse and Swift read about 112K tokens, Qwen3.8 about 145K, Strata about 243K.
- **Depth:** sessions are filled to 0.9 of the context (`fill.fraction`). The compaction point is context minus 36,864:
  94,208 at 131072, 126,976 at 163840, 225,280 at 262144. Muse's deepest depth trial peaked at 92,666 tokens; Strata's
  at 209,178.
- **Documents 100K:**
  - At 131072 a 100K document cannot be held below the compaction point. Muse compacted on all three 100K tasks and
    failed one of them.
  - Strata (262144) compacted on none.
  - At 163840, Qwen3.8 holds 100K below its compaction point.
- **Recommended:** keep each model's adopted context for the main table, and add a common-context table at 131072 for
  recall, depth and documents. Only Qwen3.8 and Strata need extra runs (Muse and Swift already run at 131072), about
  2 h in all.
- A common-context core is not worth its cost (about 3.5 h for one run each) given the compaction counts above.

### 8. Strata's request differs in thinking (medium)
Captured request bodies (`inference_requests.contract_json`):

| Field | llama.cpp profiles | Strata |
|---|---|---|
| Thinking budget | `thinking_budget_tokens: 16384` (pi) | none (pi entry has no `reasoning`) |
| Effort | template kwargs: Muse `reasoning_strength: high`, Qwen3.8 and Swift `reasoning_effort: xhigh` | top-level `reasoning_effort: max` (capture) |
| History | template default | `preserve_thinking: true` |

These are production-faithful: the owner's pi entry for Strata declares no reasoning. They are not fit limits.
Equalising them changes `configs/pi-production.json` and production, so the recommendation is to state them.

### 9. Strata's earlier results and the regrade note (medium)
Verified:
- Task identities. The latest Strata result of every mined, game and depth task has the same `task_identity_sha` as
  Muse's 11-06 runs on bench-equivalent code: 31 of 31 the same (20 mined, 6 game, 5 depth). The six documents
  differ only by their per-tokenizer label, as designed.
- Each depth task's latest Strata attempt is its only attempt on the current identity, so using the latest is not
  cherry-picking.
- The regrade flips reproduce: mined `mined-thermalright-source-isolation` fail to pass (diff_scope), and depth
  `depth-impl-reasoning-fields` fail to pass.
- No wall_cap, stall or tool_hang stops in the Strata depth or 11-04 trials.

Overstated or missing in `docs/operator-log/2026-11-06-strata-regrade.md`:
- **Not bench-v1 code.** The runs use harness SHAs `66561c4`, `995d522`, `4e4e07c` (depth, 10-31 and 11-01), `62b7cdb`
  (documents) and `d6138f9` (mined, game), all before the 11-06 fixes. DECISIONS says candidates are compared only on
  bench-v1 runs, so these need the explicit exception the plan assumes.
- **Documents 6/6 ran on a dirty harness.** `runs.harness_dirty = 1` on run `b69935603ece`: `runner.py` was being edited
  for the mined and game workload group during the run. Low impact on documents, but it cannot be reproduced from its SHA.
- **`depth-impl-reasoning-fields` is provisional.** Its 10-31 run predates `03b7bbc` (11-01 13:44), which made grading
  run the original frozen tests. Its saved `tests` pass ran against the model's edited frozen file. Under today's
  procedure it might not pass.
- **Depth used the 10-23 AGENTS.md** (`c30865d4e6a7`), not the 11-04 one.
- **The mined and game regrade is partial.** `tools/regrade_saved.py` `task_for` looks only at
  `fixtures/workload/<id>.json`, so collection tasks (mined, game, vision) load no spec. Only the frozen-file rule and
  the claim-scope answer rule were re-applied to them. The rest of the saved checks stand as recorded.

The note now carries a short addendum pointing here.

### 10 and 11. RAM gate (low, fixed)
- Available RAM now: 56,292 MiB. The floor in `llm_eval/ramcheck.py` is 57,014 MiB. The launch check also credits
  what the router returns.
- The last drill started at 57,109 MiB, so the margin is a few hundred MiB.
- `plan_ctl.py preflight` checks every open Strata row at the start of the night. A shortfall makes `"ok": false`, and
  RUNCARD step 0 then launched nothing, so the three llama.cpp rows, which need no RAM gate, were lost too.
- Fixed in `RUNCARD.md`: a RAM problem on a Strata row alone does not stop the rows that run before it.
- Also fixed: RUNCARD now says 57,014 MiB instead of the stale 54,326.

### 12. Proving a run ran on the pinned code (low, fixed)
The recorded `harness_sha` is HEAD at run time. The operator commits queue edits before every launch, so no run will
record `ff6b302`. The tag is also lightweight and can be moved. A run counts as bench code when all of these hold:

```
git diff --quiet bench-v1 <harness_sha> -- llm_eval fixtures tools configs suites   # exit 0
runs.harness_dirty = 0; runs.llama_server_version contains "build 12017" (llama.cpp rows)
plan.json workload.binary_version = 1.4.0; plan.json files[...agent/AGENTS.md] = the pinned hash
run_models.settings_json = the adopted preset section; Strata: settings.pinned_files equal across its runs
task_runs.task_identity_sha equal across models (documents: per tokenizer)
```

### 13. Plan-folder collision (low)
If `bench-muse-30b` starts before midnight on 11-06, it reuses `2026-11-06-muse-glimmer-30b-131072-plan` with
`recall-muse-recheck`. Its diagnostic folder becomes `...-diagnostic-attempt2`. `plan_ctl` reads `SUMMARY.md` from
its own offset, so this is cosmetic.

### 14. Analysis tools (low)
- `tools/flip_rate.py` leaves incomplete task-runs out, which understates noise for models that loop.
- `tools/regrade_saved.py` misses collection specs (finding 9).
- Neither is on the run path, but both are under `tools/`, so a fix moves the freeze. Fold both into the decision-1
  re-tag if it happens.

## What is already equal (checked)
- **Phases.** `--only 0,1,2` gives every runtime the same three phases: smoke (`impl-llm-eval-eval-manifest`, one run),
  diagnostic (recall, tools, chat sweep) and the 35 core tasks at one run each. The Strata-only phase 3 is not in it.
  No core task needs image input, so none is dropped for Qwen3.8 or Strata.
- **Client settings.**
  - Output: 32,768 for every candidate, from `configs/pi-production.json`, which the freeze covers. Pinning `--output`
    adds nothing and would only flip `mode.output_overridden` against the Muse full-suite runs. Leave it.
  - Harness: pi 1.4.0 for all.
  - Compaction: reserve 36,864, keep 20,000 for all.
  - Thinking level: `max`, clamped by pi to each model's top level.
  - Sampling: from each card. Muse top-k 64; Qwen3.8, Swift and Strata top-k 20; all temperature 1.0, top-p 0.95,
    min-p 0.
- **Grading.** `llm_eval/phase_verdict.py` and both grader modules have no runtime or profile branch. The only
  per-runtime code is in prepare: Strata's manifest, the tokenizer that sizes documents, and the image filter.
- **Tag.** `bench-v1` = `ff6b302`. The tracked tree is clean; the untracked dotfiles in the repo root are sandbox stubs.
  It includes the advisory format fix and output-budget packet (`04bc946`), the driver race, tool hang, question reply
  and lle-006 fixes (`fc80b36`), the SHA map (`0a5897f`), the tools verdict (`0f05585`) and the vision group
  (`9b4d3fb`). Nothing is left unmerged that the plan needs.
- **Drill.** A drill passed on the bench-v1 restore code (`a09c5aa8`, SIGINT 19:03 and SIGTERM 19:08). A change to
  `prepare.py` alone does not void it.
- **Muse full suite.** It ran at `d71d26d`, which differs from bench-v1 only by `9b4d3fb` (vision group), and its vision
  re-run came after that fix. Its code is bench-equivalent; its AGENTS.md is `69cbbc`.
- **Engines.** llama.cpp build 12017 is recorded on every llama.cpp run. Strata's 380 pinned files are identical on
  10-31, 11-04 and 11-06, apart from the preset file and the config path.

## Decisions for the owner

1. **System prompt for the comparison.**
   - **A.** (Recommended) Pin `69cbbc` in the repo and re-tag `bench-v2` before anything runs. That is the version of
     every result since 11-04, including the Muse full suite and Strata mined, game and documents, so no re-runs are
     needed. Patch:
     - add `configs/agents-md/AGENTS-69cbbc.md`, the output of `git -C ~/Workspace/dotfiles show 6f0b06b~1:agents/AGENTS.md`;
     - set `PI_AGENTS_MD` in `llm_eval/workload/prepare.py` to that file;
     - add a test that its sha256 starts `69cbbcdef4ad`, and a line in `configs/index.md`.

     `prepare.py` is not restore code, so no drill is needed. The integration tier re-stamps itself at the first launch
     (up to 30 min, no GPU).
   - **B.** Pin `bd5612` (today's agreement) the same way. Then also re-run the Muse full suite (about 3 h) and Strata
     mined, game, documents and depth (about 4 h).
   - **C.** No code change: keep reading the live file and promise not to edit it until the comparison is done. Same
     re-runs as B, checked per run by the plan's AGENTS.md hash.
2. **Qwen3.8 KV.**
   - **A.** (Recommended) Sweep q8_0 with MTP off at 163840, then at 131072 (row `sweep-qwen3.8-q8-nomtp`, about
     0.3 h), and adopt the first that clears 1,024 MiB. At 131072, `configs/pi-production.json` changes too, so the
     re-tag in decision 1 includes it.
   - **B.** Keep q4_0, MTP on, 163840 as adopted, and state it as a chosen setting, not a fit.
3. **Noise.**
   - **A.** (Recommended) Add `--runs 3` to the four bench rows, which gives core x3 and per-model flip rates. Drop
     `noise-accuracy`. About +12.6 h.
   - **B.** One core run each, plus `noise-accuracy` on muse-glimmer-30b (3 h) as a shared margin. The core order then
     stays a tie.
4. **Full suites.**
   - **A.** (Recommended) Run them for all four candidates: Qwen3.8 and Swift one night each. Re-run Strata depth and
     documents on the bench tag (about 0.6 h; this removes the 10-23 prompt, the provisional pass and the dirty run).
     Keep Strata mined and game with the finding 9 caveats.
   - **B.** Keep the top-3 gate as queued.
5. **Context.**
   - **A.** (Recommended) Main table at each model's adopted context, plus a 131072 table for recall, depth and
     documents (rows `common-ctx-qwen3.8`, `common-ctx-strata`, about 2 h; the Qwen3.8 row drops if decision 2 puts it
     at 131072).
   - **B.** Adopted contexts only, with the differences stated.
6. **Strata thinking.**
   - **A.** (Recommended) State it: no thinking budget, effort `max`, `preserve_thinking`.
   - **B.** Equalise: give Strata's pi entry `reasoning` and a 16,384 budget field. This changes production pi and
     `configs/`, and needs Strata to honour the field.

## Run order (all recommendations taken)

| Step | Runs | Hours |
|---|---|---|
| 0 | Owner decides; code change and tag `bench-v2`; `sweep-qwen3.8-q8-nomtp` and preset edit (decision 2A) | 0.3 GPU, plus up to 0.5 integration tier |
| Nights 1 and 2 | bench-muse-30b (3.0), bench-qwen3.8-27b (5.8), bench-swift-vision (6.4), screen-strata-rerun (5.2), each `--only 0,1,2 --runs 3` | about 21 |
| Night 3 | full-qwen3.8-27b (about 4.5), strata-depth-docs-bench (0.6), common-ctx-strata (1.0), common-ctx-qwen3.8 (1.0, only if Qwen3.8 stays at 163840) | about 7 |
| Night 4 | full-swift-vision (about 5, vision included) | about 5 |
| Then | Analysis with the provenance check, publishing | 0 GPU |

Total: about 34 GPU hours over four nights. Under decision 3B the bench chain is about 9 h, and the total about 23 h.
Hours come from the screens' measured phase times (core: Muse 0.88 h, Qwen3.8 1.8 h, Swift 2.0 h, Strata 1.6 h) and
Muse's full suite (2.8 h), scaled by speed.

Launch each night with Steam and games closed. Under the RUNCARD fix, a Strata RAM shortfall no longer stops the rows
before it. Keeping Strata last in the chain means a refusal costs nothing else that night.

## What remains not like for like, and why

| Difference | Why it stays | How it is reported |
|---|---|---|
| Context: Muse 131072 (its maximum), Swift 131072, Qwen3.8 163840 (or 131072, decision 2), Strata 262144 | What each fits; the owner's production windows | Main table per model; 131072 table for recall, depth, documents; compacted trials counted per model |
| KV: Muse f16, Swift q8_0, Qwen3.8 q4_0 (or q8_0), Strata int8 | Fit (Qwen3.8's is a choice unless decision 2A) | One column per model |
| Quant: Muse UD-Q4_K_XL, Qwen3.8 UD-Q4_K_M, Swift Q4_K_M, Strata IQ3_S | Fit; Strata's model is far larger | Stated |
| Speculative decoding: Muse DFlash, Qwen3.8 MTP, Strata MTP on; Swift off | Fit; the target model verifies drafts, so output quality is unchanged | Speed only |
| Swift projector in system RAM | Fit | Speed only on vision |
| Vision: Qwen3.8 and Strata take no images | Model ability | Vision compared Muse against Swift only; totals over the shared parts |
| Sampling from each card (Muse top-k 64) | Vendor settings | Stated |
| Thinking: top level per model (Muse high, Qwen3.8 and Swift xhigh); Strata effort max, no budget, preserved thinking | Production pi entries | Stated (decision 6) |
| Strata mined and game on the pre-11-06 harness, partial regrade | Re-running costs about 3 h for no expected change | Caveat beside the numbers |
| Depth and documents task sizes scale with context (fill 0.9; a 100K document compacts at 131072) | Designed to test each model at its own window | Common-context table |
| One run per task in the full suites | Cost | Gaps under about 2 tasks on a 6-task part, or 3 on mined, read as ties |
