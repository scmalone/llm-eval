# Plan: solo-local workload benchmark, cheap screen then top 3 (locked 2026-10-04)

Supersedes the run shape in `2026-10-03-workload-benchmark.md` (n=2, both modes, recall tier, one full run per profile). Task design, grading, sandbox and warehouse from the earlier plans stand.

## Purpose
Find which local model, context size and `limit.output` to use when working solely on the 4090 desktop with no hosted model (possible private data). The `-agent` entries (65,536 context, 12,288 output) exist so a hosted model can delegate a workload to the local LLM; that is a different use case and is not benchmarked here.

## Locked decisions (owner, 2026-10-04)
1. **Workload suite only, bare entry only.** The `-agent` mode cohort is backlogged as a delegation benchmark. Recall is dropped from this run (backlog). No profile is capped at 65,536.
2. **n=1** is the default runs per task.
3. **Staged.** A cheap screen runs on all 14 profiles first and every screened profile's results must be recorded. The full workload run is for the top 3 from the screen. The goal is all models eventually.
4. **Context.** Target 131,072. The workload runs at the highest qualified rung at or below it (ladder 65,536, 86,016, 98,304, 102,400, 114,688, 131,072) and records the context it actually ran at. Qualification's free-VRAM figure per rung is part of the screen's output. The owner's working target is 80,000 to 100,000 usable context, with 64,000 as the floor; usable context is the compaction point, `context - min(limit.output, 32,000)`.
5. **Output limit.** 16,384 for every profile in the screen, so output is not a variable. For the top 3, set it from each profile's measured peak output with headroom, capped at the model card's recommendation. Any task ending on a length cutoff flags the profile.
6. **Depth.** One depth task scaled to each profile's compaction point, run on the top 3 only. Nothing in the screen checks quality at 80 to 100K.
7. **Hedges.** Remove anything built against a hypothetical need, especially guardrails that data and a run should reveal. Invariants that stay: no real confidential data to a hosted model, loopback checks, production restore and verification, VRAM gates, bwrap isolation of the agent under test.
8. **Time caps.** No fixed 3600 s floor; caps are enforced mid-task; a stall watchdog (600 s of silence from opencode and the server) ends a hung turn.

## Steps
1. **Efficiency red-team** (Codex `gpt-6.1-sol`, high, 3600 s, done 2026-10-04): the screen's design, the selection rule, waste in the run path, and an inventory of everything built beyond the pilot sorted into needed / can wait / safety invariant / hedge.
2. **Adjudicate** the findings with the owner; stop before fixes. Done: see "Decided at step 2".
3. **One build pass** (done 2026-10-04, six commits, 286 tests), then tests and a STATE row, committing only touched files:
   - the context step-down fix (workload runs at the highest qualified rung, config set in the sandbox copy, actual context recorded, ladder counts target-or-fallback runs)
   - n=1 default (`acceptance.py:95`, `ladder.py:44`, `runner.py:216`, `prepare.py:80`, `tasks.py:137`) and the tests that assume 2
   - bare as the manual `prepare` default (`opencode_config.model_entry` prefers `-agent` today) and "agent" removed from the ladder's mode choices
   - output limit 16,384 in the screen; the scaled depth task for the top 3
   - the screen definition and top-3 rule from step 2; the approved hedge removals
4. **`/redteam` once** on the live-path part of that diff (`run_workload`). Optional: the owner may skip it.
5. **Integration stamp:** `python3 -m llm_eval.acceptance integration` from a plain terminal, after the edits.
6. **Preflight:** the owner closes ComfyUI and anything else holding more than 1 GiB of VRAM.
7. **Rung 1** (`prepare --next`, card, owner's yes, `run --next --approve <token>`, record). Rung 1 also checks what could not be verified offline: which of the server's `/slots` and `/metrics` counters move during generation, `max_silence_seconds` for false stalls, capture errors, the GPU sampler, the `--metrics` endpoint.
8. **Screen** across all 14 profiles, then rank and pick the top 3.
9. **Full run on the top 3** at n=1, with output limits from the screen's measured peaks and the scaled depth task.

## Backlog
- Delegation benchmark for the `-agent` entries (hosted model hands a workload to the local LLM).
- Long-context recall ladder as a diagnostic.
- Remaining profiles beyond the top 3.
- A canned-webfetch task; a local judge if tasks on private data are added.

## Decided at step 2 (2026-10-04, after the efficiency red-team)
- **Screen (7 tasks, one run each, in this order):** `lookup-smoke-test` (the live pilot), `resume-llm-eval-owner-decision`, `advisory-models-to-keep`, `ops-fans-gpu-hog`, `surgery-split-commit`, `review-llm-eval-live-recovery`, `impl-llm-eval-html-report`. One per category.
- **Ranking** (`acceptance screen`): substantive passes, then category coverage, then elapsed time on the tasks every tied profile passed. Infrastructure and capacity failures and missing results are owed, never counted as failures. A timeout is a failure kept visible. No quality-based elimination in the first screen. Review and advisory passes are audited by reading the answers before the ranking is trusted.
- **Contested third place:** the same extra task for each boundary candidate (`impl-llm-eval-output-budget`, then `review-llm-eval-memory-obligation`); if still tied, the faster profile, labelled provisional.
- **Hedges removed or deferred:** the per-rung record gate and the 16-rung ladder (now `r01-pilot`, `r02-screen`, `r03-core`, `r04-depth` in two stages, one yes for the screen); Codex and pi runbooks; further unattended hardening, the 20-pair judge calibration and the report redesign (deferred). Kept: wide raw capture, the integration stamp, the enforced stall watchdog, the per-plan approval token, the qualification climb.
- **Full-stage output limit:** twice the largest measured per-step output in the profile's screen runs, between 8,192 and 32,768, rounded up to 1,024.

## Reference
Model-card output and context figures (retrieved 2026-10-04, quotes via a summarizing fetch): only Qwen3.6-35B-A3B recommends an output length (32,768, 81,920 for hard benchmarks); Qwen3.8 and North-Mini-Code state maximums (131,072 and 65,536); Gemma 4, Muse-Glimmer, occamy, Ornith and Swift state none. Native context is 131,072 or more for every family.
