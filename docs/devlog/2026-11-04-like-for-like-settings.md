# 2026-11-04: The first two screens were not like for like, and every model's settings checked against its vendor

- Covers: 2026-11-03 23:11 to 2026-11-04 16:45 UTC, sessions 885e1c2d

## What happened

The first two llama.cpp screens finished overnight: qwen3.8-27b-unsloth-q4-mtp at 22/35 core
(2026-11-03-screen-qwen3.8-27b.md (private historical record)) and qwen3.6-35b-a3b at 13/35, with recall 0/3 at 262,144
and several ops trials looping until the wall cap (2026-11-04-screen-qwen3.6-35b.md (private historical record)).
Asked why Qwen3.6 did so badly, the agent read the captured requests: Qwen3.6's preset carried the card's
precise-coding sampling (temperature 0.6, no presence penalty), while Qwen3.8 ran the card's general thinking mode
(temperature 1.0). The two scores compare two different configurations, not two models.

## What the owner challenged

- "So I ran a benchmark of a coding specific config whereas other models are thinking?" Yes. The model-card notes
  already listed Qwen3.6's modes; nobody had compared them with the preset.
- The queue's Q6/Q9/A1 labels were hard to follow in long lists. Every row, log file and reference was renamed to
  what it runs (`screen-qwen3.6-35b`, `sweep-resize`, ...).
- Standard: context sizes are a separate question (VRAM decides them); everything else is each model's vendor default
  for general thinking.
- "The harness should just send the exact parameters needed", and, when the agent proposed building a pinned thinking
  block, "all of this information already exists via the vendors". The agent dropped the build and set each model to
  its vendor default instead.
- The owner had Perplexity research the vendor defaults per model (sampling, thinking controls, output and context
  guidance) and asked for it to be reasoned against the project.

## What the agent got wrong

- Did not check each preset's sampling against the card's general-thinking mode before the screens ran.
- Said it could not tell what reached each model; every trial's `raw/inference/contract.json` and `request.json`
  record the exact body sent.
- Began proposing harness features before checking the vendor data the model-card sweep had already gathered.
- Added a co-author trailer to four commits against the owner's rule (rewritten before push).

## Changed

- Qwen3.6 preset to thinking general: temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence 1.5 (preset repo
  `a9be2ca`; llm-eval `a8a1403`). Rerun queued as `screen-qwen3.6-35b-card-sampling`, waiting for the owner.
- Muse Glimmer: the owner's thinking level `max` used to reach `xhigh`; the benchmark's pi copy now maps xhigh to null,
  so it resolves to the template default `high` (`83fcb24`). The owner's live pi config still sends xhigh.
- Gemma, Muse and North presets: `min-p = 0`, and North `top-k = 0` (preset repo `a675e89`). Their vendors name no
  min-p (North no top-k), the GGUFs embed none, so llama.cpp had been applying its own 0.05 and 40.
- Run report (`7f441c5`): each model shows the thinking fields pi sends (for example
  `enable_thinking=True, reasoning_effort=xhigh, thinking_budget_tokens=16384`), and the workload scorecard counts
  thinking-cap turns.

## Learned

- After these changes every screened model runs its vendor's general-thinking sampling and its template's default
  thinking mode: Qwen3.6, Occamy, Ornith and North thinking on; Qwen3.8 and Swift `reasoning_effort` xhigh; Muse
  `reasoning_strength` high; Gemma `enable_thinking` true (its template defaults to off). Ornith's own card says 0.6.
- pi sends `thinking_budget_tokens: 16384` at high and above for every model, and llama.cpp enforces it by forcing
  the thinking block closed. No vendor publishes that number. It was kept (`DECISIONS.md`): it fired on 12 of 2,471
  Qwen3.6 turns, mostly the loops, and 2 of 498 Qwen3.8 turns, so it works as a loop breaker.
- `max_tokens` is 32,768 for all models, matching Qwen3.6's general recommendation; the vendors' larger figures
  (81,920, 131,072, 250,000) are benchmark settings or tied to 1M context.
- pi sends earlier reasoning back on every turn; each template keeps or drops it as its vendor intends.

## Data kept

Only `screen-qwen3.6-35b` is superseded; it stays on disk as the coding-mode data point. The Qwen3.8 screen, the
Strata runs and the speed sweeps are unaffected (no task, grader or Qwen3.8/Strata setting changed). The pending
`advisory-lle002-option` grader fix is a re-grade of saved answers, not a rerun.
