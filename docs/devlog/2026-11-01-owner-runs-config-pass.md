# 2026-11-01: The owner-started runs, the config pass, and the 1 GiB floor

## What happened

One sitting, readiness checklist C and D, started by the owner and run by the agent through `plan_ctl` and
`speed_sweep.py`:

- C1 signal drill (16:11 to 16:25): SIGINT and SIGTERM both passed, restored in about 11 s, production healthy.
- C2 config sweep (16:25 to 17:03): 21 settings, each preset as written beside one change (C2 to C6). Router
  restored healthy (service, health, four-token probe).
- C3 Strata re-run of the 14 tasks whose identity changed (17:05 to 17:56): core 6/7, documents 6/6, depth 1/1.
- D: adopted `load-mode = none` on Qwen3.6 (4.9% faster), `image-min-tokens = 1024` on Qwen3.6 and Swift, and
  `ubatch-size = 1152` on Gemma 31B (full 1,120 image tokens); the vision changes wait on the owner's image check.

## Back and forth

- The sweep said 16 of 21 settings "do not fit", the production presets as written included. The cause was the
  desktop, not the presets changing: it held about 1.8 GB (Slack and Brave open) against 0.9 to 1.2 GB on 10-31, and
  every 10-31 tuned preset lost about 910 MiB. The sweep's start-of-run warning (under 22,300 MiB free) missed it
  by 44 MiB.
- The agent first judged variants by their memory change against their own baseline, and left Gemma C2 out for
  costing 390 to 460 MiB. The owner then set the policy: plan for Brave open, and 1 GiB free is a number to track,
  not a pass or fail. The runner and the sweep changed to match and Gemma C2 went in. Within the hour the owner
  reversed it: 1 GiB free is the pass/fail mark, with Brave open as normal use. The change was reverted (`47e0764`),
  which put the runner back byte for byte on the drilled restore code, so the C1 drill still holds; the sweep's floor
  became the bare 1,024 MiB gate (the 400 MiB margin assumed a lighter desktop than the one it now runs beside).
  Gemma 26B C2 came back out (855 MiB); 31B C2 stayed (1,201). Most presets now need a re-sizing sweep.
- The owner plans to work remotely over Tailscale. tailscaled holds no GPU memory and Brave stays closed then; a
  Chromium the model starts for a visual test runs in the task sandbox, which has no GPU device, so it costs none.
- The owner asked what size is safe with Brave and a text editor open. Answer: 1 GiB at the run's lowest point,
  under about 500 MiB risky. Brave's GPU process measured 120 to 135 MiB; a 30 s `nvidia-smi -l 1` sample by the
  owner moved 68 MiB.
- Later the owner set the KV default to q8_0 (six presets moved up from q4_0) and the step-down order for a preset
  short of the floor: context to 131,072, CPU MoE layers, ubatch, then KV, q4_0 last. The sweep also turned out never
  to have read model facts or buffer sizes on build 12017 (the log no longer prints them at the default verbosity);
  it now reads expert count and trained context from the GGUF header, so Gemma 26B gets CPU MoE steps too.
- Agent mistake: the preset repo held the 10-31 sweep's edits uncommitted for a day; found when committing D1 and
  committed separately first.
- `tools/model_cards.py` reads only `*-speed-sweep` folders, so the config pass would not have reached the model
  pages; it gained a `config_results` field in `notes.json` instead.

## Learned

- A sweep's absolute fit verdict is only as good as the desktop it ran beside; compare a change with its baseline
  in the same run, and record the desktop's own usage (`gpu.jsonl` first reading with the router stopped).
- `load-mode = none` helps prefill 6 to 16% on every `n-cpu-moe` profile, but only Qwen3.6 clears 3% on the
  combined score.
- C5: the Qwen3.8 Unsloth files' own MTP layer drafts exactly as well as the separate drafter (62.2% and 65%
  acceptance either way) and frees about 770 MiB, at 1.7 to 2.5% slower. Not adopted under the 3% rule.
- Ornith's MTP layer (C4) does not fit at 262,144.

## Numbers

| Variant | Prefill t/s | Decode t/s | Accept | Lowest free MiB | Score change | Adopted |
|---|---|---|---|---|---|---|
| qwen3.6 C3 image-min-tokens | 1,481 → 1,481 | 85.8 → 86.9 | - | 801 → 814 | +1.0% | yes, pending image check |
| qwen3.6 C6 load-mode none | 1,481 → 1,716 | 85.8 → 87.5 | - | 801 → 777 | +4.9% | yes |
| gemma-26b C2 ubatch 1152 | 3,830 → 4,091 | 76.3 → 76.4 | - | 1,315 → 855 | +0.7% | no, under the 1,024 floor |
| gemma-31b C2 ubatch 1152 | 2,076 → 2,055 | 34.1 → 33.9 | - | 1,591 → 1,201 | -0.6% | yes, pending image check |
| swift C3 image-min-tokens | 1,795 → 1,795 | 63.7 → 63.7 | 52% | 637 → 641 | 0.0% | yes, pending image check |
| ornith C6 load-mode none | 2,840 → 3,061 | 101.1 → 101.5 | - | 591 → 601 | +1.5% | no, under 3% |
| ornith C4 MTP depth 2, 3 | - | - | - | OOM at load | - | no |
| qwen3.8 Q4 C5 no separate drafter | 1,826 → 1,820 | 67.2 → 65.3 | 62% → 62% | 833 → 1,601 | -2.5% | no, slower |
| qwen3.8 IQ4_XS C5 no separate drafter | 1,332 → 1,330 | 60.0 → 58.7 | 65% → 65% | 581 → 1,357 | -1.7% | no, slower |
| occamy C6 load-mode none | 2,854 → 3,064 | 95.4 → 96.6 | - | 598 → 598 | +2.2% | no, under 3% |
| north C6 load-mode none | 2,816 → 2,987 | 79.3 → 80.0 | - | 555 → 556 | +1.6% | no, under 3% |

Strata on current task identities: core 28/35, documents 6/6, depth 5/5. The one re-run failure,
router-model-path, is the model's: it never opened the preset or the unit and configured opencode instead.

Run folders (`~/Workspace/llm-eval-data/`): `2026-11-01-strata-iq3s-262144-drill-{sigint,sigterm}-attempt2`,
`2026-11-01-config-pass/` (report.md, results.jsonl, raw/, logs/), `2026-11-01-strata-iq3s-262144-{core-subset,
documents-subset,depth-subset}`.

## Sources

- Prompt: `docs/prompts/2026-11-01-owner-runs.md`. Results: `results/strata-iq3s/2026-10-31-plan.md`.
- Commits: llm-eval `4e4e07c` (model cards), `e113d40` and its revert `47e0764`, `642cc2b` (sweep floor);
  `~/.config/llama-cpp` `6771bb5`, `71bbb21`, `3539bb4`, `411065b`.
