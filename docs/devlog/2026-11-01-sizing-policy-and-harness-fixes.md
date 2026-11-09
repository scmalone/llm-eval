# 2026-11-01: Sizing policy, the disk cap, version-split scores and a check suite draft

## What happened

After the config pass the owner set how a model is sized and asked whether the approach was over- or
under-engineered. The answer changed the harness in five commits, none of them a GPU run.

- Sizing policy (`DECISIONS.md`): KV q8_0 on every preset (six moved up from q4_0); a preset short of the 1,024 MiB
  floor steps down context to 131,072, then CPU MoE layers, then ubatch (never on Gemma), then KV, q4_0 last.
  Gemma 31B left the test (it loads only 65,536).
- B1: a task home over 1 GiB ends the trial as `capacity_fail` within about 2 s; the sandbox `/tmp` is capped at 1 GiB.
- B2: every verdict carries a hash of the grading code, and the scorecard splits on harness, client version, engine
  build, preset settings, model and template file hashes, effort and graders.
- pi's context window: `tools/sync_pi_context.py` (owner-run) sets it from the preset; a run refuses a mismatch.
- Sweep: the estimate-driven jumps went (they never ran), KV no longer climbs, and `--repeat` measures noise.
- Draft check suite for any candidate (`plans/2026-11-01-check-suite.md`), awaiting the owner.

## Back and forth

- The owner asked whether the pi context switch could be automatic. Inside tests it already was (each run builds
  its own pi config at the run's context); the owner's own pi config is behind the credential wall, so it is a
  one-command script plus a refusal, not an agent edit.
- Over or under: the sweep's jump logic was dead weight (buffer sizes have been null in every result since 10-31);
  the gaps were no accuracy check per candidate and no noise measurement under a 3% speed rule.
- The check suite was first proposed from existing tasks only; the owner asked about the mined fixes and Blender,
  and three slots now wait for them (E3 to E7).
- Agent mistake: said the q8_0 preset change would take effect on a router restart; the checklist says the router
  reads the preset on its next load, so it may already be live for daily use.

## Learned

- Build 12017's server log at the default verbosity prints neither model facts nor buffer sizes; anything that
  parsed them has been silently empty. The sweep now reads expert count and trained context from the GGUF header.
- Task homes run 1 to 21 MiB (235 homes measured); the two outliers were the runaway task.

## Sources

- Commits: preset `4cc0231`; llm-eval `1f9843e`, `a6b0e3a`, `2c212f6`, `688e45a`, `ea947f3`, `acf351e`.
