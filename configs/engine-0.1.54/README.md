# Frozen configurations, engine 0.1.54 (Strata v0.2.8)

Two arms of one controlled comparison. They differ in `--max-context` and in the numbers that follow from it, and in
nothing else. Frozen 2026-10-30 after the measurements in
`plans/2026-10-30-fresh-start-engine-0.1.54.md`; every number below was measured on this machine, none is carried over
from 0.1.45.

| | `iq3s-131072.json` | `iq3s-262144.json` |
|---|---|---|
| context | 131,072 | 262,144 |
| ladder for the runner (`contexts`, anchor first) | `[131072, 114688]`, anchor 131072 | `[262144, 131072]`, anchor 262144 |
| measured expert cache | 6,320 to 6,370 slots / 12.00 to 12.09 GiB | 5,392 to 5,475 slots / 10.25 to 10.41 GiB |
| measured free VRAM after load | 2,050 to 2,090 MiB | 1,974 to 2,083 MiB |
| measured KV cost of the step | - | about 1.9 GiB of cache against the 131K arm |

## Settings, identical in both arms

- Engine: `~/Workspace/Strata/engine/strata`, 0.1.54, built from tag `v0.2.8` (`870aedd9`). Reinstalled
  2026-10-30 18:18 by Strata's `setup.py --update` (its own build flags, `-DSTRATA_Q6K_EXPERTS=ON
  -DSTRATA_MMQ_KQUANTS=ON`): binary sha256 `d6625999...`, with a matching `BUILD.json` and `libstrata_kq.so`. The
  2026-10-29 verification below was on the earlier hand build (`bf18d30f...`, plain `cmake`), which setup.py
  replaced; re-verify the cache and free-VRAM numbers on the next load.
- `--expert-cache auto --vram-reserve-mib 2048`. The reserve is what makes the runner's 1 GiB gate pass; it costs about
  854 slots at 131K against the 700 default. See the plan's step 3 table.
- `--pcie-frac 0.20`, pinned. With the flag given the engine skips its start-up probe and its refinement, so the share
  cannot drift with the link's power state. 0.20 is where a warm start settles at both contexts on this machine.
  A cold start without the pin settles at 0.10 for the whole session (plan, step 5).
- `--kv int8 --spec 4 --spec-min-p 0.5 --prefill auto --mtp <draft>`, one serving slot.
- Environment: `STRATA_ROUTE_TAIL_SKIP=0` (the accuracy-preserving side; it costs 8 to 12% decode), `STRATA_KVA=0`,
  `STRATA_ROUTE_PRIOR=0`.
- Sampling, written out rather than left to defaults: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0,
  presence_penalty 0.0, repetition_penalty 1.0. Thinking on: `reasoning_effort: high`, `reasoning_budget_tokens: 0`
  (0 means no cap).
- `fit_max_tokens: false` and `answer_room: false`, matching what the eval adapter sets for a run.

## Running one directly

```bash
.venv/bin/python serve/server.py --engine strata --config configs/engine-0.1.54/iq3s-262144.json --host 127.0.0.1 --port 8091 --parallel 1
```

The engine log goes to the `log` path in the config, under
`~/Workspace/strata-eval-runs/2026-10-30-engine-0.1.54/frozen/`.

## How the adapter must mirror them

`run_config()` in `llm_eval/strata.py` copies `settings["resolved_config"]`, strips `--max-context`, `--host`,
`--port`, `--batch`, `--parallel` and `--vision`, then re-adds `--max-context` from the plan's rung. So the plan's
`resolved_config.args` must carry `--vram-reserve-mib 2048` and `--pcie-frac 0.20` itself, and `settings.env` must carry
the three variables above. Sampling and thinking come from `settings.request`, not from the profile.

Verified 2026-10-30 by loading both files: both reach READY, the log shows zero `PCIe probe` and zero `PCIe share`
lines (the pin is in force), and the cache and free-VRAM numbers above are from those two loads.
