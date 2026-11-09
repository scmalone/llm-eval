# 2026-11-05: The resize sweep gave different answers on a heavier desktop, and the owner adopted the clean ones

- Covers: 2026-11-05 13:43 to 2026-11-06 04:10 UTC, sessions 37c105ae

## What happened
- The owner asked for `noise-speed`. It was already done on 2026-11-03 with a decode spread of 0.46%, well under the 3% that
  would make Opus widen GAIN (log), so it was not re-run.
- `sweep-resize` ran twice. The first pass (55 min) had 21,745 MiB free with the router stopped, under the 22,300 MiB the
  sweep wants, and cut most contexts. After Steam and Slack were closed the re-run had 22,617 MiB free and moved the answers
  a lot: `qwen3.6-35b-a3b` now fits at 262144 unchanged (log). The first
  pass's data is kept as `2026-11-05-speed-sweep-heavy-desktop` in the data folder.
- The owner asked why a sweep takes so long. 26 of 51 first-pass probes failed the 1,024 MiB floor at load or first chat but
  still ran the whole depth ladder, about half the probe time (per session; the probe count is from
  `2026-11-05-speed-sweep-heavy-desktop/results.jsonl`, corrected from 20 by the 2026-11-06 review). A probe now stops at the first breach (`980902b`).
- The owner asked whether MoE layers were moved and why not more CPU layers. A 13-minute probe (log)
  showed one extra CPU layer keeps 262144 on occamy, ornith and north-mini; gemma needs two.
- The owner chose option B, the sweep's own proposals, over the full-context variants. Eight preset edits went into
  `models-preset.ini` (a separate repo), the pi copy was synced, and `owner-adopt-resize` closed (`f4d41c0`).

## What the owner challenged
- "confirm the findings via A first before doing anything." The diagnosis was checked against the saved probe data before any
  edit. It held, and the proposed shorter ladder was dropped because passing probes reach their lowest VRAM in the ladder.
- "Is the headroom against now or is it estimated against proposed." It is measured at the proposed setting, on the
  sweep-time desktop, not the current one.
- "What about adding more MoE's to the cpu instead?" The sweep's rescue order never reached that step, because every
  failing preset fit at a smaller context first.

## What the agent got wrong
- It told the owner Brave was fine to keep open during the sweep. The desktop used 0.6 to 1.4 GB more than in earlier sweeps
  and most baselines failed the floor, so the first pass had to be redone.
- It labelled ubatch rows "+13% speed". For occamy, ornith and north-mini most of that headline came from the smaller
  context, and the ubatch step alone gave +3 to +4%. It corrected this when asked what the percentages meant.
- It estimated the MoE probe at 30 to 40 minutes; it took 13.

## Changed
- `ddba644`, `a08140e`, `59584ff`, `35314ec` sweep-resize logs; `980902b` early stop; `a108943` MoE probe; `f4d41c0` adoption.
- Adopted: gemma ctx 196608; occamy and ornith ctx 196608 with ubatch 2048; north-mini ctx 196608 with `n-cpu-moe` 5;
  qwen3.8-27b-unsloth-iq4xs ctx 131072 with draft depth 4; q4-mtp ctx 163840; Muse ubatch 512; qwen3.8-27b-q4 q4_0 KV with ubatch 256.
- The sync also changed the dotfiles pi models file. That change is still uncommitted, because a hook blocks the agent from
  committing it.

## Learned
- A sweep's answer depends on the desktop it ran against. The report warns under 22,300 MiB free, and that warning should
  mean a re-run, not a note.
- Muse's ubatch 512 has 74 MiB above the floor (per session), the thinnest of the lot.

## Data
- Stands: the clean re-run (14:37 to 15:38). Superseded: the heavy-desktop pass.
- The early stop has not been timed on a full sweep, and aborted probes no longer record a true lowest-free value.
- `swift-qwen3.8-27b-vision` did not fit in either pass.
