# 2026-11-06: Swift loaded after two step-downs and scored with the top two

- Covers: 2026-11-06 16:24 to 20:09 UTC, sessions 37c105ae

## What happened
- The owner said "Do this - sweep-swift-vision". The sweep measured each step on a copy of the preset:
  - Dropping MTP alone left 423 MiB free, under the 1,024 MiB floor.
  - Also moving the vision projector to system RAM left 1,527 MiB free, at 40.7 t/s decode.
  - The context stayed 131072 and the KV cache q8_0 (2026-11-06-sweep-swift-vision.md (private historical record)).
- The owner approved editing the real preset (option A). The screen then ran from 11:27 to 13:50 local:
  - Smoke 1/1.
  - Recall passed at 131072. Swift was the second model to pass, after Qwen3.8.
  - Core 23/35 (2026-11-06-screen-swift-vision-2.md (private historical record)).
- The owner asked for a synopsis of the failures: model or test? The agent's preliminary read named suspects, and
  became a prompt for an Opus review of everything to date (`docs/prompts/2026-11-06-opus-full-review.md`).
- In between the owner asked for recall results by model, the strongest models, and a time estimate for Swift.

## What the owner challenged
- "Give me a brief synopsis on the overall test failures and if they are a issue with the parameters of the test
  and/or legitimate failures."
- "I am surprised there are so many models doing poorly in the tests. I'm not surprised qwen 3.8 is running so well
  however. That's expected." This led straight to the full review in the next entry.

## What the agent got wrong
- It first said phase 1 had no numbers, then corrected it: chat sweep 5/5, recall passed, tools 10/10.
- It wrote a malformed operator-log index line (a literal `\n`), fixed in `3a72c08`.
- It called Swift's score "not directly comparable" because MTP was off and the projector sat in RAM. The review
  found the accuracy comparable: MTP changes speed only, and the screen sends no images. Only speed differs.

## Changed
- `36edd4a` in `~/.config/llama-cpp`: the Swift preset has MTP off and the projector in RAM, owner approved.
- `4dcc571`, `c5084cf` sweep; `d80829c`, `bb12d28`, `3a72c08` screen; `fb11135` Opus review prompt; `c9d0f66` STATE.

## Learned
- The step-down order paid off. Two lossless steps fit the model, where the q4_0 cache the agent first proposed was
  never needed.

## Data
- Stands: Swift core 23/35 as recorded, which the next entry re-grades to 26.
- Speed numbers for Swift now describe MTP off. Its model page was regenerated in `3ac5232`.
