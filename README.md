# llm-eval

A benchmark for one question: which local models can do my real agent work on a single 24 GB RTX 4090, and what do
the memory settings that make them fit cost in quality?

AI agents wrote almost all of the code here. I set the goal, the rules, the acceptance checks and the stop
conditions, read the results, and decided what counted. The repo is a record of that process as much as of the
results.

## Results (bench-v2, 2026-11-07 to 11-09)

Core is 35 tasks from my own work, run three times each (105 trials), graded by code checks. Every row ran on the
same frozen harness and grader (git tag `bench-v2`, grader `04468edc`).

| Model (profile) | Engine, context | Core raw | Core adjusted | Recall | Tool calls | Chat sweep | Wall time |
|---|---|---|---|---|---|---|---|
| Qwen3.8 27B, Unsloth Q4_K_M (`qwen3.8-27b-unsloth-q4-mtp`) | llama.cpp, 163,840 | 91/105 | 98/105 (101 upper bound) | 3/3 | 10/10 | 15/15 | 9.7 h |
| Qwen3.8 Flash Next IQ3_S (`strata-iq3s`) | Strata 0.1.54, 262,144 | 91/105 (1 incomplete) | n/a | 3/3 | 10/10 | 12/15 | 5.0 h |
| Swift 1.5, a Qwen3.8 27B fine-tune, Q4_K_M (`swift-qwen3.8-27b-vision`) | llama.cpp, 131,072 | 84/105 | n/a | 3/3 | 10/10 | 15/15 | 7.1 h |
| Muse Glimmer 30B (`muse-glimmer-30b`) | llama.cpp, 131,072 | 81/105 | n/a | 0/3 (17 of 36 records) | 10/10 | 15/15 | 3.3 h |

How to read it:
- The contexts differ, so this is not a head-to-head ranking. A row compares only with rows run at the same context.
- "Adjusted" removes trials that failed only the style check (an em dash in the answer) and trials where the grader
  or the task was confirmed wrong (Qwen3.8's removals: `DECISIONS.md`, 2026-11-08). Both numbers are reported because the raw
  one mostly measures formatting, and dropping it would hide that a model ignores a rule I set. Adjusted figures
  for the other three rows haven't been worked out yet.
- Qwen3.8's 101 upper bound also removes the three haystack trials, which are mostly a task fault. One of them has a
  real model error too, so 98 is the headline.
- Qwen3.8 and Swift ran with speculative decoding (MTP) off. Qwen3.8 decoded at about 45 tokens/s instead of about
  80 with MTP on, which is why its row took 9.7 h.
- Muse's recall miss at about 112K tokens repeats the two diagnostic runs before it. It's a measured limit, not noise.
- Strata is a third-party inference engine, driven through this harness's Strata adapter. Its incomplete trial is a timeout,
  counted as a failure.

Per-model setup notes are in `models/`, and the earlier Strata results in `results/`.

## How it was built

The work ran as a loop: I wrote or approved a plan, agents built it, a model from a different family reviewed it,
and I made the call. `DECISIONS.md` keeps the rules that came out of that, each with its reason. `docs/devlog/`
tells the story one milestone at a time, including what the agents got wrong and what I pushed back on.

Some of the calls that shaped it:
- Live runs fail closed. The plain CLI refuses a live plan. A live run goes through an acceptance step with an
  approval token, and the runner restores the production model server when a run ends or fails.
- Runs are checkpointed and resumable. State is written atomically, and every event is appended to an fsynced log.
- The live adapter had two independent reviews before it was enabled, and the unattended runner had signal drills.
- I read full answers before trusting scores. In the 2026-11-06 review, six of the 35 core tasks had passed on no
  model, and five of those six were the test or the grader. The advisory format check had failed 19 of 50 trials
  whose content was correct. A one-line grader bug failed eight correct document answers.
- The harness's reply is part of the experiment. Every model that acted on a question did it right after the reply
  "Use your judgment and continue", so question tasks now reply "finish with your answer and change nothing".
- I refused to compare rows across harness changes. The 11-06 fixes changed 22 of the 35 core tasks, so the
  comparison runs were frozen at one tag, and older rows stay as a record instead of a baseline.
- An operator agent works the run queue. It may restart the model server and fix operational defects, each in its
  own commit with tests, but never tasks, graders, verdicts or limits.
- Settings are like for like. Each model runs at its vendor's sampling defaults, and any sampler the vendor
  doesn't name is switched off rather than left at llama.cpp's defaults.

Built with Claude, Codex and Deepseek. pi is the agent harness under test.

## What it measures

- Core: 35 tasks in seven categories taken from my own agent logs (advisory, implementation, lookup, ops, resume,
  review, surgery), five each. Each task is proven to fail when the agent does nothing and to pass when the known
  solution is replayed.
- Diagnostics: a recall haystack sized to the context, ten tool calls, and five synthetic chat sweeps.
- Accuracy suites: depth tasks that start near the harness's compaction point, long-document questions, 20 fixes
  mined from my other repos, six game tasks and ten vision tasks.
- Memory settings: KV-cache precision (q8_0 by default, q4_0 only as a last step down), ubatch size, CPU offload of
  MoE layers, speculative decoding (MTP and DFlash drafters), and context from 131,072 up to 262,144 tokens.
- The pass mark is 1,024 MiB of GPU memory left free beside a working desktop.

The agent under test runs in a bubblewrap sandbox with a private network. Its only exits are the model server and a
canned-page proxy. It works over a fake home that holds just the task's repo and a few pinned files. llama.cpp is
pinned at build 12017, and every run records its harness commit, grader hash and mode.

## Status and limits

- The bench-v2 comparison is finished for four candidates. Next is an audit of the suites (the haystack task
  contradicts its own packet, and one matcher is inverted), then knowledge suites.
- One core run per model can't separate models within 3 points, which is why each bench row runs core three
  times. Even so, 35 tasks is a small suite.
- This repo is for reading, not for running. It expects my machine's model server, presets and run data, which
  aren't included. The mined and game tasks also need source repos that aren't here.
- This public copy has a single commit. Commit IDs in the docs refer to the private development repo, and run
  folders, transcripts and operator logs stay private.

Layout: `llm_eval/` is the harness, `fixtures/` holds the tasks, `suites/` and `configs/` hold the run definitions,
and `tests/` runs without a GPU (`python3 -m unittest discover -s tests`).

## Credits and license

The benchmark drives tools it doesn't include: llama.cpp, the Strata engine, pi and opencode. The models belong to
their publishers, linked from each page in `models/`. The code and docs here are MIT; see `LICENSE`.
