# llm-eval

Which local models can do real agent work on one 24 GB RTX 4090, and what do the memory settings that make them fit cost in quality?

AI agents wrote almost all of the code. I set the goal, rules, acceptance checks and stop conditions, read the results, and decided what counted. This repo documents both the benchmark and that process.

## Results at a glance

The core comparison uses 35 tasks from my own agent work, five in each of seven categories, run three times each. Every row below uses the same frozen task set and grader (`bench-v2`, grader `04468edc`).

| Model and profile | Engine, context | Core raw | Core adjusted | Wall time |
|---|---|---:|---:|---:|
| Qwen3.8 27B, Unsloth Q4_K_M (`qwen3.8-27b-unsloth-q4-mtp`) | llama.cpp, 163,840 | 91/105 | 98/105 (101 upper bound) | 9.7 h |
| Qwen3.8 Flash Next IQ3_S (`strata-iq3s`) | Strata 0.1.54, 262,144 | 91/105 (1 incomplete) | n/a | 5.0 h |
| Swift 1.5, Qwen3.8 27B fine-tune (`swift-qwen3.8-27b-vision`) | llama.cpp, 131,072 | 84/105 | n/a | 7.1 h |
| Muse Glimmer 30B (`muse-glimmer-30b`) | llama.cpp, 131,072 | 81/105 | n/a | 3.3 h |

The contexts differ, so these totals are not a head-to-head ranking. Adjusted scores remove confirmed style or grader faults; only Qwen3.8 has a reviewed adjusted score so far. Its 101 upper bound also removes three haystack trials, one of which contains a possible model error, so 98 is the reported adjusted result.

### Core score by category

Each cell is passed trials out of 15. Raw counts include style and task/grader issues. The haystack task and one inverted matcher need follow-up, so read the totals with the caveats above.

| Category | Qwen3.8 163K | Strata 262K | Swift 131K | Muse 131K |
|---|---:|---:|---:|---:|
| Advisory | 8/15 | 11/15 | 7/15 | 11/15 |
| Implementation | 11/15 | 12/15 | 10/15 | 11/15 |
| Lookup | 15/15 | 15/15 | 14/15 | 15/15 |
| Operations | 15/15 | 14/15 | 11/15 | 8/15 |
| Resume | 15/15 | 14/15 | 15/15 | 15/15 |
| Review | 12/15 | 12/15 | 12/15 | 8/15 |
| Surgery | 15/15 | 13/15 | 15/15 | 13/15 |

### Diagnostics

Three repeated recall prompts probe long-context retrieval; ten tool calls check basic tool-loop reliability; five chat tasks are repeated three times.

| Model and context | Recall | Tool calls | Chat sweep |
|---|---:|---:|---:|
| Qwen3.8, 163K | 3/3 (36/36 records) | 10/10 | 15/15 |
| Strata, 262K | 3/3 (36/36 records) | 10/10 | 12/15 |
| Swift, 131K | 3/3 (36/36 records) | 10/10 | 15/15 |
| Muse, 131K | 0/3 (17/36 records) | 10/10 | 15/15 |

Muse's recall miss near 112K tokens repeats earlier diagnostics. Qwen3.8's run used MTP off and decoded at about 45 tokens/s; its estimated speed with MTP on is about 80 tokens/s, but those settings are not compared in this table.

### Additional accuracy suites

These are broader task suites, reported as passed/total. They were run in separate campaigns, not as part of the three-run core comparison. A dash means the suite was not run for that profile. `*` marks a result affected by a known measurement or grader issue.

| Model | Depth | Documents | Mined fixes | Game tasks | Vision |
|---|---:|---:|---:|---:|---:|
| Qwen3.8, 163K | 3/5 (1 unmeasured) | 3/6* | 18/20 | 6/6 | — |
| Strata, 262K | 4/5 | 6/6 | 18/20 | 6/6 | — |
| Swift, 131K | 3/5 (1 unmeasured) | 3/6* | 16/20 | 4/6 | 10/10 |
| Muse, 131K | 3/5 (1 unmeasured) | 4/6 | 16/20 | 5/6 | 9/10 |

`*` Qwen3.8 and Swift each have three document answers that appear correct but fail the one-line decoy grader. Depth includes one task that did not reach its required token depth and another that fails on a missing model path in the test environment. These counts are raw grader results, not adjusted scores.

## How it was built

The work ran as a loop: I wrote or approved a plan, agents built it, a model from a different family reviewed it, and I made the calls. [DECISIONS.md](DECISIONS.md) records the rules and why they changed. The [development log](docs/devlog/) follows milestones and records what agents got wrong and where I pushed back.

- Live runs fail closed. They require an acceptance step with an approval token, and the runner restores the production model server when a run ends or fails.
- Runs are checkpointed and resumable. State is written atomically, and every event is appended to an fsynced log.
- Two independent reviews preceded enabling the live adapter. The unattended runner also went through signal drills.
- I read full answers before trusting scores. A review found six core tasks that passed even when the model did no work; five had a test or grader problem. A one-line grader bug rejected eight correct document answers.
- I froze the comparison after fixes changed 22 of 35 core tasks. Older results remain in the record but are not treated as a baseline.
- An operator agent can restart the model server and fix operational defects in separate commits, but cannot change tasks, graders, verdicts or limits during a run queue.
- Sampling follows each model publisher's defaults. Samplers the publisher does not specify are disabled rather than inherited from llama.cpp.

Built with Claude, Codex and DeepSeek. pi is the agent harness under test.

## What it measures

- Core: 35 tasks in seven categories based on my agent logs. Each task is verified to fail when the agent does nothing and pass when the known solution is replayed.
- Diagnostics: context-sized recall, ten tool calls and repeated synthetic chat sweeps.
- Accuracy suites: depth tasks near the harness compaction point, long-document questions, 20 fixes mined from other repos, six game tasks and ten vision tasks.
- Memory settings: KV-cache precision, ubatch size, CPU offload of MoE layers, speculative decoding and context from 131,072 to 262,144 tokens.

The pass mark is 1,024 MiB of free GPU memory beside a working desktop. The agent runs in a bubblewrap sandbox with a private network. Its only exits are the model server and a canned-page proxy. llama.cpp is pinned at build 12017, and each run records its harness commit, grader hash and mode.

## Status and limits

- Four profiles have completed the core comparison. The suite audit is next; the haystack task contradicts its own packet, and one matcher is inverted.
- The core suite is small. Three repeats help reveal variability, but 35 tasks cannot represent every kind of agent work.
- This repo is for reading. Running it requires the original machine's model server, presets and run data, which are not included. Mined and game tasks also depend on source repos that are not included.
- The public repo excludes the private development history. Commit IDs in the documents refer to the private development repo. Run folders, transcripts and operator logs stay private.

The harness lives in `llm_eval/`; `fixtures/` holds test tasks, and `suites/` and `configs/` hold run definitions. The Python package metadata in `pyproject.toml` defines the installable CLI.

## Credits and license

The benchmark drives tools it does not include: llama.cpp, Strata, pi and opencode. The models belong to their publishers, linked from their pages in [`models/`](models/). The code and docs are MIT licensed; see [LICENSE](LICENSE).
