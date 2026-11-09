# llm-eval

Which local models can handle real agent work on a single 24 GB RTX 4090, and what do the memory settings required to fit cost in quality?

AI agents wrote almost all of the code. The project owner set the goals, rules, acceptance checks and stop conditions, reviewed the results, and made the final calls. This repo documents both the benchmark and the process behind it.

## Results at a glance

[Download the aggregate results](results/bench-v2.json), including campaign dates, category counts, adjustment rules and measurement limitations. Raw trial records remain private.

The core comparison uses 35 tasks drawn from real agent work, five in each of seven categories, run three times each. Every row below uses the same frozen task set and grader (`bench-v2`, grader `04468edc`).

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

## Development and evaluation method

The harness was built with Claude, Codex and DeepSeek. pi is the agent framework under test. Plans define scope, acceptance checks and stop conditions; code changes are reviewed before runs, and final acceptance remains human-reviewed. [DECISIONS.md](DECISIONS.md) records the rules and their rationale. The [development log](docs/devlog/) tracks the milestones and resulting changes.

| Area | Method |
|---|---|
| Independent review | Two reviews preceded live-adapter use. The unattended runner also passed signal drills. |
| Live-run safety | Runs require an acceptance step and approval token. The runner restores the production model server on completion or failure. |
| Recovery | Runs checkpoint state atomically, append events to an fsynced log, and can resume after interruption. |
| Score integrity | Full answers are reviewed when a result is disputed. An audit found six core tasks that failed on every model; five involved a test or grader issue. A one-line grader bug rejected eight correct document answers. |
| Comparability | Fixes changed 22 of 35 core tasks, so `bench-v2` froze the task set and grader. Earlier scores remain historical records, not a baseline. |
| Operator scope | An operator agent may restart the model server and commit operational fixes separately. It cannot change tasks, graders, verdicts or limits during a run queue. |
| Sampling | Each model uses its publisher's defaults. Samplers without a published recommendation are disabled instead of inheriting llama.cpp defaults. |

## What it measures

- Core: 35 tasks in seven categories drawn from real agent work logs. Each task is verified to fail when the agent does nothing and pass when the known solution is replayed.
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
