# Strata quality and performance testing

Updated 2026-10-28. Goal: choose the most correct usable IQ3_S configuration on the
RTX 4090, Ryzen 9 7950X3D, 64 GB Linux desktop. Correct completion and total task time
are the selection criteria. Decode throughput explains performance but cannot establish accuracy.

This procedure incorporates the owner's two research reports supplied on 2026-10-28.
Their version numbers, API behavior and performance claims are leads to verify, not local results.
The October 18 Strata 0.1.45 artifacts under `~/Workspace/Strata/bench/results/2026-10-18-*`
are historical speed measurements: thinking off, temperature zero, no correctness checks.

## 1. Establish runtime and request provenance

Preserve the existing installation/configuration and model files for comparison. Record
app, Control and Qwen engine versions separately from startup logs and `/props` where
available; resolve the actual binary, config path, model/pack/tokenizer/template hashes,
MTP and PLE files, effective args/env, client version and model endpoint. Record the
hardware, driver, RAM/VRAM use, storage location, expert residency and background workloads.
Use the installed engine's help/source plus current official docs to verify supported settings:

- [Strata technical reference](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)
- [Release notes](https://github.com/Niko1221/Strata/releases)
- [Quant measurements](https://github.com/Niko1221/Strata/blob/main/docs/QUANTS.md)
- [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Strata benchmark protocol](https://github.com/Niko1221/Strata/blob/main/docs/BENCHMARK_PROTOCOL.md)

Capture request bodies and effective template behavior for the actual client. Verify
thinking is enabled, effort labels resolve correctly (`high` versus `xhigh`), sampling
values survive client overrides, and historical reasoning reaches the next turn.
Sending `preserve_thinking=true` alone is insufficient evidence. Record unsupported or
ignored fields as limitations. Inspect output caps, reasoning budgets and any `answer_room`
wrapping so a supposedly unrestricted reasoning test is not silently shortened.
Completion: a retained provenance record and resolved request/template evidence, with
unverified behaviors identified before interpreting quality results.

## 2. Freeze the quality baseline

| Dimension | Baseline |
|---|---|
| Weights and PLE | Existing original IQ3_S and standard PLE files |
| Context and serving | 131,072 configured tokens, one slot |
| KV | INT8; verify streaming policy and record actual resident cells |
| Expert cache and prefill | `auto`; record resolved sizes |
| Drafting | Existing MTP, `--spec 4 --spec-min-p 0.5` |
| Routing | Explicit `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_ROUTE_PRIOR=0` |
| Approximate prefill | Explicit `STRATA_KVA=0` if supported; record disabled status |
| Thinking | Enabled, highest supported effort, verified reasoning preservation |
| Sampler | temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1.0 |
| Output allowance | Fixed adequate cap per task, including reasoning and answer; record wrapping/truncation |
| Other policies | Release-default async adaptation; fixed vision setting across comparisons |

Verify the current model card before freezing sampler values. `--spec-min-p` is draft
confidence, separate from sampler `min_p`. Treat `--kv-resident 32768`, CPU vision,
worker counts and manual async settings as hardware candidates, not baseline requirements.
If an option is unsupported, resolve and record that limitation before proceeding.
Completion: save the complete effective baseline and its hash; retain original configs.

## 3. Use paired correctness tasks

Start with 18 fixed tasks, six each of coding, document reasoning/retrieval and tool use.
Reuse suitable existing fixtures and graders; add any missing cases only in a separately
implemented suite. Coding uses held-out tests and regression checks. Document tasks use
answer keys, supporting passages, distractors, unanswerable questions and actual prompts
near 32K and 100K tokens. Tool tasks use a mock environment with argument validators,
multi-turn tool-result interpretation and a checked final outcome.

Freeze fixtures, graders, three seed pairs, output/time caps and retry policy before the
comparison. Run A/B in alternating order, with identical reset/warm-up behavior and
controlled cache state for corresponding tasks. Score the first attempt separately from
completion after permitted retries. Count cutoffs, invalid calls, loops and crashes;
keep infrastructure failures visible and distinct from model failures. A changed cap,
fixture or budget starts a new cohort. Selected disagreement-only reruns are diagnostic,
not an unbiased overall ranking.
Completion: fixed task/seed manifest, machine-checkable outcomes and a retained raw capture.

## 4. Run comparisons in order

| Order | Comparison | Hold fixed |
|---|---|---|
| 1 | Preserved 0.1.45 versus verified current engine, if the old binary/config is available | New quality baseline, identical explicit requests and files; otherwise mark unavailable |
| 2 | Highest effort versus medium | Current engine, official sampler, routing shortcuts off |
| 3 | Tail skip 0 versus 7 | Baseline effort and all other settings; verify whether skipping actually fires |
| 4 | Route prior 0 versus 1.5 | Tail skip 0; verify support; retain as experimental |
| 5 | Hardware calibration and draft threshold 0.5 versus 0.7 | Winning quality settings; measure each hardware change independently |
| 6 | Optional diagnostics: MTP off, FP16 KV, Q8 PLE, alternative quant or 262K context | One change per comparison; separate actual context lengths and memory tiers |

Steps 3 onward are conditional on needing more speed or investigating an observed problem.
Repeat the baseline around long comparison batches to detect drift. Preserve automatic
calibration's measurements and run the correctness suite on its chosen configuration;
a speed-only calibration cannot establish unchanged task success. Keep both routing
shortcuts off until each has been tested independently. Test combinations only after
individual comparisons justify them. Wider context and higher quants are later experiments
because they can change RAM pressure, expert residency and SSD traffic simultaneously.

## 5. Record results and select

For every task/seed retain correctness, first-attempt success, final completion, retries,
loops, malformed calls, truncations, crashes, total elapsed time including retries, TTFT,
time to first answer/tool call, reasoning/output token counts, engine prompt/decode timing,
reused tokens, draft acceptance, memory use and file-tier/swap activity. Separate cold-prompt
from warm-conversation latency. A configured 262K window is not a 262K comprehension test.

Report paired outcomes by task category and seed, pass counts, unresolved regressions,
completion coverage, and time on tasks both arms finish correctly. Also report timeouts
and total time consumed across all attempts. A faster arm with worse correctness does not
replace the quality baseline automatically. Three seeds are a screen; expand disputed
comparisons before claiming equivalence. Keep token-fidelity, perplexity, needle recall,
subjective preference and task correctness as distinct evidence.

Completion: recommend a quality profile and, if supported, a faster routine profile with
observed accuracy/latency tradeoffs. Preserve raw artifacts and original settings. Record
version, source links and uncertainty beside any external comparison. Special determinism
switches belong in a separate diagnostic cohort because they can change performance.

## Execution boundary

Implementation scope and acceptance criteria: Strata adapter spec.

The adapter now uses `acceptance prepare --runtime strata`, the existing isolation,
capture, watchdog, sleep-inhibition, restore and acceptance mechanisms, and the separate
`--suite strata` roster. Commands and limitations are in RUNBOOK.md.
`llm_eval.paired` freezes and resumes three-seed alternating campaigns. This procedure
does not establish live acceptance: use STATE.md and the results ledger for measured
status. Keep production presets/services under runner ownership. Runtime upgrades remain
separate work. The llama.cpp screen retains its existing n=1 rule.
