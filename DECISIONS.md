# Benchmark decisions

Selected decisions affecting the published results. The full operational history remains in the private development repository.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-30 | Count output and wall-time cutoffs as failures, labelled incomplete. | Excluding them would reward running out of budget. |
| 2026-10-30 | Compare inference engines through paired campaigns. | Repeated runs showed variability larger than many setting effects. |
| 2026-10-31 | Match the production client's thinking parameters and output limits. | Synthetic request settings changed model behavior and could cause template errors. |
| 2026-11-01 | Exclude tasks that never reach their required depth from depth quality measurements. | A shallow answer does not measure performance at the intended depth. |
| 2026-11-01 | Pin task environments by content digest. | Read-only mounts alone do not prevent host dependencies from changing. |
| 2026-11-01 | Use q8 KV by default, q4 only as a last resort; retain at least 1,024 MiB free GPU memory. | The benchmark targets a usable desktop configuration while limiting cache precision loss. |
| 2026-11-04 | Disable samplers the model publisher does not recommend. | Inherited engine defaults would affect some models differently. |
| 2026-11-06 | Reply to question-only tasks with an explicit instruction to answer and change nothing. | The previous follow-up encouraged models to act on questions. Changed tasks cannot be compared with earlier identities. |
| 2026-11-07 | Freeze task, grader and instruction identities for bench-v2. | Changes to the harness and grading made older scores unsuitable as a comparison baseline. |
| 2026-11-08 | Publish raw and reviewed adjusted counts together. | Raw scores retain rule violations; adjusted figures expose confirmed style and grader effects. Qwen's disputed haystack removals remain an upper bound. |

Exact adjustment rules and remaining measurement limitations are in [the results artifact](results/bench-v2.json).
