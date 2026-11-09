# Process-overhead task

Status: used. `llm_eval/workload/metrics.py` records steps, tool calls by tool, invalid calls, tool errors, compactions, peak context, model and tool time, clarifications, tests pasted back, and harness stalls, apart from pass or fail.

Record time-to-useful-result, tool calls, retries, human interventions,
context consumed, and ceremony required to reach the acceptance gate. Report
these separately from answer quality so a slower but correct process is not
mistaken for an efficient one.
