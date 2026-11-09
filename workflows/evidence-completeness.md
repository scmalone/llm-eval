# Evidence-completeness / anti-skimming task

Status: used. The review tasks' graders require each planted defect by file and function (`findings` check), and `llm_eval.graders.evidence` remains the fact-plus-source checker.

Require the model to inspect a declared file set and return fact IDs paired with
valid file/line sources. Unsupported claims, missing required facts, and
claims based only on an inspected-file list are failures.
