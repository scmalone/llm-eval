# Twilight's Requiem workflow task

Status: deferred. Visual slices need a render adapter; a later commit-replay task from twilights-requiem can use its non-visual checks.
adapter before activation.

Task: implement the smallest playable/visual slice requested by the fixture,
render it, inspect the result, and make one bounded iteration.

Acceptance: artifact renders; requested visual facts are present; scope stays
within the slice; the model records what changed and why; no unrelated files
are modified. Measure wall time, tool calls, retries, and iteration count.
