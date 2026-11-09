"""Pure context admission and comparison policy; memory units are MiB."""
LADDER = (65536, 86016, 98304, 102400, 114688, 131072)
MIN_FREE = 1024
RESERVE = 2048


def classification(free_mib):
    if free_mib < MIN_FREE:
        return "unsafe"
    if free_mib < 1536:
        return "marginal"
    if free_mib < 2048:
        return "usable"
    return "comfortable"


def forecast(free_before, fixed_mib, kv_at_anchor_mib, anchor, target):
    if min(free_before, fixed_mib, kv_at_anchor_mib, anchor, target) < 0 or not anchor:
        raise ValueError("invalid memory estimate")
    return free_before - fixed_mib - kv_at_anchor_mib * target / anchor - 256


def fits(prompt_tokens, output_tokens, context, reserve=RESERVE):
    return min(prompt_tokens, output_tokens, reserve) >= 0 and prompt_tokens + output_tokens + reserve <= context


def context_label(context, measured_prompt_tokens=0):
    if context <= 65536:
        return "specialist_only"
    # Only a measured 100K-token prompt counts: recall prompts leave room for a full agent-turn
    # output budget, so a rung's number overstates the conversation it can actually hold.
    if measured_prompt_tokens >= 100000:
        return "100k_prompt_measured"
    return "above_64k_fallback"
