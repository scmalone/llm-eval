"""Process metrics from an ``opencode export`` (the process-overhead workflow), kept apart from
pass/fail so a slow correct run is not mistaken for an efficient one."""
from collections import Counter


def _text(parts):
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()


def summarize(export):
    messages = export.get("messages", [])
    tools, commands, errors, steps, compactions = Counter(), [], 0, 0, 0
    output = reasoning = peak = peak_step_output = length_finishes = final_context = 0
    model_ms = tool_ms = 0
    answer, user_turns = "", 0
    first = last = None
    for message in messages:
        info, parts = message.get("info", {}), message.get("parts", [])
        times = info.get("time") or {}
        if info.get("role") == "user":
            user_turns += 1
            first = first if first is not None else times.get("created")
            continue
        if times.get("created") is not None and times.get("completed") is not None:
            model_ms += times["completed"] - times["created"]
            last = max(last or 0, times["completed"])
        if info.get("summary"):
            compactions += 1
        text = _text(parts)
        if text:
            answer = text
        for part in parts:
            kind = part.get("type")
            if kind == "compaction":
                compactions += 1
            elif kind == "step-finish":
                steps += 1
                tokens = part.get("tokens") or {}
                cache = tokens.get("cache") or {}
                output += tokens.get("output") or 0
                peak_step_output = max(peak_step_output, tokens.get("output") or 0)
                length_finishes += part.get("reason") == "length"
                reasoning += tokens.get("reasoning") or 0
                step_context = (tokens.get("input") or 0) + (cache.get("read") or 0) + (cache.get("write") or 0)
                peak = max(peak, step_context)
                final_context = step_context  # the last step is the question's own: after any compaction
            elif kind == "tool":
                tools[part.get("tool")] += 1
                state = part.get("state") or {}
                errors += state.get("status") == "error"
                span = state.get("time") or {}
                if span.get("start") is not None and span.get("end") is not None:
                    tool_ms += span["end"] - span["start"]
                if part.get("tool") == "bash" and isinstance((state.get("input") or {}).get("command"), str):
                    commands.append(state["input"]["command"])
    return {"answer": answer, "commands": commands, "user_turns": user_turns, "steps": steps,
            "tool_calls": sum(tools.values()), "tools": dict(tools), "invalid_calls": tools.get("invalid", 0),
            "tool_errors": errors, "compactions": compactions, "peak_context_tokens": peak,
            "final_context_tokens": final_context,
            "output_tokens": output, "reasoning_tokens": reasoning,
            "peak_step_output_tokens": peak_step_output, "length_finishes": length_finishes,
            "model_seconds": round(max(0, model_ms - tool_ms) / 1000, 1), "tool_seconds": round(tool_ms / 1000, 1),
            "wall_seconds": round((last - first) / 1000, 1) if first is not None and last is not None else None,
            "ended_with_question": answer.rstrip().endswith("?")}
