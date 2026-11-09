"""Deterministic graders. Expected facts stay outside the model request."""
import json


def recall(answer, expected):
    actual = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    return {"pass": actual == expected, "expected_count": len(expected),
            "correct_positions": sum(a == b for a, b in zip(actual, expected)),
            "actual_count": len(actual)}


def tool_call(message, name, arguments):
    calls = message.get("tool_calls", [])
    if len(calls) != 1:
        return {"pass": False, "reason": "expected exactly one call"}
    function = calls[0].get("function", {})
    try:
        raw = function.get("arguments")
        actual = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return {"pass": False, "reason": "invalid JSON"}
    return {"pass": function.get("name") == name and actual == arguments}


def evidence(answer, required_fact_ids, known_sources):
    """Validate submitted fact/source records; no reward for an inspected-file list alone."""
    try:
        entries = json.loads(answer)
        if not isinstance(entries, list):
            raise ValueError()
        supported = {e["fact_id"] for e in entries
                     if e["source"] in known_sources.get(e["fact_id"], [])}
        unsupported = [e for e in entries if e["source"] not in known_sources.get(e["fact_id"], [])]
    except (ValueError, KeyError, TypeError):
        return {"pass": False, "reason": "invalid evidence envelope"}
    missing = sorted(set(required_fact_ids) - supported)
    return {"pass": not missing and not unsupported, "missing": missing,
            "unsupported_count": len(unsupported)}
