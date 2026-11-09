"""Per-request timings from a llama-server log (the build on this machine, verbosity 3).

Every request leaves ``launch_slot_`` ... ``prompt eval time`` / ``eval time`` / ``total time`` ...
``release`` lines keyed by task id, and a request that reused a cached prefix is preceded by
``selected slot by LCP similarity, f_sim_best = X``. Lines are prefixed ``M.SS.mmm.uuu`` (time
since the server started). The progress lines are skipped. A request the server never finished
(killed or timed out) has no ``eval time`` line and is returned with those fields None.
"""
import re

STAMP = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) ")
NUMBER = r"([\d.]+|inf|nan)"
LAUNCH = re.compile(r"slot launch_slot_: id +\d+ \| task (\d+) \| processing task")
SIMILARITY = re.compile(r"selected slot by LCP similarity, f_sim_best = ([\d.]+)")
PROMPT = re.compile(rf"task (\d+) \| prompt eval time = +([\d.]+) ms / +(\d+) tokens \(.*?, +{NUMBER} tokens per second\)")
EVAL = re.compile(rf"task (\d+) \| +eval time = +([\d.]+) ms / +(\d+) tokens \(.*?, +{NUMBER} tokens per second\)")
TOTAL = re.compile(r"task (\d+) \| +total time = +([\d.]+) ms / +(\d+) tokens")
GRAPHS = re.compile(r"task (\d+) \| +graphs reused = +(\d+)")
RELEASE = re.compile(r"slot +release: id +\d+ \| task (\d+) \| stop processing: n_tokens = (\d+), truncated = (\d+)")


def _seconds(line):
    match = STAMP.match(line)
    return int(match[1]) * 60 + int(match[2]) + int(match[3]) / 1000 + int(match[4]) / 1e6 if match else None


def _rate(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value == value and value != float("inf") else None


def parse_server_log(text):
    """Requests in launch order: dicts with the fields of the ``server_requests`` table (less run
    and file). ``t_log_s`` is the release time in seconds since server start."""
    requests, open_ = [], {}
    similarity = None
    for line in text.splitlines():
        if (match := SIMILARITY.search(line)):
            similarity = float(match[1])
        elif (match := LAUNCH.search(line)):
            task = int(match[1])
            open_[task] = {"task": task, "slot_similarity": similarity, "prompt_n": None, "prompt_ms": None,
                           "prompt_tps": None, "predicted_n": None, "predicted_ms": None, "predicted_tps": None,
                           "total_ms": None, "total_tokens": None, "graphs_reused": None, "ctx_tokens_after": None,
                           "truncated": None, "t_log_s": None, "cache_n_est": None}
            requests.append(open_[task])
            similarity = None
        elif (match := PROMPT.search(line)) and int(match[1]) in open_:
            r = open_[int(match[1])]
            r.update(prompt_ms=float(match[2]), prompt_n=int(match[3]), prompt_tps=_rate(match[4]))
        elif (match := EVAL.search(line)) and int(match[1]) in open_:
            r = open_[int(match[1])]
            r.update(predicted_ms=float(match[2]), predicted_n=int(match[3]), predicted_tps=_rate(match[4]))
        elif (match := TOTAL.search(line)) and int(match[1]) in open_:
            open_[int(match[1])].update(total_ms=float(match[2]), total_tokens=int(match[3]))
        elif (match := GRAPHS.search(line)) and int(match[1]) in open_:
            open_[int(match[1])]["graphs_reused"] = int(match[2])
        elif (match := RELEASE.search(line)) and int(match[1]) in open_:
            r = open_.pop(int(match[1]))
            r.update(ctx_tokens_after=int(match[2]), truncated=int(match[3]), t_log_s=_seconds(line))
            if r["prompt_n"] is not None and r["predicted_n"] is not None:
                # n_tokens at release excludes the last sampled token, so the total is one more.
                r["cache_n_est"] = max(0, r["ctx_tokens_after"] + 1 - r["prompt_n"] - r["predicted_n"])
    return requests


def last_stamp(text):
    """Seconds since start of the last stamped line, to place the log on the wall clock."""
    for line in reversed(text.splitlines()):
        seconds = _seconds(line)
        if seconds is not None:
            return seconds
    return None
