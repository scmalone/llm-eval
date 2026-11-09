#!/usr/bin/env python3
"""Find the fastest llama.cpp preset settings that fit: start high, step down on a failure, up while memory is left.

    python3 tools/speed_sweep.py --dry-run          # the knobs each preset would try; touches nothing
    python3 tools/speed_sweep.py --baseline-only    # the load test: every preset as written, nothing tuned
    python3 tools/speed_sweep.py --baseline-only --repeat 3 --presets a   # the noise check: one preset, 3 times
    python3 tools/speed_sweep.py [--presets a,b] [--speed-first a,b] [--out DIR]

One measurement loads a setting in a scratch llama-server with the router stopped, then:
- two short chat requests (the owner's template and thinking on): decode speed and draft acceptance near depth 0;
- a depth ladder through /completion on a frozen source-code corpus, each prompt a token-exact prefix extension of the
  last so the prompt cache carries it (25%, 50%, 75%, then FILL of the context): prefill speed per segment and decode
  speed at each depth. A step the cache did not carry (hybrid models can reprocess) skips the middle depths.
Knobs that move decode speed but not memory (drafter, draft depth) are compared on quick measurements (load, chat and
the 25% step), and their winner gets the full ladder once. Token prompts start with BOS (tokenized with add_special).
GPU telemetry is sampled every half second and the host every five, tagged by phase; every request and response, the
server's /props, /slots and /metrics after load and at the end, its log, argv and buffer sizes are kept. A setting
fits when the lowest free memory seen stays at or above the 1,024 MiB gate, measured beside the working desktop
(owner, 2026-11-01: Brave open, Steam and Slack closed).

The search starts from the preset (the 2026-10-31 prime configs, already sized near the top of the memory budget):
- spare memory is spent in priority order: context (up to the model's trained maximum), then speed (drafter on or off,
  draft depth, ubatch, fewer CPU MoE layers); ``--speed-first`` puts speed first. KV never climbs (q8_0 is the
  default). Every knob moves one value at a time and stops at the first failure or, for a speed knob, the first step
  that is not faster; draft depth starts at 3 or the preset's;
- a speed change is kept only when it is more than 3% faster, so a tie keeps the setting with more headroom;
- a preset that does not fit as written is stepped down one setting at a time, each step on top of the last, until
  it fits (owner, 2026-11-01): context down the ladder to 131,072 at the lowest, then more CPU MoE layers (MoE models),
  then a smaller ubatch (not on Gemma, where ubatch caps image tokens), then KV f16 -> q8_0, and q8_0 -> q4_0 only as
  the last resort. Settings the step-down moved, and context and KV, are not climbed again afterwards.
Context and KV changes are proposals for the owner (they move pi's compaction and the cache); the speed settings are
the result. Every measurement is appended to results.jsonl, so a rerun with the same --out resumes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
# Out of the default sweep (owner, 2026-10-31): gemma-4-31b-it only loads 65,536, a context the owner does not run.
SKIP = ("gemma-4-31b-it",)
sys.path.insert(0, str(REPO))

from llm_eval import inhibit, ramcheck  # noqa: E402
from llm_eval.telemetry import GpuSampler  # noqa: E402
from llm_eval.acceptance import DEFAULT_RESTORE  # noqa: E402
from llm_eval.adapters import CapacityError, InfrastructureError  # noqa: E402
from llm_eval.live import LiveAdapter, server_args  # noqa: E402
from llm_eval.preflight import DEFAULT_PRESET, read_presets  # noqa: E402
from llm_eval.runner import _uninterruptible  # noqa: E402

from llm_eval.paths import DATA as RUNS  # noqa: E402  (the data folder, outside git)
ROUTER = "llama-router.service"
GATE_MIB = 1024     # the free-memory gate every run keeps
MARGIN_MIB = 0      # none since 2026-11-01: the sweep runs beside the working desktop (Brave open), so the gate is the mark
FLOOR_MIB = GATE_MIB + MARGIN_MIB
BASELINE_WARN_MIB = 22300  # free with the router stopped; less means the desktop holds over ~2.2 GB (Steam open?)
GAIN = 0.03         # a speed step must beat this; inside it the setting with more headroom stays
FILL = 0.90         # the deepest request fills this much of the context
DEPTHS = (0.25, 0.50, 0.75)  # the ladder's middle steps, before FILL
SHORT_TOKENS, SHORT_REPEATS, DEPTH_TOKENS = 512, 2, 128
PREFILL_REF, DECODE_REF = 8000, 1500  # a reference agent turn: new prompt tokens and output tokens
ACCEPT_LOW = 0.5    # below this draft acceptance the drafter-off arm is measured
CONTEXTS = (65536, 98304, 131072, 163840, 196608, 262144)
KV_BITS = {"q4_0": 4.5, "q8_0": 8.5, "f16": 16.0}
UBATCH = (256, 512, 1024, 2048, 4096)
DRAFT_DEPTHS = {"draft-mtp": (1, 2, 3, 4, 5, 6), "draft-dflash": (4, 6, 8, 10, 12, 15)}
DRAFT_START = {"draft-mtp": 3}  # MTP starts at 3 unless the preset asks for more
DRAFT_KEYS = ("spec-type", "spec-draft-n-max", "draft-p-min", "model-draft")
PROPOSAL_KEYS = ("ctx-size", "cache-type-k", "cache-type-v")
RESCUE_MOE_STEPS = 4
CONTEXT_FLOOR = 131072      # owner, 2026-11-01: a step-down may shorten context to this and no further
NO_UBATCH_RESCUE = ("gemma-",)  # Gemma needs a whole image in one ubatch: a smaller one caps image detail
# Timeouts, sized from what a working server does: a hang costs at most one of these, never the night.
LOAD_SECONDS = 300          # a 20 GB load from page cache takes 15-60 s
CHAT_SECONDS = 240          # 512 tokens at a floor of 5 t/s, plus slack
PREFILL_FLOOR_TPS = 200     # slower than this (prefill slows with depth) is a failed setting, not a slow one
DECODE_FLOOR_TPS = 4
GPU_SAMPLE_SECONDS, HOST_SAMPLE_SECONDS = 0.5, 5.0
CORPUS_ROOT = Path.home() / "Dev" / "llama.cpp"
CORPUS_DIRS = ("src", "common", "tools/server")
CORPUS_CHARS = 1_400_000    # ~400K tokens of C++ for any vocabulary seen so far: enough for 90% of 262,144
SHORT_PROMPT = ("Write a Python module that implements an LRU cache with a size limit and per-entry expiry. "
                "Include type hints, docstrings and unittest tests.")


# A two-exchange conversation for /apply-template: the earlier assistant turn carries reasoning, so the render shows
# whether the template keeps it (cache reuse) and what reasoning level it states.
TEMPLATE_PROBE = [
    {"role": "system", "content": "Probe system prompt."},
    {"role": "user", "content": "First question."},
    {"role": "assistant", "content": "First answer.", "reasoning_content": "EARLIER-REASONING-MARK"},
    {"role": "user", "content": "Second question."},
]


def log(message, out=None):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if out is not None:
        with (out / "sweep.log").open("a") as handle:
            handle.write(line + "\n")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


# -- scoring and estimates -----------------------------------------------------------------------
def score(prefill_tps, decode_tps):
    """Seconds for the reference turn: lower is faster."""
    return PREFILL_REF / prefill_tps + DECODE_REF / decode_tps


def gain(new, old):
    """How much faster ``new`` is than ``old`` (0.05 = 5% faster). A quick measurement has only the first depth step,
    so a comparison with one uses ``quick_score``, which every measurement carries."""
    field = "quick_score" if new.get("quick") or old.get("quick") else "score"
    return old[field] / new[field] - 1


def config_key(name, settings, quick=False):
    return hashlib.sha256(json.dumps([name, sorted(settings.items())] + (["quick"] if quick else [])
                                     ).encode()).hexdigest()[:16]


def gguf_facts(model_path):
    """{"n_expert", "n_ctx_train"} from the model file's own header, or {} when it cannot be read. Build 12017's
    server log at the default verbosity no longer prints the loader's facts, so the sweep reads the file (llama.cpp's
    own gguf-py from the corpus checkout)."""
    try:
        sys.path.insert(0, str(CORPUS_ROOT / "gguf-py"))
        from gguf import GGUFReader
        fields = GGUFReader(str(Path(model_path).expanduser())).fields
    except Exception:  # noqa: BLE001 -- a missing file or library leaves the facts unknown, as before
        return {}
    finally:
        if sys.path[0] == str(CORPUS_ROOT / "gguf-py"):
            sys.path.pop(0)
    def value(suffix):
        key = next((k for k in fields if k.endswith(suffix)), None)
        return int(fields[key].parts[-1][0]) if key else None
    return {k: v for k, v in (("n_expert", value(".expert_count")), ("n_ctx_train", value(".context_length")))
            if v is not None}


def model_info(settings, result, facts=gguf_facts):
    """The server's reported model facts, with the file header filling what the log left out."""
    info = {k: v for k, v in ((result or {}).get("info") or {}).items() if v is not None}
    return {**(facts(settings.get("model", "")) if settings.get("model") else {}), **info}


# -- knobs ---------------------------------------------------------------------------------------
def with_value(key):
    return lambda settings, value: {**settings, key: str(value)}


def set_ubatch(settings, value):
    out = {**settings, "ubatch-size": str(value)}
    if int(out.get("batch-size", 2048)) < value:
        out["batch-size"] = str(value)
    return out


def set_kv(settings, value):
    return {**settings, "cache-type-k": value, "cache-type-v": value}


def drop_drafter(settings):
    return {k: v for k, v in settings.items() if k not in DRAFT_KEYS}


def knobs(settings, info, speed_first=False):
    """The knobs this preset can move, in the order spare memory is spent, and the ones it cannot with why.
    ``values`` run from the preset's value towards more memory or more speed, one step at a time; ``start`` is where
    the search begins; ``gain`` knobs must be faster to stay. KV only ever steps down (q8_0 is the default; f16 buys
    no accuracy worth its memory), so it is not a knob here."""
    found, skipped = [], []
    ctx = int(settings["ctx-size"])
    trained = info.get("n_ctx_train") or max(CONTEXTS)
    up = [c for c in CONTEXTS if ctx < c <= trained]
    if up:
        found.append({"name": "context", "key": "ctx-size", "values": [ctx, *up], "start": 0, "gain": False,
                      "apply": with_value("ctx-size")})
    else:
        skipped.append(("context", f"{ctx:,} is the trained maximum or the ladder top"))
    spec = settings.get("spec-type")
    speed = []
    if spec:
        speed.append({"name": "drafter", "drafter": True, "quick": True})
        depth = int(settings.get("spec-draft-n-max", 3))
        ladder = sorted(set(DRAFT_DEPTHS.get(spec, ())) | {depth})
        start = max(depth, DRAFT_START.get(spec, depth))
        speed.append({"name": "draft depth", "key": "spec-draft-n-max", "values": ladder, "from": depth,
                      "start": ladder.index(start), "gain": True, "sweet": True, "quick": True,
                      "apply": with_value("spec-draft-n-max")})
    else:
        skipped.append(("drafter", "no spec-type"))
    ub = int(settings.get("ubatch-size", 512))
    ladder = sorted({u for u in UBATCH if u >= ub} | {ub})
    if len(ladder) > 1:
        speed.append({"name": "ubatch", "key": "ubatch-size", "values": ladder, "start": 0, "gain": True,
                      "apply": set_ubatch})
    moe = int(settings.get("n-cpu-moe", 0))
    if moe > 0:
        speed.append({"name": "cpu moe", "key": "n-cpu-moe", "values": list(range(moe, -1, -1)), "start": 0,
                      "gain": True, "apply": with_value("n-cpu-moe")})
    else:
        skipped.append(("cpu moe", "every expert layer is already on the GPU" if info.get("n_expert")
                        else "no CPU MoE layers"))
    return (speed + found if speed_first else found + speed), skipped


def rescues(settings, info, name=""):
    """Step-downs for a preset that does not fit as written, each on top of the last: context to CONTEXT_FLOOR, more
    CPU MoE layers, a smaller ubatch, then KV down one precision at a time to q4_0 (the last resort)."""
    steps, cfg = [], dict(settings)
    for smaller in sorted((c for c in CONTEXTS if CONTEXT_FLOOR <= c < int(cfg["ctx-size"])), reverse=True):
        cfg = {**cfg, "ctx-size": str(smaller)}
        steps.append((f"ctx-size {smaller}", cfg))
    if info.get("n_expert") or "n-cpu-moe" in settings:
        for _ in range(RESCUE_MOE_STEPS):
            cfg = {**cfg, "n-cpu-moe": str(int(cfg.get("n-cpu-moe", 0)) + 1)}
            steps.append((f"n-cpu-moe {cfg['n-cpu-moe']}", cfg))
    ub = int(cfg.get("ubatch-size", 512))
    if not name.startswith(NO_UBATCH_RESCUE):
        for smaller in sorted((u for u in UBATCH if u < ub), reverse=True):
            cfg = set_ubatch(cfg, smaller)
            steps.append((f"ubatch-size {smaller}", cfg))
    k, v = cfg.get("cache-type-k", "f16"), cfg.get("cache-type-v", "f16")
    order = list(KV_BITS)
    if k == v and k in order:
        for lower in reversed(order[:order.index(k)]):
            cfg = set_kv(cfg, lower)
            steps.append((f"cache-type {lower}", cfg))
    return steps


# -- the search ----------------------------------------------------------------------------------
class Sweep:
    """The search over one preset. ``measure(name, settings)`` returns a result dict with ``fits``, ``why``,
    ``score``, ``headroom_mib``, ``accept``, ``buffers`` and ``info``; it is the only thing that touches a GPU."""

    def __init__(self, measure, say=print, facts=gguf_facts):
        self.measure, self.say, self.facts = measure, say, facts

    def note(self, knob, value, outcome, why, result=None):
        row = {"knob": knob, "value": value, "outcome": outcome, "why": why}
        if result is not None:
            row.update(key=result.get("key"), score=result.get("score"), headroom_mib=result.get("headroom_mib"))
        self.trail.append(row)
        self.say(f"  {knob} {value}: {outcome} ({why})")

    def preset(self, name, settings, speed_first=False, baseline_only=False, repeats=1):
        self.name, self.trail = name, []
        first = self.measure(name, settings)
        outcome = {"preset": name, "before": dict(settings), "baseline": first, "trail": self.trail}
        if first.get("infra"):
            self.note("baseline", "as written", "error", first["why"], first)
            return {**outcome, "status": "error", "after": dict(settings), "final": first}
        self.note("baseline", "as written", "fits" if first["fits"] else "failed", first["why"], first)
        if baseline_only:
            outcome["repeats"] = [first]
            for n in range(1, repeats):  # the noise check: the same setting again, each load from scratch
                again = self.measure(name, settings, repeat=n)
                self.note("repeat", n + 1, "fits" if again["fits"] else "failed", again["why"], again)
                outcome["repeats"].append(again)
            return {**outcome, "status": "fits" if first["fits"] else "does not fit", "after": dict(settings),
                    "final": first}
        cfg, best, held = settings, first, set()
        info = model_info(settings, first, self.facts)
        if not first["fits"]:
            cfg, best = self.rescue(settings, info)
            if best is None:
                return {**outcome, "status": "does not fit", "after": dict(settings), "final": first}
            # what the step-down moved, and context and KV, would only be climbed back into the memory it freed
            held = {k for k in cfg if cfg.get(k) != settings.get(k)} | set(PROPOSAL_KEYS)
        found, skipped = knobs(cfg, info, speed_first)
        for knob, why in skipped:
            self.note(knob, "-", "skipped", why)
        for knob in found:
            if knob.get("key") in held:
                self.note(knob["name"], "-", "skipped", "held after the step-down")
                continue
            if knob["name"] == "draft depth" and "spec-type" not in cfg:
                self.note(knob["name"], "-", "skipped", "the drafter is off")
                continue
            before = (cfg, best)
            cfg, best = self.drafter(cfg, best) if knob.get("drafter") else self.climb(knob, cfg, best)
            if best.get("quick"):
                cfg, best = self.verify(cfg, best, *before)
        return {**outcome, "status": "tuned", "after": cfg, "final": best}

    def verify(self, cfg, best, fallback_cfg, fallback_best):
        """A decode knob is chosen on quick measurements (no deep prefill); its winner gets the full ladder once."""
        full = self.measure(self.name, cfg)
        if full["fits"]:
            self.note("verify", "full ladder", "fits", full["why"], full)
            return cfg, full
        self.note("verify", "full ladder", "failed", f"{full['why']}; back to the last verified setting", full)
        return fallback_cfg, fallback_best

    def rescue(self, settings, info):
        for label, cfg in rescues(settings, info, self.name):
            result = self.measure(self.name, cfg)
            if result["fits"]:
                self.note("rescue", label, "fits", result["why"], result)
                return cfg, result
            self.note("rescue", label, "failed", result["why"], result)
        self.note("rescue", "-", "gave up",
                  f"nothing fits down to {CONTEXT_FLOOR:,} context and q4_0 KV: the owner's call")
        return settings, None

    def climb(self, knob, cfg, best):
        values, ceiling, moved = knob["values"], len(knob["values"]) - 1, False
        quick = bool(knob.get("quick"))
        i = values.index(knob["from"]) if "from" in knob else knob["start"]
        if knob["start"] != i:
            # draft depth begins at its plausible start, above the preset's value: a tie keeps the preset's
            trial = knob["apply"](cfg, values[knob["start"]])
            result = self.measure(self.name, trial, quick=quick)
            change = gain(result, best) if result["fits"] else None
            if change is not None and change > GAIN:
                self.note(knob["name"], values[knob["start"]], "kept", f"start {change:+.1%}", result)
                cfg, best, i, moved = trial, result, knob["start"], True
            else:
                self.note(knob["name"], values[knob["start"]], "failed" if change is None else "no gain",
                          result["why"] if change is None else f"start {change:+.1%}", result)
                ceiling = i  # up from the preset's value is already answered
        while i < ceiling:
            j = i + 1
            trial = knob["apply"](cfg, values[j])
            result = self.measure(self.name, trial, quick=quick)
            if not result["fits"]:
                self.note(knob["name"], values[j], "failed", result["why"], result)
                break
            change = gain(result, best)
            if knob["gain"] and change <= GAIN:
                self.note(knob["name"], values[j], "no gain", f"{change:+.1%}", result)
                break
            self.note(knob["name"], values[j], "kept", f"{change:+.1%}, headroom {result['headroom_mib']:,} MiB",
                      result)
            cfg, best, i, moved = trial, result, j, True
        if knob.get("sweet") and not moved:
            for j in range(i - 1, -1, -1):
                trial = knob["apply"](cfg, values[j])
                result = self.measure(self.name, trial, quick=quick)
                if not result["fits"]:
                    self.note(knob["name"], values[j], "failed", result["why"], result)
                    break
                change = gain(result, best)
                if change <= GAIN:
                    self.note(knob["name"], values[j], "no gain", f"{change:+.1%}", result)
                    break
                self.note(knob["name"], values[j], "kept", f"{change:+.1%}", result)
                cfg, best = trial, result
        return cfg, best

    def drafter(self, cfg, best):
        accept = best.get("accept")
        if accept is not None and accept >= ACCEPT_LOW:
            self.note("drafter", "off", "skipped", f"acceptance {accept:.0%} is at least {ACCEPT_LOW:.0%}")
            return cfg, best
        trial = drop_drafter(cfg)
        result = self.measure(self.name, trial, quick=True)
        if not result["fits"]:
            self.note("drafter", "off", "failed", result["why"], result)
            return cfg, best
        change = gain(result, best)
        seen = "unreported" if accept is None else f"{accept:.0%}"
        if change >= -GAIN:  # off and within 3% keeps the memory the drafter held
            self.note("drafter", "off", "kept", f"{change:+.1%} with acceptance {seen}", result)
            return trial, result
        self.note("drafter", "off", "no gain", f"{change:+.1%} with acceptance {seen}", result)
        return cfg, best


# -- measuring on the machine --------------------------------------------------------------------
class Recorder(threading.Thread):
    """Samples ``read()`` every ``interval`` into a JSONL file tagged with the current phase; tracks the lowest free
    GPU memory seen, overall and per phase. Reads first, so even a short measurement has a reading."""

    def __init__(self, read, path, interval, free_field=None):
        super().__init__(daemon=True)
        self.read, self.path, self.interval, self.free_field = read, path, interval, free_field
        self.phase, self.low, self.lows, self.done, self.t0 = "start", None, {}, threading.Event(), time.monotonic()

    def run(self):
        while True:
            try:
                row = self.read()
                if row is not None:
                    row = {"t": round(time.monotonic() - self.t0, 2), "wall": time.time(), "phase": self.phase, **row}
                    append_jsonl(self.path, row)
                    free = row.get(self.free_field) if self.free_field else None
                    if isinstance(free, (int, float)):
                        self.low = free if self.low is None else min(self.low, free)
                        self.lows[self.phase] = min(self.lows.get(self.phase, free), free)
            except Exception:
                pass
            if self.done.wait(self.interval):
                return

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.done.set()
        self.join(10)


def parse_buffers(text):
    """Buffer sizes and model facts from a llama-server log (first match for model facts: the target loads first)."""
    def total(pattern):
        found = [float(x) for x in re.findall(pattern, text)]
        return round(sum(found), 1) if found else None
    def each(pattern):
        found = {}
        for name, size in re.findall(pattern, text):
            found[name] = round(found.get(name, 0) + float(size), 1)
        return found
    def first(pattern, cast=int):
        match = re.search(pattern, text)
        return cast(match.group(1).strip()) if match else None
    buffers = {"kv_mib": total(r"KV buffer size\s*=\s*([0-9.]+)\s*MiB"),
               "rs_mib": total(r"RS buffer size\s*=\s*([0-9.]+)\s*MiB"),
               "compute_mib": total(r"CUDA\d+ compute buffer size\s*=\s*([0-9.]+)\s*MiB"),
               "cpu_model_mib": total(r"CPU_Mapped model buffer size\s*=\s*([0-9.]+)\s*MiB"),
               "gpu_model_mib": total(r"CUDA\d+ model buffer size\s*=\s*([0-9.]+)\s*MiB"),
               "model_by_device": each(r"(\S+) model buffer size\s*=\s*([0-9.]+)\s*MiB"),
               "kv_by_device": each(r"(\S+) KV buffer size\s*=\s*([0-9.]+)\s*MiB"),
               "compute_by_device": each(r"(\S+) compute buffer size\s*=\s*([0-9.]+)\s*MiB")}
    info = {"n_ctx_train": first(r"n_ctx_train\s*=\s*(\d+)"), "n_expert": first(r"n_expert\s*=\s*(\d+)"),
            "n_expert_used": first(r"n_expert_used\s*=\s*(\d+)"), "n_layer": first(r"n_layer\s*=\s*(\d+)"),
            "n_embd": first(r"n_embd\s*=\s*(\d+)"), "arch": first(r"arch\s*=\s*(\S+)", str),
            "file_type": first(r"file type\s*=\s*(.+)", str), "model_size": first(r"file size\s*=\s*(.+)", str),
            "model_params": first(r"model params\s*=\s*(.+)", str), "n_vocab": first(r"n_vocab\s*=\s*(\d+)"),
            "build": first(r"build:\s*(.+)", str), "graph_splits": first(r"graph splits\s*=\s*(\d+)"),
            "oom": bool(re.search(r"out of memory|cudaMalloc failed|failed to allocate", text, re.I))}
    return buffers, info


def frozen_corpus(out):
    """Source code for the depth ladder, frozen into the run folder on first use so a resumed run reads the same."""
    path = out / "corpus.txt"
    if not path.exists():
        parts, size = [], 0
        for folder in CORPUS_DIRS:
            for file in sorted((CORPUS_ROOT / folder).rglob("*")):
                if file.is_file() and file.suffix in (".cpp", ".h", ".c", ".hpp"):
                    text = f"// ===== {file.relative_to(CORPUS_ROOT)} =====\n{file.read_text(errors='replace')}\n"
                    parts.append(text)
                    size += len(text)
                    if size >= CORPUS_CHARS:
                        break
            if size >= CORPUS_CHARS:
                break
        path.write_text("".join(parts))
    return path.read_text()


class LiveMeasure:
    def __init__(self, adapter, out, fill=FILL):
        self.adapter, self.out, self.fill, self.tokens = adapter, out, fill, {}

    def corpus_tokens(self, settings):
        """The corpus in this model's tokens, from the server's own tokenizer; one list per model file."""
        model = settings["model"]
        if model not in self.tokens:
            status, body = self.adapter.host.http("POST", self.adapter.endpoint(), "/tokenize",
                                                  {"content": frozen_corpus(self.out), "add_special": True},
                                                  timeout=120)  # with BOS: a token-array prompt is used as given
            if status != 200 or not isinstance(body, dict) or "tokens" not in body:
                raise InfrastructureError(f"tokenize failed with HTTP {status}")
            self.tokens[model] = body["tokens"]
        return self.tokens[model]

    def completion(self, tokens, n, raw, label, phase, cached=0):
        """One /completion of the first ``n`` corpus tokens; returns its timings and keeps the whole exchange.
        ``cached``: the prefix the prompt cache should carry, so the timeout covers only the new tokens."""
        body = {"prompt": tokens[:n], "n_predict": DEPTH_TOKENS, "cache_prompt": True, "seed": 0, "stream": False}
        timeout = 60 + (n - cached) / PREFILL_FLOOR_TPS + DEPTH_TOKENS / DECODE_FLOOR_TPS
        phase(label)
        started = time.monotonic()
        try:
            status, reply = self.adapter.host.http("POST", self.adapter.endpoint(), "/completion", body,
                                                   timeout=timeout)
        except (ConnectionError, TimeoutError) as exc:
            self.adapter._request_failed(f"{label}: {exc}")
        append_jsonl(raw / "requests.jsonl", {"label": label, "endpoint": "/completion", "prompt_tokens": n,
                                              "timeout": round(timeout), "seconds": round(time.monotonic() - started, 3),
                                              "status": status, "response": reply})
        if status >= 500:
            self.adapter._request_failed(f"{label}: HTTP {status}")
        if status != 200 or not isinstance(reply, dict):
            raise ValueError(f"{label}: unexpected response (HTTP {status})")
        return reply.get("timings") or {}

    def rendered_template(self):
        """What the chat template makes of a fixed conversation, with the server's own default kwargs: shows the
        reasoning level the model is told (Muse Glimmer's ``Reasoning strength``) and whether earlier reasoning
        stays in the history (prompt-cache reuse in agent turns). Never fails a measurement."""
        try:
            status, body = self.adapter.host.http("POST", self.adapter.endpoint(), "/apply-template",
                                                  {"messages": TEMPLATE_PROBE}, timeout=60)
        except Exception as exc:  # noqa: BLE001 -- evidence only: nothing here may fail the measurement
            return {"error": f"{type(exc).__name__}: {exc}"[:200]}
        prompt = body.get("prompt") if isinstance(body, dict) else None
        prompt = prompt if isinstance(prompt, str) else None
        return {"status": status, "prompt": prompt,
                "earlier_reasoning_kept": None if prompt is None else "EARLIER-REASONING-MARK" in prompt,
                "reasoning_line": None if prompt is None else next(
                    (line for line in prompt.splitlines() if "easoning" in line and "MARK" not in line), None)}

    def __call__(self, name, settings, quick=False, key=None):
        """``quick``: load, chat and the first depth step only, for knobs that move decode speed, not memory.
        ``key``: a repeat's own key, so its raw data does not land on the first measurement's."""
        ctx = int(settings["ctx-size"])
        key = key or config_key(name, settings, quick)
        raw = self.out / "raw" / key
        result = {"preset": name, "key": key, "settings": settings, "quick": quick, "fits": False, "infra": False,
                  "why": "",
                  "baseline_free_mib": self.adapter.baseline, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "argv": server_args(self.adapter.binary or "llama-server", settings, ctx, 0)[1:]}
        started = time.monotonic()
        shorts, ladder = [], []
        gpu = Recorder(self.adapter.sample_gpu, raw / "gpu.jsonl", GPU_SAMPLE_SECONDS, "mem_free_mib")
        host = Recorder(self.adapter.sample_host, raw / "host.jsonl", HOST_SAMPLE_SECONDS)
        def phase(label):
            gpu.phase = host.phase = label
        def guard():
            """The floor is the lowest free memory anywhere in the run, so once it is crossed the verdict is in:
            stop here instead of filling the rest of the context."""
            if gpu.low is not None and gpu.low < FLOOR_MIB:
                raise FloorBreach(gpu.phase)
        with gpu, host:
            try:
                phase("load")
                loaded = self.adapter.load({"id": name, "settings": settings, "request": {}}, ctx)
                result["load"] = loaded
                write_json(raw / "server.loaded.json", self.adapter.server_snapshot())
                write_json(raw / "template.json", self.rendered_template())
                guard()
                for seed in range(SHORT_REPEATS):
                    phase(f"chat-{seed}")
                    reply = self.adapter.request({"kind": "smoke", "prompt": SHORT_PROMPT,
                                                  "max_output_tokens": SHORT_TOKENS, "seed": seed})
                    append_jsonl(raw / "requests.jsonl", {"label": f"chat-{seed}", "endpoint": "/v1/chat/completions",
                                                          "response": reply})
                    shorts.append(reply.get("timings") or {})
                    guard()
                phase("tokenize")
                tokens = self.corpus_tokens(settings)
                deepest = min(int(ctx * self.fill), len(tokens))
                result.update(corpus_tokens=len(tokens), short_corpus=len(tokens) < int(ctx * self.fill))
                points = [int(ctx * d) for d in DEPTHS if int(ctx * d) < deepest] + [deepest]
                if quick:
                    points = points[:1]
                previous = 0
                for n in points:
                    timings = self.completion(tokens, n, raw, f"depth-{n}", phase, previous)
                    guard()
                    reused = (timings.get("cache_n") or 0) >= previous * 0.9 if previous else None
                    ladder.append({"prompt_tokens": n, "cache_n": timings.get("cache_n"),
                                   "prompt_n": timings.get("prompt_n"), "prompt_ms": timings.get("prompt_ms"),
                                   "prefill_tps": timings.get("prompt_per_second"),
                                   "decode_tps": timings.get("predicted_per_second"),
                                   "predicted_n": timings.get("predicted_n"), "draft_n": timings.get("draft_n"),
                                   "draft_n_accepted": timings.get("draft_n_accepted"), "cache_carried": reused})
                    previous = n
                    if reused is False and n != deepest:
                        # the cache did not carry this prefix: every middle depth would reprocess, so go deepest now
                        result["cache_carried"] = False
                        ladder.append(None)
                        timings = self.completion(tokens, deepest, raw, f"depth-{deepest}", phase)
                        ladder[-1] = {"prompt_tokens": deepest, "cache_n": timings.get("cache_n"),
                                      "prompt_n": timings.get("prompt_n"), "prompt_ms": timings.get("prompt_ms"),
                                      "prefill_tps": timings.get("prompt_per_second"),
                                      "decode_tps": timings.get("predicted_per_second"),
                                      "predicted_n": timings.get("predicted_n"), "draft_n": timings.get("draft_n"),
                                      "draft_n_accepted": timings.get("draft_n_accepted"), "cache_carried": None}
                        break
                phase("end")
                write_json(raw / "server.end.json", self.adapter.server_snapshot())
            except FloorBreach as exc:
                result.update(why=f"lowest free {gpu.low:,} MiB is under the {FLOOR_MIB:,} MiB floor",
                              stopped_early=str(exc), headroom_mib=gpu.low - FLOOR_MIB)
            except CapacityError as exc:
                result["why"] = f"out of memory: {exc}"[:300]
            except (InfrastructureError, ValueError, TimeoutError, ConnectionError) as exc:
                result.update(infra=True, why=f"server error: {exc}"[:300])
            finally:
                phase("stop")
                logs = sorted(self.out.glob(f"logs/{name}-{ctx}-*.log"), key=lambda p: p.stat().st_mtime)
                self.adapter.stop()  # raises when memory does not come back: that stops the sweep
        text = logs[-1].read_text(errors="replace") if logs else ""
        result["buffers"], result["info"] = parse_buffers(text)
        result["log"] = str(logs[-1]) if logs else None
        result["min_free_mib"], result["min_free_by_phase"] = gpu.low, gpu.lows
        result["seconds"] = round(time.monotonic() - started, 1)
        result["chat"], result["ladder"] = shorts, ladder
        if result["why"]:
            return result
        if not (shorts and ladder and all(t.get("predicted_per_second") for t in shorts)
                and all(p and p["decode_tps"] and p["prefill_tps"] for p in ladder)):
            result.update(infra=True, why="the server returned no timings")
            return result
        short = sum(t["predicted_per_second"] for t in shorts) / len(shorts)
        deep = ladder[-1]["decode_tps"]
        decode = 2 / (1 / short + 1 / deep)
        if all(p["cache_carried"] is not False for p in ladder):
            # one prefill of 0 to the deepest step, in segments: the same span a cold deepest request covers
            prefill = sum(p["prompt_n"] for p in ladder) / (sum(p["prompt_ms"] for p in ladder) / 1000)
        else:
            prefill = ladder[-1]["prefill_tps"]  # the cold deepest request alone, so presets stay comparable
        first = ladder[0]
        result["quick_score"] = round(score(first["prefill_tps"], 2 / (1 / short + 1 / first["decode_tps"])), 3)
        drafted = sum(t.get("draft_n") or 0 for t in shorts) + sum(p["draft_n"] or 0 for p in ladder)
        accepted = sum(t.get("draft_n_accepted") or 0 for t in shorts) + sum(p["draft_n_accepted"] or 0 for p in ladder)
        result.update(prefill_tps=round(prefill, 1), short_decode_tps=round(short, 1), deep_decode_tps=round(deep, 1),
                      decode_tps=round(decode, 1), accept=round(accepted / drafted, 3) if drafted else None,
                      score=round(score(prefill, decode), 3), deep_tokens=ladder[-1]["prompt_tokens"],
                      headroom_mib=None if gpu.low is None else gpu.low - FLOOR_MIB)
        if gpu.low is None:
            result.update(infra=True, why="no GPU memory reading")
        elif gpu.low < FLOOR_MIB:
            result["why"] = f"lowest free {gpu.low:,} MiB is under the {FLOOR_MIB:,} MiB floor"
        else:
            result.update(fits=True, why=f"lowest free {gpu.low:,} MiB, prefill {prefill:,.0f} t/s, "
                                         f"decode {decode:.1f} t/s")
        return result


class FloorBreach(Exception):
    """Free GPU memory went under the floor; the argument is the phase it happened in."""


class Cached:
    """Measurements already in results.jsonl are reused; infrastructure errors are retried."""

    def __init__(self, measure, path, say):
        self.measure, self.path, self.say, self.seen = measure, path, say, {}
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if not row.get("infra"):
                    self.seen[row["key"]] = row

    def __call__(self, name, settings, quick=False, repeat=0):
        key = config_key(name, settings, quick) + (f"-r{repeat}" if repeat else "")
        if key in self.seen:
            self.say(f"  (measured before: {self.seen[key]['why']})")
            return self.seen[key]
        result = {**self.measure(name, settings, quick=quick, **({"key": key} if repeat else {})), "key": key}
        append_jsonl(self.path, result)
        if not result.get("infra"):
            self.seen[key] = result
        return result


# -- report --------------------------------------------------------------------------------------
def changes(outcome):
    before, after = outcome["before"], outcome["after"]
    keys = sorted(set(before) | set(after))
    moved = {k: (before.get(k), after.get(k)) for k in keys if before.get(k) != after.get(k)}
    speed = {k: v for k, v in moved.items() if k not in PROPOSAL_KEYS}
    proposals = {k: v for k, v in moved.items() if k in PROPOSAL_KEYS}
    return speed, proposals


def fmt(moved):
    return "; ".join(f"{k} {a or 'unset'} → {b or 'removed'}" for k, (a, b) in moved.items()) or "none"


def report(outcomes, out, baseline, restored=None):
    lines = [f"# Speed sweep, {time.strftime('%Y-%m-%d')}", "",
             f"Free GPU memory with the router stopped: {baseline:,} MiB. A setting fits when the lowest free memory "
             f"during load, chat and a depth ladder to {FILL:.0%} of the context stays at or above {FLOOR_MIB:,} MiB "
             f"(the gate, with the working desktop open). Score: seconds for a reference turn of {PREFILL_REF:,} prompt "
             f"and {DECODE_REF:,} output tokens (lower is faster). Speed changes are the result; proposals change "
             f"context or KV and need the owner (and pi's contextWindow for a context change). Raw data per "
             f"measurement: raw/<key>/ (telemetry, requests, server snapshots); every result: results.jsonl.", "",
             "| Preset | Status | Score before → after | Faster | Headroom after | Speed changes | Proposals |",
             "|---|---|---|---|---|---|---|"]
    for o in outcomes:
        speed, proposals = changes(o)
        b, f = o["baseline"], o["final"]
        before = f"{b['score']:.1f}" if b.get("score") else "-"
        after = f"{f['score']:.1f}" if f.get("score") else "-"
        faster = f"{gain(f, b):+.0%}" if b.get("score") and f.get("score") else "-"
        head = f"{f['headroom_mib']:,}" if f.get("headroom_mib") is not None else "-"
        lines.append(f"| {o['preset']} | {o['status']} | {before} → {after} | {faster} | {head} | {fmt(speed)} | "
                     f"{fmt(proposals)} |")
    noisy = [o for o in outcomes if len(o.get("repeats") or []) > 1]
    if noisy:
        lines += ["", "## Run-to-run noise (the same setting measured again)", "",
                  "| Preset | Runs | Score (s) | Spread | Prefill t/s | Decode t/s | Lowest free MiB |", "|---|---|---|---|---|---|---|"]
        for o in noisy:
            runs = [r for r in o["repeats"] if r.get("score")]
            if not runs:
                continue
            def span(field, form):
                values = [r[field] for r in runs if r.get(field) is not None]
                return f"{form(min(values))} to {form(max(values))}" if values else "-"
            scores = [r["score"] for r in runs]
            spread = max(scores) / min(scores) - 1
            lines.append(f"| {o['preset']} | {len(runs)} | {span('score', lambda v: f'{v:.1f}')} | {spread:.1%} | "
                         f"{span('prefill_tps', lambda v: f'{v:,.0f}')} | {span('decode_tps', lambda v: f'{v:.1f}')} | "
                         f"{span('min_free_mib', lambda v: f'{v:,.0f}')} |")
        lines += ["", f"A speed change counts only past {GAIN:.0%}; a spread near that means the rule is choosing "
                      "by chance and should widen."]
    for o in outcomes:
        speed, proposals = changes(o)
        f = o["final"]
        lines += ["", f"## {o['preset']}", ""]
        if f.get("score"):
            accept = f"{f['accept']:.0%}" if f.get("accept") is not None else "n/a"
            lines.append(f"After: prefill {f['prefill_tps']:,.0f} t/s, decode {f['decode_tps']:.1f} t/s "
                         f"(short {f['short_decode_tps']:.1f}, at {f.get('deep_tokens', 0):,} tokens "
                         f"{f['deep_decode_tps']:.1f}), draft acceptance {accept}, lowest free "
                         f"{f['min_free_mib']:,} MiB.")
            depth = [p for p in f.get("ladder") or [] if p]
            if depth:
                lines.append("Depth: " + ", ".join(f"{p['prompt_tokens']:,} → prefill {p['prefill_tps']:,.0f}, "
                                                   f"decode {p['decode_tps']:.1f}" for p in depth) + " t/s.")
        if speed:
            lines += ["", "Preset lines:", "", "```ini",
                      *[f"{k} = {b}" if b is not None else f"# remove: {k}" for k, (_, b) in speed.items()], "```"]
        if proposals:
            lines += ["", f"Proposal (owner): {fmt(proposals)}. Verified to fit with the speed changes above."]
        lines += ["", "| Knob | Value | Outcome | Why | Score | Headroom |", "|---|---|---|---|---|---|"]
        for row in o["trail"]:
            score_text = "" if row.get("score") is None else row["score"]
            head_text = "" if row.get("headroom_mib") is None else row["headroom_mib"]
            lines.append(f"| {row['knob']} | {row['value']} | {row['outcome']} | {row['why']} | {score_text} | "
                         f"{head_text} |")
    if restored is not None:
        lines += ["", f"Router restore: {json.dumps(restored)}"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_json(out / "summary.json", {"baseline_free_mib": baseline, "outcomes": outcomes, "restored": restored})


# -- main ----------------------------------------------------------------------------------------
def dry_run(names, presets, speed_first):
    for name in names:
        info = model_info(presets[name], None)
        found, skipped = knobs(presets[name], info, name in speed_first)
        print(f"{name}:")
        for knob in found:
            values = knob.get("values")
            how = "drafter" if knob.get("drafter") else "step" + (", must be faster" if knob["gain"] else "")
            start = f" from {values[knob['start']]}" if values and knob["start"] else ""
            print(f"  {knob['name']} ({how}){start}: " + (" → ".join(map(str, values)) if values else "off arm"))
        for knob, why in skipped:
            print(f"  {knob}: skipped, {why}")
        down = [label for label, _ in rescues(presets[name], info, name)]
        print("  if it does not fit as written: " + (" → ".join(down) if down else "nothing to step down"))
    print("Context ladders are capped at each model's trained maximum, read from the model file's header.")


def _stop(number, frame):
    raise KeyboardInterrupt(signal.Signals(number).name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--presets", help="comma-separated preset names (default: every preset in the file but --skip)")
    parser.add_argument("--skip", default=",".join(SKIP),
                        help=f"presets the default list leaves out (default {','.join(SKIP)}); '' keeps all")
    parser.add_argument("--speed-first", default="", help="presets that spend headroom on speed before context")
    parser.add_argument("--baseline-only", action="store_true", help="measure every preset as written; tune nothing")
    parser.add_argument("--repeat", type=int, default=1,
                        help="with --baseline-only: measure each preset this many times (the noise check)")
    parser.add_argument("--dry-run", action="store_true", help="print the knobs each preset would try")
    parser.add_argument("--preset-file", default=str(DEFAULT_PRESET))
    parser.add_argument("--out", help="run folder (default <data folder>/<date>-speed-sweep)")
    args = parser.parse_args(argv)
    if args.repeat < 1 or (args.repeat > 1 and not args.baseline_only):
        parser.error("--repeat takes a positive count and needs --baseline-only")
    _, presets = read_presets(args.preset_file)
    skip = set(filter(None, args.skip.split(",")))
    names = args.presets.split(",") if args.presets else [n for n in presets if n not in skip]
    missing = [n for n in names if n not in presets]
    if missing:
        parser.error(f"not in {args.preset_file}: {', '.join(missing)}")
    speed_first = set(filter(None, args.speed_first.split(",")))
    if args.dry_run:
        dry_run(names, presets, speed_first)
        return 0
    out = Path(args.out or RUNS / f"{time.strftime('%Y-%m-%d')}-speed-sweep").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    say = lambda message: log(message, out)  # noqa: E731
    plan = {"production": {"unit": ROUTER, "restore_model": DEFAULT_RESTORE},
            "limits": {"load_seconds": LOAD_SECONDS, "request_seconds": CHAT_SECONDS}}
    adapter = LiveAdapter(plan, out / "logs", run_id=f"speed-sweep-{int(time.time())}")
    snapshot = adapter.snapshot()
    if snapshot["unknown_gpu_clients"]:
        say(f"refused: another program holds GPU memory: {', '.join(snapshot['unknown_gpu_clients'])}")
        return 2
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, _stop)
    outcomes, restored, sampler = [], None, None
    with inhibit.sleep_inhibit("llm-eval speed sweep", guard=_uninterruptible):
        try:
            baseline = adapter.quiesce()["free_mib"]
            say(f"router stopped; {baseline:,} MiB free")
            if baseline < BASELINE_WARN_MIB:
                say(f"warning: under {BASELINE_WARN_MIB:,} MiB free with the router stopped; is Steam open? "
                    "Results are then conservative for a lighter desktop")
            try:
                write_json(out / "environment.json", {**adapter.environment(), "snapshot": snapshot,
                                                      "baseline_free_mib": baseline, "argv": sys.argv,
                                                      "preset_file": args.preset_file})
            except Exception:
                pass  # derived data: never stops the sweep
            ramcheck.snapshot(out / "ram.jsonl", "sweep_start")
            host_sample = getattr(adapter, "sample_host", None)
            if host_sample:  # RAM, load, pressure and GPU every 5 s, tagged by preset; derived data
                sampler = GpuSampler(host_sample, out / "telemetry" / "host.jsonl").start()
            sweep = Sweep(Cached(LiveMeasure(adapter, out), out / "results.jsonl", say), say)
            for name in names:
                say(f"{name}")
                if sampler is not None:
                    sampler.tag = name
                ramcheck.snapshot(out / "ram.jsonl", f"{name}_start")
                outcomes.append(sweep.preset(name, dict(presets[name]), name in speed_first, args.baseline_only,
                                             args.repeat))
                report(outcomes, out, baseline)
        except KeyboardInterrupt as exc:
            say(f"stopped ({exc or 'SIGINT'}); restoring the router")
        finally:
            with _uninterruptible():
                if sampler is not None:
                    sampler.stop()
                try:
                    adapter.stop()
                finally:
                    restored = adapter.restore(snapshot)
                    say(f"router restored: {restored}")
                    report(outcomes, out, adapter.baseline or 0, restored)
    say(f"report: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
