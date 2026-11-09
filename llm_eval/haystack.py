"""Deterministic recall haystacks sized to a real token budget.

A fixture declares ``haystack: {records, seed}`` instead of a fixed prompt. The runner
materializes it after a context is selected, so the filler is grown until the tokenizer
reports the requested prompt size. Everything derives from the seed, so a resumed run
rebuilds the same prompt.
"""
import random

SITES = ("northwind", "kestrel", "alderbrook", "tamarack", "ironwood", "solstice",
         "marlowe", "halcyon", "brackenridge", "quillfeather", "saltmarsh", "windermere",
         "ashgrove", "cormorant", "lindenhall", "fernbank", "harrowgate", "oakhaven",
         "pellucid", "ravenscar", "stonebridge", "thistledown", "umberfield", "wrenmoor")
WORDS = ("ledger", "transit", "harbor", "signal", "archive", "relay", "orchard", "census",
         "beacon", "quarry", "filament", "meridian", "tariff", "lantern", "cistern", "vellum",
         "gradient", "ferry", "bulletin", "granary", "turbine", "parcel", "outpost", "cadence",
         "shipment", "lighthouse", "survey", "depot", "canal", "foundry", "register", "pylon")
BLOCK_WORDS = 48
TOLERANCE = 0.005
INSTRUCTION = ("Return exactly one mapping per line, formatted as site:code, one for every "
               "RECORD line above in the order they appear. Output nothing else.")


def records_for(spec):
    rng = random.Random(spec["seed"])
    count = spec["records"]
    if not 1 <= count <= len(SITES):
        raise ValueError("records must be between 1 and %d" % len(SITES))
    sites = rng.sample(SITES, count)
    codes = rng.sample(range(1000, 9999), count)
    return [(site, code) for site, code in zip(sites, codes)]


def filler_block(seed, index):
    rng = random.Random(f"{seed}:{index}")
    return " ".join(rng.choice(WORDS) for _ in range(BLOCK_WORDS)) + "."


def render(spec, blocks):
    """Prompt with ``blocks`` filler paragraphs and the records spread evenly through them."""
    records = records_for(spec)
    slots = {}
    for number, (site, code) in enumerate(records):
        slots.setdefault((number * blocks) // len(records) if blocks else 0, []).append(
            f"RECORD: site={site} code={code}")
    lines = ["Read this log. Most of it is noise; only the RECORD lines matter."]
    for index in range(max(blocks, 1)):
        if index < blocks:
            lines.append(filler_block(spec["seed"], index))
        lines.extend(slots.get(index, []))
    lines.append(INSTRUCTION)
    return "\n".join(lines), [f"{site}:{code}" for site, code in records]


def build(spec, target_tokens, count_tokens, max_rounds=12):
    """Largest prompt not above ``target_tokens`` within TOLERANCE, measured by count_tokens."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    blocks = max(1, target_tokens // 80)
    best = None
    for _ in range(max_rounds):
        prompt, expected = render(spec, blocks)
        measured = count_tokens(prompt)
        if measured <= target_tokens and (best is None or measured > best[2]):
            best = (prompt, expected, measured, blocks)
        if measured <= target_tokens and measured >= target_tokens * (1 - TOLERANCE):
            break
        scaled = max(1, round(blocks * target_tokens / max(measured, 1) * (1 - TOLERANCE / 2)))
        if scaled == blocks:
            scaled += -1 if measured > target_tokens else 1
        blocks = max(1, scaled)
    if best is None:
        raise ValueError("could not fit a haystack within %d tokens" % target_tokens)
    return {"prompt": best[0], "expected": best[1], "prompt_tokens": best[2], "filler_blocks": best[3]}
