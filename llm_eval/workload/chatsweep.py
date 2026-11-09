"""Synthetic chat-depth sweep (diagnostic tier), generated once per seed into a workload task.

One conversation through real opencode: a briefing with planted facts, pasted batches of records
each with a local question, a correction that supersedes one fact, and probe turns at growing
depth that ask for an original fact, the corrected fact and a record from the first batch. Every
turn has a checkable answer, and the probes say whether the model still uses what it was told
earlier as the session nears and passes compaction. All content derives from the seed.
"""
import random

from ..haystack import SITES, WORDS

LINES_PER_BATCH = 520
BATCHES = 9
PROBE_AFTER = (3, 6, 9)


def _batch(rng, number):
    items = {}
    lines = [f"Batch {number}:"]
    for _ in range(LINES_PER_BATCH):
        name = f"{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
        qty = rng.randint(10, 999)
        items[name] = qty
        lines.append(f"item {name}: qty {qty}")
    return "\n".join(lines), items


def build(seed):
    """(prompt, scripted follow-ups, per-turn checks, solution answers) for one seed."""
    rng = random.Random(f"chatsweep:{seed}")
    sites = rng.sample(SITES, 6)
    codes = {s: rng.randint(1000, 9999) for s in sites}
    codename = rng.choice(WORDS).upper() + "-" + str(rng.randint(10, 99))
    corrected_site = sites[0]
    new_code = rng.randint(1000, 9999)
    while new_code == codes[corrected_site]:
        new_code = rng.randint(1000, 9999)
    briefing = ("For this chat, answer with only the value I ask for, nothing else, and don't use tools. "
                "Remember these site codes:\n" + "\n".join(f"{s}: {codes[s]}" for s in sites)
                + f"\nThe project codename is {codename}. Reply OK.")
    turns, checks, answers = [], {"1": {"all": [r"(?i)\bok\b"]}}, ["OK"]
    first_batch = None
    number_re = lambda n: rf"(?<!\d){n}(?!\d)"
    for batch in range(1, BATCHES + 1):
        text, items = _batch(rng, batch)
        first_batch = first_batch or items
        name = rng.choice(sorted(items))
        turns.append(f"{text}\n\nIn batch {batch}, what is the qty for item {name}? Answer with the number only.")
        checks[str(len(turns) + 1)] = {"all": [number_re(items[name])]}
        answers.append(str(items[name]))
        if batch == 3:
            turns.append(f"Correction: the code for {corrected_site} is now {new_code}. Reply OK.")
            checks[str(len(turns) + 1)] = {"all": [r"(?i)\bok\b"]}
            answers.append("OK")
        if batch in PROBE_AFTER:
            original = sites[1 + PROBE_AFTER.index(batch)]
            early = rng.choice(sorted(first_batch))
            turns.append(f"Three answers, one per line, values only: 1) the code for {original} from my first message; "
                         f"2) the current code for {corrected_site}; 3) in batch 1, the qty for item {early}.")
            checks[str(len(turns) + 1)] = {"all": [number_re(codes[original]), number_re(new_code), number_re(first_batch[early])],
                                           "none": [number_re(codes[corrected_site])]}
            answers.append(f"{codes[original]}\n{new_code}\n{first_batch[early]}")
    scripted = [{"after_turn": i + 1, "say": text} for i, text in enumerate(turns)]
    return briefing, scripted, checks, answers
