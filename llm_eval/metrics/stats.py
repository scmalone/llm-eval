"""Small, dependency-free helpers for analysing the corpus."""
import math
import random


def wilson_interval(passes, total, z=1.96):
    """Wilson score interval for a pass rate; (None, None) when there are no trials. Small samples
    are the norm here (3 to 10 runs), where the normal approximation lies."""
    if total <= 0:
        return None, None
    p = passes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def paired_bootstrap(a, b, resamples=10000, seed=0, level=0.95):
    """Mean of ``a[i] - b[i]`` over paired tasks, with a percentile bootstrap interval. ``a`` and
    ``b`` are per-task scores (for example pass rates) for two profiles on the SAME task hashes."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty paired samples")
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(resamples))
    low = means[int((1 - level) / 2 * resamples)]
    high = means[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return {"mean_diff": sum(diffs) / len(diffs), "low": low, "high": high, "n": len(diffs)}


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
