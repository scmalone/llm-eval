"""Inference-runtime selection, independent of the fake/live execution backend."""
from .live import LiveAdapter
from .strata import StrataAdapter


def adapter_class(plan):
    runtime = plan.get("runtime", "llama.cpp")
    if runtime == "llama.cpp":
        return LiveAdapter
    if runtime == "strata":
        return StrataAdapter
    raise ValueError(f"unsupported inference runtime: {runtime}")
