"""The benchmark's opencode config: production's, narrowed to one local model on loopback.

Production routes agents through role tiers (``OC_MODEL_*``) that the router serves with
different models. The benchmark has one scratch server holding the profile under test, so every
tier and agent points at that profile's production model entry, whose ``limit`` (and so the
compaction point) is kept exactly. Agent sampling options, permissions and compaction settings
are kept. Removed: every provider but the local one, MCP servers (one reaches the personal
vault), and auto-update. The result is refused unless every provider URL is literal loopback.
"""
import copy
import ipaddress
import re
from urllib.parse import urlparse

LOCAL_PROVIDER = "local-llama"
SANDBOX_PORT = 8390  # the in-sandbox relay port; arbitrary, kept off the production router's 8080
SANDBOX_BASE_URL = f"http://127.0.0.1:{SANDBOX_PORT}/v1"
TIERS = ("OC_MODEL_BALANCED", "OC_MODEL_REASONING", "OC_MODEL_CODING", "OC_MODEL_UTILITY")


def parse_jsonc(text):
    """JSON with // and /* */ comments and trailing commas (opencode's config dialect)."""
    import json
    out, i, n, in_string = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\":
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        else:
            out.append(c)
        i += 1
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)))


def is_loopback_url(url):
    parsed = urlparse(url)
    try:
        return parsed.scheme == "http" and ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        return False


def model_entry(production, profile, covers=(), entry=None):
    """The production model key for a profile. With no ``entry`` the bare profile's own entry is used (the
    ``-agent`` entries are the delegation target for hosted sessions, a different use case); ``entry`` names
    any other entry explicitly and the acceptance card always prints which one was used."""
    models = production.get("provider", {}).get(LOCAL_PROVIDER, {}).get("models", {})
    names = [n for n in (profile, *covers) if n in models]
    if entry is not None:
        if entry not in names:
            raise ValueError(f"entry {entry!r} is not a {LOCAL_PROVIDER} model entry for {profile} "
                             f"(choose from {', '.join(names) or 'none'})")
        return entry
    if names:
        return names[0]
    raise ValueError(f"no {LOCAL_PROVIDER} model entry for {profile} (or {', '.join(covers) or 'aliases'})")


def build(production, profile, covers=(), entry=None, context=None, output=None):
    """Return (config, model_ref, limit) for running ``profile`` in the sandbox. ``context`` and ``output``
    override the entry's ``limit.context`` and ``limit.output`` in the sandbox copy only (production is never edited)."""
    entry = model_entry(production, profile, covers, entry)
    ref = f"{LOCAL_PROVIDER}/{entry}"
    config = copy.deepcopy(production)
    local = copy.deepcopy(production["provider"][LOCAL_PROVIDER])
    local.setdefault("options", {})["baseURL"] = SANDBOX_BASE_URL
    config["provider"] = {LOCAL_PROVIDER: local}
    config.update(model=ref, small_model=ref, autoupdate=False, share="disabled", disabled_providers=[])
    config.pop("mcp", None)
    config.pop("plugin", None)
    for agent in config.get("agent", {}).values():
        if isinstance(agent, dict) and "model" in agent:
            agent["model"] = ref
    if context is not None:
        local["models"][entry].setdefault("limit", {})["context"] = context
    if output is not None:
        local["models"][entry].setdefault("limit", {})["output"] = output
    check(config)
    return config, ref, local["models"][entry].get("limit", {})


def check(config):
    """Refuse a config that could reach anything but loopback."""
    providers = config.get("provider", {})
    if set(providers) != {LOCAL_PROVIDER}:
        raise ValueError(f"benchmark config must have only {LOCAL_PROVIDER}, found {sorted(providers)}")
    url = providers[LOCAL_PROVIDER].get("options", {}).get("baseURL", "")
    if not is_loopback_url(url):
        raise ValueError(f"provider baseURL must be http on a loopback IP, got {url!r}")
    if config.get("mcp") or config.get("plugin") or config.get("share") != "disabled":
        raise ValueError("benchmark config must not enable MCP, plugins or sharing")
    for name, agent in (config.get("agent") or {}).items():
        model = agent.get("model") if isinstance(agent, dict) else None
        if model and not model.startswith(LOCAL_PROVIDER + "/"):
            raise ValueError(f"agent {name} points outside {LOCAL_PROVIDER}: {model}")


def environment(ref):
    """Env for the sandbox, so any leftover {env:...} reference also resolves to the profile."""
    return {**{tier: ref for tier in TIERS}, "OC_LLAMA_BASEURL": SANDBOX_BASE_URL, "OC_SHELL": "fish",
            "OC_DISABLED_PROVIDERS": ""}
