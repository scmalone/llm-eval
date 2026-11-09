"""The benchmark's pi config: one loopback-only provider for the model under test.

pi reads its agent directory (``$PI_CODING_AGENT_DIR``, default ``~/.pi/agent``) from the fake home,
so the task's own copy holds both files that matter:

* ``models.json``: the ``local-llama`` provider pointed at the sandbox relay, one model whose
  ``contextWindow``, ``maxTokens`` and sampling match the pinned request exactly;
* ``settings.json``: where compaction starts, so the harness compacts where the owner would.

With a production entry (``configs/pi-production.json``, the owner's own ``models.json`` for the local
providers) the model is described exactly as production describes it: the same ``maxTokens``, ``input``,
``reasoning`` flag, ``thinkingLevelMap`` and ``compat``, so pi builds the same request (role, token field,
thinking parameters) it builds for the owner, and compacts at the owner's reserve. The run's AGENTS.md is the
owner's working agreement, pinned into the plan's home files by the prepare step.

Thinking is pi's to send (2026-10-31): an entry with ``compat.thinkingFormat = "chat-template"`` maps pi's
thinking level to the model's own template variables (``enable_thinking``, ``reasoning_effort``,
``reasoning_strength``) and caps thinking with a per-request budget field. ``expected_request`` replays that
mapping, so the capture supplies nothing pi would not send and refuses only real drift.

The result is refused unless the endpoint is literal loopback and no other provider, MCP server or
package is reachable. Nothing here touches the real ``~/.pi``.
"""
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlparse

LOCAL_PROVIDER = "local-llama"
SANDBOX_PORT = 8390  # the in-sandbox relay port; the same one opencode uses, one harness at a time
SANDBOX_BASE_URL = f"http://127.0.0.1:{SANDBOX_PORT}/v1"
# pi's thinking levels, and the provider value each maps to. The pinned request asks for "high".
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
# Sampling fields the pinned request carries. pi merges samplingParams into the request body verbatim,
# so these arrive at the server exactly as configured (chat_template_kwargs included).
SAMPLING = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
PRODUCTION = Path(__file__).resolve().parents[2] / "configs" / "pi-production.json"
AGENT_DIR = Path(".pi", "agent")  # pi's agent directory, relative to the (fake) home
# pi's DEFAULT_COMPACTION_SETTINGS (core/compaction/compaction.js, pi 1.4.0), used where the frozen copy has none.
DEFAULT_RESERVE, DEFAULT_KEEP_RECENT = 16384, 20000
# pi-ai's thinking budgets per level (api/simple-options.js, pi 1.4.0): xhigh and max use high's, and at least
# MIN_ANSWER_TOKENS of max_tokens stay for the answer.
THINKING_BUDGETS = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384}
MIN_ANSWER_TOKENS = 1024
# Entry fields that decide what pi sends for thinking; copied from production verbatim.
THINKING_FIELDS = ("reasoning", "thinkingLevelMap", "compat")


def _frozen(path=PRODUCTION):
    return json.loads(Path(path).read_text())


def production_entry(profile, path=PRODUCTION):
    """The owner's pi entry for a benchmark profile (Strata's profile is pi's ``strata``), or None."""
    frozen = _frozen(path)
    entry = frozen["models"].get(frozen.get("aliases", {}).get(profile, profile))
    return dict(entry) if entry else None


def production_compaction(path=PRODUCTION):
    """(reserveTokens, keepRecentTokens) from the owner's settings.json, as frozen; pi's defaults otherwise."""
    compaction = _frozen(path).get("compaction")
    compaction = compaction if isinstance(compaction, dict) else {}
    return compaction.get("reserveTokens", DEFAULT_RESERVE), compaction.get("keepRecentTokens", DEFAULT_KEEP_RECENT)


def production_thinking_level(path=PRODUCTION):
    """The owner's defaultThinkingLevel, as frozen ("high" when the copy does not say)."""
    return _frozen(path).get("default_thinking_level", "high")


def supported_levels(entry):
    """pi's getSupportedThinkingLevels: a level mapped to null is unsupported; xhigh and max need a mapping."""
    if not entry.get("reasoning"):
        return ["off"]
    mapping = entry.get("thinkingLevelMap") or {}
    return [level for level in THINKING_LEVELS
            if not (level in mapping and mapping[level] is None)
            and (level not in ("xhigh", "max") or level in mapping)]


def clamp_level(entry, level):
    """pi's clampThinkingLevel: the level itself, else the nearest supported one above, else below."""
    available = supported_levels(entry)
    if level in available:
        return level
    if level not in THINKING_LEVELS:
        return available[0] if available else "off"
    index = THINKING_LEVELS.index(level)
    for candidate in THINKING_LEVELS[index:] + THINKING_LEVELS[:index][::-1]:
        if candidate in available:
            return candidate
    return "off"


def thinking_request(entry, level, output):
    """The thinking fields pi 1.4.0 adds to a chat request for this entry at ``level``, with max_tokens ``output``
    (openai-completions.js: the chat-template format and the top-level budget field). Empty for an entry that
    declares no reasoning or uses another thinking format."""
    if not entry or not entry.get("reasoning"):
        return {}
    clamped = clamp_level(entry, level)
    effort = None if clamped == "off" else clamped
    compat = entry.get("compat") or {}
    budget = None
    if effort is not None:
        wanted = THINKING_BUDGETS["high" if effort in ("xhigh", "max") else effort]
        budget = min(wanted, max(0, output - MIN_ANSWER_TOKENS)) or None
    fields = {}
    if compat.get("thinkingFormat") == "chat-template":
        mapping, kwargs = entry.get("thinkingLevelMap") or {}, {}
        for key, value in (compat.get("chatTemplateKwargs") or {}).items():
            if not isinstance(value, dict):
                kwargs[key] = value
            elif effort is None and value.get("omitWhenOff"):
                continue
            elif value.get("$var") == "thinking.enabled":
                kwargs[key] = effort is not None
            elif value.get("$var") == "thinking.budget":
                if budget is not None:
                    kwargs[key] = budget
            else:
                name = effort if effort is not None else "off"
                if name not in mapping:
                    if effort is not None:
                        kwargs[key] = effort
                elif isinstance(mapping[name], str):
                    kwargs[key] = mapping[name]
        if kwargs:
            fields["chat_template_kwargs"] = kwargs
    elif compat.get("thinkingFormat") in (None, "openai") and compat.get("supportsReasoningEffort", True):
        # OpenAI style: the level, through thinkingLevelMap, as a top-level reasoning_effort
        mapping = entry.get("thinkingLevelMap") or {}
        value = mapping.get(effort, effort) if effort is not None else mapping.get("off")
        if isinstance(value, str):
            fields["reasoning_effort"] = value
    elif compat.get("thinkingFormat") not in (None, "openai"):
        raise ValueError(f"thinking format {compat.get('thinkingFormat')!r} is not modelled by the harness")
    field = compat.get("thinkingTokenBudgetField") or ("thinking_token_budget" if compat.get("supportsThinkingTokenBudget") else None)
    if field and budget is not None:
        fields[field] = budget
    return fields


def is_loopback_url(url):
    parsed = urlparse(url)
    try:
        return parsed.scheme == "http" and ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        return False


def sampling_of(request):
    """The sampling part of the pinned request, plus any chat-template kwargs it pins."""
    sampling = {k: v for k, v in request.items() if k in SAMPLING}
    if isinstance(request.get("chat_template_kwargs"), dict):
        sampling["chat_template_kwargs"] = dict(request["chat_template_kwargs"])
    return sampling


def build(model_id, context, output, request, effort="high", compaction_at=None, production=None):
    """Return (models.json, settings.json, ref) for running ``model_id`` in the sandbox.

    ``compaction_at`` is the context size at which compaction starts. Without it, a production entry
    compacts at the owner's reserve, as the owner's pi does; otherwise one full output plus the harness
    margin stays free.
    """
    if effort not in THINKING_LEVELS:
        raise ValueError(f"unknown thinking level {effort!r}")
    default = context - (production_compaction()[0] if production is not None else output + 1024)
    reserve = context - (compaction_at if compaction_at is not None else default)
    if reserve < 0:
        raise ValueError(f"compaction_at {compaction_at} leaves no room in a {context} context")
    entry = {"id": model_id, "name": model_id, "reasoning": True, "input": ["text"],
             "contextWindow": context, "maxTokens": output,
             "thinkingLevelMap": {level: level for level in THINKING_LEVELS}}
    if production is not None:
        # Production's own thinking fields, so pi maps its thinking level to the model's template exactly as it
        # does for the owner. An entry without them (Strata) sends no thinking fields; the capture supplies those.
        entry = {"id": model_id, "name": model_id, "input": list(production.get("input", ["text"])),
                 "contextWindow": context, "maxTokens": output}
        entry.update({k: json.loads(json.dumps(production[k])) for k in THINKING_FIELDS if k in production})
    models = {"providers": {LOCAL_PROVIDER: {
        "baseUrl": SANDBOX_BASE_URL, "api": "openai-completions", "apiKey": "local-benchmark",
        "models": [{**entry, "samplingParams": sampling_of(request)}]}}}
    keep = production_compaction()[1] if production is not None else DEFAULT_KEEP_RECENT
    settings = {"compaction": {"enabled": True, "reserveTokens": reserve, "keepRecentTokens": keep},
                "showCacheMissNotices": False}
    check(models, settings)
    return models, settings, f"{LOCAL_PROVIDER}/{model_id}"


def check(models, settings):
    """Refuse a config that could reach anything but loopback, or load anything but the one model."""
    providers = models.get("providers") or {}
    if set(providers) != {LOCAL_PROVIDER}:
        raise ValueError(f"benchmark config must have only {LOCAL_PROVIDER}, found {sorted(providers)}")
    if not is_loopback_url(providers[LOCAL_PROVIDER].get("baseUrl", "")):
        raise ValueError(f"provider baseUrl must be http on a loopback IP, got {providers[LOCAL_PROVIDER].get('baseUrl')!r}")
    if models.get("mcp") or models.get("packages") or models.get("extensions"):
        raise ValueError("benchmark config must not enable MCP, packages or extensions")
    reserve = (settings.get("compaction") or {}).get("reserveTokens")
    if reserve is not None and reserve < 0:
        raise ValueError("compaction.reserveTokens must not be negative")


def expected_request(model_id, context, output, request, effort="high", production=None):
    """What the capture should see the client send, so drift is a real change and not a harness default.

    A production entry that declares reasoning makes pi the source of the thinking fields: the contract is what
    pi sends at ``effort``, and no top-level reasoning_effort is added (llama.cpp would pass it to the template,
    and Swift's template rejects "high")."""
    thinking = thinking_request(production, effort, output) if production else {}
    if thinking or (production or {}).get("reasoning"):
        expected = {**sampling_of(request), "max_tokens": output, **thinking}
    else:
        expected = {**sampling_of(request), "max_tokens": output, "reasoning_effort": effort}
    if "seed" in request:
        expected["seed"] = request["seed"]  # a paired attempt's seed: pi never sends one, so the capture supplies it
    expected.pop("reasoning_budget_tokens", None)  # pi has no such field; a missing key is not a conflict
    return expected


def write(directory, models, settings):
    """Write both files into the task's agent directory and return the paths."""
    agent = Path(directory) / AGENT_DIR
    agent.mkdir(parents=True, exist_ok=True)
    (agent / "models.json").write_text(json.dumps(models, indent=2))
    (agent / "settings.json").write_text(json.dumps(settings, indent=2))
    return agent / "models.json", agent / "settings.json"
