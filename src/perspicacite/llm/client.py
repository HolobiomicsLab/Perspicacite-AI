"""Async LLM client using LiteLLM."""

import time
from collections.abc import AsyncIterator
from typing import Any, Protocol

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from perspicacite.config.schema import LLMConfig, LLMProviderConfig
from perspicacite.llm.cache import LLMResponseCache, build_cache_key
from perspicacite.logging import get_logger

logger = get_logger("perspicacite.llm")


def _safe_completion_cost(response: Any, model: str) -> float:
    """Best-effort cost lookup via ``litellm.completion_cost``.

    Returns 0.0 on any failure (unknown model, missing usage, lookup
    table mismatch) and logs a warning so operators can spot mis-priced
    models. Never raises — token telemetry must keep flowing even when
    pricing data is stale.
    """
    try:
        import litellm as _litellm
        return float(_litellm.completion_cost(completion_response=response) or 0.0)
    except Exception as exc:  # pragma: no cover — depends on litellm tables
        logger.warning(
            "llm_cost_lookup_failed",
            model=model,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return 0.0


def _emit_usage_telemetry(
    sink: Any,
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    """Forward token + cost events to a telemetry sink (best-effort).

    Imported lazily so the LLM client has no hard dependency on
    perspicacite.rag.telemetry import order. Swallows every error —
    telemetry must never break the pipeline.
    """
    if sink is None:
        return
    try:
        from perspicacite.rag.telemetry import emit_cost, emit_tokens
        emit_tokens(
            sink,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            model=model,
            provider=provider,
        )
        emit_cost(sink, usd=cost_usd, model=model, provider=provider)
    except Exception:
        pass

# F9 (audit 2026-05-15): LiteLLM prints a "Give Feedback / Get Help"
# banner to stderr on every error, plus an "If you need to debug…" info
# line. These pollute our structured logs and operator terminals. Silence
# them at module load. The banner is gated on ``litellm.suppress_debug_info``
# (see litellm/utils.py and litellm/router.py).
import logging as _stdlib_logging

try:
    import litellm as _litellm
    _litellm.suppress_debug_info = True
except Exception:  # pragma: no cover — litellm is a hard dep
    pass
_stdlib_logging.getLogger("LiteLLM").setLevel(_stdlib_logging.ERROR)
_stdlib_logging.getLogger("litellm").setLevel(_stdlib_logging.ERROR)

# Global LiteLLM timeout fallback (Issue 1 — three-tier policy).
# Overridden by llm.default_timeout_s in config, or by timeout= kwarg per call.
DEFAULT_LLM_TIMEOUT_S: float = 60.0


def _maybe_wrap_error(exc: Exception, provider: str) -> Exception:
    """If ``exc`` is a known LLM-error pattern (rate limit, auth),
    return a fresh structured exception. Otherwise return ``exc``
    unchanged."""
    cls_name = type(exc).__name__
    msg = str(exc)

    # Rate-limit takes priority over auth — see test_rate_limit_wins_when_both_patterns_match.
    from perspicacite.llm.errors import (
        AuthError,
        RateLimitError,
        detect_auth_error,
        detect_rate_limit,
        suggested_action,
    )
    if cls_name == "RateLimitError" or cls_name.endswith(".RateLimitError"):
        return RateLimitError(
            f"{provider}: rate limit. {suggested_action(provider)}",
            provider=provider,
        )
    hit = detect_rate_limit(msg)
    if hit is not None:
        return RateLimitError(
            f"{provider}: rate limit. {suggested_action(provider)}",
            provider=provider,
            retry_after_seconds=hit.retry_after_seconds,
        )
    if cls_name in ("AuthenticationError", "PermissionDeniedError") or detect_auth_error(msg):
        return AuthError(
            f"{provider}: auth failed. {suggested_action(provider, hint=_auth_hint(msg))}",
            provider=provider,
        )
    return exc


def _auth_hint(msg: str) -> str:
    """F3 (audit 2026-05-15): distinguish invalid-key from quota-exceeded.

    The Anthropic API returns ``invalid x-api-key`` for revoked/missing keys
    and ``rate_limit_error`` / ``billing`` for quota breaches. Today both
    end up in the same ``suggested_action`` string telling the user to
    "wait for quota reset" — confusing for first-time users with a typo'd key.
    """
    lower = (msg or "").lower()
    if any(
        k in lower
        for k in ("invalid x-api-key", "invalid api key", "no api key", "unauthorized", "api key not found", "missing api key")
    ):
        return "missing_or_invalid_key"
    if any(k in lower for k in ("quota", "billing", "credit balance", "usage limit")):
        return "quota_exceeded"
    return "unknown"


def _should_trigger_free_fallback(exc: Exception) -> bool:
    """Return True when the error warrants trying free-tier fallback models.

    Triggers on:
    - Invalid / unknown model ID ("not a valid model id", "model not found")
    - Quota / billing / credit exhausted
    - Auth errors (no key, wrong key) — a free-tier key might still work

    Does NOT trigger on budget-cap breaches (user-set limit) or generic
    network / timeout errors (those should retry on the same model).
    """
    from perspicacite.llm.errors import AuthError
    if isinstance(exc, AuthError):
        return True
    msg = str(exc).lower()
    return bool(any(k in msg for k in ("not a valid model", "model not found", "no endpoints found", "quota", "billing", "credit balance", "usage limit", "insufficient", "invalid api key", "authentication")))


def _is_deterministic_fail(exc: Exception) -> bool:
    """F1 (audit 2026-05-15): tenacity predicate for "do not retry".

    Re-raises immediately if the wrapped form of ``exc`` is an ``AuthError``
    or ``BudgetExceededError`` — neither will resolve on its own.
    """
    from perspicacite.llm.errors import AuthError

    if isinstance(exc, AuthError):
        return True
    # Late import — BudgetExceededError lives in the budget module.
    try:
        from perspicacite.llm.budget import BudgetExceededError
        if isinstance(exc, BudgetExceededError):
            return True
    except Exception:  # pragma: no cover — module always importable
        pass
    # Detect the wrapped class without invoking _maybe_wrap_error
    # (avoids circular logic with the retry decorator).
    cls_name = type(exc).__name__
    if cls_name in ("AuthenticationError", "PermissionDeniedError"):
        return True
    msg = str(exc).lower()
    from perspicacite.llm.errors import detect_auth_error
    return bool(detect_auth_error(msg))


def resolve_stage_model(
    config: Any,
    stage: str,
) -> tuple[str, str]:
    """Pick ``(provider, model)`` for an LLM call site by stage name.

    Stages live in ``config.llm.models`` / ``config.llm.providers_per_stage``
    keyed by the stage name (e.g. ``"routing"``, ``"screening"``,
    ``"synthesis_basic"``). Falls back to the global defaults when the
    stage isn't pinned. Returns the pair the caller should pass to
    :meth:`AsyncLLMClient.complete`.

    Why a helper and not a config method: kb_router / screen / rephrase
    are imported by code paths that don't see the full config object,
    they receive an already-constructed ``llm_client``. This function
    lets the orchestrator MCP / CLI / REST layer resolve once and pass
    the explicit pair down.
    """
    if config is None:
        return "anthropic", "claude-haiku-4-5"
    llm_cfg = getattr(config, "llm", None)
    if llm_cfg is None:
        return "anthropic", "claude-haiku-4-5"
    default_provider = llm_cfg.default_provider or "anthropic"
    default_model = llm_cfg.default_model or "claude-sonnet-4-5"
    models = getattr(llm_cfg, "models", {}) or {}
    providers = getattr(llm_cfg, "providers_per_stage", {}) or {}
    return (
        providers.get(stage, default_provider),
        models.get(stage, default_model),
    )


def resolve_stage_chain(
    config: Any,
    stage: str,
) -> list[tuple[str, str]]:
    """Return the ordered ``[(provider, model), ...]`` fallback chain
    for a stage.

    ``providers_per_stage[stage]`` may be a single provider string
    (single-element chain) or a list of providers (multi-element).
    Missing stages produce a one-element chain from
    ``(default_provider, default_model)``.

    The same model is used for every chain entry (per-provider model
    overrides are a documented Wave 3.2 followup). Agent-CLI providers
    apply their own model_aliases internally, so a list like
    ``["anthropic", "claude_cli"]`` with model ``"claude-sonnet-4-5"``
    works out of the box.

    See ``docs/superpowers/specs/2026-05-14-fallback-chain-design.md``.
    """
    if config is None:
        return [("anthropic", "claude-haiku-4-5")]
    llm_cfg = getattr(config, "llm", None)
    if llm_cfg is None:
        return [("anthropic", "claude-haiku-4-5")]
    default_provider = llm_cfg.default_provider or "anthropic"
    default_model = llm_cfg.default_model or "claude-sonnet-4-5"
    models = getattr(llm_cfg, "models", {}) or {}
    providers = getattr(llm_cfg, "providers_per_stage", {}) or {}

    model = models.get(stage, default_model)
    pinned = providers.get(stage, default_provider)
    if isinstance(pinned, str):
        return [(pinned, model)]
    # list[str]
    return [(p, model) for p in pinned]


def build_cached_messages(
    *,
    system: str | None = None,
    cacheable_context: str | None = None,
    user_message: str,
    provider: str = "anthropic",
) -> list[dict[str, Any]]:
    """Build a message list with Anthropic prompt-caching markers.

    Anthropic's prompt-caching API charges 90% less on cached prefix
    tokens (cache writes cost 1.25×, cache reads cost 0.1× of base).
    A 5-minute TTL covers typical scientific-research session
    cadence. Big wins are anywhere we re-send the same large prefix
    across calls:

      - kb_router: KB context block (descriptions + sampled titles)
        is identical for every ``kb_name="auto"`` query.
      - contextual retrieval: the source document is re-sent for
        every chunk of the same paper.
      - RAG synthesis: large system prompts shared across questions.

    On non-Anthropic providers we collapse into a plain string message
    so call sites don't have to branch.

    Args:
        system: Optional system prompt. Sent as its own ``system``
            message (Anthropic-style) when provider == "anthropic",
            otherwise prepended to user content.
        cacheable_context: The big repeated prefix. Marked with
            ``cache_control={"type": "ephemeral"}`` on Anthropic.
        user_message: The variable per-call portion (question, chunk
            being summarized, etc.). Never marked cacheable.
        provider: Routing key; ``"anthropic"`` enables caching.

    Returns:
        Messages list ready to pass to :meth:`AsyncLLMClient.complete`.
    """
    if provider != "anthropic":
        # OpenAI / DeepSeek / Ollama: plain text concatenation.
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        body_parts: list[str] = []
        if cacheable_context:
            body_parts.append(cacheable_context)
        body_parts.append(user_message)
        msgs.append({"role": "user", "content": "\n\n".join(body_parts)})
        return msgs

    # Anthropic via LiteLLM accepts content as either a string or a
    # list of typed blocks; the ``cache_control`` flag attaches only
    # to a block, so we always emit list form for user content here.
    msgs = []
    if system:
        msgs.append({
            "role": "system",
            "content": [{"type": "text", "text": system}],
        })
    user_content: list[dict[str, Any]] = []
    if cacheable_context:
        user_content.append({
            "type": "text",
            "text": cacheable_context,
            "cache_control": {"type": "ephemeral"},
        })
    user_content.append({"type": "text", "text": user_message})
    msgs.append({"role": "user", "content": user_content})
    return msgs


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str: ...

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        provider: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...


class AsyncLLMClient:
    """
    Async LLM client using LiteLLM.

    Features:
    - Multi-provider support (OpenAI, Anthropic, DeepSeek, Gemini, etc.)
    - Automatic retries with exponential backoff
    - Streaming and non-streaming completion
    - Structured logging of all calls
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._litellm = None
        # Cache one AgentCLIClient instance per provider key (claude_cli,
        # agent_cli, plus any user-defined alias).
        self._agent_clis: dict[str, Any] = {}
        # Disk cache (Wave 2.1). Constructed lazily on first access so
        # callers that disable caching never touch the filesystem.
        self._cache: LLMResponseCache | None = None

    def _get_cache(self) -> LLMResponseCache | None:
        """Lazy-init the disk cache. Returns None when disabled."""
        if not getattr(self.config, "cache_enabled", False):
            return None
        if self._cache is None:
            self._cache = LLMResponseCache(
                path=self.config.cache_path,
                ttl_hours=self.config.cache_ttl_hours,
            )
        return self._cache

    def _get_agent_cli_client(self, provider: str) -> Any:
        """Lazy-init an :class:`AgentCLIClient` for the given provider key.

        ``claude_cli`` is a preset — the legacy
        :func:`ClaudeCLIClient` factory supplies Claude Code's flag
        defaults so the user doesn't have to spell them out in YAML.

        Any other provider key (notably ``agent_cli``, but users can
        define their own) reads every flag from
        :class:`LLMProviderConfig` — see ``llm/agent_cli.py`` for the
        config shape.
        """
        cached = self._agent_clis.get(provider)
        if cached is not None:
            return cached

        cli_cfg = self.config.providers.get(provider)
        if cli_cfg is None:
            raise ValueError(
                f"Provider '{provider}' is not configured. Add it under "
                "llm.providers in your config.yml."
            )

        if provider == "claude_cli":
            # Backwards compat: ClaudeCLIClient still works as a
            # factory that produces a pre-configured AgentCLIClient.
            from perspicacite.llm.claude_cli import ClaudeCLIClient
            kw: dict[str, Any] = {}
            if getattr(cli_cfg, "timeout", None):
                kw["timeout"] = float(cli_cfg.timeout)
            if getattr(cli_cfg, "executable", None):
                kw["executable"] = cli_cfg.executable
            if getattr(cli_cfg, "cwd", None):
                kw["cwd"] = cli_cfg.cwd
            if getattr(cli_cfg, "env_extra", None):
                kw["env_extra"] = dict(cli_cfg.env_extra)
            if getattr(cli_cfg, "usage_input_tokens_path", None):
                kw["usage_input_tokens_path"] = cli_cfg.usage_input_tokens_path
            if getattr(cli_cfg, "usage_output_tokens_path", None):
                kw["usage_output_tokens_path"] = cli_cfg.usage_output_tokens_path
            # F4 (audit 2026-05-15): forward rich result fields.
            if getattr(cli_cfg, "cost_usd_path", None):
                kw["cost_usd_path"] = cli_cfg.cost_usd_path
            if getattr(cli_cfg, "cache_read_tokens_path", None):
                kw["cache_read_tokens_path"] = cli_cfg.cache_read_tokens_path
            if getattr(cli_cfg, "cache_creation_tokens_path", None):
                kw["cache_creation_tokens_path"] = cli_cfg.cache_creation_tokens_path
            client = ClaudeCLIClient(**kw)
        else:
            from perspicacite.llm.agent_cli import AgentCLIClient
            if not getattr(cli_cfg, "executable", None):
                raise ValueError(
                    f"Provider '{provider}' uses the agent_cli path but "
                    "has no `executable` set. Configure "
                    f"llm.providers.{provider}.executable in your YAML, "
                    "or copy one of the config.{claude_code,codex,openclaw,"
                    "hermes}.example.yml presets."
                )
            client = AgentCLIClient(
                executable=cli_cfg.executable,
                provider_label=provider,
                prompt_via=cli_cfg.prompt_via,
                prompt_flag=cli_cfg.prompt_flag,
                system_flag=cli_cfg.system_flag,
                model_flag=cli_cfg.model_flag,
                extra_args=list(cli_cfg.extra_args),
                output_format=cli_cfg.output_format,
                result_json_path=cli_cfg.result_json_path,
                output_file_flag=cli_cfg.output_file_flag,
                usage_input_tokens_path=cli_cfg.usage_input_tokens_path,
                usage_output_tokens_path=cli_cfg.usage_output_tokens_path,
                cost_usd_path=cli_cfg.cost_usd_path,
                cache_read_tokens_path=cli_cfg.cache_read_tokens_path,
                cache_creation_tokens_path=cli_cfg.cache_creation_tokens_path,
                timeout=float(cli_cfg.timeout),
                cwd=cli_cfg.cwd,
                env_extra=dict(cli_cfg.env_extra),
                model_aliases=dict(cli_cfg.model_aliases),
            )
        self._agent_clis[provider] = client
        return client

    def _is_agent_cli_provider(self, provider: str) -> bool:
        """Return True when ``provider`` should route through agent_cli.

        Two cases:
        - The provider key is in our agent-CLI allowlist
          (``claude_cli``, ``agent_cli``).
        - A user-defined provider sets ``executable`` — opt-in for
          custom presets without modifying core code.
        """
        if provider in ("claude_cli", "agent_cli"):
            return True
        cfg = self.config.providers.get(provider)
        return bool(cfg and getattr(cfg, "executable", None))

    def _get_litellm(self) -> Any:
        """Lazy import litellm."""
        if self._litellm is None:
            import litellm

            # Configure litellm
            litellm.set_verbose = False
            litellm.drop_params = True  # Drop unsupported params
            self._litellm = litellm
        return self._litellm

    def _get_provider_config(self, provider: str) -> LLMProviderConfig:
        """Get configuration for a provider."""
        if provider not in self.config.providers:
            raise ValueError(f"Unknown provider: {provider}")
        return self.config.providers[provider]

    def _build_model_string(self, provider: str, model: str) -> str:
        """Build the model string for LiteLLM (e.g., 'anthropic/claude-3-5-sonnet')."""
        # LiteLLM format: provider/model
        # For Minimax, the actual API call uses minimax/{model} format directly
        return f"{provider}/{model}"

    @retry(
        # F1 (audit 2026-05-15): never retry on deterministic-fail errors
        # — auth errors won't suddenly become valid; budget breaches won't
        # heal. Retry every OTHER *Exception*.
        #
        # `isinstance(e, Exception)` is load-bearing: tenacity catches
        # BaseException, so without it the predicate would retry on
        # asyncio.CancelledError (a BaseException, not an Exception). That
        # swallows cancellation from asyncio.wait_for/timeout and makes every
        # caller-side timeout ineffective: all 3 retry attempts run their full
        # HTTP timeout to completion before the (now-stale) deadline is
        # noticed. Letting cancellation propagate is what makes those timeouts
        # actually bound wall-clock.
        retry=retry_if_exception(
            lambda e: isinstance(e, Exception) and not _is_deterministic_fail(e)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _complete_primary(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Internal retry-decorated completion. Call :meth:`complete` instead."""
        if provider is None:
            provider = self.config.default_provider
        if model is None:
            model = self.config.default_model

        stage_label = kwargs.pop("stage", "llm")
        # Optional telemetry sink — RAG modes / MCP adapter pass one in
        # so token + cost events reach external observers. Pop so it's
        # not forwarded to LiteLLM as a request kwarg.
        sink = kwargs.pop("sink", None)

        # ---- disk cache lookup (Wave 2.1) -----------------------------
        # Cache key is computed from the resolved (provider, model)
        # pair plus everything that affects the response. Volatile
        # kwargs (stage, cache, timeout) are filtered inside
        # build_cache_key.
        cache_bypass = kwargs.pop("cache", True) is False
        cache = None if cache_bypass else self._get_cache()
        cache_key: str | None = None
        if cache is not None:
            cache_key = build_cache_key(
                provider=provider, model=model,
                messages=messages, temperature=temperature,
                max_tokens=max_tokens, extra_kwargs=kwargs,
            )
            hit = await cache.get(cache_key)
            if hit is not None:
                logger.info(
                    "llm_cache_hit",
                    stage=stage_label, provider=provider, model=model,
                    age_seconds=int(time.time()) - hit.created_at,
                )
                from perspicacite.provenance.context import get_collector
                _c = get_collector()
                if _c is not None:
                    _c.add_llm_call(
                        stage_label=stage_label,
                        provider=provider,
                        model=model,
                        prompt_messages=messages,
                        response_text=hit.response,
                        prompt_tokens=hit.input_tokens,
                        completion_tokens=hit.output_tokens,
                        latency_ms=hit.latency_ms,
                    )
                # Cache hits cost $0 — still emit telemetry so MCP
                # clients see a sensible token/cost stream.
                _emit_usage_telemetry(
                    sink, provider=provider, model=model,
                    prompt_tokens=int(hit.input_tokens or 0),
                    completion_tokens=int(hit.output_tokens or 0),
                    cost_usd=0.0,
                )
                return hit.response
            logger.debug(
                "llm_cache_miss",
                stage=stage_label, provider=provider, model=model,
            )

        # ---- budget (Wave 2.4) ---------------------------------------
        from perspicacite.llm.budget import get_budget_tracker
        tracker = get_budget_tracker()
        if tracker is not None:
            tracker.check()

        # MCP sampling — first try the connected client's
        # sampling/createMessage protocol when enabled and a ctx is
        # bound. Falls through on capability error.
        if getattr(self.config, "use_mcp_sampling", False):
            from perspicacite.llm.mcp_sampling import try_sample
            sampled = await try_sample(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if sampled is not None:
                logger.info(
                    "llm_completion_via_sampling",
                    stage=stage_label, output_len=len(sampled),
                )
                return sampled

        # Agent CLIs (Claude Code, Codex, OpenClaw, Hermes, ...) take a
        # completely different path — subprocess to a binary, no
        # LiteLLM, uses the user's subscription / local install.
        # Branch here so we don't need to pretend LiteLLM understands them.
        if self._is_agent_cli_provider(provider):
            cli = self._get_agent_cli_client(provider)
            content = await cli.complete(
                messages=messages, model=model, provider=provider,
                temperature=temperature, max_tokens=max_tokens,
                stage=stage_label, **kwargs,
            )
            if cache is not None and cache_key is not None:
                await cache.put(
                    key=cache_key, provider=provider, model=model,
                    response=content, latency_ms=0.0,
                    input_tokens=0, output_tokens=0,
                )
            # TODO: budget — agent-CLI usage plumbing. Emit a zero-cost
            # event so token/cost telemetry stays present on this code
            # path (subscriptions are flat-priced; per-call USD is N/A).
            _emit_usage_telemetry(
                sink, provider=provider, model=model,
                prompt_tokens=0, completion_tokens=0, cost_usd=0.0,
            )
            return content

        provider_config = self._get_provider_config(provider)
        model_str = self._build_model_string(provider, model)

        logger.info(
            "llm_completion_start",
            provider=provider,
            model=model,
            message_count=len(messages),
            temperature=temperature,
        )

        try:
            litellm = self._get_litellm()

            # Three-tier timeout: per-call kwarg > provider config > global config > code constant
            _call_timeout = kwargs.pop("timeout", None)
            if _call_timeout is not None:
                _effective_timeout = float(_call_timeout)
            elif provider_config.timeout is not None:
                _effective_timeout = float(provider_config.timeout)
            else:
                _effective_timeout = float(
                    getattr(self.config, "default_timeout_s", DEFAULT_LLM_TIMEOUT_S)
                )

            # Prepare API call parameters
            completion_kwargs = {
                "model": model_str,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": _effective_timeout,
            }

            # TODO: Minimax implementation needs fixes
            # There are response parsing issues with the Anthropic-compatible API
            # Consider using DeepSeek or other providers as alternative
            # Special handling for Minimax (Anthropic-compatible API via LiteLLM)
            if provider == "minimax":
                import os
                # For Minimax, use standard acompletion with api_base
                # Model format: minimax/MiniMax-M2.7 (or just MiniMax-M2.7 with custom api_base)
                minimax_api_key = os.environ.get("MINIMAX_API_KEY")
                if not minimax_api_key:
                    raise ValueError("MINIMAX_API_KEY environment variable not set")

                t0 = time.monotonic()
                response = await litellm.acompletion(
                    model=f"minimax/{model}",  # Use minimax/MiniMax-M2.7 format
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=_effective_timeout,
                    api_key=minimax_api_key,
                    api_base=provider_config.base_url,
                    **kwargs,
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                # Standard OpenAI-compatible response format
                content = response.choices[0].message.content
                usage = response.get("usage", {})

                logger.info(
                    "llm_completion_success",
                    provider=provider,
                    model=model,
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    content_length=len(content or ""),
                )
                from perspicacite.provenance.context import get_collector
                _c = get_collector()
                if _c is not None:
                    _c.add_llm_call(
                        stage_label=stage_label,
                        provider=provider,
                        model=model,
                        prompt_messages=messages,
                        response_text=content or "",
                        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                        latency_ms=latency_ms,
                    )
                if tracker is not None:
                    tracker.record(
                        provider=provider, model=model,
                        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        output_tokens=int(usage.get("completion_tokens", 0) or 0),
                    )
                if cache is not None and cache_key is not None:
                    await cache.put(
                        key=cache_key, provider=provider, model=model,
                        response=content or "", latency_ms=latency_ms,
                        input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        output_tokens=int(usage.get("completion_tokens", 0) or 0),
                    )
                _pt = int(usage.get("prompt_tokens", 0) or 0)
                _ct = int(usage.get("completion_tokens", 0) or 0)
                _cost = _safe_completion_cost(response, model)
                logger.info(
                    "llm_completion_usage",
                    stage=stage_label, provider=provider, model=model,
                    tokens_in=_pt, tokens_out=_ct, cost_usd=_cost,
                )
                _emit_usage_telemetry(
                    sink, provider=provider, model=model,
                    prompt_tokens=_pt, completion_tokens=_ct, cost_usd=_cost,
                )
                return content

            completion_kwargs.update(kwargs)

            t0 = time.monotonic()
            response = await litellm.acompletion(**completion_kwargs)
            latency_ms = (time.monotonic() - t0) * 1000.0

            content = response.choices[0].message.content
            usage = response.get("usage", {})

            logger.info(
                "llm_completion_success",
                provider=provider,
                model=model,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                content_length=len(content or ""),
            )
            from perspicacite.provenance.context import get_collector
            _c = get_collector()
            if _c is not None:
                _c.add_llm_call(
                    stage_label=stage_label,
                    provider=provider,
                    model=model,
                    prompt_messages=messages,
                    response_text=content or "",
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    latency_ms=latency_ms,
                )
            if tracker is not None:
                tracker.record(
                    provider=provider, model=model,
                    input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    output_tokens=int(usage.get("completion_tokens", 0) or 0),
                )
            if cache is not None and cache_key is not None:
                await cache.put(
                    key=cache_key, provider=provider, model=model,
                    response=content or "", latency_ms=latency_ms,
                    input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    output_tokens=int(usage.get("completion_tokens", 0) or 0),
                )

            _pt = int(usage.get("prompt_tokens", 0) or 0)
            _ct = int(usage.get("completion_tokens", 0) or 0)
            _cost = _safe_completion_cost(response, model)
            logger.info(
                "llm_completion_usage",
                stage=stage_label, provider=provider, model=model,
                tokens_in=_pt, tokens_out=_ct, cost_usd=_cost,
            )
            _emit_usage_telemetry(
                sink, provider=provider, model=model,
                prompt_tokens=_pt, completion_tokens=_ct, cost_usd=_cost,
            )

            return content

        except Exception as e:
            logger.error(
                "llm_completion_error",
                provider=provider,
                model=model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise _maybe_wrap_error(e, provider) from e

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        provider: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion from the LLM.

        Args:
            messages: List of message dicts
            model: Model name
            provider: Provider name. Uses default if None.
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        if provider is None:
            provider = self.config.default_provider

        stage_label = kwargs.pop("stage", "llm")

        provider_config = self._get_provider_config(provider)
        model_str = self._build_model_string(provider, model)

        # Three-tier timeout: per-call kwarg > provider config > global config > code constant
        _call_timeout = kwargs.pop("timeout", None)
        if _call_timeout is not None:
            _effective_timeout = float(_call_timeout)
        elif provider_config.timeout is not None:
            _effective_timeout = float(provider_config.timeout)
        else:
            _effective_timeout = float(
                getattr(self.config, "default_timeout_s", DEFAULT_LLM_TIMEOUT_S)
            )

        logger.info(
            "llm_stream_start",
            provider=provider,
            model=model,
            message_count=len(messages),
        )

        try:
            litellm = self._get_litellm()

            # TODO: Minimax implementation needs fixes
            # Special handling for Minimax
            if provider == "minimax":
                import os
                # Use standard acompletion with api_base for streaming
                minimax_api_key = os.environ.get("MINIMAX_API_KEY")
                if not minimax_api_key:
                    raise ValueError("MINIMAX_API_KEY environment variable not set")

                t0 = time.monotonic()
                stream = await litellm.acompletion(
                    model=f"minimax/{model}",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=_effective_timeout,
                    api_key=minimax_api_key,
                    api_base=provider_config.base_url,
                    stream=True,
                    **kwargs,
                )

                accum: list[str] = []
                async for chunk in stream:
                    piece = chunk.choices[0].delta.content
                    if piece:
                        accum.append(piece)
                        yield piece

                latency_ms = (time.monotonic() - t0) * 1000.0
                logger.info("llm_stream_complete", provider=provider, model=model)
                from perspicacite.provenance.context import get_collector
                _c = get_collector()
                if _c is not None:
                    _c.add_llm_call(
                        stage_label=stage_label,
                        provider=provider,
                        model=model,
                        prompt_messages=messages,
                        response_text="".join(accum),
                        prompt_tokens=0,
                        completion_tokens=0,
                        latency_ms=latency_ms,
                    )
                return

            # Standard OpenAI-compatible streaming. include_usage=True asks
            # the OpenAI-compatible backend to emit a final delta-chunk
            # with token totals (F-23 — without it provenance records
            # prompt_tokens=0 / completion_tokens=0).
            completion_kwargs = {
                "model": model_str,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": _effective_timeout,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            completion_kwargs.update(kwargs)

            t0 = time.monotonic()
            response = await litellm.acompletion(**completion_kwargs)

            accum2: list[str] = []
            stream_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
            async for chunk in response:
                if chunk.choices:
                    piece = chunk.choices[0].delta.content
                    if piece:
                        accum2.append(piece)
                        yield piece
                # Final chunk often carries .usage when include_usage=True.
                usage_obj = getattr(chunk, "usage", None)
                if usage_obj is not None:
                    pt = getattr(usage_obj, "prompt_tokens", None)
                    ct = getattr(usage_obj, "completion_tokens", None)
                    if pt:
                        stream_usage["prompt_tokens"] = int(pt)
                    if ct:
                        stream_usage["completion_tokens"] = int(ct)

            latency_ms = (time.monotonic() - t0) * 1000.0
            logger.info(
                "llm_stream_complete", provider=provider, model=model,
                prompt_tokens=stream_usage["prompt_tokens"],
                completion_tokens=stream_usage["completion_tokens"],
            )
            from perspicacite.provenance.context import get_collector
            _c = get_collector()
            if _c is not None:
                _c.add_llm_call(
                    stage_label=stage_label,
                    provider=provider,
                    model=model,
                    prompt_messages=messages,
                    response_text="".join(accum2),
                    prompt_tokens=stream_usage["prompt_tokens"],
                    completion_tokens=stream_usage["completion_tokens"],
                    latency_ms=latency_ms,
                )

        except Exception as e:
            logger.error(
                "llm_stream_error",
                provider=provider,
                model=model,
                error=str(e),
            )
            raise _maybe_wrap_error(e, provider) from e

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Complete a conversation with the LLM.

        **Free-auto mode** (``llm.free_auto_mode: true`` in config):
        When the caller does not pass an explicit ``model``/``provider``,
        ``free_tier_fallback_models`` is used as the *primary* rotation chain
        via OpenRouter. Models are tried in order; on any error the next model
        is attempted automatically. No credits are needed — a free
        openrouter.ai account (``OPENROUTER_API_KEY``) is sufficient.

        **Normal mode** (``free_auto_mode: false``, default):
        The configured ``default_model`` / ``default_provider`` is tried first
        (with automatic retries on transient errors). If it fails with a
        quota-exceeded, invalid-model-ID, or auth error *and*
        ``free_tier_fallback_models`` is set, those models are tried next.

        Callers that pass an explicit ``model=`` always use that model
        regardless of ``free_auto_mode``.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: Model name. Uses the free chain or ``default_model`` when None.
            provider: Provider name. Uses the chain or ``default_provider`` when None.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Forwarded to the provider (e.g. ``stage``, ``cache``).

        Returns:
            Generated text.

        Raises:
            Exception: All attempted models (primary + fallbacks) failed.
        """
        free_models: list[str] = getattr(
            self.config, "free_tier_fallback_models", []
        ) or []
        free_auto: bool = getattr(self.config, "free_auto_mode", False)

        # ---- free-auto mode: no explicit model → use free chain as primary ----
        # complete_with_chain() calls self.complete(model=X, provider=Y) with
        # explicit values, so the free_auto block will NOT recurse.
        if free_auto and free_models and model is None and provider is None:
            chain = [("openrouter", m) for m in free_models]
            logger.info(
                "llm_free_auto_chain",
                chain_length=len(chain),
                first_model=free_models[0],
            )
            return await self.complete_with_chain(
                messages, chain=chain,
                temperature=temperature, max_tokens=max_tokens, **kwargs,
            )

        # ---- normal path: primary model with retry, then free fallback --------
        try:
            return await self._complete_primary(
                messages, model=model, provider=provider,
                temperature=temperature, max_tokens=max_tokens, **kwargs,
            )
        except Exception as primary_exc:
            if not free_models or not _should_trigger_free_fallback(primary_exc):
                raise

            # Don't cascade into the free chain when the failing model is
            # *already* a free-tier model — complete_with_chain() handles
            # rotation itself and we'd only duplicate retries.
            current_model = model or self.config.default_model
            if (
                current_model.endswith(":free")
                or current_model in free_models
                or current_model == "openrouter/free"
            ):
                raise

            logger.warning(
                "llm_primary_failed_trying_free_fallback",
                primary_model=model or self.config.default_model,
                primary_provider=provider or self.config.default_provider,
                error=str(primary_exc),
                free_tier_count=len(free_models),
            )

            # Pop `cache` kwarg — free-tier fallback responses should not be
            # cached under the primary model's key; pass cache=False to avoid
            # serving a cached free-tier response for the primary model later.
            kwargs_no_cache = {**kwargs, "cache": False}

            last_exc: Exception = primary_exc
            for free_model in free_models:
                try:
                    result = await self._complete_primary(
                        messages,
                        model=free_model,
                        provider="openrouter",
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs_no_cache,
                    )
                    logger.info(
                        "llm_free_fallback_success",
                        free_model=free_model,
                        original_model=model or self.config.default_model,
                    )
                    return result
                except Exception as free_exc:
                    last_exc = free_exc
                    logger.warning(
                        "llm_free_fallback_step_failed",
                        free_model=free_model,
                        error=str(free_exc),
                    )

            # All free-tier models also failed — re-raise the last error.
            # `from None` suppresses the implicit "during handling of
            # primary_exc" chaining; last_exc already carries the real cause.
            raise last_exc from None

    async def complete_with_fallback(
        self,
        messages: list[dict[str, str]],
        primary_model: str,
        primary_provider: str | None = None,
        fallback_model: str | None = None,
        fallback_provider: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Complete with automatic fallback on failure.

        Args:
            messages: List of messages
            primary_model: Primary model to try
            primary_provider: Primary provider
            fallback_model: Fallback model if primary fails
            fallback_provider: Fallback provider
            **kwargs: Additional parameters

        Returns:
            Generated text from primary or fallback
        """
        try:
            return await self.complete(
                messages=messages,
                model=primary_model,
                provider=primary_provider,
                **kwargs,
            )
        except Exception as e:
            logger.warning(
                "llm_primary_failed",
                primary_model=primary_model,
                error=str(e),
                fallback_model=fallback_model,
            )

            if fallback_model is None:
                raise

            return await self.complete(
                messages=messages,
                model=fallback_model,
                provider=fallback_provider,
                **kwargs,
            )

    async def complete_with_chain(
        self,
        messages: list[dict[str, Any]],
        chain: list[tuple[str, str]],
        **kwargs: Any,
    ) -> str:
        """Try each ``(provider, model)`` in order. Returns the first
        success. On :class:`RateLimitError` or other ``Exception`` (but
        not :class:`BudgetExceededError`), logs and tries the next.
        Raises the last exception when all fail.

        See docs/superpowers/specs/2026-05-14-fallback-chain-design.md.
        """
        if not chain:
            raise ValueError("complete_with_chain requires a non-empty chain")

        from perspicacite.llm.budget import BudgetExceededError

        last_exc: Exception | None = None
        for i, (provider, model) in enumerate(chain):
            try:
                return await self.complete(
                    messages=messages,
                    model=model,
                    provider=provider,
                    **kwargs,
                )
            except BudgetExceededError:
                # Switching providers won't help a budget breach.
                raise
            except Exception as e:
                last_exc = e
                logger.warning(
                    "llm_chain_step_failed",
                    attempt=i + 1,
                    chain_length=len(chain),
                    provider=provider,
                    model=model,
                    error=str(e),
                    error_type=type(e).__name__,
                )
        # All steps failed.
        assert last_exc is not None  # chain non-empty → at least one attempt
        raise last_exc
