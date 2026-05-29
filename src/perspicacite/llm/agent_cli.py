"""Generic agent-CLI subprocess LLM provider.

Drives any single-shot completion CLI (Claude Code, OpenAI Codex,
OpenClaw, Hermes, opencode, OpenHands, etc.) as a subprocess. Flags
and output-parsing rules come from :class:`LLMProviderConfig` so each
agent only needs a config preset, not a new Python module.

When to choose this over the LiteLLM-backed providers:

- ✅ You have a subscription / local install for an agent CLI and
  want to route Perspicacité's internal LLM calls through it without
  paying API rates on top.
- ✅ Routing, screening, rephrase, contextual-retrieval, synthesis —
  any stage that ``AsyncLLMClient.complete`` powers.
- ❌ Production / unattended use that runs concurrently with your
  interactive agent session (rate limits are shared).
- ❌ Streaming output (this provider returns the whole completion at
  once — most agent CLIs don't surface streaming over their stdout
  interface anyway).
- ❌ Strict per-call ``temperature`` / ``max_tokens`` control (agent
  CLIs typically don't expose these in non-interactive mode).

For Claude Code specifically, the long-term path is MCP sampling
once upstream lands client-side support (see
``docs/superpowers/specs/2026-05-14-claude-code-sampling-integration-design.md``).
This subprocess shim is the bridge until then — and it doubles as
the integration point for every other agent CLI.

Configuration shape (``config.yml``):

.. code:: yaml

    llm:
      default_provider: agent_cli
      default_model:    sonnet         # or whatever the CLI accepts
      providers:
        agent_cli:
          executable: claude
          prompt_via: stdin            # or "arg"
          extra_args:                  # always appended
            - "-p"
            - "--output-format"
            - "json"
            - "--no-session-persistence"
          system_flag: "--append-system-prompt"
          model_flag: "--model"
          output_format: json          # or "text"
          result_json_path: result     # dotted path inside the JSON
          timeout: 180
          model_aliases:               # optional: rewrite user-facing
            claude-sonnet-4-5: sonnet  # names into CLI aliases
            claude-haiku-4-5:  haiku

See ``config/providers/claude-code.yml``, ``config/providers/codex.yml``,
``config/providers/openclaw.yml``, and ``config/providers/hermes.yml`` for
ready-made presets.

**Caveat: rate limits are shared with your interactive agent
session.** A heavy Perspicacité run (contextual retrieval on a
multi-paper ingest) can freeze you out of the agent CLI for hours.
Pair with ``pdf_download.cache_pdfs`` + the existing prompt cache so
re-ingest doesn't burn fresh requests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.llm.agent_cli")


def _flatten_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Reduce a chat-style messages list to ``(system_prompt, user_prompt)``.

    Agent CLIs accept a single prompt, so we join non-system messages
    by role tag. Anthropic-style content blocks
    (``[{"type": "text", "text": "..."}]``) are flattened by
    concatenating their ``text`` fields and dropping ``cache_control``
    markers — agent CLIs don't surface prompt caching.
    """
    def _to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(blk.get("text", "")) if isinstance(blk, dict) else str(blk)
                for blk in content
            )
        return str(content)

    system_parts: list[str] = []
    body_parts: list[str] = []
    for m in messages:
        role = m.get("role")
        text = _to_text(m.get("content", ""))
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            body_parts.append(f"[Assistant prior turn]\n{text}")
        else:  # user (or anything else)
            body_parts.append(text)
    return "\n\n".join(system_parts), "\n\n".join(body_parts)


def _walk_json_path(payload: Any, path: str) -> Any:
    """Walk a dotted JSON path, e.g. ``message.content[0].text``.

    Supports dict keys (``a.b``) and list indices (``a[0]``). Returns
    ``None`` if any segment doesn't resolve.
    """
    if not path:
        return payload
    cursor: Any = payload
    # Tokenise: split on '.' but also peel ``[n]`` off the end of each
    # token. Cheap and good enough for the shapes agent CLIs emit.
    for raw in path.split("."):
        # Handle ``key[i]`` and ``key[i][j]`` forms.
        key, _, rest = raw.partition("[")
        if key:
            if not isinstance(cursor, dict):
                return None
            cursor = cursor.get(key)
            if cursor is None:
                return None
        while rest:
            idx_str, _, rest = rest.partition("]")
            try:
                idx = int(idx_str)
            except ValueError:
                return None
            if not isinstance(cursor, list) or not (0 <= idx < len(cursor)):
                return None
            cursor = cursor[idx]
            # Skip the leading ``[`` of the next bracket group, if any.
            rest = rest.lstrip(".[")
            # Loop ends naturally when ``rest`` is empty.
    return cursor


class AgentCLIClient:
    """LLM client that drives an agent CLI as a subprocess.

    Same async ``complete(messages, model, provider, temperature,
    max_tokens, stage) -> str`` contract as :class:`AsyncLLMClient`,
    so it drops into any call site that takes an LLM client.

    All flag plumbing comes from the config (see module docstring).
    """

    def __init__(
        self,
        *,
        executable: str,
        provider_label: str = "agent_cli",
        prompt_via: str = "stdin",
        prompt_flag: str | None = None,
        system_flag: str | None = None,
        model_flag: str | None = None,
        extra_args: list[str] | None = None,
        output_format: str = "text",
        result_json_path: str | None = None,
        output_file_flag: str | None = None,
        usage_input_tokens_path: str | None = None,
        usage_output_tokens_path: str | None = None,
        cost_usd_path: str | None = None,
        cache_read_tokens_path: str | None = None,
        cache_creation_tokens_path: str | None = None,
        timeout: float = 180.0,
        cwd: str | None = None,
        env_extra: dict[str, str] | None = None,
        model_aliases: dict[str, str] | None = None,
    ):
        self.executable = executable
        self.provider_label = provider_label
        self.prompt_via = prompt_via
        self.prompt_flag = prompt_flag
        self.system_flag = system_flag
        self.model_flag = model_flag
        self.extra_args = list(extra_args or [])
        self.output_format = output_format
        self.result_json_path = result_json_path
        # When set, a tempfile is created per call and passed to the
        # CLI via this flag (e.g. Codex's `--output-last-message`).
        # The result is read from that file instead of stdout — useful
        # when the CLI prints banner / progress output to stdout that
        # we'd otherwise need to scrape.
        self.output_file_flag = output_file_flag
        self.usage_input_tokens_path = usage_input_tokens_path
        self.usage_output_tokens_path = usage_output_tokens_path
        # F4 (audit 2026-05-15): rich result fields.
        self.cost_usd_path = cost_usd_path
        self.cache_read_tokens_path = cache_read_tokens_path
        self.cache_creation_tokens_path = cache_creation_tokens_path
        self.timeout = timeout
        self.cwd = cwd
        self.env_extra = dict(env_extra or {})
        self.model_aliases = dict(model_aliases or {})

    def _resolve_model(self, model: str | None) -> str | None:
        if not model:
            return None
        # Exact alias match wins.
        if model in self.model_aliases:
            return self.model_aliases[model]
        # Fuzzy contains-match — useful for Claude Code where any of
        # ``claude-sonnet-4-5``, ``claude-sonnet-4-5-20250929``, or
        # ``anthropic/claude-sonnet-4-5`` should collapse to ``sonnet``.
        lowered = model.lower()
        for needle, alias in self.model_aliases.items():
            if needle.lower() in lowered:
                return alias
        return model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        provider: str | None = None,
        temperature: float = 0.7,  # accepted for API compat, unused
        max_tokens: int = 4096,    # accepted for API compat, unused
        **kwargs: Any,
    ) -> str:
        stage = kwargs.pop("stage", "llm")
        system, body = _flatten_messages(messages)
        resolved_model = self._resolve_model(model)

        cmd: list[str] = [self.executable, *self.extra_args]
        if resolved_model and self.model_flag:
            cmd.extend([self.model_flag, resolved_model])
        if system and self.system_flag:
            cmd.extend([self.system_flag, system])
        # When output_file_flag is set, the CLI writes its final
        # message to a tempfile instead of (or in addition to)
        # stdout. We allocate the tempfile here and read it after the
        # subprocess exits. Done before prompt_via handling because
        # the file arg should appear in the flag block, not at the
        # tail where the prompt goes for arg-mode CLIs.
        out_path: str | None = None
        if self.output_file_flag:
            import tempfile
            fd, out_path = tempfile.mkstemp(
                prefix=f"{self.provider_label}_", suffix=".out"
            )
            os.close(fd)
            cmd.extend([self.output_file_flag, out_path])
        # When prompt_via == "arg", append the body as an argument.
        if self.prompt_via == "arg":
            if self.prompt_flag:
                cmd.extend([self.prompt_flag, body])
            else:
                cmd.append(body)
        stdin_bytes: bytes | None = (
            body.encode("utf-8") if self.prompt_via == "stdin" else None
        )

        env = os.environ.copy()
        env.update(self.env_extra)
        # Be polite: identify ourselves so the agent's telemetry knows.
        env.setdefault("PERSPICACITE_AGENT_CLI", "1")

        logger.info(
            "agent_cli_call_start",
            provider=self.provider_label,
            executable=self.executable,
            stage=stage,
            model=resolved_model,
            system_len=len(system),
            body_len=len(body),
        )
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"{self.provider_label}: CLI not found at '{self.executable}'. "
                f"Install it or set llm.providers.{self.provider_label}.executable "
                "to the absolute path."
            ) from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin_bytes),
                timeout=self.timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"{self.provider_label}: CLI timed out after {self.timeout}s"
            ) from None
        latency_ms = (time.monotonic() - t0) * 1000.0

        if proc.returncode != 0:
            err_full = (stderr or b"").decode("utf-8", errors="replace")
            out_str = (stdout or b"").decode("utf-8", errors="replace")
            err = err_full[:500]
            if out_path:
                with contextlib.suppress(OSError):
                    os.unlink(out_path)
            # Detect rate-limit signals — raise structured error so the
            # orchestrator / Wave 3.2 fallback chain can react.
            from perspicacite.llm.errors import (
                RateLimitError,
                detect_rate_limit,
                suggested_action,
            )
            hit = detect_rate_limit(err_full) or detect_rate_limit(out_str)
            if hit is not None:
                raise RateLimitError(
                    f"{self.provider_label}: rate limit. "
                    f"{suggested_action(self.provider_label)}",
                    provider=self.provider_label,
                    retry_after_seconds=hit.retry_after_seconds,
                )
            from perspicacite.llm.errors import (
                AuthError,
                detect_auth_error,
            )
            if detect_auth_error(err_full) or detect_auth_error(out_str):
                raise AuthError(
                    f"{self.provider_label}: auth failed. "
                    f"{suggested_action(self.provider_label)}",
                    provider=self.provider_label,
                )
            raise RuntimeError(
                f"{self.provider_label}: CLI exited {proc.returncode}: {err}"
            )

        # Pick the source of truth: tempfile when output_file_flag is
        # set (cleanest — many CLIs print banner/progress to stdout
        # that would otherwise need scraping), stdout otherwise.
        if out_path:
            try:
                with open(out_path, encoding="utf-8", errors="replace") as fh:
                    raw = fh.read().strip()
            finally:
                with contextlib.suppress(OSError):
                    os.unlink(out_path)
        else:
            raw = stdout.decode("utf-8", errors="replace").strip()
        text, in_tokens, out_tokens, details = self._parse_output_full(raw)
        cost_usd = details.get("cost_usd")
        cache_read = details.get("cache_read_tokens", 0)
        cache_create = details.get("cache_creation_tokens", 0)

        logger.info(
            "agent_cli_call_done",
            provider=self.provider_label,
            stage=stage,
            latency_ms=int(latency_ms),
            output_len=len(text),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost_usd,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )
        # Provenance — log even though cost is $0 from the user's
        # perspective, so ``perspicacite report cost`` accounts for
        # these calls.
        try:
            from perspicacite.provenance.context import get_collector
            _c = get_collector()
            if _c is not None:
                _c.add_llm_call(
                    stage_label=stage,
                    provider=self.provider_label,
                    model=resolved_model or "default",
                    prompt_messages=messages,
                    response_text=text,
                    prompt_tokens=in_tokens,
                    completion_tokens=out_tokens,
                    latency_ms=latency_ms,
                )
        except Exception as exc:
            logger.debug("provenance_collector_emit_failed", provider=self.provider_label, error=str(exc))

        # F2 (audit 2026-05-15): budget tracker — push usage even when
        # no provenance collector is active. Prefer the CLI-reported
        # exact cost (F4); fall back to PRICING_TABLE lookup keyed on
        # (provider, model).
        try:
            from perspicacite.llm.budget import get_budget_tracker
            tracker = get_budget_tracker()
            if tracker is not None:
                if cost_usd is not None:
                    tracker.record_cost(
                        provider=self.provider_label,
                        model=resolved_model or "default",
                        cost_usd=cost_usd,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                    )
                else:
                    tracker.record(
                        provider=self.provider_label,
                        model=resolved_model or "default",
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                    )
        except Exception as e:  # never break a call on bookkeeping
            logger.debug("agent_cli_budget_record_failed", error=str(e))

        return text

    def _parse_output(self, raw: str) -> str:
        """Extract assistant text from raw stdout.

        - ``output_format == "text"``: return the raw stdout as-is.
        - ``output_format == "json"``: parse and walk
          ``result_json_path``. Fall back to common keys
          (``result`` / ``message`` / ``text`` / ``content``) and
          finally to the raw stdout if all attempts fail.
        """
        if self.output_format != "json":
            return raw
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw  # CLI emitted plain text despite our flag

        if self.result_json_path:
            picked = _walk_json_path(payload, self.result_json_path)
            if isinstance(picked, str):
                return picked
            if picked is not None:
                return json.dumps(picked)

        # Fallback shapes seen in the wild.
        if isinstance(payload, dict):
            for key in ("result", "message", "text", "content", "output"):
                v = payload.get(key)
                if isinstance(v, str):
                    return v
                if isinstance(v, list):
                    # OpenAI / Anthropic-style content blocks.
                    parts = [
                        b.get("text", "")
                        for b in v
                        if isinstance(b, dict) and "text" in b
                    ]
                    if parts:
                        return "\n".join(parts)
        return raw

    def _parse_output_with_usage(self, raw: str) -> tuple[str, int, int]:
        """Return ``(assistant_text, input_tokens, output_tokens)``.

        Backwards compat: ``_parse_output`` still returns just the
        text. This wider variant is used by :meth:`complete` so the
        provenance row records honest counts.

        Zeros are returned when:
        - ``output_format != "json"`` (no payload to walk).
        - No usage paths configured.
        - The JSON is malformed.
        - A path resolves to a non-int value.
        """
        text, in_t, out_t, _details = self._parse_output_full(raw)
        return text, in_t, out_t

    def _parse_output_full(
        self, raw: str
    ) -> tuple[str, int, int, dict[str, Any]]:
        """F4 (audit 2026-05-15): richer parse that also extracts
        ``total_cost_usd`` and the two Anthropic cache-token paths.

        Returns ``(assistant_text, input_tokens, output_tokens, details)``
        where ``details`` is a dict with optional keys:

        - ``cost_usd``  — float, if ``cost_usd_path`` resolved
        - ``cache_read_tokens`` — int
        - ``cache_creation_tokens`` — int

        Callers that don't care can keep using ``_parse_output_with_usage``.
        """
        text = self._parse_output(raw)
        details: dict[str, Any] = {}

        if self.output_format != "json":
            return text, 0, 0, details

        # Even if no usage paths are configured, we might still have
        # cost_usd / cache paths set — so don't early-return on that
        # alone.
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return text, 0, 0, details

        def _walk_int(path: str | None) -> int:
            if not path:
                return 0
            v = _walk_json_path(payload, path)
            if isinstance(v, bool):
                return 0
            if isinstance(v, int):
                return v
            return 0

        def _walk_float(path: str | None) -> float | None:
            if not path:
                return None
            v = _walk_json_path(payload, path)
            if isinstance(v, bool):
                return None
            if isinstance(v, (int, float)):
                return float(v)
            return None

        in_t = _walk_int(self.usage_input_tokens_path)
        out_t = _walk_int(self.usage_output_tokens_path)
        cost = _walk_float(self.cost_usd_path)
        if cost is not None:
            details["cost_usd"] = cost
        cr = _walk_int(self.cache_read_tokens_path) if self.cache_read_tokens_path else 0
        cc = _walk_int(self.cache_creation_tokens_path) if self.cache_creation_tokens_path else 0
        if cr:
            details["cache_read_tokens"] = cr
        if cc:
            details["cache_creation_tokens"] = cc
        return text, in_t, out_t, details
