"""Shared web aggregator search helper for RAG modes."""

import contextlib
from typing import Any

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.rag.web_search")


async def _emit_telemetry(sink: Any, event: dict) -> None:
    """Dispatch one event to a sink, supporting both list-style and
    callback-style sinks. Plain ``list``s only get .append() (sync); a
    ``CallbackTelemetrySink`` gets on_event_async (live notification).
    """
    if sink is None:
        return
    if hasattr(sink, "on_event_async"):
        try:
            await sink.on_event_async(event)
            return
        except Exception:
            return
    # Fallback for plain list (legacy).
    with contextlib.suppress(Exception):
        sink.append(event)


async def run_web_aggregator_search(
    *,
    keyword_query: str,
    context: str | None,
    optimize_enabled: bool | None,
    databases: list[str] | None,
    max_docs: int,
    apis: list[str] | None = None,
    scilex_apis: list[str] | None = None,
    allowed_provider_names: set[str] | None = None,
    app_state: Any,
    telemetry: Any = None,
) -> list[Any]:
    """Run the shared web aggregator search with query optimization.

    Runs the shared query optimizer first (when enabled), then invokes
    the aggregator with the rewritten string. Optimizer failures fall
    back to ``keyword_query`` automatically — the aggregator call always
    happens with a valid query.

    ``scilex_apis`` and ``allowed_provider_names`` may be pre-computed by
    the caller (``_web_fallback_papers``).  When called directly (e.g. from
    tests), they are derived from ``databases`` on the fly.
    """
    # --- resolve provider sets when not pre-computed ---
    _apis = apis or (databases or ["semantic_scholar", "openalex", "pubmed"])
    if scilex_apis is None or allowed_provider_names is None:
        SCILEX_BACKED = {"semantic_scholar", "openalex", "pubmed", "arxiv"}
        PROVIDER_NAMES = {
            "europepmc": "europepmc",
            "core": "core",
            "inspire": "inspire",
            "pubchem": "pubchem",
            "google_scholar": "google_scholar",
            "dblp_sparql": "dblp_sparql",
        }
        _selected = {(d or "").lower() for d in (databases or [])}
        scilex_apis = [d for d in _selected if d in SCILEX_BACKED]
        extra_providers = {PROVIDER_NAMES[d] for d in _selected if d in PROVIDER_NAMES}
        if not scilex_apis and not extra_providers:
            scilex_apis = ["semantic_scholar", "openalex", "pubmed"]
        allowed_provider_names = set(extra_providers)
        if scilex_apis:
            allowed_provider_names.add("scilex")
        # Standalone providers can share names with SciLEx-fanout APIs —
        # notably "pubmed", which is registered as both a standalone
        # PubMed-Entrez provider and a SciLEx API. Picking "pubmed" must
        # keep the standalone alongside (or instead of) SciLEx, especially
        # when SciLEx is disabled. Selections not matching any standalone
        # provider are harmless — the filter loop just won't see them.
        allowed_provider_names |= _selected

    # --- query optimization ---
    # app_state is passed explicitly by callers (threaded from RAGRequest or
    # MinimalAppState). No global fallback needed — callers own injection.

    effective_query = keyword_query
    if app_state is not None and getattr(app_state, "config", None) is not None:
        import perspicacite.search.query_optimizer as _qo_mod
        try:
            opt = await _qo_mod.optimize_query(
                query=keyword_query,
                context=context,
                app_state=app_state,
                optimize_enabled=optimize_enabled,
            )
            effective_query = opt.searched_query
            if opt.applied:
                logger.info(
                    "web_aggregator_query_rewritten",
                    original=keyword_query,
                    rewritten=effective_query,
                )
                await _emit_telemetry(telemetry, {
                    "kind": "query_rephrased",
                    "by": "keyword_optimizer",
                    "original": keyword_query,
                    "rewritten": effective_query,
                })
        except Exception as _opt_exc:
            logger.warning(
                "web_aggregator_optimizer_failed",
                error=str(_opt_exc),
            )
            # Fall through: use keyword_query unchanged.

    # --- aggregator / SciLEx call ---
    config = getattr(app_state, "config", None) if app_state is not None else None
    try:
        if config is not None:
            from perspicacite.search.domain_aggregator import build_aggregator

            aggregator = build_aggregator(config)
            providers_attr = getattr(aggregator, "_providers", [])
            kept_providers = []
            for p in providers_attr:
                name = (getattr(p, "name", "") or type(p).__name__).lower()
                if name in allowed_provider_names:
                    kept_providers.append(p)
            if kept_providers:
                aggregator._providers = kept_providers  # type: ignore[attr-defined]
                # The aggregator's built-in domain classifier filters
                # providers whose ``domains`` don't intersect the
                # classified domain of the query (e.g. EuropePMC is
                # tagged "biomedical" and gets dropped on a "general"
                # query). The user already explicitly picked these
                # providers — bypass the filter so their choice sticks.
                aggregator._select_providers = lambda _domains: list(  # type: ignore[attr-defined]
                    kept_providers
                )
            elif databases:
                # User explicitly picked one or more databases, but none
                # of them resolve to a provider this server actually
                # built — e.g. picked "google_scholar" on a config that
                # omits ``search.enabled_providers: [..., google_scholar]``,
                # or picked "pubmed" when standalone PubMed was skipped
                # due to a missing ``pdf_download.unpaywall_email``.
                # Honour the selection by narrowing to NO providers and
                # surfacing why, instead of silently running unrelated
                # databases (which produced misleading "I picked Scholar,
                # got CORE results" behaviour).
                aggregator._providers = []  # type: ignore[attr-defined]
                aggregator._select_providers = lambda _domains: []  # type: ignore[attr-defined]
                _available_names = sorted(
                    (getattr(p, "name", "") or type(p).__name__).lower()
                    for p in providers_attr
                )
                _requested = sorted(d.lower() for d in databases if d)
                logger.warning(
                    "web_aggregator_selection_unavailable",
                    requested=_requested,
                    available=_available_names,
                )
                await _emit_telemetry(telemetry, {
                    "kind": "selection_unavailable",
                    "message": (
                        f"Selected databases unavailable on this server: "
                        f"{_requested}. None match the providers built by "
                        f"this config (available: {_available_names}). "
                        f"Either pick from the available list or enable "
                        f"the requested providers in config.yml "
                        f"(search.enabled_providers / google_scholar.enabled / "
                        f"pdf_download.unpaywall_email)."
                    ),
                    "requested": _requested,
                    "available": _available_names,
                })
            # ``apis`` here is the SciLEx fan-out list — only the SciLEx
            # provider reads it; standalone providers ignore it and run
            # against their own endpoints regardless.
            _provider_names = [
                (getattr(p, "name", None) or type(p).__name__).lower()
                for p in getattr(aggregator, "_providers", [])
            ]
            if telemetry is not None and _provider_names:
                telemetry.append({
                    "kind": "provider_progress",
                    "phase": "start",
                    "providers": _provider_names,
                    "scilex_apis": list(scilex_apis or _apis),
                    # Surface the actual query string the aggregator is
                    # about to send so the frontend trail can show users
                    # the keywords being searched (not just the count).
                    "searched_query": effective_query,
                })
            web_papers = await aggregator.search(
                query=effective_query,
                max_results=max_docs * 6,
                apis=scilex_apis or _apis,
            )
            logger.info(
                "web_aggregator_search_done",
                providers=_provider_names,
                returned=len(web_papers),
            )
            if telemetry is not None:
                # Per-provider counts from the returned papers' metadata
                # sources. SciLEx-wrapped papers tag both "scilex" and the
                # underlying API; we collapse "scilex" out so counts reflect
                # the real upstream APIs.
                from collections import Counter as _Counter

                _per_source: _Counter = _Counter()
                for _p in web_papers:
                    _ms = (getattr(_p, "metadata", None) or {}).get("sources") or []
                    if isinstance(_ms, list):
                        # Count this paper under EVERY upstream API that
                        # returned it (a multi-DB hit). Previously a stray
                        # ``break`` counted only the first one, undercounting
                        # cross-DB matches in the Retrieval panel.
                        for _s in _ms:
                            _sl = str(_s).lower()
                            if _sl and _sl != "scilex":
                                _per_source[_sl] += 1
                    else:
                        _src_obj = getattr(_p, "source", None)
                        _val = getattr(_src_obj, "value", None) or (
                            str(_src_obj).replace("PaperSource.", "").lower()
                            if _src_obj else ""
                        )
                        if _val and _val != "scilex":
                            _per_source[_val] += 1
                telemetry.append({
                    "kind": "provider_progress",
                    "phase": "done",
                    "total": len(web_papers),
                    "by_provider": dict(_per_source),
                })
        else:
            from perspicacite.search.scilex_adapter import SciLExAdapter

            web_papers = await SciLExAdapter().search(
                query=effective_query, max_results=max_docs * 6, apis=_apis,
            )
    except Exception as e:
        logger.warning("web_aggregator_search_failed", error=str(e))
        return []

    return web_papers
