"""Tests for Scholar wiring in build_aggregator."""
from __future__ import annotations

from types import SimpleNamespace


def _make_config(enabled_providers=None, scholar_enabled=False):
    from perspicacite.config.schema import GoogleScholarConfig
    return SimpleNamespace(
        search=SimpleNamespace(
            enabled_providers=enabled_providers or [],
            provider_timeout_s=20.0,
            max_results_per_provider=25,
            core_api_key="",
            ads_api_key="",
        ),
        google_scholar=GoogleScholarConfig(enabled=scholar_enabled),
        pdf_download=SimpleNamespace(unpaywall_email=""),
    )


def test_scholar_not_in_aggregator_when_disabled():
    """google_scholar in enabled_providers but google_scholar.enabled=False → excluded."""
    from perspicacite.search.domain_aggregator import build_aggregator

    cfg = _make_config(enabled_providers=["google_scholar"], scholar_enabled=False)
    agg = build_aggregator(cfg)
    names = [getattr(p, "name", "") for p in agg._providers]
    assert "google_scholar" not in names


def test_scholar_in_aggregator_when_enabled():
    """google_scholar in enabled_providers AND google_scholar.enabled=True → included."""
    from perspicacite.search.domain_aggregator import build_aggregator

    cfg = _make_config(enabled_providers=["google_scholar"], scholar_enabled=True)
    agg = build_aggregator(cfg)
    names = [getattr(p, "name", "") for p in agg._providers]
    assert "google_scholar" in names
