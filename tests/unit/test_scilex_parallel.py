"""Per-backend fan-out should be concurrent (ThreadPoolExecutor), not
serial. A failure in one backend must not delay or poison the others."""
import asyncio
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

from perspicacite.search.scilex_adapter import SciLExAdapter


class TestParallelFanOut:
    """Verify _collect_all_backends fans out per-backend collection
    concurrently rather than serially."""

    def test_per_backend_collection_runs_concurrently(self):
        """Three slow (0.3s) backends should complete in ~0.3s parallel,
        not ~0.9s serial."""
        adapter = SciLExAdapter()
        adapter._scilex_available = True

        # Stub collector: each run_job_collects call sleeps 0.3s
        collector = MagicMock()

        def slow_collect(api_collect_list):
            time.sleep(0.3)

        collector.run_job_collects.side_effect = slow_collect

        queries_by_api = {
            "SemanticScholar": [{"q": "x"}],
            "OpenAlex": [{"q": "x"}],
            "PubMed": [{"q": "x"}],
        }

        # The helper we'll add wraps the per-backend dispatch loop
        t0 = time.monotonic()
        successful, failed = adapter._collect_all_backends(
            collector=collector,
            queries_by_api=queries_by_api,
            max_results=10,
        )
        elapsed = time.monotonic() - t0

        assert elapsed < 0.7, (
            f"Fan-out appears serial — took {elapsed:.2f}s for 3x0.3s backends. "
            "Expected ~0.3-0.5s parallel."
        )
        assert sorted(successful) == sorted(queries_by_api.keys())
        assert failed == []

    def test_per_backend_failure_does_not_block_or_poison_others(self):
        """If one backend raises, the others still complete and we get
        partial success."""
        adapter = SciLExAdapter()
        adapter._scilex_available = True

        collector = MagicMock()

        def flaky_collect(api_collect_list):
            api = api_collect_list[0]["api"]
            if api == "OpenAlex":
                raise RuntimeError("openalex flaked")

        collector.run_job_collects.side_effect = flaky_collect

        queries_by_api = {
            "SemanticScholar": [{"q": "x"}],
            "OpenAlex": [{"q": "x"}],
            "PubMed": [{"q": "x"}],
        }

        successful, failed = adapter._collect_all_backends(
            collector=collector,
            queries_by_api=queries_by_api,
            max_results=10,
        )

        assert "OpenAlex" in failed
        assert "SemanticScholar" in successful
        assert "PubMed" in successful


class TestCollectDeadlineReturnsPartial:
    """A collection deadline must return the fast backends' results instead
    of letting one slow backend cause the outer provider timeout to discard
    everything (the databases-filter → 0 results bug)."""

    def test_deadline_abandons_slow_backend_keeps_fast(self):
        adapter = SciLExAdapter()
        adapter._scilex_available = True

        collector = MagicMock()

        def collect(api_collect_list):
            api = api_collect_list[0]["api"]
            if api == "SemanticScholar":
                time.sleep(2.0)  # straggler — would blow the budget

        collector.run_job_collects.side_effect = collect

        queries_by_api = {
            "SemanticScholar": [{"q": "x"}],  # slow
            "OpenAlex": [{"q": "x"}],         # fast
            "Arxiv": [{"q": "x"}],            # fast
        }

        report: dict = {}
        t0 = time.monotonic()
        successful, failed = adapter._collect_all_backends(
            collector=collector,
            queries_by_api=queries_by_api,
            max_results=10,
            collect_deadline_s=0.4,
            report=report,
        )
        elapsed = time.monotonic() - t0

        # Returned at the deadline, not after the 2s straggler.
        assert elapsed < 1.5, f"deadline not enforced — took {elapsed:.2f}s"
        # Fast backends completed and are reported.
        assert "OpenAlex" in successful
        assert "Arxiv" in successful
        # Slow backend was abandoned: neither succeeded nor failed.
        assert "SemanticScholar" not in successful
        assert "SemanticScholar" not in failed
        # A structured partial-results warning is recorded for the caller —
        # in the call-local report, never on the shared adapter.
        partial = report["partial_collection"]
        assert partial["kind"] == "partial_results_timeout"
        assert "SemanticScholar" in partial["abandoned"]
        assert adapter._last_partial_collection is None, (
            "worker thread wrote adapter state; it can outlive its own call"
        )

    def test_no_deadline_waits_for_all(self):
        """collect_deadline_s=None keeps legacy behaviour: wait for every
        backend, no partial-collection warning."""
        adapter = SciLExAdapter()
        adapter._scilex_available = True

        collector = MagicMock()

        def collect(api_collect_list):
            time.sleep(0.2)

        collector.run_job_collects.side_effect = collect

        queries_by_api = {
            "OpenAlex": [{"q": "x"}],
            "Arxiv": [{"q": "x"}],
        }

        report: dict = {}
        successful, failed = adapter._collect_all_backends(
            collector=collector,
            queries_by_api=queries_by_api,
            max_results=10,
            collect_deadline_s=None,
            report=report,
        )

        assert sorted(successful) == ["Arxiv", "OpenAlex"]
        assert failed == []
        assert "partial_collection" not in report


class TestStragglerDoesNotDiscardResults:
    """A backend abandoned at the deadline keeps writing into the scratch
    directory. Removing that directory races those writes and raises
    "Directory not empty" — which would discard everything already collected,
    re-triggering the very bug the deadline exists to prevent."""

    def _install_fake_scilex(self, monkeypatch, collector):
        """Register minimal stand-ins for the scilex modules the search imports."""
        aggregate = MagicMock()
        for converter in (
            "OpenAlextoZoteroFormat", "SemanticScholartoZoteroFormat",
            "ArxivtoZoteroFormat", "PubMedtoZoteroFormat",
            "IEEEtoZoteroFormat", "SpringertoZoteroFormat",
            "DBLPtoZoteroFormat",
        ):
            setattr(aggregate, converter, MagicMock())
        aggregate.deduplicate = lambda df: df

        collector_module = MagicMock()
        collector_module.CollectCollection = collector

        monkeypatch.setitem(sys.modules, "scilex", MagicMock())
        monkeypatch.setitem(sys.modules, "scilex.crawlers", MagicMock())
        monkeypatch.setitem(
            sys.modules, "scilex.crawlers.collector_collection", collector_module
        )
        monkeypatch.setitem(sys.modules, "scilex.crawlers.aggregate", aggregate)

    def test_abandoned_backend_writing_during_cleanup_returns_normally(
        self, monkeypatch
    ):
        adapter = SciLExAdapter()
        adapter._scilex_available = True

        stop = threading.Event()
        scratch_dirs: list[str] = []

        fake_collector = MagicMock()
        fake_collector.queryCompositor.return_value = {
            "OpenAlex": [{"q": "x"}],           # finishes inside the deadline
            "SemanticScholar": [{"q": "x"}],    # straggler, abandoned
        }

        def collect(api_collect_list):
            if api_collect_list[0]["api"] != "SemanticScholar":
                return
            root = Path(scratch_dirs[0]) / "straggler"
            attempt = 0
            while not stop.is_set():
                try:
                    written = root / f"batch{attempt}"
                    written.mkdir(parents=True, exist_ok=True)
                    (written / "records.json").write_text("{}")
                except OSError:
                    pass  # scratch dir already removed — expected
                attempt += 1
                time.sleep(0.001)  # exercise the race without flooding the disk

        fake_collector.run_job_collects.side_effect = collect

        def build_collector(main_config, api_config):
            scratch_dirs.append(main_config["output_dir"])
            return fake_collector

        self._install_fake_scilex(monkeypatch, build_collector)

        report: dict = {}
        try:
            result = adapter._scilex_search_sync(
                query="anything",
                max_results=10,
                year_min=None,
                year_max=None,
                apis=["openalex", "semantic_scholar"],
                collect_deadline_s=0.2,
                report=report,
            )
        finally:
            stop.set()

        # The point of the test: cleanup raced a live writer and we still
        # returned instead of throwing away the collected results.
        assert result == []
        assert "SemanticScholar" in report["partial_collection"]["abandoned"]

        # And the scratch directory is reclaimed once the straggler stops,
        # rather than left on disk to grow back.
        scratch = Path(scratch_dirs[0])
        deadline = time.monotonic() + 10
        while scratch.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not scratch.exists(), f"scratch directory leaked: {scratch}"

    def test_returned_backend_lists_are_snapshots(self):
        """Abandoned threads keep appending; the caller's lists must not move."""
        adapter = SciLExAdapter()
        adapter._scilex_available = True

        stop = threading.Event()
        collector = MagicMock()

        def collect(api_collect_list):
            if api_collect_list[0]["api"] != "SemanticScholar":
                return
            stop.wait(timeout=2.0)  # still running when the deadline fires

        collector.run_job_collects.side_effect = collect

        try:
            successful, _failed = adapter._collect_all_backends(
                collector=collector,
                queries_by_api={
                    "OpenAlex": [{"q": "x"}],
                    "SemanticScholar": [{"q": "x"}],
                },
                max_results=10,
                collect_deadline_s=0.2,
            )
            observed = list(successful)
            stop.set()
            time.sleep(0.3)  # let the straggler finish and try to append
            assert successful == observed, (
                "returned list mutated by an abandoned backend thread"
            )
        finally:
            stop.set()


class TestCollectAllBackendsPreservesQueryStructure:
    """Make sure the new helper still builds api_collect_list correctly
    (max_articles_per_query, output_dir, api fields)."""

    def test_each_backend_receives_proper_query_dict(self):
        adapter = SciLExAdapter()
        adapter._scilex_available = True

        collector = MagicMock()
        captured: list[list] = []

        def capture(api_collect_list):
            captured.append(list(api_collect_list))

        collector.run_job_collects.side_effect = capture

        queries_by_api = {
            "SemanticScholar": [{"q": "x"}],
        }

        adapter._collect_all_backends(
            collector=collector,
            queries_by_api=queries_by_api,
            max_results=7,
        )

        assert len(captured) == 1
        items = captured[0]
        assert len(items) == 1
        item = items[0]
        assert item["api"] == "SemanticScholar"
        assert item["query"]["q"] == "x"
        # max_articles_per_query should be 2x max_results per the existing pattern
        assert item["query"]["max_articles_per_query"] == 14


class TestDeadlineIsPerCallNotPerPass:
    """search() retries a title-like query with a normalized title. Both passes
    share one budget: giving each the full deadline would take twice the
    caller's timeout, which cancels the call and discards both passes — the
    failure the deadline exists to prevent."""

    def _adapter_recording_deadlines(self, deadlines: list):
        adapter = SciLExAdapter()
        adapter._scilex_available = True

        async def fake_search_once(*, collect_deadline_s=None, report=None, **kw):
            deadlines.append(collect_deadline_s)
            await asyncio.sleep(0.05)  # a pass that leaves budget for the retry
            return []

        adapter._search_once = fake_search_once
        return adapter

    async def test_retry_gets_only_the_remaining_budget(self):
        deadlines: list = []
        adapter = self._adapter_recording_deadlines(deadlines)

        t0 = time.monotonic()
        await adapter.search(
            query="Deep learning for protein structure prediction: a review",
            collect_deadline_s=0.4,
        )
        elapsed = time.monotonic() - t0

        assert len(deadlines) == 2, "the normalize-retry should have run"
        # The retry gets what is LEFT of the budget, not a fresh copy of it.
        assert deadlines[1] < deadlines[0], (
            f"retry was given a fresh budget instead of the remainder: {deadlines}"
        )
        assert elapsed < 0.4 * 1.5, (
            f"call overran its budget ({elapsed:.2f}s for a 0.4s deadline)"
        )

    async def test_retry_is_skipped_when_the_budget_is_already_spent(self):
        deadlines: list = []
        adapter = SciLExAdapter()
        adapter._scilex_available = True

        async def burn_the_budget(*, collect_deadline_s=None, report=None, **kw):
            deadlines.append(collect_deadline_s)
            await asyncio.sleep(0.25)  # spends the whole deadline
            return []

        adapter._search_once = burn_the_budget

        await adapter.search(query="A title: with a colon", collect_deadline_s=0.2)

        assert len(deadlines) == 1, "retry ran with no budget left"

    async def test_no_deadline_still_retries(self):
        """collect_deadline_s=None keeps the legacy unbounded retry."""
        deadlines: list = []
        adapter = self._adapter_recording_deadlines(deadlines)

        await adapter.search(query="A title: with a colon", collect_deadline_s=None)

        assert deadlines == [None, None]

