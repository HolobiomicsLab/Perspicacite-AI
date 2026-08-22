# Search-to-KB: Building a Knowledge Base from a Literature Search

The `search-to-kb` workflow lets you build a focused knowledge base without a
pre-existing `.bib` file. One command runs a SciLEx multi-database search, filters and
optionally screens the results, downloads full texts, and indexes everything into a new
or existing KB.

---

## Prerequisites

- SciLEx installed: `uv pip install -e ".[scilex]"`
- A `config.yml` with an LLM key (for `--screen llm`) and `pdf_download.unpaywall_email`
- The server does not need to be running — `search-to-kb` is a standalone command

---

## Basic usage

```bash
# Build a new KB from the top 30 hits on a query since 2020
perspicacite -c config.yml search-to-kb \
  --query "nitrogen vacancy diamond magnetometry" \
  --kb diamond_sensors \
  --max-results 30 \
  --min-year 2020
```

If `diamond_sensors` already exists, papers are appended; duplicates are skipped.

---

## Filtering before ingest

Filters apply client-side, before any PDF fetch. They reduce unnecessary network
calls and keep KBs focused:

| Flag | Description |
|------|-------------|
| `--min-year YEAR` | Drop papers published before this year |
| `--max-year YEAR` | Drop papers published after this year |
| `--min-citations N` | Drop papers with fewer than N citations |
| `--require-abstract` | Drop papers without an abstract |
| `--article-type TYPE` | Filter by article type (e.g., `journal-article`) |

Papers without a DOI are filtered out too — the download pipeline is keyed on DOIs,
so a hit without one cannot be fetched. That drop is not silent: it shows up as
`no_doi` in the run's filter reasons, and `--resolve-missing-dois` (below) can
recover most of those hits instead.

```bash
perspicacite -c config.yml search-to-kb \
  --query "LLM literature screening" \
  --kb llm_screen \
  --max-results 50 \
  --min-year 2022 \
  --min-citations 5 \
  --require-abstract
```

---

## Relevance screening

After filtering, an optional screen pass scores each candidate paper's abstract against
the query. Papers below the threshold are dropped before ingest.

```bash
# BM25 screen (free, no LLM calls)
perspicacite -c config.yml search-to-kb \
  --query "metabolomics annotation methods" \
  --kb metabo \
  --max-results 40 \
  --screen bm25 \
  --screen-threshold 0.3

# LLM screen (one Haiku-grade call per paper, more accurate)
perspicacite -c config.yml search-to-kb \
  --query "metabolomics annotation methods" \
  --kb metabo \
  --max-results 40 \
  --screen llm \
  --screen-threshold 0.5
```

The `--screen-threshold` range is 0.0–1.0. A threshold of 0.5 for LLM screening
keeps papers the model rates as clearly relevant.

---

## KB-aware query expansion

When `--kb-aware` is set and the target KB already exists, Perspicacité extracts topic
terms from the KB's description and a sample of its paper titles, then appends them
to the search query. This biases SciLEx toward papers adjacent to what you already
have:

```bash
perspicacite -c config.yml search-to-kb \
  --query "magnetometry" \
  --kb diamond_sensors \
  --kb-aware \
  --max-results 20
```

---

## Multi-variant rephrasing

`--rephrase N` generates N alternate phrasings of the query using one cheap LLM call,
fans them all out across SciLEx, and merges the deduped results. This is useful for
keyword-sensitive databases (DBLP, HAL) where the exact phrasing matters:

```bash
perspicacite -c config.yml search-to-kb \
  --query "metabolite annotation LLM" \
  --kb metabo \
  --rephrase 3 \
  --max-results 10
```

With `--rephrase 3`, this fires 4 queries (original + 3 variants) and merges the
deduplicated results. Combine with `--kb-aware` and `--screen llm` for the most
thorough coverage:

```bash
perspicacite -c config.yml search-to-kb \
  --query "metabolite annotation LLM" \
  --kb metabo \
  --rephrase 3 \
  --kb-aware \
  --screen llm \
  --screen-threshold 0.5 \
  --max-results 10
```

---

## Dry-run mode

See which DOIs would be ingested without actually running the download pipeline:

```bash
perspicacite -c config.yml search-to-kb \
  --query "nitrogen vacancy diamond" \
  --kb diamond_sensors \
  --max-results 30 \
  --min-year 2020 \
  --dry-run
```

Output: a list of DOIs that passed all filters and screens, with their titles and
citation counts.

---

## When a search returns nothing

`searched=0, candidates=0` has two very different causes, and they must not be
confused:

- **The query genuinely matched nothing.** Broaden it, drop a filter, or widen
  the year range.
- **Every database refused the request.** The literature is there; you never got
  to see it.

The second case is now reported explicitly. When a backend answers `429`, the
run logs `scilex_backends_throttled` naming the affected databases, and callers
using `search_with_warnings` (or the `build_kb_from_search` MCP tool) receive a
`rate_limit_blocked` warning carrying the provider list and the longest
`Retry-After` the server asked for:

```json
{
  "kind": "rate_limit_blocked",
  "providers": ["OpenAlex"],
  "retry_after_s": 14917,
  "advice": "One or more databases answered 429 and returned nothing, ..."
}
```

**Never read a throttled run as evidence that a field is empty.** A systematic
survey built on a throttled search will silently under-report its coverage.

Two common causes:

- **OpenAlex meters a daily credit budget** (1000 credits/day on the free tier)
  that resets at midnight UTC. A large DOI ingest earlier in the day can exhaust
  it, after which every search request returns `429` with a multi-hour
  `Retry-After`. See <https://openalex.org/pricing>.
- **Semantic Scholar's public tier throttles hard** without an API key. Set
  `pdf_download.semantic_scholar_api_key` in `config.yml` to lift it.

## Google Scholar as a fallback database

When OpenAlex is out of credit, Google Scholar still works. It is driven by a
headless Chromium, so it is slower (~6s/query) than the REST backends, but it
covers preprints and proceedings the metadata APIs miss:

```bash
perspicacite -c config.yml search-to-kb \
  --query "agent benchmark shared computing cluster" \
  --kb my_kb -d google_scholar --max-results 20 --dry-run
```

It needs the browser extra, which is **not** part of the default install:

```bash
uv pip install -e ".[browser]" --inexact
uv run playwright install chromium
```

Without it the provider logs `google_scholar_playwright_missing` and returns
zero hits — another zero that does not mean "no literature".

Scholar returns many results with no DOI in the link. Three recovery steps run
in order, cheapest first:

1. **From the URL.** doi.org and publisher landing pages give the DOI directly.
   arXiv pages expose none but imply one (`arxiv.org/abs/<id>` →
   `10.48550/arXiv.<id>`). bioRxiv/medRxiv version and view tails
   (`…/10.64898/2025.12.02.691830v1.abstract`) are trimmed, because the DOI
   with the tail attached resolves nowhere.
2. **From the title**, with `--resolve-missing-dois` — see the next section.
3. **Neither works** — aclanthology, OpenReview, proceedings and thesis pages
   for work that was never registered with a DOI. These are reported under
   `no_doi` and skipped. That is the correct outcome: there is nothing to fetch.

A live 2026-08-22 measurement on `-d google_scholar`: 10 hits, 5 already carried
a DOI from step 1, and step 2 resolved 3 of the remaining 5 in 1.9 s — 8 of 10
hits ingestable, against 5 before.

Note that the default database set is `semantic_scholar, openalex, pubmed` —
all SciLEx backends. Providers such as `google_scholar`, `europepmc` and
`core` are only queried when named explicitly with `-d`, even though
`search.enabled_providers` lists them.

## Recovering DOIs from titles (`--resolve-missing-dois`)

Google Scholar, DBLP and other scrape-backed providers return a title, an author
line and a link — but often no DOI. Since the ingest path is keyed on DOIs, those
hits used to be dropped, and a Scholar-only run could report `candidates=0` while
having found perfectly good papers.

`--resolve-missing-dois` inserts a title → DOI lookup between the search and the
filter, for the hits that lack a DOI only:

```bash
perspicacite -c config.yml search-to-kb \
  --query "non-uniform sampling fast 2D NMR" \
  --kb nmr_methods -d google_scholar \
  --resolve-missing-dois --dry-run
```

```
  • searched=10 candidates=8 filtered_out=2
  • DOI backfill: resolved 3/5 attempted, 5 hits had no DOI
  • filter reasons: no_doi=2
```

**Every match is verified.** The lookup walks OpenAlex → Crossref → Semantic
Scholar → arXiv and accepts a candidate only when the author tokens overlap, the
year is within ±1, and the titles pass a Jaccard similarity floor. The optional
`--resolve-doi-browser` tier scrapes Google Scholar with headless Chromium and
additionally confirms each scraped DOI through Crossref. A miss is always
preferred to a loose match: a wrong DOI in a bibliography is worse than a paper
that never got ingested.

**It costs network round-trips**, so it is off by default and bounded:

| Flag | Default | Purpose |
|------|---------|---------|
| `--resolve-missing-dois` | off | Enable the lookup at all |
| `--resolve-doi-budget N` | 25 | Max lookups per run; hits beyond it are reported under `over_budget`, never hidden |
| `--resolve-doi-browser` | off | Add the headless-Chromium Scholar tier (slow; needs the `browser` extra) |

Concurrency is capped at 4, and results — hits *and* misses — are memoised for
the life of the process, so re-running overlapping queries in one server session
doesn't re-pay for the same titles.

On the 2026-08-22 NMR sample the four HTTP tiers resolved 3 of 5 in ~2 s, and
`--resolve-doi-browser` added nothing beyond them for ~1.5 s more. Reach for the
browser tier only when the HTTP tiers keep missing a paper you know exists.

Over MCP, the same behaviour is `resolve_missing_dois=true` and
`resolve_doi_budget` on `build_kb_from_search`. Counts come back in the report's
`doi_backfill` block.

## SciLEx re-ranks, and that hurts fast-moving topics

The SciLEx adapter does not preserve the upstream API's own relevance ordering. After
deduplication it re-sorts the whole pool with SciLEx's composite score
(`aggregate_collect._apply_relevance_ranking`): **keywords 45%, quality 25%, itemtype
20%, citations 10%**. The re-rank was added for a good reason — SciLEx collects
year-by-year, so keeping collection order meant a plain truncation returned only the
earliest year's block — but it has a consequence worth knowing.

On a topic where the entire relevant literature is one or two years old and uncited,
the 35% of the score carried by `quality` + `citations`, plus an `itemtype` term that
favours journal articles over preprints, systematically promotes older, heavily-cited,
loosely-matching work. A 2026-08-22 sweep on LLM-agent/HPC queries produced a block of
2002 grid-computing papers through SciLEx, while the *same query against the Semantic
Scholar API directly* returned on-topic 2025–2026 work in its first three hits.

**For a recent topic, query the API directly and take the top-ranked hits**, or treat
the SciLEx result as a recall-oriented pool to be screened (`--screen llm`) rather than
a ranked shortlist. This is a property of the ranking design, not a throttling or key
problem — it happens with a valid Semantic Scholar key and HTTP 200 responses.

Databases needing paid keys (`ieee`, `springer`, `elsevier`) are `enabled: false`
in `config.example.yml`. Passing `-d ieee` while it is disabled is a silent
no-op — the flag is accepted and the backend is never queried.

---

## Via MCP

The same workflow is available as the `build_kb_from_search` MCP tool:

```python
await build_kb_from_search(
    query="LLM literature screening accuracy",
    kb_name="llm_screening",
    max_results=20,
    min_year=2023,
    screen_method="llm",
    screen_threshold=0.5,
)
# → {"added_papers": 14, "added_chunks": 142, "skipped_duplicate": 3, ...}
```

---

## Related topics

- [guides/expand-via-citations.md](expand-via-citations.md) — grow the KB further
  by following citation links from the papers you just ingested
- [guides/ingest-bibtex.md](ingest-bibtex.md) — alternative import from a `.bib`
- [concepts/knowledge-bases.md](../concepts/knowledge-bases.md) — KB storage internals
- [reference/cli.md](../reference/cli.md) — all `search-to-kb` flags
