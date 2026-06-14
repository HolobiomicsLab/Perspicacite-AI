# KB Grounding Loop — Perspicacité Closed-Loop Architecture

**Status:** v0 specification (2026-06-14) · **Audience:** ASB developers, ASBB release engineers, KB maintainers
**Scope:** Per-paper OpenAI-large KB lifecycle, immutable metadata tracking, single-writer choreography with release loop

---

## Overview

The KB-grounding loop is a **closed-loop system** that ensures ASB builds are grounded against reproducible, versioned knowledge bases. It bridges three subsystems:

1. **Per-paper KB creation** — Perspicacité builds domain-specific KBs (one per input paper, named `asb-paper-<doi-slug>`)
2. **Immutable metadata tracking** — Embedding model, chunking config, and embedding-provider version are recorded as provenance hash
3. **Single-writer choreography** — The production Perspicacité server (:8002, openai-large) is single-writer; ingest + snapshot cycles are serialized via release.yml

This document specifies the lifecycle, the provenance schema, and integration points with ASB + the release train.

---

## 1. Architecture: Three Layers

### 1.1 Layer 1 — Per-Paper KB Build (ASB-side)

**When:** During ASB factory run (Phase 1.3, task 1.2).

**Who:** `ensure_paper_kb()` in `AgenticScienceBuilder/src/agentic_science_builder/perspicacite_kb_gen.py`

**Input:**
- Paper DOI (e.g., `10.1234/example`)
- Local PDF paths (from the research package)
- Perspicacité server endpoint (default `:8002/api`)

**Process:**

1. **Name derivation:** KB name = `asb-paper-` + DOI slug (e.g., `asb-paper-10-1234-example`)
2. **Reachability check:** Call `/api/health` on the server; fail-soft if unreachable (logs a WARNING, continues without grounding)
3. **Idempotency check:** Call `/api/kb/{kb_name}/stats` to see if the KB already has chunks (from a prior run)
   - If chunks exist, skip ingest (the KB is immutable once populated)
   - If no chunks, proceed to ingest
4. **Async ingest:** POST `/api/kb/{kb_name}/local-paths` with `pdf_paths` + `doi` + `title` metadata
   - Returns a `job_id`
5. **Poll to completion:** GET `/api/jobs/{job_id}` every 3 seconds, timeout 240 seconds
   - Terminal states: `done`, `completed`, `success` (return `added_chunks`), `error`, `failed` (return 0)
6. **Record metadata** in the build's `MANIFEST.gen.json`:
   - KB name
   - Chunk count (from the job result or `/api/kb/{kb_name}/stats`)
   - Build timestamp (ISO 8601)
   - Perspicacité server version (from `/api/health` `version` field, if present)

**Output:** Populated KB at `~/.local/share/perspicacite/chroma_db/asb-paper-<slug>/` (on the local Perspicacité server host) with chunks indexed by `text-embedding-3-large`.

**Failure modes:**
- Server unreachable → continue (grounding downgrades gracefully; logged as WARNING in the build; card validation later emits `inferred` flags)
- Ingest times out (>240s) → return 0 chunks, log WARNING
- Job fails → return 0 chunks, log WARNING
- Idempotent: if KB already has chunks, skip and return chunk count

### 1.2 Layer 2 — Immutable Metadata Provenance (Perspicacité-side)

**Where:** `config_openai_large.yml` (the production config pinned in asb-skill-collections + the running instance)

**Immutable metadata for v0:**

```yaml
# File: /Users/nothiasl/git/Perspicacite-AI/config_openai_large.yml
embedding_model: "text-embedding-3-large"
chunk_size: 1000                    # tokens (via tokenCounter)
chunk_overlap: 200                  # tokens
chunking_method: "token"            # not character-based
reranker_model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
use_two_pass: true                  # reranker only on top-k
```

**Provenance hash (sha256):**

Each KB metadata record includes a hash that unambiguously identifies the embedding + chunking configuration:

```json
{
  "kb_name": "asb-paper-10-1234-example",
  "schema_version": "0.1.0",
  "config_hash": "sha256:abc123...",
  "config_provenance": {
    "embedding_model": "text-embedding-3-large",
    "embedding_provider": "openai",
    "chunk_size_tokens": 1000,
    "chunk_overlap_tokens": 200,
    "chunking_method": "token",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "perspicacite_version": "0.5.0+20260614",
    "llm_provider": "openrouter",
    "llm_model": "anthropic/claude-haiku-4-5"
  },
  "created_at": "2026-06-14T10:30:45Z",
  "paper_doi": "10.1234/example",
  "chunk_count": 47,
  "size_bytes": 1_234_567
}
```

**How the hash is computed:**

```python
import json, hashlib

provenance = {
    "embedding_model": "text-embedding-3-large",
    "chunk_size_tokens": 1000,
    "chunk_overlap_tokens": 200,
    "chunking_method": "token",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "perspicacite_version": "0.5.0+20260614",
}
canonical = json.dumps(provenance, sort_keys=True, separators=(',', ':'))
config_hash = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
```

**Storage:**

Perspicacité stores the metadata in `data/perspicacite.db`, table `kb_metadata`:

```sql
CREATE TABLE kb_metadata (
  id INTEGER PRIMARY KEY,
  kb_name TEXT UNIQUE NOT NULL,
  config_hash TEXT NOT NULL,
  embedding_model TEXT,
  chunk_size_tokens INTEGER,
  chunk_overlap_tokens INTEGER,
  reranker_model TEXT,
  perspicacite_version TEXT,
  paper_doi TEXT,
  created_at TEXT,
  chunk_count INTEGER,
  size_bytes INTEGER
);
```

### 1.3 Layer 3 — Single-Writer Choreography (Release Loop)

**Problem:** Chroma is a single-writer system. If multiple processes ingest simultaneously, or if a snapshot happens during ingest, the KB can become corrupt (torn/inconsistent writes).

**Solution:** The Perspicacité server (:8002) is the single writer. All ASB builds and snapshot operations go through the release loop in a serialized manner.

**Workflow:**

```
┌─────────────────────────────────────────────────────────────────┐
│ ASB Factory (Phase 1.3) — per-domain per-paper                 │
│                                                                  │
│  1. Poll if :8002/api/health answers (timeout 5s)              │
│  2. If unreachable: continue (degrade gracefully)              │
│  3. If reachable:                                               │
│       a. Derive KB name from DOI                               │
│       b. Check if KB exists + has chunks (idempotent)          │
│       c. If empty: POST /api/kb/{kb_name}/local-paths          │
│          - Await job completion (poll 240s)                    │
│       d. Record metadata in MANIFEST.gen.json                  │
│                                                                  │
│  4. Continue with card validation / ground-synthesis           │
│     (both use :8002/api endpoints, fail-soft)                  │
│                                                                  │
│  5. Emit MANIFEST.gen.json with:                               │
│     {                                                           │
│       "generation_id": "...",                                  │
│       "timestamp": "2026-06-14T10:30:45Z",                     │
│       "kbs": [                                                 │
│         {                                                       │
│           "kb_name": "asb-paper-10-1234-...",                 │
│           "config_hash": "sha256:abc123...",                   │
│           "chunk_count": 47,                                   │
│           "paper_doi": "10.1234/..."                           │
│         }                                                       │
│       ]                                                         │
│     }                                                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Release Gate (Phase 1.4) — human review + signature             │
│                                                                  │
│  1. Load gate_report.json (strip-verbatim + similarity check)  │
│  2. Manual review: inspect 10–20% of public artifacts for      │
│     similarity to sources + DOI/license correctness            │
│  3. Sign-off: human reviewer writes reviews/GATE_REPORT.md      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Publish to Collections (Phase 1.5) — collection→HF mirror      │
│                                                                  │
│  1. Zip collection/<domain>/v<N> → collections-<domain>-v1.zip │
│  2. Publish benchmark → HF Datasets asb-<domain>-v1             │
│  3. Publish skill bundles (each skill carries skill_kb.json      │
│     with per-skill doc manifest + related papers from :8002)   │
│  4. Zenodo: ONE concept DOI per collection. The release uploads │
│     into the collection's existing deposition as a new VERSION  │
│     (new version DOI under the same concept DOI). The KB        │
│     snapshot tarball rides along as a FILE in that deposition — │
│     it does NOT get its own separate DOI.                       │
│                                                                  │
│  NOTE: Full KB (Chroma) is NOT published to HF                 │
│  (KBs stay private in Perspicacite-ai_kb LFS repo; the          │
│   public snapshot ships only as a file inside the collection's  │
│   Zenodo deposition, governed by the OA-only gate)             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Snapshot (Post-Release) — preserve immutable state              │
│                                                                  │
│  1. Signal ASB factory: "stop ingesting, :8002 is entering      │
│     snapshot mode" (convention: snapshot only after release)    │
│  2. Stop the :8002 server (single-writer guard enforced by      │
│     snapshot.sh: refuses to run while :8002 answers)           │
│  3. rsync Chroma database to Perspicacite-ai_kb/chroma_db/      │
│  4. rsync KB metadata from data/perspicacite.db                │
│  5. Commit + push (LFS) to Perspicacite-ai_kb main              │
│                                                                  │
│  (Subsequent builds re-clone from LFS snapshot for              │
│   reproducibility; or build against a running :8002 that        │
│   is synced from the snapshot)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Immutable KB Metadata Schema

**Table: `KB Provenance Record` (stored in MANIFEST.gen.json)**

```json
{
  "schema_version": "0.1.0",
  "generation_id": "asbb-metabolomics-v1-20260614-<hash>",
  "timestamp": "2026-06-14T10:30:45Z",
  "domain": "metabolomics",
  "collection_concept_doi": "TODO: Zenodo concept DOI for the metabolomics collection (one concept DOI per collection)",
  "collection_version_doi": "TODO: Zenodo version DOI for this specific release (metabolomics-v1)",
  "asb_version": "0.5.0",
  "asb_profile": "profiles/asbb-v0.yaml",
  "perspicacite_version": "0.5.0+20260614",
  "perspicacite_config_ref": "config_openai_large.yml@<sha1>",
  "open_access_ingest_only": true,
  "kbs": [
    {
      "kb_name": "asb-paper-10-1234-example",
      "paper_doi": "10.1234/example",
      "paper_title": "Example Research on Metabolomics",
      "config_hash": "sha256:abc123def456...",
      "config_provenance": {
        "embedding_model": "text-embedding-3-large",
        "embedding_provider": "openai",
        "embedding_provider_version": "2024-12-19",
        "chunk_size_tokens": 1000,
        "chunk_overlap_tokens": 200,
        "chunking_method": "token",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "llm_provider": "openrouter",
        "llm_model": "anthropic/claude-haiku-4-5"
      },
      "chunk_count": 47,
      "size_bytes": 1_234_567,
      "created_at": "2026-06-14T10:30:45Z",
      "ingest_status": "success",
      "ingest_method": "local-paths"
    }
  ],
  "allow_non_oa_ingest": false,
  "release_gate_passed": true,
  "release_gate_report_sha256": "TODO: SHA256 of gate_report.json",
  "notes": "v0 release — all sources OA-verified; KB immutable post-release"
}
```

**Key fields:**

- **`config_hash`**: SHA256 of the embedding + chunking config. Used to validate that all papers in a collection used the same KB setup.
- **`open_access_ingest_only`**: Boolean. In v0, always `true`. Controls whether non-OA papers can be added to the KB post-release.
- **`allow_non_oa_ingest`**: Boolean (inverse of above, for clarity in gate reports). In v0, always `false`.
- **`ingest_status`**: One of `success`, `empty` (0 chunks), `timeout`, `server_unreachable`, `error`. When `success`, downstream grounding is confident. Otherwise, it's degraded.
- **`release_gate_passed`**: Set by human reviewer during Phase 1.4. When `false`, the KB cannot be published.

---

## 3. Integration with ASB Factory

### 3.1 Build-Time Grounding

**Where:** `AgenticScienceBuilder/src/agentic_science_builder/build_manifest.py` + `perspicacite_kb_gen.py`

**When:** Phase 1.3 (card synthesis + validation)

**Steps:**

1. **Before Agent 2b (card synthesis):** Call `ensure_paper_kb()` with the paper's DOI and local PDFs.
   ```python
   from agentic_science_builder.perspicacite_kb_gen import ensure_paper_kb
   
   kb_status = ensure_paper_kb(
       kb_name=f"asb-paper-{doi_slug}",
       pdf_paths=["/path/to/paper.pdf"],
       mcp_url="http://127.0.0.1:8002",  # via config or --perspicacite-mcp
       doi=doi,
       title=paper_title,
       mode="paper",
       poll_timeout=240.0
   )
   # kb_status = {
   #   "status": "success" | "unreachable" | "exists" | "error" | ...
   #   "kb": "asb-paper-...",
   #   "chunks": 47
   # }
   ```

2. **During Agent 2b:** When generating tools + expected_outputs (card synthesis), use the KB for grounding:
   - Query the KB: `GET /api/kb/{kb_name}/search?q=<tool_name>&top_k=5`
   - For each tool, check if it has semantic support in the paper (chunk cosine >0.7)
   - Mark tools as `grounded` (confident) or `inferred` (ungrounded)
   - Store grounding confidence in `card_validation.json`

3. **After Agent 2b:** The `synthesis_grounding.json` sidecar records:
   ```json
   {
     "kb_name": "asb-paper-10-1234-...",
     "config_hash": "sha256:abc123...",
     "timestamp": "2026-06-14T10:30:45Z",
     "grounding_checks": [
       {
         "card_id": "task_1",
         "tool_name": "example_tool",
         "query": "example_tool",
         "top_results": [
           {
             "chunk_id": "chunk_123",
             "text": "...",
             "cosine_similarity": 0.82,
             "chunk_start_page": 5,
             "grounded": true
           }
         ]
       }
     ]
   }
   ```

4. **Record in MANIFEST.gen.json** (emitted at the end of the build):
   ```json
   {
     "generation_id": "asbb-metabolomics-v1-20260614-abc123",
     "kbs": [
       {
         "kb_name": "asb-paper-10-1234-example",
         "config_hash": "sha256:abc123...",
         "chunk_count": 47,
         "ingest_status": "success"
       }
     ]
   }
   ```

### 3.2 Validation-Time Grounding

**Where:** `AgenticScienceBuilder/src/agentic_science_builder/run_eval.py` (if `--card-validation` is used)

**When:** After card synthesis, before releasing

**Steps:**

1. For each card, query the KB again (post-publication, to verify tools are still grounded):
   ```python
   GET /api/kb/{kb_name}/search?q={tool_name}&top_k=3
   ```

2. If the KB returns results with cosine >0.7, mark `GROUNDING_OK`.
3. If cosine <0.5 or KB is unreachable, mark `GROUNDING_INFERRED` (still okay, but advisory).
4. Emit `card_validation.json` per capsule.

---

## 4. Immutable KB Snapshot & Release Loop

### 4.1 Snapshot Choreography

**Location:** `/Users/nothiasl/git/Perspicacite-ai_kb/snapshot.sh`

**When:** After a release tag is cut (Phase 1.5+)

**Single-writer guard:**

```bash
# Refuse if :8002 is alive (prevents torn writes)
if curl -s -m 3 -o /dev/null "http://localhost:8002/api/kb/_any_/chunks?limit=1"; then
  echo "!! Perspicacité server UP — stop it first"
  exit 1
fi
```

**What it snapshots:**

| Item | Path | Purpose |
|------|------|---------|
| Chroma DB | `chroma_db/` | Vector chunks (immutable per release) |
| KB metadata | `data/perspicacite.db` | KB registry + config hashes |
| Paper PDFs | `data/papers/` | Source corpus (for provenance) |
| Capsules | `data/capsules/` | Generated figures + summaries (if exists) |
| Claim graphs | `data/claim_graphs/` | Indicium assertions (if exists) |
| Provenance | `data/provenance/` | Citation/ingest logs |

**Exclusions (regenerable):**

- `github_cache/` (1.3 GB, re-clonable third-party repos)
- `url_cache/`
- `llm_cache.db` (LLM response cache)
- `kb_logs/` (append log, can be reconstructed)
- `cookies.txt` (secrets)

**Commit message:**

```
snapshot 2026-06-14 — 3 collections

Perspicacité KB snapshot post-release:
- metabolomics-v1: 47 papers, 2,341 chunks
- epigenomics-v1: 32 papers, 1,805 chunks
- transcriptomics-v1: 38 papers, 2,104 chunks

config_hash: sha256:abc123...
perspicacite_version: 0.5.0+20260614
```

### 4.2 Release Train Integration

**Location:** `asb-skill-collections/.github/workflows/release.yml`

**Steps (in order):**

1. Extract tag: `metabolomics-v1` → `slug=metabolomics`, `version=1`
2. Run CI gates (pytest)
3. Call `scripts/regen_catalogue.py` (regenerate catalogue.jsonld with versioned KB references)
4. Upload collection to Zenodo (if ZENODO_TOKEN set; fail-soft otherwise)
   - **Topology:** ONE Zenodo concept DOI per collection. Each release creates a **new version** of that single deposition (a new version DOI resolving under the stable concept DOI) — collections are *not* re-minted as brand-new records per release.
   - **KB snapshot as a file, not a DOI:** the OA-only KB snapshot tarball is uploaded as a **file inside the collection's deposition**. It does **not** receive its own separate DOI; it is cited via the collection's version DOI + the file path within that record.
   - **Important:** The upload includes the MANIFEST.gen.json so the KB provenance is preserved in the archive
5. Update CITATION.cff with the minted **version DOI** (and reference the stable concept DOI)
6. Trigger `mirror-to-hf.yml` via workflow_dispatch
7. **Post-release:** Operator runs `Perspicacite-ai_kb/snapshot.sh` manually (or via a follow-up scheduled task); the resulting snapshot tarball is what rides into the collection deposition as a file on the next/amended Zenodo upload

**Why the manual snapshot?**

- Snapshot.sh enforces single-writer invariant (refuses to run while :8002 answers)
- The release.yml runs asynchronously; snapshot should happen only after release is complete + human has verified
- Future automation: hook snapshot into release.yml with a 10-minute delay + verification step

---

## 5. OA-Redaction Policy (v0 Default)

**Default:** `allow_non_oa_ingest = False`

**Behavior:**

1. **Ingest phase:** When `ensure_paper_kb()` is called, check the paper's OA status (via Unpaywall + Crossref)
   ```python
   # Pseudo-code
   is_oa = check_open_access(doi)
   if not is_oa and not allow_non_oa_ingest:
       logger.warning(f"Paper {doi} is not OA; skipping KB ingest")
       return {"status": "skipped", "reason": "non-oa", "kb": None}
   ```

2. **Release gate:** The gate report flags any non-OA papers that were ingested (even if they did get in):
   ```json
   {
     "non_oa_papers": [
       {
         "doi": "10.1234/closed",
         "reason": "behind paywall",
         "action": "will be removed before public release"
       }
     ]
   }
   ```

3. **Public artifact policy:** The released KBs (published to HF) contain only chunks from OA papers.

**Future:** Post-v0, when the gate is proven + legal review is done, this can be relaxed to `allow_non_oa_ingest = True` with stronger similarity checks + verbatim stripping.

---

## 6. Known Issues & Workarounds

### 6.1 Double-Fire Bug (release.yml + mirror-to-hf.yml)

**Issue:** Both `release.yml` and `mirror-to-hf.yml` trigger on `*-v[0-9]*` tags, causing workflows to run twice.

**Status:** Known in grounded reality; do not fix yet (another step handles it).

**Workaround:** Monitor Actions UI; ignore the second fire (it's idempotent).

### 6.2 ZENODO_TOKEN Not Set

**Issue:** `release.yml` step "Upload to Zenodo" fails gracefully but skips minting a DOI.

**Expectation:** In production (after v0), ZENODO_TOKEN must be set in asb-skill-collections secrets.

**v0 Current:** Placeholder DOI in CITATION.cff (TODO: real Zenodo account setup).

### 6.3 Server Unreachable Degrades Gracefully

**Issue:** If `:8002` is offline during ASB build, KB ingest is skipped (status = `unreachable`).

**Expectation:** Card validation marks tools as `inferred` instead of `grounded`.

**Advisory:** This is acceptable for v0 (graceful degradation); downstream gates will catch the lack of grounding.

---

## 7. File Locations & Paths

| Role | Path | Description |
|------|------|-------------|
| **Config** | `/Users/nothiasl/git/Perspicacite-AI/config_openai_large.yml` | Immutable v0 KB config (pinned in releases) |
| **KB Data** | `~/.local/share/perspicacite/chroma_db/` | Live Chroma instance (local to ASB builder) |
| **KB Snapshot** | `/Users/nothiasl/git/Perspicacite-ai_kb/chroma_db/` | LFS-tracked snapshot (immutable per release) |
| **Metadata** | `/Users/nothiasl/git/Perspicacite-ai_kb/data/perspicacite.db` | KB registry + config hashes (LFS) |
| **Snapshot Script** | `/Users/nothiasl/git/Perspicacite-ai_kb/snapshot.sh` | Single-writer snapshot orchestrator |
| **Release Workflow** | `/Users/nothiasl/git/asb-skill-collections/.github/workflows/release.yml` | Collection → Zenodo → HF pipeline |
| **ASB KB Gen** | `/Users/nothiasl/git/AgenticScienceBuilder/src/agentic_science_builder/perspicacite_kb_gen.py` | Per-paper KB builder |
| **ASB Manifest** | `outputs/<run_id>/MANIFEST.gen.json` | Build provenance (includes KB metadata) |

---

## 8. Checklist for v0 Release

**Phase 1 (Factory + Gate):**

- [ ] Metadata schema in MANIFEST.gen.json matches `kb_metadata` table (pg. 2.2)
- [ ] `ensure_paper_kb()` records `config_hash` per KB
- [ ] `synthesis_grounding.json` includes KB name + config hash
- [ ] Release gate validates: all papers in collection used the same `config_hash` (no mixed embeddings)
- [ ] Human reviewer checks MANIFEST.gen.json + gate_report.json side-by-side

**Phase 1.5 (Publish):**

- [ ] `scripts/regen_catalogue.py` includes KB provenance in catalogue.jsonld (version + config_hash)
- [ ] MANIFEST.gen.json is archived to Zenodo (inside the collection zip)
- [ ] Release is uploaded as a **new version of the collection's single concept-DOI deposition** (not a new record)
- [ ] OA-only KB snapshot tarball is included as a **file** in that deposition (no separate DOI)
- [ ] CITATION.cff in collection dir includes the minted **version DOI** (+ stable concept DOI + config_hash reference)

**Phase 1.6 (Snapshot):**

- [ ] Perspicacité server is stopped before `snapshot.sh` runs (single-writer guard enforced)
- [ ] `snapshot.sh` commits with message that includes config_hash
- [ ] Perspicacite-ai_kb/chroma_db has `.gitattributes` entries for LFS (`chroma.sqlite3`)

**Testing:**

- [ ] Reproducibility test: clone Perspicacite-ai_kb snapshot, verify KB queries return same results as live build
- [ ] Build two papers with the same config; verify they generate identical config_hash values
- [ ] Verify MANIFEST.gen.json schema validates against `MANIFEST.schema.json` (TODO: create schema)

---

## 9. Future Extensions (Post-v0)

1. **Multi-LLM variants** (Phase 2.4): New KBs with different reranker models or embedding dimensions → new config_hash values; compare retrieval performance
2. **Semantic admission gate:** Use KB's own retrieval to gate papers (e.g., reject papers with zero semantic overlap to the domain)
3. **KB versioning & rollback:** Use `KBLogWriter` to record per-paper ingest events; enable rollback-to-timestamp for failed ingest batches
4. **Public KB snapshots:** If OA-only gate proves sufficient, ship per-domain KB snapshots alongside benchmarks — as files inside each collection's existing Zenodo deposition (under that collection's single concept DOI), not as separately minted datasets/DOIs
5. **KB integrity checksums:** Compute & store Chroma DB checksum per release so reproducible builds can validate they're using the right snapshot

---

## 10. References

- **Perspicacité config:** `/Users/nothiasl/git/Perspicacite-AI/config_openai_large.yml`
- **Perspicacité KB versioning:** `/Users/nothiasl/git/Perspicacite-AI/docs/versioned-kbs-2026-05-14.md`
- **ASB KB gen:** `/Users/nothiasl/git/AgenticScienceBuilder/src/agentic_science_builder/perspicacite_kb_gen.py`
- **ASB SPEC:** `/Users/nothiasl/git/agenticsciencebuilder_dev/docs/asbb/SPEC.md`
- **ASBB PLAN:** `/Users/nothiasl/git/agenticsciencebuilder_dev/docs/asbb/PLAN.md`
- **indicium:** `/Users/nothiasl/git/indicium` (claim/evidence standard, pinned to **indicium 1.12.0**, co-released with ASB)

---

## Human-Only Input TODOs

The following items require human decision/input and are marked as TODO:

1. **Collection concept DOIs:** Each domain **collection** (`metabolomics`, etc.) needs ONE Zenodo concept DOI; every release is a new version under it (per-release version DOI). The KB snapshot is a file inside that deposition, not a separate DOI.
   - TODO: Create one Zenodo deposition (concept DOI) per collection; record both `collection_concept_doi` and per-release `collection_version_doi` in MANIFEST.gen.json

2. **Real Perspicacité version:** Update the config_openai_large.yml with the actual Perspicacité release version tag when v0.5.0+ is tagged
   - TODO: Pin Perspicacité version to a specific tag (e.g., `v0.5.0+20260614`) in the pinned profile

3. **ZENODO_TOKEN:** Must be added to asb-skill-collections GitHub secrets for release.yml to mint DOIs
   - TODO: Create Zenodo API token + add as `ZENODO_TOKEN` secret

4. **Real ORCID:** The release metadata includes a placeholder ORCID (0000-0002-XXXX-XXXX)
   - TODO: Replace with actual HolobiomicsLab ORCID (if registered) or use a generic lab identifier

5. **Author/CRediT list:** CITATION.cff files need author names + ORCID + CRediT roles
   - TODO: Gather author list from the lab; add to CITATION.cff template

6. **KB snapshot schedule:** Post-release, snapshot.sh should run automatically (not manual)
   - TODO: Add a GitHub Actions scheduled task (or cron hook) to run snapshot.sh after release.yml completes

7. **Perspicacité deployment:** The pinned profile assumes a :8002 instance is always available during ASB builds
   - TODO: Document operational SOP for running the :8002 server as a long-lived service (systemd unit, Docker, etc.)

8. **Config hash provenance:** MANIFEST.gen.json must include the exact SHA-1 of config_openai_large.yml used in the build
   - TODO: Add a step to AsB pipeline to compute `git rev-parse HEAD:config_openai_large.yml` (or file SHA-256) and record it
