# PDF extraction: fast text + optional docling tables/figures

Perspicacité extracts PDF content in **two independent layers**:

| Layer | Engine | Runs | Output | Speed |
|-------|--------|------|--------|-------|
| **Text** (always on) | PyMuPDF (`fitz`) → `pdfplumber` fallback | every PDF ingest | full body text + sections | fast (sub-second) |
| **Tables + figures** (opt-in, advanced) | docling layout model | only when enabled | structured tables as retrievable chunks | slow (CPU-bound, ~minutes/page) |

The layers are decoupled: **text never depends on docling.** If the `[docling]`
extra is not installed, the PDF exceeds the page cap, or docling errors/times
out, you still get the full fitz text — you simply don't get the table chunks.
Enabling docling can only *add* content, never break ingest.

## Why docling is off by default

Docling runs the RT-DETR layout model + TableFormer. On CPU this is roughly
**~45–50 s per page (~10 min for a typical paper)**. On Apple Silicon the GPU
(MPS) path is currently **unusable** — the upstream `transformers` RT-DETRv2
positional embedding hard-codes `float64`, which MPS does not support
(see [huggingface/transformers#28334](https://github.com/huggingface/transformers/issues/28334));
`PYTORCH_ENABLE_MPS_FALLBACK=1` does not help. So docling here is a deliberate,
batch/offline choice, not a hot-path default. A CUDA machine makes it fast
enough for routine use.

## Enabling docling

1. Install the optional extra (one-time, heavy — pulls torch + layout models):

   ```bash
   uv sync --extra docling
   ```

2. In `config.yml` under `knowledge_base:`:

   ```yaml
   docling_extract_tables_figures: true   # default: false
   docling_max_pages: 40                  # PDFs larger than this skip docling (text-only)
   docling_timeout_s: 600                 # per-document wall-clock cap; on timeout, keep text, skip extras
   ```

3. Ingest **local PDF files** (the local-files / dropzone path). Each PDF gets
   fitz text **plus** any tables docling extracts, added as searchable chunks
   tagged `content_type="table"` (caption + page preserved in metadata).

## Guard behaviour (config knobs)

- `docling_extract_tables_figures` (bool, default `false`) — master switch for
  the advanced layer.
- `docling_max_pages` (int, default `40`) — documents with more pages skip
  docling and use text-only fitz (avoids the worst-case multi-minute cost).
- `docling_timeout_s` (int, default `600`) — per-document wall-clock cap. docling
  runs in a worker process; on timeout it is abandoned and ingest falls back to
  the already-extracted fitz text. Every fallback logs one structured
  `docling_fallback` event (`reason=oversized|timeout|error`).

## Scope and current limits

- **Wired for the local-file ingest path** (`integrations/local_docs.py`). The
  DOI/BibTeX download path is text-only for now (adding table chunks there needs
  a `Paper.tables` field — a follow-up).
- **Tables become chunks today; figures are extracted but not yet consumed.**
  Docling figure records are produced (caption + image, dimensions populated)
  and mapped to the existing multimodal record shape, but feeding figure images
  into the answer/vision pipeline is a follow-up.
- **CPU-only in practice** on Apple Silicon (see above). Prefer a CUDA host or a
  remote docling service for large batches.

## Implementation pointers

- Converter + record mapping: `src/perspicacite/pipeline/parsers/docling_pdf.py`
  (`DoclingPDFParser`, `DoclingTable`, `DoclingFigure`,
  `figure_to_multimodal_record`). The converter forces
  `AcceleratorDevice.CPU` and enables `generate_picture_images` + `images_scale=2.0`
  (without picture-image rendering, `PictureItem.get_image()` returns `None` and
  every figure is dropped).
- Backend guard + worker: `src/perspicacite/pipeline/parsers/pdf.py`
  (`_should_run_docling_extras`, `_run_docling_with_timeout`, `_docling_importable`).
- Table → chunk: `src/perspicacite/pipeline/chunking_dispatch.py`
  (`table_records_to_chunks`).
- Config: `src/perspicacite/config/schema.py` (`KnowledgeBaseConfig`).

## Note on full text vs. abstracts

If a knowledge base shows only abstracts, that is a **source** issue, not a
docling one: a Zotero `.bib` carries abstracts only. To get full text, ingest
the actual **PDFs** (local-file path) — the fast fitz layer already returns the
complete body text, no docling required. Enable docling only when you also want
the papers' **tables** as retrievable content.
