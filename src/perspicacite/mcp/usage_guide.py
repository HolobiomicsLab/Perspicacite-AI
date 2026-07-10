"""Authoritative, hand-authored guide to using Perspicacité over MCP.

This module is a curated lookup table consumed by the ``get_usage_guide`` MCP
tool. The ``_TOOL_ENTRIES`` list is authored by hand (NOT derived from
``_TOOL_NAMES``) so the drift test in ``tests/unit/test_usage_guide.py`` is a
meaningful check: when a new MCP tool is registered, the drift test fails until
a human documents it here.
"""

from __future__ import annotations

# One entry per registered MCP tool. Keep each field terse: a one-line
# ``purpose``, a one-line ``when_to_use``, and a short ``key_knobs`` list.
_TOOL_ENTRIES: list[dict] = [
    {
        "name": "get_usage_guide",
        "purpose": "Return this authoritative guide to using Perspicacité over MCP.",
        "when_to_use": "Call FIRST when planning any multi-step research task.",
        "key_knobs": [],
    },
    {
        "name": "suggest_databases",
        "purpose": "Recommend which literature databases to search for a query.",
        "when_to_use": "Before search_literature/generate_report when unsure which databases to pass.",
        "key_knobs": ["hints"],
    },
    {
        "name": "search_literature",
        "purpose": "Live multi-database academic search with enrichment and rerank.",
        "when_to_use": "Find candidate papers for a topic from the open literature.",
        "key_knobs": [
            "databases",
            "max_results",
            "optimize_query",
            "year_min",
            "year_max",
            "min_relevance",
        ],
    },
    {
        "name": "web_search",
        "purpose": "Live academic web search across user-selected databases.",
        "when_to_use": "Quick provider sweep with per-provider telemetry; lighter than search_literature.",
        "key_knobs": ["databases", "max_results", "optimize_query", "enrich"],
    },
    {
        "name": "get_paper_content",
        "purpose": "Fetch full text / structured content / abstract for a DOI.",
        "when_to_use": "Read the actual content of a single paper by DOI.",
        "key_knobs": [],
    },
    {
        "name": "get_paper_references",
        "purpose": "List the reference/citation entries of a paper.",
        "when_to_use": "Explore what a paper cites.",
        "key_knobs": [],
    },
    {
        "name": "fetch_paper_resources",
        "purpose": "Fetch associated resources (PDF, supplementary, etc.) for a paper.",
        "when_to_use": "Collect downloadable artefacts attached to a paper.",
        "key_knobs": [],
    },
    {
        "name": "fetch_supplementary",
        "purpose": "Download supplementary materials for a paper.",
        "when_to_use": "Need supplementary files specifically.",
        "key_knobs": [],
    },
    {
        "name": "list_knowledge_bases",
        "purpose": "List all available knowledge bases (KBs).",
        "when_to_use": "Discover which KBs exist before querying or ingesting.",
        "key_knobs": [],
    },
    {
        "name": "create_knowledge_base",
        "purpose": "Create a new empty knowledge base.",
        "when_to_use": "Start a fresh KB for a topic before adding papers.",
        "key_knobs": [],
    },
    {
        "name": "delete_knowledge_base",
        "purpose": "Delete a knowledge base and its vectors.",
        "when_to_use": "Remove an obsolete KB.",
        "key_knobs": [],
    },
    {
        "name": "search_knowledge_base",
        "purpose": "Semantic search over papers in one or more KBs.",
        "when_to_use": "Retrieve relevant papers already ingested into a KB.",
        "key_knobs": ["kb_names", "k"],
    },
    {
        "name": "search_skill_kb",
        "purpose": "EDAM-IRI pre-filtered semantic search over an ASB-Skill collection KB.",
        "when_to_use": (
            "L2 layer of layered skill routing: called by the collection router after "
            "L1 (router SKILL.md) has identified the domain. Cuts candidate set ~10× "
            "before embedding ranking when EDAM filters are supplied. Returns top-k "
            "skill IRIs alongside filtered chunks."
        ),
        "key_knobs": ["edam_operation", "edam_topics", "top_k"],
    },
    {
        "name": "search_by_passage",
        "purpose": "Retrieve KB passages similar to an arbitrary input text, with source/license records.",
        "when_to_use": "Find citable supporting passages for a given sentence/paragraph.",
        "key_knobs": ["kb_names", "k", "min_score"],
    },
    {
        "name": "get_relevant_passages",
        "purpose": "Keyword-style passage retrieval with optional adaptive re-query.",
        "when_to_use": "Pull passages for a search query; set adaptive on sparse KBs.",
        "key_knobs": ["kb_names", "k", "paper_doi", "adaptive"],
    },
    {
        "name": "extract_parameters_from_passages",
        "purpose": "LLM-extract structured numeric parameters (thresholds, ranges) from passages.",
        "when_to_use": "After retrieving passages, pull out quantitative settings.",
        "key_knobs": ["context", "parameter_families", "model"],
    },
    {
        "name": "extract_failure_modes_from_passages",
        "purpose": "LLM-extract structured failure modes from passages.",
        "when_to_use": "After retrieving passages, enumerate documented failure modes.",
        "key_knobs": ["context", "model"],
    },
    {
        "name": "extract_claims_from_passages",
        "purpose": "Turn retrieved passages into typed, SHACL-validated scientific claims (indicium standard).",
        "when_to_use": "After retrieval, when you need machine-readable claims (5-slot SuperPattern + ECO-typed evidence) rather than prose.",
        "key_knobs": ["passages (required)", "context", "model"],
    },
    {
        "name": "generate_report",
        "purpose": "Run a full RAG report over a KB (basic/advanced/deep_research/agentic/literature_survey/contradiction; \"profound\" is a deprecated alias for deep_research).",
        "when_to_use": "Synthesise an answer with citations from KB content.",
        "key_knobs": [
            "mode",
            "kb_names",
            "max_papers",
            "recency_weight",
            "screen_method",
            "screen_threshold",
            "databases",
            "bm25_weight (hybrid retrieval keyword bias, 0.0–1.0)",
            "vector_weight (hybrid retrieval semantic bias, 0.0–1.0)",
        ],
        "response_fields": (
            "report, sources, papers_used, iteration_count, completion_reason, "
            "diagnostic (cycles_completed/papers_retrieved for deep_research), "
            "message_id, task_id, asb_metadata"
        ),
    },
    {
        "name": "screen_papers",
        "purpose": "Score candidate papers (DOIs/titles/dicts) by relevance to a query.",
        "when_to_use": "Triage a candidate list before ingesting or reporting.",
        "key_knobs": ["method", "threshold", "max_results"],
    },
    {
        "name": "add_papers_to_kb",
        "purpose": "Add papers (from a list/source) to a knowledge base.",
        "when_to_use": "Grow a KB with already-identified papers.",
        "key_knobs": [],
    },
    {
        "name": "add_dois_to_kb",
        "purpose": "Add papers to a KB by DOI, fetching content via the pipeline.",
        "when_to_use": "Ingest specific DOIs into a KB.",
        "key_knobs": [],
    },
    {
        "name": "ensure_kb",
        "purpose": (
            "Idempotently create and ingest a per-paper KB for a DOI, using the ASB "
            "binder slug convention. Returns immediately if the KB already has chunks."
        ),
        "when_to_use": (
            "Before grounding questions against a single paper, to guarantee its KB "
            "exists without re-ingesting it on every call."
        ),
        "key_knobs": ["doi (required)", "mode — reserved, currently unused"],
    },
    {
        "name": "ground_paper",
        "purpose": (
            "Answer a research question against one paper's dedicated KB, ensuring "
            "that KB exists first (via ensure_kb)."
        ),
        "when_to_use": (
            "Ask a question of a specific paper by DOI in one call. Set tier='si' to "
            "prefer evidence from the paper's supplementary information."
        ),
        "key_knobs": ["doi (required)", "question (required)", "tier (paper|si) — default: paper"],
    },
    {
        "name": "ingest_local_documents",
        "purpose": "Ingest local files (PDF/text) into a KB.",
        "when_to_use": "Add documents already on disk to a KB when the filename is a sufficient identifier.",
        "key_knobs": [],
    },
    {
        "name": "add_local_papers_to_kb",
        "purpose": "Ingest local files into a KB with user-provided metadata (title, authors, year, abstract).",
        "when_to_use": "Use instead of ingest_local_documents when you have metadata to attach — proposal PDFs, preprints without DOIs, lab reports. Gives proper titles in search results rather than raw filenames.",
        "key_knobs": [
            "file (required)",
            "title (required)",
            "authors",
            "year",
            "abstract",
            "keywords",
            "doi",
        ],
    },
    {
        "name": "ingest_asb_run",
        "purpose": "Ingest an ASB (Agentic Science Benchmark) workflow-run directory into a KB.",
        "when_to_use": "Make a completed ASB run (cards, skills, tool outputs) searchable. Pass the run directory path and a KB name.",
        "key_knobs": ["run_dir (required)", "kb_name (required)"],
    },
    {
        "name": "ingest_github_repo",
        "purpose": "Ingest a GitHub repository's docs/code into a KB.",
        "when_to_use": "Make a codebase or documentation repo searchable. Chunks markdown, Python docstrings, and notebooks. Use ingest_skill_bundle instead when the repo contains a bundle.yml.",
        "key_knobs": ["url (required)", "kb_name (required)", "include", "exclude"],
    },
    {
        "name": "ingest_skill_bundle",
        "purpose": "Ingest an ASB skill bundle (local path or GitHub URL with bundle.yml) into a KB.",
        "when_to_use": "Load a skill bundle's papers and code for retrieval. Resolves DOIs from the bundle manifest when ingest_linked_papers=True.",
        "key_knobs": ["source (required)", "kb_name", "ingest_linked_papers"],
    },
    {
        "name": "build_kb_from_search",
        "purpose": "Build a KB directly from a literature search.",
        "when_to_use": "One-shot: search a topic and populate a new KB.",
        "key_knobs": ["databases", "max_results"],
    },
    {
        "name": "build_kbs_from_zotero",
        "purpose": "Build KBs from Zotero collections.",
        "when_to_use": "Turn an existing Zotero library into searchable KBs.",
        "key_knobs": [],
    },
    {
        "name": "expand_kb_via_citations",
        "purpose": "Expand a KB by following citation links of its papers.",
        "when_to_use": "Broaden a KB along its citation graph.",
        "key_knobs": [],
    },
    {
        "name": "enrich_kb_from_cite_graph_tool",
        "purpose": "Enrich KB metadata using the citation graph.",
        "when_to_use": "Backfill citation-derived metadata for KB papers.",
        "key_knobs": [],
    },
    {
        "name": "export_kb",
        "purpose": "Export a KB (e.g. to BibTeX/JSON).",
        "when_to_use": "Hand off KB contents to another tool.",
        "key_knobs": [],
    },
    {
        "name": "route_kbs",
        "purpose": "Pick the most relevant KB(s) for a query.",
        "when_to_use": "Decide which KB to query when several exist.",
        "key_knobs": [],
    },
    {
        "name": "build_capsule",
        "purpose": "Build an evidence capsule for a paper/passage set.",
        "when_to_use": "Package retrieved evidence into a citable capsule.",
        "key_knobs": [],
    },
    {
        "name": "build_capsules_for_kb",
        "purpose": "Build evidence capsules for all papers in a KB.",
        "when_to_use": "Bulk-capsule an entire KB.",
        "key_knobs": [],
    },
    {
        "name": "push_to_zotero",
        "purpose": "Push papers to a Zotero library/collection.",
        "when_to_use": "Save results into Zotero.",
        "key_knobs": [],
    },
    {
        "name": "push_notes_to_zotero",
        "purpose": "Attach text notes to existing Zotero items.",
        "when_to_use": (
            "After push_to_zotero or when you have a Zotero item_key/DOI and want to "
            "store a RAG summary, annotation, or drafting note alongside the reference."
        ),
        "key_knobs": [
            "content (required)",
            "item_key or doi (one required per note)",
            "tags",
        ],
    },
    {
        "name": "zotero_list_collections",
        "purpose": "List Zotero collections.",
        "when_to_use": "Discover Zotero collections before ingesting.",
        "key_knobs": [],
    },
    {
        "name": "zotero_get_collection_items",
        "purpose": "List items in a Zotero collection.",
        "when_to_use": "Inspect a Zotero collection's papers.",
        "key_knobs": [],
    },
    {
        "name": "zotero_get_paper_resources",
        "purpose": "Fetch resources for a Zotero paper.",
        "when_to_use": "Get attachments for a specific Zotero item.",
        "key_knobs": [],
    },
    {
        "name": "zotero_ingest_collection_to_kb",
        "purpose": "Ingest a Zotero collection into a KB.",
        "when_to_use": "Load a single Zotero collection into a KB.",
        "key_knobs": [],
    },
    {
        "name": "cancel_task",
        "purpose": "Cancel a running long task by task_id.",
        "when_to_use": "Stop an in-flight report/ingest you no longer need.",
        "key_knobs": ["task_id"],
    },
    {
        "name": "export_astra",
        "purpose": "Project indicium claims to ASTRA Insights for interoperable analysis records.",
        "when_to_use": "After extracting claims, to hand them to an ASTRA-consuming tool.",
        "key_knobs": ["claims"],
    },
    {
        "name": "build_claim_graph",
        "purpose": "Build/refresh the indicium claim graph for a KB (oxigraph-backed).",
        "when_to_use": "Run before first RAGMode.REASONING query or after new papers.",
        "key_knobs": ["kb_name (required)", "refresh", "max_pairs_per_claim", "model"],
    },
    {
        "name": "claim_graph_status",
        "purpose": "Return the per-KB claim-graph manifest + last build summary.",
        "when_to_use": "Check KB claim-graph freshness before reasoning; detects schema drift.",
        "key_knobs": ["kb_name (required)"],
    },
    {
        "name": "query_claim_graph",
        "purpose": (
            "Run a typed traversal query (claims_supporting / disputing / "
            "trace / pattern / neighbors) against a KB's claim graph."
        ),
        "when_to_use": (
            "Direct graph queries when reasoning mode's planner isn't "
            "suitable or for tool composition."
        ),
        "key_knobs": ["kb_name (required)", "query_name (required)", "kwargs"],
    },
    {
        "name": "claim_graph_export",
        "purpose": (
            "Export a KB's indicium claim graph as N-Quads (default), Turtle, JSON-LD, or RO-Crate."
        ),
        "when_to_use": (
            "Archive or hand off a KB's claim graph to an external triple-store, "
            "analysis pipeline, or research-object archive."
        ),
        "key_knobs": ["kb_name (required)", "format (nquads|turtle|jsonld|rocrate) — default: nquads"],
    },
    {
        "name": "get_claim_figures",
        "purpose": (
            "Return Figure nodes (microscopy, plots, diagrams) associated with "
            "a claim IRI in the KB claim graph."
        ),
        "when_to_use": (
            "After build_claim_graph, when you need the visual evidence behind "
            "a specific claim — e.g. to surface supporting figures in a report."
        ),
        "key_knobs": ["kb_name (required)", "claim_iri (required)"],
    },
    {
        "name": "get_claim_links",
        "purpose": (
            "Return ClaimLink nodes (supports, contradicts, refines, etc.) "
            "for a claim IRI in the KB claim graph."
        ),
        "when_to_use": (
            "After build_claim_graph, when you need the full typed-edge context "
            "around a specific claim — e.g. to surface contradicting claims, "
            "argument chains, or methodological refinements in a report."
        ),
        "key_knobs": ["kb_name (required)", "claim_iri (required)"],
    },
]

_CAPABILITIES: list[str] = [
    "Live multi-database literature search with enrichment and rerank.",
    "Personal knowledge bases (KBs): build, ingest (DOIs/local/GitHub/Zotero), search, export.",
    "RAG report generation in modes basic/advanced/deep_research/agentic/literature_survey/contradiction (\"profound\" still accepted as alias).",
    "Passage-level retrieval and citable source/license records.",
    "LLM extraction of numeric parameters and failure modes from passages.",
    "Citation-graph expansion and evidence-capsule building.",
]

_DECISION_RULES: list[str] = [
    "Translate non-English queries to English before searching.",
    "Set optimize_query on for literature search unless you need a verbatim query.",
    "For author searches, set optimize_query=false (or filter by ORCID/OpenAlex "
    "author id): the rewrite is tuned for topical recall and may drop bare "
    "surnames it does not recognise as scientific terms.",
    "Call suggest_databases first when unsure which databases to target.",
    "Pick the tool: search_literature/web_search for discovery; "
    "search_knowledge_base/search_by_passage/get_relevant_passages for KB retrieval; "
    "generate_report to synthesise an answer.",
    "Pick the mode (advanced default; deep_research for depth; contradiction for claim conflicts; \"profound\" is a deprecated alias for deep_research) "
    "and screening (screen_method/screen_threshold) for generate_report.",
    "Always read the {success: true/false} envelope on every response before continuing.",
]

_KNOB_DEFAULTS: dict = {
    "optimize_query": "on for literature search",
    "screen_threshold": 0.0,
    "mode": "advanced",
}


def build_usage_guide() -> dict:
    """Return the authoritative MCP usage guide as a plain dict."""
    return {
        "capabilities": list(_CAPABILITIES),
        "decision_rules": list(_DECISION_RULES),
        "tools": [dict(entry) for entry in _TOOL_ENTRIES],
        "knob_defaults": dict(_KNOB_DEFAULTS),
    }
