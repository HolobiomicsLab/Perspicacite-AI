"""SPARQL templates + namespace constants for the claim graph.

Subplan A: namespace constants, graph-name helpers, and the single
``cito_edges_for_claim`` helper used by the typed-contradiction strategy.
Subplan B will extend this with the five typed-traversal queries
(claims_supporting / claims_disputing / evidence_trace / ...).
"""

from __future__ import annotations

from typing import Any

# ---------- Namespaces (single source of truth across the package) ----------

ASB_NS = "https://asb.holobiomics.org/ns/asb#"
CITO_NS = "http://purl.org/spar/cito/"
PROV_NS = "http://www.w3.org/ns/prov#"
FABIO_NS = "http://purl.org/spar/fabio/"
DOCO_NS = "http://purl.org/spar/doco/"
OA_NS = "http://www.w3.org/ns/oa#"
DCT_NS = "http://purl.org/dc/terms/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
DCAT_NS = "http://www.w3.org/ns/dcat#"
INDICIUM_NS = "https://w3id.org/indicium/"

# Indicium v1.2 figure / script / dataset type IRIs
IRI_FIGURE = f"{DOCO_NS}Figure"
IRI_SCRIPT = f"{FABIO_NS}ComputerProgram"
IRI_DATA_ASSET = f"{DCAT_NS}Dataset"

# Indicium v1.2 figure property IRIs
IRI_FIGURE_ID = f"{ASB_NS}figureId"
IRI_CAPTION = f"{DCT_NS}description"
IRI_FIGURE_TYPE = f"{ASB_NS}figureType"
IRI_SOURCE_DOI = f"{ASB_NS}sourceDoi"

SPARQL_PREFIXES = f"""
PREFIX asb: <{ASB_NS}>
PREFIX cito: <{CITO_NS}>
PREFIX prov: <{PROV_NS}>
PREFIX fabio: <{FABIO_NS}>
PREFIX doco: <{DOCO_NS}>
PREFIX oa: <{OA_NS}>
PREFIX dct: <{DCT_NS}>
PREFIX rdf: <{RDF_NS}>
PREFIX xsd: <{XSD_NS}>
PREFIX dcat: <{DCAT_NS}>
PREFIX indicium: <{INDICIUM_NS}>
"""

# ---------- Type IRIs (frequently referenced by builder + strategies) ----------

IRI_CLAIM = f"{ASB_NS}Claim"
IRI_EVIDENCE = f"{ASB_NS}Evidence"
IRI_TEXT_CHUNK = f"{DOCO_NS}TextChunk"
IRI_RESEARCH_PAPER = f"{FABIO_NS}ResearchPaper"
IRI_RUN_ACTIVITY = f"{PROV_NS}Activity"

IRI_CONTEXT = f"{ASB_NS}context"
IRI_SUBJECT = f"{ASB_NS}subject"
IRI_QUALIFIER = f"{ASB_NS}qualifier"
IRI_RELATION = f"{ASB_NS}relation"
IRI_OBJECT = f"{ASB_NS}object"
IRI_EVIDENCE_PROP = f"{ASB_NS}evidence"
IRI_EVIDENCE_TYPE = f"{ASB_NS}evidenceType"
IRI_DERIVED_FROM_PASSAGE = f"{ASB_NS}derivedFromPassage"
IRI_WAS_DERIVED_FROM = f"{PROV_NS}wasDerivedFrom"
IRI_WAS_GENERATED_BY = f"{PROV_NS}wasGeneratedBy"
IRI_CREATED = f"{DCT_NS}created"
IRI_RDF_TYPE = f"{RDF_NS}type"

# R3 anchor provenance IRIs (project-side asb: predicates + W3C Open Annotation)
IRI_QUOTE_EXACT = f"{ASB_NS}quoteExact"
IRI_ANCHOR_STATUS = f"{ASB_NS}anchorStatus"
IRI_TEXT_QUOTE_SELECTOR = f"{OA_NS}TextQuoteSelector"
IRI_OA_EXACT = f"{OA_NS}exact"
IRI_OA_PREFIX = f"{OA_NS}prefix"
IRI_OA_SUFFIX = f"{OA_NS}suffix"

# Indicium v1.4 ClaimLink property IRIs
IRI_CLAIM_LINK       = f"{INDICIUM_NS}ClaimLink"    # indicium:ClaimLink (v1.4: class_uri removed from schema)
IRI_FROM_CLAIM       = f"{INDICIUM_NS}from_claim"
IRI_TO_CLAIM         = f"{INDICIUM_NS}to_claim"
IRI_LINK_TYPE        = f"{INDICIUM_NS}link_type"
IRI_CLAIM_STATUS     = f"{INDICIUM_NS}claim_status"
IRI_ASSERTED_BY      = f"{PROV_NS}wasAttributedTo"   # slot_uri: prov:wasAttributedTo
IRI_DECISION_CONTEXT = f"{INDICIUM_NS}decision_context"

# ---------- Named graph helpers ----------


def cito_graph_iri(kb_name: str) -> str:
    return f"kb://{kb_name}/graphs/cito"


def runs_graph_iri(kb_name: str) -> str:
    return f"kb://{kb_name}/graphs/runs"


# ---------- Edge readers ----------


def cito_edges_for_claim(store: Any, kb_name: str, claim_iri: str) -> list[dict[str, str]]:
    """Return CiTO edges where `claim_iri` is the subject.

    Each row: ``{"predicate": ..., "object": ..., "confidence": ..., "run": ...}``.
    """
    g = cito_graph_iri(kb_name)
    sparql = (
        SPARQL_PREFIXES
        + f"""
        SELECT ?predicate ?object ?confidence ?run
        WHERE {{
            GRAPH <{g}> {{
                ?meta rdf:subject <{claim_iri}> ;
                      rdf:predicate ?predicate ;
                      rdf:object ?object ;
                      asb:confidence ?confidence ;
                      prov:wasGeneratedBy ?run .
            }}
        }}
    """
    )
    return store.select(sparql)


# ---------- ECO grading helpers ----------

_ECO_TIER_RANK = {
    "data": 0,
    "citation": 1,
    "knowledge": 2,
    "inference": 3,
    "speculation": 4,
}
_ECO_IRI_BY_TIER = {
    "data": "http://purl.obolibrary.org/obo/ECO_0000006",
    "citation": "http://purl.obolibrary.org/obo/ECO_0000033",
    "knowledge": "http://purl.obolibrary.org/obo/ECO_0000302",
    "inference": "http://purl.obolibrary.org/obo/ECO_0000361",
    "speculation": "http://purl.obolibrary.org/obo/ECO_0000034",
}


def _eco_iris_at_or_above(min_tier: str | None) -> list[str]:
    if min_tier is None:
        return []
    rank = _ECO_TIER_RANK.get(min_tier, 0)
    return [iri for tier, iri in _ECO_IRI_BY_TIER.items() if _ECO_TIER_RANK[tier] <= rank]


# ---------- Five typed traversal queries (Phase 2) ----------


def claims_supporting(
    store: Any,
    kb_name: str,
    subject_or_iri: str,
    *,
    min_eco_grade: str | None = None,
) -> list[dict[str, str]]:
    """Return claims whose subject contains the given lemma (or IRI matches).

    Optionally filter by minimum ECO tier (data > citation > knowledge >
    inference > speculation).
    """
    if subject_or_iri.startswith("kb://") or subject_or_iri.startswith("doi:"):
        subject_clause = f"FILTER(?claim = <{subject_or_iri}>)"
    else:
        lit = subject_or_iri.replace('"', '\\"')
        subject_clause = f'FILTER(CONTAINS(LCASE(STR(?subject)), LCASE("{lit}")))'
    eco_filter = ""
    if min_eco_grade:
        iris = _eco_iris_at_or_above(min_eco_grade)
        iri_list = ", ".join(f"<{i}>" for i in iris)
        eco_filter = f"?claim asb:evidenceTypeIri ?eco . FILTER(?eco IN ({iri_list}))"
    sparql = (
        SPARQL_PREFIXES
        + f"""
        SELECT ?claim ?subject ?object ?paper ?eco WHERE {{
            ?claim rdf:type asb:Claim ;
                   asb:subject ?subject ;
                   asb:object ?object ;
                   prov:wasDerivedFrom ?paper .
            OPTIONAL {{ ?claim asb:evidenceTypeIri ?eco }} .
            {eco_filter}
            {subject_clause}
        }}
    """
    )
    return store.select(sparql)


def claims_disputing(store: Any, kb_name: str, target_iri: str) -> list[dict[str, str]]:
    """Return claims that dispute the given claim IRI."""
    g = cito_graph_iri(kb_name)
    sparql = (
        SPARQL_PREFIXES
        + f"""
        SELECT ?from ?confidence
        FROM <{g}>
        WHERE {{
            ?meta rdf:object <{target_iri}> ;
                  rdf:predicate cito:disputes ;
                  rdf:subject ?from ;
                  asb:confidence ?confidence .
        }}
    """
    )
    return store.select(sparql)


def evidence_trace(
    store: Any,
    kb_name: str,
    claim_iri: str,
    *,
    max_depth: int = 3,
) -> list[dict[str, str]]:
    """BFS along cito:supports + cito:qualifies, up to max_depth.

    Returns a list of ``{"claim": iri, "depth": d}`` rows in BFS order.
    Uses repeated SELECT calls in Python to avoid relying on SPARQL property
    paths, which rdflib supports inconsistently across versions.
    """
    g = cito_graph_iri(kb_name)
    visited: set[str] = {claim_iri}
    frontier: list[str] = [claim_iri]
    out: list[dict[str, str]] = [{"claim": claim_iri, "depth": "0"}]
    for depth in range(1, max_depth + 1):
        next_frontier: list[str] = []
        for node in frontier:
            sparql = (
                SPARQL_PREFIXES
                + f"""
                SELECT ?o
                FROM <{g}>
                WHERE {{
                    ?meta rdf:subject <{node}> ;
                          rdf:predicate ?p ;
                          rdf:object ?o .
                    FILTER(?p IN (cito:supports, cito:qualifies))
                }}
            """
            )
            for row in store.select(sparql):
                neighbour = row["o"]
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                next_frontier.append(neighbour)
                out.append({"claim": neighbour, "depth": str(depth)})
        if not next_frontier:
            break
        frontier = next_frontier
    return out


def papers_with_claim_pattern(
    store: Any,
    kb_name: str,
    *,
    subject: str | None = None,
    relation: str | None = None,
    object: str | None = None,
) -> list[dict[str, str]]:
    """Return papers whose claims match a (subject, relation, object) pattern.

    Slot filters use case-insensitive substring match on the literal.
    """
    filters: list[str] = []
    if subject:
        lit = subject.replace('"', '\\"')
        filters.append(f'FILTER(CONTAINS(LCASE(STR(?subject)), LCASE("{lit}")))')
    if relation:
        lit = relation.replace('"', '\\"')
        filters.append(f'FILTER(CONTAINS(LCASE(STR(?relation)), LCASE("{lit}")))')
    if object:
        lit = object.replace('"', '\\"')
        filters.append(f'FILTER(CONTAINS(LCASE(STR(?object)), LCASE("{lit}")))')
    filter_block = "\n            ".join(filters)
    sparql = (
        SPARQL_PREFIXES
        + f"""
        SELECT DISTINCT ?paper ?subject ?relation ?object WHERE {{
            ?claim rdf:type asb:Claim ;
                   prov:wasDerivedFrom ?paper .
            OPTIONAL {{ ?claim asb:subject ?subject }}
            OPTIONAL {{ ?claim asb:relation ?relation }}
            OPTIONAL {{ ?claim asb:object ?object }}
            {filter_block}
        }}
    """
    )
    return store.select(sparql)


def figures_for_claim(
    store: Any,
    kb_name: str,
    claim_iri: str,
) -> list[dict[str, str]]:
    """Return Figure nodes linked to a claim via prov:wasDerivedFrom.

    Each row: {"figure": iri, "figure_id": str, "caption": str,
               "figure_type": str, "source_doi": str}
    """
    sparql = (
        SPARQL_PREFIXES
        + f"""
        PREFIX dcat: <{DCAT_NS}>
        SELECT ?figure ?figure_id ?caption ?figure_type ?source_doi WHERE {{
            <{claim_iri}> prov:wasDerivedFrom ?figure .
            ?figure rdf:type <{IRI_FIGURE}> .
            OPTIONAL {{ ?figure asb:figureId ?figure_id }}
            OPTIONAL {{ ?figure dct:description ?caption }}
            OPTIONAL {{ ?figure asb:figureType ?figure_type }}
            OPTIONAL {{ ?figure asb:sourceDoi ?source_doi }}
        }}
        """
    )
    return store.select(sparql)


def neighbors(
    store: Any,
    kb_name: str,
    claim_iri: str,
    *,
    edge_types: list[str] | None = None,
) -> list[dict[str, str]]:
    """Return CiTO-graph neighbours of a claim (both directions).

    edge_types: e.g. ["supports", "qualifies"]. None = all CiTO edges.
    """
    g = cito_graph_iri(kb_name)
    if edge_types:
        type_iris = ", ".join(f"cito:{t}" for t in edge_types)
        pred_filter = f"FILTER(?p IN ({type_iris}))"
    else:
        pred_filter = ""
    sparql = (
        SPARQL_PREFIXES
        + f"""
        SELECT ?neighbor ?direction ?predicate ?confidence
        FROM <{g}>
        WHERE {{
            {{
                ?meta rdf:subject <{claim_iri}> ;
                      rdf:predicate ?p ;
                      rdf:object ?neighbor ;
                      asb:confidence ?confidence .
                BIND("outgoing" AS ?direction)
                BIND(STR(?p) AS ?predicate)
                {pred_filter}
            }} UNION {{
                ?meta rdf:object <{claim_iri}> ;
                      rdf:predicate ?p ;
                      rdf:subject ?neighbor ;
                      asb:confidence ?confidence .
                BIND("incoming" AS ?direction)
                BIND(STR(?p) AS ?predicate)
                {pred_filter}
            }}
        }}
    """
    )
    return store.select(sparql)


def claim_links_for_claim(
    store,
    kb_name: str,
    claim_iri: str,
) -> list[dict]:
    """Return ClaimLink nodes where claim_iri is from_claim or to_claim.

    Queries the cito graph for all ClaimLink nodes (typed indicium:ClaimLink per
    the Indicium v1.4 schema) where the given claim IRI appears as either
    from_claim (outgoing edge) or to_claim (incoming edge).

    Args:
        store:     Store object with a .select(sparql) method.
        kb_name:   KB name (used to resolve the cito named graph IRI).
        claim_iri: IRI of the claim to query.

    Returns:
        List of dicts: {link_iri, from_claim, to_claim, link_type, direction}.
    """
    g = cito_graph_iri(kb_name)
    sparql = (
        SPARQL_PREFIXES
        + f"""
        SELECT ?link ?from_claim ?to_claim ?link_type ?direction
        WHERE {{
            {{
                GRAPH <{g}> {{
                    ?link rdf:type <{IRI_CLAIM_LINK}> ;
                          indicium:from_claim <{claim_iri}> ;
                          indicium:to_claim   ?to_claim ;
                          indicium:link_type  ?link_type .
                }}
                BIND(<{claim_iri}> AS ?from_claim)
                BIND("outgoing" AS ?direction)
            }} UNION {{
                GRAPH <{g}> {{
                    ?link rdf:type <{IRI_CLAIM_LINK}> ;
                          indicium:to_claim   <{claim_iri}> ;
                          indicium:from_claim ?from_claim ;
                          indicium:link_type  ?link_type .
                }}
                BIND(<{claim_iri}> AS ?to_claim)
                BIND("incoming" AS ?direction)
            }}
        }}
    """
    )
    return store.select(sparql)
