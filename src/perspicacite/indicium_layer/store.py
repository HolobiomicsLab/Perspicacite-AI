"""ClaimGraphStore — pyoxigraph (production) / rdflib (tests) façade.

All triples flow through a single small surface:

    store.add(s, p, o, *, graph=None)              # plain triple/quad
    store.add_edge_with_confidence(s, p, o, ...)   # reified edge + metadata
    store.select(sparql)                           # SPARQL SELECT
    store.contains_iri(iri)                        # cheap "any triple with this s?"
    store.close()

Confidence-bearing edges use classic RDF reification (rdf:subject / predicate
/ object + asb:confidence + prov:wasGeneratedBy) rather than RDF-star — this
works identically across both backends. The spec's RDF-star aspiration is a
follow-up; the current shape is forward-compatible (the asserted edge is
still present alongside the reification node).
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any
from typing import Literal as LiteralType

from rdflib import Dataset, URIRef
from rdflib import Literal as RdflibLiteral

# rdflib ≥ 6.0 deprecates ConjunctiveGraph in favour of Dataset.
# Dataset is a drop-in for our usage (add/query named graphs).
ConjunctiveGraph = Dataset  # alias for backward compat in type hints below

_RDF_SUBJECT = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject")
_RDF_PREDICATE = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate")
_RDF_OBJECT = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#object")
_ASB_CONFIDENCE = URIRef("https://asb.holobiomics.org/ns/asb#confidence")
_PROV_WAS_GENERATED_BY = URIRef("http://www.w3.org/ns/prov#wasGeneratedBy")
_XSD_DECIMAL = "http://www.w3.org/2001/XMLSchema#decimal"

LiteralTuple = tuple[LiteralType["literal"], Any, str | None]


def _edge_meta_iri(kb_name: str, s: str, p: str, o: str) -> str:
    digest = hashlib.sha256(f"{s}|{p}|{o}".encode()).hexdigest()[:16]
    return f"kb://{kb_name}/edge-meta/{digest}"


class ClaimGraphStore:
    """Claim graph store façade. Backends: ``oxigraph`` | ``memory``."""

    def __init__(
        self,
        kb_name: str,
        *,
        data_dir: pathlib.Path | None = None,
        backend: str = "auto",
    ) -> None:
        self.kb_name = kb_name
        if backend == "auto":
            backend = "oxigraph" if data_dir is not None else "memory"
        self._backend = backend
        if backend == "memory":
            self._g: ConjunctiveGraph | None = ConjunctiveGraph()
            self._oxistore = None
        elif backend == "oxigraph":
            self._g = None
            self._oxistore = _open_oxigraph(data_dir, kb_name)
        else:
            raise ValueError(f"unknown backend: {backend}")

    # ------------------------------------------------------------------ public

    def add(
        self,
        s: str,
        p: str,
        o: str | LiteralTuple,
        *,
        graph: str | None = None,
    ) -> None:
        if self._backend == "memory":
            self._add_rdflib(s, p, o, graph)
        else:
            self._add_oxi(s, p, o, graph)

    def add_edge_with_confidence(
        self,
        s: str,
        p: str,
        o: str,
        *,
        confidence: float,
        run_iri: str,
        graph: str,
    ) -> None:
        """Assert (s, p, o) in `graph` and a reification node carrying confidence."""
        self.add(s, p, o, graph=graph)
        meta_iri = _edge_meta_iri(self.kb_name, s, p, o)
        self.add(meta_iri, str(_RDF_SUBJECT), s, graph=graph)
        self.add(meta_iri, str(_RDF_PREDICATE), p, graph=graph)
        self.add(meta_iri, str(_RDF_OBJECT), o, graph=graph)
        self.add(
            meta_iri,
            str(_ASB_CONFIDENCE),
            ("literal", f"{confidence:.4f}", _XSD_DECIMAL),
            graph=graph,
        )
        self.add(meta_iri, str(_PROV_WAS_GENERATED_BY), run_iri, graph=graph)

    def select(self, sparql: str) -> list[dict[str, str]]:
        if self._backend == "memory":
            return self._select_rdflib(sparql)
        return self._select_oxi(sparql)

    def contains_iri(self, iri: str) -> bool:
        if self._backend == "memory":
            assert self._g is not None
            return any(self._g.quads((URIRef(iri), None, None, None)))
        # oxigraph: cheap ASK
        assert self._oxistore is not None  # guaranteed by __init__ when backend == "oxigraph"
        ask = f"ASK {{ {{ <{iri}> ?p ?o }} UNION {{ GRAPH ?g {{ <{iri}> ?p ?o }} }} }}"
        return bool(self._oxistore.query(ask))

    def close(self) -> None:
        # rdflib has no close; oxigraph store flushes on drop
        self._g = None
        self._oxistore = None

    # ------------------------------------------------------------------ rdflib

    def _add_rdflib(self, s: str, p: str, o: str | LiteralTuple, graph: str | None) -> None:
        assert self._g is not None
        triple = (URIRef(s), URIRef(p), _rdflib_obj(o))
        if graph is None:
            # Add to the default graph (Dataset.add delegates to the default graph)
            self._g.add(triple)
        else:
            self._g.get_context(URIRef(graph)).add(triple)

    def _select_rdflib(self, sparql: str) -> list[dict[str, str]]:
        assert self._g is not None
        results = self._g.query(sparql)
        out: list[dict[str, str]] = []
        for row in results:
            d: dict[str, str] = {}
            for var, val in row.asdict().items():  # type: ignore[union-attr]  # pyoxigraph ResultRow always has asdict(); union includes non-row types due to broad query return stub
                if val is None:
                    continue
                d[str(var)] = str(val.toPython()) if isinstance(val, RdflibLiteral) else str(val)
            out.append(d)
        return out

    # ------------------------------------------------------------------ oxigraph

    def _add_oxi(self, s: str, p: str, o: str | LiteralTuple, graph: str | None) -> None:
        import pyoxigraph as oxi

        assert self._oxistore is not None  # guaranteed by __init__ when backend == "oxigraph"
        subj = oxi.NamedNode(s)
        pred = oxi.NamedNode(p)
        obj: Any  # NamedNode or Literal depending on branch; both are valid RDF term types
        if isinstance(o, tuple):
            _, val, dt = o
            obj = oxi.Literal(
                str(val),
                datatype=oxi.NamedNode(dt) if dt else None,
            )
        else:
            obj = oxi.NamedNode(o)
        ctx = oxi.NamedNode(graph) if graph else oxi.DefaultGraph()
        self._oxistore.add(oxi.Quad(subj, pred, obj, ctx))

    def _select_oxi(self, sparql: str) -> list[dict[str, str]]:
        assert self._oxistore is not None  # guaranteed by __init__ when backend == "oxigraph"
        solutions = self._oxistore.query(sparql)
        out: list[dict[str, str]] = []
        for sol in solutions:
            d: dict[str, str] = {}
            for var in sol.variables:
                node = sol[var]
                if node is None:
                    continue
                d[var.value] = node.value if hasattr(node, "value") else str(node)
            out.append(d)
        return out


def _rdflib_obj(o: str | LiteralTuple):
    if isinstance(o, tuple):
        _, val, dt = o
        if dt:
            return RdflibLiteral(val, datatype=URIRef(dt))
        return RdflibLiteral(val)
    return URIRef(o)


def _open_oxigraph(data_dir: pathlib.Path | None, kb_name: str):
    import pyoxigraph as oxi

    if data_dir is None:
        data_dir = pathlib.Path("data/claim_graphs") / kb_name
    data_dir.mkdir(parents=True, exist_ok=True)
    return oxi.Store(str(data_dir / "graph.db"))
