"""Recording external boundaries for real passage retrieval integration."""

from types import SimpleNamespace

from perspicacite.rag.dynamic_kb import DynamicKnowledgeBase, KnowledgeBaseConfig
from perspicacite.retrieval.multi_kb import MultiKBRetriever


def hit(chunk_id, text, score, paper="paper", metadata=True):
    source = SimpleNamespace(paper_id=paper, doi=f"10.0000/{paper}", license_id="CC-BY-4.0")
    chunk = SimpleNamespace(id=chunk_id, text=text, metadata=source if metadata else None)
    return SimpleNamespace(chunk=chunk, score=score)


class Embedding:
    dimension = 2
    model_name = "fixture-embedding"

    def __init__(self, served_model=None):
        self.calls = []
        self.last_used_model = self.model_name
        self.served_model = served_model or self.model_name

    async def embed_query(self, texts):
        self.calls.append(texts)
        self.last_used_model = self.served_model
        return [[0.2, 0.4]]


class Store:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def search(self, collection, query_embedding, top_k=10, filters=None):
        self.calls.append((collection, query_embedding, top_k, filters))
        rows = self.rows[collection]
        if isinstance(rows, Exception):
            raise rows
        return rows


def metadata(name, model="fixture-embedding"):
    return SimpleNamespace(name=name, collection_name=f"kb_{name}", embedding_model=model)


def retriever(kind, rows, *, model="fixture-embedding", served_model=None, floor=0.0):
    store, embedding = Store(rows), Embedding(served_model)
    if kind == "single":
        result = DynamicKnowledgeBase(
            store, embedding, KnowledgeBaseConfig(min_relevance_score=floor)
        )
        result.collection_name = "kb_a"
        result.kb_name = "a"
        result.expected_embedding_model = model
        result._initialized = True
    else:
        result = MultiKBRetriever(
            store, embedding, [metadata(name[3:], model) for name in rows], default_min_score=floor
        )
    return result, store, embedding
