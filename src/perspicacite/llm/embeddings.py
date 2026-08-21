"""Embedding providers for vector search."""

import math
from pathlib import Path
from typing import Any, Protocol

from perspicacite.logging import get_logger

logger = get_logger("perspicacite.llm.embeddings")

# OpenAI text-embedding-* models hard-cap each input at 8192 tokens; a single
# oversized chunk (e.g. a long PMC section ingested via the heading-aware DOI
# path) otherwise 400s the WHOLE batch and drops the entire paper to 0 chunks.
# Truncate each input to a safe budget before sending so one big section can
# never poison its batch.  cl100k_base is the tokenizer for text-embedding-3-*.
_EMBED_MAX_TOKENS = 8000  # margin under the 8192 hard cap
_CHARS_PER_TOKEN_FALLBACK = 3  # conservative when tiktoken is unavailable
_ENC_CACHE: dict[str, Any] = {}


def _truncate_for_embedding(text: str, max_tokens: int = _EMBED_MAX_TOKENS) -> str:
    """Return ``text`` truncated to at most ``max_tokens`` embedding tokens."""
    if not text:
        return text
    try:
        import tiktoken

        enc = _ENC_CACHE.get("cl100k_base")
        if enc is None:
            enc = tiktoken.get_encoding("cl100k_base")
            _ENC_CACHE["cl100k_base"] = enc
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        return enc.decode(toks[:max_tokens])
    except Exception:
        # tiktoken missing/failed → conservative char-based cap.
        cap = max_tokens * _CHARS_PER_TOKEN_FALLBACK
        return text if len(text) <= cap else text[:cap]


class EmbeddingFailedError(RuntimeError):
    """An embedding could not be produced, so no vector may be stored or queried.

    Raised instead of substituting a zero vector. A zero vector is not a
    neutral placeholder: in cosine space it sits at distance 1.0 from every
    other vector, which silently turns retrieval into a constant score over an
    arbitrary ordering rather than an error.
    """


def is_zero_vector(vector: list[float], tolerance: float = 1e-12) -> bool:
    """Return True when the vector's L2 norm is indistinguishable from zero."""
    return math.sqrt(sum(component * component for component in vector)) <= tolerance


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Contract for implementers: raise on failure, never substitute a zero
    vector for a text that has content. A zero vector is only ever a valid
    answer for input that is itself empty or whitespace.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (document mode)."""
        ...

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts.

        Instruct-tuned embedding models (stella, gte-Qwen2) require a
        task-specific instruction prefix when encoding *queries* but NOT
        when encoding *documents*.  Callers that embed a query should call
        ``embed_query`` instead of ``embed`` so that the correct prompt is
        applied.  For models that do not use query prompts (MiniLM, OpenAI,
        SPECTER2, …) this method falls back to ``embed``.
        """
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...


class LiteLLMEmbeddingProvider:
    """
    Embedding provider using LiteLLM.

    Supports OpenAI, Cohere, Voyage, and other providers via LiteLLM.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 32,
    ):
        self.model = model
        self.batch_size = batch_size
        self._litellm = None
        self._dimension = self._get_dimension()

    def _get_litellm(self) -> Any:
        """Lazy import litellm."""
        if self._litellm is None:
            import litellm

            self._litellm = litellm
        return self._litellm

    def _get_dimension(self) -> int:
        """Get embedding dimension for the model.

        Handles bare OpenAI names and provider-prefixed routes (e.g.
        ``openrouter/qwen/qwen3-embedding-8b``) by matching on the final
        model segment, case-insensitively.
        """
        dimensions = {
            # OpenAI
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            # OpenRouter-hosted (verified 2026-05-28 via litellm openrouter/ route)
            "codestral-embed-2505": 1536,   # Mistral code embedding — good for code KBs
            "codestral-embed": 1536,
            "mistral-embed-2312": 1024,
            "mistral-embed": 1024,
            "qwen3-embedding-8b": 4096,
            "qwen3-embedding-4b": 2560,
            "qwen3-embedding-0.6b": 1024,
            "bge-m3": 1024,
            "bge-large-en-v1.5": 1024,
            "bge-base-en-v1.5": 768,
            "gemini-embedding-001": 3072,
            "gemini-embedding-2": 3072,
            "multilingual-e5-large": 1024,
            "e5-large-v2": 1024,
            "e5-base-v2": 768,
        }
        # Exact match first, then final-segment match for prefixed routes.
        if self.model in dimensions:
            return dimensions[self.model]
        tail = self.model.rsplit("/", 1)[-1].lower()
        return dimensions.get(tail, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts using LiteLLM.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts; truncate each to the model's token cap so a
        # single oversized chunk can never 400 (and thus drop) its whole batch.
        valid_texts = [_truncate_for_embedding(t) for t in texts if t and t.strip()]
        if not valid_texts:
            return [[0.0] * self.dimension for _ in texts]

        logger.debug("embedding_start", text_count=len(valid_texts), model=self.model)

        try:
            litellm = self._get_litellm()

            # Process in batches
            all_embeddings = []
            for i in range(0, len(valid_texts), self.batch_size):
                batch = valid_texts[i : i + self.batch_size]

                try:
                    response = await litellm.aembedding(
                        model=self.model,
                        input=batch,
                    )
                    batch_embeddings = [item["embedding"] for item in response["data"]]
                except Exception as batch_err:
                    # One poisoned input must not drop the whole paper: fall back
                    # to embedding the batch item-by-item, zero-filling only the
                    # individual texts that still fail.
                    logger.warning(
                        "embedding_batch_fallback",
                        model=self.model,
                        batch_size=len(batch),
                        error=str(batch_err),
                    )
                    batch_embeddings = []
                    for one in batch:
                        try:
                            r1 = await litellm.aembedding(model=self.model, input=[one])
                            batch_embeddings.append(r1["data"][0]["embedding"])
                        except Exception as item_err:
                            logger.error(
                                "embedding_item_skipped",
                                model=self.model,
                                error=str(item_err),
                            )
                            batch_embeddings.append([0.0] * self.dimension)
                all_embeddings.extend(batch_embeddings)

            logger.debug(
                "embedding_complete",
                text_count=len(valid_texts),
                dimension=self.dimension,
            )

            return all_embeddings

        except Exception as e:
            logger.error(
                "embedding_error",
                model=self.model,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts. API embedding models (OpenAI, Cohere, …) do not
        use separate query prompts — delegates to ``embed``."""
        return await self.embed(texts)


def _best_device() -> str:
    """Return the best available torch device.

    Selection order:
      * macOS (Darwin) with a working Metal backend → ``"mps"``
      * any platform with an NVIDIA GPU → ``"cuda"``
      * otherwise → ``"cpu"``

    MPS is Apple-Metal-only, so it is gated on ``platform.system() == "Darwin"``
    to be explicit (and to avoid a stray ``mps`` from an unusual torch build on
    a non-Mac host). Any import/availability error degrades safely to CPU.
    """
    import platform

    try:
        import torch

        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# Models that require trust_remote_code=True to load their custom forward pass.
# Without trust_remote_code, sentence-transformers loads the canonical HuggingFace
# implementation instead, which may use different attention/pooling logic.
#
# stella_en_1.5B_v5: requires trust_remote_code for custom modeling_qwen.py which
# implements the correct attention pattern.  The cached modeling_qwen.py was patched
# to fix transformers 5.x API incompatibilities (rope_theta, DynamicCache, get_usable_length).
# Models that require trust_remote_code=True to load their custom forward pass.
#
# stella_en_1.5B_v5 CANNOT use trust_remote_code=True with transformers 5.x.
# The custom modeling_qwen.py was written for transformers ~4.42 and the attention
# mask computation changed too much in 5.x (AttentionMaskConverter deprecated;
# different causal mask generation).  Patching rope_theta / DynamicCache / get_usable_length
# is not enough — the forward pass produces random-looking embeddings.
#
# Leave this dict empty until a compatible version of transformers is available.
_ST_TRUST_REMOTE_CODE: dict[str, bool] = {}

# Named prompts baked into the sentence-transformers model config.
# These are applied via ``model.encode(texts, prompt_name=<name>)``.
# Used for query-side encoding only — document embeddings use no prompt.
#
# IMPORTANT: only register models here when trust_remote_code=True works correctly.
# With mean pooling (the fallback when trust_remote_code fails), adding a query
# prompt DEGRADES retrieval because instruction tokens enter the mean and are absent
# from document embeddings, creating an embedding-space mismatch.
#
# stella_en_1.5B_v5 diagnosis (2026-05-27):
#   - trust_remote_code=False (std forward, no prompt): SciFact 0.737, NFCorpus 0.098
#   - trust_remote_code=True (patched, no prompt): random embeddings (broken)
#   - trust_remote_code=True (patched, s2p_query): random embeddings (broken)
# Root cause: transformers 5.x AttentionMaskConverter changes break the custom forward.
# Resolution: keep trust_remote_code=False; accept asymmetric retrieval penalty.
_ST_QUERY_PROMPT_NAMES: dict[str, str] = {}

# String prefixes for instruct models that use text-level instructions.
# Applied as a literal prefix: ``"<prefix>" + query``.
# Used when the model lacks a named built-in prompt.
#
# Same caution as _ST_QUERY_PROMPT_NAMES: only register when trust_remote_code
# is confirmed working.
#
# TODO: test gte-Qwen2-7B pooling + trust_remote_code compatibility before enabling:
#   "Alibaba-NLP/gte-Qwen2-7B-instruct": (
#       "Instruct: Given a web search query, retrieve relevant passages "
#       "that answer the query\nQuery: "
#   ),
_ST_QUERY_PROMPT_PREFIXES: dict[str, str] = {}


class SentenceTransformerEmbeddingProvider:
    """
    Local embedding provider using sentence-transformers.

    Falls back to this if API embeddings fail or for offline use.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model
        self.batch_size = batch_size
        self.device = device or _best_device()
        self._model = None

    def _get_model(self) -> Any:
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                trust_remote_code = _ST_TRUST_REMOTE_CODE.get(self.model_name, False)
                logger.info(
                    "loading_sentence_transformer",
                    model=self.model_name,
                    device=self.device,
                    trust_remote_code=trust_remote_code,
                )
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=trust_remote_code,
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._model is None:
            # Common dimensions — 768 is the default for scientific/BERT-size models.
            # Add entries here when using a model whose pre-load dimension is needed
            # for ChromaDB collection creation (create_collection is called before the
            # first embed(), so the lazy-loaded model isn't available yet).
            dimensions = {
                "all-MiniLM-L6-v2": 384,
                "all-MiniLM-L12-v2": 384,
                "all-mpnet-base-v2": 768,
                # SPECTER2 and SPECTER v1 scientific embedding models
                "allenai/specter2_base": 768,
                "allenai/specter2": 768,
                "sentence-transformers/allenai-specter": 768,
                # PubMed BERT / BioMedical ST models (768-dim BERT backbone)
                "pritamdeka/S-PubMedBert-MS-MARCO": 768,
                "NeuML/pubmedbert-base-embeddings": 768,
                # BGE family (BAAI) — multi-functionality retrieval models
                "BAAI/bge-m3": 1024,
                "BAAI/bge-large-en-v1.5": 1024,
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-small-en-v1.5": 384,
                # LLM-based general embedding models (large dimension)
                # bge-en-icl: Mistral-7B backbone, 4096-dim output
                "BAAI/bge-en-icl": 4096,
                # stella_en_1.5B_v5: 1.5B model, 1024-dim matryoshka output
                "dunzhang/stella_en_1.5B_v5": 1024,
                # GTE-Qwen2 family (Alibaba-NLP) — Qwen2 LLM backbone
                "Alibaba-NLP/gte-Qwen2-7B-instruct": 3584,
                "Alibaba-NLP/gte-Qwen2-1.5B-instruct": 1536,
            }
            return dimensions.get(self.model_name, 768)
        return self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts locally.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[0.0] * self.dimension for _ in texts]

        logger.debug(
            "local_embedding_start",
            text_count=len(valid_texts),
            model=self.model_name,
        )

        try:
            import asyncio

            model = self._get_model()

            # Run in thread pool since sentence-transformers is CPU-bound
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(
                    valid_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ),
            )

            # Convert to list of lists
            embeddings_list = embeddings.tolist()

            logger.debug(
                "local_embedding_complete",
                text_count=len(valid_texts),
                dimension=len(embeddings_list[0]) if embeddings_list else 0,
            )

            return embeddings_list

        except Exception as e:
            logger.error(
                "local_embedding_error",
                model=self.model_name,
                error=str(e),
            )
            raise

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts using a model-specific query prompt when available.

        For instruct-tuned embedding models (stella, gte-Qwen2-7B) the query
        must be encoded with a task-specific instruction to align with the
        document embedding space.  Without the instruction the query falls
        into a generic embedding region that does not match document vectors,
        causing near-random retrieval (NDCG@10 drops from ~0.42 to ~0.10 for
        stella on NFCorpus).

        Models without a registered prompt delegate straight to ``embed()``.
        """
        if not texts:
            return []

        # Check for named prompt (stella-style)
        prompt_name = _ST_QUERY_PROMPT_NAMES.get(self.model_name)
        # Check for string prefix (gte-Qwen2-style)
        prompt_prefix = _ST_QUERY_PROMPT_PREFIXES.get(self.model_name)

        if prompt_name is None and prompt_prefix is None:
            # No query-specific prompt for this model — plain embed.
            return await self.embed(texts)

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return [[0.0] * self.dimension for _ in texts]

        logger.debug(
            "local_query_embedding_start",
            text_count=len(valid_texts),
            model=self.model_name,
            prompt_name=prompt_name,
            has_prefix=prompt_prefix is not None,
        )

        try:
            import asyncio

            model = self._get_model()

            if prompt_name is not None:
                # Use the model's named built-in prompt (e.g. stella "s2p_query")
                embeddings = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: model.encode(
                        valid_texts,
                        prompt_name=prompt_name,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    ),
                )
            else:
                # Apply a string prefix to each query text
                prefixed = [prompt_prefix + t for t in valid_texts]
                embeddings = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: model.encode(
                        prefixed,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    ),
                )

            return embeddings.tolist()

        except Exception as e:
            logger.error(
                "local_query_embedding_error",
                model=self.model_name,
                error=str(e),
            )
            raise


class FallbackEmbeddingProvider:
    """
    Embedding provider with automatic fallback.

    Tries primary provider first, falls back to secondary on failure.
    """

    def __init__(
        self,
        primary: EmbeddingProvider,
        fallback: EmbeddingProvider,
    ):
        self.primary = primary
        self.fallback = fallback
        self._dimension = primary.dimension
        # F8 (audit 2026-05-15): track which inner provider actually
        # served the latest embed() call. KB-metadata writers should
        # use ``last_used_model`` rather than the legacy "A|B" tag —
        # vectors from the primary path are dimension/topology-
        # incompatible with vectors from the fallback, so recording
        # both as a single string is a latent footgun.
        self._last_used_model: str = primary.model_name
        self._fallback_triggered_count: int = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return the primary model name only.

        Historically returned ``"primary|fallback"`` for diagnostic
        purposes, but that string was being persisted as the KB's
        embedding fingerprint and broke multi-KB compatibility checks
        against KBs created from the MCP path (which uses
        ``LiteLLMEmbeddingProvider`` directly and so stamps the bare
        primary name). The composite is still available via
        :attr:`composite_name`. Vectors stored in a KB come from the
        primary unless ``primary_embedding_failed`` actually fires, in
        which case ``last_used_model`` records the truth.
        """
        return self.primary.model_name

    @property
    def composite_name(self) -> str:
        """Legacy ``"primary|fallback"`` string — diagnostic use only."""
        return f"{self.primary.model_name}|{self.fallback.model_name}"

    @property
    def last_used_model(self) -> str:
        """F8: actual model that served the most-recent embed() call.

        Use this for KB-metadata storage. ``model_name`` keeps the
        legacy "A|B" syntax for back-compat with existing rows.
        """
        return self._last_used_model

    @property
    def fallback_triggered_count(self) -> int:
        return self._fallback_triggered_count

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed with fallback."""
        try:
            out = await self.primary.embed(texts)
            self._last_used_model = self.primary.model_name
            return out
        except Exception as e:
            self._fallback_triggered_count += 1
            logger.warning(
                "primary_embedding_failed",
                primary=self.primary.model_name,
                fallback=self.fallback.model_name,
                error=str(e),
                fallback_triggered_count=self._fallback_triggered_count,
            )
            out = await self.fallback.embed(texts)
            self._last_used_model = self.fallback.model_name
            return out

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts with fallback — delegates to primary.embed_query()."""
        try:
            out = await self.primary.embed_query(texts)
            self._last_used_model = self.primary.model_name
            return out
        except Exception as e:
            self._fallback_triggered_count += 1
            logger.warning(
                "primary_query_embedding_failed",
                primary=self.primary.model_name,
                fallback=self.fallback.model_name,
                error=str(e),
                fallback_triggered_count=self._fallback_triggered_count,
            )
            out = await self.fallback.embed_query(texts)
            self._last_used_model = self.fallback.model_name
            return out


class TypedEmbeddingProvider:
    """Routes ``embed`` calls to different inner providers by content type.

    Sub-project B (2026-05-15 design). When the caller passes a parallel
    ``content_types`` list, the texts are partitioned per-type and each
    partition is routed through the matching inner provider (or the
    ``default`` when no specific provider is registered for that type).
    Vectors are stitched back into the original input order.

    When ``content_types`` is ``None`` the call collapses to a single
    invocation of the default provider — identical to today's behaviour
    for callers that don't know about per-type routing.

    Cost shape: at most ``1 + len(by_content_type)`` underlying API
    calls per ``embed`` invocation (one per inner provider that has
    any texts assigned to it).
    """

    def __init__(
        self,
        *,
        default: EmbeddingProvider,
        by_content_type: dict[str, EmbeddingProvider],
    ) -> None:
        self._default = default
        self._by_type = dict(by_content_type)

    @property
    def model_name(self) -> str:
        # Default first, then sorted-key inner providers as "type:model".
        suffix = "+".join(
            f"{k}:{self._by_type[k].model_name}"
            for k in sorted(self._by_type)
        )
        return self._default.model_name + (f"+{suffix}" if suffix else "")

    @property
    def dimension(self) -> int | None:
        """Returns the common dimension when all inner providers agree,
        else None so callers know they must split per-type at write time."""
        dims = {self._default.dimension}
        for p in self._by_type.values():
            dims.add(p.dimension)
        if len(dims) == 1:
            return next(iter(dims))
        return None

    async def embed(
        self,
        texts: list[str],
        *,
        content_types: list[str] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        if content_types is None:
            return await self._default.embed(texts)
        if len(content_types) != len(texts):
            raise ValueError(
                "content_types length must match texts length: "
                f"got {len(content_types)} vs {len(texts)}"
            )

        # Partition input by routed provider; preserve original index.
        buckets: dict[int, tuple[EmbeddingProvider, list[int], list[str]]] = {}
        for i, (t, ctype) in enumerate(zip(texts, content_types)):
            prov = self._by_type.get(ctype, self._default)
            key = id(prov)
            if key not in buckets:
                buckets[key] = (prov, [], [])
            buckets[key][1].append(i)
            buckets[key][2].append(t)

        # Run each bucket through its provider.
        out: list[list[float] | None] = [None] * len(texts)
        for prov, indices, batch_texts in buckets.values():
            vecs = await prov.embed(batch_texts)
            for idx, v in zip(indices, vecs):
                out[idx] = v

        if any(v is None for v in out):
            raise RuntimeError("internal: TypedEmbeddingProvider left a None slot")
        return out  # type: ignore[return-value]

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts — delegates to the default provider's embed_query()."""
        return await self._default.embed_query(texts)


class CachedEmbeddingProvider:
    """Wraps an :class:`EmbeddingProvider`, consulting an on-disk cache
    before forwarding uncached texts to the inner provider.

    Per-text keying: two overlapping batches share entries. Empty /
    whitespace inputs pass through to the zero-vector contract without
    touching the cache. See
    docs/superpowers/specs/2026-05-14-embedding-cache-design.md.
    """

    def __init__(self, *, inner: Any, cache: Any) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    async def embed(
        self,
        texts: list[str],
        cache: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        # Build per-text keys, but only for non-empty texts (matches
        # the inner providers' empty-input contract).
        from perspicacite.llm.embedding_cache import build_embedding_cache_key

        zero = [0.0] * self.inner.dimension
        keys: list[str | None] = []
        for t in texts:
            if not t or not t.strip():
                keys.append(None)
            else:
                keys.append(
                    build_embedding_cache_key(model=self.inner.model_name, text=t)
                )

        # Cache-bypass: straight to inner, no read, no write.
        if not cache:
            # Inner provider already handles empties → zero vec.
            return await self.inner.embed(texts)

        # Batch read.
        non_null_keys = [k for k in keys if k is not None]
        hits = await self.cache.get_many(non_null_keys) if non_null_keys else {}

        # Build the result list, collecting misses to send to inner.
        out: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for i, (t, k) in enumerate(zip(texts, keys)):
            if k is None:
                out[i] = zero  # empty/whitespace stays zero-vector
            elif k in hits:
                out[i] = hits[k]
            else:
                miss_indices.append(i)
                miss_texts.append(t)

        if miss_texts:
            new_vecs = await self.inner.embed(miss_texts)
            # Write to cache + slot into out in original order.
            put_items: list[tuple[str, str, list[float]]] = []
            for idx, vec in zip(miss_indices, new_vecs):
                out[idx] = vec
                k = keys[idx]
                if k is not None:
                    put_items.append((k, self.inner.model_name, vec))
            if put_items:
                await self.cache.put_many(put_items)

        # Final result — every slot is filled.
        return [v if v is not None else zero for v in out]

    async def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed query texts with a query-specific cache key.

        Instruct models produce different embeddings for the same text depending
        on whether it is encoded as a query or a document.  We distinguish the
        two by using a ``"q:" + model_name`` namespace in the cache key so that
        query and document embeddings for the same text are stored separately.
        """
        if not texts:
            return []

        from perspicacite.llm.embedding_cache import build_embedding_cache_key

        # "q:" prefix distinguishes query-mode embeddings from document-mode
        # embeddings in the cache (critical for instruct models like stella).
        query_model_ns = f"q:{self.inner.model_name}"
        zero = [0.0] * self.inner.dimension
        keys: list[str | None] = []
        for t in texts:
            if not t or not t.strip():
                keys.append(None)
            else:
                keys.append(build_embedding_cache_key(model=query_model_ns, text=t))

        # Batch read.
        non_null_keys = [k for k in keys if k is not None]
        hits = await self.cache.get_many(non_null_keys) if non_null_keys else {}

        out: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        for i, (t, k) in enumerate(zip(texts, keys)):
            if k is None:
                out[i] = zero
            elif k in hits:
                out[i] = hits[k]
            else:
                miss_indices.append(i)
                miss_texts.append(t)

        if miss_texts:
            new_vecs = await self.inner.embed_query(miss_texts)
            put_items: list[tuple[str, str, list[float]]] = []
            for idx, vec in zip(miss_indices, new_vecs):
                out[idx] = vec
                k = keys[idx]
                if k is not None:
                    put_items.append((k, query_model_ns, vec))
            if put_items:
                await self.cache.put_many(put_items)

        return [v if v is not None else zero for v in out]


def create_embedding_provider(
    model: str,
    use_local_fallback: bool = True,
    *,
    cache_enabled: bool = False,
    cache_path: "Path | str | None" = None,
    cache_ttl_days: int = 0,
    embedding_models_per_type: dict[str, str] | None = None,
) -> EmbeddingProvider:
    """
    Factory function to create an embedding provider.

    Args:
        model: Model name (e.g., 'text-embedding-3-small' or 'all-MiniLM-L6-v2')
        use_local_fallback: Whether to set up local fallback for API providers.
        cache_enabled: When True, wrap the returned provider in a
            :class:`CachedEmbeddingProvider`. The cache key is
            ``sha256(model || \\x00 || text)``, so switching models
            transparently invalidates the cache.
        cache_path: SQLite file backing the cache. Required when
            ``cache_enabled`` is True.
        cache_ttl_days: Days until a cached vector expires. 0 (default) =
            keep forever. Embeddings are deterministic per (model, text),
            so this is safe.
        embedding_models_per_type: Optional content-type → model string
            map (sub-project B). When non-empty, the returned provider
            is a :class:`TypedEmbeddingProvider` that dispatches by
            ``content_type`` and uses ``model`` as its default. When
            empty or None, behaviour is identical to today (single
            provider).

    Returns:
        EmbeddingProvider (or TypedEmbeddingProvider when per-type
        routing is configured). Caching wraps the outer object so the
        cache key reflects the routed model.
    """
    def _build_single(m: str) -> EmbeddingProvider:
        # Inner-provider selection rule:
        # sentence-transformers for:
        #   - models starting with "all-" (e.g. all-MiniLM-L6-v2)
        #   - bare names without "/" and without "embedding"
        #     (e.g. custom local model names)
        #   - "st:" prefix as an explicit override  (e.g. st:allenai/specter2_base)
        #   - models on the ST-hosted HuggingFace hub that are known to load
        #     via sentence_transformers.SentenceTransformer directly — detected
        #     by a known-namespace prefix list (allenai/, sentence-transformers/,
        #     pritamdeka/, cross-encoder/)
        # Otherwise → LiteLLM-backed with optional local fallback.
        _st_namespaces = (
            "allenai/",
            "sentence-transformers/",
            "pritamdeka/",
            "cross-encoder/",
            # BGE family (BAAI) and other HF-hosted retrieval models
            "BAAI/",
            "NeuML/",
            "ncats/",
            "microsoft/Biomed",
            # LLM-based embedding models — require sentence-transformers, not LiteLLM
            # (stored in KB metadata without st: prefix; must be auto-routed)
            "dunzhang/",       # stella_en_1.5B_v5 and related
            "Alibaba-NLP/",    # gte-Qwen2-7B-instruct and related
        )
        if m.startswith("st:"):
            return SentenceTransformerEmbeddingProvider(model=m[3:])
        if m.startswith("all-") or ("/" not in m and "embedding" not in m):
            return SentenceTransformerEmbeddingProvider(model=m)
        if any(m.startswith(ns) for ns in _st_namespaces):
            return SentenceTransformerEmbeddingProvider(model=m)
        primary = LiteLLMEmbeddingProvider(model=m)
        if use_local_fallback:
            return FallbackEmbeddingProvider(primary, SentenceTransformerEmbeddingProvider())
        return primary

    inner: EmbeddingProvider = _build_single(model)

    if embedding_models_per_type:
        by_type: dict[str, EmbeddingProvider] = {}
        for ctype, ctype_model in embedding_models_per_type.items():
            by_type[ctype] = _build_single(ctype_model)
        inner = TypedEmbeddingProvider(default=inner, by_content_type=by_type)

    if not cache_enabled:
        return inner

    if cache_path is None:
        raise ValueError(
            "create_embedding_provider(cache_enabled=True) requires cache_path"
        )

    # Import lazily to avoid importing numpy unless the cache is used.
    from perspicacite.llm.embedding_cache import EmbeddingCache

    cache = EmbeddingCache(path=cache_path, ttl_days=cache_ttl_days)
    return CachedEmbeddingProvider(inner=inner, cache=cache)
