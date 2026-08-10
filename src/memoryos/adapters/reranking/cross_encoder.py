"""Local sentence-transformers cross-encoder.

Same disciplines as the embedder adapter, for the same reasons: loaded once per
process behind a lock, sized from the model rather than from a constant, and
never asked to read text it will silently discard.

**The truncation here is the M1.6.1 lesson applied to a second model.** That
defect was a chunker sized to 512 tokens against a model that read 256: nothing
errored, nothing failed a test, and half of every long chunk was thrown away
before it reached the encoder. Retrieval was quietly worse for a milestone and a
half. A cross-encoder has the same failure available to it and a worse version
of it — the pair is `[CLS] query [SEP] document [SEP]`, so a long document does
not merely lose its tail, it can push the *query* out of the window and leave
the model scoring a document against nothing. So the pair is truncated here,
deliberately and measurably, rather than left to whatever the tokenizer would
have done.
"""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from memoryos.application.ports import Reranker

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import CrossEncoder

logger = structlog.get_logger(__name__)

# Small, fast, and benchmarked for exactly this: reranking a shortlist of
# passages against a short query. 6 layers against the bi-encoder's 12, and it
# still outperforms it on pair relevance because it gets to read both sides.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Bumped by hand when the scores this adapter produces could change. Unlike the
# embedder's, this revision is not part of any cache key — reranking is computed
# per query and never stored — so it exists for the breakdown and the logs.
REVISION = "1"

# Same sentinel handling as the embedder: some tokenizers ship
# `model_max_length` as 1e30 meaning "unset".
FALLBACK_MAX_LENGTH = 512
_IMPLAUSIBLE_LENGTH = 100_000

# Pairs per forward pass. The shortlist is 50, so this is one or two batches —
# large enough that the per-batch overhead disappears, small enough that a
# larger shortlist does not allocate unboundedly.
DEFAULT_BATCH_SIZE = 32

# Special tokens in `[CLS] query [SEP] document [SEP]`, plus one of margin.
# Counted rather than assumed, but a floor is kept in case a tokenizer reports
# something odd.
_SPECIAL_TOKEN_BUDGET = 4


class CrossEncoderReranker(Reranker):
    """Scores query-document pairs with a cross-encoder, loaded once per process."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model: CrossEncoder | None = None
        self._tokenizer: Any | None = None
        # A worker and an API request can both rerank; two threads racing to
        # load would allocate the model twice.
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return f"{self._model_name}@{REVISION}"

    @property
    def max_length(self) -> int:
        """Tokens the model reads per pair, from the model rather than a constant."""
        model = self._load()
        # `max_seq_length` first: sentence-transformers renamed the attribute and
        # reading the old one emits a DeprecationWarning on every call. Both are
        # tried, because the fallback is what older pinned versions expose.
        for attribute in ("max_seq_length", "max_length"):
            configured = getattr(model, attribute, None)
            if isinstance(configured, int) and 0 < configured <= _IMPLAUSIBLE_LENGTH:
                return configured
        window = getattr(self._load_tokenizer(), "model_max_length", None)
        if isinstance(window, int) and 0 < window <= _IMPLAUSIBLE_LENGTH:
            return window
        return FALLBACK_MAX_LENGTH

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance for each document against the query, in the order given."""
        if not documents:
            return []

        pairs: Any = [[query, self.fit(query, document)] for document in documents]
        model = self._load()
        scores = model.predict(
            pairs, batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(score) for score in scores]

    def fit(self, query: str, document: str) -> str:
        """The document truncated to what will actually fit beside the query.

        Public because the truncation is a claim worth testing directly rather
        than inferring from a score. A document short enough to fit comes back
        unchanged, so the common case costs one token count.

        The query is never truncated. It is the shorter side by a wide margin,
        and a model scoring a full document against half a question would be
        worse than one scoring half a document against the whole question.
        """
        budget = self.max_length - self._count(query) - _SPECIAL_TOKEN_BUDGET
        if budget <= 0:
            # A query long enough to fill the window on its own. Nothing useful
            # remains for the document, and saying so beats silently scoring
            # every candidate against an empty string.
            logger.warning(
                "reranker.query_fills_window",
                model=self._model_name,
                max_length=self.max_length,
                query_tokens=self._count(query),
            )
            return ""

        tokenizer = self._load_tokenizer()
        token_ids = tokenizer.encode(document, add_special_tokens=False)
        if len(token_ids) <= budget:
            return document

        logger.info(
            "reranker.document_truncated",
            model=self._model_name,
            document_tokens=len(token_ids),
            budget=budget,
        )
        return str(tokenizer.decode(token_ids[:budget], skip_special_tokens=True))

    def _count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._load_tokenizer().encode(text, add_special_tokens=False))

    def _load(self) -> "CrossEncoder":
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                logger.info("reranker.loading", model=self._model_name)
                model = CrossEncoder(
                    self._model_name,
                    cache_folder=str(self._cache_dir) if self._cache_dir else None,
                )
                self._model = model
                logger.info(
                    "reranker.loaded",
                    model=self._model_name,
                    max_length=getattr(model, "max_seq_length", None),
                )
            return self._model

    def _load_tokenizer(self) -> Any:
        """The tokenizer alone, for counting and truncating.

        Reached through the loaded model rather than fetched separately: a
        second download of the same tokenizer would be a second thing that can
        disagree with the model actually doing the scoring.
        """
        if self._tokenizer is None:
            self._tokenizer = self._load().tokenizer
        return self._tokenizer
