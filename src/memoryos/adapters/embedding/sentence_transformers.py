"""Local sentence-transformers embedder."""

import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import structlog

from memoryos.application.ports import Embedder

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Matches the VECTOR(384) column fixed in M1.1. That column is a declared
# width, so this is a compatibility constraint rather than a preference.
DEFAULT_DIMENSION = 384

# Bumped by hand whenever the weights behind `model_name` could differ — a
# pinned revision change, a different fine-tune. The cache key contains this
# string, so a stale value means silently reusing vectors from another model.
REVISION = "1"


class DimensionMismatch(RuntimeError):
    """The loaded model does not produce the width the column expects."""


class SentenceTransformerEmbedder(Embedder):
    """Wraps a sentence-transformers model, loaded once per process.

    Loading takes seconds and allocates hundreds of megabytes. Doing it per
    batch would dominate the runtime of everything else in this milestone, so
    the model is loaded on first use and then held.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        dimension: int = DEFAULT_DIMENSION,
        cache_dir: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._cache_dir = cache_dir
        self._model: SentenceTransformer | None = None
        # The worker may embed from more than one task; two threads racing to
        # load the same model would allocate it twice.
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return f"{self._model_name}@{REVISION}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalizes(self) -> bool:
        # Unit vectors make cosine similarity and inner product identical, and
        # inner product is cheaper. M1.6's index will rely on that.
        return True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def _load(self) -> "SentenceTransformer":
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            if self._cache_dir is not None:
                # transformers and tokenizers read HF_HOME directly, so setting
                # it keeps every layer of the stack pointed at one directory.
                os.environ.setdefault("HF_HOME", str(self._cache_dir))

            from sentence_transformers import SentenceTransformer

            logger.info("embedder.loading", model=self._model_name)
            model = SentenceTransformer(
                self._model_name,
                cache_folder=str(self._cache_dir) if self._cache_dir else None,
            )

            actual = _dimension_of(model)
            if actual != self._dimension:
                # Caught here rather than at insert time, where it surfaces as
                # an opaque pgvector width error a long way from its cause.
                raise DimensionMismatch(
                    f"{self._model_name} produces {actual}-dimensional vectors, "
                    f"but this pipeline and the memory_chunks.embedding column "
                    f"expect {self._dimension}"
                )

            self._model = cast("SentenceTransformer", model)
            logger.info("embedder.loaded", model=self._model_name, dimension=actual)
            return self._model


def build_embedder(settings: Any) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model, cache_dir=Path(settings.hf_home)
    )


def _dimension_of(model: object) -> int | None:
    """The model's output width, across sentence-transformers versions.

    `get_sentence_embedding_dimension` was renamed to `get_embedding_dimension`;
    both are tried so this keeps working either side of that change rather than
    emitting a deprecation warning on every load.
    """
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        method = getattr(model, name, None)
        if callable(method):
            return cast("int | None", method())
    return None
