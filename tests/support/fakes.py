"""Test doubles for ports this project owns.

A fake embedder is legitimate precisely because the `Embedder` port is ours:
the contract it honours is one we wrote and can change. Mocking
sentence-transformers itself would be a different thing — it would assert our
beliefs about that library rather than its behaviour, and those beliefs are
exactly what the one slow test exists to check.
"""

import hashlib
import math
import re
from collections.abc import Callable, Sequence

from memoryos.application.ports import Embedder, Reranker

FAKE_MODEL_ID = "fake/deterministic@1"

_TOKEN = re.compile(r"\w+|[^\w\s]")


class FakeEmbedder(Embedder):
    """Deterministic unit vectors derived from a hash of the text.

    Same text in, same vector out, so cache behaviour is testable. Different
    text gives an unrelated direction, which is all any test here needs — no
    test asserts that the geometry is *meaningful*, because a fake cannot
    establish that.
    """

    def __init__(
        self,
        model_id: str = FAKE_MODEL_ID,
        dimension: int = 384,
        *,
        broken_dimension: int | None = None,
        max_sequence_tokens: int = 512,
        query_prefix: str = "",
    ) -> None:
        self._model_id = model_id
        self._dimension = dimension
        # Configurable so tests can drive the window boundary — including a
        # deliberately tiny one — without loading a model.
        self._window = max_sequence_tokens
        # When set, `embed_passage` returns vectors of this width instead — for
        # the test that a mismatch is caught before anything is written.
        self._broken_dimension = broken_dimension
        # Empty by default, matching the port's symmetric default. A test that
        # needs the two roles to diverge sets one.
        self._query_prefix = query_prefix
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalizes(self) -> bool:
        return True

    @property
    def max_sequence_tokens(self) -> int:
        return self._window

    def count_tokens(self, text: str) -> int:
        """A deterministic stand-in for WordPiece.

        Words and punctuation, plus an extra token per long word, so that the
        fake overcounts dense identifiers roughly the way a real tokenizer
        does. It does not need to match any model; it needs to be stable and
        to punish long tokens.
        """
        pieces = _TOKEN.findall(text)
        return sum(1 + len(piece) // 8 for piece in pieces)

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)

    def embed_passage(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        width = self._broken_dimension or self._dimension
        return [self._vector(text, width) for text in texts]

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._query_prefix:
            return self.embed_passage(texts)
        return self.embed_passage([self._query_prefix + text for text in texts])

    def _vector(self, text: str, width: int) -> list[float]:
        seed = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
        # Stretch the digest to the required width, then normalise, so the
        # fake honours the port's `normalizes = True` claim.
        raw = [seed[index % len(seed)] / 255.0 - 0.5 for index in range(width)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]


class FakeReranker(Reranker):
    """Deterministic pair scores, so a test never loads a cross-encoder.

    Legitimate for the same reason `FakeEmbedder` is: `Reranker` is a port this
    project owns, and what the fake honours is a contract we wrote. It cannot
    establish that reranking *improves* anything — that is what the golden set
    and the one slow test are for — but it can establish that the pipeline
    reorders by whatever the reranker says, truncates before asking, and records
    the answer.

    The default scores by input position descending, which reverses the
    shortlist. A reversal is the strongest possible signal that the pipeline
    honours the reranker rather than quietly keeping the fused order: any
    partial ordering could be a coincidence, and the identity ordering would be
    indistinguishable from ignoring the model entirely.
    """

    def __init__(
        self,
        model_id: str = "fake/cross-encoder@1",
        *,
        max_length: int = 64,
        scorer: Callable[[str, str], float] | None = None,
    ) -> None:
        self._model_id = model_id
        self._max_length = max_length
        self._scorer = scorer
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_length(self) -> int:
        return self._max_length

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self._scorer is not None:
            return [self._scorer(query, document) for document in documents]
        # Descending in input position: the first candidate scores lowest, so a
        # pipeline that honours the reranker returns the shortlist reversed.
        return [float(index) for index in range(len(documents))]

    @property
    def pairs_scored(self) -> int:
        return sum(len(documents) for _, documents in self.calls)
