"""Test doubles for ports this project owns.

A fake embedder is legitimate precisely because the `Embedder` port is ours:
the contract it honours is one we wrote and can change. Mocking
sentence-transformers itself would be a different thing — it would assert our
beliefs about that library rather than its behaviour, and those beliefs are
exactly what the one slow test exists to check.
"""

import hashlib
import math
from collections.abc import Sequence

from memoryos.application.ports import Embedder

FAKE_MODEL_ID = "fake/deterministic@1"


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
    ) -> None:
        self._model_id = model_id
        self._dimension = dimension
        # When set, `embed` returns vectors of this width instead — for the
        # test that a mismatch is caught before anything is written.
        self._broken_dimension = broken_dimension
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
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        width = self._broken_dimension or self._dimension
        return [self._vector(text, width) for text in texts]

    def _vector(self, text: str, width: int) -> list[float]:
        seed = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
        # Stretch the digest to the required width, then normalise, so the
        # fake honours the port's `normalizes = True` claim.
        raw = [seed[index % len(seed)] / 255.0 - 0.5 for index in range(width)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]
