"""The one test that loads the real model.

Everything else in this milestone runs against a fake, which is right — a fake
proves the plumbing without a 90MB download on every run. But a fake cannot
tell you whether the vectors mean anything, and a pipeline that fills the
column with plausible-looking garbage passes every other test in the suite.
This is the only thing standing between that and us.

Marked `slow` and excluded from the default run; `make test-slow` runs it.
"""

import math

import pytest

from memoryos.adapters.embedding.sentence_transformers import (
    DEFAULT_DIMENSION,
    DimensionMismatch,
    SentenceTransformerEmbedder,
)

pytestmark = pytest.mark.slow

SIMILAR_A = "The worker claims a job from the queue and runs its handler."
SIMILAR_B = "A worker takes the next queued task and executes it."
UNRELATED = "Sourdough needs a long cold proof to develop its flavour."


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


def test_the_model_produces_the_width_the_column_expects(
    embedder: SentenceTransformerEmbedder,
) -> None:
    (vector,) = embedder.embed(["hello"])
    assert len(vector) == DEFAULT_DIMENSION == 384


def test_vectors_are_unit_length(embedder: SentenceTransformerEmbedder) -> None:
    """`normalizes` claims this, and M1.6's index will depend on it.

    With unit vectors, cosine similarity and inner product are the same number,
    and inner product is cheaper.
    """
    vectors = embedder.embed([SIMILAR_A, SIMILAR_B, UNRELATED])
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        assert norm == pytest.approx(1.0, abs=1e-5)


def test_similar_sentences_are_closer_than_unrelated_ones(
    embedder: SentenceTransformerEmbedder,
) -> None:
    """The assertion that the vectors are not garbage.

    A hash-based fake would satisfy every structural test in this suite and
    fail this one, because it has no notion of meaning at all.
    """
    similar_a, similar_b, unrelated = embedder.embed([SIMILAR_A, SIMILAR_B, UNRELATED])

    related = cosine(similar_a, similar_b)
    unrelated_a = cosine(similar_a, unrelated)
    unrelated_b = cosine(similar_b, unrelated)

    assert related > unrelated_a + 0.25, (
        f"two sentences about job queues scored {related:.3f} against "
        f"{unrelated_a:.3f} for an unrelated one — not a clear margin"
    )
    assert related > unrelated_b + 0.25


def test_the_same_text_embeds_identically(
    embedder: SentenceTransformerEmbedder,
) -> None:
    # The cache assumes this. If the model were nondeterministic, a cache hit
    # and a fresh call would disagree and nothing would ever notice.
    first, second = embedder.embed([SIMILAR_A, SIMILAR_A])
    assert first == pytest.approx(second)


def test_the_model_id_carries_a_revision(
    embedder: SentenceTransformerEmbedder,
) -> None:
    assert embedder.model_id.endswith("@1")
    assert "bge-small-en-v1.5" in embedder.model_id


def test_the_window_is_larger_than_the_model_it_replaced(
    embedder: SentenceTransformerEmbedder,
) -> None:
    # Measured, not assumed: bge-small reports 512 against MiniLM's 256, at the
    # same 384 dimensions, so the column and the index are unchanged.
    assert embedder.max_sequence_tokens == 512


def test_text_past_the_window_still_changes_the_embedding(
    embedder: SentenceTransformerEmbedder,
) -> None:
    """The defect M1.6.1 exists to fix.

    Under the old configuration — 640-token chunks against a 256-token model —
    two chunks sharing a long prefix embedded identically, because everything
    past the window was discarded. This asserts the pair the chunker can now
    actually produce differs.
    """
    window = embedder.max_sequence_tokens
    prefix = "The worker claims a job from the queue and holds a lease on it. "
    # Just inside the window, which is the largest chunk the chunker will emit.
    while embedder.count_tokens(prefix) < window - 40:
        prefix += "The worker claims a job from the queue and holds a lease on it. "

    first, second = embedder.embed([prefix + "ENDING ONE", prefix + "ENDING TWO"])

    assert first != second
    similarity = sum(a * b for a, b in zip(first, second, strict=True))
    assert similarity < 0.9999, "endings past the window were discarded"


def test_the_window_is_what_the_chunker_was_sized_against(
    embedder: SentenceTransformerEmbedder,
) -> None:
    from memoryos.adapters.chunking.structural import StructuralChunker

    chunker = StructuralChunker(embedder)
    # The invariant the startup assertion enforces, checked against the real
    # tokenizer rather than a fake.
    assert chunker.max_tokens <= embedder.max_sequence_tokens


def test_the_tokenizer_counts_code_far_above_the_old_heuristic(
    embedder: SentenceTransformerEmbedder,
) -> None:
    """Why counting words was not close enough.

    Identifiers split into several WordPieces, so a words-and-punctuation
    heuristic undercounts code by roughly 2.7x — enough that a chunk sized
    "safely" under the old scheme overflowed the window badly.
    """
    import re

    code = "\n".join(
        f"    intermediate_value_{n} = compute_something(intermediate_value_{n - 1})"
        for n in range(1, 60)
    )
    heuristic = len(re.findall(r"\w+|[^\w\s]", code))
    real = embedder.count_tokens(code)

    assert real > heuristic * 2


def test_a_wrong_expected_dimension_is_caught_at_load() -> None:
    # Better here, with both numbers named, than as an opaque pgvector width
    # error at insert time.
    wrong = SentenceTransformerEmbedder(dimension=512)
    with pytest.raises(DimensionMismatch, match="384-dimensional"):
        wrong.embed(["hello"])
