"""The embedding pipeline's pure parts, with a fake model."""

import pytest

from memoryos.adapters.db.embedding_cache import cache_key_for
from memoryos.application.embed import _batched
from tests.support.fakes import FakeEmbedder

MODEL_A = "sentence-transformers/all-MiniLM-L6-v2@1"
MODEL_B = "sentence-transformers/all-MiniLM-L6-v2@2"


# --------------------------------------------------------------------------
# Cache keys
# --------------------------------------------------------------------------


def test_the_same_model_and_text_give_the_same_key() -> None:
    assert cache_key_for(MODEL_A, "hello") == cache_key_for(MODEL_A, "hello")


def test_changing_the_text_changes_the_key() -> None:
    assert cache_key_for(MODEL_A, "hello") != cache_key_for(MODEL_A, "hello there")


def test_changing_the_model_changes_the_key() -> None:
    """The correctness requirement, not an optimisation.

    Keying on text alone would let a model upgrade silently reuse the old
    model's vectors. Nothing errors: the index just ends up holding two
    incompatible coordinate systems, and similarity between them is
    arithmetically valid and semantically meaningless.
    """
    assert cache_key_for(MODEL_A, "hello") != cache_key_for(MODEL_B, "hello")


def test_a_bare_revision_bump_is_enough_to_change_the_key() -> None:
    assert cache_key_for("m@1", "text") != cache_key_for("m@2", "text")


def test_the_null_separator_prevents_a_boundary_collision() -> None:
    """Without a separator, ("ab", "c") and ("a", "bc") concatenate alike.

    A null byte cannot appear in a model id, so no text can be crafted to slide
    the boundary and claim another pairing's vector.
    """
    assert cache_key_for("ab", "c") != cache_key_for("a", "bc")
    assert cache_key_for("model", "") != cache_key_for("mode", "l")


def test_a_text_that_starts_with_a_null_cannot_forge_a_key() -> None:
    # The obvious attack on the separator: put the delimiter in the payload.
    assert cache_key_for("model", "\0extra") != cache_key_for("model\0extra", "")


def test_keys_are_lowercase_hex_of_the_expected_width() -> None:
    key = cache_key_for(MODEL_A, "hello")
    assert len(key) == 64
    assert key == key.lower()
    int(key, 16)


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "size", "expected"),
    [
        (0, 32, []),
        (1, 32, [1]),
        (32, 32, [32]),
        (33, 32, [32, 1]),
        (64, 32, [32, 32]),
        (65, 32, [32, 32, 1]),
        (5, 2, [2, 2, 1]),
        (4, 2, [2, 2]),
    ],
)
def test_batching_splits_at_multiples_and_remainders(
    count: int, size: int, expected: list[int]
) -> None:
    batches = _batched([f"text-{n}" for n in range(count)], size)
    assert [len(batch) for batch in batches] == expected


def test_batching_preserves_order_and_loses_nothing() -> None:
    texts = [f"text-{n}" for n in range(70)]
    assert [text for batch in _batched(texts, 32) for text in batch] == texts


# --------------------------------------------------------------------------
# The fake, which the integration tests lean on
# --------------------------------------------------------------------------


def test_the_fake_is_deterministic() -> None:
    embedder = FakeEmbedder()
    assert embedder.embed(["hello"]) == embedder.embed(["hello"])


def test_the_fake_gives_different_texts_different_vectors() -> None:
    (first,), (second,) = FakeEmbedder().embed(["a"]), FakeEmbedder().embed(["b"])
    assert first != second


def test_the_fake_returns_unit_vectors() -> None:
    # It claims `normalizes = True`, so it had better.
    (vector,) = FakeEmbedder().embed(["hello"])
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_the_fake_counts_what_it_was_asked_to_embed() -> None:
    embedder = FakeEmbedder()
    embedder.embed(["a", "b"])
    embedder.embed(["c"])
    assert embedder.calls == [["a", "b"], ["c"]]
    assert embedder.texts_embedded == 3


def test_the_fake_embeds_nothing_for_an_empty_list() -> None:
    assert FakeEmbedder().embed([]) == []
