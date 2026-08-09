"""The embedding pipeline's pure parts, with a fake model."""

import pytest

from memoryos.adapters.db.embedding_cache import cache_key_for
from memoryos.application.embed import _batched
from memoryos.domain.values import EmbeddingRole
from tests.support.fakes import FakeEmbedder

MODEL_A = "sentence-transformers/all-MiniLM-L6-v2@1"
MODEL_B = "sentence-transformers/all-MiniLM-L6-v2@2"

PASSAGE = EmbeddingRole.PASSAGE
QUERY = EmbeddingRole.QUERY


def key(model_id: str, text: str, role: EmbeddingRole = PASSAGE) -> str:
    return cache_key_for(model_id, text, role=role)


# --------------------------------------------------------------------------
# Cache keys
# --------------------------------------------------------------------------


def test_the_same_model_and_text_give_the_same_key() -> None:
    assert key(MODEL_A, "hello") == key(MODEL_A, "hello")


def test_changing_the_text_changes_the_key() -> None:
    assert key(MODEL_A, "hello") != key(MODEL_A, "hello there")


def test_changing_the_model_changes_the_key() -> None:
    """The correctness requirement, not an optimisation.

    Keying on text alone would let a model upgrade silently reuse the old
    model's vectors. Nothing errors: the index just ends up holding two
    incompatible coordinate systems, and similarity between them is
    arithmetically valid and semantically meaningless.
    """
    assert key(MODEL_A, "hello") != key(MODEL_B, "hello")


def test_a_bare_revision_bump_is_enough_to_change_the_key() -> None:
    assert key("m@1", "text") != key("m@2", "text")


def test_the_same_text_in_the_two_roles_does_not_collide() -> None:
    """The defect the role exists to prevent.

    An asymmetric model encodes the same sentence differently as a query than as
    a passage. Without the role in the key those two calls land on one entry,
    and whichever ran second silently receives the first one's vector — no
    error, just a query compared in the wrong half of the geometry. Exactly the
    failure the model id in the key already guards against, one level down.
    """
    assert key(MODEL_A, "leases expire", PASSAGE) != key(MODEL_A, "leases expire", QUERY)


def test_the_role_is_stable_for_the_same_text() -> None:
    assert key(MODEL_A, "hello", QUERY) == key(MODEL_A, "hello", QUERY)


def test_the_null_separator_prevents_a_boundary_collision() -> None:
    """Without a separator, ("ab", "c") and ("a", "bc") concatenate alike.

    A null byte cannot appear in a model id or a role name, so no text can be
    crafted to slide a boundary and claim another triple's vector.
    """
    assert key("ab", "c") != key("a", "bc")
    assert key("model", "") != key("mode", "l")


def test_a_text_that_starts_with_a_null_cannot_forge_a_key() -> None:
    # The obvious attack on the separator: put the delimiter in the payload.
    assert key("model", "\0extra") != key("model\0extra", "")
    # Including one that tries to forge the role field.
    assert key(MODEL_A, "x", QUERY) != key(MODEL_A, f"{QUERY.value}\0x", PASSAGE)


def test_a_role_cannot_be_defaulted_at_the_call_site() -> None:
    """`role` is keyword-only and has no default, so nobody can guess it."""
    with pytest.raises(TypeError):
        cache_key_for(MODEL_A, "hello")  # type: ignore[call-arg]


def test_keys_are_lowercase_hex_of_the_expected_width() -> None:
    value = key(MODEL_A, "hello")
    assert len(value) == 64
    assert value == value.lower()
    int(value, 16)


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
    assert embedder.embed_passage(["hello"]) == embedder.embed_passage(["hello"])


def test_the_fake_gives_different_texts_different_vectors() -> None:
    (first,) = FakeEmbedder().embed_passage(["a"])
    (second,) = FakeEmbedder().embed_passage(["b"])
    assert first != second


def test_the_fake_returns_unit_vectors() -> None:
    # It claims `normalizes = True`, so it had better.
    (vector,) = FakeEmbedder().embed_passage(["hello"])
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_the_fake_counts_what_it_was_asked_to_embed() -> None:
    embedder = FakeEmbedder()
    embedder.embed_passage(["a", "b"])
    embedder.embed_passage(["c"])
    assert embedder.calls == [["a", "b"], ["c"]]
    assert embedder.texts_embedded == 3


def test_the_fake_embeds_nothing_for_an_empty_list() -> None:
    assert FakeEmbedder().embed_passage([]) == []


# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------


def test_a_symmetric_embedder_treats_the_two_roles_identically() -> None:
    """The port's default, and what a model with no documented prefix wants."""
    embedder = FakeEmbedder()
    assert embedder.embed_query(["hello"]) == embedder.embed_passage(["hello"])


def test_an_asymmetric_embedder_separates_them() -> None:
    embedder = FakeEmbedder(query_prefix="Query: ")
    assert embedder.embed_query(["hello"]) != embedder.embed_passage(["hello"])


def test_the_prefix_is_what_makes_the_query_vector_differ() -> None:
    # Not some other divergence: the query vector is the prefixed text's vector.
    embedder = FakeEmbedder(query_prefix="Query: ")
    assert embedder.embed_query(["hello"]) == embedder.embed_passage(["Query: hello"])
