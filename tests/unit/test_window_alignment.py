"""The guard that stops this defect returning."""

import pytest

from memoryos.adapters.chunking.structural import ChunkerConfig, StructuralChunker
from memoryos.container import WindowMisalignment, _assert_window_alignment
from tests.support.fakes import FakeEmbedder


def test_a_matched_pair_starts() -> None:
    embedder = FakeEmbedder(max_sequence_tokens=512)
    _assert_window_alignment(StructuralChunker(embedder), embedder)


@pytest.mark.parametrize("window", [64, 128, 256, 512, 8192])
def test_a_derived_chunker_always_matches_its_embedder(window: int) -> None:
    # Deriving from the window is what makes this true by construction rather
    # than by somebody remembering to update two numbers together.
    embedder = FakeEmbedder(max_sequence_tokens=window)
    _assert_window_alignment(StructuralChunker(embedder), embedder)


def test_a_chunker_larger_than_the_model_refuses_to_start() -> None:
    """The most important line in the hotfix.

    Parameters drift and models get swapped; neither side complains when they
    stop agreeing, because the extra text is discarded silently. This turns
    that into a startup failure.
    """
    embedder = FakeEmbedder(max_sequence_tokens=256)
    oversized = StructuralChunker(embedder, ChunkerConfig.for_window(1024))

    with pytest.raises(WindowMisalignment) as caught:
        _assert_window_alignment(oversized, embedder)

    message = str(caught.value)
    # Both numbers named, so the message says what to change.
    assert "1008" in message
    assert "256" in message
    assert "discarded" in message


def test_the_message_suggests_the_fix() -> None:
    embedder = FakeEmbedder(max_sequence_tokens=256)
    oversized = StructuralChunker(embedder, ChunkerConfig.for_window(512))

    with pytest.raises(WindowMisalignment, match=r"for_window\(256\)"):
        _assert_window_alignment(oversized, embedder)


def test_the_old_m14_configuration_would_now_be_refused() -> None:
    # 640-target chunks against a 256-token model: the exact defect, expressed
    # as the assertion that would have caught it.
    embedder = FakeEmbedder(max_sequence_tokens=256)
    old = StructuralChunker(
        embedder, ChunkerConfig(model_window=256, maximum=1024, target=640, overlap=80, minimum=120)
    )

    with pytest.raises(WindowMisalignment):
        _assert_window_alignment(old, embedder)


def test_settings_and_the_adapter_agree_on_the_model() -> None:
    """One source of truth for the model name.

    They were two, briefly, and the CLI silently ran a different model from the
    tests — which is the same class of drift as the window misalignment itself.
    """
    from memoryos.adapters.embedding.sentence_transformers import DEFAULT_MODEL
    from memoryos.config import Settings

    assert Settings().embedding_model == DEFAULT_MODEL


def test_every_way_of_building_the_embedder_agrees_on_the_query_prefix() -> None:
    """The same drift, one level down, caught in the fast suite.

    The prefix follows the model name through `__init__`, so the factory the CLI
    uses and the bare constructor the tests use cannot disagree about it.
    Deciding it in `build_embedder` instead would have reintroduced exactly the
    defect above: a test measuring one thing and production doing another. No
    model is loaded here — the prefix is settled before any weights are read.
    """
    from memoryos.adapters.embedding.sentence_transformers import (
        DEFAULT_MODEL,
        SentenceTransformerEmbedder,
        build_embedder,
        query_prefix_for,
    )
    from memoryos.config import Settings

    expected = query_prefix_for(DEFAULT_MODEL)
    assert build_embedder(Settings()).query_prefix == expected
    assert SentenceTransformerEmbedder().query_prefix == expected


def test_a_documented_prefix_is_not_applied_until_it_is_measured() -> None:
    """Recorded and applied are separate, and this is the seam between them.

    bge-v1.5's card documents an instruction and then says to pick by measuring.
    We measured, on this corpus it loses, so it is documented and not applied.
    `tests/slow/test_query_prefix.py` is what re-checks that verdict.
    """
    from memoryos.adapters.embedding.sentence_transformers import (
        APPLY_QUERY_PREFIX,
        DEFAULT_MODEL,
        DOCUMENTED_QUERY_PREFIXES,
        query_prefix_for,
    )

    assert DOCUMENTED_QUERY_PREFIXES[DEFAULT_MODEL], "the documented text was lost"
    applied = DEFAULT_MODEL in APPLY_QUERY_PREFIX
    assert bool(query_prefix_for(DEFAULT_MODEL)) is applied


def test_a_model_with_no_documented_prefix_stays_symmetric() -> None:
    # Guessing a prefix for a model that was not trained with one degrades it.
    from memoryos.adapters.embedding.sentence_transformers import query_prefix_for

    assert query_prefix_for("sentence-transformers/all-MiniLM-L6-v2") == ""


def test_the_configured_pair_is_aligned() -> None:
    # The real pair the application starts with, not a fabricated one.
    from memoryos.adapters.embedding.sentence_transformers import build_embedder
    from memoryos.config import Settings

    embedder = build_embedder(Settings())
    _assert_window_alignment(StructuralChunker(embedder), embedder)
