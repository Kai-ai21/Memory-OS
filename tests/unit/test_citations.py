"""Excerpt windows and explanation arithmetic.

Pure functions, no database. What is under test is the part that gets silently
wrong: an offset that looks right, a window that starts mid-word, a percentage
that does not add up.
"""

import pytest

from memoryos.domain.citation import build_excerpt
from memoryos.domain.explanation import build_explanation

PROSE = (
    "The worker claims a task from the queue. It then holds a lease on that task "
    "while the handler runs. Renewing the lease is how a long task keeps its hold "
    "on the work it started. A sweeper reclaims anything whose lease has lapsed."
)


def test_excerpt_boundaries_snap_to_sentence_breaks_and_never_split_a_word() -> None:
    """A citation that starts mid-word reads as corrupt and discredits the quote."""
    span_start = PROSE.index("Renewing")
    span_end = span_start + len("Renewing the lease is how a long task keeps its hold")

    excerpt = build_excerpt(PROSE, span_start, span_end, context_chars=60)

    # Context was added on both sides...
    assert len(excerpt.text) > span_end - span_start
    # ...and neither edge lands inside a word.
    assert not excerpt.text.startswith(" ")
    assert excerpt.text[0].isupper() or excerpt.text[0] == "\n"
    first_word = excerpt.text.split()[0]
    assert first_word in PROSE.split(), first_word
    last_word = excerpt.text.split()[-1]
    assert last_word in PROSE.split(), last_word

    # It began at a sentence boundary rather than 60 characters back.
    assert excerpt.text.startswith("It then holds") or excerpt.text.startswith("The worker")


def test_excerpt_offsets_locate_the_span_inside_the_excerpt() -> None:
    """The offsets are the point: a UI highlights with them and redoes no maths."""
    span_start = PROSE.index("Renewing")
    span_end = span_start + len("Renewing the lease")

    excerpt = build_excerpt(PROSE, span_start, span_end, context_chars=80)

    # The span located *within the excerpt* is the same text as the span located
    # within the memory. This is the identity the whole citation rests on.
    assert excerpt.span == PROSE[span_start:span_end] == "Renewing the lease"
    assert excerpt.text[excerpt.span_start : excerpt.span_end] == "Renewing the lease"

    # At the very start of a document there is nothing to truncate.
    whole = build_excerpt(PROSE, 0, 10, context_chars=500)
    assert whole.truncated_start is False
    assert whole.truncated_end is False
    assert whole.span_start == 0
    assert whole.span == PROSE[0:10]

    # A span with no room for context still returns the span itself.
    tight = build_excerpt(PROSE, span_start, span_end, context_chars=0)
    assert tight.span == "Renewing the lease"

    with pytest.raises(ValueError, match="invalid span"):
        build_excerpt(PROSE, 10, 5)


def test_signal_shares_sum_to_one() -> None:
    """The shares are recomputed from the fusion, so they have to reconstruct it.

    If they stop summing to 1.0 the explanation has drifted from the arithmetic
    that produced the ranking, and an explanation that does not describe the
    ranking is worse than none.
    """
    explanation = build_explanation(
        final_rank=2,
        fused_score=0.0325,
        ranks={
            "semantic": (8, 0.71, 1.0),
            "keyword": (1, 0.10, 1.0),
            "recency": (3, 0.9, 0.3),
        },
        rrf_k=60,
        rerank_score=4.2,
        previous_rank=5,
    )

    assert sum(item.share for item in explanation.contributions) == pytest.approx(1.0)
    # Ordered by share, so the first clause of `why` is the reason that mattered.
    shares = [item.share for item in explanation.contributions]
    assert shares == sorted(shares, reverse=True)
    assert explanation.contributions[0].name == "keyword"

    # A ranking at weight zero contributed nothing and is not listed as a reason.
    off = build_explanation(
        final_rank=1,
        fused_score=0.016,
        ranks={"semantic": (1, 0.8, 1.0), "importance": (2, 0.4, 0.0)},
        rrf_k=60,
    )
    assert [item.name for item in off.contributions] == ["semantic"]
    assert sum(item.share for item in off.contributions) == pytest.approx(1.0)

    # Nothing found it: no contributions, no division, and a sentence that says so.
    empty = build_explanation(final_rank=1, fused_score=0.0, ranks={}, rrf_k=60)
    assert empty.contributions == []
    assert "no ranking signal" in empty.why


def test_why_is_a_sentence_assembled_from_the_numbers() -> None:
    explanation = build_explanation(
        final_rank=2,
        fused_score=0.0325,
        ranks={"semantic": (8, 0.71, 1.0), "keyword": (1, 0.10, 1.0)},
        rrf_k=60,
        rerank_score=4.2,
        previous_rank=5,
    )

    assert explanation.why == (
        "Ranked 2nd: strong keyword match (rank 1), "
        "moderate semantic match (rank 8), reranked up from 5th."
    )

    # `previous_rank` counts memories, like `final_rank`. Equal means the
    # reranker considered it and left it alone, which is worth saying.
    unchanged = build_explanation(
        final_rank=3,
        fused_score=0.02,
        ranks={"semantic": (2, 0.7, 1.0)},
        rrf_k=60,
        previous_rank=3,
    )
    assert unchanged.why.endswith("unchanged by reranking.")

    # And with no reranking at all the sentence says nothing about it.
    plain = build_explanation(
        final_rank=1, fused_score=0.02, ranks={"keyword": (12, 0.05, 1.0)}, rrf_k=60
    )
    assert "rerank" not in plain.why
    assert plain.why == "Ranked 1st: weak keyword match (rank 12)."
