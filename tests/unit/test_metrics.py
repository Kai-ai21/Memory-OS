"""The four metrics, against numbers worked out by hand.

Hand-computed rather than property-based on purpose. A property test would say
recall never exceeds 1.0 and would pass just as happily against an
implementation that divided by the wrong denominator; the only way to catch that
is to know what the answer should be before running the code.

The fixed ranking below is chosen so that all four disagree with each other. The
relevant set contains `z`, which is never retrieved, so recall and precision
cannot coincide; the first hit is at rank 2, so MRR is not 1.0; and only one of
three relevant items comes back, so nDCG is neither 0 nor 1.
"""

import pytest

from memoryos.application.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "e", "z"}


def test_metrics_match_hand_computed_values() -> None:
    # Top 3 is a, b, c. One of the three relevant items is in it.
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    # `b` at rank 2.
    assert reciprocal_rank(RETRIEVED, RELEVANT) == pytest.approx(0.5)
    # DCG = 1/log2(3) = 0.630930, one hit at rank 2 and nothing else in the top 3.
    # IDCG = 1 + 1/log2(3) + 1/log2(4) = 2.130930, the best three-hit ranking.
    assert ndcg_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(0.2960819109658652)

    # Widening the cutoff finds `e` at rank 5: recall doubles, precision falls
    # because the denominator grew faster than the numerator.
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 5)

    # Nothing relevant retrieved: all zero, and no ZeroDivisionError from the
    # empty intersection.
    assert recall_at_k(["a", "c", "d"], {"z"}, 3) == 0.0
    assert precision_at_k(["a", "c", "d"], {"z"}, 3) == 0.0
    assert reciprocal_rank(["a", "c", "d"], {"z"}) == 0.0
    assert ndcg_at_k(["a", "c", "d"], {"z"}, 3) == 0.0

    # Nothing to find, and nothing found. The degenerate inputs that would divide
    # by zero if the guards were removed.
    for metric in (recall_at_k, precision_at_k, ndcg_at_k):
        assert metric([], set(), 3) == 0.0
        assert metric(RETRIEVED, RELEVANT, 0) == 0.0
    assert reciprocal_rank([], set()) == 0.0


def test_ndcg_is_one_when_the_order_is_perfect() -> None:
    """1.0 means "as good as this query could have scored", not "all of them"."""
    assert ndcg_at_k(["b", "e", "a", "c"], {"b", "e"}, 4) == pytest.approx(1.0)

    # Still 1.0 when k cuts the ranking short of the relevant set: two relevant
    # items cannot both be in the top one, so the ideal is one hit at rank 1.
    assert ndcg_at_k(["b", "e", "a"], {"b", "e"}, 1) == pytest.approx(1.0)

    # And below 1.0 the moment an irrelevant result is promoted above a relevant
    # one, with nothing else changed.
    assert ndcg_at_k(["a", "b", "e", "c"], {"b", "e"}, 4) < 1.0


@pytest.mark.parametrize("position", range(1, 6))
def test_reciprocal_rank_is_one_over_the_first_hit(position: int) -> None:
    retrieved = [f"miss-{index}" for index in range(1, 6)]
    retrieved[position - 1] = "hit"
    # A second relevant item further down must not change the answer: the metric
    # is about the *first* one.
    if position != 5:
        retrieved[4] = "also-relevant"

    assert reciprocal_rank(retrieved, {"hit", "also-relevant"}) == pytest.approx(
        1 / position
    )
