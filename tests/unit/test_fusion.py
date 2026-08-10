"""Reciprocal rank fusion, against arithmetic done by hand.

The numbers below are written out rather than computed in the test, because a
test that recomputes the formula it is checking asserts only that Python is
deterministic. If `k` or the shape of the sum ever changes, these have to be
recalculated by a person — which is the point.
"""

import pytest

from memoryos.domain.fusion import reciprocal_rank_fusion


def test_scores_match_the_formula_worked_out_by_hand() -> None:
    """Two rankings, k=60, every document's score written out.

    `a` is first then second:   1/61 + 1/62 = 0.032522…
    `c` is third then first:    1/63 + 1/61 = 0.032266…
    `b` is second in one only:  1/62        = 0.016129…
    `d` is third in one only:   1/63        = 0.015873…
    """
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])

    assert [identifier for identifier, _ in fused] == ["a", "c", "b", "d"]
    assert dict(fused) == pytest.approx(
        {
            "a": 0.03252247488101534,
            "c": 0.032266458495966696,
            "b": 0.016129032258064516,
            "d": 0.015873015873015872,
        }
    )

    # Weights scale a ranking's whole contribution: `a` at rank 1 under weight 2
    # and rank 2 under weight 0.5 is 2/61 + 0.5/62.
    weighted = dict(
        reciprocal_rank_fusion(
            [["a", "b", "c"], ["c", "a", "d"]], weights=[2.0, 0.5]
        )
    )
    assert weighted["a"] == pytest.approx(0.0408514013749339)

    # A ranking that disagrees about how many there are is a caller bug, not
    # something to silently pad.
    with pytest.raises(ValueError, match="one to one"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_agreement_beats_enthusiasm() -> None:
    """Third by both outranks first by one. The reason RRF exists.

    1/63 + 1/63 = 0.031746 against 1/61 = 0.016393. Without this property
    fusion would just be two rankings interleaved, and the retriever that
    happened to be confident would win every disagreement.
    """
    fused = reciprocal_rank_fusion([["x", "y", "both"], ["p", "q", "both"]])

    assert fused[0][0] == "both"
    assert fused[0][1] == pytest.approx(0.031746031746031744)
    # Not a close thing: agreement at rank 3 is worth nearly twice a single
    # first place, which is what makes the property survive a k change.
    assert fused[0][1] > 1.9 * 0.01639344262295082

    # `x` and `p` are both rank 1 in one ranking and absent from the other, so
    # they tie exactly. The tie breaks on the id, ascending, so a rerun of the
    # same query cannot reorder them.
    assert [identifier for identifier, _ in fused[1:3]] == ["p", "x"]
    assert fused[1][1] == pytest.approx(fused[2][1])
