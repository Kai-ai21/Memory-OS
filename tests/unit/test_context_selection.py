"""Which items fit, checked against hand-worked cases and no database.

Two of the milestone's four required properties live here, because both are
decisions about a list rather than about a corpus: an item that does not fit is
dropped whole rather than truncated, and no category may take more than its cap.
Testing either through the engine would test the database's ability to hold rows.

The third — that a memory found by two sources appears once and ranks higher — is
RRF's property rather than selection's, and is asserted in the integration file
against the real fusion.
"""

import pytest

from memoryos.domain.context import (
    DEFAULT_CATEGORY_SHARE,
    Candidate,
    ContextCategory,
    Rejection,
    category_of,
    cosine,
    maximal_marginal_relevance,
    select,
)
from memoryos.domain.values import MemoryKind


def item(
    key: str,
    *,
    tokens: int = 10,
    relevance: float = 0.05,
    category: ContextCategory = ContextCategory.PROSE,
    vector: tuple[float, ...] | None = None,
) -> Candidate:
    return Candidate(
        key=key,
        category=category,
        tokens=tokens,
        relevance=relevance,
        vector=vector,
    )


# --------------------------------------------------------------------------
# The budget: dropped whole, never truncated
# --------------------------------------------------------------------------


def test_an_item_over_the_remaining_budget_is_dropped_not_truncated() -> None:
    """M2.6's rule, and it matters more here.

    A truncated memory may lose the very thing that made it relevant, and
    nothing downstream can tell that it was cut — so the item leaves whole or
    stays whole. Nothing in this API can express a partial item, which is the
    strongest form of the guarantee: it is not enforced, it is unrepresentable.
    """
    selection = select(
        [
            item("a", tokens=60, relevance=0.09),
            item("b", tokens=60, relevance=0.08),
        ],
        token_budget=100,
        max_items=10,
    )

    assert selection.keys == ["a"]
    assert selection.tokens_used == 60
    assert selection.rejected == [("b", Rejection.OVER_BUDGET)]


def test_a_shorter_item_further_down_still_fits() -> None:
    """The loop continues past a rejection rather than stopping.

    Stopping at the first item that does not fit would leave 39 of 100 tokens
    unspent for no reason — and the budget exists to be spent, not to be
    survived.
    """
    selection = select(
        [
            item("big", tokens=90, relevance=0.09),
            item("huge", tokens=90, relevance=0.08),
            item("small", tokens=9, relevance=0.07),
        ],
        token_budget=100,
        max_items=10,
    )

    assert selection.keys == ["big", "small"]
    assert dict(selection.rejected) == {"huge": Rejection.OVER_BUDGET}


def test_an_item_with_no_text_is_rejected_rather_than_counted() -> None:
    # A zero-token item costs nothing and contributes nothing, so admitting it
    # would spend a slot in `max_items` on a blank line.
    selection = select([item("empty", tokens=0)], token_budget=100, max_items=5)

    assert selection.keys == []
    assert selection.rejected == [("empty", Rejection.EMPTY)]


def test_max_items_binds_before_the_budget_runs_out() -> None:
    # `category_share=1.0` so the cap cannot be what rejects anything. One rule
    # at a time: with the default share these ten same-category items would be
    # cut to one by the cap and this would silently stop testing `max_items`.
    selection = select(
        [item(str(index), tokens=1) for index in range(10)],
        token_budget=1000,
        max_items=3,
        category_share=1.0,
    )

    assert len(selection.chosen) == 3
    assert sum(1 for _, reason in selection.rejected if reason is Rejection.MAX_ITEMS) == 7


def test_a_zero_budget_is_refused_rather_than_returning_nothing() -> None:
    # Returning an empty context would look like "the corpus has nothing", which
    # is a different and much more alarming answer than "you asked for no room".
    with pytest.raises(ValueError, match="token_budget"):
        select([item("a")], token_budget=0, max_items=5)
    with pytest.raises(ValueError, match="max_items"):
        select([item("a")], token_budget=100, max_items=0)


# --------------------------------------------------------------------------
# Category caps
# --------------------------------------------------------------------------


def test_no_category_may_take_more_than_its_share() -> None:
    """Twelve code files is a worse answer than nine files and three decisions.

    MMR cannot see this: twelve files can be mutually dissimilar by cosine and
    still be one *sort* of thing. The cap is the rule that knows the difference,
    and it is applied during selection rather than as a filter afterwards, so
    the slot a rejected file would have taken goes to the next admissible item.
    """
    candidates = [
        item(f"code-{index}", category=ContextCategory.CODE, relevance=0.09 - index / 1000)
        for index in range(10)
    ] + [
        item("decision", category=ContextCategory.DECISION, relevance=0.01),
        item("note", category=ContextCategory.PROSE, relevance=0.005),
    ]

    selection = select(candidates, token_budget=10_000, max_items=8)

    chosen = set(selection.keys)
    assert sum(1 for key in chosen if key.startswith("code")) == 4  # 8 * 0.5
    # And the slots the cap freed went to the other categories rather than being
    # left empty.
    assert "decision" in chosen
    assert "note" in chosen
    assert any(reason is Rejection.CATEGORY_FULL for _, reason in selection.rejected)


def test_a_cap_that_would_round_to_zero_still_admits_one() -> None:
    """The category most likely to round away is the one worth having.

    `max_items=1` gives a cap of `int(0.5) == 0`, which would make every
    category unreachable and return an empty context from a corpus full of
    candidates.
    """
    selection = select(
        [item("a", category=ContextCategory.DECISION)], token_budget=100, max_items=1
    )

    assert selection.keys == ["a"]


def test_the_cap_scales_with_the_request() -> None:
    # A literal cap of five is most of a request for six and almost none of a
    # request for thirty. The failure it prevents is proportional, so it is too.
    code = [
        item(f"code-{index}", category=ContextCategory.CODE, relevance=0.09)
        for index in range(30)
    ]

    small = select(code, token_budget=10_000, max_items=4)
    large = select(code, token_budget=10_000, max_items=20)

    assert len(small.chosen) == 2
    assert len(large.chosen) == 10
    assert len(large.chosen) / 20 == pytest.approx(DEFAULT_CATEGORY_SHARE)


def test_an_unmapped_memory_kind_falls_to_other() -> None:
    # Deliberate: a new `MemoryKind` should not silently join whichever category
    # happens to be listed first, and `OTHER` has its own cap so it cannot flood.
    assert category_of(MemoryKind.CODE) is ContextCategory.CODE
    assert category_of(MemoryKind.NOTE) is ContextCategory.PROSE
    assert category_of(MemoryKind.BOOKMARK) is ContextCategory.OTHER


# --------------------------------------------------------------------------
# Diversity
# --------------------------------------------------------------------------


def test_the_first_pick_is_the_most_relevant() -> None:
    # Nothing is chosen yet, so the penalty is zero for everybody and MMR must
    # not reorder. A first pick that was not the top hit would mean the
    # diversity term is being applied against an empty set.
    chosen = maximal_marginal_relevance(
        [
            item("low", relevance=0.01),
            item("high", relevance=0.09),
            item("mid", relevance=0.05),
        ]
    )

    assert [row.key for row in chosen] == ["high", "mid", "low"]
    assert chosen[0].redundancy == 0.0


def test_a_near_duplicate_is_pushed_below_a_distinct_alternative() -> None:
    """The whole reason MMR is here.

    `twin` is more relevant than `other` and is nearly identical to the item
    already chosen. Pure top-k returns two paragraphs about the same thing; MMR
    trades that sliver of relevance for the perspective that is actually
    different.
    """
    chosen = maximal_marginal_relevance(
        [
            item("first", relevance=0.09, vector=(1.0, 0.0)),
            item("twin", relevance=0.08, vector=(0.99, 0.14)),
            item("other", relevance=0.05, vector=(0.0, 1.0)),
        ]
    )

    assert [row.key for row in chosen] == ["first", "other", "twin"]
    # And the number that caused it is recorded, so `--explain` can show it.
    assert chosen[2].redundancy > 0.9
    assert chosen[1].redundancy < 0.1


def test_lambda_seven_would_not_have_reordered_anything() -> None:
    """Why the default is 0.5, pinned so a future tidy-up cannot raise it.

    At 0.7 the penalty is capped at 0.3 while the relevance gap it must overcome
    is 0.7 times the normalised difference — so MMR reorders only items that
    were already near-tied, which is not the failure it was added to fix. This
    is the same fixture as the test above, and the near-duplicate wins.
    """
    chosen = maximal_marginal_relevance(
        [
            item("first", relevance=0.09, vector=(1.0, 0.0)),
            item("twin", relevance=0.08, vector=(0.99, 0.14)),
            item("other", relevance=0.05, vector=(0.0, 1.0)),
        ],
        lambda_=0.7,
    )

    assert [row.key for row in chosen] == ["first", "twin", "other"]


def test_lambda_one_ignores_diversity_entirely() -> None:
    # The control arm. At lambda=1 this is pure relevance ranking, which is what
    # makes the test above evidence about the diversity term rather than about
    # the ordering machinery.
    chosen = maximal_marginal_relevance(
        [
            item("first", relevance=0.09, vector=(1.0, 0.0)),
            item("twin", relevance=0.08, vector=(0.99, 0.14)),
            item("other", relevance=0.05, vector=(0.0, 1.0)),
        ],
        lambda_=1.0,
    )

    assert [row.key for row in chosen] == ["first", "twin", "other"]


def test_an_item_without_a_vector_is_never_penalised_and_never_penalises() -> None:
    """A decision has no embedding, and must not therefore win every round.

    Treating a missing vector as the zero vector would make every decision
    maximally novel — cosine 0 against everything — and at any lambda below 1
    they would take every slot. Treating it as identical would be worse: one
    unembedded item would suppress the entire rest of the list.
    """
    chosen = maximal_marginal_relevance(
        [
            item("a", relevance=0.09, vector=(1.0, 0.0)),
            # Distinct from `a`, so nothing about this test turns on redundancy
            # between the two embedded items.
            item("b", relevance=0.08, vector=(0.0, 1.0)),
            item("decision", relevance=0.02, vector=None),
        ]
    )

    by_key = {row.key: row for row in chosen}
    assert by_key["decision"].redundancy == 0.0
    # And it did not jump the queue on the strength of that: with nothing for
    # the novelty term to reward it for, relevance still decided.
    assert by_key["decision"].position == 3


def test_an_unembedded_item_does_outrank_a_perfect_duplicate() -> None:
    # The other half of the same rule, and the behaviour that changed when
    # lambda came down to 0.5. `b` is a pixel-identical repeat of `a`; a
    # decision nobody can measure similarity for is a better second item than a
    # second copy of the first, even at a fifth of the relevance.
    chosen = maximal_marginal_relevance(
        [
            item("a", relevance=0.09, vector=(1.0, 0.0)),
            item("b", relevance=0.08, vector=(1.0, 0.0)),
            item("decision", relevance=0.02, vector=None),
        ]
    )

    assert [row.key for row in chosen] == ["a", "decision", "b"]


def test_identical_relevance_leaves_ranking_to_novelty() -> None:
    # The normaliser maps a flat set to 1.0 for everybody, so the relevance term
    # is constant and MMR ranks purely on how different each item is. That is
    # the correct behaviour when relevance cannot distinguish anything.
    chosen = maximal_marginal_relevance(
        [
            item("a", relevance=0.05, vector=(1.0, 0.0)),
            item("twin", relevance=0.05, vector=(1.0, 0.0)),
            item("distinct", relevance=0.05, vector=(0.0, 1.0)),
        ]
    )

    assert [row.key for row in chosen][1] == "distinct"


def test_lambda_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="lambda"):
        maximal_marginal_relevance([item("a")], lambda_=1.5)


# --------------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------------


def test_cosine_is_computed_rather_than_assumed_normalised() -> None:
    """The embedder normalizes; relying on that is how M1.6.1 happened.

    A dot product would give 4.0 for these two, which is not a similarity and
    would swamp every relevance term in the MMR score.
    """
    assert cosine((2.0, 0.0), (2.0, 0.0)) == pytest.approx(1.0)
    assert cosine((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)
    assert cosine((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)


def test_a_missing_or_mismatched_vector_is_maximally_dissimilar() -> None:
    # 0.0 rather than 1.0. A missing embedding must not read as "the same as
    # everything", which would let one unembedded item suppress the whole list.
    assert cosine((), (1.0,)) == 0.0
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0
    assert cosine((1.0, 0.0), (1.0, 0.0, 0.0)) == 0.0
