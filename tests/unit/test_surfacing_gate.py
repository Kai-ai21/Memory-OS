"""The gate's arithmetic, checked against worked examples rather than a corpus.

Every rule M6.3 is built on is decidable without a database: what clears the
bar, what counts as the same interruption, and how feedback moves the threshold.
The four properties the milestone asks for are asserted end to end against
Postgres in `tests/integration/test_surfacing.py`; these are the pieces those
tests are built out of, pinned where they can be reasoned about.

The most important one is `test_a_single_route_can_never_clear_the_bar`. The base
threshold is not a tuned number — it is the smallest multiple of what one ranking
can contribute that no single-source item can reach — and a later tidy-up that
lowered it would silently turn this feature into search with a louder voice.
"""

from datetime import UTC, datetime, timedelta

import pytest

from memoryos.domain.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from memoryos.domain.surfacing import (
    BASE_MULTIPLE,
    CEILING_MULTIPLE,
    DISMISSAL_WINDOW,
    FLOOR_MULTIPLE,
    REPEAT_WINDOW,
    SINGLE_ROUTE_BEST,
    PriorSurfacing,
    SurfaceReason,
    TopItem,
    context_hash,
    decide,
    names_the_focus,
    overlap,
    threshold_for,
)

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
DEFAULT = threshold_for(dismissed=0, acted_on=0)


def item(score: float, *, key: str = "memory:a", routes: int = 2) -> TopItem:
    return TopItem(key=key, title=key, score=score, routes=routes)


def prior(
    keys: tuple[str, ...],
    *,
    surfaced_ago: timedelta,
    dismissed_ago: timedelta | None = None,
) -> PriorSurfacing:
    return PriorSurfacing(
        context_hash=context_hash(keys),
        keys=keys,
        surfaced_at=NOW - surfaced_ago,
        dismissed_at=None if dismissed_ago is None else NOW - dismissed_ago,
    )


# --------------------------------------------------------------------------
# The threshold, and the property it exists to have
# --------------------------------------------------------------------------


def test_a_single_route_can_never_clear_the_bar() -> None:
    """**The reason the base is 1.8 and not a number somebody liked.**

    RRF scores are not on a scale that means anything on their own, with one
    exception: the most a single ranking can contribute is `1 / (k + 1)`, by
    putting an item first. Any threshold above that line is a structural
    requirement for a second route to have agreed — which is the difference
    between answering a question and volunteering something.

    Asserted against the real fusion function rather than against arithmetic
    repeated here, so a change to `reciprocal_rank_fusion` breaks this rather
    than leaving it asserting about a formula the system no longer uses.
    """
    only_route_first = reciprocal_rank_fusion([["memory:a", "memory:b"]])
    best_one_route_can_do = only_route_first[0][1]

    assert best_one_route_can_do == pytest.approx(SINGLE_ROUTE_BEST)
    assert best_one_route_can_do < DEFAULT

    decision = decide(
        top=item(best_one_route_can_do, routes=1),
        keys=["memory:a"],
        threshold=DEFAULT,
        recent=[],
        now=NOW,
    )
    assert decision.surface is False
    assert decision.reason is SurfaceReason.BELOW_THRESHOLD


def test_two_routes_near_the_top_do_clear_it() -> None:
    """The case the bar is calibrated for: agreement, not enthusiasm."""
    fused = reciprocal_rank_fusion(
        [["memory:a", "memory:b"], ["memory:c", "memory:a"]]
    )
    top_key, top_score = fused[0]

    assert top_key == "memory:a"
    assert top_score > DEFAULT

    decision = decide(
        top=item(top_score),
        keys=["memory:a", "memory:b"],
        threshold=DEFAULT,
        recent=[],
        now=NOW,
    )
    assert decision.surface is True
    assert decision.reason is SurfaceReason.CLEARED


def test_the_floor_holds_the_single_route_property_at_any_amount_of_praise() -> None:
    """No quantity of "this was useful" buys an item in on one route.

    The adaptation is allowed to lower the bar, and the floor is exactly one
    route's best score — so a focus with a long record of useful context still
    requires two routes to agree. A floor below that line would make the
    invariant a property of the *default* rather than of the system.
    """
    generous = threshold_for(dismissed=0, acted_on=100)

    assert generous == pytest.approx(FLOOR_MULTIPLE * SINGLE_ROUTE_BEST)
    # `decide` compares with `<=`, so an item at exactly the floor is refused.
    assert (
        decide(
            top=item(SINGLE_ROUTE_BEST, routes=1),
            keys=["memory:a"],
            threshold=generous,
            recent=[],
            now=NOW,
        ).surface
        is False
    )


def test_dismissals_raise_and_usefulness_lowers_asymmetrically() -> None:
    """Fast towards silence, slow away from it.

    Being wrong loudly costs more than being wrong quietly, so one dismissal
    moves the bar three times as far as one "useful" does. Three dismissals
    reach the ceiling; it takes nine good ones to walk the same distance back.
    """
    base = threshold_for(dismissed=0, acted_on=0)
    dismissed_once = threshold_for(dismissed=1, acted_on=0)
    useful_once = threshold_for(dismissed=0, acted_on=1)

    assert dismissed_once > base > useful_once
    assert (dismissed_once - base) == pytest.approx(3 * (base - useful_once))

    assert threshold_for(dismissed=10, acted_on=0) == pytest.approx(
        CEILING_MULTIPLE * SINGLE_ROUTE_BEST
    )
    assert base == pytest.approx(BASE_MULTIPLE * SINGLE_ROUTE_BEST)


def test_the_bar_is_much_higher_than_searchs() -> None:
    """Search returns anything a retriever found; this needs `k` to be irrelevant.

    Stated as a ratio because the milestone's requirement is comparative — "much
    higher than search's" — and the only honest way to compare a gate against a
    retriever with no threshold at all is against the weakest score a retriever
    will still return.
    """
    weakest_search_will_return = 1.0 / (DEFAULT_RRF_K + 30)
    assert DEFAULT / weakest_search_will_return > 2.5


# --------------------------------------------------------------------------
# What "the same context" means
# --------------------------------------------------------------------------


def test_the_same_items_in_a_different_order_are_the_same_context() -> None:
    assert context_hash(["b", "a", "c"]) == context_hash(["a", "b", "c"])


def test_similarity_is_overlap_rather_than_identity() -> None:
    """One item different is a different hash and the same interruption.

    Which is why suppression compares overlap and the hash is only the row's
    identity. A gate that suppressed on the hash would be defeated by MMR
    swapping the eleventh item.
    """
    shown = ("a", "b", "c", "d", "e")
    almost = ("a", "b", "c", "d", "f")

    assert context_hash(shown) != context_hash(almost)
    assert overlap(shown, almost) >= 0.6


def test_two_empty_contexts_are_not_the_same_interruption() -> None:
    """Zero, not one. An empty context is not an interruption at all, and
    letting two of them match would make the first empty assembly suppress
    every later one for that focus."""
    assert overlap([], []) == 0.0


@pytest.mark.parametrize(
    ("external_key", "focus", "expected"),
    [
        ("src/memoryos/application/search.py", "src/memoryos/application/search.py", True),
        # The watcher may be started inside a package, so either side may be the
        # tail of the other.
        ("src/memoryos/application/search.py", "application/search.py", True),
        ("application/search.py", "src/memoryos/application/search.py", True),
        # Anchored on the separator, or every file would match its own suffix.
        ("src/memoryos/application/research.py", "search.py", False),
        ("src/memoryos/application/search.py", "Weekly planning sync", False),
        (None, "anything", False),
    ],
)
def test_which_items_are_the_thing_already_open(
    external_key: str | None, focus: str, expected: bool
) -> None:
    assert names_the_focus(external_key, focus) is expected


# --------------------------------------------------------------------------
# Refusals, and the order they are reported in
# --------------------------------------------------------------------------


def test_a_context_of_only_the_focused_file_is_nothing_new() -> None:
    """`_top_item` returns None for it, and this is what the gate does then.

    Distinguished from an empty context, because the two mean different things:
    one is a corpus with nothing to say about this file and the other is a
    corpus that has only the file itself.
    """
    assert (
        decide(top=None, keys=["memory:a"], threshold=DEFAULT, recent=[], now=NOW).reason
        is SurfaceReason.NOTHING_NEW
    )
    assert (
        decide(top=None, keys=[], threshold=DEFAULT, recent=[], now=NOW).reason
        is SurfaceReason.NO_CONTEXT
    )


def test_below_threshold_is_reported_before_suppression() -> None:
    """A context that would not have been shown anyway was not *suppressed*.

    The distinction is not pedantry: `surfacing stats` reports suppression
    separately precisely so the reader can tell "the windows are doing work"
    from "the bar is high", and folding a refused context into the suppressed
    count would make those two numbers the same number.
    """
    keys = ["memory:a", "memory:b"]
    decision = decide(
        top=item(SINGLE_ROUTE_BEST, routes=1),
        keys=keys,
        threshold=DEFAULT,
        recent=[prior(tuple(keys), surfaced_ago=timedelta(minutes=5))],
        now=NOW,
    )
    assert decision.reason is SurfaceReason.BELOW_THRESHOLD


def test_a_dismissal_outranks_a_recent_showing_whatever_the_row_order() -> None:
    """Both windows are checked against every similar prior before either wins.

    A context shown twice and dismissed once is dismissed, and a suppression
    that depended on which row a query returned first would be one that stopped
    holding the day somebody changed an ORDER BY.
    """
    keys = ("memory:a", "memory:b", "memory:c")
    shown_recently = prior(keys, surfaced_ago=timedelta(minutes=10))
    dismissed_ages_ago = prior(
        keys, surfaced_ago=timedelta(days=3), dismissed_ago=timedelta(days=3)
    )

    for recent in ([shown_recently, dismissed_ages_ago], [dismissed_ages_ago, shown_recently]):
        decision = decide(
            top=item(DEFAULT * 2),
            keys=list(keys),
            threshold=DEFAULT,
            recent=recent,
            now=NOW,
        )
        assert decision.reason is SurfaceReason.DISMISSED


def test_the_windows_are_two_orders_of_magnitude_apart() -> None:
    """Not a tuning choice. Repeating something somebody explicitly refused is
    the fastest way to be ignored permanently, so the dismissal window is set by
    what a mistake costs rather than by a half-life."""
    assert DISMISSAL_WINDOW > REPEAT_WINDOW * 100


def test_a_refusal_still_carries_the_arithmetic() -> None:
    """The first question about a refusal is how close it came, and a decision
    that only said "no" could not answer it."""
    decision = decide(
        top=item(DEFAULT * 0.9),
        keys=["memory:a"],
        threshold=DEFAULT,
        recent=[],
        now=NOW,
    )
    assert decision.surface is False
    assert decision.margin < 0
    assert decision.top is not None
    assert decision.explanation
