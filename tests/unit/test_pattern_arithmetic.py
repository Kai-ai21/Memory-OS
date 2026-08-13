"""The numbers every behavioural claim rests on, checked against hand computation.

Unit tests with no database, because that is the point of the arithmetic living
in `domain/`. A confidence formula nobody can reproduce is decoration with a
decimal point, and the way to keep it reproducible is to compute it by hand once
and pin the result.

The interval is the important half. It is what makes `patterns discover` stay
quiet on a small corpus, and a change that widened or narrowed it would change
what the system is willing to claim about somebody.
"""

from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from memoryos.domain.patterns import (
    DEFAULT_MIN_SUPPORT,
    MAX_CONFIDENCE,
    classify_confidence,
    is_emittable,
    is_miscalibrated,
    pattern_confidence,
    wilson_interval,
)
from memoryos.domain.values import ConfidenceHorizon, TimeProvenance

DECIDED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

# --------------------------------------------------------------------------
# The Wilson interval, worked by hand
# --------------------------------------------------------------------------


def test_the_interval_for_fourteen_of_fourteen_matches_hand_computation() -> None:
    """The case this corpus actually produced, computed on paper.

        p-hat = 1.0, z = 1.959964, z-squared = 3.841459
        denominator = 1 + 3.841459/14                      = 1.274390
        centre      = (1.0 + 3.841459/28) / 1.274390       = 0.892345
        margin      = (1.959964/1.274390) * sqrt(3.841459/784)
                    = 1.537954 * 0.070002                  = 0.107655
        interval    = 0.784689 .. 1.0  (clamped at 1)
    """
    interval = wilson_interval(14, 14)

    assert interval.observed == 1.0
    assert interval.low == pytest.approx(0.784689, abs=1e-6)
    assert interval.high == 1.0
    assert interval.n == 14


def test_the_interval_stays_finite_at_the_boundaries() -> None:
    """Why Wilson and not the textbook normal approximation.

    The normal interval's width is proportional to sqrt(p-hat(1-p-hat)), which is zero
    at 0 and at 1 — so fourteen assumptions that all held would report "100%,
    plus or minus nothing" and every calibration comparison would find a gap.
    Both boundaries have to have real width or this whole module reports noise.
    """
    for successes, n in ((0, 4), (4, 4), (0, 20), (20, 20)):
        interval = wilson_interval(successes, n)
        assert interval.high - interval.low > 0.15, (successes, n)


def test_a_larger_sample_narrows_the_interval() -> None:
    # The property the whole design leans on: more evidence makes a claim
    # possible, and there is no threshold that substitutes for it.
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(5, 10)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_an_empty_sample_supports_nothing() -> None:
    interval = wilson_interval(0, 0)
    assert interval.low == 0.0
    assert interval.high == 1.0
    # Which means every stated confidence is "consistent" with no data at all,
    # so nothing can be miscalibrated on zero observations.
    assert not is_miscalibrated(0.9, 0, 0)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_a_stated_confidence_inside_the_interval_is_not_miscalibration() -> None:
    """The gate that keeps this quiet, on the corpus's own numbers.

    Fourteen assumptions stated at a mean of 0.85, all of which held. The gap
    to 100% looks like underconfidence and is not evidence of it: a run of
    fourteen cannot distinguish being right 85% of the time from being right
    always.
    """
    assert not is_miscalibrated(0.846, 14, 14)
    # Six decisions, all worked, stated 0.89 — the same story with a wider
    # interval because the sample is smaller.
    assert not is_miscalibrated(0.892, 6, 6)
    # Four assumptions stated at 0.41, none of which held.
    assert not is_miscalibrated(0.413, 0, 4)


def test_a_stated_confidence_outside_the_interval_is_miscalibration() -> None:
    # Forty observations, ninety per cent claimed, half delivered. That gap the
    # sample can resolve, and it should be reported.
    assert is_miscalibrated(0.9, 20, 40)
    # And the mirror: a claim of near-ignorance from somebody who is nearly
    # always right.
    assert is_miscalibrated(0.2, 38, 40)


def test_the_same_gap_is_evidence_at_one_sample_size_and_not_another() -> None:
    """The heart of it, stated as a single comparison.

    An observed 100% against a stated 0.85 is not evidence at n=14 and is
    evidence at n=60. Nothing about the *gap* changed; only how much could have
    produced it by chance.
    """
    assert not is_miscalibrated(0.85, 14, 14)
    assert is_miscalibrated(0.85, 60, 60)


# --------------------------------------------------------------------------
# Confidence, worked by hand
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("supporting", "contradicting", "expected"),
    [
        # agreement 1.00 * sufficiency 3/6 = 0.50
        (3, 0, 0.50),
        # agreement 1.00 * sufficiency 1.00 = 1.00, capped
        (6, 0, MAX_CONFIDENCE),
        # agreement 6/8 = 0.75 * sufficiency 1.00
        (6, 2, 0.75),
        # agreement 4/7 = 0.571 * sufficiency 4/6 = 0.667
        (4, 3, 0.381),
        # No support is no confidence, not a weak pattern.
        (0, 5, 0.0),
    ],
)
def test_confidence_matches_hand_computation(
    supporting: int, contradicting: int, expected: float
) -> None:
    assert pattern_confidence(supporting, contradicting) == pytest.approx(
        expected, abs=1e-3
    )


def test_counter_evidence_lowers_confidence() -> None:
    """The property that makes counter-evidence more than decoration.

    A pattern that collected only agreeing cases would score identically
    whether or not the corpus contained counter-examples. Every contradicting
    decision has to move the number down, monotonically, or the search for them
    is theatre.
    """
    scores = [pattern_confidence(6, counter) for counter in range(0, 6)]
    assert scores == sorted(scores, reverse=True)
    assert all(later < earlier for earlier, later in pairwise(scores))


def test_confidence_never_reaches_certainty() -> None:
    # A rules-based detector over one person's own records, which that person
    # also wrote, cannot be certain however much evidence agrees.
    assert pattern_confidence(1000, 0) == MAX_CONFIDENCE


# --------------------------------------------------------------------------
# What may be emitted at all
# --------------------------------------------------------------------------


def test_support_below_the_minimum_is_never_emittable() -> None:
    assert not is_emittable(2, 0)
    assert is_emittable(3, 0)
    # And the bar moves with the argument rather than being baked in, so a
    # caller can be stricter.
    assert not is_emittable(3, 0, min_support=5)


def test_a_candidate_with_more_counter_evidence_is_not_a_pattern() -> None:
    """Not a weak pattern — not a pattern.

    Four supporting against three contradicting means the corpus holds almost
    as many counter-examples as examples. Emitting it with a low confidence
    would still put the claim in front of somebody, which is the failure this
    milestone is arranged against.
    """
    assert not is_emittable(4, 4)
    assert not is_emittable(4, 5)
    assert is_emittable(4, 3)


def test_the_default_minimum_is_three() -> None:
    # Written down as a test because it is the number the milestone turns on,
    # and a change to it should be deliberate rather than incidental.
    assert DEFAULT_MIN_SUPPORT == 3


# --------------------------------------------------------------------------
# Which confidences may be calibrated against at all
#
# Phase 5's retrospective called this the largest single defect in the phase: a
# calibration table built on confidences reconstructed after the fact measures
# hindsight and is indistinguishable from one built on foresight.
# --------------------------------------------------------------------------


def test_a_recovered_date_means_a_recovered_confidence() -> None:
    """The test that decides it on real data.

    `parsed` means the date was read out of a document; `filesystem` means it
    came from an mtime. If nobody wrote the date down at the time, nobody wrote
    the confidence down at the time either — both came from the same act of
    reconstruction.
    """
    for source in (TimeProvenance.PARSED, TimeProvenance.FILESYSTEM):
        assert (
            classify_confidence(
                0.8,
                decided_at=DECIDED_AT,
                decided_at_source=source,
                # Recorded the same second, so the window test cannot be what
                # rejects it.
                recorded_at=DECIDED_AT,
            )
            is ConfidenceHorizon.HINDSIGHT
        )


def test_a_declared_date_recorded_at_the_time_is_foresight() -> None:
    # The positive control. Without it every assertion here passes on a function
    # that returns HINDSIGHT unconditionally.
    assert (
        classify_confidence(
            0.8,
            decided_at=DECIDED_AT,
            decided_at_source=TimeProvenance.DECLARED,
            recorded_at=DECIDED_AT + timedelta(hours=2),
        )
        is ConfidenceHorizon.FORESIGHT
    )


def test_a_declared_date_entered_a_week_later_is_not() -> None:
    assert (
        classify_confidence(
            0.8,
            decided_at=DECIDED_AT,
            decided_at_source=TimeProvenance.DECLARED,
            recorded_at=DECIDED_AT + timedelta(days=7),
        )
        is ConfidenceHorizon.HINDSIGHT
    )


def test_an_outcome_already_observed_settles_it_whatever_the_dates_say() -> None:
    assert (
        classify_confidence(
            0.8,
            decided_at=DECIDED_AT,
            decided_at_source=TimeProvenance.DECLARED,
            recorded_at=DECIDED_AT,
            outcome_already_observed=True,
        )
        is ConfidenceHorizon.HINDSIGHT
    )


def test_no_confidence_is_unknown_rather_than_either_verdict() -> None:
    # There is no number, so there is nothing to have been early or late about.
    # A CHECK constraint pairs this with a null `confidence`.
    assert (
        classify_confidence(
            None,
            decided_at=DECIDED_AT,
            decided_at_source=TimeProvenance.DECLARED,
            recorded_at=DECIDED_AT,
        )
        is ConfidenceHorizon.UNKNOWN
    )
