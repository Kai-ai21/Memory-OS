"""The two signals, at their boundaries.

Both are proxies, and neither test asserts that they are *right* — a test cannot
establish that recency predicts relevance, which is what `tune-weights` is for.
What can be asserted is that they behave the way the docstrings claim at the
points where a caller would be surprised: the half-life, the null date, and the
extremes that the database's CHECK constraint would reject.
"""

from datetime import UTC, datetime, timedelta

import pytest

from memoryos.domain.signals import (
    DEFAULT_HALF_LIFE_DAYS,
    UNKNOWN_RECENCY,
    importance_score,
    recency_score,
)

NOW = datetime(2026, 8, 10, tzinfo=UTC)


def test_recency_halves_at_one_half_life_and_declines_to_call_a_null_old() -> None:
    assert recency_score(NOW, NOW) == pytest.approx(1.0)
    assert recency_score(
        NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS), NOW
    ) == pytest.approx(0.5)
    assert recency_score(
        NOW - timedelta(days=2 * DEFAULT_HALF_LIFE_DAYS), NOW
    ) == pytest.approx(0.25)

    # **A null date is not evidence of age.** Scoring it 0.0 would rank every
    # undated item below every dated one, which the data does not support — the
    # same mistake M1.1's CHECK constraint exists to prevent. 0.5 places it at
    # one half-life: neither fresh nor stale.
    assert recency_score(None, NOW) == UNKNOWN_RECENCY == 0.5
    assert recency_score(None, NOW) == recency_score(
        NOW - timedelta(days=DEFAULT_HALF_LIFE_DAYS), NOW
    )

    # Decay never reaches zero, so there is no cutoff past which every old
    # document is equally worthless — the reason it is exponential.
    ancient = recency_score(NOW - timedelta(days=3650), NOW)
    assert 0.0 < ancient < 0.001

    # A future date scores above 1.0 rather than being clamped: that is a clock
    # problem and flattening it would hide it.
    assert recency_score(NOW + timedelta(days=180), NOW) == pytest.approx(2.0)

    with pytest.raises(ValueError, match="half_life_days"):
        recency_score(NOW, NOW, half_life_days=0)


@pytest.mark.parametrize(
    ("chunks", "versions"),
    [(0, 0), (0, 1), (1, 1), (1000, 50), (1000, 1), (0, 50), (10_000, 500)],
)
def test_importance_stays_inside_the_unit_interval(chunks: int, versions: int) -> None:
    """The column carries `CHECK (importance BETWEEN 0.0 AND 1.0)`.

    A score of 1.0000000000000002 fails an insert rather than a calculation, and
    the failure would surface a long way from here.
    """
    for last_edited in (None, NOW, NOW - timedelta(days=4000), NOW + timedelta(days=30)):
        value = importance_score(
            chunk_count=chunks,
            version_count=versions,
            last_edited_at=last_edited,
            now=NOW,
        )
        assert 0.0 <= value <= 1.0, (chunks, versions, last_edited, value)


def test_importance_is_ordered_by_the_evidence_it_claims_to_use() -> None:
    """Bounded is not enough; it also has to be monotone in each input.

    Saturating, though: the difference between 200 chunks and 400 is not
    information, and a linear count would make this a proxy for file size — the
    largest file in most repositories is a lock file.
    """
    base = {"version_count": 3, "last_edited_at": NOW, "now": NOW}
    small = importance_score(chunk_count=2, **base)  # type: ignore[arg-type]
    medium = importance_score(chunk_count=20, **base)  # type: ignore[arg-type]
    large = importance_score(chunk_count=1000, **base)  # type: ignore[arg-type]
    assert small < medium < large
    # Saturation: the jump from 2 to 20 chunks says more than 20 to 1000.
    assert (medium - small) > (large - medium)

    revised = importance_score(
        chunk_count=10, version_count=8, last_edited_at=NOW, now=NOW
    )
    once = importance_score(chunk_count=10, version_count=1, last_edited_at=NOW, now=NOW)
    assert revised > once

    stale = importance_score(
        chunk_count=10,
        version_count=3,
        last_edited_at=NOW - timedelta(days=720),
        now=NOW,
    )
    fresh = importance_score(
        chunk_count=10, version_count=3, last_edited_at=NOW, now=NOW
    )
    assert fresh > stale

    with pytest.raises(ValueError, match="never negative"):
        importance_score(chunk_count=-1, version_count=1, last_edited_at=NOW, now=NOW)
