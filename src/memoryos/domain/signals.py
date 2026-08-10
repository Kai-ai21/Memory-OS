"""Ranking signals that are properties of an item rather than of a query.

Pure functions. Nothing here knows what was searched for — these describe the
*document*, and fusion is where they meet the query.

Both are deliberately weak claims about the world, and both are labelled as
proxies rather than measurements. Real importance is behavioural — what somebody
actually returns to and acts on — and this system has no behaviour to observe
yet. That arrives in Phase 5. What is available now is the shape of the corpus,
and the honest thing is to use only that and say so.
"""

import math
from datetime import datetime

# Six months. Long enough that a working repository's files do not all collapse
# to the same score, short enough that a year-old note ranks visibly below a
# current one. Not tuned — `tune-weights` tunes how much recency counts, which
# is a different and more answerable question than how fast it should decay.
DEFAULT_HALF_LIFE_DAYS = 180.0

# What an undated item scores. Explicitly not 0.0: see `recency_score`.
UNKNOWN_RECENCY = 0.5

# Above this many chunks a document is simply "long" and the count stops
# carrying information. The log already compresses the tail; this bounds it.
_CHUNK_SATURATION = 64.0
# Likewise for revisions. A file edited fifty times is not five times more
# important than one edited ten times.
_VERSION_SATURATION = 12.0

# How the three pieces of evidence combine. Chunk count is weighted lowest
# because it is the one most contaminated by something other than importance —
# file length — even after the log.
_WEIGHT_SIZE = 0.3
_WEIGHT_REVISIONS = 0.45
_WEIGHT_FRESHNESS = 0.25


def recency_score(
    occurred_at: datetime | None,
    now: datetime,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """Exponential decay. 1.0 at `now`, 0.5 at one half-life.

    Exponential rather than linear. A linear ramp needs an arbitrary cutoff
    where the score hits zero, and everything past it is equally worthless — a
    two-year-old file and a ten-year-old one become indistinguishable, while the
    difference between one day and thirty days gets the same treatment as the
    difference between 700 days and 730. Decay has no cutoff and its derivative
    is largest where the differences actually matter.

    **A null date scores 0.5, not 0.0.** An unknown date is not evidence of age.
    Scoring it zero would rank every undated item below every dated one, which
    is a claim the data does not support — and it is the same mistake M1.1's
    CHECK constraint exists to prevent, where substituting `ingested_at` for a
    missing `occurred_at` would have quietly fabricated a history. 0.5 places an
    undated item at one half-life: neither fresh nor stale, which is the truth.

    Future timestamps score above 1.0 rather than being clamped. A file dated
    next week is a clock problem, and flattening it to 1.0 would hide that from
    whoever has to notice.
    """
    if occurred_at is None:
        return UNKNOWN_RECENCY
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be > 0, got {half_life_days}")

    age_days = (now - occurred_at).total_seconds() / 86400.0
    return float(2.0 ** (-age_days / half_life_days))


def importance_score(
    *,
    chunk_count: int,
    version_count: int,
    last_edited_at: datetime | None,
    now: datetime,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    """A proxy for importance, from observable evidence only. Always in 0..1.

    **This is not importance.** It is three things that correlate with it weakly
    and can be measured without asking anybody:

    * **Chunk count, log-scaled.** A document somebody kept adding to is more
      likely to matter than a stub. Linear would make this a proxy for file
      size, and the largest file in a repository is usually a lock file.
    * **Version count.** A file revised repeatedly is one somebody keeps coming
      back to. This is the strongest of the three and is weighted accordingly.
    * **Freshness of the last edit.** Same decay as `recency_score`, on the same
      reasoning.

    No model scores anything here, and nothing is inferred from content. A
    plausible-sounding number that came from a language model would be
    indistinguishable from a measured one downstream, and this column is
    consumed by a ranker that cannot tell the difference.

    Saturating rather than unbounded, because the column is declared `0.0..1.0`
    by a CHECK constraint, and because the difference between 200 chunks and 400
    is not information.
    """
    if chunk_count < 0 or version_count < 0:
        raise ValueError("chunk and version counts are counts, never negative")

    # log1p over a saturation point: 0 chunks -> 0.0, and the curve is steepest
    # where small documents differ from each other.
    size = math.log1p(min(float(chunk_count), _CHUNK_SATURATION)) / math.log1p(
        _CHUNK_SATURATION
    )
    # Version 1 is every file's starting point and says nothing, so the scale
    # starts at the first *re*-vision.
    revisions = math.log1p(
        min(float(max(version_count - 1, 0)), _VERSION_SATURATION)
    ) / math.log1p(_VERSION_SATURATION)
    freshness = recency_score(last_edited_at, now, half_life_days=half_life_days)

    score = (
        _WEIGHT_SIZE * size
        + _WEIGHT_REVISIONS * revisions
        + _WEIGHT_FRESHNESS * min(freshness, 1.0)
    )
    # The weights sum to 1 and every term is already bounded, so this clamp is
    # unreachable arithmetic — kept because the column has a CHECK constraint
    # and a float that lands at 1.0000000000000002 would fail the insert rather
    # than the calculation.
    return max(0.0, min(1.0, score))
