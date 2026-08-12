"""M4.0's four claims, each about a way the layer could be quietly wrong.

Every one of them is a case where a plausible implementation returns a number
rather than an error: a null treated as a date, a bucket that swallows a month
boundary, an `as_of` that reads `is_current`, a gap threshold applied to the
wrong side of the comparison. None of those fail loudly, and all of them produce
a chart somebody would believe.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db.repositories import (
    SqlAlchemyArtifactRepository,
    SqlAlchemyMemoryRepository,
)
from memoryos.application import temporal
from memoryos.application.temporal import SourceScope
from memoryos.domain.entities import Memory, RawArtifact, Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash, MemoryKind, Period, TimeProvenance
from tests.integration.conftest import add_source

pytestmark = pytest.mark.integration

JANUARY = datetime(2026, 1, 1, tzinfo=UTC)
MARCH = datetime(2026, 3, 1, tzinfo=UTC)

ARTIFACT = RawArtifact(content_hash=ContentHash.of(b"temporal fixture"), byte_size=16)


@pytest.fixture
async def corpus(sessions: async_sessionmaker[AsyncSession], tmp_path: Path) -> Source:
    """A committed source and artifact to hang synthetic memories from.

    Committed rather than flushed, because every function under test opens its
    own session through the factory — which is the point of them being functions
    over a session factory — and would not see an open transaction's writes.
    """
    source = await add_source(sessions, "temporal", tmp_path)
    async with sessions.begin() as session:
        await SqlAlchemyArtifactRepository(session).add(ARTIFACT)
    return source


async def write(
    sessions: async_sessionmaker[AsyncSession],
    source: Source,
    key: str,
    *,
    occurred_at: datetime | None,
    ingested_at: datetime | None = None,
    kind: MemoryKind = MemoryKind.NOTE,
    title: str | None = None,
) -> Memory:
    """One memory, with both clocks set by the test rather than by the database."""
    memory = Memory(
        id=new_id(),
        source_id=source.id,
        external_key=key,
        content_hash=ARTIFACT.content_hash,
        kind=kind,
        title=title,
        occurred_at=occurred_at,
        occurred_at_source=(
            TimeProvenance.UNKNOWN if occurred_at is None else TimeProvenance.DECLARED
        ),
        ingested_at=ingested_at,
    )
    async with sessions.begin() as session:
        await SqlAlchemyMemoryRepository(session).add_version(memory)
    return memory


async def test_range_queries_exclude_undated_memories_rather_than_placing_them(
    sessions: async_sessionmaker[AsyncSession], corpus: Source
) -> None:
    """An unknown date is not evidence of any date.

    The tempting implementation coalesces `occurred_at` to `ingested_at` so that
    every memory lands somewhere. It would put the undated one below, which was
    written in 2019 and read this morning, on this morning's bar — and the chart
    would show a spike that no event produced. Asserted from both directions: it
    is absent from a range covering its ingestion time *and* from one covering
    the whole of recorded history.
    """
    inside = datetime(2026, 1, 15, tzinfo=UTC)
    await write(sessions, corpus, "dated.md", occurred_at=inside)
    await write(sessions, corpus, "outside.md", occurred_at=datetime(2025, 6, 1, tzinfo=UTC))
    await write(
        sessions,
        corpus,
        "undated.md",
        occurred_at=None,
        ingested_at=datetime(2026, 1, 20, tzinfo=UTC),
    )

    found = await temporal.memories_in_range(sessions, JANUARY, MARCH)
    assert [memory.external_key for memory in found] == ["dated.md"]

    # The range that contains the undated memory's *ingestion*, which is the
    # date a coalescing implementation would have given it.
    around_ingestion = await temporal.memories_in_range(
        sessions, datetime(2026, 1, 19, tzinfo=UTC), datetime(2026, 1, 21, tzinfo=UTC)
    )
    assert around_ingestion == []

    everything = await temporal.memories_in_range(
        sessions, datetime(1970, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC)
    )
    assert sorted(memory.external_key for memory in everything) == [
        "dated.md",
        "outside.md",
    ]

    # And the same exclusion in the aggregate, where it is easier to miss: the
    # buckets have to sum to the dated memories, not to the corpus.
    buckets = await temporal.activity_by_period(
        sessions,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2027, 1, 1, tzinfo=UTC),
        period=Period.MONTH,
    )
    assert sum(bucket.count for bucket in buckets) == 2

    # The profile is where the undated ones *are* counted, because a corpus with
    # a large unknown band needs to say so rather than to look smaller.
    profile = await temporal.provenance_profile(sessions)
    assert {band.provenance: band.count for band in profile} == {
        TimeProvenance.DECLARED: 2,
        TimeProvenance.UNKNOWN: 1,
    }


async def test_activity_by_period_buckets_across_a_month_boundary(
    sessions: async_sessionmaker[AsyncSession], corpus: Source
) -> None:
    """The last day of a month and the first of the next are two buckets.

    Two failures live here. A half-open interval applied at one end only makes
    midnight on the 1st belong to both months, so the bars sum to more than the
    corpus; and a month advanced by a fixed 30 days drifts off the first within a
    year, so the boundary itself moves. February is in the fixture because it is
    the month that catches the second one.
    """
    for day, key in (
        (datetime(2026, 1, 30, 12, tzinfo=UTC), "jan-30.md"),
        (datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC), "jan-31.md"),
        # Exactly the boundary instant, which is the one that gets counted twice.
        (datetime(2026, 2, 1, 0, 0, 0, tzinfo=UTC), "feb-01.md"),
        (datetime(2026, 2, 2, 8, tzinfo=UTC), "feb-02.md"),
    ):
        await write(sessions, corpus, key, occurred_at=day)

    months = await temporal.activity_by_period(sessions, JANUARY, MARCH, period=Period.MONTH)

    assert [(bucket.start, bucket.count) for bucket in months] == [
        (JANUARY, 2),
        (datetime(2026, 2, 1, tzinfo=UTC), 2),
    ]
    # Each bucket ends where the next begins, and the lengths are the calendar's
    # rather than a constant: 31 days then 28.
    assert [bucket.end for bucket in months] == [datetime(2026, 2, 1, tzinfo=UTC), MARCH]
    assert [bucket.end - bucket.start for bucket in months] == [
        timedelta(days=31),
        timedelta(days=28),
    ]
    assert sum(bucket.count for bucket in months) == 4

    # Daily, across the same boundary: the empty days between the 1st and the
    # 30th are present with a count of zero, because a histogram that omits them
    # draws a corpus with no shape as a corpus with no gaps.
    days = await temporal.activity_by_period(
        sessions,
        datetime(2026, 1, 29, tzinfo=UTC),
        datetime(2026, 2, 3, tzinfo=UTC),
        period=Period.DAY,
    )
    assert [(bucket.start.day, bucket.count) for bucket in days] == [
        (29, 0),
        (30, 1),
        (31, 1),
        (1, 1),
        (2, 1),
    ]


async def test_as_of_excludes_memories_ingested_after_the_query_time(
    sessions: async_sessionmaker[AsyncSession], corpus: Source
) -> None:
    """What the system knew then, reconstructed rather than read off `is_current`.

    Three claims, and the second is the one that makes this function worth
    having. A memory ingested afterwards is absent. A memory *revised*
    afterwards is present **as the version that existed at the time** — reading
    `is_current` would return today's text and call it last Tuesday's, which is
    the failure that makes past retrieval look reproducible when it is not. And
    a memory deleted afterwards is still there, because the tombstone updates a
    column rather than appending a row, and filtering on it naively would erase
    an item retroactively from every past view.
    """
    early = datetime(2026, 5, 1, tzinfo=UTC)
    query_time = datetime(2026, 5, 15, tzinfo=UTC)
    late = datetime(2026, 6, 1, tzinfo=UTC)

    revised = await write(
        sessions,
        corpus,
        "revised.md",
        occurred_at=early,
        ingested_at=early,
        title="first",
    )
    await write(sessions, corpus, "deleted.md", occurred_at=early, ingested_at=early)
    await write(sessions, corpus, "later.md", occurred_at=early, ingested_at=late)
    # A second version of the first item, ingested after the query time.
    await write(
        sessions, corpus, "revised.md", occurred_at=early, ingested_at=late, title="second"
    )

    async with sessions.begin() as session:
        repository = SqlAlchemyMemoryRepository(session)
        doomed = await repository.get_current(corpus.id, "deleted.md")
        assert doomed is not None
        await repository.tombstone(doomed.id, late)

    view = await temporal.as_of(sessions, query_time)

    assert view.query_time == query_time
    assert sorted(memory.external_key for memory in view.memories) == [
        "deleted.md",
        "revised.md",
    ]
    assert view.count == 2
    assert view.latest_ingested_at == early

    stored = {memory.external_key: memory for memory in view.memories}
    assert stored["revised.md"].title == "first"
    assert stored["revised.md"].version == 1
    assert stored["revised.md"].id == revised.id
    # The one whose deletion had not happened yet is present, tombstone and all.
    assert stored["deleted.md"].deleted_at == late

    # And afterwards, all three, with the revision superseding its first version.
    after = await temporal.as_of(sessions, datetime(2026, 6, 2, tzinfo=UTC))
    assert after.count == 2
    assert {memory.external_key for memory in after.memories} == {
        "revised.md",
        "later.md",
    }
    assert next(
        memory.title for memory in after.memories if memory.external_key == "revised.md"
    ) == "second"


async def test_find_gaps_reports_the_long_silence_and_not_the_short_one(
    sessions: async_sessionmaker[AsyncSession], corpus: Source
) -> None:
    """A gap has activity on both sides, and is longer than the threshold.

    The fixture contains one 39-day silence and one of four days, so a threshold
    of 30 days has to return exactly one of them. It also ends with 20 days of
    nothing after the newest memory, which is *not* a gap: it has activity on one
    side only, and reporting it would mean every source in every corpus is always
    abandoned as of its last write.
    """
    for day, key in (
        (1, "a.md"),
        (3, "b.md"),
        # Four days, which is a weekend and not an abandonment.
        (7, "c.md"),
        # Thirty-nine days, which is the one being looked for.
        (46, "d.md"),
        (48, "e.md"),
    ):
        await write(sessions, corpus, key, occurred_at=JANUARY + timedelta(days=day))

    scope = SourceScope(corpus.id)
    gaps = await temporal.find_gaps(sessions, scope, min_gap=timedelta(days=30))

    assert len(gaps) == 1
    gap = gaps[0]
    assert gap.duration == timedelta(days=39)
    assert gap.start == JANUARY + timedelta(days=7)
    assert gap.end == JANUARY + timedelta(days=46)
    assert (gap.before.external_key, gap.after.external_key) == ("c.md", "d.md")

    # Below the short gap, both show up; above the long one, neither does.
    assert len(await temporal.find_gaps(sessions, scope, min_gap=timedelta(days=3))) == 2
    assert await temporal.find_gaps(sessions, scope, min_gap=timedelta(days=40)) == []

    # A threshold equal to the gap includes it: `min_gap` is the shortest silence
    # worth reporting, not the shortest one it exceeds.
    exact = await temporal.find_gaps(sessions, scope, min_gap=timedelta(days=39))
    assert len(exact) == 1


async def test_out_of_order_measures_the_lag_and_only_in_one_direction(
    sessions: async_sessionmaker[AsyncSession], corpus: Source
) -> None:
    """A fifth test, for the fifth function, which the milestone did not list.

    It has a caller — `timeline` prints the backfill summary above every
    histogram — and a function with a caller and no test is the arrangement this
    project keeps paying for.

    The direction is the claim worth pinning. An absolute difference would lump
    backfilled content in with a source whose clock runs fast, and the two are
    not the same phenomenon: one says the corpus was assembled, the other says a
    timestamp is wrong. `future.md` below occurred *after* it was ingested and is
    excluded at every threshold.
    """
    ingested = datetime(2026, 6, 1, tzinfo=UTC)
    await write(
        sessions,
        corpus,
        "backfilled.md",
        occurred_at=datetime(2019, 3, 4, tzinfo=UTC),
        ingested_at=ingested,
    )
    await write(
        sessions,
        corpus,
        "recent.md",
        occurred_at=datetime(2026, 5, 30, tzinfo=UTC),
        ingested_at=ingested,
    )
    await write(
        sessions,
        corpus,
        "future.md",
        occurred_at=datetime(2026, 9, 1, tzinfo=UTC),
        ingested_at=ingested,
    )
    await write(sessions, corpus, "undated.md", occurred_at=None, ingested_at=ingested)

    # Ordered by lag, longest first, so the head is the most backfilled thing.
    everything = await temporal.out_of_order(sessions, timedelta(0))
    assert [memory.external_key for memory in everything] == [
        "backfilled.md",
        "recent.md",
    ]

    assert [
        memory.external_key
        for memory in await temporal.out_of_order(sessions, timedelta(days=365))
    ] == ["backfilled.md"]
    assert await temporal.out_of_order(sessions, timedelta(days=3000)) == []
