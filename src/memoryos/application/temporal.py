"""Queries over *when*, rather than over what the text says.

M1.1 stored `occurred_at` beside `ingested_at` and recorded in
`occurred_at_source` how each was derived. Six milestones later nothing had read
those columns, which is the only reason this milestone is a query layer and not a
migration: adding the column now would have been an afternoon, and recovering the
values it should have held would have been impossible — a source moves, a file is
rewritten, and the mtime that would have been last March becomes today.

Two clocks, and everything here turns on which one a question is asking about:

* **`occurred_at`** — when the thing happened in the world. `memories_in_range`,
  `activity_by_period` and `find_gaps` are about this one. It is the clock a
  person means by "when".
* **`ingested_at`** — when this system learned about it. `as_of` is about this
  one, and only this one. It is the clock a *debugger* means by "when".

`out_of_order` is the one function that reads both, because the distance between
them is itself the signal: content whose world-time is far behind its
ingestion-time was backfilled, and a corpus that was assembled rather than
accumulated looks different here from one that grew.

**Nulls are excluded, never defaulted.** `occurred_at IS NULL` means the date is
unknown, and the domain already refuses to let it mean anything else — `Memory`
raises unless a null timestamp is paired with `TimeProvenance.UNKNOWN`, and a
CHECK constraint says the same thing to every other writer. Substituting
`ingested_at` for a missing `occurred_at` would stack every undated memory onto
the day the corpus was read and invent a spike that no event produced. An unknown
date is not evidence of any date, so an undated memory is in no range and in no
bucket, and the counts here are counts of what is *known* to have happened then.

Everything is a function over a session factory rather than a method on a
service, because none of it holds state and none of it decides anything. The
ranking layer is not touched: this milestone makes the bitemporal data legible,
and M4.3 is where any of it is allowed to affect what retrieval returns.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from uuid import UUID

from sqlalchemy import ColumnElement, Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from memoryos.adapters.db import models
from memoryos.adapters.db.mappers import to_memory
from memoryos.domain.entities import Memory
from memoryos.domain.values import MemoryKind, Period, TimeProvenance

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bucket:
    """One period of the histogram, including the empty ones.

    `count` of zero is a real answer and is why these are generated rather than
    read straight out of a `GROUP BY`: a histogram assembled only from the
    periods that have rows draws no bar where nothing happened, which is exactly
    the shape a reader is looking for. The absence has to be plotted, not
    omitted.
    """

    start: datetime
    end: datetime
    count: int


@dataclass(frozen=True, slots=True)
class TemporalView:
    """The memory projection as it stood at one instant, by `ingested_at`.

    **What it is not:** a snapshot of retrieval. Chunks are deleted and rewritten
    in place by re-chunking, embeddings carry only the time they were last
    computed, and entity extraction records a version rather than a history — so
    the *text* the system held at a past instant is reconstructible from this and
    the ranking it would have produced is not. Stated here rather than discovered
    later, because a view that quietly claimed the second thing would be worse
    than no view at all.
    """

    query_time: datetime
    memories: tuple[Memory, ...]

    @property
    def count(self) -> int:
        return len(self.memories)

    @property
    def latest_ingested_at(self) -> datetime | None:
        """The most recent thing the system had learned. None if it knew nothing."""
        stamps = [m.ingested_at for m in self.memories if m.ingested_at is not None]
        return max(stamps) if stamps else None

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for memory in self.memories:
            counts[memory.kind.value] = counts.get(memory.kind.value, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


@dataclass(frozen=True, slots=True)
class Gap:
    """A stretch with activity on both sides of it and none inside.

    Bounded by the two memories that bracket it rather than by round dates,
    because the question this exists for — "when did I stop working on this" — is
    answered by *what was last touched* and *what came next*, and a gap reported
    as a bare pair of timestamps sends the reader back to the corpus to find out
    what they were.

    A gap is by construction interior: the stretch since the newest memory is not
    one, however long it has run. That silence has activity on one side only, so
    nothing distinguishes "abandoned" from "still going, nothing written lately"
    — and the open-ended version is the one that would fire on every source in
    every corpus, every time.
    """

    start: datetime
    end: datetime
    before: Memory
    after: Memory

    @property
    def duration(self) -> timedelta:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class ProvenanceBand:
    """How many memories got their date each way, and what range those cover.

    Printed above every histogram, and that placement is the argument for it
    existing. A timeline drawn from `filesystem` mtimes and one drawn from dates
    the source declared look identical and are worth completely different
    amounts: the first says when files were last written or checked out, which on
    a fresh clone is one afternoon regardless of when the work happened. Reported
    beside the chart so nobody reads the chart without it.
    """

    provenance: TimeProvenance
    count: int
    earliest: datetime | None
    latest: datetime | None


@dataclass(frozen=True, slots=True)
class SourceScope:
    """Gaps in one source's activity."""

    source_id: UUID


@dataclass(frozen=True, slots=True)
class EntityScope:
    """Gaps in the activity of memories that mention one entity.

    A merged-away id resolves to its winner rather than returning nothing: M3.2
    repoints mentions onto the winner, so the loser's id is a name for the same
    thing with none of the rows, and a caller holding one from before a merge
    should get an answer rather than an empty list.
    """

    entity_id: UUID


GapScope = SourceScope | EntityScope


# --------------------------------------------------------------------------
# Period arithmetic
#
# Pure, and separate from the query, because these are the functions the
# histogram's correctness actually rests on and they are worth being able to
# check without a database. They mirror Postgres' `date_trunc` exactly — weeks
# start Monday, months on the first — since one truncates the counts and the
# other generates the buckets those counts are placed into.
# --------------------------------------------------------------------------


def truncate(moment: datetime, period: Period) -> datetime:
    """`moment` snapped back to the start of its period, in UTC."""
    utc = _require_aware(moment, "moment").astimezone(UTC)
    midnight = utc.replace(hour=0, minute=0, second=0, microsecond=0)
    if period is Period.DAY:
        return midnight
    if period is Period.WEEK:
        return midnight - timedelta(days=midnight.weekday())
    return midnight.replace(day=1)


def advance(moment: datetime, period: Period) -> datetime:
    """The start of the period after `moment`'s.

    Calendar arithmetic rather than a fixed offset. Months are 28 to 31 days and
    `+30 days` would walk off the first of the month within a year; days and
    weeks are added as durations, which is deliberately *not* the same as adding
    24 hours to a local wall clock — everything here is UTC, where they agree.
    """
    start = truncate(moment, period)
    if period is Period.DAY:
        return start + timedelta(days=1)
    if period is Period.WEEK:
        return start + timedelta(days=7)
    year, month = divmod(start.month, 12)
    return start.replace(year=start.year + year, month=month + 1)


def bucket_starts(start: datetime, end: datetime, period: Period) -> list[datetime]:
    """Every period start covering `[start, end)`, in order.

    The first bucket is the one *containing* `start`, not `start` itself, so a
    range beginning mid-month is reported against the calendar month it falls in.
    A bucket labelled with a partial period would be a bar of a different width
    from every other bar in the chart.
    """
    if end <= start:
        return []
    starts: list[datetime] = []
    cursor = truncate(start, period)
    while cursor < end:
        starts.append(cursor)
        cursor = advance(cursor, period)
    return starts


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


async def provenance_profile(
    sessions: async_sessionmaker[AsyncSession], *, source_id: UUID | None = None
) -> list[ProvenanceBand]:
    """The temporal signal in the corpus, by how each date was derived.

    Every band the corpus actually has, ordered by size, with the undated band
    last whether or not it is empty — a zero there is the useful reading, and a
    row that disappears when the count reaches zero cannot be read at all.
    """
    stmt = (
        select(
            models.Memory.occurred_at_source,
            func.count(),
            func.min(models.Memory.occurred_at),
            func.max(models.Memory.occurred_at),
        )
        .where(*_current_predicates())
        .group_by(models.Memory.occurred_at_source)
    )
    if source_id is not None:
        stmt = stmt.where(models.Memory.source_id == source_id)

    async with sessions() as session:
        rows = (await session.execute(stmt)).all()

    bands = {
        TimeProvenance(provenance): ProvenanceBand(
            provenance=TimeProvenance(provenance),
            count=count,
            earliest=earliest,
            latest=latest,
        )
        for provenance, count, earliest, latest in rows
    }
    dated = sorted(
        (band for band in bands.values() if band.provenance is not TimeProvenance.UNKNOWN),
        key=lambda band: (-band.count, band.provenance.value),
    )
    unknown = bands.get(
        TimeProvenance.UNKNOWN,
        ProvenanceBand(TimeProvenance.UNKNOWN, 0, None, None),
    )
    return [*dated, unknown]


def observed_bounds(bands: Iterable[ProvenanceBand]) -> tuple[datetime, datetime] | None:
    """The span the dated memories cover, or None if none of them are dated."""
    earliest = [band.earliest for band in bands if band.earliest is not None]
    latest = [band.latest for band in bands if band.latest is not None]
    if not earliest or not latest:
        return None
    return min(earliest), max(latest)


async def memories_in_range(
    sessions: async_sessionmaker[AsyncSession],
    start: datetime,
    end: datetime,
    *,
    source_id: UUID | None = None,
    kinds: Iterable[MemoryKind] | None = None,
) -> list[Memory]:
    """Current memories that happened in `[start, end)`, oldest first.

    Half-open, because the ranges callers build are adjacent: closed at both ends
    means a memory at midnight belongs to two consecutive months, and a histogram
    whose bars sum to more than the corpus is a histogram nobody can reason from.
    """
    stmt = _in_range(_current(), start, end)
    if source_id is not None:
        stmt = stmt.where(models.Memory.source_id == source_id)
    kind_values = [kind.value for kind in kinds] if kinds is not None else None
    if kind_values:
        stmt = stmt.where(models.Memory.kind.in_(kind_values))

    stmt = stmt.order_by(models.Memory.occurred_at, models.Memory.external_key)
    async with sessions() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [to_memory(row) for row in rows]


async def activity_by_period(
    sessions: async_sessionmaker[AsyncSession],
    start: datetime,
    end: datetime,
    *,
    period: Period,
    source_id: UUID | None = None,
) -> list[Bucket]:
    """How many memories happened in each period of `[start, end)`.

    Counted in Postgres and densified in Python. The counting has to be in SQL —
    it is the one operation here that must not pull the corpus across the wire —
    and the empty buckets have to come from somewhere, since a `GROUP BY` cannot
    return a row for a group with no rows.
    """
    _require_aware(start, "start")
    _require_aware(end, "end")
    # Three-argument `date_trunc` (PG16+), so the truncation happens in UTC
    # rather than in whatever the session's TimeZone happens to be. The two-arg
    # form would make the histogram depend on the client's locale, which is a
    # bug that only appears on somebody else's machine.
    bucket = func.date_trunc(period.value, models.Memory.occurred_at, "UTC")
    stmt = _in_range(select(bucket, func.count()), start, end).where(*_current_predicates())
    if source_id is not None:
        stmt = stmt.where(models.Memory.source_id == source_id)

    async with sessions() as session:
        rows = (await session.execute(stmt.group_by(bucket))).all()

    counts = {moment: count for moment, count in rows}
    return [
        Bucket(
            start=moment,
            end=advance(moment, period),
            count=counts.get(moment, 0),
        )
        for moment in bucket_starts(start, end, period)
    ]


async def as_of(
    sessions: async_sessionmaker[AsyncSession], query_time: datetime
) -> TemporalView:
    """What the system knew at `query_time`, by `ingested_at`.

    **The one people skip and later need.** Without it, "why did this query
    return that last Tuesday" has no answer: the corpus has moved on, the
    ranking is reproducible only against the inputs it actually had, and a
    retrieval bug reported against a corpus that no longer exists is a bug that
    cannot be re-run.

    Reconstructed from the version history rather than read off `is_current`,
    which is the whole difficulty. `is_current` is a fact about *now*: at a past
    instant the current version of an item was whichever version had been
    ingested by then, and the row wearing the flag today may not have existed.
    So this takes the newest version per item at or before `query_time`, and
    keeps items whose deletion had not yet happened — `deleted_at` is a column
    the tombstone *updates*, so a memory deleted afterwards must still appear,
    and filtering it in the same pass that picks the version would have dropped
    it.
    """
    moment = _require_aware(query_time, "query_time")
    known = (
        select(models.Memory)
        .where(models.Memory.ingested_at <= moment)
        .distinct(models.Memory.source_id, models.Memory.external_key)
        .order_by(
            models.Memory.source_id,
            models.Memory.external_key,
            models.Memory.ingested_at.desc(),
            models.Memory.version.desc(),
        )
        .subquery()
    )
    version = aliased(models.Memory, known)
    stmt = (
        select(version)
        .where(or_(version.deleted_at.is_(None), version.deleted_at > moment))
        .order_by(version.ingested_at, version.external_key)
    )

    async with sessions() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return TemporalView(query_time=moment, memories=tuple(to_memory(row) for row in rows))


async def out_of_order(
    sessions: async_sessionmaker[AsyncSession],
    threshold: timedelta,
    *,
    source_id: UUID | None = None,
) -> list[Memory]:
    """Memories whose `occurred_at` precedes their `ingested_at` by more than `threshold`.

    Backfilled content, and a real fact about how a corpus was assembled: a
    system that has been running since the content was written has a small lag
    here, and one pointed at ten years of accumulated files has a large one. The
    threshold is the parameter because *some* lag is unavoidable — nothing is
    ingested at the instant it happens — so the interesting question is not
    whether the two differ but by how much.

    The reverse case, `occurred_at` in the future of `ingested_at`, is not
    reported here and is not the same phenomenon: it is a clock problem or a
    source that lies, and lumping it in with backfill under one absolute
    difference would hide both.
    """
    lag = models.Memory.ingested_at - models.Memory.occurred_at
    stmt = (
        select(models.Memory)
        .where(
            *_current_predicates(),
            models.Memory.occurred_at.is_not(None),
            lag > threshold,
        )
        .order_by(lag.desc(), models.Memory.external_key)
    )
    if source_id is not None:
        stmt = stmt.where(models.Memory.source_id == source_id)

    async with sessions() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [to_memory(row) for row in rows]


async def find_gaps(
    sessions: async_sessionmaker[AsyncSession],
    entity_or_source: GapScope,
    *,
    min_gap: timedelta,
) -> list[Gap]:
    """Stretches of `min_gap` or longer with activity on both sides and none inside.

    **The capability this milestone exists for.** "When did I abandon this" has
    no document to retrieve, because abandonment is not written down anywhere —
    it is the absence of anything after a point, and an absence is invisible to
    every retriever in this system. Vector search finds text that means what the
    question means, keyword search finds text that says it, and neither can
    return a document that was never written. Only aggregation over time can see
    a hole.

    Walked in Python over an ordered list rather than computed with a window
    function, and that is a size judgement rather than a style one: it is one
    pass over the memories in scope, the scope is a source or an entity, and the
    corpus this runs against is measured in hundreds. A `lag()` over the whole
    table would be the right answer at a scale this project does not have, and
    would put the definition of a gap into SQL where the test cannot read it.
    """
    if min_gap <= timedelta(0):
        raise ValueError(f"min_gap must be positive, got {min_gap!r}")

    stmt = select(models.Memory).where(
        *_current_predicates(), models.Memory.occurred_at.is_not(None)
    )
    if isinstance(entity_or_source, SourceScope):
        stmt = stmt.where(models.Memory.source_id == entity_or_source.source_id)
    else:
        stmt = stmt.where(
            models.Memory.id.in_(
                select(models.EntityMention.memory_id).where(
                    models.EntityMention.entity_id.in_(_resolved_entity(entity_or_source))
                )
            )
        )

    async with sessions() as session:
        rows = (await session.execute(stmt.order_by(models.Memory.occurred_at))).scalars().all()

    ordered = [to_memory(row) for row in rows]
    return [
        Gap(start=before.occurred_at, end=after.occurred_at, before=before, after=after)
        # The null checks are unreachable — the query above excludes them — and
        # they are how the type checker learns that, which is cheaper than an
        # ignore comment that would go on lying if the predicate were removed.
        for before, after in pairwise(ordered)
        if before.occurred_at is not None
        and after.occurred_at is not None
        and after.occurred_at - before.occurred_at >= min_gap
    ]


# --------------------------------------------------------------------------
# Shared predicates
# --------------------------------------------------------------------------


def _current_predicates() -> Sequence[ColumnElement[bool]]:
    """Eligibility, matching what search considers eligible.

    `is_current` and not deleted, the same pair `memory_predicates` applies to
    retrieval. A timeline drawn over superseded versions would count one file
    once per revision and call it activity, and one drawn over tombstoned items
    would show work on files that are gone. The deleted ones are still in the
    event log, which is where a question about what was removed belongs.
    """
    return (models.Memory.is_current.is_(True), models.Memory.deleted_at.is_(None))


def _current() -> Select[tuple[models.Memory]]:
    return select(models.Memory).where(*_current_predicates())


def _in_range[T: tuple[object, ...]](stmt: Select[T], start: datetime, end: datetime) -> Select[T]:
    """Add the half-open `occurred_at` window, nulls excluded explicitly.

    The `IS NOT NULL` is redundant against these comparisons — SQL's three-valued
    logic drops nulls from both — and it stays because the exclusion is a
    decision rather than a side effect. Somebody will one day make `start`
    optional, and on that day the comparison stops carrying it.
    """
    _require_aware(start, "start")
    _require_aware(end, "end")
    return stmt.where(
        models.Memory.occurred_at.is_not(None),
        models.Memory.occurred_at >= start,
        models.Memory.occurred_at < end,
    )


def _resolved_entity(scope: EntityScope) -> Select[tuple[UUID]]:
    """The id itself, or the winner it was merged into."""
    return select(func.coalesce(models.Entity.merged_into_id, models.Entity.id)).where(
        models.Entity.id == scope.entity_id
    )


def _require_aware(moment: datetime, name: str) -> datetime:
    """Reject naive datetimes at the boundary.

    The columns are `timestamptz` and every comparison here is against one, so a
    naive datetime is not a slightly-worse input — it is an instant with no
    defined value, and Postgres would resolve it against the session's time zone
    rather than refuse it. Refused here so the error names the argument.
    """
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise ValueError(f"{name} must be timezone-aware, got {moment!r}")
    return moment
