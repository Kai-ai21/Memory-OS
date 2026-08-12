"""M4.0's temporal layer over HTTP.

Every handler here calls `application.temporal` and nothing else. That is the
same rule `/stats` and `/doctor` follow, for the same reason: two
implementations of "how many memories happened in August" is how a chart ends up
disagreeing with `memoryos timeline` and neither of them is obviously wrong.

**The provenance travels with every date.** `TimelineOut` carries the profile,
and `MemoryAtOut` carries `occurred_at_source` per row, because the UI's job in
this milestone is to stop a filesystem mtime and a date an email declared from
looking like the same claim. An endpoint that shipped the timestamps without the
provenance would make that impossible downstream, however careful the frontend
was.
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from memoryos.adapters.db import models
from memoryos.application import temporal
from memoryos.container import Container
from memoryos.domain.entities import Memory
from memoryos.domain.values import Period

router = APIRouter(tags=["timeline"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


# --------------------------------------------------------------------------
# Response models
# --------------------------------------------------------------------------


class BandOut(BaseModel):
    """One `occurred_at_source` band of the corpus."""

    provenance: str
    count: int
    earliest: datetime | None
    latest: datetime | None


class BucketOut(BaseModel):
    start: datetime
    end: datetime
    count: int
    # Summed to `count` by construction — one `GROUP BY` produces both — so the
    # stacked bar and the number above it cannot disagree.
    by_kind: dict[str, int] = Field(default_factory=dict)


class TimelineOut(BaseModel):
    """The histogram, plus what its dates are worth.

    The profile is part of the same response rather than a second endpoint on
    purpose. A caller that has to make two requests to find out whether the
    chart it just drew is built on declared dates or on mtimes is a caller that
    will draw the chart first.
    """

    start: datetime
    end: datetime
    period: Period
    total: int
    buckets: list[BucketOut]
    provenance: list[BandOut]


class MemoryAtOut(BaseModel):
    id: UUID
    external_key: str
    source_name: str
    kind: str
    title: str | None
    occurred_at: datetime | None
    occurred_at_source: str
    # Nullable because the domain entity's is, and passed through rather than
    # defaulted. Anything read from the database has one; inventing a value for
    # the case that cannot happen is how a bug becomes a plausible timestamp.
    ingested_at: datetime | None


class MemoriesAtOut(BaseModel):
    start: datetime
    end: datetime
    total: int
    memories: list[MemoryAtOut]


class GapEndOut(BaseModel):
    """What was active on one side of a gap."""

    id: UUID
    external_key: str
    kind: str
    occurred_at: datetime | None
    occurred_at_source: str


class GapOut(BaseModel):
    start: datetime
    end: datetime
    days: float
    source_name: str
    before: GapEndOut
    after: GapEndOut


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


@router.get("/timeline", response_model=TimelineOut)
async def get_timeline(
    container: ContainerDep,
    period: Period = Period.MONTH,
    start: Annotated[
        datetime | None,
        Query(alias="from", description="ISO instant. Defaults to the earliest dated memory."),
    ] = None,
    end: Annotated[
        datetime | None,
        Query(alias="to", description="ISO instant, exclusive. Defaults past the latest."),
    ] = None,
    source: Annotated[str | None, Query(description="Source name.")] = None,
) -> TimelineOut:
    """Activity per period, with the provenance profile it should be read against.

    The window defaults to what the corpus actually covers rather than to a
    fixed span like "the last year". A default window that missed the data would
    render an empty chart, and an empty chart is indistinguishable from an empty
    corpus.
    """
    sessions = container.database.session_factory
    source_id = await _resolve_source(container, source)

    bands = await temporal.provenance_profile(sessions, source_id=source_id)
    observed = temporal.observed_bounds(bands)

    if observed is None and (start is None or end is None):
        # Nothing dated. An empty timeline with the profile attached, so the
        # caller can say *why* it is empty rather than drawing nothing.
        now = datetime.now(UTC)
        return TimelineOut(
            start=start or now,
            end=end or now,
            period=period,
            total=0,
            buckets=[],
            provenance=[_band(band) for band in bands],
        )

    assert observed is not None
    window_start = _aware(start) if start else observed[0]
    window_end = _aware(end) if end else temporal.advance(observed[1], period)
    if window_end <= window_start:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"'from' ({window_start.isoformat()}) must be before "
            f"'to' ({window_end.isoformat()})",
        )

    buckets = await temporal.activity_by_period(
        sessions, window_start, window_end, period=period, source_id=source_id
    )
    return TimelineOut(
        start=window_start,
        end=window_end,
        period=period,
        total=sum(bucket.count for bucket in buckets),
        buckets=[
            BucketOut(
                start=bucket.start,
                end=bucket.end,
                count=bucket.count,
                by_kind=dict(bucket.by_kind),
            )
            for bucket in buckets
        ],
        provenance=[_band(band) for band in bands],
    )


@router.get("/memories/at", response_model=MemoriesAtOut)
async def memories_at(
    container: ContainerDep,
    date: Annotated[datetime, Query(description="ISO instant. The start of the window.")],
    window_days: Annotated[
        float,
        Query(gt=0, le=3660, description="Window length in days, forward from `date`."),
    ] = 1.0,
    source: Annotated[str | None, Query(description="Source name.")] = None,
    limit: Annotated[int, Query(gt=0, le=1000)] = 200,
) -> MemoriesAtOut:
    """The memories that happened in `[date, date + window_days)`.

    **The window runs forward from `date` rather than straddling it**, which is
    the less obvious reading of the name and the one that makes the timeline
    exact. A bucket the user clicked has a start and a length — 31 days for
    January, 28 for February — and a centred window could not express either
    without inventing a midpoint. Half-open at the far end, matching
    `memories_in_range`, so clicking two adjacent bars never returns the same
    memory twice.
    """
    start = _aware(date)
    end = start + timedelta(days=window_days)
    source_id = await _resolve_source(container, source)

    found = await temporal.memories_in_range(
        container.database.session_factory, start, end, source_id=source_id
    )
    names = await _source_names(container)
    return MemoriesAtOut(
        start=start,
        end=end,
        # The honest total, before the limit, so a truncated list says so.
        total=len(found),
        memories=[
            MemoryAtOut(
                id=memory.id,
                external_key=memory.external_key,
                source_name=names.get(memory.source_id, "—"),
                kind=memory.kind.value,
                title=memory.title,
                occurred_at=memory.occurred_at,
                occurred_at_source=memory.occurred_at_source.value,
                ingested_at=memory.ingested_at,
            )
            for memory in found[:limit]
        ],
    )


@router.get("/gaps", response_model=list[GapOut])
async def get_gaps(
    container: ContainerDep,
    min_days: Annotated[float, Query(gt=0, description="Shortest silence worth reporting.")] = 30.0,
    source: Annotated[str | None, Query(description="Source name.")] = None,
) -> list[GapOut]:
    """Stretches with activity either side and none during, per source.

    Per source rather than corpus-wide: a silence in one source that another was
    busy through is not a silence, and merging them would report a gap nobody
    experienced.
    """
    sessions = container.database.session_factory
    names = await _source_names(container)
    wanted = await _resolve_source(container, source)

    gaps: list[GapOut] = []
    for source_id, name in sorted(names.items(), key=lambda item: item[1]):
        if wanted is not None and source_id != wanted:
            continue
        for gap in await temporal.find_gaps(
            sessions, temporal.SourceScope(source_id), min_gap=timedelta(days=min_days)
        ):
            gaps.append(
                GapOut(
                    start=gap.start,
                    end=gap.end,
                    days=gap.duration.total_seconds() / 86400,
                    source_name=name,
                    before=_gap_end(gap.before),
                    after=_gap_end(gap.after),
                )
            )
    return sorted(gaps, key=lambda gap: gap.start)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _band(band: temporal.ProvenanceBand) -> BandOut:
    return BandOut(
        provenance=band.provenance.value,
        count=band.count,
        earliest=band.earliest,
        latest=band.latest,
    )


def _gap_end(memory: Memory) -> GapEndOut:
    return GapEndOut(
        id=memory.id,
        external_key=memory.external_key,
        kind=memory.kind.value,
        occurred_at=memory.occurred_at,
        occurred_at_source=memory.occurred_at_source.value,
    )


def _aware(moment: datetime) -> datetime:
    """A query parameter as an instant.

    A date with no offset means UTC, not the server's zone. The alternative
    makes the same URL select different rows on two machines, and the difference
    is a few hours — exactly the size nobody notices.
    """
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


async def _resolve_source(container: Container, name: str | None) -> UUID | None:
    if name is None:
        return None
    async with container.database.session_factory() as session:
        found = (
            await session.execute(select(models.Source.id).where(models.Source.name == name))
        ).scalars().first()
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no source named {name!r}")
    return found


async def _source_names(container: Container) -> dict[UUID, str]:
    async with container.database.session_factory() as session:
        return {
            row[0]: row[1]
            for row in await session.execute(select(models.Source.id, models.Source.name))
        }
