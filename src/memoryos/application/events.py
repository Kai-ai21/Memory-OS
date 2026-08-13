"""Receiving events, and turning each one into work somebody else does.

**Dispatch enqueues jobs. It does not run handlers.** That is the whole design of
this module and every property worth having follows from it.

Running handlers inline inside `dispatch` would be four lines shorter and wrong
in four ways at once. The HTTP request that delivered the event would block until
every handler finished, so a slow handler becomes a slow plugin. A handler that
raised would abort the ones after it, so the order of a `subscribe` call would
decide whose work happened. A crash mid-dispatch would lose the remaining
handlers with no record that they were owed. And a retry would mean re-running
the handlers that already succeeded, because nothing would remember which had.

Enqueueing one job per handler answers all four with machinery M1.2 already
built: each job retries on its own schedule with its own backoff, dead-letters on
its own after five attempts, and carries its own error and traceback in a row
anybody can query. **One handler failing cannot block another, because they were
never in the same unit of work.**

The cost is that dispatch is not synchronous, so an event is *received* long
before it is *processed*, and the gap is the number `events stats` reports.
That number is the point of this milestone rather than a diagnostic: M6.1 puts
context behind these triggers, and context that arrives after the meeting starts
is worth nothing.

`processed_at` is written by the job, never by dispatch. An event whose jobs are
all still pending has had nothing done about it, and stamping it at enqueue would
make the latency measurement report the speed of an INSERT.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.domain.events import Event, EventKind, RateLimit
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobSpec, JobType

logger = structlog.get_logger(__name__)


class EventRateLimited(RuntimeError):
    """One source delivered more than its allowance. Nothing was stored."""

    def __init__(self, source: str, limit: RateLimit) -> None:
        super().__init__(
            f"source {source!r} has delivered {limit.limit} events in the last "
            f"{int(limit.window.total_seconds())}s; try again in "
            f"{limit.retry_after()}s"
        )
        self.source = source
        self.retry_after = limit.retry_after()


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------


class EventHandler(Protocol):
    """Something that wants to know when an event happens.

    `name` is part of the protocol rather than derived from the class, because
    it is written into a job payload and read back by a worker that may be a
    different process on a different build. A class name is a refactor away from
    changing; this is a string somebody chose.

    `kinds` is a frozenset rather than a single kind, because the interesting
    handlers in M6.1 will care about two or three of the four, and a handler
    registered once per kind would be dispatched to twice for an event that
    matched two.
    """

    @property
    def name(self) -> str: ...

    @property
    def kinds(self) -> frozenset[EventKind]: ...

    async def handle(self, event: Event) -> None: ...


class LogEventHandler:
    """Records that the event was dispatched, and does nothing else.

    **The point of M6.0 is that the plumbing works and can be watched working.**
    This handler is what makes that observable end to end: an event arrives, a
    job is enqueued, a worker claims it, this runs, `processed_at` is stamped,
    and `events stats` reports the latency between the two ends. Every part of
    that path is exercised without a single line of the thing M6.1 will actually
    do, which is what keeps this milestone's failures about the bus rather than
    about context assembly.

    Subscribed to every kind on purpose. A handler that quietly ignored a kind
    would leave those events unprocessed forever and make the pending count in
    `events stats` mean two different things at once.
    """

    name = "log"
    kinds = frozenset(EventKind)

    async def handle(self, event: Event) -> None:
        logger.info(
            "event.handled",
            handler=self.name,
            event_id=str(event.id),
            kind=event.kind.value,
            source=event.source,
            payload_keys=sorted(event.payload),
            delivery_lag_ms=(
                None
                if event.delivery_lag is None
                else int(event.delivery_lag.total_seconds() * 1000)
            ),
        )


# --------------------------------------------------------------------------
# The bus
# --------------------------------------------------------------------------


class EventBus:
    """Which handlers care about which kinds, and one job each when one fires.

    Deliberately not a pub/sub abstraction over a broker. It is a dictionary and
    an enqueue loop, because the durability, retry, backoff and dead-lettering
    that a broker would be brought in for are already in `jobs` — and buying them
    twice would mean two systems that can disagree about whether a handler ran.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, EventHandler] = {}

    def subscribe(self, handler: EventHandler) -> None:
        """Register a handler. Registering the same name twice is refused.

        Refused rather than replaced: two handlers under one name means a job
        payload that no longer identifies which code should run, and the failure
        would appear as a worker silently doing the wrong work rather than as an
        error at startup.
        """
        if handler.name in self._handlers:
            raise ValueError(f"a handler named {handler.name!r} is already subscribed")
        self._handlers[handler.name] = handler

    def handler(self, name: str) -> EventHandler | None:
        """The handler a job payload names, or None on a build that lacks it.

        None rather than a raise, so the *worker* decides what an unknown handler
        means. It already treats an unregistered job type as permanent rather
        than retrying forever, and this is the same situation one level down.
        """
        return self._handlers.get(name)

    def subscribers(self, kind: EventKind) -> list[EventHandler]:
        return [
            handler
            for handler in self._handlers.values()
            if kind in handler.kinds
        ]

    async def dispatch(self, session: AsyncSession, event: Event) -> list[UUID]:
        """Enqueue one job per subscribed handler, in the caller's transaction.

        **The session is a parameter and not a convenience.** M6.0 shipped this
        taking the `JobQueue` port, which opens its own transaction — so the
        event was committed by one transaction and its jobs by another, leaving a
        window in which the event exists and the work that processes it does not.
        A crash inside that window strands the event forever: nothing re-reads
        the table looking for events without jobs, and `events stats` reports it
        as "pending", which is exactly what a correctly-queued event looks like.

        That window is the one thing a jobs table exists to not have.
        `application/sync.py` says so in its own comment — "this is the entire
        reason the queue is a table rather than a broker: there is no window in
        which the memory exists and the job that processes it does not" — and
        M6.0, the first milestone with a new writer, opened one anyway.

        So this enqueues through `enqueue_in`, the function that exists for
        precisely this, and the caller commits both or neither.

        The dedupe key is `(event id, handler name)`, which makes a re-dispatch
        idempotent per handler rather than per event.

        An event nobody subscribes to returns an empty list and is *not* marked
        processed. That is the honest state — nothing was done about it — and the
        pending count in `events stats` is where it shows up.
        """
        handlers = self.subscribers(event.kind)
        if not handlers:
            logger.info(
                "event.no_subscribers", event_id=str(event.id), kind=event.kind.value
            )
            return []

        enqueued: list[UUID] = []
        for handler in handlers:
            job_id = await enqueue_in(
                session,
                JobSpec(
                    job_type=JobType.HANDLE_EVENT,
                    payload={"event_id": str(event.id), "handler": handler.name},
                    dedupe_key=f"{event.id}:{handler.name}",
                ),
            )
            if job_id is not None:
                enqueued.append(job_id)

        logger.info(
            "event.dispatched",
            event_id=str(event.id),
            kind=event.kind.value,
            handlers=len(handlers),
            jobs=len(enqueued),
        )
        return enqueued


# --------------------------------------------------------------------------
# Receiving
# --------------------------------------------------------------------------


DEFAULT_RATE_LIMIT = RateLimit(limit=60, window=timedelta(minutes=1))


@dataclass(frozen=True, slots=True)
class Received:
    """What happened to a delivery, including when nothing did."""

    event: Event
    # False when an unprocessed event with the same (kind, dedupe_key) already
    # existed. The row returned is the original one, so a client that retries
    # gets the id it was given the first time rather than a 409 to interpret.
    created: bool
    jobs: list[UUID] = field(default_factory=list)


async def receive(
    sessions: async_sessionmaker[AsyncSession],
    bus: EventBus,
    *,
    kind: EventKind,
    source: str,
    payload: dict[str, Any] | None = None,
    occurred_at: datetime | None = None,
    dedupe_key: str | None = None,
    rate_limit: RateLimit = DEFAULT_RATE_LIMIT,
) -> Received:
    """Store one event and dispatch it, or explain why neither happened.

    `occurred_at=None` becomes `received_at`, and the resulting equality is the
    provenance: it means the client did not say when. That is M1.1's rule
    adapted rather than abandoned — there is no `occurred_at_source` column
    because there is only one way an event can get a time, so the column would
    have exactly one non-default value.
    """
    now = datetime.now(UTC)
    await _enforce_rate_limit(sessions, source=source, limit=rate_limit, now=now)

    event = Event(
        id=new_id(),
        kind=kind,
        source=source,
        payload=payload or {},
        occurred_at=occurred_at or now,
        received_at=now,
        dedupe_key=dedupe_key,
    )

    stored, created, jobs = await _store_and_dispatch(sessions, bus, event)
    if not created:
        # The tenth keystroke. Not an error and not a new unit of work: the
        # first delivery's job is still pending, and re-dispatching would only
        # collide on the job queue's own dedupe key one layer down.
        logger.info(
            "event.deduplicated",
            event_id=str(stored.id),
            kind=stored.kind.value,
            dedupe_key=dedupe_key,
        )
        return Received(event=stored, created=False)

    return Received(event=stored, created=True, jobs=jobs)


async def _enforce_rate_limit(
    sessions: async_sessionmaker[AsyncSession],
    *,
    source: str,
    limit: RateLimit,
    now: datetime,
) -> None:
    """Count what this source has already delivered, and refuse past the line.

    Counted against stored rows rather than a token bucket held in memory. A
    bucket in the API process is per replica and resets on every deploy, while
    the queue it protects is shared by all of them — so two replicas would allow
    twice the limit and a restart would allow it again immediately.
    """
    async with sessions() as session:
        count = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(models.ExternalEvent)
                    .where(
                        models.ExternalEvent.source == source,
                        models.ExternalEvent.received_at > now - limit.window,
                    )
                )
            ).scalar_one()
        )
    if limit.exceeded(count):
        logger.warning("event.rate_limited", source=source, in_window=count)
        raise EventRateLimited(source, limit)


async def _store_and_dispatch(
    sessions: async_sessionmaker[AsyncSession], bus: EventBus, event: Event
) -> tuple[Event, bool, list[UUID]]:
    """Insert the event and enqueue its jobs, or neither.

    **One transaction, which is the whole point.** Either the event and every job
    that processes it are committed together, or nothing is. There is no state in
    which an event is stored with no work queued against it, so nothing needs to
    reconcile one later — and `events stats` reporting "pending" means one thing
    rather than two.

    The `flush()` is load-bearing and is not tidiness. Without it the INSERT is
    deferred to commit time, so the unique index would fire *after* the jobs were
    enqueued — and a duplicate keystroke would roll back having already done the
    work of deciding it was a duplicate. Flushing forces the constraint to
    arbitrate first.

    The collision is caught rather than checked for. A `SELECT` first would be a
    race: two deliveries of the same keystroke can both find nothing and both
    insert, and the partial unique index is what actually decides. Catching
    `IntegrityError` means the database arbitrates, which is the only arbiter
    that is right under concurrency.
    """
    row = models.ExternalEvent(
        id=event.id,
        kind=event.kind.value,
        source=event.source,
        payload=event.payload,
        occurred_at=event.occurred_at,
        received_at=event.received_at,
        dedupe_key=event.dedupe_key,
    )
    try:
        async with sessions.begin() as session:
            session.add(row)
            await session.flush()
            jobs = await bus.dispatch(session, event)
    except IntegrityError:
        existing = await _pending_with_key(sessions, event.kind, event.dedupe_key)
        if existing is None:
            # The index did not fire, so something else did. Nothing here can
            # make that a duplicate.
            raise
        return existing, False, []
    return event, True, jobs


async def _pending_with_key(
    sessions: async_sessionmaker[AsyncSession], kind: EventKind, dedupe_key: str | None
) -> Event | None:
    if dedupe_key is None:
        return None
    async with sessions() as session:
        row = (
            await session.execute(
                select(models.ExternalEvent).where(
                    models.ExternalEvent.kind == kind.value,
                    models.ExternalEvent.dedupe_key == dedupe_key,
                    models.ExternalEvent.processed_at.is_(None),
                )
            )
        ).scalar_one_or_none()
    return None if row is None else _to_event(row)


def _to_event(row: models.ExternalEvent) -> Event:
    return Event(
        id=row.id,
        kind=EventKind(row.kind),
        source=row.source,
        payload=dict(row.payload),
        occurred_at=row.occurred_at,
        received_at=row.received_at,
        processed_at=row.processed_at,
        dedupe_key=row.dedupe_key,
    )


async def load(sessions: async_sessionmaker[AsyncSession], event_id: UUID) -> Event | None:
    async with sessions() as session:
        row = await session.get(models.ExternalEvent, event_id)
    return None if row is None else _to_event(row)


async def mark_processed(
    sessions: async_sessionmaker[AsyncSession], event_id: UUID
) -> None:
    """Stamp `processed_at`, unconditionally, after a handler has run.

    Every handler job does this, so the stored value ends up being the last
    handler to finish — which is the number `events stats` should report, since
    the event is only fully dealt with when all of them have.

    Unconditional rather than "only if null", and the difference matters for what
    the latency means. First-writer-wins would report the time to the *fastest*
    handler, which is the most flattering number available and the least useful.
    """
    async with sessions.begin() as session:
        row = await session.get(models.ExternalEvent, event_id)
        if row is None:
            return
        row.processed_at = datetime.now(UTC)


# --------------------------------------------------------------------------
# Watching it work
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KindStats:
    kind: EventKind
    total: int
    processed: int
    # Mean seconds from `received_at` to `processed_at`, over processed events
    # only. None when nothing of this kind has been processed — an average over
    # zero rows reported as 0.0 would read as "instant".
    mean_latency_seconds: float | None
    # Mean seconds from `occurred_at` to `received_at`. Separated from the
    # latency above because a slow queue is this system's fault and a slow
    # delivery is the network's, and one number covering both blames whichever
    # component the reader already suspected.
    mean_delivery_lag_seconds: float | None

    @property
    def pending(self) -> int:
        return self.total - self.processed


@dataclass(frozen=True, slots=True)
class EventStats:
    by_kind: list[KindStats]

    @property
    def total(self) -> int:
        return sum(row.total for row in self.by_kind)

    @property
    def processed(self) -> int:
        return sum(row.processed for row in self.by_kind)

    @property
    def pending(self) -> int:
        return self.total - self.processed

    @property
    def mean_latency_seconds(self) -> float | None:
        """Weighted by how many events each kind actually processed.

        A plain mean of the per-kind means would let one processed
        `meeting_upcoming` outweigh two hundred `file_focused`, and the number
        M6.1 depends on would be dominated by whichever kind is rarest.
        """
        weighted = [
            (row.mean_latency_seconds, row.processed)
            for row in self.by_kind
            if row.mean_latency_seconds is not None and row.processed > 0
        ]
        if not weighted:
            return None
        total = sum(count for _, count in weighted)
        return sum(mean * count for mean, count in weighted) / total


async def stats(sessions: async_sessionmaker[AsyncSession]) -> EventStats:
    """Counts and latencies per kind, computed in the database.

    The latency is `AVG(processed_at - received_at)` over processed rows, in SQL
    rather than in Python, because the alternative is pulling every event to
    average four columns — and the table this reads is the one designed to grow
    fastest in the whole schema.
    """
    latency = func.avg(
        func.extract(
            "epoch", models.ExternalEvent.processed_at - models.ExternalEvent.received_at
        )
    )
    lag = func.avg(
        func.extract(
            "epoch", models.ExternalEvent.received_at - models.ExternalEvent.occurred_at
        )
    )
    stmt = (
        select(
            models.ExternalEvent.kind,
            func.count(),
            func.count(models.ExternalEvent.processed_at),
            latency,
            lag,
        )
        .group_by(models.ExternalEvent.kind)
        .order_by(models.ExternalEvent.kind)
    )
    async with sessions() as session:
        rows = list(await session.execute(stmt))

    return EventStats(
        by_kind=[
            KindStats(
                kind=EventKind(kind),
                total=int(total),
                processed=int(processed),
                mean_latency_seconds=None if mean is None else float(mean),
                mean_delivery_lag_seconds=None if delivery is None else float(delivery),
            )
            for kind, total, processed, mean, delivery in rows
        ]
    )


async def tail(
    sessions: async_sessionmaker[AsyncSession],
    *,
    kind: EventKind | None = None,
    limit: int = 20,
    since: datetime | None = None,
) -> list[Event]:
    """The most recent events, oldest first.

    Queried newest-first so the index does the limiting, then reversed, so a
    tail of twenty over a million rows reads twenty rows. Returned oldest-first
    because that is the order a log is read in.

    `since` is what makes `--follow` cheap: each poll asks only for what arrived
    after the last one it saw, rather than re-reading the same window and
    filtering client-side.
    """
    stmt = select(models.ExternalEvent).order_by(
        models.ExternalEvent.received_at.desc()
    ).limit(limit)
    if kind is not None:
        stmt = stmt.where(models.ExternalEvent.kind == kind.value)
    if since is not None:
        stmt = stmt.where(models.ExternalEvent.received_at > since)

    async with sessions() as session:
        rows = list((await session.execute(stmt)).scalars())
    return [_to_event(row) for row in reversed(rows)]


def build_default_bus(handlers: Sequence[EventHandler] | None = None) -> EventBus:
    """The bus a worker and an API process both build.

    Both build it from the same function on purpose: the API dispatches by
    handler *name* and the worker looks that name up, so two lists that could
    drift would show up as jobs no worker can run.
    """
    bus = EventBus()
    for handler in handlers if handlers is not None else [LogEventHandler()]:
        bus.subscribe(handler)
    return bus
