"""The event bus: what is refused at the edge, and what reaches the queue.

Four properties this milestone exists to hold, and each is one test:

* an unknown kind is refused rather than stored,
* the partial unique index collapses ten deliveries into one unit of work,
* dispatch enqueues one job per subscribed handler,
* and one handler failing does not stop another running.

The last two are the reason dispatch enqueues rather than calling handlers
inline. Inline, the fourth property is false by construction — the first raise
ends the loop — and no amount of try/except around it recovers the retry,
backoff and dead-lettering that M1.2 already provides per job.

The bus is driven directly rather than through HTTP wherever the assertion is
about the bus. Two tests go through the API, because the refusal of an unknown
kind and the rate limit are properties of the *edge* and asserting them below it
would be asserting them somewhere they are not enforced.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import PostgresJobQueue
from memoryos.api.app import create_app
from memoryos.application.events import (
    ContextEventHandler,
    EventBus,
    EventRateLimited,
    LogEventHandler,
    build_default_bus,
    focus_of,
    load,
    mark_processed,
    receive,
    stats,
    tail,
)
from memoryos.application.jobs.handlers import make_event_handler
from memoryos.application.jobs.registry import JobContext
from memoryos.config import Settings
from memoryos.container import Container
from memoryos.domain.events import Event, EventKind, RateLimit, UnknownEventKind, parse_kind
from memoryos.domain.jobs import JobStatus, JobType, PermanentError, TransientError
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration


class RecordingHandler:
    """A handler that remembers what it was given, and optionally refuses it."""

    def __init__(
        self,
        name: str,
        *,
        kinds: frozenset[EventKind] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.name = name
        self.kinds = kinds if kinds is not None else frozenset(EventKind)
        self._raises = raises
        self.seen: list[UUID] = []

    async def handle(self, event: Event) -> None:
        self.seen.append(event.id)
        if self._raises is not None:
            raise self._raises


async def a_bus(sessions: async_sessionmaker[AsyncSession], *handlers: Any) -> EventBus:
    bus = EventBus()
    for handler in handlers:
        bus.subscribe(handler)
    return bus


async def count_jobs(
    sessions: async_sessionmaker[AsyncSession], job_type: JobType = JobType.HANDLE_EVENT
) -> int:
    async with sessions() as session:
        return int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(models.Job)
                    .where(models.Job.job_type == job_type.value)
                )
            ).scalar_one()
        )


async def run_job(
    sessions: async_sessionmaker[AsyncSession], bus: EventBus, job: Any
) -> None:
    """Run one claimed job through the real `HANDLE_EVENT` handler."""
    handler = make_event_handler(sessions, bus)
    async with sessions() as session:
        await handler(JobContext(job=job, session=session))


# --------------------------------------------------------------------------
# 1. An unknown kind is rejected, not stored
# --------------------------------------------------------------------------


def test_an_unknown_kind_is_refused_with_the_list_of_known_ones() -> None:
    """The message is the point, not the refusal.

    A schema validator would already reject this with "value is not a valid
    enumeration member", which is true and tells whoever is writing the plugin
    nothing about what to send instead.
    """
    with pytest.raises(UnknownEventKind) as caught:
        parse_kind("editor_closed")

    message = str(caught.value)
    assert "editor_closed" in message
    for member in EventKind:
        assert member.value in message


async def test_the_endpoint_refuses_an_unknown_kind_and_stores_nothing(
    client: AsyncClient, harness: Harness
) -> None:
    response = await client.post(
        "/events",
        json={"kind": "editor_closed", "source": "vscode", "payload": {}},
    )

    assert response.status_code == 422
    assert "editor_closed" in response.json()["detail"]
    # And nothing was written. A row nothing subscribes to would sit unprocessed
    # forever, and a queue full of those cannot be told from one that is behind.
    async with harness.sessions() as session:
        assert (
            await session.execute(
                select(func.count()).select_from(models.ExternalEvent)
            )
        ).scalar_one() == 0


# --------------------------------------------------------------------------
# 2. The dedupe index collapses repeats
# --------------------------------------------------------------------------


async def test_ten_deliveries_of_one_focus_produce_one_unit_of_work(
    harness: Harness,
) -> None:
    """The milestone's headline defence, driven the way a plugin would drive it."""
    handler = RecordingHandler("recorder")
    bus = await a_bus(harness.sessions, handler)

    results = [
        await receive(
            harness.sessions,
            bus,
            kind=EventKind.FILE_FOCUSED,
            source="vscode",
            payload={"path": "src/memoryos/cli.py"},
            dedupe_key="src/memoryos/cli.py",
        )
        for _ in range(10)
    ]

    assert sum(1 for result in results if result.created) == 1
    # Every delivery gets the *same* id back, so a client that retries is not
    # left holding an id nothing will ever process.
    assert len({result.event.id for result in results}) == 1
    async with harness.sessions() as session:
        assert (
            await session.execute(
                select(func.count()).select_from(models.ExternalEvent)
            )
        ).scalar_one() == 1
    assert await count_jobs(harness.sessions) == 1


async def test_the_same_key_is_new_work_once_the_first_is_processed(
    harness: Harness,
) -> None:
    """Why the index is partial.

    The same file focused again tomorrow is genuinely new work. An index over
    every row rather than over the unprocessed ones would refuse the second
    focus forever, which is a system that stops responding to you after the
    first time you open a file.
    """
    bus = await a_bus(harness.sessions, RecordingHandler("recorder"))
    first = await receive(
        harness.sessions,
        bus,
        kind=EventKind.FILE_FOCUSED,
        source="vscode",
        dedupe_key="src/memoryos/cli.py",
    )
    await mark_processed(harness.sessions, first.event.id)

    second = await receive(
        harness.sessions,
        bus,
        kind=EventKind.FILE_FOCUSED,
        source="vscode",
        dedupe_key="src/memoryos/cli.py",
    )

    assert second.created
    assert second.event.id != first.event.id


async def test_two_kinds_sharing_a_key_do_not_collide(harness: Harness) -> None:
    # The index is on `(kind, dedupe_key)`, and a path is a natural key for both
    # "opened" and "focused". Keyed on the string alone, opening a file would
    # swallow the focus event that follows it a millisecond later.
    bus = await a_bus(harness.sessions, RecordingHandler("recorder"))
    opened = await receive(
        harness.sessions, bus, kind=EventKind.EDITOR_OPENED,
        source="vscode", dedupe_key="src/memoryos/cli.py",
    )
    focused = await receive(
        harness.sessions, bus, kind=EventKind.FILE_FOCUSED,
        source="vscode", dedupe_key="src/memoryos/cli.py",
    )

    assert opened.created and focused.created
    assert await count_jobs(harness.sessions) == 2


async def test_events_without_a_key_are_never_deduplicated(harness: Harness) -> None:
    # `dedupe_key IS NOT NULL` is in the index predicate, because Postgres treats
    # nulls as distinct anyway — and a client that sends no key is asserting that
    # each delivery is its own event.
    bus = await a_bus(harness.sessions, RecordingHandler("recorder"))
    for _ in range(3):
        result = await receive(
            harness.sessions, bus, kind=EventKind.MANUAL, source="cli"
        )
        assert result.created
    assert await count_jobs(harness.sessions) == 3


# --------------------------------------------------------------------------
# 3. One job per subscribed handler
# --------------------------------------------------------------------------


async def test_dispatch_enqueues_one_job_per_subscribed_handler(
    harness: Harness,
) -> None:
    first = RecordingHandler("first")
    second = RecordingHandler("second")
    # Subscribed to a kind this event is not, so it must get nothing. Without it
    # this test would pass on a bus that ignores `kinds` entirely.
    elsewhere = RecordingHandler("elsewhere", kinds=frozenset({EventKind.MANUAL}))
    bus = await a_bus(harness.sessions, first, second, elsewhere)

    result = await receive(
        harness.sessions, bus, kind=EventKind.MEETING_UPCOMING, source="calendar"
    )

    assert len(result.jobs) == 2
    assert await count_jobs(harness.sessions) == 2
    async with harness.sessions() as session:
        payloads = [
            row.payload
            for row in (
                await session.execute(
                    select(models.Job).where(
                        models.Job.job_type == JobType.HANDLE_EVENT.value
                    )
                )
            ).scalars()
        ]
    assert {payload["handler"] for payload in payloads} == {"first", "second"}
    assert {payload["event_id"] for payload in payloads} == {str(result.event.id)}


async def test_dispatching_twice_does_not_double_the_work(harness: Harness) -> None:
    # The job dedupe key is `(event id, handler name)`, so a POST that stored its
    # event and then failed to enqueue can be dispatched again without doubling
    # the work of the handlers that were already queued.
    bus = await a_bus(harness.sessions, RecordingHandler("first"))
    result = await receive(
        harness.sessions, bus, kind=EventKind.MANUAL, source="cli"
    )

    async with harness.sessions.begin() as session:
        again = await bus.dispatch(session, result.event)

    assert again == []
    assert await count_jobs(harness.sessions) == 1


async def test_an_event_nobody_subscribes_to_is_stored_and_stays_pending(
    harness: Harness,
) -> None:
    """The honest state, and the one `events stats` has to show.

    Nothing was done about it. Marking it processed would make the pending count
    mean two things at once, and dropping it would lose the only record that a
    kind is arriving with nothing behind it.
    """
    bus = EventBus()

    result = await receive(
        harness.sessions, bus, kind=EventKind.FILE_FOCUSED, source="vscode"
    )

    assert result.created
    assert result.jobs == []
    assert await count_jobs(harness.sessions) == 0
    stored = await load(harness.sessions, result.event.id)
    assert stored is not None
    assert stored.processed_at is None


def test_two_handlers_cannot_share_a_name() -> None:
    # A job payload names the handler that should run it. Two under one name is a
    # payload that no longer identifies any code, and the failure would show up
    # as a worker silently doing the wrong work.
    bus = EventBus()
    bus.subscribe(RecordingHandler("log"))
    with pytest.raises(ValueError, match="already subscribed"):
        bus.subscribe(RecordingHandler("log"))


# --------------------------------------------------------------------------
# 4. A failing handler does not take another down with it
# --------------------------------------------------------------------------


async def test_one_handler_failing_does_not_prevent_another_from_running(
    harness: Harness,
) -> None:
    """The property that makes one job per handler worth the extra rows.

    Inline dispatch makes this false by construction: the first raise ends the
    loop, and whichever handler was registered second never runs. Here the two
    are separate units of work, so the failure is contained in one job's row —
    with its own error, its own traceback and its own retry schedule.
    """
    broken = RecordingHandler("broken", raises=TransientError("provider is down"))
    working = RecordingHandler("working")
    bus = await a_bus(harness.sessions, broken, working)

    result = await receive(
        harness.sessions, bus, kind=EventKind.MEETING_UPCOMING, source="calendar"
    )
    assert len(result.jobs) == 2

    queue = PostgresJobQueue(harness.sessions)
    failures = 0
    for _ in range(2):
        job = await queue.claim("worker-a", timedelta(minutes=1))
        assert job is not None
        try:
            await run_job(harness.sessions, bus, job)
        except TransientError:
            failures += 1
            await queue.reschedule(
                job.id, "worker-a", datetime.now(UTC) + timedelta(minutes=5),
                "provider is down", "traceback",
            )
        else:
            await queue.complete(job.id, "worker-a")

    assert failures == 1
    # The working handler ran, and it ran despite the other having raised.
    assert working.seen == [result.event.id]
    assert broken.seen == [result.event.id]
    # And the event is marked processed by the one that succeeded, while the
    # failed job waits on its own backoff rather than blocking anything.
    stored = await load(harness.sessions, result.event.id)
    assert stored is not None
    assert stored.processed_at is not None
    async with harness.sessions() as session:
        statuses = [
            row.status
            for row in (
                await session.execute(
                    select(models.Job).where(
                        models.Job.job_type == JobType.HANDLE_EVENT.value
                    )
                )
            ).scalars()
        ]
    assert sorted(statuses) == sorted([JobStatus.PENDING.value, JobStatus.SUCCEEDED.value])


async def test_a_job_for_an_event_that_no_longer_exists_fails_permanently(
    harness: Harness,
) -> None:
    """A full replay truncates `events` — they are operational and discardable.

    Any job still pending then refers to a row that no longer exists. Retrying
    that five times would turn a routine rebuild into a burst of dead letters, so
    it fails on the first attempt instead.
    """
    bus = await a_bus(harness.sessions, RecordingHandler("first"))
    result = await receive(harness.sessions, bus, kind=EventKind.MANUAL, source="cli")
    async with harness.sessions.begin() as session:
        await session.delete(
            await session.get_one(models.ExternalEvent, result.event.id)
        )

    queue = PostgresJobQueue(harness.sessions)
    job = await queue.claim("worker-a", timedelta(minutes=1))
    assert job is not None
    with pytest.raises(PermanentError, match="no event"):
        await run_job(harness.sessions, bus, job)


async def test_a_handler_this_build_does_not_have_fails_permanently(
    harness: Harness,
) -> None:
    # An older worker draining a queue a newer one wrote to. Permanent for the
    # same reason the worker treats an unregistered job type as permanent:
    # retrying cannot make code appear.
    bus = await a_bus(harness.sessions, RecordingHandler("newer"))
    result = await receive(harness.sessions, bus, kind=EventKind.MANUAL, source="cli")

    queue = PostgresJobQueue(harness.sessions)
    job = await queue.claim("worker-a", timedelta(minutes=1))
    assert job is not None
    older = EventBus()
    with pytest.raises(PermanentError, match="no handler"):
        await run_job(harness.sessions, older, job)
    # And the event is untouched, so nothing reports it as dealt with.
    stored = await load(harness.sessions, result.event.id)
    assert stored is not None
    assert stored.processed_at is None


# --------------------------------------------------------------------------
# The rate limit, and the stream
# --------------------------------------------------------------------------


async def test_a_source_past_its_limit_is_refused_and_others_are_not(
    harness: Harness,
) -> None:
    """Per source, because the point is to stop one misbehaving plugin.

    A global limit would let that plugin lock out every well-behaved client,
    which is the same outage with a different cause.
    """
    bus = await a_bus(harness.sessions, RecordingHandler("recorder"))
    limit = RateLimit(limit=3, window=timedelta(minutes=1))

    for _ in range(3):
        await receive(
            harness.sessions, bus, kind=EventKind.FILE_FOCUSED,
            source="noisy", rate_limit=limit,
        )
    with pytest.raises(EventRateLimited):
        await receive(
            harness.sessions, bus, kind=EventKind.FILE_FOCUSED,
            source="noisy", rate_limit=limit,
        )

    # The well-behaved client is unaffected.
    quiet = await receive(
        harness.sessions, bus, kind=EventKind.MANUAL, source="cli", rate_limit=limit
    )
    assert quiet.created


async def test_the_endpoint_answers_429_with_retry_after(
    settings: Settings, clean_database: None
) -> None:
    app = create_app(settings.model_copy(update={"event_rate_limit": 2}))
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        body = {"kind": "manual", "source": "cli", "payload": {}}
        for _ in range(2):
            assert (await client.post("/events", json=body)).status_code == 202
        response = await client.post("/events", json=body)

    assert response.status_code == 429
    # Sent as a header as well as prose, because a well-behaved client backs off
    # on the header without parsing English.
    assert response.headers["Retry-After"] == "60"


async def test_the_endpoint_stores_dispatches_and_reports_the_duplicate(
    client: AsyncClient, harness: Harness
) -> None:
    body = {
        "kind": "file_focused",
        "source": "vscode",
        "payload": {"path": "a.py"},
        "dedupe_key": "a.py",
    }
    first = await client.post("/events", json=body)
    second = await client.post("/events", json=body)

    assert first.status_code == 202
    assert first.json()["created"] is True
    # One job per subscribed handler, and the default bus subscribes two: M6.0's
    # log handler and M6.1's context handler, both of which take file focuses.
    assert first.json()["jobs"] == 2
    # 200 rather than 202: nothing was accepted for processing the second time.
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["id"] == first.json()["id"]
    assert await count_jobs(harness.sessions) == 2


async def test_an_absent_occurred_at_becomes_received_at(client: AsyncClient) -> None:
    """The equality is the provenance: it means the client did not say when."""
    response = await client.post(
        "/events", json={"kind": "manual", "source": "cli", "payload": {}}
    )

    body = response.json()
    assert body["occurred_at"] == body["received_at"]


async def test_a_client_that_asserts_a_time_keeps_it(
    client: AsyncClient, harness: Harness
) -> None:
    # M1.1's rule unchanged: an event delivered after a network hiccup happened
    # earlier than it arrived, and collapsing the two would re-date every event
    # to whenever the connection recovered.
    earlier = datetime.now(UTC) - timedelta(minutes=5)
    response = await client.post(
        "/events",
        json={
            "kind": "meeting_upcoming",
            "source": "calendar",
            "payload": {},
            "occurred_at": earlier.isoformat(),
        },
    )

    stored = await load(harness.sessions, UUID(response.json()["id"]))
    assert stored is not None
    assert stored.delivery_lag is not None
    assert stored.delivery_lag >= timedelta(minutes=4)


async def test_stats_separates_pending_from_processed_and_reports_latency(
    harness: Harness,
) -> None:
    bus = await a_bus(harness.sessions, LogEventHandler())
    processed = await receive(
        harness.sessions, bus, kind=EventKind.MANUAL, source="cli"
    )
    await receive(
        harness.sessions, bus, kind=EventKind.FILE_FOCUSED, source="vscode"
    )
    await mark_processed(harness.sessions, processed.event.id)

    report = await stats(harness.sessions)

    by_kind = {row.kind: row for row in report.by_kind}
    assert by_kind[EventKind.MANUAL].processed == 1
    assert by_kind[EventKind.MANUAL].pending == 0
    assert by_kind[EventKind.FILE_FOCUSED].processed == 0
    assert by_kind[EventKind.FILE_FOCUSED].pending == 1
    assert report.total == 2
    assert report.pending == 1
    # A real duration, not a placeholder.
    latency = by_kind[EventKind.MANUAL].mean_latency_seconds
    assert latency is not None
    assert latency >= 0.0
    # And nothing of a kind that has processed nothing, rather than 0.0 — a mean
    # over zero rows reported as zero reads as "instant".
    assert by_kind[EventKind.FILE_FOCUSED].mean_latency_seconds is None


async def test_the_tail_is_oldest_first_and_filterable(harness: Harness) -> None:
    bus = await a_bus(harness.sessions, RecordingHandler("recorder"))
    for index in range(3):
        await receive(
            harness.sessions, bus, kind=EventKind.MANUAL, source=f"cli-{index}"
        )
    await receive(harness.sessions, bus, kind=EventKind.FILE_FOCUSED, source="vscode")

    everything = await tail(harness.sessions, limit=10)
    manual = await tail(harness.sessions, kind=EventKind.MANUAL, limit=10)

    assert len(everything) == 4
    assert [event.source for event in manual] == ["cli-0", "cli-1", "cli-2"]
    # Oldest first, which is the order a log is read in.
    received = [event.received_at for event in everything]
    assert received == sorted(received)  # type: ignore[type-var]


async def test_the_default_bus_subscribes_the_log_handler(harness: Harness) -> None:
    # The API and the worker both build the bus from this function, because the
    # API writes a handler name into a payload the worker looks up. Two lists
    # that could drift would show up as jobs no worker can run.
    bus = build_default_bus()

    assert bus.handler("log") is not None
    assert {handler.name for handler in bus.subscribers(EventKind.MEETING_UPCOMING)} == {
        "log"
    }


async def test_the_worker_this_project_ships_can_run_a_handle_event_job(
    settings: Settings, harness: Harness
) -> None:
    """The registration this milestone shipped without, found by running it.

    The bus was wired into the API and into the container, and *not* into the
    registry the worker builds — so every dispatched job dead-lettered on its
    first attempt with "no handler registered for job_type 'handle_event'". Every
    unit test passed: they build their own registry, which is precisely the kind
    of test that cannot see a missing line in the composition root.

    So this asserts against the container's own registry rather than a fixture's.
    It is the only place the real wiring is visible.
    """
    container = Container.build(settings)
    try:
        assert JobType.HANDLE_EVENT.value in container.registry().registered_types()
    finally:
        await container.dispose()


async def test_an_event_is_never_stored_without_the_work_that_processes_it(
    harness: Harness,
) -> None:
    """The window M6.0 shipped, closed.

    That milestone stored the event in one transaction and enqueued its jobs in
    another, so a crash in between left an event nothing would ever process —
    permanently, because nothing re-reads the table looking for events without
    jobs, and `events stats` reports such an event as "pending", which is exactly
    what a correctly-queued event looks like.

    `application/sync.py` states the invariant M6.0 broke in its own comment:
    "this is the entire reason the queue is a table rather than a broker: there
    is no window in which the memory exists and the job that processes it does
    not, or the reverse."
    """

    class BrokenBus(EventBus):
        async def dispatch(self, session: Any, event: Event) -> list[UUID]:
            raise RuntimeError("the queue went away mid-dispatch")

    bus = BrokenBus()
    bus.subscribe(RecordingHandler("recorder"))

    with pytest.raises(RuntimeError, match="mid-dispatch"):
        await receive(harness.sessions, bus, kind=EventKind.MANUAL, source="cli")

    # Neither the event nor any job. The alternative — an event with no job — is
    # the state that cannot be detected after the fact.
    async with harness.sessions() as session:
        assert (
            await session.execute(
                select(func.count()).select_from(models.ExternalEvent)
            )
        ).scalar_one() == 0
    assert await count_jobs(harness.sessions) == 0


async def test_a_duplicate_rolls_back_before_enqueueing_anything(
    harness: Harness,
) -> None:
    # The `flush()` in `_store_and_dispatch` is what makes this true. Without it
    # the INSERT is deferred to commit, so the unique index fires *after* the
    # jobs are enqueued and the transaction rolls back having already decided the
    # duplicate was work.
    bus = await a_bus(harness.sessions, RecordingHandler("recorder"))
    first = await receive(
        harness.sessions, bus, kind=EventKind.FILE_FOCUSED,
        source="vscode", dedupe_key="a.py",
    )
    assert len(first.jobs) == 1

    second = await receive(
        harness.sessions, bus, kind=EventKind.FILE_FOCUSED,
        source="vscode", dedupe_key="a.py",
    )

    assert not second.created
    assert second.jobs == []
    assert await count_jobs(harness.sessions) == 1


# --------------------------------------------------------------------------
# M6.1's handler, and the precomputation policy
# --------------------------------------------------------------------------


async def test_context_is_precomputed_for_scheduled_triggers_only(
    harness: Harness,
) -> None:
    """The whole precomputation policy, asserted rather than described.

    A meeting is scheduled and an editor opening predicts a session; both are
    worth building ahead of being asked. A file focus fires on every file
    somebody glances at, and assembling for each burns compute continuously to
    produce output nobody asked for — which is the push-system failure Phase 6
    opened by naming.
    """
    assembled: list[str] = []

    async def assemble(focus: str, trigger: Event) -> None:
        assembled.append(focus)

    handler = ContextEventHandler(assemble)

    await handler.handle(
        Event(
            id=uuid4(),
            kind=EventKind.MEETING_UPCOMING,
            source="calendar",
            payload={"title": "phase 6 review"},
        )
    )
    await handler.handle(
        Event(
            id=uuid4(),
            kind=EventKind.EDITOR_OPENED,
            source="vscode",
            payload={"workspace": "Memory-OS"},
        )
    )
    await handler.handle(
        Event(
            id=uuid4(),
            kind=EventKind.FILE_FOCUSED,
            source="vscode",
            payload={"path": "src/memoryos/cli.py"},
        )
    )

    assert assembled == ["phase 6 review", "Memory-OS"]
    # And it is still subscribed to the one it skips, so the decision is visible
    # in one place with its reason rather than being an omission from a set.
    assert EventKind.FILE_FOCUSED in handler.kinds


async def test_a_payload_with_no_focus_assembles_nothing(harness: Harness) -> None:
    # A context assembled about "" is a context about the whole corpus, which is
    # the least useful possible answer and the most expensive to compute. Logged
    # rather than raised: a malformed payload is one client's mistake, and
    # dead-lettering would make it look like a broken queue.
    calls: list[str] = []

    async def assemble(focus: str, trigger: Event) -> None:
        calls.append(focus)

    await ContextEventHandler(assemble).handle(
        Event(
            id=uuid4(),
            kind=EventKind.MEETING_UPCOMING,
            source="calendar",
            payload={"organiser": "someone"},
        )
    )

    assert calls == []


def test_the_focus_is_read_from_the_payload_key_each_kind_uses() -> None:
    for kind, payload, expected in (
        (EventKind.FILE_FOCUSED, {"path": "a.py"}, "a.py"),
        (EventKind.EDITOR_OPENED, {"workspace": "  repo  "}, "repo"),
        (EventKind.MEETING_UPCOMING, {"subject": "standup"}, "standup"),
        (EventKind.MANUAL, {"focus": "chunking"}, "chunking"),
        (EventKind.MANUAL, {"unrelated": "x"}, ""),
    ):
        event = Event(id=uuid4(), kind=kind, source="test", payload=payload)
        assert focus_of(event) == expected
