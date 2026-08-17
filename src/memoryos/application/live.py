"""Telling an open page that a background job finished.

**The problem is process boundaries, not transport.** The worker that normalizes,
embeds and extracts is a separate process from the API that has the browser's
connection, so "push the connection line when extraction finishes" needs the two
to talk. Everything else in this system that crosses that boundary goes through
Postgres — the queue is a table for exactly this reason — and so does this:
`NOTIFY` on commit, `LISTEN` in the API, no broker and no new dependency.

**SSE rather than WebSockets**, which the milestone asked for and which is right
for a reason worth writing down: this is server-to-client only. The browser never
sends anything up this channel, so a bidirectional protocol would buy a direction
nobody uses and cost an upgrade handshake, a framing layer, and a reconnect loop
somebody has to write. `EventSource` reconnects on its own and replays its last
event id without being asked.

**Replay is bounded and says so.** Reconnecting clients send `Last-Event-ID` and
get everything after it *that this process still holds* — a ring buffer, in
memory, lost on restart. That is a real limit rather than a hidden one: the
alternative is a durable event log, which is a table, a retention policy and a
migration for the sake of a few seconds of connection loss on a UI that can
recover by refetching. `Stream.resumable` tells a client which it got, so the
interface can refetch instead of quietly missing an update.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger(__name__)

# The Postgres channel every process agrees on. One channel rather than one per
# memory: `NOTIFY` payloads are limited to 8000 bytes and channels are a global
# namespace, so a channel per memory would be a name nobody could unsubscribe
# from and a listener that had to know its subscriptions before they existed.
CHANNEL = "memoryos_live"

# How many events a reconnecting client can catch up on. Sized for a laptop lid
# closing rather than for a day offline — see the module note on why this is not
# durable.
BUFFER_SIZE = 256

# How often the stream sends a comment when nothing has happened. Proxies and
# load balancers close idle connections at thirty to sixty seconds, and a stream
# that only speaks when there is news is a stream that gets cut during quiet.
HEARTBEAT_SECONDS = 15.0

# How long the listener waits before reconnecting. Fixed rather than exponential:
# a database that is briefly gone is the ordinary case, and the cost of missing
# notifications during it is a late connection line that the interface refetches
# anyway. Backing off would make recovery slower than the failure.
RECONNECT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class LiveEvent:
    """One thing that happened, addressed to whoever is watching.

    `id` is monotonic within one API process and is what `Last-Event-ID` carries.
    It is deliberately not a database id: the ordering that matters to a client is
    the order it would have received things in, which is a property of the stream
    rather than of the corpus.
    """

    id: int
    kind: str
    data: dict[str, Any]
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def render(self) -> str:
        """One SSE frame.

        `id:` first so a client that disconnects mid-frame does not record an id
        for an event it never fully received.
        """
        return (
            f"id: {self.id}\n"
            f"event: {self.kind}\n"
            f"data: {json.dumps({**self.data, 'at': self.at.isoformat()})}\n\n"
        )


class LiveBus:
    """Fan-out for one API process, fed by Postgres notifications.

    One instance per process, held on `app.state`. Subscribers are unbounded
    queues rather than callbacks, because a slow client must not be able to block
    the notification handler — and a bounded queue would mean choosing between
    dropping events for that client and blocking every other one.
    """

    def __init__(self, buffer_size: int = BUFFER_SIZE) -> None:
        self._next_id = 1
        self._buffer: list[LiveEvent] = []
        self._buffer_size = buffer_size
        self._subscribers: set[asyncio.Queue[LiveEvent]] = set()

    def publish(self, kind: str, data: dict[str, Any]) -> LiveEvent:
        """Record an event and hand it to every open stream."""
        event = LiveEvent(id=self._next_id, kind=kind, data=data)
        self._next_id += 1

        self._buffer.append(event)
        if len(self._buffer) > self._buffer_size:
            del self._buffer[: len(self._buffer) - self._buffer_size]

        for queue in self._subscribers:
            queue.put_nowait(event)
        return event

    def since(self, last_id: int | None) -> tuple[list[LiveEvent], bool]:
        """Buffered events after `last_id`, and whether that was answerable.

        The boolean is the honest half. `False` means the client asked to resume
        from an id this process no longer holds — a restart, or a gap longer than
        the buffer — and the interface must refetch rather than assume it has
        seen everything in between. Returning an empty list without saying so
        would be indistinguishable from "nothing happened".
        """
        if last_id is None:
            return [], True
        if not self._buffer:
            # Nothing buffered at all. Resumable only if the client is already up
            # to date; otherwise this process has restarted and lost the events.
            return [], last_id >= self._next_id - 1
        if last_id < self._buffer[0].id - 1:
            return [], False
        return [event for event in self._buffer if event.id > last_id], True

    @contextlib.contextmanager
    def subscribe(self) -> Any:
        """A queue fed every event published while it is open."""
        queue: asyncio.Queue[LiveEvent] = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    @property
    def subscribers(self) -> int:
        return len(self._subscribers)


async def stream(
    bus: LiveBus, *, last_event_id: int | None = None
) -> AsyncIterator[str]:
    """One client's SSE body: replay, then live, with heartbeats between.

    Subscribing happens *before* the replay is read, which is the ordering that
    makes this lossless across the handover: an event published between "read the
    buffer" and "start listening" would otherwise fall through the gap, and it
    would fall through it exactly when the system is busy.
    """
    with bus.subscribe() as queue:
        missed, resumable = bus.since(last_event_id)
        if not resumable:
            # Said rather than papered over. The client refetches; a silent empty
            # replay would let it believe it had missed nothing.
            gap = {"from": last_event_id, "reason": "events no longer buffered"}
            yield f"event: gap\ndata: {json.dumps(gap)}\n\n"
        for event in missed:
            yield event.render()
            # Anything already delivered from the buffer must not be delivered
            # again from the queue.
            last_event_id = event.id

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                # A comment, which SSE ignores and every proxy counts as traffic.
                yield ": keep-alive\n\n"
                continue
            if last_event_id is not None and event.id <= last_event_id:
                continue
            yield event.render()


def notify_sql(kind: str, data: dict[str, Any]) -> tuple[str, str]:
    """The channel and payload for `pg_notify`, as bound parameters.

    Returned rather than formatted into SQL, because a payload is user-adjacent
    data and `NOTIFY` takes its channel as an identifier that cannot be
    parameterised — `pg_notify(:channel, :payload)` is the function form that can
    be, and is what every caller here uses.
    """
    return CHANNEL, json.dumps({"kind": kind, **data})


def memory_ready(memory_id: UUID, job_type: str) -> dict[str, Any]:
    """The payload for "a job finished with this memory".

    Deliberately just the id and what finished. The connection line is *not*
    computed here and not carried in the notification: the API re-reads
    `chat.status` when it receives this, so what reaches the browser is the state
    of the corpus at delivery rather than at publication. Those differ whenever
    two jobs finish close together, and the second one is the one worth showing.
    """
    return {"memory_id": str(memory_id), "job_type": job_type}


async def listen(bus: LiveBus, dsn: str) -> None:
    """Feed the bus from Postgres, forever, reconnecting when the socket dies.

    A dedicated asyncpg connection rather than one from the pool, and that is
    forced rather than chosen: a listening connection is *held* for the process's
    lifetime and is unusable for queries while it waits, so borrowing one from the
    pool would remove it from circulation permanently and eventually deadlock
    every request.

    Reconnects on its own with a fixed delay. A database that is briefly gone is
    the ordinary case — a restart, a failover — and the consequence of missing
    notifications during it is that a page shows a connection line late, which the
    interface's own refetch already covers. Backing off exponentially would make
    the recovery slower than the failure.
    """
    # Imported here rather than at module scope, and untyped on purpose. asyncpg
    # ships no `py.typed`, so this is the one place in the codebase that reaches
    # past SQLAlchemy to the raw driver — because a `LISTEN` connection is held
    # rather than borrowed and has no SQLAlchemy equivalent that does not consume
    # a pool slot forever.
    import asyncpg  # type: ignore[import-untyped]

    while True:
        connection = None
        try:
            connection = await asyncpg.connect(dsn)

            def on_notify(
                _connection: object, _pid: int, _channel: str, payload: str
            ) -> None:
                try:
                    body = json.loads(payload)
                except json.JSONDecodeError:
                    logger.warning("live.unreadable_payload", payload=payload[:200])
                    return
                bus.publish(str(body.pop("kind", "unknown")), body)

            await connection.add_listener(CHANNEL, on_notify)
            logger.info("live.listening", channel=CHANNEL)
            # Nothing to do but stay alive; the callback does the work. Sleeping
            # forever rather than polling, because asyncpg dispatches
            # notifications on its own reader task.
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("live.listener_failed", exc_info=True)
            await asyncio.sleep(RECONNECT_SECONDS)
        finally:
            if connection is not None:
                with contextlib.suppress(Exception):
                    await connection.close()


def asyncpg_dsn(database_url: str) -> str:
    """`postgresql+asyncpg://…` as the DSN asyncpg itself takes.

    SQLAlchemy's URL carries its driver in the scheme and asyncpg rejects that,
    which is a two-line translation and exactly the kind of thing that is worth a
    named function rather than an inline replace at the one call site — there will
    be a second one the moment anything else needs a raw connection.
    """
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
