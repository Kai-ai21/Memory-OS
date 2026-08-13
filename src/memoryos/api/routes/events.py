"""The edge Phase 6 pushes through, and the two things it refuses.

One endpoint. It stores an event and dispatches it, and dispatch only enqueues —
so the request returns as soon as the rows are written rather than waiting for
any handler. A plugin that blocked on this would be a plugin that makes an editor
feel slow, which is the fastest possible way to have it uninstalled.

**An unknown kind is refused with the list of known ones, never stored.** A
schema validator would already reject it, with "value is not a valid enumeration
member" — true, and useless to whoever is writing the plugin. So `kind` arrives
as a plain string and `domain/events.parse_kind` names the four. Storing it
instead would be worse than either: a row nothing subscribes to sits unprocessed
forever, and a queue full of those cannot be told from a queue that is behind.

**A source past its rate limit gets 429 with `Retry-After`.** Deduplication is
the first defence and it is better, but it only helps a client that sets a dedupe
key; this is what stops the one that does not.

There is deliberately no `GET /events`. Reading the stream is an operator's job,
not a browser's — `memoryos events tail` and `events stats` — and an endpoint
that paginated the fastest-growing table in the schema would be the first thing
to make this API slow. M6.1 may need one; it can add it with a use case behind
it.
"""

from datetime import datetime, timedelta
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from memoryos.application import events
from memoryos.container import Container
from memoryos.domain.events import EventKind, RateLimit, UnknownEventKind, parse_kind

router = APIRouter(tags=["events"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class EventIn(BaseModel):
    # A plain string, validated in the handler, so the refusal can name every
    # kind that exists. See the module docstring.
    kind: str = Field(min_length=1)
    source: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    # Optional. Absent means the client is not asserting when it happened, and
    # the event gets `received_at` — which makes the two equal, and that equality
    # is the provenance.
    occurred_at: datetime | None = None
    # Optional, and the difference between one unit of work and ten. A client
    # that fires on every keystroke should send the file path here.
    dedupe_key: str | None = None


class EventOut(BaseModel):
    id: UUID
    kind: EventKind
    source: str
    occurred_at: datetime
    received_at: datetime
    # False when this collided with an unprocessed event carrying the same key.
    # The id returned is the original's, so a client that retries gets the id it
    # was given the first time rather than a 409 to interpret.
    created: bool
    # How many handler jobs were enqueued. Zero on a duplicate, and zero when
    # nothing subscribes to the kind — which is a fact worth returning rather
    # than an error, because the event is stored either way.
    jobs: int


@router.post("/events", response_model=EventOut, status_code=status.HTTP_202_ACCEPTED)
async def ingest_event(
    body: EventIn, container: ContainerDep, response: Response
) -> EventOut:
    """Accept one event, store it, and enqueue a job per subscribed handler.

    `202` rather than `201`: the event is recorded and the work is queued, and
    the work has not happened. A `201` would promise a created *result*, which is
    exactly the thing that has not been produced yet.
    """
    settings = container.settings
    try:
        kind = parse_kind(body.kind)
    except UnknownEventKind as exc:
        # 422 rather than 400: the request was well-formed and one field was
        # unprocessable, which is the distinction FastAPI already draws
        # everywhere else in this API.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    try:
        received = await events.receive(
            container.database.session_factory,
            container.event_bus(),
            kind=kind,
            source=body.source,
            payload=body.payload,
            occurred_at=body.occurred_at,
            dedupe_key=body.dedupe_key,
            rate_limit=RateLimit(
                limit=settings.event_rate_limit,
                window=timedelta(seconds=settings.event_rate_window_seconds),
            ),
        )
    except events.EventRateLimited as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            str(exc),
            # Sent as a header as well as in the body, because a well-behaved
            # client backs off on the header without parsing prose.
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc

    if not received.created:
        # The row already existed, so nothing was created. The body says which.
        response.status_code = status.HTTP_200_OK

    event = received.event
    assert event.occurred_at is not None  # set by `receive`, never null in the row
    assert event.received_at is not None
    return EventOut(
        id=event.id,
        kind=event.kind,
        source=event.source,
        occurred_at=event.occurred_at,
        received_at=event.received_at,
        created=received.created,
        jobs=len(received.jobs),
    )
