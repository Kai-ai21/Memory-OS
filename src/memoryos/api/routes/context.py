"""Context over HTTP, for an editor panel that cannot afford to wait.

**This endpoint never assembles.** M6.1 measured assembly at 0.9-1.8s warm and
18-21s cold, because a cold process loads an embedder and a cross-encoder — and
the caller here is a sidebar in an editor, redrawn every time the active tab
changes. Holding an HTTP request open for either number is the difference
between a panel somebody glances at and a panel somebody closes.

So it serves the cache and nothing else. On a miss it enqueues
`ASSEMBLE_CONTEXT` and answers `202` immediately; the worker builds it, and the
client's next poll finds it. That is Step 3's "cached-or-loading rather than
blocking", implemented as the only thing the endpoint can do rather than as a
mode it can be put into.

**A miss is what makes it worth building, and that is a change from M6.1.** That
milestone refused to precompute for `FILE_FOCUSED` because the event fires on
every file glanced at. This does precompute, for the same file, and the two are
consistent: an event is somebody's editor mentioning a path, while a request here
is a panel *open on that file with somebody looking at it*. The second is a much
better predictor than the first, and it is the one that pays for the second of
compute.

There is no POST. Assembly is enqueued as a side effect of a read, which is
unusual enough to say out loud — the alternative is a POST the client must
remember to call first, and a panel that renders nothing because it forgot.
"""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response, status
from pydantic import BaseModel

from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.application.context_engine import (
    DEFAULT_MAX_ITEMS,
    DEFAULT_TOKEN_BUDGET,
    ContextRequest,
    cache_key_for,
    corpus_fingerprint,
    read_cached,
)
from memoryos.container import Container
from memoryos.domain.context import ContextCategory
from memoryos.domain.jobs import JobSpec, JobType

router = APIRouter(tags=["context"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class ContextItemOut(BaseModel):
    position: int
    title: str
    category: ContextCategory
    text: str
    tokens: int
    # Which of M6.1's four sources proposed it, and where it ranked in each.
    # Two entries means two independent routes agreed, which is why it is as
    # high as it is — and it is the only thing on screen that lets a reader
    # judge an item rather than take it.
    sources: dict[str, int]
    # For the "open in the web UI" link. Exactly one of these is set.
    memory_id: UUID | None = None
    decision_id: UUID | None = None
    external_key: str | None = None


class ContextOut(BaseModel):
    focus: str
    # False when nothing was cached and a build was enqueued. The client shows
    # its loading state on this rather than on an error, because "not yet" and
    # "went wrong" need different faces.
    ready: bool
    token_budget: int
    tokens_used: int
    items: list[ContextItemOut]


@router.get("/context", response_model=ContextOut)
async def get_context(
    container: ContainerDep,
    response: Response,
    focus: Annotated[str, Query(min_length=1)],
    token_budget: Annotated[int, Query(ge=1, le=32_000)] = DEFAULT_TOKEN_BUDGET,
    max_items: Annotated[int, Query(ge=1, le=50)] = DEFAULT_MAX_ITEMS,
) -> ContextOut:
    """Cached context for a focus, or `202` and a build enqueued.

    The fingerprint is recomputed on every call — two indexed aggregates — so a
    sync that landed a second ago invalidates the entry rather than being served
    around. That cost is the price of the cache being safe rather than merely
    fast, and it is four orders of magnitude below the assembly it avoids.
    """
    sessions = container.database.session_factory
    request = ContextRequest(
        focus=focus, token_budget=token_budget, max_items=max_items
    )
    key = cache_key_for(request, await corpus_fingerprint(sessions))

    cached = await read_cached(sessions, key)
    if cached is not None:
        return ContextOut(
            focus=cached.focus,
            ready=True,
            token_budget=cached.token_budget,
            tokens_used=cached.tokens_used,
            items=[_item_out(item) for item in cached.items],
        )

    # Deduped on the cache key rather than on the focus, so two panels open at
    # different budgets queue two builds and two panels at the same budget queue
    # one. The key already encodes everything that changes the answer.
    async with sessions.begin() as session:
        await enqueue_in(
            session,
            JobSpec(
                job_type=JobType.ASSEMBLE_CONTEXT,
                payload={
                    "focus": focus,
                    "token_budget": token_budget,
                    "max_items": max_items,
                },
                dedupe_key=key,
                # Above the ingestion chain. Somebody is looking at a panel
                # waiting for this, and a normalize job for a file nobody has
                # opened is not more urgent than that.
                priority=10,
            ),
        )

    response.status_code = status.HTTP_202_ACCEPTED
    return ContextOut(
        focus=focus,
        ready=False,
        token_budget=token_budget,
        tokens_used=0,
        items=[],
    )


def _item_out(item: Any) -> ContextItemOut:
    return ContextItemOut(
        position=item.position,
        title=item.title,
        category=item.category,
        text=item.text,
        tokens=item.tokens,
        sources={source.value: rank for source, rank in item.sources.items()},
        memory_id=item.memory_id,
        decision_id=item.decision_id,
        external_key=item.external_key,
    )
