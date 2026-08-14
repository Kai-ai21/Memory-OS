"""What was volunteered, and the two clicks that judge it.

Three endpoints and no fourth one. There is deliberately **no POST that asks the
gate to run**: surfacing happens because an event arrived, and an endpoint that
could be called to produce an interruption would be a pull path wearing a push
path's clothes — the exact confusion this milestone is about. `GET /surfacing` is
a read of decisions already made.

`GET /surfacing` defaults to what a panel wants: things that were shown for one
focus and have not been judged yet. `?include_refused=true` is the other view,
and it is the one that answers *why didn't it show me anything* — every decision
in order with the reason and how close it came.

The two verdicts are separate routes rather than one taking a body. They are not
symmetric in what they mean: dismissing raises this focus's bar and suppresses
the same context for a month, marking useful lowers it and does nothing else. One
endpoint with a `verdict` field would hide that a month of suppression is on one
side of it.
"""

from datetime import timedelta
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from memoryos.application import surfacing
from memoryos.container import Container
from memoryos.domain.surfacing import EXPLANATIONS, SurfaceReason

router = APIRouter(tags=["surfacing"])

# How far back the panel's default view reaches.
#
# "Was this useful?" about something from last week is a question whose answer
# would be noise, so it stops being asked. The row stays and is still counted in
# the dismissal rate — expiring rows out of that denominator would improve the
# number by forgetting the interruptions nobody valued enough to rate.
PANEL_WINDOW = timedelta(hours=12)


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class SurfacedOut(BaseModel):
    id: UUID
    focus: str
    reason: SurfaceReason
    # The sentence, not just the enum. Three clients render this and three
    # phrasings of one rule is three chances for one of them to describe
    # behaviour that has changed.
    explanation: str
    score: float
    threshold: float
    top_key: str | None
    top_title: str | None
    item_count: int
    trigger_kind: str | None
    decided_at: str
    surfaced: bool
    # "dismissed", "useful", or null when nobody has said. Null is the state a
    # panel offers buttons for.
    verdict: str | None


@router.get("/surfacing", response_model=list[SurfacedOut])
async def list_surfacing(
    container: ContainerDep,
    focus: Annotated[str | None, Query()] = None,
    include_refused: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[SurfacedOut]:
    """Recent surfacing decisions. Shown-and-unjudged by default."""
    rows = await surfacing.recent(
        container.database.session_factory,
        focus=focus,
        surfaced_only=not include_refused,
        unrated_only=not include_refused,
        within=None if include_refused else PANEL_WINDOW,
        limit=limit,
    )
    return [_out(row) for row in rows]


@router.post("/surfacing/{surfacing_id}/dismiss", status_code=status.HTTP_204_NO_CONTENT)
async def dismiss(container: ContainerDep, surfacing_id: UUID) -> None:
    """Not worth having been shown. Raises this focus's bar for a month."""
    await _rate(container, surfacing_id, dismissed=True)


@router.post("/surfacing/{surfacing_id}/useful", status_code=status.HTTP_204_NO_CONTENT)
async def mark_useful(container: ContainerDep, surfacing_id: UUID) -> None:
    """Worth having been shown. Lowers this focus's bar, slowly."""
    await _rate(container, surfacing_id, dismissed=False)


async def _rate(container: Container, surfacing_id: UUID, *, dismissed: bool) -> None:
    sessions = container.database.session_factory
    try:
        found = (
            await surfacing.dismiss(sessions, surfacing_id)
            if dismissed
            else await surfacing.mark_useful(sessions, surfacing_id)
        )
    except surfacing.AlreadyRated as exc:
        # 409 rather than 200: the click did nothing, and a client that showed a
        # confirmation would be reporting a state change that did not happen.
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if not found:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"no surfaced context {surfacing_id}",
        )


def _out(row: surfacing.LoggedDecision) -> SurfacedOut:
    return SurfacedOut(
        id=row.id,
        focus=row.focus,
        reason=row.reason,
        explanation=EXPLANATIONS[row.reason],
        score=row.score,
        threshold=row.threshold,
        top_key=row.top_key,
        top_title=row.top_title,
        item_count=row.item_count,
        trigger_kind=row.trigger_kind,
        decided_at=row.decided_at.isoformat(),
        surfaced=row.surfaced_at is not None,
        verdict=row.rated,
    )
