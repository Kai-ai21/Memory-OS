"""Capturing and exporting human verdicts on search results."""

from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from memoryos.application.judgements import (
    InvalidJudgement,
    JudgementInput,
    export_golden_set,
    record,
    summarise,
)
from memoryos.container import Container
from memoryos.domain.values import Verdict

router = APIRouter(tags=["judgements"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class JudgementIn(BaseModel):
    query_text: str = Field(min_length=1)
    # The durable identity of the judged item. Not the memory id, which a rebuild
    # replaces; see `adapters/db/models.QueryJudgement`.
    source_name: str = Field(min_length=1)
    external_key: str = Field(min_length=1)
    verdict: Verdict
    # Part of the identity, not a snapshot: null judges the memory, a number
    # judges that one chunk of it. Ordinals are 0-based and survive a rebuild.
    chunk_ordinal: int | None = Field(default=None, ge=0)
    # Snapshots of what the system said when the verdict was given.
    memory_id: UUID | None = None
    chunk_id: UUID | None = None
    rank_at_judgement: int | None = Field(default=None, ge=1)
    score_at_judgement: float | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


class JudgementOut(BaseModel):
    id: UUID


class QuerySummaryOut(BaseModel):
    query_text: str
    relevant: int
    not_relevant: int
    missing: int
    total: int
    last_judged_at: datetime


@router.post("/judgements", response_model=JudgementOut, status_code=status.HTTP_201_CREATED)
async def create_judgement(body: JudgementIn, container: ContainerDep) -> JudgementOut:
    """Record a verdict, replacing any previous one for the same query and item."""
    try:
        judgement_id = await record(
            container.database.session_factory,
            JudgementInput(
                query_text=body.query_text,
                source_name=body.source_name,
                external_key=body.external_key,
                verdict=body.verdict,
                chunk_ordinal=body.chunk_ordinal,
                memory_id=body.memory_id,
                chunk_id=body.chunk_id,
                rank_at_judgement=body.rank_at_judgement,
                score_at_judgement=body.score_at_judgement,
                filters=body.filters,
            ),
        )
    except InvalidJudgement as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return JudgementOut(id=judgement_id)


@router.get("/judgements", response_model=list[QuerySummaryOut])
async def list_judgements(container: ContainerDep) -> list[QuerySummaryOut]:
    """One row per judged query, so the golden set can be watched as it grows."""
    return [
        QuerySummaryOut(
            query_text=summary.query_text,
            relevant=summary.relevant,
            not_relevant=summary.not_relevant,
            missing=summary.missing,
            total=summary.total,
            last_judged_at=summary.last_judged_at,
        )
        for summary in await summarise(container.database.session_factory)
    ]


@router.get("/judgements/export")
async def export(container: ContainerDep) -> dict[str, Any]:
    """The golden set as JSON. M2.0's direct input.

    Untyped `dict` on purpose: this is a data interchange format whose consumer
    is an evaluation harness that does not exist yet, and pinning it into a
    response model now would be inventing a contract for it.
    """
    golden = await export_golden_set(
        container.database.session_factory, now=datetime.now(UTC)
    )
    return golden.as_dict()
