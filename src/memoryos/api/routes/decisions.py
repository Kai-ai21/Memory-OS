"""Decisions over HTTP: capture, read, edit, and the review queue.

**No route here writes a decision from a suggestion implicitly.** `POST
/decisions/suggestions/{id}/accept` exists and is the only path from the queue
into the table, so a client cannot accept by mistake and there is exactly one
place to look when asking whether something was reviewed. `GET
/decisions/suggestions` returns the source passage with every draft, because a
review UI that showed only the draft would be asking the reviewer to judge how
well it reads.

`POST /decisions/suggest` is deliberately absent. Running the extractor costs a
model call per passage, and an endpoint that spends money is one an over-eager
client can spend a daily quota on before anybody notices. It is a CLI command,
the same way `evolution` refuses to summarise on a GET.
"""

from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from memoryos.application import decision_suggest, decisions
from memoryos.container import Container
from memoryos.domain.values import (
    DecisionStatus,
    EvidenceRelation,
    SuggestionStatus,
    TimeProvenance,
)

router = APIRouter(tags=["decisions"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


class OptionIn(BaseModel):
    description: str = Field(min_length=1)
    rejected_because: str | None = None


class AssumptionIn(BaseModel):
    statement: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class EvidenceIn(BaseModel):
    # The durable identity, not a memory id: the same rule `/judgements` follows,
    # and for the same reason — a rebuild replaces every id.
    source_name: str = Field(min_length=1)
    external_key: str = Field(min_length=1)
    relation: EvidenceRelation = EvidenceRelation.INFORMED
    chunk_ordinal: int | None = Field(default=None, ge=0)


class DecisionIn(BaseModel):
    question: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    reasoning: str | None = None
    # At the time of deciding. There is no route that updates it afterwards.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_outcome: str | None = None
    # At least one is required by the use case, not by this model: the error
    # explains why a decision needs alternatives, and a 422 from pydantic would
    # say "too_short".
    options: list[OptionIn] = Field(default_factory=list)
    assumptions: list[AssumptionIn] = Field(default_factory=list)
    evidence: list[EvidenceIn] = Field(default_factory=list)
    decided_at: datetime | None = None


class DecisionEditIn(BaseModel):
    """What may be amended.

    `confidence` and `decided_at` are absent by design; see
    `application/decisions.DecisionEdit`. `options` and `assumptions` replace
    the existing sets when present.
    """

    question: str | None = None
    chosen: str | None = None
    reasoning: str | None = None
    expected_outcome: str | None = None
    status: DecisionStatus | None = None
    options: list[OptionIn] | None = None
    assumptions: list[AssumptionIn] | None = None


class DecisionOut(BaseModel):
    id: UUID


class OptionOut(BaseModel):
    id: UUID
    description: str
    was_chosen: bool
    rejected_because: str | None


class AssumptionOut(BaseModel):
    id: UUID
    statement: str
    confidence: float | None
    # Null until M5.2 evaluates it, and deliberately not `false`.
    held: bool | None
    evaluated_at: datetime | None


class EvidenceOut(BaseModel):
    id: UUID
    memory_id: UUID
    chunk_id: UUID | None
    source_name: str
    external_key: str
    chunk_ordinal: int | None
    relation: EvidenceRelation


class DecisionSummaryOut(BaseModel):
    id: UUID
    question: str
    chosen: str
    status: DecisionStatus
    confidence: float | None
    decided_at: datetime
    # Carried to the client so the UI can mark an undeclared date, exactly as
    # M4.1's timeline does. A list that rendered an mtime like an assertion
    # would be a chart that lies confidently.
    decided_at_source: TimeProvenance
    options: int
    assumptions: int
    evidence: int


class DecisionDetailOut(BaseModel):
    id: UUID
    question: str
    chosen: str
    reasoning: str | None
    confidence: float | None
    expected_outcome: str | None
    status: DecisionStatus
    decided_at: datetime
    decided_at_source: TimeProvenance
    created_at: datetime
    updated_at: datetime
    options: list[OptionOut]
    assumptions: list[AssumptionOut]
    evidence: list[EvidenceOut]


class SuggestionOut(BaseModel):
    id: UUID
    # The draft, as the model proposed it. Untyped on purpose: it is a proposal
    # rather than a record, and pinning it into a response model here would
    # invent a second contract for a shape `DecisionDraft` already owns.
    draft: dict[str, Any]
    # The passage the draft came from, shown beside it. This is the field that
    # makes accepting a judgement about evidence.
    source_text: str
    source_name: str
    external_key: str
    chunk_ordinal: int | None
    status: SuggestionStatus
    model_id: str
    suggested_at: datetime
    reviewed_at: datetime | None
    decision_id: UUID | None


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@router.post(
    "/decisions", response_model=DecisionOut, status_code=status.HTTP_201_CREATED
)
async def create_decision(body: DecisionIn, container: ContainerDep) -> DecisionOut:
    """Record one decision, with its options, assumptions and evidence."""
    try:
        decision_id = await decisions.record(
            container.database.session_factory,
            decisions.DecisionDraft(
                question=body.question,
                chosen=body.chosen,
                reasoning=body.reasoning,
                confidence=body.confidence,
                expected_outcome=body.expected_outcome,
                options=tuple(
                    decisions.OptionInput(
                        description=option.description,
                        rejected_because=option.rejected_because,
                    )
                    for option in body.options
                ),
                assumptions=tuple(
                    decisions.AssumptionInput(
                        statement=item.statement, confidence=item.confidence
                    )
                    for item in body.assumptions
                ),
                evidence=tuple(
                    decisions.EvidenceInput(
                        source_name=item.source_name,
                        external_key=item.external_key,
                        relation=item.relation,
                        chunk_ordinal=item.chunk_ordinal,
                    )
                    for item in body.evidence
                ),
            ),
            decided_at=body.decided_at or _now(),
            # A date somebody typed into a form is a date somebody declared,
            # which is what M1.1's `declared` means. Nothing on this path reads
            # a file, so no other provenance is reachable.
            decided_at_source=TimeProvenance.DECLARED,
        )
    except decisions.InvalidDecision as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except decisions.UnresolvedEvidence as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return DecisionOut(id=decision_id)


def _now() -> datetime:
    return datetime.now(UTC)


@router.get("/decisions", response_model=list[DecisionSummaryOut])
async def list_decisions(
    container: ContainerDep,
    decision_status: Annotated[
        DecisionStatus | None,
        Query(alias="status", description="open, settled or reversed"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[DecisionSummaryOut]:
    rows = await decisions.list_decisions(
        container.database.session_factory, status=decision_status, limit=limit
    )
    return [
        DecisionSummaryOut(
            id=row.id,
            question=row.question,
            chosen=row.chosen,
            status=row.status,
            confidence=row.confidence,
            decided_at=row.decided_at,
            decided_at_source=row.decided_at_source,
            options=row.options,
            assumptions=row.assumptions,
            evidence=row.evidence,
        )
        for row in rows
    ]


# Registered before `/decisions/{decision_id}`, because FastAPI matches in
# registration order rather than by specificity: the parameterised route would
# otherwise claim `/decisions/suggestions` and answer 422 for a path that was
# never meant to be a UUID. The same trap `app.py` documents for `/memories/at`.
@router.get("/decisions/suggestions", response_model=list[SuggestionOut])
async def list_suggestions(
    container: ContainerDep,
    suggestion_status: Annotated[
        SuggestionStatus | None,
        Query(alias="status", description="pending, accepted or rejected"),
    ] = SuggestionStatus.PENDING,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[SuggestionOut]:
    """The review queue, each draft beside the passage it came from."""
    rows = await decision_suggest.list_suggestions(
        container.database.session_factory, status=suggestion_status, limit=limit
    )
    return [
        SuggestionOut(
            id=row.id,
            draft=row.draft.as_dict(),
            source_text=row.source_text,
            source_name=row.source_name,
            external_key=row.external_key,
            chunk_ordinal=row.chunk_ordinal,
            status=row.status,
            model_id=row.model_id,
            suggested_at=row.suggested_at,
            reviewed_at=row.reviewed_at,
            decision_id=row.decision_id,
        )
        for row in rows
    ]


@router.post(
    "/decisions/suggestions/{suggestion_id}/accept",
    response_model=DecisionOut,
    status_code=status.HTTP_201_CREATED,
)
async def accept_suggestion(
    suggestion_id: UUID, container: ContainerDep, body: DecisionIn | None = None
) -> DecisionOut:
    """Turn a draft into a decision, optionally with the reviewer's edits.

    The body is the accept-with-changes path and is the expected one: a reviewer
    who has read the passage usually knows a confidence and at least one
    assumption the model could not have. Accepting unedited is allowed and
    leaves those fields empty, which is honest — an empty confidence is not the
    same as a confidence of nothing.
    """
    edited = None
    if body is not None:
        edited = decisions.DecisionDraft(
            question=body.question,
            chosen=body.chosen,
            reasoning=body.reasoning,
            confidence=body.confidence,
            expected_outcome=body.expected_outcome,
            options=tuple(
                decisions.OptionInput(
                    description=option.description,
                    rejected_because=option.rejected_because,
                )
                for option in body.options
            ),
            assumptions=tuple(
                decisions.AssumptionInput(
                    statement=item.statement, confidence=item.confidence
                )
                for item in body.assumptions
            ),
        )
    try:
        decision_id = await decision_suggest.accept(
            container.database.session_factory, suggestion_id, edited=edited
        )
    except decisions.UnknownDecision as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except decision_suggest.AlreadyReviewed as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except decisions.InvalidDecision as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return DecisionOut(id=decision_id)


@router.post(
    "/decisions/suggestions/{suggestion_id}/reject",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def reject_suggestion(suggestion_id: UUID, container: ContainerDep) -> None:
    """Mark a draft as not a decision. The row stays, and is the only measurement
    of what the extractor gets wrong."""
    try:
        await decision_suggest.reject(
            container.database.session_factory, suggestion_id
        )
    except decisions.UnknownDecision as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except decision_suggest.AlreadyReviewed as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.get("/decisions/{decision_id}", response_model=DecisionDetailOut)
async def show_decision(decision_id: UUID, container: ContainerDep) -> DecisionDetailOut:
    try:
        detail = await decisions.show(container.database.session_factory, decision_id)
    except decisions.UnknownDecision as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return _detail_out(detail)


@router.patch("/decisions/{decision_id}", response_model=DecisionDetailOut)
async def edit_decision(
    decision_id: UUID, body: DecisionEditIn, container: ContainerDep
) -> DecisionDetailOut:
    try:
        await decisions.edit(
            container.database.session_factory,
            decision_id,
            decisions.DecisionEdit(
                question=body.question,
                chosen=body.chosen,
                reasoning=body.reasoning,
                expected_outcome=body.expected_outcome,
                status=body.status,
                options=(
                    tuple(
                        decisions.OptionInput(
                            description=option.description,
                            rejected_because=option.rejected_because,
                        )
                        for option in body.options
                    )
                    if body.options is not None
                    else None
                ),
                assumptions=(
                    tuple(
                        decisions.AssumptionInput(
                            statement=item.statement, confidence=item.confidence
                        )
                        for item in body.assumptions
                    )
                    if body.assumptions is not None
                    else None
                ),
            ),
        )
    except decisions.UnknownDecision as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except decisions.InvalidDecision as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _detail_out(
        await decisions.show(container.database.session_factory, decision_id)
    )


@router.post(
    "/decisions/{decision_id}/evidence", status_code=status.HTTP_201_CREATED
)
async def link_evidence(
    decision_id: UUID, body: EvidenceIn, container: ContainerDep
) -> DecisionOut:
    """Attach one memory to a decision, resolving its natural key to ids."""
    try:
        evidence_id = await decisions.link_evidence(
            container.database.session_factory,
            decision_id,
            decisions.EvidenceInput(
                source_name=body.source_name,
                external_key=body.external_key,
                relation=body.relation,
                chunk_ordinal=body.chunk_ordinal,
            ),
        )
    except decisions.UnknownDecision as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except decisions.UnresolvedEvidence as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return DecisionOut(id=evidence_id)


def _detail_out(detail: decisions.DecisionDetail) -> DecisionDetailOut:
    return DecisionDetailOut(
        id=detail.id,
        question=detail.question,
        chosen=detail.chosen,
        reasoning=detail.reasoning,
        confidence=detail.confidence,
        expected_outcome=detail.expected_outcome,
        status=detail.status,
        decided_at=detail.decided_at,
        decided_at_source=detail.decided_at_source,
        created_at=detail.created_at,
        updated_at=detail.updated_at,
        options=[
            OptionOut(
                id=option.id,
                description=option.description,
                was_chosen=option.was_chosen,
                rejected_because=option.rejected_because,
            )
            for option in detail.options
        ],
        assumptions=[
            AssumptionOut(
                id=item.id,
                statement=item.statement,
                confidence=item.confidence,
                held=item.held,
                evaluated_at=item.evaluated_at,
            )
            for item in detail.assumptions
        ],
        evidence=[
            EvidenceOut(
                id=item.id,
                memory_id=item.memory_id,
                chunk_id=item.chunk_id,
                source_name=item.source_name,
                external_key=item.external_key,
                chunk_ordinal=item.chunk_ordinal,
                relation=item.relation,
            )
            for item in detail.evidence
        ],
    )
