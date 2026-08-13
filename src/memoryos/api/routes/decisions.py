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

from memoryos.application import (
    assumptions,
    decision_suggest,
    decisions,
    outcome_suggest,
    outcomes,
    patterns,
)
from memoryos.container import Container
from memoryos.domain.values import (
    AssumptionVerdict,
    DecisionStatus,
    EvidenceKind,
    EvidenceRelation,
    OutcomeVerdict,
    PatternKind,
    PatternRelation,
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
    # `held | failed | partially`, or null when nobody has judged it — which is
    # deliberately not `failed`. M5.2 widened this from a boolean because almost
    # nothing anybody assumes is cleanly right or wrong.
    held: AssumptionVerdict | None
    evaluated_at: datetime | None
    # The evaluator's reasoning, separate from the statement they were judging.
    note: str | None = None


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


class OutcomeEvidenceOut(BaseModel):
    id: UUID
    memory_id: UUID
    chunk_id: UUID | None
    source_name: str
    external_key: str
    chunk_ordinal: int | None
    # A snapshot of the evidence memory's clock at link time, not a join. The
    # gap between it and the decision's date is the claim being made.
    occurred_at: datetime | None


class OutcomeOut(BaseModel):
    id: UUID
    description: str
    verdict: OutcomeVerdict
    observed_at: datetime
    observed_at_source: TimeProvenance
    # Testimony or a model's reading. Sent to the client on every outcome
    # because an interface that rendered them identically would be asserting
    # they are the same kind of claim.
    evidence_kind: EvidenceKind
    confidence: float | None
    created_at: datetime
    evidence: list[OutcomeEvidenceOut]


class DismissIn(BaseModel):
    # Required, and the CHECK constraint agrees: a rejection nobody explained is
    # one the next reader cannot tell from a stale row.
    reason: str = Field(min_length=1)


class OutcomeIn(BaseModel):
    description: str = Field(min_length=1)
    verdict: OutcomeVerdict
    observed_at: datetime | None = None
    evidence: list[EvidenceIn] = Field(default_factory=list)
    # No `confidence` and no `evidence_kind`. This route is the declared path:
    # it records what somebody observed, at confidence 1.0, because saying you
    # observed something is what certainty about an observation means. A reading
    # you are unsure about belongs in the suggestion queue.


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
    outcomes: list[OutcomeOut]


class OutcomeSuggestionOut(BaseModel):
    id: UUID
    decision_id: UUID
    # The decision travels with the candidate. "Is this an outcome" is not a
    # question anybody can answer without both on screen.
    decision_question: str
    decision_decided_at: datetime
    draft: dict[str, Any]
    source_text: str
    source_name: str
    external_key: str
    candidate_occurred_at: datetime
    # The temporal claim, stated rather than folded into a score.
    gap_days: float
    window_days: float
    shared_entities: list[str]
    # 'applied' or 'unavailable' — whether the entity test could be run at all.
    entity_filter: str
    status: SuggestionStatus
    model_id: str
    suggested_at: datetime
    reviewed_at: datetime | None
    outcome_id: UUID | None


class AssumptionDetailOut(BaseModel):
    id: UUID
    decision_id: UUID
    # The decision travels with the assumption. A claim made in service of a
    # choice, read away from the choice, is a sentence with its subject removed.
    decision_question: str
    statement: str
    confidence: float | None
    held: AssumptionVerdict | None
    evaluated_at: datetime | None
    note: str | None
    group_id: UUID | None
    group_label: str | None
    # Context, never a gate. An assumption on a `too_early` decision is still
    # evaluable — some beliefs are checkable long before the decision they
    # supported is.
    outcome_verdict: OutcomeVerdict | None
    evidence: list[OutcomeEvidenceOut]


class AssumptionGroupOut(BaseModel):
    id: UUID
    label: str
    strategy: str
    members: int
    evaluated: int
    held: int
    failed: int
    partially: int
    # None when nothing in the group is evaluated. Zero would read as "none of
    # these held", and a group nobody has looked at says nothing at all.
    hold_rate: float | None
    # Deliberately not one minus the hold rate: `partially` counts as a failure
    # here and not as a success there. A belief that half held is a belief that
    # half broke, and the view that surfaces recurring trouble should show it.
    failure_rate: float | None
    statements: list[str]


class AssumptionStatsOut(BaseModel):
    total: int
    evaluated: int
    unevaluated: int
    held: int
    failed: int
    partially: int
    hold_rate: float | None
    groups: list[AssumptionGroupOut]


class PatternEvidenceOut(BaseModel):
    decision_id: UUID
    decision_question: str
    decided_at: datetime
    relation: PatternRelation
    # Why this decision counts for or against, written by the detector. Shown
    # verbatim: "supports" beside a title is not something a reader can check.
    note: str | None


class PatternOut(BaseModel):
    id: UUID
    statement: str
    kind: PatternKind
    detector: str
    support_count: int
    contradiction_count: int
    confidence: float | None
    first_observed: datetime | None
    last_observed: datetime | None
    # How long the supporting decisions span. Three made in one afternoon are
    # three observations of one mood, and the client needs to be able to say so.
    span_days: float | None
    discovered_at: datetime
    dismissed_at: datetime | None
    dismissed_reason: str | None
    # Two lists, never merged and never one flag on a shared list. The interface
    # renders them at equal weight, which is the whole point of finding
    # counter-evidence in the first place.
    supporting: list[PatternEvidenceOut]
    contradicting: list[PatternEvidenceOut]


class CalibrationBandOut(BaseModel):
    low: float
    high: float
    stated: float
    observed: float
    interval_low: float
    interval_high: float
    n: int
    # True only when the stated confidence falls outside the interval its sample
    # supports. Everything else is consistent with being exactly as reliable as
    # claimed.
    miscalibrated: bool


class CalibrationOut(BaseModel):
    decisions: list[CalibrationBandOut]
    assumptions: list[CalibrationBandOut]


class SuccessRateOut(BaseModel):
    worked: int
    failed: int
    mixed: int
    # Reported beside the rate rather than inside it, and `undecided` beside
    # that: a decision it is too soon to judge and a decision nobody has looked
    # at are different facts.
    too_early: int
    undecided: int
    resolved: int
    # None rather than 0.0 when nothing is resolved — zero would read as
    # "everything failed", which on this corpus is the opposite of the truth.
    rate: float | None


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


@router.get("/outcomes/suggestions", response_model=list[OutcomeSuggestionOut])
async def list_outcome_suggestions(
    container: ContainerDep,
    suggestion_status: Annotated[
        SuggestionStatus | None, Query(alias="status")
    ] = SuggestionStatus.PENDING,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[OutcomeSuggestionOut]:
    """The outcome review queue, closest temporal gap first."""
    rows = await outcome_suggest.list_suggestions(
        container.database.session_factory, status=suggestion_status, limit=limit
    )
    return [
        OutcomeSuggestionOut(
            id=row.id,
            decision_id=row.decision_id,
            decision_question=row.decision_question,
            decision_decided_at=row.decision_decided_at,
            draft=row.draft.as_dict(),
            source_text=row.source_text,
            source_name=row.source_name,
            external_key=row.external_key,
            candidate_occurred_at=row.candidate_occurred_at,
            gap_days=row.gap_days,
            window_days=row.window_days,
            shared_entities=row.shared_entities,
            entity_filter=row.entity_filter,
            status=row.status,
            model_id=row.model_id,
            suggested_at=row.suggested_at,
            reviewed_at=row.reviewed_at,
            outcome_id=row.outcome_id,
        )
        for row in rows
    ]


@router.post(
    "/outcomes/suggestions/{suggestion_id}/accept",
    response_model=DecisionOut,
    status_code=status.HTTP_201_CREATED,
)
async def accept_outcome_suggestion(
    suggestion_id: UUID, container: ContainerDep
) -> DecisionOut:
    """Write the outcome as `inferred`, keeping the candidate as its evidence.

    Never `declared`, whoever accepted it. Accepting means the reading is worth
    keeping, not that anybody watched it happen — and an accepted suggestion
    promoted to testimony would be indistinguishable from an observation to
    M5.3, which is the one thing `evidence_kind` exists to prevent.
    """
    try:
        outcome_id = await outcome_suggest.accept(
            container.database.session_factory, suggestion_id
        )
    except decisions.UnknownDecision as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except outcome_suggest.AlreadyReviewed as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except outcomes.InvalidOutcome as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return DecisionOut(id=outcome_id)


@router.post(
    "/outcomes/suggestions/{suggestion_id}/reject",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def reject_outcome_suggestion(
    suggestion_id: UUID, container: ContainerDep
) -> None:
    try:
        await outcome_suggest.reject(container.database.session_factory, suggestion_id)
    except decisions.UnknownDecision as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except outcome_suggest.AlreadyReviewed as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.get("/assumptions", response_model=list[AssumptionDetailOut])
async def list_assumptions_route(
    container: ContainerDep,
    decision: Annotated[UUID | None, Query()] = None,
    unevaluated: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[AssumptionDetailOut]:
    """Assumptions with their decision, outcome, group and evidence."""
    rows = await assumptions.list_assumptions(
        container.database.session_factory,
        decision_id=decision,
        unevaluated_only=unevaluated,
        limit=limit,
    )
    return [
        AssumptionDetailOut(
            id=row.id,
            decision_id=row.decision_id,
            decision_question=row.decision_question,
            statement=row.statement,
            confidence=row.confidence,
            held=row.held,
            evaluated_at=row.evaluated_at,
            note=row.note,
            group_id=row.group_id,
            group_label=row.group_label,
            outcome_verdict=row.outcome_verdict,
            evidence=[
                OutcomeEvidenceOut(
                    id=item.id,
                    memory_id=item.memory_id,
                    chunk_id=item.chunk_id,
                    source_name=item.source_name,
                    external_key=item.external_key,
                    chunk_ordinal=item.chunk_ordinal,
                    occurred_at=item.occurred_at,
                )
                for item in row.evidence
            ],
        )
        for row in rows
    ]


@router.get("/assumptions/stats", response_model=AssumptionStatsOut)
async def assumption_stats(container: ContainerDep) -> AssumptionStatsOut:
    """Totals, hold rate, and every group with its rates.

    `unevaluated` is beside the rate rather than inside it, the same way
    `too_early` sits beside a success rate: a percentage over whatever happened
    to get attention is not a measurement.
    """
    report = await assumptions.stats(container.database.session_factory)
    return AssumptionStatsOut(
        total=report.total,
        evaluated=report.evaluated,
        unevaluated=report.unevaluated,
        held=report.held,
        failed=report.failed,
        partially=report.partially,
        hold_rate=report.hold_rate,
        groups=[
            AssumptionGroupOut(
                id=group.id,
                label=group.label,
                strategy=group.strategy,
                members=group.members,
                evaluated=group.evaluated,
                held=group.held,
                failed=group.failed,
                partially=group.partially,
                hold_rate=group.hold_rate,
                failure_rate=group.failure_rate,
                statements=group.statements,
            )
            # Only the recurring ones. A group of one is an assumption nothing
            # else resembles — a fact about the corpus, not a finding about
            # anybody's judgement — and a view listing them would bury the two
            # that mean something under thirty that do not.
            for group in report.recurring
        ],
    )


@router.get("/patterns", response_model=list[PatternOut])
async def list_patterns_route(
    container: ContainerDep,
    kind: Annotated[PatternKind | None, Query()] = None,
    include_dismissed: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[PatternOut]:
    """Patterns with both evidence lists.

    Both always, and never a truncated one: a client that had to ask again for
    the counter-evidence would render the supporting side first and the
    contradicting side after a spinner, which is how a tool becomes a flatterer.
    """
    rows = await patterns.list_patterns(
        container.database.session_factory,
        kind=kind,
        include_dismissed=include_dismissed,
        limit=limit,
    )
    return [_pattern_out(row) for row in rows]


@router.get("/patterns/calibration", response_model=CalibrationOut)
async def pattern_calibration(container: ContainerDep) -> CalibrationOut:
    """Stated confidence against actual verdicts, band by band.

    Returned whether or not any band is a finding, because "no patterns" and
    "here are the bands, and every stated confidence falls inside what its
    sample supports" are the same result and only the second is legible.
    """
    bands = await patterns.calibration(container.database.session_factory)
    return CalibrationOut(
        decisions=[_band_out(band) for band in bands.get("decision_calibration", [])],
        assumptions=[
            _band_out(band) for band in bands.get("assumption_calibration", [])
        ],
    )


@router.post("/patterns/{pattern_id}/dismiss", status_code=status.HTTP_204_NO_CONTENT)
async def dismiss_pattern(
    pattern_id: UUID, body: DismissIn, container: ContainerDep
) -> None:
    """Reject a pattern permanently. Discovery will not re-propose the subject."""
    try:
        await patterns.dismiss(
            container.database.session_factory, pattern_id, reason=body.reason
        )
    except patterns.UnknownPattern as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


def _pattern_out(row: patterns.PatternRow) -> PatternOut:
    return PatternOut(
        id=row.id,
        statement=row.statement,
        kind=row.kind,
        detector=row.detector,
        support_count=row.support_count,
        contradiction_count=row.contradiction_count,
        confidence=row.confidence,
        first_observed=row.first_observed,
        last_observed=row.last_observed,
        span_days=row.span_days,
        discovered_at=row.discovered_at,
        dismissed_at=row.dismissed_at,
        dismissed_reason=row.dismissed_reason,
        supporting=[_evidence_out(item) for item in row.supporting],
        contradicting=[_evidence_out(item) for item in row.contradicting],
    )


def _evidence_out(item: patterns.EvidenceRow) -> PatternEvidenceOut:
    return PatternEvidenceOut(
        decision_id=item.decision_id,
        decision_question=item.decision_question,
        decided_at=item.decided_at,
        relation=item.relation,
        note=item.note,
    )


def _band_out(band: patterns.CalibrationBand) -> CalibrationBandOut:
    return CalibrationBandOut(
        low=band.low,
        high=band.high,
        stated=band.stated,
        observed=band.interval.observed,
        interval_low=band.interval.low,
        interval_high=band.interval.high,
        n=band.interval.n,
        miscalibrated=band.miscalibrated,
    )


@router.get("/outcomes/rate", response_model=SuccessRateOut)
async def get_success_rate(container: ContainerDep) -> SuccessRateOut:
    """How decisions turned out, with `too_early` outside the rate."""
    rate = await outcomes.success_rate(container.database.session_factory)
    return SuccessRateOut(
        worked=rate.worked,
        failed=rate.failed,
        mixed=rate.mixed,
        too_early=rate.too_early,
        undecided=rate.undecided,
        resolved=rate.resolved,
        rate=rate.rate,
    )


@router.get("/decisions/{decision_id}", response_model=DecisionDetailOut)
async def show_decision(decision_id: UUID, container: ContainerDep) -> DecisionDetailOut:
    try:
        detail = await decisions.show(container.database.session_factory, decision_id)
    except decisions.UnknownDecision as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    recorded = await outcomes.for_decision(
        container.database.session_factory, decision_id
    )
    return _detail_out(detail, recorded)


@router.post(
    "/decisions/{decision_id}/outcomes",
    response_model=DecisionOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_outcome(
    decision_id: UUID, body: OutcomeIn, container: ContainerDep
) -> DecisionOut:
    """Record an observed outcome. Declared, confidence 1.0."""
    try:
        outcome_id = await outcomes.record(
            container.database.session_factory,
            decision_id,
            outcomes.OutcomeDraft(
                description=body.description,
                verdict=body.verdict,
                evidence=tuple(
                    outcomes.OutcomeEvidenceInput(
                        source_name=item.source_name,
                        external_key=item.external_key,
                        chunk_ordinal=item.chunk_ordinal,
                    )
                    for item in body.evidence
                ),
            ),
            observed_at=body.observed_at or _now(),
            observed_at_source=TimeProvenance.DECLARED,
            evidence_kind=EvidenceKind.DECLARED,
        )
    except decisions.UnknownDecision as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (outcomes.InvalidOutcome, outcomes.UnresolvedEvidence) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return DecisionOut(id=outcome_id)


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
        await decisions.show(container.database.session_factory, decision_id),
        await outcomes.for_decision(container.database.session_factory, decision_id),
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


def _detail_out(
    detail: decisions.DecisionDetail, recorded: list[outcomes.OutcomeRow]
) -> DecisionDetailOut:
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
                note=item.note,
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
        outcomes=[
            OutcomeOut(
                id=outcome.id,
                description=outcome.description,
                verdict=outcome.verdict,
                observed_at=outcome.observed_at,
                observed_at_source=outcome.observed_at_source,
                evidence_kind=outcome.evidence_kind,
                confidence=outcome.confidence,
                created_at=outcome.created_at,
                evidence=[
                    OutcomeEvidenceOut(
                        id=item.id,
                        memory_id=item.memory_id,
                        chunk_id=item.chunk_id,
                        source_name=item.source_name,
                        external_key=item.external_key,
                        chunk_ordinal=item.chunk_ordinal,
                        occurred_at=item.occurred_at,
                    )
                    for item in outcome.evidence
                ],
            )
            for outcome in recorded
        ],
    )
