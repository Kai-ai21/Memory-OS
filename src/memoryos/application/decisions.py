"""Recording what was decided, what else was on the table, and what had to be true.

The second thing in this system that nobody can regenerate, and the first one
that is the product rather than the measurement. `query_judgements` records an
opinion about a search result; a decision records an opinion about the world,
held at a moment, under uncertainty that nobody can reconstruct afterwards.
Everything here is shaped by three facts.

**A decision without alternatives is a description.** "We used Postgres" is a
statement about the present tense. "We used Postgres rather than Celery, because
enqueueing and writing the row it refers to had to be one transaction" is a
decision, and only the second one can be learned from — the first has no
counterfactual in it, so no outcome can ever tell you whether it was right.
`record` refuses the first, and that refusal is the single most opinionated line
in this module.

**Confidence is recorded at the time of deciding and never refreshed.** Its
entire value is that it was written down before the answer was known. An edit
may correct a typo in the reasoning; it may not raise a confidence after the
fact, because a calibration measured in hindsight measures nothing.

**Assumptions are asked for explicitly, one at a time, and "none" is allowed.**
Most people cannot list what they were taking for granted unless somebody asks,
so the interactive prompt does real work rather than filling in a form. An
empty answer is recorded as an empty answer rather than as a skipped question:
a decision that genuinely rested on nothing is a finding, and it is not the same
as a decision whose assumptions nobody wrote down.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.ids import new_id
from memoryos.domain.values import (
    AssumptionVerdict,
    DecisionStatus,
    EvidenceRelation,
    TimeProvenance,
)

logger = structlog.get_logger(__name__)


class InvalidDecision(ValueError):
    """A decision that cannot be stored as given."""


class UnknownDecision(LookupError):
    """No decision with that id."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptionInput:
    """One thing that was on the table.

    `was_chosen` is not supplied by callers of `record`: the chosen option is
    derived from the decision's `chosen` text, so the two cannot disagree. It
    exists on this type because `edit` legitimately moves the choice from one
    option to another.
    """

    description: str
    rejected_because: str | None = None
    was_chosen: bool = False


@dataclass(frozen=True, slots=True)
class AssumptionInput:
    statement: str
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class EvidenceInput:
    """A memory that informed a decision, records it, or argues against it.

    Identified by its natural key, exactly as a judgement is, with the ids
    resolved at write time. A caller that already holds ids may pass them; a
    caller working from search results holds `(source_name, external_key)` and
    an ordinal, which are what survive a rebuild.
    """

    source_name: str
    external_key: str
    relation: EvidenceRelation = EvidenceRelation.INFORMED
    chunk_ordinal: int | None = None


@dataclass(frozen=True, slots=True)
class DecisionDraft:
    """Everything a decision record holds, before it is a row.

    Shared by the manual path and the extraction path, which is deliberate: a
    suggestion is a draft that has not been accepted, not a different kind of
    object, so both go through the same validation and neither can write
    something the other would have refused.
    """

    question: str
    chosen: str
    reasoning: str | None = None
    confidence: float | None = None
    expected_outcome: str | None = None
    options: tuple[OptionInput, ...] = ()
    assumptions: tuple[AssumptionInput, ...] = ()
    evidence: tuple[EvidenceInput, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """The JSON shape stored in `decision_suggestions.draft`."""
        return {
            "question": self.question,
            "chosen": self.chosen,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "expected_outcome": self.expected_outcome,
            "options": [
                {
                    "description": option.description,
                    "rejected_because": option.rejected_because,
                    "was_chosen": option.was_chosen,
                }
                for option in self.options
            ],
            "assumptions": [
                {"statement": item.statement, "confidence": item.confidence}
                for item in self.assumptions
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DecisionDraft":
        """Read one back, tolerating a model's missing keys but not its lies.

        Missing keys become absent fields; a key holding the wrong type raises
        here rather than at the database, because a `TypeError` from psycopg
        names a column and this names the field the model got wrong.
        """
        return cls(
            question=str(payload.get("question") or ""),
            chosen=str(payload.get("chosen") or ""),
            reasoning=_optional_text(payload.get("reasoning")),
            confidence=_optional_float(payload.get("confidence")),
            expected_outcome=_optional_text(payload.get("expected_outcome")),
            options=tuple(
                OptionInput(
                    description=str(item.get("description") or ""),
                    rejected_because=_optional_text(item.get("rejected_because")),
                    was_chosen=bool(item.get("was_chosen", False)),
                )
                for item in payload.get("options") or []
            ),
            assumptions=tuple(
                AssumptionInput(
                    statement=str(item.get("statement") or ""),
                    confidence=_optional_float(item.get("confidence")),
                )
                for item in payload.get("assumptions") or []
            ),
        )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidDecision(f"confidence must be a number, got {value!r}") from exc


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionSummary:
    id: UUID
    question: str
    chosen: str
    status: DecisionStatus
    confidence: float | None
    decided_at: datetime
    decided_at_source: TimeProvenance
    options: int
    assumptions: int
    evidence: int


@dataclass(frozen=True, slots=True)
class OptionRow:
    id: UUID
    description: str
    was_chosen: bool
    rejected_because: str | None


@dataclass(frozen=True, slots=True)
class AssumptionRow:
    id: UUID
    statement: str
    confidence: float | None
    # Written by M5.2, which widened it from a boolean to `held | failed |
    # partially`: almost nothing anybody assumes is cleanly right or wrong, and
    # a binary forced the interesting cases into the wrong box. None still means
    # nobody has judged it, and is deliberately not `FAILED`.
    held: AssumptionVerdict | None
    evaluated_at: datetime | None
    # The evaluator's reasoning. Separate from `statement`, which is what was
    # believed at the time and is never rewritten to match what happened.
    note: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    id: UUID
    memory_id: UUID
    chunk_id: UUID | None
    source_name: str
    external_key: str
    chunk_ordinal: int | None
    relation: EvidenceRelation


@dataclass(frozen=True, slots=True)
class DecisionDetail:
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
    options: list[OptionRow] = field(default_factory=list)
    assumptions: list[AssumptionRow] = field(default_factory=list)
    evidence: list[EvidenceRow] = field(default_factory=list)

    @property
    def rejected(self) -> list[OptionRow]:
        return [option for option in self.options if not option.was_chosen]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


async def record(
    session_factory: async_sessionmaker[AsyncSession],
    draft: DecisionDraft,
    *,
    decided_at: datetime,
    decided_at_source: TimeProvenance,
    status: DecisionStatus = DecisionStatus.OPEN,
) -> UUID:
    """Write one decision, its options, its assumptions and its evidence.

    One transaction for all four tables. A decision that committed without its
    options would be exactly the shape this module refuses to accept, and it
    would be in the database rather than in an error message.

    The chosen option is written from `chosen` rather than taken from the
    caller's option list, so the winner named in the decision and the winner
    flagged in the options cannot disagree. A caller that also passes an option
    matching `chosen` gets one row, not two.
    """
    validated = _validate(draft, decided_at=decided_at, decided_at_source=decided_at_source)
    decision_id = new_id()

    async with session_factory.begin() as session:
        session.add(
            models.Decision(
                id=decision_id,
                question=validated.question,
                chosen=validated.chosen,
                reasoning=validated.reasoning,
                confidence=validated.confidence,
                expected_outcome=validated.expected_outcome,
                decided_at=decided_at,
                decided_at_source=decided_at_source.value,
                status=status.value,
            )
        )
        await _write_options(session, decision_id, validated)
        await _write_assumptions(session, decision_id, validated.assumptions)
        for link in validated.evidence:
            await _link_one(session, decision_id, link)

    logger.info(
        "decision.recorded",
        decision_id=str(decision_id),
        question=validated.question,
        options=len(validated.options) + 1,
        assumptions=len(validated.assumptions),
    )
    return decision_id


def _validate(
    draft: DecisionDraft,
    *,
    decided_at: datetime,
    decided_at_source: TimeProvenance,
) -> DecisionDraft:
    """Everything that must be true before a decision becomes a row.

    The alternatives check is the one with an opinion in it. The rest are the
    ordinary shape rules, enforced here as well as by CHECK constraints so that
    the message names the field rather than the constraint.
    """
    question = draft.question.strip()
    chosen = draft.chosen.strip()
    if not question:
        raise InvalidDecision("a decision needs the question it answers")
    if not chosen:
        raise InvalidDecision("a decision needs what was chosen")

    # The rule. Options equal to the chosen text are the winner rather than an
    # alternative, so they do not count towards it.
    alternatives = tuple(
        option
        for option in draft.options
        if option.description.strip()
        and option.description.strip().casefold() != chosen.casefold()
    )
    if not alternatives:
        raise InvalidDecision(
            f"{question!r} has no alternatives, so it is a description of what "
            f"happened rather than a decision. Record at least one option that "
            f"was considered and not taken, with the reason it lost — that is "
            f"the part a later outcome can be read against."
        )

    if draft.confidence is not None and not 0.0 <= draft.confidence <= 1.0:
        raise InvalidDecision(
            f"confidence is a probability between 0 and 1, got {draft.confidence}"
        )
    for assumption in draft.assumptions:
        if not assumption.statement.strip():
            raise InvalidDecision("an assumption with no statement says nothing")
        if assumption.confidence is not None and not 0.0 <= assumption.confidence <= 1.0:
            raise InvalidDecision(
                f"assumption confidence is a probability between 0 and 1, got "
                f"{assumption.confidence}"
            )

    if decided_at_source is TimeProvenance.UNKNOWN:
        raise InvalidDecision(
            "a decision needs a date whose provenance is known. `declared` for one "
            "a person stated, `parsed` for one read out of a document, `filesystem` "
            "for one taken from a file's mtime — the M1.1 rules, unchanged."
        )
    if decided_at.tzinfo is None:
        raise InvalidDecision("decided_at must carry a timezone")

    return DecisionDraft(
        question=question,
        chosen=chosen,
        reasoning=_optional_text(draft.reasoning),
        confidence=draft.confidence,
        expected_outcome=_optional_text(draft.expected_outcome),
        options=alternatives,
        assumptions=tuple(
            AssumptionInput(item.statement.strip(), item.confidence)
            for item in draft.assumptions
        ),
        evidence=draft.evidence,
    )


async def _write_options(
    session: AsyncSession, decision_id: UUID, draft: DecisionDraft
) -> None:
    session.add(
        models.DecisionOption(
            id=new_id(),
            decision_id=decision_id,
            description=draft.chosen,
            was_chosen=True,
            rejected_because=None,
        )
    )
    for option in draft.options:
        session.add(
            models.DecisionOption(
                id=new_id(),
                decision_id=decision_id,
                description=option.description.strip(),
                was_chosen=False,
                rejected_because=_optional_text(option.rejected_because),
            )
        )


async def _write_assumptions(
    session: AsyncSession, decision_id: UUID, assumptions: Sequence[AssumptionInput]
) -> None:
    for assumption in assumptions:
        session.add(
            models.DecisionAssumption(
                id=new_id(),
                decision_id=decision_id,
                statement=assumption.statement.strip(),
                confidence=assumption.confidence,
            )
        )


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------


class UnresolvedEvidence(LookupError):
    """The item a link names is not in the corpus."""


async def link_evidence(
    session_factory: async_sessionmaker[AsyncSession],
    decision_id: UUID,
    link: EvidenceInput,
) -> UUID:
    """Attach one memory to one decision, resolving the natural key to ids."""
    async with session_factory.begin() as session:
        if await session.get(models.Decision, decision_id) is None:
            raise UnknownDecision(f"no decision {decision_id}")
        return await _link_one(session, decision_id, link)


async def _link_one(
    session: AsyncSession, decision_id: UUID, link: EvidenceInput
) -> UUID:
    """Resolve `(source_name, external_key, chunk_ordinal)` and write the row.

    Resolved here rather than trusted from the caller because the ids are the
    part that goes stale: a suggestion made before a replay carries ids that no
    longer exist, and writing them would produce a foreign key violation at
    best and a link to somebody else's memory at worst.
    """
    memory_id = (
        await session.execute(
            select(models.Memory.id)
            .join(models.Source, models.Source.id == models.Memory.source_id)
            .where(
                models.Source.name == link.source_name,
                models.Memory.external_key == link.external_key,
                models.Memory.is_current.is_(True),
                models.Memory.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if memory_id is None:
        raise UnresolvedEvidence(
            f"{link.external_key!r} is not a current memory in source "
            f"{link.source_name!r}, so there is nothing to link to"
        )

    chunk_id: UUID | None = None
    if link.chunk_ordinal is not None:
        chunk_id = (
            await session.execute(
                select(models.MemoryChunk.id).where(
                    models.MemoryChunk.memory_id == memory_id,
                    models.MemoryChunk.ordinal == link.chunk_ordinal,
                )
            )
        ).scalar_one_or_none()
        if chunk_id is None:
            raise UnresolvedEvidence(
                f"{link.external_key!r} has no chunk {link.chunk_ordinal}"
            )

    evidence_id = new_id()
    session.add(
        models.DecisionEvidence(
            id=evidence_id,
            decision_id=decision_id,
            memory_id=memory_id,
            chunk_id=chunk_id,
            source_name=link.source_name,
            external_key=link.external_key,
            chunk_ordinal=link.chunk_ordinal,
            relation=link.relation.value,
        )
    )
    return evidence_id


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


async def list_decisions(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: DecisionStatus | None = None,
    limit: int = 100,
) -> list[DecisionSummary]:
    """Every decision, newest first, with the counts that say how complete it is.

    The three counts are the list's real content. A decision with two options,
    no assumptions and no evidence is a decision nothing later in Phase 5 can do
    anything with, and the only place that is visible is beside the ones that
    are complete.
    """
    counts = {
        "options": _child_count(models.DecisionOption),
        "assumptions": _child_count(models.DecisionAssumption),
        "evidence": _child_count(models.DecisionEvidence),
    }
    stmt = (
        select(
            models.Decision,
            counts["options"].label("options"),
            counts["assumptions"].label("assumptions"),
            counts["evidence"].label("evidence"),
        )
        .order_by(models.Decision.decided_at.desc(), models.Decision.created_at.desc())
        .limit(limit)
    )
    if status is not None:
        stmt = stmt.where(models.Decision.status == status.value)

    async with session_factory() as session:
        rows = list(await session.execute(stmt))

    return [
        DecisionSummary(
            id=row[0].id,
            question=row[0].question,
            chosen=row[0].chosen,
            status=DecisionStatus(row[0].status),
            confidence=row[0].confidence,
            decided_at=row[0].decided_at,
            decided_at_source=TimeProvenance(row[0].decided_at_source),
            options=row[1],
            assumptions=row[2],
            evidence=row[3],
        )
        for row in rows
    ]


def _child_count(model: type[Any]) -> Any:
    """A correlated count of one decision's children.

    A scalar subquery per table rather than three joins, because joining three
    one-to-many tables multiplies their rows together and every count comes back
    as the product of the other two.
    """
    return (
        select(func.count())
        .select_from(model)
        .where(model.decision_id == models.Decision.id)
        .scalar_subquery()
    )


async def show(
    session_factory: async_sessionmaker[AsyncSession], decision_id: UUID
) -> DecisionDetail:
    """One decision with everything hanging off it."""
    async with session_factory() as session:
        row = await session.get(models.Decision, decision_id)
        if row is None:
            raise UnknownDecision(f"no decision {decision_id}")

        options = list(
            (
                await session.execute(
                    select(models.DecisionOption)
                    .where(models.DecisionOption.decision_id == decision_id)
                    # The chosen one first, then the alternatives in a stable
                    # order: this is read top to bottom by a person.
                    .order_by(
                        models.DecisionOption.was_chosen.desc(),
                        models.DecisionOption.description,
                    )
                )
            ).scalars()
        )
        assumptions = list(
            (
                await session.execute(
                    select(models.DecisionAssumption)
                    .where(models.DecisionAssumption.decision_id == decision_id)
                    .order_by(models.DecisionAssumption.statement)
                )
            ).scalars()
        )
        evidence = list(
            (
                await session.execute(
                    select(models.DecisionEvidence)
                    .where(models.DecisionEvidence.decision_id == decision_id)
                    .order_by(
                        models.DecisionEvidence.relation,
                        models.DecisionEvidence.external_key,
                    )
                )
            ).scalars()
        )

    return DecisionDetail(
        id=row.id,
        question=row.question,
        chosen=row.chosen,
        reasoning=row.reasoning,
        confidence=row.confidence,
        expected_outcome=row.expected_outcome,
        status=DecisionStatus(row.status),
        decided_at=row.decided_at,
        decided_at_source=TimeProvenance(row.decided_at_source),
        created_at=row.created_at,
        updated_at=row.updated_at,
        options=[
            OptionRow(
                id=option.id,
                description=option.description,
                was_chosen=option.was_chosen,
                rejected_because=option.rejected_because,
            )
            for option in options
        ],
        assumptions=[
            AssumptionRow(
                id=item.id,
                statement=item.statement,
                confidence=item.confidence,
                held=AssumptionVerdict(item.held) if item.held else None,
                evaluated_at=item.evaluated_at,
                note=item.note,
            )
            for item in assumptions
        ],
        evidence=[
            EvidenceRow(
                id=item.id,
                memory_id=item.memory_id,
                chunk_id=item.chunk_id,
                source_name=item.source_name,
                external_key=item.external_key,
                chunk_ordinal=item.chunk_ordinal,
                relation=EvidenceRelation(item.relation),
            )
            for item in evidence
        ],
    )


# --------------------------------------------------------------------------
# Editing
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DecisionEdit:
    """What an edit may change.

    Deliberately not everything. `decided_at` and `decided_at_source` are absent
    because moving a decision in time silently rewrites the only ordering M5.1
    has to work with, and `confidence` is absent because the number's entire
    value is that it was written before the outcome was known. Correcting either
    is a delete and a re-record, which leaves the correction visible.

    `options` and `assumptions`, when given, replace the existing set rather than
    merging into it. A merge would need identity for rows a person is editing by
    hand, and "the list, as it now reads" is what an editor is actually holding.
    """

    question: str | None = None
    chosen: str | None = None
    reasoning: str | None = None
    expected_outcome: str | None = None
    status: DecisionStatus | None = None
    options: tuple[OptionInput, ...] | None = None
    assumptions: tuple[AssumptionInput, ...] | None = None


async def edit(
    session_factory: async_sessionmaker[AsyncSession],
    decision_id: UUID,
    changes: DecisionEdit,
) -> None:
    """Apply an edit, re-running the rules the original had to pass."""
    async with session_factory.begin() as session:
        row = await session.get(models.Decision, decision_id)
        if row is None:
            raise UnknownDecision(f"no decision {decision_id}")

        if changes.question is not None:
            if not changes.question.strip():
                raise InvalidDecision("a decision needs the question it answers")
            row.question = changes.question.strip()
        if changes.chosen is not None:
            if not changes.chosen.strip():
                raise InvalidDecision("a decision needs what was chosen")
            row.chosen = changes.chosen.strip()
        if changes.reasoning is not None:
            row.reasoning = _optional_text(changes.reasoning)
        if changes.expected_outcome is not None:
            row.expected_outcome = _optional_text(changes.expected_outcome)
        if changes.status is not None:
            row.status = changes.status.value

        if changes.options is not None:
            alternatives = tuple(
                option
                for option in changes.options
                if option.description.strip()
                and option.description.strip().casefold() != row.chosen.casefold()
            )
            if not alternatives:
                raise InvalidDecision(
                    "an edit cannot remove the last alternative: a decision with "
                    "nothing rejected is a description of what happened"
                )
            await session.execute(
                delete(models.DecisionOption).where(
                    models.DecisionOption.decision_id == decision_id
                )
            )
            await _write_options(
                session,
                decision_id,
                DecisionDraft(
                    question=row.question, chosen=row.chosen, options=alternatives
                ),
            )
        elif changes.chosen is not None:
            # The winner's text moved but the option list did not, so the row
            # flagged `was_chosen` still describes the old choice.
            await session.execute(
                delete(models.DecisionOption).where(
                    models.DecisionOption.decision_id == decision_id,
                    models.DecisionOption.was_chosen.is_(True),
                )
            )
            session.add(
                models.DecisionOption(
                    id=new_id(),
                    decision_id=decision_id,
                    description=row.chosen,
                    was_chosen=True,
                )
            )

        if changes.assumptions is not None:
            for assumption in changes.assumptions:
                if not assumption.statement.strip():
                    raise InvalidDecision("an assumption with no statement says nothing")
            await session.execute(
                delete(models.DecisionAssumption).where(
                    models.DecisionAssumption.decision_id == decision_id
                )
            )
            await _write_assumptions(session, decision_id, changes.assumptions)

        row.updated_at = datetime.now(UTC)

    logger.info("decision.edited", decision_id=str(decision_id))
