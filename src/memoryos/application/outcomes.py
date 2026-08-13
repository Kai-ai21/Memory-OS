"""What happened after a decision, and how much anybody actually knows about it.

M5.0 recorded the choice, the alternatives and the assumptions. This connects a
decision to what followed. Three things shape everything here.

**`too_early` is a verdict.** Most decisions in a young project have no outcome
yet, and recording that honestly is better than forcing a judgement or leaving a
hole. It is excluded from every success rate — from the numerator *and* the
denominator — so a corpus with two wins and thirty unresolved decisions reports
two out of two rather than a number that reads like a track record. A system
that counted `too_early` as a failure would punish caution; one that counted it
as a success would be lying.

**Declared and inferred outcomes are different kinds of claim.** A declared
outcome is testimony: somebody watched the deployment, read the incident, saw
the number move, and `record` stamps it confidence 1.0 because that is what
observing something means. An inferred outcome is a correlation in time plus a
language model's opinion that the correlation means something, and the schema
forbids one from claiming 1.0. M5.3 must weight them differently, and it can
only do that because the distinction is a column rather than a convention.

**An outcome does not replace the one before it.** A decision can work in the
first month and fail in the sixth, and there is no unique constraint saying a
decision has one outcome. `latest_verdict` reads the most recent by
`observed_at` for the places that need a single answer; the sequence stays
intact for M5.3, which is the milestone that cares about it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.decisions import UnknownDecision
from memoryos.domain.ids import new_id
from memoryos.domain.values import (
    RESOLVED_VERDICTS,
    EvidenceKind,
    OutcomeVerdict,
    TimeProvenance,
)

logger = structlog.get_logger(__name__)


class InvalidOutcome(ValueError):
    """An outcome that cannot be stored as given."""


class UnknownOutcome(LookupError):
    """No outcome with that id."""


class UnresolvedEvidence(LookupError):
    """The item a link names is not in the corpus."""


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OutcomeEvidenceInput:
    """A memory that shows the outcome happened.

    Identified by its natural key, exactly as `decision_evidence` is, with the
    ids resolved at write time. `occurred_at` is not supplied by the caller: it
    is read off the memory as it stands now and stored as a snapshot, because
    the gap between it and the decision's date is the claim being made.
    """

    source_name: str
    external_key: str
    chunk_ordinal: int | None = None


@dataclass(frozen=True, slots=True)
class OutcomeDraft:
    """Everything an outcome holds, before it is a row.

    Shared by the manual path and the suggestion path so both go through the
    same validation, exactly as `DecisionDraft` is. What differs between them is
    `evidence_kind` and `confidence`, and neither is on this type for that
    reason — `record` takes them as arguments, so a caller cannot accidentally
    write an inferred outcome that claims to have been observed.
    """

    description: str
    verdict: OutcomeVerdict
    # The model's own account of why it thinks this is an outcome. Empty for a
    # manual one, where the description is the account.
    rationale: str | None = None
    evidence: tuple[OutcomeEvidenceInput, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """The JSON shape stored in `outcome_suggestions.draft`."""
        return {
            "description": self.description,
            "verdict": self.verdict.value,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OutcomeDraft":
        raw = str(payload.get("verdict") or "").strip().lower()
        try:
            verdict = OutcomeVerdict(raw)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in OutcomeVerdict)
            raise InvalidOutcome(
                f"{raw!r} is not a verdict; expected one of {allowed}"
            ) from exc
        rationale = payload.get("rationale")
        return cls(
            description=str(payload.get("description") or ""),
            verdict=verdict,
            rationale=str(rationale).strip() or None if rationale else None,
        )


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    id: UUID
    memory_id: UUID
    chunk_id: UUID | None
    source_name: str
    external_key: str
    chunk_ordinal: int | None
    occurred_at: datetime | None


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    id: UUID
    decision_id: UUID
    description: str
    verdict: OutcomeVerdict
    observed_at: datetime
    observed_at_source: TimeProvenance
    evidence_kind: EvidenceKind
    confidence: float | None
    created_at: datetime
    evidence: list[EvidenceRow]

    @property
    def resolved(self) -> bool:
        """Whether this outcome says anything a success rate can count."""
        return self.verdict in RESOLVED_VERDICTS


@dataclass(frozen=True, slots=True)
class SuccessRate:
    """What the corpus says about how decisions turned out.

    `too_early` is reported beside the rate rather than inside it, and
    `undecided` — decisions with no outcome at all — beside that. Three numbers
    because they mean three different things: a decision that turned out well, a
    decision it is too soon to judge, and a decision nobody has looked at. A
    single "success rate" over a corpus like this one would be a percentage of
    almost nothing, quoted with a decimal point.
    """

    worked: int
    failed: int
    mixed: int
    too_early: int
    undecided: int

    @property
    def resolved(self) -> int:
        return self.worked + self.failed + self.mixed

    @property
    def rate(self) -> float | None:
        """Successes over resolved outcomes, or None when nothing is resolved.

        None rather than 0.0, which would read as "everything failed" — and on a
        corpus where every decision is `too_early` that is the opposite of what
        the data says.
        """
        if self.resolved == 0:
            return None
        return self.worked / self.resolved

    def as_dict(self) -> dict[str, Any]:
        return {
            "worked": self.worked,
            "failed": self.failed,
            "mixed": self.mixed,
            "too_early": self.too_early,
            "undecided": self.undecided,
            "resolved": self.resolved,
            "rate": self.rate,
        }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


async def record(
    session_factory: async_sessionmaker[AsyncSession],
    decision_id: UUID,
    draft: OutcomeDraft,
    *,
    observed_at: datetime,
    observed_at_source: TimeProvenance,
    evidence_kind: EvidenceKind = EvidenceKind.DECLARED,
    confidence: float | None = None,
) -> UUID:
    """Write one outcome and its evidence, in one transaction.

    `confidence` defaults to 1.0 for a declared outcome and is required to be
    below 1.0 for an inferred one. That asymmetry is the point of the two kinds:
    saying you observed something *is* certainty about the observation, while a
    model's reading of a correlation never is, and a schema that let the second
    claim the first would make M5.3's weighting meaningless.
    """
    description = draft.description.strip()
    if not description:
        raise InvalidOutcome("an outcome needs a description of what happened")
    if observed_at.tzinfo is None:
        raise InvalidOutcome("observed_at must carry a timezone")
    if observed_at_source is TimeProvenance.UNKNOWN:
        raise InvalidOutcome(
            "an outcome needs a date whose provenance is known — the M1.1 rules, "
            "unchanged"
        )

    if confidence is None:
        # Declared means observed, and observing something is what confidence
        # 1.0 is for. An inferred outcome with no stated confidence is not
        # given one, because a default here would be a number nobody produced.
        confidence = 1.0 if evidence_kind is EvidenceKind.DECLARED else None
    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise InvalidOutcome(
            f"confidence is a probability between 0 and 1, got {confidence}"
        )
    if evidence_kind is EvidenceKind.INFERRED and confidence == 1.0:
        raise InvalidOutcome(
            "an inferred outcome cannot claim certainty: it is a correlation in "
            "time and a model's opinion of it, not something anybody observed"
        )

    outcome_id = new_id()
    async with session_factory.begin() as session:
        if await session.get(models.Decision, decision_id) is None:
            raise UnknownDecision(f"no decision {decision_id}")
        session.add(
            models.DecisionOutcome(
                id=outcome_id,
                decision_id=decision_id,
                description=description,
                verdict=draft.verdict.value,
                observed_at=observed_at,
                observed_at_source=observed_at_source.value,
                evidence_kind=evidence_kind.value,
                confidence=confidence,
            )
        )
        for link in draft.evidence:
            await _link_one(session, outcome_id, link)

    logger.info(
        "outcome.recorded",
        decision_id=str(decision_id),
        outcome_id=str(outcome_id),
        verdict=draft.verdict.value,
        evidence_kind=evidence_kind.value,
    )
    return outcome_id


async def link_evidence(
    session_factory: async_sessionmaker[AsyncSession],
    outcome_id: UUID,
    link: OutcomeEvidenceInput,
) -> UUID:
    async with session_factory.begin() as session:
        if await session.get(models.DecisionOutcome, outcome_id) is None:
            raise UnknownOutcome(f"no outcome {outcome_id}")
        return await _link_one(session, outcome_id, link)


async def _link_one(
    session: AsyncSession, outcome_id: UUID, link: OutcomeEvidenceInput
) -> UUID:
    """Resolve the natural key to ids, and snapshot the memory's own clock.

    Resolved here rather than trusted from the caller, for the reason
    `decisions._link_one` does it: a suggestion made before a replay carries ids
    that no longer exist, and writing them would produce a foreign key violation
    at best and a link to somebody else's memory at worst.
    """
    found = (
        await session.execute(
            select(models.Memory.id, models.Memory.occurred_at)
            .join(models.Source, models.Source.id == models.Memory.source_id)
            .where(
                models.Source.name == link.source_name,
                models.Memory.external_key == link.external_key,
                models.Memory.is_current.is_(True),
                models.Memory.deleted_at.is_(None),
            )
        )
    ).first()
    if found is None:
        raise UnresolvedEvidence(
            f"{link.external_key!r} is not a current memory in source "
            f"{link.source_name!r}, so there is nothing to link to"
        )
    memory_id, occurred_at = found

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
        models.OutcomeEvidence(
            id=evidence_id,
            outcome_id=outcome_id,
            memory_id=memory_id,
            chunk_id=chunk_id,
            source_name=link.source_name,
            external_key=link.external_key,
            chunk_ordinal=link.chunk_ordinal,
            occurred_at=occurred_at,
        )
    )
    return evidence_id


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


async def for_decision(
    session_factory: async_sessionmaker[AsyncSession], decision_id: UUID
) -> list[OutcomeRow]:
    """Every outcome recorded for one decision, oldest first.

    Oldest first because the sequence is the information. "Worked, then failed"
    and "failed, then worked" are different stories about the same decision, and
    a reverse-chronological list makes the second one look like the first.
    """
    async with session_factory() as session:
        rows = list(
            (
                await session.execute(
                    select(models.DecisionOutcome)
                    .where(models.DecisionOutcome.decision_id == decision_id)
                    .order_by(models.DecisionOutcome.observed_at)
                )
            ).scalars()
        )
        evidence = list(
            (
                await session.execute(
                    select(models.OutcomeEvidence)
                    .where(
                        models.OutcomeEvidence.outcome_id.in_(
                            [row.id for row in rows] or [new_id()]
                        )
                    )
                    .order_by(models.OutcomeEvidence.external_key)
                )
            ).scalars()
        )

    by_outcome: dict[UUID, list[EvidenceRow]] = {}
    for item in evidence:
        by_outcome.setdefault(item.outcome_id, []).append(
            EvidenceRow(
                id=item.id,
                memory_id=item.memory_id,
                chunk_id=item.chunk_id,
                source_name=item.source_name,
                external_key=item.external_key,
                chunk_ordinal=item.chunk_ordinal,
                occurred_at=item.occurred_at,
            )
        )

    return [_to_row(row, by_outcome.get(row.id, [])) for row in rows]


def _to_row(row: models.DecisionOutcome, evidence: list[EvidenceRow]) -> OutcomeRow:
    return OutcomeRow(
        id=row.id,
        decision_id=row.decision_id,
        description=row.description,
        verdict=OutcomeVerdict(row.verdict),
        observed_at=row.observed_at,
        observed_at_source=TimeProvenance(row.observed_at_source),
        evidence_kind=EvidenceKind(row.evidence_kind),
        confidence=row.confidence,
        created_at=row.created_at,
        evidence=evidence,
    )


async def counts_by_decision(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[UUID, int]:
    """How many outcomes each decision has. What the list view needs."""
    async with session_factory() as session:
        rows = await session.execute(
            select(models.DecisionOutcome.decision_id, func.count())
            .group_by(models.DecisionOutcome.decision_id)
        )
    return {row[0]: row[1] for row in rows}


async def latest_verdict(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[UUID, OutcomeVerdict]:
    """The most recent verdict per decision, for the places that need one answer.

    `DISTINCT ON` rather than a group-by-max join, because the question is "the
    verdict belonging to the newest row" and a max over `observed_at` alone
    would need a second pass to find out which row that was.
    """
    stmt = (
        select(models.DecisionOutcome.decision_id, models.DecisionOutcome.verdict)
        .distinct(models.DecisionOutcome.decision_id)
        .order_by(
            models.DecisionOutcome.decision_id,
            models.DecisionOutcome.observed_at.desc(),
            models.DecisionOutcome.created_at.desc(),
        )
    )
    async with session_factory() as session:
        rows = await session.execute(stmt)
    return {row[0]: OutcomeVerdict(row[1]) for row in rows}


async def success_rate(
    session_factory: async_sessionmaker[AsyncSession],
) -> SuccessRate:
    """The corpus-wide picture, with `too_early` outside the rate.

    Computed over each decision's *latest* verdict rather than over every
    outcome row, because a decision that worked and then failed is one decision
    that failed rather than one of each. The sequence is still there for M5.3;
    this is the summary, and a summary that double-counted would overstate the
    corpus in both directions at once.
    """
    latest = await latest_verdict(session_factory)
    async with session_factory() as session:
        total = int(
            (
                await session.execute(select(func.count()).select_from(models.Decision))
            ).scalar_one()
        )

    tally = {verdict: 0 for verdict in OutcomeVerdict}
    for verdict in latest.values():
        tally[verdict] += 1

    return SuccessRate(
        worked=tally[OutcomeVerdict.WORKED],
        failed=tally[OutcomeVerdict.FAILED],
        mixed=tally[OutcomeVerdict.MIXED],
        too_early=tally[OutcomeVerdict.TOO_EARLY],
        # Decisions with no outcome row at all. Distinct from `too_early`, which
        # is somebody having looked.
        undecided=max(total - len(latest), 0),
    )


def verdicts_for_rate(outcomes: Sequence[OutcomeRow]) -> list[OutcomeRow]:
    """The outcomes a success rate is computed over.

    A function rather than a comprehension at each call site, because "which
    verdicts count" is a decision and two places making it separately is how a
    dashboard and a report end up disagreeing about the same corpus. The set
    itself lives in `domain.values.RESOLVED_VERDICTS`.
    """
    return [outcome for outcome in outcomes if outcome.resolved]
