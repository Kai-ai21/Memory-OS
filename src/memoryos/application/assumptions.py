"""Which assumptions held, which did not, and what that says across decisions.

**Assumptions matter more than outcomes, and this is the module that says why.**
An outcome tells you a decision worked: one bit, about one decision,
transferable to nothing. "pgvector will be fast enough at my scale" holding or
failing teaches you something you can apply to the next storage decision.
"The pgvector decision worked out" teaches you nothing at all.

Three things shape everything here.

**Nothing in this module decides.** `assumption_suggest` retrieves memories that
bear on an assumption and prints them; a person reads them and says whether the
belief held. A language model asked "did this assumption hold" produces a
fluent guess dressed as an evaluation, and M5.4's reflections would then be
built on it — a claim about how somebody thinks, resting on a model's opinion
about a sentence that person wrote. The evaluation is the one thing here that
has to be testimony.

**`partially` is a first-class verdict.** Almost nothing anybody assumes is
cleanly right or wrong, and a binary forces the interesting cases into the wrong
box. It counts in the denominator of a hold rate and not the numerator, which is
a judgement rather than an obvious truth and is reported separately so the
choice stays visible.

**Unevaluated is not failed.** It is absent from both halves of every rate, the
same way `too_early` is absent from a success rate. A hold rate over a corpus
where most assumptions have never been looked at would be a percentage of
whatever happened to get attention.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.ids import new_id
from memoryos.domain.values import AssumptionVerdict, OutcomeVerdict

logger = structlog.get_logger(__name__)


class UnknownAssumption(LookupError):
    """No assumption with that id."""


class InvalidEvaluation(ValueError):
    """An evaluation that cannot be stored as given."""


class UnresolvedEvidence(LookupError):
    """The item a link names is not in the corpus."""


@dataclass(frozen=True, slots=True)
class EvidenceInput:
    """A memory the evaluator actually used, named by its natural key."""

    source_name: str
    external_key: str
    chunk_ordinal: int | None = None


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
class AssumptionRow:
    """One assumption with everything needed to judge or report on it.

    Carries the decision it belongs to and that decision's latest outcome,
    because neither `review` nor the stats table is readable without them: an
    assumption is a claim made *in service of* a choice, and reading it away
    from the choice is reading a sentence with its subject removed.
    """

    id: UUID
    decision_id: UUID
    decision_question: str
    decision_decided_at: datetime
    statement: str
    confidence: float | None
    held: AssumptionVerdict | None
    evaluated_at: datetime | None
    note: str | None
    group_id: UUID | None
    group_label: str | None
    # The decision's most recent verdict, or None when nobody has recorded one.
    outcome_verdict: OutcomeVerdict | None
    evidence: list[EvidenceRow]

    @property
    def evaluated(self) -> bool:
        return self.held is not None


@dataclass(frozen=True, slots=True)
class GroupStats:
    """One group of assumptions that say the same thing, and how it went.

    `hold_rate` is over evaluated members only. A group of four with one
    evaluation is not a 100% hold rate, and `evaluated` is beside the number so
    that cannot be read off it by mistake.
    """

    id: UUID
    label: str
    strategy: str
    members: int
    evaluated: int
    held: int
    failed: int
    partially: int
    statements: list[str]

    @property
    def hold_rate(self) -> float | None:
        """Held over evaluated, or None when nothing in the group is evaluated.

        None rather than 0.0 for the reason `SuccessRate.rate` is: zero reads as
        "none of these held", and a group nobody has evaluated says nothing at
        all.
        """
        if self.evaluated == 0:
            return None
        return self.held / self.evaluated

    @property
    def failure_rate(self) -> float | None:
        """What the groups view sorts by.

        `partially` counts as a failure here and not in `hold_rate`, and the two
        are deliberately not complements. A belief that half held is a belief
        that half broke, and the view whose job is to surface recurring trouble
        should show it — while the rate quoted as "how often this held" should
        not credit it.
        """
        if self.evaluated == 0:
            return None
        return (self.failed + self.partially) / self.evaluated


@dataclass(frozen=True, slots=True)
class AssumptionStats:
    total: int
    evaluated: int
    held: int
    failed: int
    partially: int
    groups: list[GroupStats]

    @property
    def unevaluated(self) -> int:
        return self.total - self.evaluated

    @property
    def hold_rate(self) -> float | None:
        if self.evaluated == 0:
            return None
        return self.held / self.evaluated

    @property
    def recurring(self) -> list[GroupStats]:
        """Groups with more than one member — the only ones that mean anything.

        A group of one is an assumption nothing else resembles, which is a fact
        about the corpus rather than a finding about anybody's judgement. M5.3
        needs recurrence, and recurrence starts at two.
        """
        return [group for group in self.groups if group.members > 1]


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


async def list_assumptions(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    decision_id: UUID | None = None,
    unevaluated_only: bool = False,
    limit: int = 200,
) -> list[AssumptionRow]:
    """Assumptions with their decision, outcome and evidence.

    Ordered by the decision's date and then the statement, so `review` walks a
    corpus in the order it was written rather than in whatever order the planner
    returned — an evaluator working through them is reconstructing a period, and
    jumping between months makes that impossible.
    """
    stmt = (
        select(models.DecisionAssumption, models.Decision, models.AssumptionGroup.label)
        .join(
            models.Decision,
            models.Decision.id == models.DecisionAssumption.decision_id,
        )
        .outerjoin(
            models.AssumptionGroup,
            models.AssumptionGroup.id == models.DecisionAssumption.group_id,
        )
        .order_by(models.Decision.decided_at, models.DecisionAssumption.statement)
        .limit(limit)
    )
    if decision_id is not None:
        stmt = stmt.where(models.DecisionAssumption.decision_id == decision_id)
    if unevaluated_only:
        stmt = stmt.where(models.DecisionAssumption.held.is_(None))

    async with session_factory() as session:
        rows = list(await session.execute(stmt))
        assumption_ids = [row[0].id for row in rows]
        evidence = (
            list(
                (
                    await session.execute(
                        select(models.AssumptionEvidence)
                        .where(
                            models.AssumptionEvidence.assumption_id.in_(assumption_ids)
                        )
                        .order_by(models.AssumptionEvidence.external_key)
                    )
                ).scalars()
            )
            if assumption_ids
            else []
        )
        latest = await _latest_outcomes(session, [row[1].id for row in rows])

    by_assumption: dict[UUID, list[EvidenceRow]] = {}
    for item in evidence:
        by_assumption.setdefault(item.assumption_id, []).append(
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

    return [
        AssumptionRow(
            id=row[0].id,
            decision_id=row[1].id,
            decision_question=row[1].question,
            decision_decided_at=row[1].decided_at,
            statement=row[0].statement,
            confidence=row[0].confidence,
            held=AssumptionVerdict(row[0].held) if row[0].held else None,
            evaluated_at=row[0].evaluated_at,
            note=row[0].note,
            group_id=row[0].group_id,
            group_label=row[2],
            outcome_verdict=latest.get(row[1].id),
            evidence=by_assumption.get(row[0].id, []),
        )
        for row in rows
    ]


async def _latest_outcomes(
    session: AsyncSession, decision_ids: Sequence[UUID]
) -> dict[UUID, OutcomeVerdict]:
    """The most recent verdict per decision, for context while evaluating.

    Context only. **An assumption on a `too_early` decision is still evaluable**
    — "the free tier's limits are workable" can be checked the first time a
    quota is hit, long before anybody can say whether the decision it supported
    was right — so nothing here filters on the outcome, and the CLI shows it
    beside the question rather than using it to decide what to ask about.
    """
    if not decision_ids:
        return {}
    stmt = (
        select(models.DecisionOutcome.decision_id, models.DecisionOutcome.verdict)
        .where(models.DecisionOutcome.decision_id.in_(decision_ids))
        .distinct(models.DecisionOutcome.decision_id)
        .order_by(
            models.DecisionOutcome.decision_id,
            models.DecisionOutcome.observed_at.desc(),
            models.DecisionOutcome.created_at.desc(),
        )
    )
    rows = await session.execute(stmt)
    return {row[0]: OutcomeVerdict(row[1]) for row in rows}


async def get(
    session_factory: async_sessionmaker[AsyncSession], assumption_id: UUID
) -> AssumptionRow:
    rows = await _by_ids(session_factory, [assumption_id])
    if not rows:
        raise UnknownAssumption(f"no assumption {assumption_id}")
    return rows[0]


async def _by_ids(
    session_factory: async_sessionmaker[AsyncSession], ids: Sequence[UUID]
) -> list[AssumptionRow]:
    every = await list_assumptions(session_factory, limit=10_000)
    wanted = set(ids)
    return [row for row in every if row.id in wanted]


# --------------------------------------------------------------------------
# Evaluating
# --------------------------------------------------------------------------


async def evaluate(
    session_factory: async_sessionmaker[AsyncSession],
    assumption_id: UUID,
    verdict: AssumptionVerdict,
    *,
    note: str | None = None,
    evidence: Sequence[EvidenceInput] = (),
    evaluated_at: datetime | None = None,
) -> None:
    """Record whether an assumption held, and what the evaluator read.

    Re-evaluating replaces. One verdict per assumption, the same rule
    `query_judgements` applies to a search result and for the same reason: two
    contradictory opinions about whether the same belief held is not richer
    data, it is data nobody can use. The note and the evidence are replaced
    along with it, because they are the reasoning *for* that verdict and keeping
    the old ones would leave a verdict explained by an argument for a different
    one.

    `evidence` names memories by natural key and resolves them here — the link
    is a claim the evaluator made about what they used, not a copy of whatever
    `suggest` proposed.
    """
    statement_note = (note or "").strip() or None
    async with session_factory.begin() as session:
        row = await session.get(models.DecisionAssumption, assumption_id)
        if row is None:
            raise UnknownAssumption(f"no assumption {assumption_id}")

        row.held = verdict.value
        row.evaluated_at = evaluated_at or datetime.now(UTC)
        row.note = statement_note

        await session.execute(
            delete(models.AssumptionEvidence).where(
                models.AssumptionEvidence.assumption_id == assumption_id
            )
        )
        for link in evidence:
            await _link_one(session, assumption_id, link)

    logger.info(
        "assumption.evaluated",
        assumption_id=str(assumption_id),
        held=verdict.value,
        evidence=len(evidence),
    )


async def _link_one(
    session: AsyncSession, assumption_id: UUID, link: EvidenceInput
) -> UUID:
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
        models.AssumptionEvidence(
            id=evidence_id,
            assumption_id=assumption_id,
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
# Stats
# --------------------------------------------------------------------------


async def stats(session_factory: async_sessionmaker[AsyncSession]) -> AssumptionStats:
    """Totals, verdict counts, and every group with its hold rate.

    The group table is the output that matters. A group of four with a 25% hold
    rate is a real finding about how somebody estimates; the corpus-wide hold
    rate is a number that mostly reflects which assumptions were easy to check.
    """
    async with session_factory() as session:
        total = int(
            (
                await session.execute(
                    select(func.count()).select_from(models.DecisionAssumption)
                )
            ).scalar_one()
        )
        tally = {
            AssumptionVerdict(row[0]): row[1]
            for row in await session.execute(
                select(models.DecisionAssumption.held, func.count())
                .where(models.DecisionAssumption.held.is_not(None))
                .group_by(models.DecisionAssumption.held)
            )
        }
        group_rows = list(
            await session.execute(
                select(
                    models.AssumptionGroup,
                    models.DecisionAssumption.statement,
                    models.DecisionAssumption.held,
                )
                .join(
                    models.DecisionAssumption,
                    models.DecisionAssumption.group_id == models.AssumptionGroup.id,
                )
                .order_by(
                    models.AssumptionGroup.created_at, models.DecisionAssumption.statement
                )
            )
        )

    grouped: dict[UUID, list[tuple[models.AssumptionGroup, str, str | None]]] = {}
    for group, statement, held in group_rows:
        grouped.setdefault(group.id, []).append((group, statement, held))

    groups = []
    for members in grouped.values():
        group = members[0][0]
        verdicts = [held for _, _, held in members if held is not None]
        groups.append(
            GroupStats(
                id=group.id,
                label=group.label,
                strategy=group.strategy,
                members=len(members),
                evaluated=len(verdicts),
                held=verdicts.count(AssumptionVerdict.HELD.value),
                failed=verdicts.count(AssumptionVerdict.FAILED.value),
                partially=verdicts.count(AssumptionVerdict.PARTIALLY.value),
                statements=[statement for _, statement, _ in members],
            )
        )

    # Worst first: the whole point of the view is to surface recurring trouble,
    # and a group nobody has evaluated sorts last rather than as a zero.
    groups.sort(key=lambda group: (-(group.failure_rate or -1.0), -group.members))

    return AssumptionStats(
        total=total,
        evaluated=sum(tally.values()),
        held=tally.get(AssumptionVerdict.HELD, 0),
        failed=tally.get(AssumptionVerdict.FAILED, 0),
        partially=tally.get(AssumptionVerdict.PARTIALLY, 0),
        groups=groups,
    )
