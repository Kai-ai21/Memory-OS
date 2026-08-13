"""Finding memories that bear on an assumption, and stopping there.

**The system proposes evidence; you judge.** That sentence is the whole design.
An assumption is a claim somebody made about the world, and asking a language
model whether it held produces a fluent guess dressed as an evaluation — M5.4's
reflections read these values, so a model's opinion here becomes a claim about
how a person thinks, stated as fact and impossible to falsify.

So there is **no `LanguageModel` in this module at all**, which is a deliberate
absence rather than an oversight. It is the only proposal path in Phase 5 built
that way: M5.0 asks a model to draft decisions and M5.1 asks one to judge
outcomes, both behind a review queue. Here the retrieval *is* the proposal, and
what comes back is a list of passages with the reason each surfaced. Nothing is
queued, nothing carries a verdict, and there is nothing to accept — the
evaluator reads them and runs `memoryos assumption <id> --held ...`.

Two mechanisms, and each answers a different question:

* **Retrieval**, over the assumption's own text. This is the one that works on
  prose: "the free tier's rate limits are workable" finds the passage about a
  token cap because the words are related, not because anything was extracted.
* **The temporal filter**, which narrows to memories that occurred *after* the
  decision. A memory that predates the decision cannot be evidence about whether
  the belief later held — it is part of what the belief was formed from. M4.0's
  `occurred_at`, nulls excluded rather than defaulted.

The graph adds a third when the corpus can support it: memories sharing a
resolved entity with the decision's evidence. On a corpus with no extraction
that returns nothing, and the report says `unavailable` rather than pretending
the test was run and passed — the same distinction M5.1 draws, for the same
reason.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.assumptions import AssumptionRow, list_assumptions
from memoryos.application.ports import SearchFilters
from memoryos.application.search import SearchMemories
from memoryos.domain.values import TimeProvenance

logger = structlog.get_logger(__name__)

# How many passages to retrieve per assumption before the temporal filter runs.
# Wider than what is shown, because the filter removes most of them on this
# corpus: nearly every memory predates nearly every decision.
RETRIEVAL_DEPTH = 25

# How many to actually print. A reader evaluating twenty assumptions will not
# read ten passages for each, and a list nobody finishes is a list that gets
# skimmed — the failure this whole phase is arranged against.
SHOWN_PER_ASSUMPTION = 4


@dataclass(frozen=True, slots=True)
class ProposedEvidence:
    """One memory that might bear on an assumption, and why it surfaced.

    `why` is a sentence rather than a score. "0.71" tells a reader nothing they
    can act on; "retrieved for the statement, 2.3 days after the decision,
    shares `postgres`" is a claim they can check against the passage.
    """

    memory_id: UUID
    source_name: str
    external_key: str
    occurred_at: datetime | None
    occurred_at_source: TimeProvenance
    excerpt: str
    score: float
    gap_days: float | None
    shared_entities: tuple[str, ...]

    @property
    def why(self) -> str:
        parts = [f"retrieved at {self.score:.4f}"]
        if self.gap_days is not None:
            parts.append(f"{self.gap_days:.1f} days after the decision")
        if self.shared_entities:
            parts.append(f"shares {', '.join(self.shared_entities)}")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class AssumptionEvidenceProposal:
    assumption: AssumptionRow
    evidence: list[ProposedEvidence]
    # `applied` or `unavailable`, exactly as M5.1's suggestion rows carry it. A
    # corpus with no extraction cannot fail the entity test; it cannot take it.
    entity_filter: str


@dataclass(slots=True)
class SuggestReport:
    assumptions: int = 0
    with_evidence: int = 0
    retrieved: int = 0
    # Dropped because they predate the decision. Reported because on this corpus
    # it is almost all of them, and a run that showed nothing should say whether
    # retrieval found nothing or whether time removed it.
    dropped_before_decision: int = 0
    without_entity_coverage: int = 0
    proposals: list[AssumptionEvidenceProposal] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "assumptions": self.assumptions,
            "with_evidence": self.with_evidence,
            "retrieved": self.retrieved,
            "dropped_before_decision": self.dropped_before_decision,
            "without_entity_coverage": self.without_entity_coverage,
        }


class SuggestAssumptionEvidence:
    """Retrieve passages bearing on each unevaluated assumption. Decide nothing.

    Stated in the class as well as the module, for the reason M5.0 and M5.1
    state their equivalents: somebody looking for "the thing that evaluates
    assumptions" must not find something that could be made to.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        search: SearchMemories,
    ) -> None:
        self._sessions = session_factory
        self._search = search

    async def __call__(
        self,
        *,
        decision_id: UUID | None = None,
        unevaluated_only: bool = True,
        limit: int = 20,
    ) -> SuggestReport:
        report = SuggestReport()
        assumptions = await list_assumptions(
            self._sessions,
            decision_id=decision_id,
            unevaluated_only=unevaluated_only,
            limit=limit,
        )
        report.assumptions = len(assumptions)

        for assumption in assumptions:
            evidence, entity_filter = await self._for_one(assumption, report)
            if evidence:
                report.with_evidence += 1
            if entity_filter == "unavailable":
                report.without_entity_coverage += 1
            report.proposals.append(
                AssumptionEvidenceProposal(
                    assumption=assumption,
                    evidence=evidence,
                    entity_filter=entity_filter,
                )
            )

        logger.info("assumptions.evidence_suggested", **report.as_dict())
        return report

    async def _for_one(
        self, assumption: AssumptionRow, report: SuggestReport
    ) -> tuple[list[ProposedEvidence], str]:
        result = await self._search(
            assumption.statement,
            k=RETRIEVAL_DEPTH,
            filters=SearchFilters(),
            rerank=True,
        )
        report.retrieved += len(result.hits)

        decision_entities = await _decision_entities(self._sessions, assumption.decision_id)
        entity_filter = "applied" if decision_entities else "unavailable"
        memory_ids = tuple(hit.memory_id for hit in result.hits)
        by_memory = (
            await _entities_by_memory(self._sessions, memory_ids)
            if decision_entities
            else {}
        )

        proposals: list[ProposedEvidence] = []
        for hit in result.hits:
            occurred_at = _as_datetime(hit.occurred_at)
            if occurred_at is None or occurred_at <= assumption.decision_decided_at:
                # Not evidence about whether the belief later held. A memory
                # that predates the decision is part of what the belief was
                # formed from, and offering it as a test of that belief would be
                # circular. Undated memories go the same way: an unknown date is
                # not evidence of any date.
                report.dropped_before_decision += 1
                continue

            overlap = decision_entities.keys() & by_memory.get(hit.memory_id, set())
            best = max(hit.matched_chunks, key=lambda chunk: chunk.score)
            proposals.append(
                ProposedEvidence(
                    memory_id=hit.memory_id,
                    source_name=hit.source_name,
                    external_key=hit.external_key,
                    occurred_at=occurred_at,
                    occurred_at_source=_provenance(hit),
                    excerpt=" ".join(best.text.split())[:400],
                    score=hit.score,
                    gap_days=(
                        occurred_at - assumption.decision_decided_at
                    ).total_seconds()
                    / 86400.0,
                    shared_entities=tuple(
                        sorted(decision_entities[key] for key in overlap)
                    ),
                )
            )

        # Entity overlap first, then retrieval score. A passage that shares a
        # resolved entity with the decision's own evidence is a stronger lead
        # than one that merely reads similarly, and on a corpus of prose about
        # one project reading similarly is close to free.
        proposals.sort(key=lambda item: (-len(item.shared_entities), -item.score))
        return proposals[:SHOWN_PER_ASSUMPTION], entity_filter


def _as_datetime(value: object) -> datetime | None:
    """`SearchHit.occurred_at` is typed `object` at the port. Narrow it here."""
    return value if isinstance(value, datetime) else None


def _provenance(hit: object) -> TimeProvenance:
    """The hit's own provenance if it carries one, else `unknown`.

    Search results do not currently carry `occurred_at_source`, so this reports
    `unknown` rather than inventing `filesystem` — which would be right for this
    corpus and wrong as a rule, and is exactly the substitution M1.1's CHECK
    constraint exists to forbid everywhere else.
    """
    source = getattr(hit, "occurred_at_source", None)
    if isinstance(source, TimeProvenance):
        return source
    if isinstance(source, str):
        try:
            return TimeProvenance(source)
        except ValueError:
            return TimeProvenance.UNKNOWN
    return TimeProvenance.UNKNOWN


async def _decision_entities(
    sessions: async_sessionmaker[AsyncSession], decision_id: UUID
) -> dict[UUID, str]:
    """Resolved entities named by the memories that informed this decision."""
    winner = func.coalesce(models.Entity.merged_into_id, models.Entity.id)
    stmt = (
        select(winner, models.Entity.canonical_name)
        .join(models.EntityMention, models.EntityMention.entity_id == models.Entity.id)
        .join(
            models.DecisionEvidence,
            models.DecisionEvidence.memory_id == models.EntityMention.memory_id,
        )
        .where(models.DecisionEvidence.decision_id == decision_id)
        .distinct()
    )
    async with sessions() as session:
        rows = await session.execute(stmt)
    return {row[0]: row[1] for row in rows}


async def _entities_by_memory(
    sessions: async_sessionmaker[AsyncSession], memory_ids: tuple[UUID, ...]
) -> dict[UUID, set[UUID]]:
    if not memory_ids:
        return {}
    winner = func.coalesce(models.Entity.merged_into_id, models.Entity.id)
    stmt = (
        select(models.EntityMention.memory_id, winner)
        .join(models.Entity, models.Entity.id == models.EntityMention.entity_id)
        .where(models.EntityMention.memory_id.in_(memory_ids))
        .distinct()
    )
    async with sessions() as session:
        rows = await session.execute(stmt)
    grouped: dict[UUID, set[UUID]] = {}
    for memory_id, entity_id in rows:
        grouped.setdefault(memory_id, set()).add(entity_id)
    return grouped
