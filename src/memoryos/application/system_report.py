"""M8.2: one command that says what the system currently is.

**The artifact you show somebody.** Eight phases produced eight ways of asking a
question — `stats`, `evaluate`, `graph verify`, `decisions list`, `patterns
list`, `model show`, `agent evaluate`, `doctor` — and the answer to "does this
work?" was assembled by hand from nine terminal scrollbacks and a memory of
which numbers were current. That assembly is where a project stops being
checkable: nobody re-runs nine commands to verify a claim in a README, so the
claims drift and nothing catches it.

### Every number here is measured on the run, or labelled as not

Two things in this report are not recomputed, and both say so on the page rather
than in a footnote:

* **The retrieval baselines.** `var/baseline*.json` are recorded runs. The
  current run is live and the deltas are against those files, which is what a
  baseline is for.
* **The agent trajectory scores.** These are always the recorded run in
  `var/baseline-agent.json`, never a live one, and the page says so in those
  words with the date beside them. Scoring the eight golden questions costs a
  live model call each against a free tier that M7.3 already measured as unable
  to serve three passes in a day, so running them inside a report somebody might
  call twice is not a cost this command is allowed to impose. Printing a
  recorded file *as though it were current* is the drift this command exists to
  stop; printing it labelled is the honest alternative. Run `memoryos agent
  evaluate` for a fresh one.

Everything else — corpus counts, graph divergence, decisions, patterns, facets,
model stability, doctor — is computed against the database at the moment the
command runs.

### It is allowed to report that things are empty

Most of Phase 5's patterns and all of Phase 8's derived facets are empty on this
corpus, and the report prints the emptiness with the reason. A summary that
omitted the zero rows would describe a system twice as capable as this one.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.user_model import Stability

# The recorded retrieval runs every later run is measured against, and what each
# one was holding fixed. Named here rather than discovered by globbing `var/`, so
# that a stray file somebody left there does not silently become a baseline.
BASELINES: tuple[tuple[str, str], ...] = (
    ("baseline.json", "M2.2 vector-only, before hybrid retrieval"),
    ("baseline-hybrid.json", "M2.3 hybrid, before reranking"),
    ("baseline-graph-off.json", "M3.4 hybrid + rerank, graph expansion off"),
    ("baseline-temporal-off.json", "M4.2 hybrid + rerank, temporal intent off"),
)

AGENT_BASELINE = "baseline-agent.json"


@dataclass(frozen=True, slots=True)
class CorpusSection:
    memories: int
    current_memories: int
    chunks: int
    embedded_chunks: int
    sources: int
    entities: int
    mentions: int
    relationships: int
    # Memories whose date came from the source rather than from a file mtime.
    # The single number behind M4.0's habits gap and worth carrying here for it.
    declared_dates: int
    # Distinct memories entity extraction has actually reached. Not the same as
    # `mentions`, and the difference is the whole reason M8.0's `workflows`
    # dimension declines: one memory with forty mentions is one memory.
    extracted_memories: int

    @property
    def coverage(self) -> float:
        return self.embedded_chunks / self.chunks if self.chunks else 0.0

    @property
    def extraction_coverage(self) -> float:
        if not self.current_memories:
            return 0.0
        return self.extracted_memories / self.current_memories


@dataclass(frozen=True, slots=True)
class DecisionSection:
    decisions: int
    options: int
    assumptions: int
    grouped_assumptions: int
    evaluated_assumptions: int
    groups: int
    outcomes: int
    by_verdict: dict[str, int] = field(default_factory=dict)
    by_assumption_verdict: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BehaviourSection:
    """Phase 5's patterns and Phase 8's facets, with what they lack.

    One section because they are one claim in two tables — a pattern is a
    regularity over decisions, a facet is a claim about the person derived from
    those regularities — and reporting them apart would let a page show seven
    dimensions of a user model above a line saying no patterns exist.
    """

    patterns: int
    dismissed_patterns: int
    reflections: int
    facets: int
    asserted_facets: int
    derived_facets: int
    withdrawn_facets: int
    dismissed_facets: int
    # Every dimension, with its count or the reason it has none.
    assessments: tuple[Any, ...] = ()
    stability: tuple[Stability, ...] = ()


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """One recorded run and the live run's deltas against it."""

    name: str
    what: str
    ran_at: str
    queries: int
    # metric -> (baseline mean, current mean)
    deltas: tuple[tuple[str, float, float], ...] = ()
    regressions: int = 0
    missing: str = ""


async def gather_corpus(session: AsyncSession) -> CorpusSection:
    async def count(*criteria: Any, table: Any) -> int:
        stmt = select(func.count()).select_from(table)
        for clause in criteria:
            stmt = stmt.where(clause)
        return int((await session.execute(stmt)).scalar_one())

    memories = await count(table=models.Memory)
    current = await count(models.Memory.is_current.is_(True), table=models.Memory)
    chunks = await count(table=models.MemoryChunk)
    embedded = await count(
        models.MemoryChunk.embedding.is_not(None), table=models.MemoryChunk
    )
    declared = await count(
        models.Memory.is_current.is_(True),
        models.Memory.occurred_at_source != "filesystem",
        table=models.Memory,
    )
    reached = int(
        (
            await session.execute(
                select(func.count(func.distinct(models.EntityMention.memory_id)))
            )
        ).scalar_one()
    )
    return CorpusSection(
        memories=memories,
        current_memories=current,
        chunks=chunks,
        embedded_chunks=embedded,
        sources=await count(table=models.Source),
        entities=await count(table=models.Entity),
        mentions=await count(table=models.EntityMention),
        relationships=await count(table=models.EntityRelationship),
        declared_dates=declared,
        extracted_memories=reached,
    )


async def gather_decisions(session: AsyncSession) -> DecisionSection:
    async def count(*criteria: Any, table: Any) -> int:
        stmt = select(func.count()).select_from(table)
        for clause in criteria:
            stmt = stmt.where(clause)
        return int((await session.execute(stmt)).scalar_one())

    by_verdict = {
        str(verdict): int(total)
        for verdict, total in (
            await session.execute(
                select(models.DecisionOutcome.verdict, func.count()).group_by(
                    models.DecisionOutcome.verdict
                )
            )
        ).all()
    }
    by_assumption = {
        str(held): int(total)
        for held, total in (
            await session.execute(
                select(models.DecisionAssumption.held, func.count())
                .where(models.DecisionAssumption.held.is_not(None))
                .group_by(models.DecisionAssumption.held)
            )
        ).all()
    }
    return DecisionSection(
        decisions=await count(table=models.Decision),
        options=await count(table=models.DecisionOption),
        assumptions=await count(table=models.DecisionAssumption),
        grouped_assumptions=await count(
            models.DecisionAssumption.group_id.is_not(None),
            table=models.DecisionAssumption,
        ),
        evaluated_assumptions=await count(
            models.DecisionAssumption.held.is_not(None),
            table=models.DecisionAssumption,
        ),
        groups=await count(table=models.AssumptionGroup),
        outcomes=await count(table=models.DecisionOutcome),
        by_verdict=by_verdict,
        by_assumption_verdict=by_assumption,
    )


async def gather_behaviour(
    session: AsyncSession,
    *,
    assessments: tuple[Any, ...],
    stability: tuple[Stability, ...],
) -> BehaviourSection:
    async def count(*criteria: Any, table: Any) -> int:
        stmt = select(func.count()).select_from(table)
        for clause in criteria:
            stmt = stmt.where(clause)
        return int((await session.execute(stmt)).scalar_one())

    return BehaviourSection(
        patterns=await count(models.Pattern.dismissed_at.is_(None), table=models.Pattern),
        dismissed_patterns=await count(
            models.Pattern.dismissed_at.is_not(None), table=models.Pattern
        ),
        reflections=await count(table=models.Reflection),
        facets=await count(
            models.UserModelFacet.superseded_at.is_(None),
            models.UserModelFacet.dismissed_at.is_(None),
            table=models.UserModelFacet,
        ),
        asserted_facets=await count(
            models.UserModelFacet.origin == "asserted",
            models.UserModelFacet.superseded_at.is_(None),
            models.UserModelFacet.dismissed_at.is_(None),
            table=models.UserModelFacet,
        ),
        derived_facets=await count(
            models.UserModelFacet.origin == "derived",
            models.UserModelFacet.superseded_at.is_(None),
            models.UserModelFacet.dismissed_at.is_(None),
            table=models.UserModelFacet,
        ),
        # M8.2's own count: claims the system used to make and retired because
        # their evidence went away. Zero here is a fact about the corpus rather
        # than about the mechanism — nothing has yet been derived to withdraw.
        withdrawn_facets=await count(
            models.UserModelFacet.superseded_at.is_not(None),
            models.UserModelFacet.superseded_by.is_(None),
            table=models.UserModelFacet,
        ),
        dismissed_facets=await count(
            models.UserModelFacet.dismissed_at.is_not(None),
            table=models.UserModelFacet,
        ),
        assessments=assessments,
        stability=stability,
    )


def read_baselines(root: Path) -> list[tuple[str, str, dict[str, Any] | None]]:
    """Every named baseline, loaded, with a null for one that is not there.

    A missing baseline is reported rather than skipped. The set of things this
    system has been measured against is itself a claim the report makes, and a
    silently shorter list would make an unmeasured configuration look like one
    that had never existed.
    """
    loaded: list[tuple[str, str, dict[str, Any] | None]] = []
    for name, what in BASELINES:
        path = root / name
        if not path.exists():
            loaded.append((name, what, None))
            continue
        loaded.append((name, what, json.loads(path.read_text())))
    return loaded


def agent_baseline(root: Path) -> dict[str, Any] | None:
    path = root / AGENT_BASELINE
    return json.loads(path.read_text()) if path.exists() else None


async def gather(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    assessments: tuple[Any, ...],
    stability: tuple[Stability, ...],
) -> tuple[CorpusSection, DecisionSection, BehaviourSection, datetime]:
    """The three database sections, in one connection."""
    async with session_factory() as session:
        corpus = await gather_corpus(session)
        decisions = await gather_decisions(session)
        behaviour = await gather_behaviour(
            session, assessments=assessments, stability=stability
        )
    return corpus, decisions, behaviour, datetime.now(UTC)
