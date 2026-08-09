"""Recording what a human thought of a search result.

This is the only part of the system that writes data nobody can regenerate, and
the whole of M2.0a exists to produce it. Everything here is shaped by two facts.

**Identity is the natural key.** A judgement points at `(source_name,
external_key)`, not at a memory id, because a replay mints new ids for every row
and the golden set has to outlive that. `memory_id`, `rank_at_judgement` and
`score_at_judgement` are all snapshots of what the system said at the moment
somebody disagreed with it — kept because that is the only way to tell later
whether a change fixed *that* case, and never refreshed.

**Re-judging replaces.** One verdict per (query, item). Two contradictory
opinions about the same pair is not richer data, it is data nobody can use.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.ids import new_id
from memoryos.domain.values import Verdict

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class JudgementInput:
    query_text: str
    source_name: str
    external_key: str
    verdict: Verdict
    memory_id: UUID | None = None
    chunk_id: UUID | None = None
    rank_at_judgement: int | None = None
    score_at_judgement: float | None = None
    filters: dict[str, Any] | None = None


class InvalidJudgement(ValueError):
    """A judgement that cannot be stored as given."""


async def record(
    session_factory: async_sessionmaker[AsyncSession], judgement: JudgementInput
) -> UUID:
    """Insert or replace one verdict, returning the row's id.

    `ON CONFLICT DO UPDATE` rather than a read-then-write: two tabs judging the
    same result would otherwise race and one would fail on the unique index for
    no reason a user could act on.
    """
    query_text = judgement.query_text.strip()
    if not query_text:
        raise InvalidJudgement("a judgement needs the query it is about")
    if judgement.verdict is Verdict.MISSING and judgement.rank_at_judgement is not None:
        # The point of `missing` is that the item was not in the ranking. A rank
        # on it would silently corrupt any recall computed from this table; the
        # CHECK constraint refuses it too, but this says why.
        raise InvalidJudgement(
            "a 'missing' verdict cannot carry a rank: it names something the "
            "search did not return"
        )

    values = {
        "id": new_id(),
        "query_text": query_text,
        "source_name": judgement.source_name,
        "external_key": judgement.external_key,
        "memory_id": judgement.memory_id,
        "chunk_id": judgement.chunk_id,
        "verdict": judgement.verdict.value,
        "rank_at_judgement": judgement.rank_at_judgement,
        "score_at_judgement": judgement.score_at_judgement,
        "filters": judgement.filters or {},
    }

    stmt = (
        insert(models.QueryJudgement)
        .values(**values)
        .on_conflict_do_update(
            constraint="uq_query_judgements_query_item",
            # Everything except the identity is replaced, including the
            # snapshots: a re-judgement is a fresh opinion about a fresh ranking,
            # so keeping the old rank would misattribute it.
            set_={
                "verdict": values["verdict"],
                "memory_id": values["memory_id"],
                "chunk_id": values["chunk_id"],
                "rank_at_judgement": values["rank_at_judgement"],
                "score_at_judgement": values["score_at_judgement"],
                "filters": values["filters"],
                "judged_at": func.now(),
            },
        )
        .returning(models.QueryJudgement.id)
    )

    async with session_factory.begin() as session:
        judgement_id: UUID = (await session.execute(stmt)).scalar_one()

    logger.info(
        "judgement.recorded",
        query=query_text,
        item=judgement.external_key,
        verdict=judgement.verdict.value,
    )
    return judgement_id


@dataclass(frozen=True, slots=True)
class QuerySummary:
    query_text: str
    relevant: int
    not_relevant: int
    missing: int
    last_judged_at: datetime

    @property
    def total(self) -> int:
        return self.relevant + self.not_relevant + self.missing


async def summarise(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[QuerySummary]:
    """One row per query, with its verdict counts. What the /judgements page shows."""
    stmt = (
        select(
            models.QueryJudgement.query_text,
            func.count().filter(models.QueryJudgement.verdict == Verdict.RELEVANT.value),
            func.count().filter(
                models.QueryJudgement.verdict == Verdict.NOT_RELEVANT.value
            ),
            func.count().filter(models.QueryJudgement.verdict == Verdict.MISSING.value),
            func.max(models.QueryJudgement.judged_at),
        )
        .group_by(models.QueryJudgement.query_text)
        .order_by(func.max(models.QueryJudgement.judged_at).desc())
    )
    async with session_factory() as session:
        return [
            QuerySummary(
                query_text=row[0],
                relevant=row[1],
                not_relevant=row[2],
                missing=row[3],
                last_judged_at=row[4],
            )
            for row in await session.execute(stmt)
        ]


@dataclass(frozen=True, slots=True)
class GoldenItem:
    external_key: str
    source_name: str
    verdict: str
    rank_at_judgement: int | None
    score_at_judgement: float | None
    judged_at: str
    # Resolved at export time from the natural key. Null means the item is no
    # longer in the corpus — the judgement is still valid history, but nothing
    # can be scored against it, and a consumer needs to be able to tell.
    memory_id: str | None


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    query_text: str
    filters: dict[str, Any]
    items: list[GoldenItem]

    @property
    def relevant_keys(self) -> list[str]:
        """The answer key: everything a good ranking should return for this query."""
        return sorted(
            item.external_key
            for item in self.items
            if item.verdict in (Verdict.RELEVANT.value, Verdict.MISSING.value)
        )


@dataclass(frozen=True, slots=True)
class GoldenSet:
    generated_at: str
    queries: list[GoldenQuery]

    @property
    def totals(self) -> dict[str, int]:
        items = [item for query in self.queries for item in query.items]
        return {
            "queries": len(self.queries),
            "judgements": len(items),
            "relevant": sum(1 for i in items if i.verdict == Verdict.RELEVANT.value),
            "not_relevant": sum(
                1 for i in items if i.verdict == Verdict.NOT_RELEVANT.value
            ),
            "missing": sum(1 for i in items if i.verdict == Verdict.MISSING.value),
            "unresolved": sum(1 for i in items if i.memory_id is None),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "totals": self.totals,
            "queries": [
                {
                    "query_text": query.query_text,
                    "filters": query.filters,
                    "relevant_keys": query.relevant_keys,
                    "items": [
                        {
                            "external_key": item.external_key,
                            "source_name": item.source_name,
                            "verdict": item.verdict,
                            "rank_at_judgement": item.rank_at_judgement,
                            "score_at_judgement": item.score_at_judgement,
                            "judged_at": item.judged_at,
                            "memory_id": item.memory_id,
                        }
                        for item in query.items
                    ],
                }
                for query in self.queries
            ],
        }


async def export_golden_set(
    session_factory: async_sessionmaker[AsyncSession], *, now: datetime
) -> GoldenSet:
    """The labelled data, grouped by query, with ids re-resolved.

    `memory_id` is looked up from `(source_name, external_key)` against the
    *current* corpus rather than read from the stored snapshot. That is the whole
    reason identity is a natural key: after a rebuild the stored pointer is
    stale, and an export that handed M2.0 stale ids would produce an evaluation
    that silently scored nothing.

    `now` is passed in rather than read here, so a caller can produce a
    byte-stable export in a test.
    """
    stmt = select(models.QueryJudgement).order_by(
        models.QueryJudgement.query_text,
        models.QueryJudgement.rank_at_judgement.nulls_last(),
        models.QueryJudgement.external_key,
    )
    async with session_factory() as session:
        rows = list((await session.execute(stmt)).scalars())

        current = {
            (row[0], row[1]): str(row[2])
            for row in await session.execute(
                select(
                    models.Source.name,
                    models.Memory.external_key,
                    models.Memory.id,
                )
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(
                    models.Memory.is_current.is_(True),
                    models.Memory.deleted_at.is_(None),
                )
            )
        }

    grouped: dict[str, list[models.QueryJudgement]] = {}
    filters: dict[str, dict[str, Any]] = {}
    for row in rows:
        grouped.setdefault(row.query_text, []).append(row)
        # The filters of the most recent judgement for the query. They are
        # recorded per row because a verdict is only meaningful under the filters
        # it was given, but a query-level export needs one answer.
        filters[row.query_text] = dict(row.filters)

    return GoldenSet(
        generated_at=now.isoformat(),
        queries=[
            GoldenQuery(
                query_text=query_text,
                filters=filters[query_text],
                items=[
                    GoldenItem(
                        external_key=row.external_key,
                        source_name=row.source_name,
                        verdict=row.verdict,
                        rank_at_judgement=row.rank_at_judgement,
                        score_at_judgement=(
                            None
                            if row.score_at_judgement is None
                            else round(float(row.score_at_judgement), 6)
                        ),
                        judged_at=row.judged_at.isoformat(),
                        memory_id=current.get((row.source_name, row.external_key)),
                    )
                    for row in judgements
                ],
            )
            for query_text, judgements in sorted(grouped.items())
        ],
    )
