"""Reading and hand-editing the merge ledger.

The review queue is the half of resolution that is not automatic, and it exists
because the alternative is worse in both directions. A resolver that merged
everything it suspected would corrupt the graph with paths nobody wrote; one
that merged only what it was certain of would leave most of the duplicates
standing. The middle band is real, and the honest thing to do with it is show it
to somebody.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.values import MergeStatus


@dataclass(frozen=True, slots=True)
class MergeRow:
    """One ledger entry, with both names resolved for reading."""

    id: UUID
    winner_id: UUID
    loser_id: UUID
    winner_name: str
    loser_name: str
    entity_type: str
    strategy: str
    status: str
    confidence: float
    evidence: str
    moved_mentions: int


async def list_merges(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: MergeStatus | None = None,
    strategy: str | None = None,
    limit: int = 100,
) -> list[MergeRow]:
    """The ledger, newest and most confident first.

    Both entity names are joined in rather than left as ids. A review queue of
    UUID pairs is not a review queue — the reviewer would have to look up every
    row by hand to make the judgement being asked of them.
    """
    winner = models.Entity.__table__.alias("winner")
    loser = models.Entity.__table__.alias("loser")

    stmt = (
        select(
            models.EntityMerge.id,
            models.EntityMerge.winner_id,
            models.EntityMerge.loser_id,
            winner.c.name,
            loser.c.name,
            winner.c.type,
            models.EntityMerge.strategy,
            models.EntityMerge.status,
            models.EntityMerge.confidence,
            models.EntityMerge.evidence,
            func.jsonb_array_length(models.EntityMerge.moved_mention_ids),
        )
        .join(winner, winner.c.id == models.EntityMerge.winner_id)
        .join(loser, loser.c.id == models.EntityMerge.loser_id)
        .order_by(models.EntityMerge.confidence.desc(), models.EntityMerge.proposed_at)
        .limit(limit)
    )
    if status is not None:
        stmt = stmt.where(models.EntityMerge.status == status.value)
    if strategy is not None:
        stmt = stmt.where(models.EntityMerge.strategy == strategy)

    async with session_factory() as session:
        return [
            MergeRow(
                id=row[0],
                winner_id=row[1],
                loser_id=row[2],
                winner_name=row[3],
                loser_name=row[4],
                entity_type=row[5],
                strategy=row[6],
                status=row[7],
                confidence=row[8],
                evidence=row[9],
                moved_mentions=row[10],
            )
            for row in await session.execute(stmt)
        ]


async def find_entity(
    session_factory: async_sessionmaker[AsyncSession], reference: str
) -> UUID:
    """An entity id from an id or a name, for the hand-merge commands.

    Names are accepted because nobody types a UUID by choice, and the id is
    accepted because names are ambiguous. An ambiguous name raises rather than
    picking one: a merge command that guessed which "config" was meant would be
    a destructive operation resolving a coin toss.
    """
    try:
        return UUID(reference)
    except ValueError:
        pass

    async with session_factory() as session:
        matches = [
            (row[0], row[1], row[2])
            for row in await session.execute(
                select(models.Entity.id, models.Entity.name, models.Entity.type).where(
                    func.lower(models.Entity.name) == reference.lower(),
                    models.Entity.merged_into_id.is_(None),
                )
            )
        ]

    if not matches:
        raise LookupError(f"no active entity named {reference!r}")
    if len(matches) > 1:
        listed = ", ".join(f"{entity_id} ({kind})" for entity_id, _, kind in matches)
        raise LookupError(
            f"{reference!r} matches {len(matches)} entities: {listed}. "
            f"Use an id — a merge is destructive and this one is a guess."
        )
    found: UUID = matches[0][0]
    return found


async def mention_counts(
    session_factory: async_sessionmaker[AsyncSession], names: list[str]
) -> dict[str, tuple[int, int]]:
    """Per name: how many active entities carry it, and their total mentions.

    The measurement M3.2 is judged by. "postgres went from 4 nodes with 12
    mentions to 1 node with 12 mentions" is the milestone working; the same
    mentions under fewer nodes is exactly the shape of a successful merge, and
    a change in the mention total would mean one leaked.
    """
    result: dict[str, tuple[int, int]] = {}
    async with session_factory() as session:
        for name in names:
            row = (
                await session.execute(
                    select(
                        func.count(func.distinct(models.Entity.id)),
                        func.count(models.EntityMention.id),
                    )
                    .select_from(models.Entity)
                    .outerjoin(
                        models.EntityMention,
                        models.EntityMention.entity_id == models.Entity.id,
                    )
                    .where(
                        models.Entity.merged_into_id.is_(None),
                        func.lower(models.Entity.name).like(f"%{name.lower()}%"),
                    )
                )
            ).one()
            result[name] = (row[0], row[1])
    return result
