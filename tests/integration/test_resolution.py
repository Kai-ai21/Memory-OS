"""Resolution against a real database.

The three required tests that need one: a merge repoints and marks rather than
deletes, an unmerge restores exactly, and a below-threshold candidate is
recorded rather than applied.

`FakeEmbedder` throughout. Whether the real model puts "Postgres" and
"PostgreSQL" close together is the corpus measurement's question and cannot be
settled here; what these establish is what the system does with a candidate once
it has one.
"""

from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.resolution import (
    MergeCandidate,
    ResolveEntities,
)
from memoryos.domain.ids import new_id
from memoryos.domain.values import EntityType, MergeStatus, MergeStrategy
from tests.integration.conftest import Pipeline
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


async def make_entity(
    sessions: async_sessionmaker[AsyncSession],
    name: str,
    *,
    canonical: str | None = None,
    entity_type: EntityType = EntityType.TECHNOLOGY,
) -> UUID:
    entity_id = new_id()
    async with sessions.begin() as session:
        session.add(
            models.Entity(
                id=entity_id,
                name=name,
                canonical_name=canonical or name.lower(),
                type=entity_type.value,
                confidence=0.9,
            )
        )
    return entity_id


async def add_mention(
    sessions: async_sessionmaker[AsyncSession],
    entity_id: UUID,
    memory_id: UUID,
    chunk_id: UUID,
    char_start: int,
) -> UUID:
    mention_id = new_id()
    async with sessions.begin() as session:
        session.add(
            models.EntityMention(
                id=mention_id,
                entity_id=entity_id,
                memory_id=memory_id,
                chunk_id=chunk_id,
                char_start=char_start,
                char_end=char_start + 5,
                confidence=0.9,
                extractor_version="test@1",
            )
        )
    return mention_id


async def a_chunk(pipeline: Pipeline) -> tuple[UUID, UUID]:
    """One real (memory_id, chunk_id) to hang mentions from."""
    await pipeline.ingest()
    async with pipeline.sessions() as session:
        row = (
            await session.execute(
                select(models.MemoryChunk.memory_id, models.MemoryChunk.id).limit(1)
            )
        ).one()
    return row[0], row[1]


def resolver(pipeline: Pipeline, **kwargs: float) -> ResolveEntities:
    return ResolveEntities(pipeline.sessions, FakeEmbedder(), **kwargs)


async def mentions_of(
    sessions: async_sessionmaker[AsyncSession], entity_id: UUID
) -> int:
    async with sessions() as session:
        return int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(models.EntityMention)
                    .where(models.EntityMention.entity_id == entity_id)
                )
            ).scalar_one()
        )


# --------------------------------------------------------------------------
# A merge repoints and marks; it does not delete
# --------------------------------------------------------------------------


async def test_a_merge_repoints_mentions_and_marks_the_loser(
    pipeline: Pipeline,
) -> None:
    """The shape of every merge, and the reason nothing is deleted.

    Resolution is never perfect — this milestone's own report names the merges
    it got wrong. A merge that removed the losing row would make each of those
    errors permanent: the name, the surface form and the evidence all vanish,
    and no amount of re-running the resolver brings back an entity whose only
    record was that row.
    """
    memory_id, chunk_id = await a_chunk(pipeline)
    winner = await make_entity(pipeline.sessions, "React")
    loser = await make_entity(pipeline.sessions, "React.js", canonical="react.js")
    await add_mention(pipeline.sessions, winner, memory_id, chunk_id, 0)
    moved_mention = await add_mention(pipeline.sessions, loser, memory_id, chunk_id, 50)

    moved = await resolver(pipeline).apply(
        MergeCandidate(winner, loser, MergeStrategy.EXACT, 1.0, "both -> react")
    )

    assert moved == 1
    assert await mentions_of(pipeline.sessions, winner) == 2
    assert await mentions_of(pipeline.sessions, loser) == 0

    async with pipeline.sessions() as session:
        loser_row = await session.get(models.Entity, loser)
        assert loser_row is not None, "the losing entity must survive the merge"
        assert loser_row.merged_into_id == winner
        assert loser_row.name == "React.js", "the surface form is the evidence"

        moved_row = await session.get(models.EntityMention, moved_mention)
        assert moved_row is not None
        assert moved_row.entity_id == winner

        merge = (
            await session.execute(select(models.EntityMerge))
        ).scalar_one()
        assert merge.status == MergeStatus.APPLIED.value
        assert merge.merged_at is not None
        assert merge.moved_mention_ids == [str(moved_mention)]


async def test_the_entity_with_more_mentions_wins(pipeline: Pipeline) -> None:
    """Winners are chosen, not taken in argument order.

    The heavier entity survives because most of the corpus already points at it,
    so the merge moves the fewest rows — and because the rule has to be total,
    or two runs over the same corpus disagree about which name survived.
    """
    memory_id, chunk_id = await a_chunk(pipeline)
    light = await make_entity(pipeline.sessions, "pg", canonical="pg")
    heavy = await make_entity(pipeline.sessions, "postgres", canonical="postgres")
    await add_mention(pipeline.sessions, light, memory_id, chunk_id, 0)
    for offset in (10, 20, 30):
        await add_mention(pipeline.sessions, heavy, memory_id, chunk_id, offset)

    # Passed light-first, so order cannot be what decides.
    await resolver(pipeline).apply(
        MergeCandidate(light, heavy, MergeStrategy.ALIAS, 0.99, "alias")
    )

    async with pipeline.sessions() as session:
        heavy_row = await session.get(models.Entity, heavy)
        light_row = await session.get(models.Entity, light)
        assert heavy_row is not None and light_row is not None
        assert heavy_row.merged_into_id is None
        assert light_row.merged_into_id == heavy


# --------------------------------------------------------------------------
# Unmerge restores exactly
# --------------------------------------------------------------------------


async def test_unmerge_restores_the_previous_state_exactly(
    pipeline: Pipeline,
) -> None:
    """Exactly, which is why the moved ids are recorded rather than inferred.

    After a repoint, nothing distinguishes the mentions that came from the loser
    from the ones the winner always had. An unmerge that guessed would take the
    winner's own mentions with it on any entity that had some — and this winner
    has some, which is what makes the assertion meaningful.
    """
    memory_id, chunk_id = await a_chunk(pipeline)
    winner = await make_entity(pipeline.sessions, "React")
    loser = await make_entity(pipeline.sessions, "React.js", canonical="react.js")
    winner_own = await add_mention(pipeline.sessions, winner, memory_id, chunk_id, 0)
    loser_own = await add_mention(pipeline.sessions, loser, memory_id, chunk_id, 50)

    resolve = resolver(pipeline)
    await resolve.apply(
        MergeCandidate(winner, loser, MergeStrategy.EXACT, 1.0, "both -> react")
    )

    async with pipeline.sessions() as session:
        merge_id = (
            await session.execute(select(models.EntityMerge.id))
        ).scalar_one()

    restored = await resolve.revert(merge_id)

    assert restored == 1
    async with pipeline.sessions() as session:
        # Each mention is back where it started — including the winner's own,
        # which must not have moved in either direction.
        winner_row = await session.get(models.EntityMention, winner_own)
        loser_row = await session.get(models.EntityMention, loser_own)
        loser_entity = await session.get(models.Entity, loser)
        assert winner_row is not None and loser_row is not None
        assert loser_entity is not None
        assert winner_row.entity_id == winner
        assert loser_row.entity_id == loser
        assert loser_entity.merged_into_id is None

        merge = await session.get(models.EntityMerge, merge_id)
        assert merge is not None
        assert merge.status == MergeStatus.REVERTED.value
        assert merge.reverted_at is not None


async def test_a_reverted_merge_cannot_be_reverted_twice(pipeline: Pipeline) -> None:
    """The second revert would move mentions that are already home."""
    memory_id, chunk_id = await a_chunk(pipeline)
    winner = await make_entity(pipeline.sessions, "React")
    loser = await make_entity(pipeline.sessions, "React.js", canonical="react.js")
    await add_mention(pipeline.sessions, loser, memory_id, chunk_id, 0)

    resolve = resolver(pipeline)
    await resolve.apply(MergeCandidate(winner, loser, MergeStrategy.EXACT, 1.0, "e"))
    async with pipeline.sessions() as session:
        merge_id = (await session.execute(select(models.EntityMerge.id))).scalar_one()

    await resolve.revert(merge_id)
    with pytest.raises(ValueError, match="reverted"):
        await resolve.revert(merge_id)


# --------------------------------------------------------------------------
# Below the threshold, nothing is merged
# --------------------------------------------------------------------------


async def test_a_below_threshold_candidate_is_queued_and_not_merged(
    pipeline: Pipeline,
) -> None:
    """The review queue, and the asymmetry that justifies it.

    A false merge invents a path the corpus does not contain, and every
    traversal through it reports a connection nobody wrote. So the uncertain
    band is recorded for a person rather than guessed at — and recording it is
    what stops the same pair being re-proposed on every run forever.
    """
    memory_id, chunk_id = await a_chunk(pipeline)
    left = await make_entity(pipeline.sessions, "memory_id", canonical="memory_id")
    right = await make_entity(pipeline.sessions, "memories", canonical="memories")
    await add_mention(pipeline.sessions, left, memory_id, chunk_id, 0)
    await add_mention(pipeline.sessions, right, memory_id, chunk_id, 50)

    resolve = resolver(pipeline, threshold=0.95)
    created = await resolve.record_pending(
        MergeCandidate(left, right, MergeStrategy.EMBEDDING, 0.88, "cosine 0.880")
    )

    assert created is True
    # Nothing moved, nothing marked.
    assert await mentions_of(pipeline.sessions, left) == 1
    assert await mentions_of(pipeline.sessions, right) == 1
    async with pipeline.sessions() as session:
        left_row = await session.get(models.Entity, left)
        right_row = await session.get(models.Entity, right)
        assert left_row is not None and right_row is not None
        assert left_row.merged_into_id is None
        assert right_row.merged_into_id is None

        merge = (await session.execute(select(models.EntityMerge))).scalar_one()
        assert merge.status == MergeStatus.PENDING.value
        assert merge.merged_at is None
        assert "cosine" in merge.evidence, "a reviewer cannot judge a bare number"


async def test_the_same_pair_is_not_queued_twice(pipeline: Pipeline) -> None:
    """A re-run must not grow the review queue by a copy of itself."""
    left = await make_entity(pipeline.sessions, "memory_id", canonical="memory_id")
    right = await make_entity(pipeline.sessions, "memories", canonical="memories")
    candidate = MergeCandidate(left, right, MergeStrategy.EMBEDDING, 0.88, "cosine")

    resolve = resolver(pipeline)
    assert await resolve.record_pending(candidate) is True
    assert await resolve.record_pending(candidate) is False

    async with pipeline.sessions() as session:
        assert (
            await session.execute(select(func.count()).select_from(models.EntityMerge))
        ).scalar_one() == 1


async def test_extraction_after_a_merge_does_not_resurrect_the_loser(
    pipeline: Pipeline,
) -> None:
    """The silent-undo this milestone had to guard against.

    `_upsert_entity` matches on `(canonical_name, type)`, and a merged-away row
    still holds both. Without following the pointer, the next extraction
    re-attaches mentions to an entity that was merged away — no error, no log,
    just a duplicate quietly coming back to life and the counts drifting up
    after every sync.
    """
    from memoryos.application.extraction import ExtractEntities
    from tests.support.fakes import FakeEntityExtractor

    memory_id, chunk_id = await a_chunk(pipeline)
    winner = await make_entity(pipeline.sessions, "React")
    loser = await make_entity(pipeline.sessions, "React.js", canonical="react.js")
    # The winner needs the heavier mention count, because that — not argument
    # order — is what `_pick_winner` uses. Giving the loser more would make it
    # win, and this test would be asserting the opposite of its own name.
    for offset in (0, 10, 20):
        await add_mention(pipeline.sessions, winner, memory_id, chunk_id, offset)
    await add_mention(pipeline.sessions, loser, memory_id, chunk_id, 100)

    await resolver(pipeline).apply(
        MergeCandidate(winner, loser, MergeStrategy.EXACT, 1.0, "e")
    )

    # An extractor that finds exactly the losing name again.
    extract = ExtractEntities(
        pipeline.sessions,
        FakeEntityExtractor(pattern=r"React\.js", entity_type=EntityType.TECHNOLOGY),
    )
    async with pipeline.sessions.begin() as session:
        chunk = await session.get(models.MemoryChunk, chunk_id)
        assert chunk is not None
        chunk.content = "React.js is mentioned here again."

    await extract(memory_id)

    async with pipeline.sessions() as session:
        loser_row = await session.get(models.Entity, loser)
        assert loser_row is not None
        assert loser_row.merged_into_id == winner
        # Every new mention landed on the winner, not the merged-away row.
        assert (
            await session.execute(
                select(func.count())
                .select_from(models.EntityMention)
                .where(models.EntityMention.entity_id == loser)
            )
        ).scalar_one() == 0
