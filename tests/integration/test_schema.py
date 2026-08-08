import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from memoryos.adapters.db import models
from memoryos.adapters.db.mappers import to_memory_row
from memoryos.adapters.db.repositories import SqlAlchemyArtifactRepository
from memoryos.domain.entities import RawArtifact, Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash
from tests.integration.conftest import OCCURRED_AT, build_memory

pytestmark = pytest.mark.integration


async def test_inserting_the_same_content_hash_twice_raises(
    session: AsyncSession, artifact: RawArtifact
) -> None:
    # Identity is a pure function of content, so a re-read of an unchanged file
    # collides on the primary key. That collision is what makes ingestion
    # idempotent rather than duplicating rows.
    duplicate = RawArtifact(content_hash=artifact.content_hash, byte_size=999)
    with pytest.raises(IntegrityError):
        await SqlAlchemyArtifactRepository(session).add(duplicate)


async def test_malformed_content_hash_is_rejected_by_the_database(
    session: AsyncSession,
) -> None:
    # The domain value object refuses this too; the CHECK covers every other
    # writer, psql included.
    with pytest.raises(IntegrityError, match="ck_raw_artifacts_content_hash_hex"):
        await session.execute(
            text(
                "INSERT INTO raw_artifacts (content_hash, byte_size) "
                "VALUES ('NOT-A-HASH', 0)"
            )
        )


async def test_two_current_versions_of_one_item_violate_the_partial_unique_index(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    first = build_memory(source, artifact)
    session.add(to_memory_row(first))
    await session.flush()

    second = build_memory(source, artifact, id=new_id(), version=2)
    session.add(to_memory_row(second))

    with pytest.raises(IntegrityError, match="uq_memories_current_version"):
        await session.flush()


async def test_superseded_versions_may_share_the_source_and_key(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    # The index is partial for exactly this reason: history has to be allowed to
    # accumulate under the same (source_id, external_key).
    session.add(to_memory_row(build_memory(source, artifact, is_current=False)))
    session.add(to_memory_row(build_memory(source, artifact, version=2, is_current=False)))
    await session.flush()


async def test_deleting_a_memory_cascades_to_its_chunks(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    memory = build_memory(source, artifact)
    session.add(to_memory_row(memory))
    await session.flush()

    for ordinal in range(3):
        session.add(
            models.MemoryChunk(
                id=new_id(),
                memory_id=memory.id,
                ordinal=ordinal,
                content=f"chunk {ordinal}",
                token_count=2,
                char_start=ordinal * 10,
                char_end=ordinal * 10 + 9,
                chunker_version="test-v1",
                content_hash=ContentHash.of(f"chunk {ordinal}".encode()).value,
            )
        )
    await session.flush()
    assert await chunk_count(session, memory.id) == 3

    await session.execute(delete(models.Memory).where(models.Memory.id == memory.id))

    # Without ON DELETE CASCADE these rows would survive, stay in the vector
    # index, and keep surfacing in search results for a memory that is gone.
    assert await chunk_count(session, memory.id) == 0


async def test_null_occurred_at_with_known_provenance_violates_the_check(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    # Written as raw SQL so the entity's __post_init__ cannot be what catches
    # it: the guarantee has to hold at the database level.
    with pytest.raises(IntegrityError, match="ck_memories_occurred_at_provenance"):
        await session.execute(
            text(
                "INSERT INTO memories "
                "(id, source_id, external_key, content_hash, kind, "
                " occurred_at, occurred_at_source) "
                "VALUES (:id, :source_id, 'notes/raw.md', :content_hash, 'note', "
                " NULL, 'declared')"
            ),
            {
                "id": new_id(),
                "source_id": source.id,
                "content_hash": artifact.content_hash.value,
            },
        )


async def test_known_occurred_at_with_unknown_provenance_violates_the_check(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    with pytest.raises(IntegrityError, match="ck_memories_occurred_at_provenance"):
        await session.execute(
            text(
                "INSERT INTO memories "
                "(id, source_id, external_key, content_hash, kind, "
                " occurred_at, occurred_at_source) "
                "VALUES (:id, :source_id, 'notes/raw.md', :content_hash, 'note', "
                " :occurred_at, 'unknown')"
            ),
            {
                "id": new_id(),
                "source_id": source.id,
                "content_hash": artifact.content_hash.value,
                "occurred_at": OCCURRED_AT,
            },
        )


async def test_importance_outside_the_unit_interval_violates_the_check(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    session.add(to_memory_row(build_memory(source, artifact)))
    await session.flush()

    with pytest.raises(IntegrityError, match="ck_memories_importance_range"):
        await session.execute(
            text("UPDATE memories SET importance = 1.5 WHERE source_id = :source_id"),
            {"source_id": source.id},
        )


async def test_chunk_span_must_be_non_empty_at_the_database_level(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    memory = build_memory(source, artifact)
    session.add(to_memory_row(memory))
    await session.flush()

    session.add(
        models.MemoryChunk(
            id=new_id(),
            memory_id=memory.id,
            ordinal=0,
            content="",
            token_count=1,
            char_start=10,
            char_end=10,
            chunker_version="test-v1",
            content_hash=ContentHash.of(b"").value,
        )
    )
    with pytest.raises(IntegrityError, match="ck_memory_chunks_char_range"):
        await session.flush()


async def chunk_count(session: AsyncSession, memory_id: object) -> int:
    stmt = select(func.count()).select_from(models.MemoryChunk).where(
        models.MemoryChunk.memory_id == memory_id
    )
    return (await session.execute(stmt)).scalar_one()
