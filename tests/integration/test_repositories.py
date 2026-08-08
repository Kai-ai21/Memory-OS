from dataclasses import replace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import (
    SqlAlchemyArtifactRepository,
    SqlAlchemyEventLog,
    SqlAlchemyMemoryRepository,
    SqlAlchemySourceRepository,
)
from memoryos.domain.entities import IngestionEvent, RawArtifact, Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import (
    ContentHash,
    EventType,
    SourceKind,
    TimeProvenance,
)
from tests.integration.conftest import build_memory

pytestmark = pytest.mark.integration


async def test_source_round_trips_through_the_repository(
    session: AsyncSession, source: Source
) -> None:
    repository = SqlAlchemySourceRepository(session)

    stored = await repository.get(source.id)
    assert stored is not None
    # created_at is the database's to assign, so the entity that went in had
    # None there and the one that came back does not.
    assert stored.created_at is not None
    assert stored == replace(source, created_at=stored.created_at)

    assert await repository.get_by_name(SourceKind.FILESYSTEM, source.name) == stored
    assert await repository.get_by_name(SourceKind.FILESYSTEM, "no-such-source") is None
    assert await repository.get(new_id()) is None


async def test_update_cursor_replaces_the_stored_sync_state(
    session: AsyncSession, source: Source
) -> None:
    repository = SqlAlchemySourceRepository(session)

    await repository.update_cursor(source.id, {"last_path": "notes/z.md"})

    reloaded = await repository.get(source.id)
    assert reloaded is not None
    assert reloaded.cursor == {"last_path": "notes/z.md"}


async def test_artifact_exists_reflects_what_was_added(
    session: AsyncSession, artifact: RawArtifact
) -> None:
    repository = SqlAlchemyArtifactRepository(session)

    assert await repository.exists(artifact.content_hash) is True
    assert await repository.exists(ContentHash.of(b"never ingested")) is False


async def test_add_version_supersedes_the_previous_version(
    session: AsyncSession, source: Source, artifact: RawArtifact, other_artifact: RawArtifact
) -> None:
    repository = SqlAlchemyMemoryRepository(session)

    await repository.add_version(build_memory(source, artifact))
    await repository.add_version(
        build_memory(source, other_artifact, id=new_id(), title="revised")
    )

    stmt = (
        select(models.Memory)
        .where(models.Memory.source_id == source.id)
        .order_by(models.Memory.version)
    )
    rows = (await session.execute(stmt)).scalars().all()

    assert [(row.version, row.is_current) for row in rows] == [(1, False), (2, True)]

    current = await repository.get_current(source.id, "notes/example.md")
    assert current is not None
    assert current.version == 2
    assert current.title == "revised"
    assert current.content_hash == other_artifact.content_hash


async def test_add_version_starts_at_one_for_an_unseen_item(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    repository = SqlAlchemyMemoryRepository(session)

    await repository.add_version(build_memory(source, artifact, version=7))

    current = await repository.get_current(source.id, "notes/example.md")
    assert current is not None
    # The version is the repository's to assign, not the caller's.
    assert current.version == 1


async def test_get_current_returns_none_for_an_unknown_item(
    session: AsyncSession, source: Source
) -> None:
    repository = SqlAlchemyMemoryRepository(session)
    assert await repository.get_current(source.id, "notes/never-seen.md") is None


async def test_tombstone_marks_the_memory_deleted_without_removing_it(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    repository = SqlAlchemyMemoryRepository(session)
    memory = build_memory(source, artifact)
    await repository.add_version(memory)

    await repository.tombstone(memory.id)

    current = await repository.get_current(source.id, "notes/example.md")
    assert current is not None
    assert current.deleted_at is not None


async def test_append_then_replay_returns_events_in_seq_order(
    session: AsyncSession, source: Source, artifact: RawArtifact
) -> None:
    log = SqlAlchemyEventLog(session)
    keys = ["notes/a.md", "notes/b.md", "notes/c.md"]

    for key in keys:
        await log.append(
            IngestionEvent(
                id=new_id(),
                event_type=EventType.ARTIFACT_OBSERVED,
                source_id=source.id,
                external_key=key,
                occurred_at_source=TimeProvenance.UNKNOWN,
                content_hash=artifact.content_hash,
                payload={"path": key},
            )
        )

    replayed = await log.replay()

    assert [event.external_key for event in replayed] == keys
    sequences = [event.seq for event in replayed]
    assert None not in sequences
    assert sequences == sorted(sequences)  # type: ignore[type-var]
    assert replayed[0].payload == {"path": "notes/a.md"}
    assert replayed[0].schema_version == 1


async def test_replay_resumes_after_a_sequence_and_respects_the_limit(
    session: AsyncSession, source: Source
) -> None:
    log = SqlAlchemyEventLog(session)

    for key in ["notes/a.md", "notes/b.md", "notes/c.md"]:
        await log.append(
            IngestionEvent(
                id=new_id(),
                event_type=EventType.ITEM_DELETED,
                source_id=source.id,
                external_key=key,
                occurred_at_source=TimeProvenance.UNKNOWN,
            )
        )

    everything = await log.replay()
    first_seq = everything[0].seq
    assert first_seq is not None

    assert [event.external_key for event in await log.replay(after_seq=first_seq)] == [
        "notes/b.md",
        "notes/c.md",
    ]
    assert len(await log.replay(limit=2)) == 2


async def test_deletion_events_carry_no_content_hash(
    session: AsyncSession, source: Source
) -> None:
    log = SqlAlchemyEventLog(session)
    await log.append(
        IngestionEvent(
            id=new_id(),
            event_type=EventType.ITEM_DELETED,
            source_id=source.id,
            external_key="notes/gone.md",
            occurred_at_source=TimeProvenance.UNKNOWN,
        )
    )

    (event,) = await log.replay()
    assert event.content_hash is None
    assert event.event_type is EventType.ITEM_DELETED
