from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from memoryos.adapters.db.repositories import (
    SqlAlchemyArtifactRepository,
    SqlAlchemySourceRepository,
)
from memoryos.domain.entities import Memory, RawArtifact, Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash, MemoryKind, SourceKind, TimeProvenance

OCCURRED_AT = datetime(2023, 4, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
async def source(session: AsyncSession) -> Source:
    record = Source(id=new_id(), kind=SourceKind.FILESYSTEM, name=f"vault-{new_id()}")
    await SqlAlchemySourceRepository(session).add(record)
    return record


@pytest.fixture
async def artifact(session: AsyncSession) -> RawArtifact:
    record = RawArtifact(content_hash=ContentHash.of(b"first revision"), byte_size=14)
    await SqlAlchemyArtifactRepository(session).add(record)
    return record


@pytest.fixture
async def other_artifact(session: AsyncSession) -> RawArtifact:
    record = RawArtifact(content_hash=ContentHash.of(b"second revision"), byte_size=15)
    await SqlAlchemyArtifactRepository(session).add(record)
    return record


def build_memory(source: Source, artifact: RawArtifact, **overrides: object) -> Memory:
    fields: dict[str, object] = {
        "id": new_id(),
        "source_id": source.id,
        "external_key": "notes/example.md",
        "content_hash": artifact.content_hash,
        "kind": MemoryKind.NOTE,
        "occurred_at": OCCURRED_AT,
        "occurred_at_source": TimeProvenance.FILESYSTEM,
    }
    fields.update(overrides)
    return Memory(**fields)  # type: ignore[arg-type]
