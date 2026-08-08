from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.db.repositories import (
    SqlAlchemyArtifactRepository,
    SqlAlchemySourceRepository,
)
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.embed import EmbedMemory, EmbedReport
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.sync import SyncSource
from memoryos.domain.entities import Memory, RawArtifact, Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import ContentHash, MemoryKind, SourceKind, TimeProvenance
from tests.support.fakes import FakeEmbedder

OCCURRED_AT = datetime(2023, 4, 1, 12, 0, tzinfo=UTC)

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog and keeps running onward. "
    "Every good boy deserves fudge and a reasonable amount of rest. "
)


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


@dataclass(slots=True)
class Pipeline:
    root: Path
    source: Source
    sync: SyncSource
    normalize: NormalizeMemory
    embedder: FakeEmbedder
    cache: PostgresEmbeddingCache
    sessions: async_sessionmaker[AsyncSession]
    batch_size: int = 32

    def embedder_for(self, embedder: FakeEmbedder | None = None) -> EmbedMemory:
        return EmbedMemory(
            self.sessions, embedder or self.embedder, self.cache, self.batch_size
        )

    async def ingest(self) -> None:
        """Sync and normalize, leaving chunks with null embeddings."""
        await self.sync(self.source.id, full=True)
        for memory_id in await self.job_targets(JobType.NORMALIZE_MEMORY):
            await self.normalize(memory_id)
        await self.clear_jobs(JobType.NORMALIZE_MEMORY)

    async def embed_all(self, embedder: FakeEmbedder | None = None) -> list[EmbedReport]:
        embed = self.embedder_for(embedder)
        targets = await self.job_targets(JobType.EMBED_MEMORY)
        reports = [await embed(memory_id) for memory_id in targets]
        await self.clear_jobs(JobType.EMBED_MEMORY)
        return reports

    async def job_targets(self, job_type: JobType) -> list[UUID]:
        async with self.sessions() as session:
            rows = await session.execute(
                select(models.Job.payload).where(models.Job.job_type == job_type.value)
            )
            return [UUID(row[0]["memory_id"]) for row in rows]

    async def clear_jobs(self, job_type: JobType) -> None:
        async with self.sessions.begin() as session:
            await session.execute(
                delete(models.Job).where(models.Job.job_type == job_type.value)
            )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "guide.md").write_text("# Guide\n\n" + PARAGRAPH * 6 + "\n")
    (root / "notes.txt").write_text(PARAGRAPH * 3 + "\n")
    return root


@pytest.fixture
async def pipeline(
    tree: Path, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> Pipeline:
    source = Source(
        id=new_id(),
        kind=SourceKind.FILESYSTEM,
        name="corpus",
        config={"root": str(tree)},
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    embedder = FakeEmbedder()
    return Pipeline(
        root=tree,
        source=source,
        sync=SyncSource(sessions, FilesystemConnector(blobs), blobs),
        normalize=NormalizeMemory(sessions, blobs, build_parsers(), StructuralChunker(embedder)),
        embedder=embedder,
        cache=PostgresEmbeddingCache(sessions),
        sessions=sessions,
    )
