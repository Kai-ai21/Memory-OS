from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from neo4j import AsyncGraphDatabase
from sqlalchemy import delete, func, select
from sqlalchemy import text as sa_text
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
from memoryos.adapters.db.shadow import SHADOW_SCHEMA, PostgresShadowSchema
from memoryos.adapters.graph.neo4j_store import Neo4jGraphStore
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.embed import EmbedMemory, EmbedReport
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.replay import ReplayCorpus
from memoryos.application.sync import SyncSource
from memoryos.application.verification import Snapshot, snapshot
from memoryos.config import Settings
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


# --------------------------------------------------------------------------
# The replay harness
#
# Shared rather than local to test_replay.py, because test_judgements.py needs
# the same corpus: the guarantee it checks is that a replay of a real corpus
# leaves human-authored rows alone, and that needs a real corpus to replay.
# --------------------------------------------------------------------------

QUEUE_TEXT = (
    "The worker claims a task from the queue and holds a lease on it while the "
    "handler runs to completion. Renewing that lease is how a long task keeps its "
    "hold on the work it started. "
)

BREAD_TEXT = (
    "A wild yeast starter is fed flour and water until it doubles reliably, then "
    "folded gently and given a long cold rest in the refrigerator. "
)


@dataclass(slots=True)
class Harness:
    """Everything needed to ingest a corpus and then rebuild it.

    The replay is built the way the container builds it — the real
    `NormalizeMemory` and `EmbedMemory`, constructed per session factory so a
    shadow rebuild writes through a different one. A test double for the pipeline
    would prove the double works.
    """

    root: Path
    source: Source
    sessions: async_sessionmaker[AsyncSession]
    blobs: FilesystemBlobStore
    embedder: FakeEmbedder
    sync: SyncSource
    replay: ReplayCorpus

    async def ingest(self) -> None:
        """Sync, then drain the queue inline the way a worker would."""
        await self.sync(self.source.id, full=True)
        for job_type, handler in (
            (JobType.NORMALIZE_MEMORY, self.normalize()),
            (JobType.EMBED_MEMORY, self.embed()),
        ):
            async with self.sessions() as session:
                targets = [
                    UUID(row[0]["memory_id"])
                    for row in await session.execute(
                        select(models.Job.payload).where(
                            models.Job.job_type == job_type.value
                        )
                    )
                ]
            for memory_id in targets:
                await handler(memory_id)
            async with self.sessions.begin() as session:
                await session.execute(
                    delete(models.Job).where(models.Job.job_type == job_type.value)
                )

        # M3.1: embedding queues an extraction job per memory, and M3.4 has
        # `SyncSource` queue a graph sync per memory. The harness has neither an
        # extractor nor a graph — these tests are about replay, not entities — so
        # both are dropped rather than run. Dropped here rather than suppressed at
        # the source, so the harness still exercises the real `SyncSource` and
        # `EmbedMemory`, followup enqueues included, and leaves the queue empty the
        # way a drained worker would.
        async with self.sessions.begin() as session:
            await session.execute(
                delete(models.Job).where(
                    models.Job.job_type.in_(
                        [JobType.EXTRACT_ENTITIES.value, JobType.SYNC_GRAPH.value]
                    )
                )
            )

    def normalize(self) -> NormalizeMemory:
        return NormalizeMemory(
            self.sessions, self.blobs, build_parsers(), StructuralChunker(self.embedder)
        )

    def embed(self) -> EmbedMemory:
        return EmbedMemory(
            self.sessions, self.embedder, PostgresEmbeddingCache(self.sessions)
        )

    async def snapshot(self) -> Snapshot:
        return await snapshot(self.sessions)

    async def count(self, table: type[models.Base]) -> int:
        async with self.sessions() as session:
            return int(
                (
                    await session.execute(select(func.count()).select_from(table))
                ).scalar_one()
            )

    async def chunk_ids(self) -> list[str]:
        async with self.sessions() as session:
            return sorted(
                str(row[0])
                for row in await session.execute(select(models.MemoryChunk.id))
            )


def build_harness(
    root: Path,
    blobs_root: Path,
    sessions: async_sessionmaker[AsyncSession],
    source: Source,
    settings: Settings,
) -> Harness:
    blobs = FilesystemBlobStore(blobs_root)
    embedder = FakeEmbedder()
    chunker = StructuralChunker(embedder)
    replay = ReplayCorpus(
        sessions,
        make_normalize=lambda factory: NormalizeMemory(
            factory, blobs, build_parsers(), chunker, enqueue_followup=False
        ),
        make_embed=lambda factory: EmbedMemory(
            factory,
            embedder,
            PostgresEmbeddingCache(factory),
            # Mirrors the container: a replay must not queue paid extraction
            # work. Without this the harness diverges from the wiring it exists
            # to stand in for, and a rebuild enqueues a model call per memory.
            enqueue_followup=False,
        ),
        make_shadow=lambda: PostgresShadowSchema(settings.database_url),
        blobs=blobs,
    )
    return Harness(
        root=root,
        source=source,
        sessions=sessions,
        blobs=blobs,
        embedder=embedder,
        sync=SyncSource(sessions, FilesystemConnector(blobs), blobs),
        replay=replay,
    )


async def shadow_schemas(sessions: async_sessionmaker[AsyncSession]) -> set[str]:
    """Any workspace schema still present. Should always be empty afterwards."""
    async with sessions() as session:
        return {
            row[0]
            for row in await session.execute(
                sa_text(
                    "SELECT nspname FROM pg_namespace WHERE nspname = :schema"
                ).bindparams(schema=SHADOW_SCHEMA)
            )
        }


async def add_source(
    sessions: async_sessionmaker[AsyncSession], name: str, root: Path
) -> Source:
    source = Source(
        id=new_id(), kind=SourceKind.FILESYSTEM, name=name, config={"root": str(root)}
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)
    return source


# --------------------------------------------------------------------------
# The graph harness
#
# Isolation here works differently from everywhere else in this suite, and it
# has to. `clean_database` truncates, because Postgres gives the suite its own
# database to truncate — `memos_test`, which exists precisely so that truncation
# can never reach a working corpus. Neo4j Community Edition has no equivalent:
# it supports exactly one user database, so a test run and a developer's own
# graph share it, and a `clear()` in a fixture would do to the graph exactly what
# the M2.0a incidents did to the corpus.
#
# So isolation is by identity instead of by database. Every id a test uses comes
# from `GraphFixture.new_id`, which records it; teardown deletes those nodes and
# nothing else. Fresh UUIDs cannot collide with anything already in the graph, so
# the tests are isolated from existing data in both directions without emptying
# it.
# --------------------------------------------------------------------------


@dataclass(slots=True)
class GraphFixture:
    """A graph store, plus the ids it is allowed to delete afterwards."""

    store: Neo4jGraphStore
    uri: str
    user: str
    password: str
    minted: list[str] = field(default_factory=list)

    def new_id(self) -> UUID:
        """A fresh id, remembered so teardown can find the node again."""
        value = new_id()
        self.minted.append(str(value))
        return value

    async def cleanup(self) -> None:
        """Delete only what this test created.

        Its own driver rather than the store's, because deleting by id is test
        infrastructure and not something `GraphStore` should offer — a port with
        a node-level delete on it is a port that invites a use case to keep the
        graph up to date by hand instead of rebuilding it.
        """
        if not self.minted:
            return
        driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            await driver.execute_query(
                "MATCH (n) WHERE n.memory_id IN $ids OR n.entity_id IN $ids "
                "OR n.source_id IN $ids DETACH DELETE n",
                {"ids": self.minted},
            )
        finally:
            await driver.close()


@pytest.fixture
async def graph(settings: Settings) -> AsyncIterator[GraphFixture]:
    """A reachable graph store, or a skipped test.

    Skipped rather than failed when Neo4j is absent, which is the same judgement
    the milestone makes everywhere else: the graph is optional infrastructure,
    and a developer without it running should still be able to run the suite for
    everything Phase 1 and Phase 2 do. CI starts a Neo4j service so the skip does
    not quietly become permanent.
    """
    store = Neo4jGraphStore(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
    )
    try:
        await store.verify()
    except Exception as exc:
        await store.close()
        pytest.skip(f"no Neo4j at {settings.neo4j_uri}: {exc}")

    fixture = GraphFixture(
        store=store,
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    try:
        yield fixture
    finally:
        await fixture.cleanup()
        await store.close()


@pytest.fixture
async def harness(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession], settings: Settings
) -> Harness:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "queue.md").write_text("# Queue\n\n" + QUEUE_TEXT * 4 + "\n")
    (root / "bread.txt").write_text(BREAD_TEXT * 4 + "\n")
    (root / "mod.py").write_text(
        "def handler(payload):\n"
        + "\n".join(f"    step_{n} = transform(payload, {n})" for n in range(60))
        + "\n    return payload\n"
    )
    # Two identical files, so the cache and the deduplication path are exercised.
    (root / "copy-a.txt").write_text(BREAD_TEXT * 2 + "\n")
    (root / "copy-b.txt").write_text(BREAD_TEXT * 2 + "\n")

    source = await add_source(sessions, "corpus", root)
    built = build_harness(root, tmp_path / "blobs", sessions, source, settings)
    await built.ingest()
    return built
