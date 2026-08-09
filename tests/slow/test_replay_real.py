"""Replay the real corpus with the real model, and ask the real questions.

The fast suite proves the rebuild is byte-identical against a deterministic fake.
That is the stronger structural assertion, but it cannot answer the question an
operator actually has: after a rebuild, does search still work?

So this ingests this repository — the corpus the M1.6 report was written against —
records the top result for each of the four assessment queries, rebuilds the whole
corpus from the log, and asks again. Same answers, or the replay is not usable
however clean its diff looked.
"""

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
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.db.shadow import PostgresShadowSchema
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.embedding.sentence_transformers import SentenceTransformerEmbedder
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.embed import EmbedMemory
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.replay import ReplayCorpus, truncate_derived
from memoryos.application.search import SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.application.verification import compare, snapshot
from memoryos.config import Settings
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import SourceKind

pytestmark = [pytest.mark.slow, pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The four queries the M1.6 report was assessed on, two of which M1.6.1 fixed.
# Reused verbatim, because the point is that a rebuild changes none of them.
ASSESSMENT_QUERIES = (
    "how does the job queue claim work",
    "why do we store two timestamps",
    "content addressing and deduplication",
    "what happens when a file is deleted",
)


class Corpus:
    """This repository, ingested, plus the pieces needed to rebuild and search it."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        blobs: FilesystemBlobStore,
        embedder: SentenceTransformerEmbedder,
        settings: Settings,
    ) -> None:
        self.sessions = sessions
        self.blobs = blobs
        self.embedder = embedder
        self.chunker = StructuralChunker(embedder)
        self.search = SearchMemories(sessions, embedder, PgVectorStore(sessions, embedder))
        self.replay = ReplayCorpus(
            sessions,
            make_normalize=lambda factory: NormalizeMemory(
                factory, blobs, build_parsers(), self.chunker, enqueue_followup=False
            ),
            make_embed=lambda factory: EmbedMemory(
                factory, embedder, PostgresEmbeddingCache(factory)
            ),
            make_shadow=lambda: PostgresShadowSchema(settings.database_url),
        )

    async def top_results(self) -> dict[str, str]:
        """The best-ranked memory for each assessment query."""
        results: dict[str, str] = {}
        for query in ASSESSMENT_QUERIES:
            found = await self.search(query, k=5)
            assert found.hits, f"{query!r} returned nothing at all"
            results[query] = found.hits[0].external_key
        return results


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@pytest.fixture
async def corpus(
    tmp_path: Path,
    sessions: async_sessionmaker[AsyncSession],
    embedder: SentenceTransformerEmbedder,
    settings: Settings,
) -> Corpus:
    source = Source(
        id=new_id(),
        kind=SourceKind.FILESYSTEM,
        name="self",
        config={"root": str(REPO_ROOT)},
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    built = Corpus(sessions, blobs, embedder, settings)
    await SyncSource(sessions, FilesystemConnector(blobs), blobs)(source.id, full=True)

    normalize = NormalizeMemory(sessions, blobs, build_parsers(), built.chunker)
    embed = EmbedMemory(sessions, embedder, PostgresEmbeddingCache(sessions))
    for job_type, handler in (
        (JobType.NORMALIZE_MEMORY, normalize),
        (JobType.EMBED_MEMORY, embed),
    ):
        async with sessions() as session:
            targets = [
                UUID(row[0]["memory_id"])
                for row in await session.execute(
                    select(models.Job.payload).where(models.Job.job_type == job_type.value)
                )
            ]
        for memory_id in targets:
            await handler(memory_id)
        async with sessions.begin() as session:
            await session.execute(
                delete(models.Job).where(models.Job.job_type == job_type.value)
            )
    return built


async def test_the_repository_corpus_rebuilds_identically(corpus: Corpus) -> None:
    """The real thing: a real corpus, a real model, a real rebuild."""
    before = await snapshot(corpus.sessions)
    assert len(before.memories) > 50, f"only {len(before.memories)} memories ingested"
    assert before.counts["embedded_chunks"] == before.counts["chunks"]

    await truncate_derived(corpus.sessions, clear_cache=False)
    report = await corpus.replay()

    after = await snapshot(corpus.sessions)
    result = compare(before, after)
    assert result.identical, result.render()
    assert report.chunks == before.counts["chunks"]


async def test_verification_passes_on_the_real_corpus(corpus: Corpus) -> None:
    before = await snapshot(corpus.sessions)

    async with corpus.replay.rebuild_into_shadow() as (_, shadow_sessions):
        after = await snapshot(shadow_sessions)

    result = compare(before, after)
    assert result.identical, result.render()


async def test_the_four_assessment_queries_survive_a_rebuild(corpus: Corpus) -> None:
    """The operator's question, not the auditor's.

    A diff can be clean while retrieval is broken — if every vector were replaced
    by a different but internally consistent set, the rows would match on the
    columns compared and search would return nonsense. These queries are the
    independent check on that, and they are the same four the last two milestones
    were assessed on.
    """
    before = await corpus.top_results()

    await truncate_derived(corpus.sessions, clear_cache=False)
    await corpus.replay()

    after = await corpus.top_results()
    assert after == before, {
        query: (before[query], after[query])
        for query in before
        if before[query] != after[query]
    }


async def test_the_queries_survive_a_rebuild_that_recomputes_every_vector(
    corpus: Corpus,
) -> None:
    """The same, with the cache cleared, so the model actually runs again.

    With the cache kept, identical vectors are consistent with the embedder never
    having been called. This is the version that proves the model reproduces its
    own output over the same text — the assumption the cache has rested on since
    M1.5.
    """
    before = await corpus.top_results()

    report = await corpus.replay(clear_cache=True)
    assert report.vectors_computed > 0, "the cache was not actually cleared"

    assert await corpus.top_results() == before
