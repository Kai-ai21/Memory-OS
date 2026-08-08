"""The acceptance criterion for M1.6.1.

M1.6's report found two of four assessment queries returning irrelevant
results. The diagnosis was that the answers lived in chunks whose tails never
reached the model. This ingests the real repository and asserts those two
queries now find what they are asking for.

This is the test that decides whether the hotfix worked. Everything else in
this branch is a mechanism; this is the outcome.
"""

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.embedding.sentence_transformers import SentenceTransformerEmbedder
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.embed import EmbedMemory
from memoryos.application.evaluation import measure_recall
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.search import MemoryHit, SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import SourceKind

pytestmark = [pytest.mark.slow, pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@pytest.fixture
async def searcher(
    tmp_path: Path,
    sessions: async_sessionmaker[AsyncSession],
    embedder: SentenceTransformerEmbedder,
) -> SearchMemories:
    """Ingest this repository — the corpus the M1.6 report was written against."""
    source = Source(
        id=new_id(),
        kind=SourceKind.FILESYSTEM,
        name="self",
        config={"root": str(REPO_ROOT)},
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    await SyncSource(sessions, FilesystemConnector(blobs), blobs)(source.id, full=True)

    chunker = StructuralChunker(embedder)
    normalize = NormalizeMemory(sessions, blobs, build_parsers(), chunker)
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

    return SearchMemories(sessions, embedder, PgVectorStore(sessions, embedder))


def matched_text(hits: list[MemoryHit], limit: int = 3) -> str:
    """All chunk text from the top hits, lowercased."""
    return " ".join(
        chunk.text.lower() for hit in hits[:limit] for chunk in hit.matched_chunks
    )


async def test_query_two_finds_the_two_timestamps_rationale(
    searcher: SearchMemories,
) -> None:
    """M1.6 returned a generic docstring, a different timestamp pair, and hash validation.

    The answer — occurred_at versus ingested_at — is stated plainly in several
    places in this repository.
    """
    result = await searcher("why do we store two timestamps", k=5)

    assert result.hits
    text = matched_text(result.hits)
    assert "occurred_at" in text and "ingested_at" in text, [
        (hit.external_key, round(hit.score, 3)) for hit in result.hits
    ]


async def test_query_three_finds_content_addressing(
    searcher: SearchMemories,
) -> None:
    """M1.6 returned a streaming-iterator docstring and the HNSW migration."""
    result = await searcher("content addressing and deduplication", k=5)

    assert result.hits
    text = matched_text(result.hits)
    assert "raw_artifacts" in text or "content hash" in text or "content_hash" in text, [
        (hit.external_key, round(hit.score, 3)) for hit in result.hits
    ]


async def test_the_two_queries_that_already_worked_still_work(
    searcher: SearchMemories,
) -> None:
    # A fix that improves two queries by breaking two others is not a fix.
    claim = await searcher("how does the job queue claim work", k=5)
    assert "claim" in matched_text(claim.hits)

    deletion = await searcher("what happens when a file is deleted", k=5)
    assert "delet" in matched_text(deletion.hits)


async def test_self_retrieval_is_near_perfect(
    searcher: SearchMemories,
    sessions: async_sessionmaker[AsyncSession],
    embedder: SentenceTransformerEmbedder,
) -> None:
    """It was 0.98, and every miss was a tie at exactly 1.0.

    Those ties were distinct chunks sharing a prefix long enough that the model
    saw only the shared part. With the window aligned, they are distinguishable.
    """
    rows = await measure_recall(
        sessions,
        embedder,
        PgVectorStore(sessions, embedder),
        queries=200,
        k=10,
        ef_search_values=[100],
    )

    (row,) = rows
    assert row.self_retrieval >= 0.99, row


async def test_no_chunk_exceeds_the_model_window(
    searcher: SearchMemories,
    sessions: async_sessionmaker[AsyncSession],
    embedder: SentenceTransformerEmbedder,
) -> None:
    # The invariant, checked against the real corpus and the real tokenizer.
    async with sessions() as session:
        contents = list(
            (await session.execute(select(models.MemoryChunk.content))).scalars()
        )

    oversized = [c for c in contents if embedder.count_tokens(c) > embedder.max_sequence_tokens]
    assert oversized == [], f"{len(oversized)} chunks exceed the window"
