"""The lexical retriever, against a real GIN index.

Rows are built directly rather than through the ingest pipeline. What is under
test is a column Postgres maintains and a query that reads it, and routing that
through sync/normalize/embed would put a chunker between the assertion and the
thing it asserts — the four tests below would then also fail whenever chunk
boundaries moved, which is a different milestone's problem.

The corpus is written so the vector half would plausibly get each of these
wrong: an opaque SQL fragment, a query that is nothing but stop words, a
tombstone, and text that changes underneath a stored index.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.mappers import to_memory_row
from memoryos.application.ports import SearchFilters
from memoryos.domain.entities import RawArtifact, Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash
from tests.integration.conftest import build_memory

pytestmark = pytest.mark.integration

DELETED_AT = datetime(2024, 6, 1, tzinfo=UTC)

# The clause itself, in one place in the corpus, surrounded by text that shares
# none of its terms. This is the query M2.0 measured at MRR 0.000 for the vector
# half.
CLAIM = (
    "The claim query takes the oldest pending job and marks it running. "
    "FOR UPDATE SKIP LOCKED on the inner select is the clause that makes two "
    "workers claim different rows instead of queueing behind each other."
)
BREAD = (
    "A wild yeast starter is fed flour and water until it doubles reliably, "
    "then folded gently and given a long cold rest in the refrigerator."
)
LEASE = (
    "Renewing the lease is how a long running handler keeps its hold on the "
    "work it started, so a sweeper does not reclaim a job that is progressing."
)


async def seed(
    session: AsyncSession,
    source: Source,
    artifact: RawArtifact,
    *,
    external_key: str,
    content: str,
    deleted: bool = False,
) -> models.MemoryChunk:
    """One memory with one chunk. Returns the chunk row."""
    memory = build_memory(source, artifact, external_key=external_key)
    row = to_memory_row(memory)
    if deleted:
        # A tombstone, which is what a deleted file leaves behind: the row stays
        # so history and replay still work, and retrieval must not see it.
        row.deleted_at = DELETED_AT
    session.add(row)
    await session.flush()

    chunk = models.MemoryChunk(
        id=new_id(),
        memory_id=memory.id,
        ordinal=0,
        content=content,
        token_count=len(content.split()),
        char_start=0,
        char_end=len(content),
        chunker_version="test-v1",
        content_hash=ContentHash.of(content.encode()).value,
    )
    session.add(chunk)
    await session.flush()
    await session.commit()
    return chunk


@pytest.fixture
async def store(sessions: async_sessionmaker[AsyncSession]) -> PostgresKeywordStore:
    return PostgresKeywordStore(sessions)


async def test_a_rare_exact_token_ranks_first(
    session: AsyncSession,
    source: Source,
    artifact: RawArtifact,
    store: PostgresKeywordStore,
) -> None:
    """The whole argument for this milestone, as one assertion.

    `SKIP LOCKED` embeds to almost nothing — M2.0 measured the vector half at
    MRR 0.000 for this exact query on the real corpus. Lexically it is trivial:
    the terms are rare and they occur in one chunk.
    """
    claim = await seed(
        session, source, artifact, external_key="job_queue.py", content=CLAIM
    )
    await seed(session, source, artifact, external_key="bread.md", content=BREAD)
    await seed(session, source, artifact, external_key="lease.py", content=LEASE)

    found = await store.search("SKIP LOCKED", k=5, filters=SearchFilters())

    assert found, "the clause is in the corpus and the index should find it"
    assert found[0].chunk_id == claim.id
    assert found[0].score > 0
    # And it is not a case of everything matching: the other two chunks share no
    # lexeme with the query, so a hit on either would mean the `@@` filter is
    # not doing anything.
    assert [chunk.chunk_id for chunk in found] == [claim.id]


async def test_a_query_of_only_stop_words_returns_nothing_rather_than_erroring(
    session: AsyncSession,
    source: Source,
    artifact: RawArtifact,
    store: PostgresKeywordStore,
) -> None:
    """`to_tsquery` would raise here. That is why the adapter does not use it.

    Postgres reduces "the and of" to an empty tsquery and emits a NOTICE; an
    empty tsquery matches no row. A user typing three common words gets an empty
    result, which is an answer, not a failure.
    """
    await seed(session, source, artifact, external_key="job_queue.py", content=CLAIM)

    for query in ("the and of", "   ", "", "a"):
        assert await store.search(query, k=5, filters=SearchFilters()) == []

    # The store is not simply broken: the same corpus and the same call answer a
    # real query.
    assert await store.search("claim", k=5, filters=SearchFilters())


async def test_a_tombstoned_memorys_chunks_never_surface(
    session: AsyncSession,
    source: Source,
    artifact: RawArtifact,
    store: PostgresKeywordStore,
) -> None:
    """The same eligibility rule the vector store enforces, and for the reason.

    A deleted item's chunks staying searchable is a correctness failure rather
    than a ranking one — the file is gone and the system would still quote it.
    """
    await seed(
        session,
        source,
        artifact,
        external_key="deleted.py",
        content=CLAIM,
        deleted=True,
    )

    assert await store.search("SKIP LOCKED", k=5, filters=SearchFilters()) == []

    # Reachable only by asking for it, which is what the flag is for.
    including = await store.search(
        "SKIP LOCKED", k=5, filters=SearchFilters(include_deleted=True)
    )
    assert len(including) == 1


async def test_the_generated_column_follows_content_without_being_told(
    session: AsyncSession,
    source: Source,
    artifact: RawArtifact,
    store: PostgresKeywordStore,
) -> None:
    """Why it is a generated column and not a trigger or a pipeline step.

    Nothing in this test updates `search_vector`. It is not writable, no
    application code knows it exists, and a rechunk that rewrites `content`
    therefore cannot leave the index describing text that is no longer there.
    """
    chunk = await seed(
        session, source, artifact, external_key="job_queue.py", content=CLAIM
    )
    assert await store.search("SKIP LOCKED", k=5, filters=SearchFilters())

    async with session.begin():
        await session.execute(
            update(models.MemoryChunk)
            .where(models.MemoryChunk.id == chunk.id)
            .values(content=BREAD)
        )

    assert await store.search("SKIP LOCKED", k=5, filters=SearchFilters()) == []
    refreshed = await store.search("yeast starter", k=5, filters=SearchFilters())
    assert [found.chunk_id for found in refreshed] == [chunk.id]
