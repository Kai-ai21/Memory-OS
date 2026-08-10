"""The guarantee, and proof that the check can fail.

`verify-citations` asserts one identity on real rows:

    memory.content[char_start:char_end] == chunk.content[prefix_chars:]

M1.4a broke exactly that and nothing noticed: row counts were right, offsets
were in bounds, every test passed, and highlights pointed a few hundred
characters away from the answer. A check that only ever passes would have been
just as useless, so this corrupts a chunk on purpose and requires the check to
catch it.
"""

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.sync import SyncSource
from memoryos.application.verify_citations import verify_citations
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import SourceKind
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

PARAGRAPH = (
    "The worker claims a task from the queue and holds a lease while the handler "
    "runs to completion. Renewing that lease is how a long task keeps its hold on "
    "the work it started, and a sweeper reclaims anything whose lease has lapsed. "
)


@pytest.fixture
async def corpus(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> async_sessionmaker[AsyncSession]:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "queue.md").write_text("# Queue\n\n" + PARAGRAPH * 8 + "\n")
    (root / "notes.md").write_text("# Notes\n\n" + PARAGRAPH * 4 + "\n")

    source = Source(
        id=new_id(), kind=SourceKind.FILESYSTEM, name="fixture", config={"root": str(root)}
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    await SyncSource(sessions, FilesystemConnector(blobs), blobs)(source.id, full=True)

    normalize = NormalizeMemory(
        sessions, blobs, build_parsers(), StructuralChunker(FakeEmbedder())
    )
    async with sessions() as session:
        targets = [
            UUID(row[0]["memory_id"])
            for row in await session.execute(
                select(models.Job.payload).where(
                    models.Job.job_type == JobType.NORMALIZE_MEMORY.value
                )
            )
        ]
    for memory_id in targets:
        await normalize(memory_id)
    async with sessions.begin() as session:
        await session.execute(delete(models.Job))

    return sessions


async def test_a_clean_corpus_verifies(
    corpus: async_sessionmaker[AsyncSession],
) -> None:
    """The identity holds for every chunk the pipeline produced.

    Non-vacuous by assertion: several chunks with a borrowed overlap head, which
    is the case the old documented meaning of the offsets got wrong.
    """
    report = await verify_citations(corpus)

    assert report.ok
    assert report.checked > 1
    assert report.mismatches == []
    assert report.unverifiable == 0

    async with corpus() as session:
        borrowed = (
            await session.execute(
                select(models.MemoryChunk.id).where(models.MemoryChunk.prefix_chars > 0)
            )
        ).all()
    assert borrowed, "the fixture must contain chunks that borrow an overlap head"


async def test_a_corrupted_offset_is_caught_and_named(
    corpus: async_sessionmaker[AsyncSession],
) -> None:
    """A verification that cannot fail proves nothing.

    The corruption is deliberately small — the offsets slide by forty characters
    while staying in bounds and keeping the right length. That is the shape of
    the real defect: nothing raises, nothing is out of range, and the citation
    quotes text from beside the answer.
    """
    # The longer file, whose first chunk has text after it to slide into. A
    # chunk that already reaches the end of its memory would trip the bounds
    # check instead, which is a different — also correct — detection.
    async with corpus() as session:
        chunk_id, start, end = (
            await session.execute(
                select(
                    models.MemoryChunk.id,
                    models.MemoryChunk.char_start,
                    models.MemoryChunk.char_end,
                )
                .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
                .where(
                    models.Memory.external_key == "queue.md",
                    models.MemoryChunk.ordinal == 0,
                )
            )
        ).one()

    async with corpus.begin() as session:
        await session.execute(
            update(models.MemoryChunk)
            .where(models.MemoryChunk.id == chunk_id)
            .values(char_start=start + 40, char_end=end + 40)
        )

    report = await verify_citations(corpus)

    assert not report.ok
    assert len(report.mismatches) == 1
    mismatch = report.mismatches[0]
    assert mismatch.ordinal == 0
    # It says *why*, because "same length, different text" and "runs past the
    # end" have different causes and lead somewhere different.
    assert "offsets point elsewhere" in mismatch.reason
    assert mismatch.expected != mismatch.actual

    # And the scoped form catches it too, so a golden-set run is not a weaker
    # check than a full sweep on the memories it does cover.
    scoped = await verify_citations(corpus, memory_ids=[mismatch.memory_id])
    assert not scoped.ok
