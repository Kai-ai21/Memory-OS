"""Normalization and chunking against a real database.

The two properties this milestone exists to establish are asserted here: a
line-ending change writes no chunks, and re-normalizing an unchanged memory
writes nothing at all.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from pypdf import PdfWriter
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import ChunkerConfig, StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.normalize import (
    NormalizeMemory,
    NormalizeOutcome,
    NormalizeReport,
)
from memoryos.application.rechunk import enqueue_rechunk, find_stale
from memoryos.application.sync import SyncReport, SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType, PermanentError
from memoryos.domain.values import SourceKind
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog and keeps running onward. "
    "Every good boy deserves fudge and a reasonable amount of rest. "
)


@dataclass(slots=True)
class Pipeline:
    root: Path
    source: Source
    sync: SyncSource
    normalize: NormalizeMemory
    sessions: async_sessionmaker[AsyncSession]

    async def run_sync(self, *, full: bool = True) -> SyncReport:
        return await self.sync(self.source.id, full=full)

    async def normalize_all(self) -> list[NormalizeReport]:
        """Drain the normalize queue, the way a worker would.

        Jobs are consumed, not merely read: leaving them behind would make the
        next drain re-run work a real worker had already completed.
        """
        pending = await self.pending_normalize_ids()
        reports = [await self.normalize(memory_id) for memory_id in pending]
        async with self.sessions.begin() as session:
            await session.execute(
                delete(models.Job).where(
                    models.Job.job_type == JobType.NORMALIZE_MEMORY.value
                )
            )
        return reports

    async def pending_normalize_ids(self) -> list[UUID]:
        async with self.sessions() as session:
            rows = await session.execute(
                select(models.Job.payload).where(
                    models.Job.job_type == JobType.NORMALIZE_MEMORY.value
                )
            )
            return [UUID(row[0]["memory_id"]) for row in rows]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "guide.md").write_text(
        "---\ntitle: The Guide\n---\n\n# The Guide\n\n"
        + PARAGRAPH * 8
        + "\n\n## Details\n\n"
        + PARAGRAPH * 8
        + "\n"
    )
    (root / "notes.txt").write_text(PARAGRAPH * 4 + "\n")
    (root / "mod.py").write_text(
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    )
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
        sessions=sessions,
    )


async def chunk_rows(
    sessions: async_sessionmaker[AsyncSession],
    external_key: str | None = None,
    *,
    current_only: bool = True,
) -> list[models.MemoryChunk]:
    stmt = (
        select(models.MemoryChunk)
        .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
        .order_by(models.Memory.external_key, models.MemoryChunk.ordinal)
    )
    if current_only:
        stmt = stmt.where(models.Memory.is_current.is_(True))
    if external_key is not None:
        stmt = stmt.where(models.Memory.external_key == external_key)
    async with sessions() as session:
        return list((await session.execute(stmt)).scalars())


async def memory_for(
    sessions: async_sessionmaker[AsyncSession], external_key: str
) -> models.Memory:
    async with sessions() as session:
        row = (
            await session.execute(
                select(models.Memory).where(
                    models.Memory.external_key == external_key,
                    models.Memory.is_current.is_(True),
                )
            )
        ).scalar_one()
        return row


async def count_of(sessions: async_sessionmaker[AsyncSession], model: type) -> int:
    async with sessions() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


# --------------------------------------------------------------------------
# The pipeline end to end
# --------------------------------------------------------------------------


async def test_syncing_then_normalizing_produces_chunks(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    reports = await pipeline.normalize_all()

    assert [report.outcome for report in reports] == [NormalizeOutcome.CHUNKED] * 3

    chunks = await chunk_rows(pipeline.sessions)
    assert chunks
    version = StructuralChunker(FakeEmbedder()).version
    for chunk in chunks:
        assert chunk.chunker_version == version
        # M1.5 fills these; M1.4 must leave them empty.
        assert chunk.embedding is None
        assert chunk.embedding_model is None
        assert chunk.embedded_at is None
        assert chunk.token_count > 0
        assert chunk.char_end > chunk.char_start


async def test_normalizing_sets_content_hash_title_and_kind(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    await pipeline.normalize_all()

    guide = await memory_for(pipeline.sessions, "guide.md")
    assert guide.content is not None
    assert guide.content.startswith("# The Guide")
    assert guide.normalized_hash is not None
    assert guide.title == "The Guide"
    assert guide.kind == "note"

    module = await memory_for(pipeline.sessions, "mod.py")
    # The connector guessed from the suffix; the parser confirms from the bytes.
    assert module.kind == "code"


async def test_chunk_offsets_index_into_the_stored_content(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    await pipeline.normalize_all()

    guide = await memory_for(pipeline.sessions, "guide.md")
    assert guide.content is not None

    for chunk in await chunk_rows(pipeline.sessions, "guide.md"):
        # What a Phase 2 citation will highlight.
        assert chunk.content.endswith(guide.content[chunk.char_start : chunk.char_end])


async def test_normalizing_enqueues_an_embed_job(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    await pipeline.normalize_all()

    async with pipeline.sessions() as session:
        jobs = list(
            (
                await session.execute(
                    select(models.Job).where(
                        models.Job.job_type == JobType.EMBED_MEMORY.value
                    )
                )
            ).scalars()
        )

    assert len(jobs) == 3
    assert all(job.dedupe_key == f"embed:{job.payload['memory_id']}" for job in jobs)


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


async def test_re_normalizing_an_unchanged_memory_does_nothing(
    pipeline: Pipeline,
) -> None:
    """The second of the milestone's two idempotency properties."""
    await pipeline.run_sync()
    await pipeline.normalize_all()

    before = {chunk.id for chunk in await chunk_rows(pipeline.sessions)}
    guide = await memory_for(pipeline.sessions, "guide.md")

    report = await pipeline.normalize(guide.id)

    assert report.outcome is NormalizeOutcome.SKIPPED
    assert report.chunks == 0
    # Not merely the same count — the same rows. A delete-and-reinsert would
    # produce identical counts with entirely new ids.
    assert {chunk.id for chunk in await chunk_rows(pipeline.sessions)} == before


async def test_changing_only_line_endings_writes_no_chunks(pipeline: Pipeline) -> None:
    """The first idempotency property, and the point of the second hash level.

    Different bytes, so the sync correctly records a new artifact and a new
    memory version — the system does not pretend the file is unchanged. But the
    normalized text is identical, so no chunk is parsed, split, or created. The
    existing chunk rows move to the new version, ids and all, which is what will
    also spare the embeddings in M1.5.
    """
    await pipeline.run_sync()
    await pipeline.normalize_all()

    chunks_before = {chunk.id for chunk in await chunk_rows(pipeline.sessions)}
    artifacts_before = await count_of(pipeline.sessions, models.RawArtifact)
    hash_before = (await memory_for(pipeline.sessions, "guide.md")).normalized_hash

    source_file = pipeline.root / "guide.md"
    source_file.write_bytes(source_file.read_text().replace("\n", "\r\n").encode())

    sync_report = await pipeline.run_sync()
    assert sync_report.modified == 1
    # A genuinely new artifact: the bytes really are different.
    assert await count_of(pipeline.sessions, models.RawArtifact) == artifacts_before + 1

    guide = await memory_for(pipeline.sessions, "guide.md")
    assert guide.version == 2

    reports = await pipeline.normalize_all()

    assert [report.outcome for report in reports] == [NormalizeOutcome.REUSED]
    # Zero chunk writes: the very same rows, now attached to version 2.
    assert {chunk.id for chunk in await chunk_rows(pipeline.sessions)} == chunks_before
    assert (await memory_for(pipeline.sessions, "guide.md")).normalized_hash == hash_before


async def test_a_reused_version_does_not_re_enqueue_embedding(
    pipeline: Pipeline,
) -> None:
    # The saving that matters most in M1.5: the vectors travel with the rows, so
    # nothing needs re-embedding either.
    await pipeline.run_sync()
    await pipeline.normalize_all()

    async with pipeline.sessions.begin() as session:
        await session.execute(delete(models.Job))

    source_file = pipeline.root / "guide.md"
    source_file.write_bytes(source_file.read_text().replace("\n", "\r\n").encode())
    await pipeline.run_sync()
    await pipeline.normalize_all()

    async with pipeline.sessions() as session:
        embed_jobs = (
            await session.execute(
                select(func.count())
                .select_from(models.Job)
                .where(models.Job.job_type == JobType.EMBED_MEMORY.value)
            )
        ).scalar_one()
    assert embed_jobs == 0


async def test_the_normalized_hash_is_identical_across_line_endings(
    pipeline: Pipeline,
) -> None:
    # Stated directly at the database level, without the version machinery in
    # the way.
    await pipeline.run_sync()
    await pipeline.normalize_all()
    before = (await memory_for(pipeline.sessions, "notes.txt")).normalized_hash

    notes = pipeline.root / "notes.txt"
    notes.write_bytes(notes.read_text().replace("\n", "\r\n").encode())
    await pipeline.run_sync()
    await pipeline.normalize_all()

    assert (await memory_for(pipeline.sessions, "notes.txt")).normalized_hash == before


async def test_a_real_edit_replaces_the_chunks(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    await pipeline.normalize_all()

    guide = await memory_for(pipeline.sessions, "guide.md")
    before = {chunk.id for chunk in await chunk_rows(pipeline.sessions, "guide.md")}
    hash_before = guide.normalized_hash

    (pipeline.root / "guide.md").write_text(
        "# The Guide\n\nCompletely different content now.\n\n" + PARAGRAPH * 10
    )
    await pipeline.run_sync()
    await pipeline.normalize_all()

    after_memory = await memory_for(pipeline.sessions, "guide.md")
    assert after_memory.normalized_hash != hash_before

    after = {chunk.id for chunk in await chunk_rows(pipeline.sessions, "guide.md")}
    # Boundaries move when text changes, so chunks are replaced wholesale
    # rather than diffed. None of the old ids survive on the new version.
    assert after.isdisjoint(before)


# --------------------------------------------------------------------------
# Re-chunking
# --------------------------------------------------------------------------


async def test_rechunk_finds_only_stale_memories(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    await pipeline.normalize_all()

    current = StructuralChunker(FakeEmbedder()).version
    assert await find_stale(pipeline.sessions, current_version=current) == []

    # A different chunker: every existing chunk is now stale.
    improved = StructuralChunker(FakeEmbedder(), ChunkerConfig(target=320)).version
    stale = await find_stale(pipeline.sessions, current_version=improved)

    assert sorted(memory.external_key for memory in stale) == [
        "guide.md",
        "mod.py",
        "notes.txt",
    ]


async def test_rechunk_enqueues_normalize_jobs(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    await pipeline.normalize_all()

    async with pipeline.sessions.begin() as session:
        await session.execute(delete(models.Job))

    improved = StructuralChunker(FakeEmbedder(), ChunkerConfig(target=320)).version
    stale = await find_stale(pipeline.sessions, current_version=improved)
    enqueued = await enqueue_rechunk(pipeline.sessions, stale)

    assert enqueued == 3
    assert await count_of(pipeline.sessions, models.Job) == 3


async def test_rechunk_can_be_scoped_to_one_source(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    await pipeline.normalize_all()

    improved = StructuralChunker(FakeEmbedder(), ChunkerConfig(target=320)).version

    assert (
        len(await find_stale(pipeline.sessions, current_version=improved, source="corpus"))
        == 3
    )
    assert (
        await find_stale(pipeline.sessions, current_version=improved, source="other") == []
    )


async def test_a_new_chunker_version_actually_re_chunks(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    await pipeline.normalize_all()

    guide = await memory_for(pipeline.sessions, "guide.md")
    before = {chunk.id for chunk in await chunk_rows(pipeline.sessions, "guide.md")}

    smaller = NormalizeMemory(
        pipeline.sessions,
        FilesystemBlobStore(pipeline.root.parent / "blobs"),
        build_parsers(),
        StructuralChunker(FakeEmbedder(), ChunkerConfig(target=200, minimum=40)),
    )
    report = await smaller(guide.id)

    # Same text, different chunker: not skipped.
    assert report.outcome is NormalizeOutcome.CHUNKED
    after = await chunk_rows(pipeline.sessions, "guide.md")
    assert {chunk.id for chunk in after}.isdisjoint(before)
    assert {chunk.chunker_version for chunk in after} == {
        StructuralChunker(FakeEmbedder(), ChunkerConfig(target=200, minimum=40)).version
    }


async def test_rechunk_populates_prefix_chars_on_existing_memories(
    pipeline: Pipeline,
) -> None:
    """The repair path for corpora written before M1.4a.

    Migration 0008 gives every existing row `prefix_chars = 0`, which is wrong
    for every chunk after the first — the borrowed head is not recoverable from
    a migration, and 28% of this repository's stored chunk text was borrowed. So
    the column ships provisional, the chunker version bumps in the same change
    to make every row stale, and the rechunk is what writes the real value.
    """
    await pipeline.run_sync()
    await pipeline.normalize_all()

    # The state migration 0008 leaves behind: the column is there, every row
    # claims to have borrowed nothing, and the stamp is the old chunker's.
    async with pipeline.sessions.begin() as session:
        await session.execute(
            update(models.MemoryChunk).values(
                prefix_chars=0, chunker_version="structural-v2:superseded"
            )
        )
        await session.execute(delete(models.Job))

    current = StructuralChunker(FakeEmbedder()).version
    stale = await find_stale(pipeline.sessions, current_version=current)
    assert await enqueue_rechunk(pipeline.sessions, stale) == 3

    await pipeline.normalize_all()

    rows = await chunk_rows(pipeline.sessions)
    documents = {
        memory.id: memory.content
        for memory in [
            await memory_for(pipeline.sessions, key)
            for key in ("guide.md", "notes.txt", "mod.py")
        ]
    }
    assert any(row.prefix_chars > 0 for row in rows), "nothing borrowed after a rechunk"
    for row in rows:
        document = documents[row.memory_id] or ""
        assert row.content[row.prefix_chars :] == document[row.char_start : row.char_end]


# --------------------------------------------------------------------------
# Difficult inputs
# --------------------------------------------------------------------------


async def test_a_scanned_pdf_fails_permanently(pipeline: Pipeline) -> None:
    """No text layer means no amount of retrying will produce one.

    Storing the empty result would create a memory that retrieves for nothing
    and looks, in the table, exactly like a document that legitimately had no
    content.
    """
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    (pipeline.root / "scanned.pdf").write_bytes(buffer.getvalue())

    await pipeline.run_sync()
    scanned = await memory_for(pipeline.sessions, "scanned.pdf")

    with pytest.raises(PermanentError, match="scanned"):
        await pipeline.normalize(scanned.id)

    assert await chunk_rows(pipeline.sessions, "scanned.pdf") == []


async def test_invalid_utf8_still_parses_and_records_the_codec(
    pipeline: Pipeline,
) -> None:
    (pipeline.root / "latin.txt").write_bytes(
        "caf\xe9 latte. ".encode("latin-1") * 40
    )

    await pipeline.run_sync()
    await pipeline.normalize_all()

    memory = await memory_for(pipeline.sessions, "latin.txt")
    assert memory.content is not None
    assert "caf" in memory.content
    assert memory.meta["parsed"]["codec"] == "latin-1"
    assert await chunk_rows(pipeline.sessions, "latin.txt")


async def test_a_tombstoned_memory_is_not_chunked(pipeline: Pipeline) -> None:
    await pipeline.run_sync()
    guide = await memory_for(pipeline.sessions, "guide.md")

    (pipeline.root / "guide.md").unlink()
    await pipeline.run_sync()

    report = await pipeline.normalize(guide.id)

    assert report.outcome is NormalizeOutcome.DELETED
    assert await chunk_rows(pipeline.sessions, "guide.md") == []


async def test_normalizing_a_missing_memory_fails_permanently(
    pipeline: Pipeline,
) -> None:
    with pytest.raises(PermanentError, match="no such memory"):
        await pipeline.normalize(new_id())


async def test_the_enclosing_definition_reaches_the_stored_row(
    pipeline: Pipeline,
) -> None:
    """The chunker has always known this; until now it was discarded.

    It is computed here, during normalization, and needed at query time, which
    is a different process minutes or months later. A value that exists only in
    the memory of the step that derived it is not available to the step that
    needs it.
    """
    body = "\n".join(
        f"    total = total + compute_partial_result_{n}(total, {n})" for n in range(80)
    )
    (pipeline.root / "big.py").write_text(f"def enormous(total):\n{body}\n    return total\n")

    await pipeline.run_sync()
    await pipeline.normalize_all()

    chunks = await chunk_rows(pipeline.sessions, "big.py")
    assert chunks
    assert all(chunk.meta.get("definition") == "enormous" for chunk in chunks), [
        chunk.meta for chunk in chunks
    ]


async def test_a_span_inside_no_definition_stores_an_empty_object(
    pipeline: Pipeline,
) -> None:
    # Not null. "The chunker recorded nothing" and "nobody looked" are the same
    # state, and an empty object spares every reader a null check.
    await pipeline.run_sync()
    await pipeline.normalize_all()

    for chunk in await chunk_rows(pipeline.sessions, "notes.txt"):
        assert chunk.meta == {}
