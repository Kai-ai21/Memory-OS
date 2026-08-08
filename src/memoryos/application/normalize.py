"""Turn a stored artifact into normalized text and chunks.

This is where the second hash level earns its keep. A file saved with different
line endings is genuinely different bytes, so it is genuinely a new artifact and
a new memory version — but its normalized text is identical, so nothing here
does any work, and neither will the embedder in M1.5.
"""

import time
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, cast
from uuid import UUID

import structlog
from sqlalchemy import CursorResult, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.adapters.parsers.registry import ParserRegistry
from memoryos.application.ports import BlobStore, Chunker, ParsedDocument, TextChunk
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobSpec, JobType, PermanentError
from memoryos.domain.normalization import normalized_hash
from memoryos.domain.values import ContentHash

logger = structlog.get_logger(__name__)


class NormalizeOutcome(StrEnum):
    SKIPPED = auto()
    REUSED = auto()
    CHUNKED = auto()
    DELETED = auto()


@dataclass(slots=True)
class NormalizeReport:
    outcome: NormalizeOutcome
    chunks: int = 0
    tokens: int = 0
    duration_ms: int = 0

    def as_dict(self) -> dict[str, str | int]:
        return {
            "outcome": self.outcome.value,
            "chunks": self.chunks,
            "tokens": self.tokens,
            "duration_ms": self.duration_ms,
        }


class NormalizeMemory:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        blob_store: BlobStore,
        parsers: ParserRegistry,
        chunker: Chunker | None = None,
    ) -> None:
        self._sessions = session_factory
        self._blobs = blob_store
        self._parsers = parsers
        self._chunker = chunker or StructuralChunker()

    @property
    def chunker_version(self) -> str:
        return self._chunker.version

    async def __call__(self, memory_id: UUID) -> NormalizeReport:
        started = time.monotonic()
        log = logger.bind(memory_id=str(memory_id))

        memory = await self._load(memory_id)
        if memory is None:
            # Retrying cannot make a memory exist.
            raise PermanentError(f"no such memory: {memory_id}")

        if memory.deleted_at is not None:
            # Tombstoned between the enqueue and the claim. Chunking it now
            # would put a deleted item back into the retrieval set.
            log.info("normalize.skipped_deleted")
            return _finish(NormalizeReport(NormalizeOutcome.DELETED), started)

        document = await self._parse(memory)
        fresh_hash = normalized_hash(document.text)

        if await self._is_current(memory, document, fresh_hash):
            # Same text, same chunker. A rerun could not produce anything
            # different, so it writes nothing at all.
            log.info("normalize.skipped_unchanged")
            return _finish(NormalizeReport(NormalizeOutcome.SKIPPED), started)

        adopted = await self._adopt_from_previous_version(memory, document, fresh_hash)
        if adopted is not None:
            log.info("normalize.reused_chunks", chunks=adopted)
            return _finish(
                NormalizeReport(NormalizeOutcome.REUSED, chunks=adopted), started
            )

        chunks = self._chunker.chunk(document)
        await self._store(memory, document, fresh_hash, chunks)

        report = NormalizeReport(
            outcome=NormalizeOutcome.CHUNKED,
            chunks=len(chunks),
            tokens=sum(chunk.token_count for chunk in chunks),
        )
        log.info("normalize.chunked", **report.as_dict())
        return _finish(report, started)

    async def _load(self, memory_id: UUID) -> models.Memory | None:
        async with self._sessions() as session:
            return await session.get(models.Memory, memory_id)

    async def _parse(self, memory: models.Memory) -> ParsedDocument:
        data = await self._blobs.get(ContentHash(memory.content_hash))
        return self._parsers.parse(
            data,
            media_type=str(memory.meta.get("media_type") or "") or None,
            external_key=memory.external_key,
        )

    async def _is_current(
        self, memory: models.Memory, document: ParsedDocument, fresh_hash: ContentHash
    ) -> bool:
        """Whether stored state already reflects this text and this chunker.

        Both halves matter. Matching text alone would leave chunks produced by
        an older chunker in place — exactly what `rechunk` exists to find.
        Matching the chunker alone would ignore an edit.
        """
        if memory.normalized_hash != fresh_hash.value:
            return False

        async with self._sessions() as session:
            versions = set(
                (
                    await session.execute(
                        select(models.MemoryChunk.chunker_version)
                        .where(models.MemoryChunk.memory_id == memory.id)
                        .distinct()
                    )
                )
                .scalars()
                .all()
            )

        if versions:
            return versions == {self._chunker.version}
        # A document with no text legitimately has no chunks, so "no chunks" is
        # current for it and stale for everything else.
        return not document.text.strip()

    async def _adopt_from_previous_version(
        self, memory: models.Memory, document: ParsedDocument, fresh_hash: ContentHash
    ) -> int | None:
        """Move an earlier version's chunks onto this one, if the text is identical.

        This is what makes a cosmetic edit free. Saving a file with CRLF line
        endings produces different bytes, so the sync correctly records a new
        artifact and a new memory version — but the normalized text is the same,
        so the chunks that already exist are the chunks this version needs.

        They are moved rather than recreated, and that matters more in M1.5 than
        it does here: the embedding travels with the row, so a line-ending
        change costs no chunking *and* no re-embedding. Recreating identical
        rows would throw away the vectors and quietly make the expensive half of
        the pipeline run again.

        Returns the number of chunks adopted, or None if there was nothing to
        adopt.
        """
        async with self._sessions.begin() as session:
            previous = (
                await session.execute(
                    select(models.Memory)
                    .where(
                        models.Memory.source_id == memory.source_id,
                        models.Memory.external_key == memory.external_key,
                        models.Memory.id != memory.id,
                        models.Memory.normalized_hash == fresh_hash.value,
                    )
                    .order_by(models.Memory.version.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if previous is None:
                return None

            versions = set(
                (
                    await session.execute(
                        select(models.MemoryChunk.chunker_version)
                        .where(models.MemoryChunk.memory_id == previous.id)
                        .distinct()
                    )
                )
                .scalars()
                .all()
            )
            if versions != {self._chunker.version}:
                # An older chunker produced them, so they are not what this
                # version needs either.
                return None

            moved = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(models.MemoryChunk)
                    .where(models.MemoryChunk.memory_id == previous.id)
                    .values(memory_id=memory.id)
                ),
            ).rowcount

            row = await session.get(models.Memory, memory.id)
            if row is None:
                raise PermanentError(f"memory vanished during normalization: {memory.id}")
            self._apply_document(row, document, fresh_hash)
            return moved

    def _apply_document(
        self, row: models.Memory, document: ParsedDocument, fresh_hash: ContentHash
    ) -> None:
        row.content = document.text
        row.normalized_hash = fresh_hash.value
        # The connector guessed the kind from a file suffix; the parser that
        # actually read the bytes knows better.
        row.kind = document.kind.value
        if document.title:
            row.title = document.title
        if document.metadata:
            row.meta = {**row.meta, "parsed": document.metadata}

    async def _store(
        self,
        memory: models.Memory,
        document: ParsedDocument,
        fresh_hash: ContentHash,
        chunks: list[TextChunk],
    ) -> None:
        """Replace the memory's text and chunks in one transaction.

        Chunks are deleted and reinserted rather than diffed. Boundaries move
        when the text changes, so there is rarely a stable correspondence to
        diff against, and a wrong correspondence would silently keep stale
        spans alive. These are explicit deletes scoped to one memory; the
        `ON DELETE CASCADE` on the table is a different guarantee for a
        different situation.
        """
        async with self._sessions.begin() as session:
            row = await session.get(models.Memory, memory.id)
            if row is None:
                raise PermanentError(f"memory vanished during normalization: {memory.id}")

            self._apply_document(row, document, fresh_hash)

            # This memory's chunks, and those of any superseded version of the
            # same item. A version nobody can retrieve leaving chunks behind is
            # the same failure as a deleted memory leaving chunks behind: they
            # stay in the vector index and keep surfacing for content that is no
            # longer current.
            superseded = select(models.Memory.id).where(
                models.Memory.source_id == memory.source_id,
                models.Memory.external_key == memory.external_key,
            )
            await session.execute(
                delete(models.MemoryChunk).where(
                    models.MemoryChunk.memory_id.in_(superseded)
                )
            )

            for chunk in chunks:
                session.add(
                    models.MemoryChunk(
                        id=new_id(),
                        memory_id=memory.id,
                        ordinal=chunk.ordinal,
                        content=chunk.text,
                        token_count=chunk.token_count,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        chunker_version=self._chunker.version,
                        content_hash=ContentHash.of(chunk.text.encode("utf-8")).value,
                        # M1.5 fills these.
                        embedding=None,
                        embedding_model=None,
                        embedded_at=None,
                    )
                )

            # Same transaction as the chunks it refers to, for the same reason
            # the sync enqueues in the transaction that creates the memory.
            await enqueue_in(
                session,
                JobSpec(
                    job_type=JobType.EMBED_MEMORY,
                    payload={"memory_id": str(memory.id)},
                    dedupe_key=f"embed:{memory.id}",
                ),
            )


def _finish(report: NormalizeReport, started: float) -> NormalizeReport:
    report.duration_ms = int((time.monotonic() - started) * 1000)
    return report
