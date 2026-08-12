"""Extract entities for a memory, store them, and queue the graph to catch up.

**This use case does not write to Neo4j.** It commits to Postgres and enqueues a
`SYNC_GRAPH` job naming the memory that changed. M3.1 projected inline, and that
was a use case writing to two stores — which makes the graph a second source of
truth in everything but name, however carefully the write is ordered.

The ordering that mattered survives: Postgres commits, and only then is anything
queued. A crash between them leaves a corpus that is correct and a graph that is
behind, which the next sync repairs because every graph write is a `MERGE`. The
other order leaves a graph asserting entities no query can join to.

What the enqueue buys beyond the principle is failure handling. An inline
projection had to catch its own errors and report a degraded graph, because the
rows were already committed and taking the job down would have turned a
projection outage into a failed extraction. A queued sync has no such conflict: it
raises, the worker retries it with backoff, and a Neo4j that was down for a minute
is a pause rather than a divergence nobody records.

Idempotency follows M1.4 and M1.5 exactly: a memory whose mentions already carry
the current `extractor_version` is skipped without a model call. That is what
makes re-running the command free, makes a prompt change a targeted redo rather
than a corpus rebuild, and makes the job safe to retry after a failure partway
through.
"""

import time
from dataclasses import dataclass
from enum import StrEnum, auto
from uuid import UUID

import structlog
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.graph_sync import enqueue_sync
from memoryos.application.ports import (
    EntityExtractor,
    ExtractedEntity,
    JobQueue,
)
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import PermanentError
from memoryos.domain.normalization import canonical_entity_name
from memoryos.domain.values import MemoryKind

logger = structlog.get_logger(__name__)


class ExtractOutcome(StrEnum):
    SKIPPED = auto()
    EXTRACTED = auto()
    DELETED = auto()
    EMPTY = auto()


@dataclass(slots=True)
class ExtractReport:
    outcome: ExtractOutcome
    chunks: int = 0
    entities: int = 0
    mentions: int = 0
    # Candidates refused at the storage boundary because their offsets did not
    # hold. Should always be zero with the real extractor, which verifies before
    # returning; a non-zero count means an extractor broke the port's contract,
    # which is worth seeing rather than silently correcting.
    unverified: int = 0
    # Whether a projection update was queued. Not whether the graph is current:
    # that is the sync job's business, and conflating the two is what made the
    # inline projection's `projected=True` mean "we tried".
    sync_enqueued: bool = False
    duration_ms: int = 0

    def as_dict(self) -> dict[str, str | int | bool]:
        return {
            "outcome": self.outcome.value,
            "chunks": self.chunks,
            "entities": self.entities,
            "mentions": self.mentions,
            "unverified": self.unverified,
            "sync_enqueued": self.sync_enqueued,
            "duration_ms": self.duration_ms,
        }


class ExtractEntities:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        extractor: EntityExtractor,
        queue: JobQueue | None = None,
    ) -> None:
        self._sessions = session_factory
        self._extractor = extractor
        # Optional, because the graph is a projection and Phase 1 and 2 work
        # without it. No queue means the rows are written and nothing is
        # projected, which `graph verify` reports as a divergence and
        # `graph rebuild` fixes.
        self._queue = queue

    async def __call__(self, memory_id: UUID) -> ExtractReport:
        started = time.monotonic()
        version = self._extractor.version
        log = logger.bind(memory_id=str(memory_id), extractor=version)

        memory = await self._load(memory_id)
        if memory is None:
            raise PermanentError(f"no such memory: {memory_id}")
        if memory.deleted_at is not None:
            log.info("extract.skipped_deleted")
            return _finish(ExtractReport(ExtractOutcome.DELETED), started)

        if await self._already_current(memory_id, version):
            # The M1.4/M1.5 skip. No model call, no cost, no rewrite.
            log.info("extract.skipped_current")
            return _finish(ExtractReport(ExtractOutcome.SKIPPED), started)

        chunks = await self._chunks(memory_id)
        if not chunks:
            log.info("extract.no_chunks")
            return _finish(ExtractReport(ExtractOutcome.EMPTY), started)

        kind = MemoryKind(memory.kind)
        found = await self._extract(chunks, kind)

        report = await self._store(memory_id, chunks, found, version)
        # After the commit, never inside it. See the module docstring.
        report.sync_enqueued = await self._queue_sync(memory_id)
        # Logged after the enqueue rather than before, so the field in the log is
        # the outcome rather than its default. Logging first reported
        # `projected=False` on every successful run.
        log.info("extract.stored", **report.as_dict())
        return _finish(report, started)

    async def _extract(
        self, chunks: list[tuple[UUID, str]], kind: MemoryKind
    ) -> list[list[ExtractedEntity]]:
        """One pass over the chunks, batched if the extractor supports it.

        `extract_batch` is not part of the port — the port's surface is
        `extract`, one text at a time. It is used when present because a request
        per chunk exhausts the free tier's daily cap on a corpus of any size,
        and falling back to the port keeps a simpler extractor (the test fake,
        or a future local model with no rate limit) perfectly usable.
        """
        texts = [text for _, text in chunks]
        batch = getattr(self._extractor, "extract_batch", None)
        if callable(batch):
            results: list[list[ExtractedEntity]] = await batch(texts, kind=kind)
            return results
        return [await self._extractor.extract(text, kind=kind) for text in texts]

    async def _load(self, memory_id: UUID) -> models.Memory | None:
        async with self._sessions() as session:
            return await session.get(models.Memory, memory_id)

    async def _already_current(self, memory_id: UUID, version: str) -> bool:
        """Whether this memory has already been extracted at this version.

        Reads the marker on `memories` rather than probing for mentions, and the
        difference is the whole point: "has mentions" and "has been extracted"
        are different questions, and they disagree for every memory that
        legitimately contains no entities. Probing for mentions never marks
        those done, so each run paid to extract them again and the pending
        count never reached zero.
        """
        stmt = select(models.Memory.entity_extractor_version).where(
            models.Memory.id == memory_id
        )
        async with self._sessions() as session:
            return (await session.execute(stmt)).scalar_one_or_none() == version

    async def _chunks(self, memory_id: UUID) -> list[tuple[UUID, str]]:
        stmt = (
            select(models.MemoryChunk.id, models.MemoryChunk.content)
            .where(models.MemoryChunk.memory_id == memory_id)
            .order_by(models.MemoryChunk.ordinal)
        )
        async with self._sessions() as session:
            return [(row[0], row[1]) for row in await session.execute(stmt)]

    async def _store(
        self,
        memory_id: UUID,
        chunks: list[tuple[UUID, str]],
        found: list[list[ExtractedEntity]],
        version: str,
    ) -> ExtractReport:
        """Upsert the entities, replace this memory's mentions, in one transaction.

        Mentions from *other* extractor versions are deleted here, and that is
        the point of doing it in the same transaction as the insert: a memory
        must never carry two versions' opinions at once, or every count over the
        table double-counts and no query can tell which mention is current.
        """
        entities = 0
        mentions = 0
        unverified = 0

        async with self._sessions.begin() as session:
            await session.execute(
                delete(models.EntityMention).where(
                    models.EntityMention.memory_id == memory_id
                )
            )
            # In the same transaction as the mentions, so a memory is never
            # marked extracted without the rows that extraction produced — and
            # never has the rows without the mark.
            await session.execute(
                update(models.Memory)
                .where(models.Memory.id == memory_id)
                .values(entity_extractor_version=version)
            )

            for (chunk_id, chunk_text), candidates in zip(chunks, found, strict=True):
                for candidate in candidates:
                    if not _offsets_hold(chunk_text, candidate):
                        # The port says an extractor has already verified this,
                        # and it is checked again anyway — the same reasoning as
                        # `EmbedMemory._check_widths`, which refuses a
                        # wrong-width vector the embedder should never have
                        # produced. The invariant this table exists to carry is
                        # `content[char_start:char_end] == name`, and a
                        # guarantee that depends on every present and future
                        # extractor keeping its word is not a guarantee. One
                        # string slice makes it unconditional.
                        unverified += 1
                        logger.warning(
                            "extract.offsets_rejected",
                            name=candidate.name,
                            chunk_id=str(chunk_id),
                        )
                        continue
                    entity_id, created = await self._upsert_entity(session, candidate)
                    entities += int(created)
                    await session.execute(
                        insert(models.EntityMention)
                        .values(
                            id=new_id(),
                            entity_id=entity_id,
                            memory_id=memory_id,
                            chunk_id=chunk_id,
                            char_start=candidate.char_start,
                            char_end=candidate.char_end,
                            confidence=candidate.confidence,
                            extractor_version=version,
                        )
                        # The same entity at the same offset in the same chunk is
                        # one mention. Reachable when a batch returns a duplicate
                        # the in-adapter dedup did not catch across retries.
                        .on_conflict_do_nothing(
                            constraint="uq_entity_mentions_span"
                        )
                    )
                    mentions += 1

        return ExtractReport(
            outcome=ExtractOutcome.EXTRACTED,
            chunks=len(chunks),
            entities=entities,
            mentions=mentions,
            unverified=unverified,
        )

    async def _upsert_entity(
        self, session: AsyncSession, candidate: ExtractedEntity
    ) -> tuple[UUID, bool]:
        """The entity id for this name and type, creating the row if new.

        `ON CONFLICT DO NOTHING` then `SELECT`, rather than `DO UPDATE`, because
        an entity's stored `name` is the surface form as *first* seen and later
        sightings must not overwrite it — that first form is the evidence M3.2
        will resolve against. The insert returns nothing on conflict, so the
        select is what resolves an existing row.
        """
        canonical = canonical_entity_name(candidate.name)
        entity_type = candidate.type.value

        inserted = await session.execute(
            insert(models.Entity)
            .values(
                id=new_id(),
                name=candidate.name,
                canonical_name=canonical,
                type=entity_type,
                confidence=candidate.confidence,
            )
            .on_conflict_do_nothing(constraint="uq_entities_canonical_type")
            .returning(models.Entity.id)
        )
        entity_id = inserted.scalar_one_or_none()
        if entity_id is not None:
            return entity_id, True

        existing = await session.execute(
            select(models.Entity.id, models.Entity.merged_into_id).where(
                models.Entity.canonical_name == canonical,
                models.Entity.type == entity_type,
            )
        )
        entity_id, merged_into = existing.one()
        if merged_into is None:
            return entity_id, False

        # M3.2: this name was merged away, so its mentions belong to the winner.
        # Without this the next extraction re-attaches mentions to a merged-away
        # row and silently undoes the resolution — no error, no log, just a
        # duplicate quietly coming back to life and the entity counts drifting
        # upwards after every sync.
        return await _follow_merge(session, merged_into), False

    async def _queue_sync(self, memory_id: UUID) -> bool:
        """Ask for this memory's neighbourhood of the graph to be re-projected.

        The job names the memory rather than the entities, and the sync widens it
        to the entities Postgres says the memory mentions. That is the difference
        between a payload describing a *change* and one describing a projection:
        the in-memory candidates here are not the whole answer — an entity this
        memory mentions may have been merged since, and only Postgres knows.

        A `False` return means no queue is wired, not that anything failed. The
        enqueue itself is not caught: it is one insert into the same database that
        has just committed, so a failure here is a database problem rather than a
        graph problem, and swallowing it would leave the projection behind with
        nothing recording that it is.
        """
        if self._queue is None:
            return False
        await enqueue_sync(self._queue, memory_ids=[memory_id])
        return True


async def _follow_merge(session: AsyncSession, entity_id: UUID, depth: int = 8) -> UUID:
    """The entity a merged-away one now belongs to.

    Bounded, for the reason `resolution._follow` is: a chain is data, and an
    unbounded walk over cyclic data is an infinite loop holding a transaction
    open. The last id reached is returned rather than raising, because a
    too-deep chain should cost a misfiled mention, not a failed extraction.
    """
    current = entity_id
    for _ in range(depth):
        row = await session.get(models.Entity, current)
        if row is None or row.merged_into_id is None:
            return current
        current = row.merged_into_id
    logger.warning("extract.merge_chain_too_deep", entity_id=str(entity_id))
    return current


def _offsets_hold(chunk_text: str, candidate: ExtractedEntity) -> bool:
    """Whether the span really says what the entity claims it says."""
    if candidate.char_start < 0 or candidate.char_end > len(chunk_text):
        return False
    return chunk_text[candidate.char_start : candidate.char_end] == candidate.name


def _finish(report: ExtractReport, started: float) -> ExtractReport:
    report.duration_ms = int((time.monotonic() - started) * 1000)
    return report
