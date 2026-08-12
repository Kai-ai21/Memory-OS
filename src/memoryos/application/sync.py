"""The sync use case: turn what a connector observed into stored state.

The shape worth noticing is that nothing in this file knows about filesystems.
It takes a `Connector` and drives it. Every source has a different walking
problem; this is the identical downstream half, and keeping it that way is what
makes the second connector cheap.
"""

import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.adapters.db.repositories import (
    SqlAlchemyArtifactRepository,
    SqlAlchemyEventLog,
    SqlAlchemyMemoryRepository,
    SqlAlchemySourceRepository,
)
from memoryos.application.graph_sync import graph_sync_spec
from memoryos.application.ports import BlobStore, Connector, ObservedItem
from memoryos.application.projection import memory_from_event, recorded_at_of
from memoryos.domain.entities import IngestionEvent, RawArtifact, Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobSpec, JobType
from memoryos.domain.values import EventType, TimeProvenance

logger = structlog.get_logger(__name__)

CURSOR_SEEN = "seen"


@dataclass(slots=True)
class SyncReport:
    observed: int = 0
    new: int = 0
    modified: int = 0
    deleted: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "observed": self.observed,
            "new": self.new,
            "modified": self.modified,
            "deleted": self.deleted,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


class SyncSource:
    """Drive a connector and record everything it reports.

    One transaction per item, never one per sync. Ten thousand files inside a
    single transaction holds locks for minutes, produces enormous WAL, and
    discards all of it on any failure. Per-item transactions make a sync
    resumable: a crash at file eight thousand keeps the first seven thousand
    nine hundred and ninety-nine.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        connector: Connector,
        blob_store: BlobStore,
    ) -> None:
        self._sessions = session_factory
        self._connector = connector
        self._blobs = blob_store

    async def __call__(self, source_id: UUID, *, full: bool = False) -> SyncReport:
        started = time.monotonic()
        report = SyncReport()

        source = await self._load_source(source_id)
        log = logger.bind(source=source.name, source_id=str(source_id), full=full)
        log.info("sync.started")

        # Keys only, never items. Holding the items would put the entire source
        # in memory and undo the streaming the connector went to the trouble of
        # providing.
        observed_keys: set[str] = set()
        cursor_seen: dict[str, list[Any]] = dict(source.cursor.get(CURSOR_SEEN) or {})

        async for item in self._connector.observe(source, full=full):
            report.observed += 1
            observed_keys.add(item.external_key)

            try:
                changed = await self._ingest(source, item)
            except Exception:
                # One unwritable file must not end a sync of ten thousand. It is
                # counted, logged with its traceback, and the walk continues.
                report.errors += 1
                log.exception("item.failed", key=item.external_key)
                continue

            if changed is None:
                report.skipped += 1
            elif changed:
                report.modified += 1
            else:
                report.new += 1

            if item.fingerprint is not None:
                cursor_seen[item.external_key] = item.fingerprint

        if full:
            report.deleted = await self._detect_deletions(source, observed_keys, log)
            # A full sync saw everything there is, so anything left in the
            # cursor is gone and its entry is dead weight.
            cursor_seen = {k: v for k, v in cursor_seen.items() if k in observed_keys}

        await self._finish(source, cursor_seen, full=full)

        report.duration_ms = int((time.monotonic() - started) * 1000)
        log.info("sync.finished", **report.as_dict())
        return report

    async def _load_source(self, source_id: UUID) -> Source:
        async with self._sessions() as session:
            source = await SqlAlchemySourceRepository(session).get(source_id)
        if source is None:
            raise LookupError(f"no such source: {source_id}")
        return source

    async def _ingest(self, source: Source, item: ObservedItem) -> bool | None:
        """Record one item in one transaction.

        Returns None if nothing changed, False if the memory is new, True if it
        is a new version of something already known.
        """
        async with self._sessions.begin() as session:
            memories = SqlAlchemyMemoryRepository(session)
            artifacts = SqlAlchemyArtifactRepository(session)

            current = await memories.get_current(source.id, item.external_key)
            artifact_known = await artifacts.exists(item.content_hash)

            if (
                artifact_known
                and current is not None
                and current.content_hash == item.content_hash
                and current.deleted_at is None
            ):
                # Unchanged: no blob write, no artifact, no event, no version,
                # no job. This is the path that makes re-running a sync free,
                # which is what makes running it often a reasonable thing to do.
                return None

            if not artifact_known:
                # Bytes before the row that promises them: an artifact
                # referencing a blob nobody stored is a dangling promise. A
                # connector that hashes by streaming into the blob store has
                # already done this; one that cannot stream leaves it to us.
                if not await self._blobs.exists(item.content_hash):
                    await self._blobs.put(item.content_hash, await item.read_bytes())
                await artifacts.add(
                    RawArtifact(
                        content_hash=item.content_hash,
                        byte_size=item.byte_size,
                        media_type=item.media_type,
                    )
                )

            event = await SqlAlchemyEventLog(session).append(
                IngestionEvent(
                    id=new_id(),
                    event_type=EventType.ARTIFACT_OBSERVED,
                    source_id=source.id,
                    external_key=item.external_key,
                    occurred_at=item.occurred_at,
                    occurred_at_source=item.occurred_at_source,
                    content_hash=item.content_hash,
                    payload={"byte_size": item.byte_size, "media_type": item.media_type},
                )
            )

            memory_id = new_id()
            await memories.add_version(
                memory_from_event(event, memory_id=memory_id),
            )

            # In the same transaction as the memory it refers to. This is the
            # entire reason the queue is a table rather than a broker: there is
            # no window in which the memory exists and the job that processes it
            # does not, or the reverse.
            await enqueue_in(
                session,
                JobSpec(
                    job_type=JobType.NORMALIZE_MEMORY,
                    payload={"memory_id": str(memory_id)},
                    # Per memory version, so a re-run cannot queue the same work
                    # twice while the first attempt is still in flight.
                    dedupe_key=str(memory_id),
                ),
            )
            # And the graph, in the same transaction and for the same reason.
            #
            # The graph projects *every* current memory, not only the ones
            # extraction has reached, so a new memory is a projection change on
            # its own — before anything has parsed it, and whether or not this
            # deployment has an API key to extract entities with. Without this
            # enqueue the only thing that ever announced a memory to the graph was
            # extraction, and `graph verify` would report a permanent divergence
            # for every file nobody had extracted yet.
            #
            # It also handles the superseded version implicitly: the projection
            # reads `is_current`, so re-projecting this memory's neighbourhood
            # removes the node for the version this one has just replaced.
            await enqueue_in(session, graph_sync_spec(memory_ids=[memory_id]))

            return current is not None

    async def _detect_deletions(
        self, source: Source, observed_keys: set[str], log: structlog.BoundLogger
    ) -> int:
        """Find items that are no longer there.

        This only runs on a full sync, and it can only run on a full sync. A
        deleted file produces no observation at all — absence is not an event —
        so the only way to notice is to compare the complete observed set
        against the complete known set. That is exactly why `sources` carries
        `last_full_sync_at` separately from `last_sync_at`.
        """
        async with self._sessions() as session:
            stmt = select(models.Memory.external_key, models.Memory.id).where(
                models.Memory.source_id == source.id,
                models.Memory.is_current.is_(True),
                models.Memory.deleted_at.is_(None),
            )
            known: dict[str, UUID] = {
                str(key): memory_id for key, memory_id in (await session.execute(stmt))
            }

        deleted = 0
        for external_key in sorted(set(known) - observed_keys):
            async with self._sessions.begin() as session:
                event = await SqlAlchemyEventLog(session).append(
                    IngestionEvent(
                        id=new_id(),
                        event_type=EventType.ITEM_DELETED,
                        source_id=source.id,
                        external_key=external_key,
                        # We know when we noticed the absence — that is
                        # `recorded_at`. We do not know when the deletion
                        # happened, and claiming otherwise would be a
                        # fabrication of exactly the kind the provenance column
                        # exists to prevent.
                        occurred_at=None,
                        occurred_at_source=TimeProvenance.UNKNOWN,
                        content_hash=None,
                    )
                )
                # Stamped from the event rather than from `now()`, so that a
                # replay of this event lands on the same `deleted_at`.
                await SqlAlchemyMemoryRepository(session).tombstone(
                    known[external_key], recorded_at_of(event)
                )
                # A tombstoned memory leaves the projection: `graph_projection`
                # reads `deleted_at IS NULL`, so the sync prunes the node and
                # writes nothing back. Queued in the tombstone's own transaction,
                # so there is no window where the row is deleted and the job that
                # removes its node does not exist.
                await enqueue_in(
                    session, graph_sync_spec(memory_ids=[known[external_key]])
                )
            log.info("item.deleted", key=external_key)
            deleted += 1

        return deleted

    async def _finish(
        self, source: Source, cursor_seen: dict[str, list[Any]], *, full: bool
    ) -> None:
        values: dict[str, Any] = {
            "cursor": {CURSOR_SEEN: cursor_seen},
            "last_sync_at": func.now(),
        }
        # Only a full sweep can claim to have reconciled deletions, so only a
        # full sweep moves this timestamp.
        if full:
            values["last_full_sync_at"] = func.now()

        async with self._sessions.begin() as session:
            await session.execute(
                update(models.Source).where(models.Source.id == source.id).values(**values)
            )
