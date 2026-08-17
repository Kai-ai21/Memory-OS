"""Taking a memory out, at two levels, and meaning it.

**"Users can permanently delete memories" has been a stated guarantee since Phase
1, and until M10.4 there was no button for it.** Tombstoning existed — the sync
writes one when a file disappears — but the only thing that could ask for it was a
full sweep noticing an absence. Nothing in the product could delete anything on
purpose, which made the guarantee a claim about the schema rather than about the
system.

## Two levels, and the difference is not a detail

**Remove from view** writes the tombstone the sync already writes: an
`item_deleted` event and a `deleted_at` stamp. Every retrieval path filters on that
column — `SearchFilters.include_deleted` defaults false, the graph projection reads
`deleted_at IS NULL`, citations and the timeline do the same — so the memory stops
being findable and stops being cited. The row, the chunks, the vectors and the
bytes are all still there, which is exactly why it is recoverable: re-ingesting the
same key un-deletes it through the ordinary version path.

**Delete permanently** removes the content. Every version of the memory, its
chunks and therefore its vectors, its entity mentions, its graph nodes, its tags,
the transcript rows that carry the same text, and the blob itself. What survives is
the log.

## Why the log survives, said plainly

The append-only log is the system of record; the corpus is a projection of it.
Deleting rows from it would not be a deletion, it would be a *rewrite of history* —
and the corpus would then disagree with the log in a way no replay could resolve.
So the purge is itself an event: `item_purged`, appended, saying that the content
observed at this key was removed.

The consequence is the crypto-shredding tension Phase 1 named, and it is worth
stating without softening because the confirmation dialog states it too: **the log
still records that something was observed here — a hash, a byte size, a date — and
the thing observed is gone.** That is less than "as though it never happened" and
more than any system can honestly offer while keeping an audit log. The one thing
this module refuses to do is claim the stronger version.

Replay closes the loop. It reads the purge, and skips the key entirely — including
the `artifact_observed` events before it, whose blobs were deliberately shredded.
A rebuild after a purge produces a corpus without the item and does not fail
looking for bytes nobody kept.

## What a cascade does and does not reach

`memory_chunks`, `entity_mentions`, `entity_relationships`, `change_summaries` and
the three evidence tables hold `ON DELETE CASCADE` foreign keys into `memories`, so
deleting the memory row takes all of them — including the vectors, which live on
the chunks and are the reason the cascade was specified in M1.1 rather than
discovered here.

Three things are *not* reachable by a cascade and are deleted explicitly, because
each one holds a copy of the content or a pointer that would outlive it:

* `chat_messages`, which stores the text of a typed message a second time. Leaving
  it would make "delete permanently" false in the most visible place in the
  product — the conversation the message was typed into.
* `memory_tags`, keyed by name rather than by id precisely so it survives a replay,
  which means it also survives a cascade.
* the blob, which is a file rather than a row, and only when no surviving memory or
  attachment still references those bytes. Two uploads of one document share an
  artifact; shredding it because one of them was purged would corrupt the other.

`raw_artifacts` stays. It is the hash, the byte size and the media type — the
metadata the log's foreign key points at, and no content. Deleting it would break
that key, which is the same rewrite-of-history the log's append-only rule forbids.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import ColumnElement, delete, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.adapters.db.mappers import to_source
from memoryos.adapters.db.repositories import (
    SqlAlchemyEventLog,
    SqlAlchemyMemoryRepository,
)
from memoryos.application.graph_sync import graph_sync_spec
from memoryos.application.ingest import ingest_item
from memoryos.application.ports import BlobStore, ObservedItem
from memoryos.application.projection import recorded_at_of
from memoryos.domain.entities import IngestionEvent, Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash, EventType, TimeProvenance

logger = structlog.get_logger(__name__)


class NoSuchMemory(LookupError):
    """A memory id, or an item, that nothing has."""


class NotDeletable(ValueError):
    """A deletion that would not mean what it says.

    Its own type because the transports answer it differently from a missing row:
    the caller asked about something that exists and the operation does not apply
    to it, which is a 409 rather than a 404.
    """


class BlobsSurvived(RuntimeError):
    """The corpus was purged and some bytes could not be removed.

    Raised *after* the database transaction has committed, which is deliberate and
    is the lesser of the two available failures. The alternative ordering — shred
    first, then commit — can leave bytes destroyed for a memory that still exists,
    and a rollback cannot put a file back.

    So the corpus is correct, the content is unreachable through every read path,
    and one or more files are still on disk. That is worth an exception rather than
    a log line: "delete permanently" was asked for, and the honest report is that
    it was three-quarters done and which quarter is missing.
    """


@dataclass(frozen=True, slots=True)
class Item:
    """One item's durable identity, which is what a deletion works on.

    Not a memory id. A memory id names *one version*, and every operation here
    applies to the item as a whole: tombstoning marks the current version,
    correcting adds one, and purging removes all of them. Addressing versions
    would make "delete this memory" mean "delete this revision of it", leaving
    the previous text in the corpus — which is the failure mode that makes a
    deletion guarantee worthless.
    """

    source_id: UUID
    source_name: str
    external_key: str


@dataclass(frozen=True, slots=True)
class PurgeScope:
    """Exactly what a permanent deletion will remove, counted before it runs.

    **The confirmation names counts because a person cannot consent to an
    unspecified amount of loss.** "Delete this memory?" is answerable; "delete
    this memory, its 14 chunks, the 9 entity mentions that were the only evidence
    for two entities, and the 3 turns of conversation it appears in?" is a
    different question, and it is the true one.

    Read in one pass before anything is written, so the numbers in the dialog are
    the numbers the operation will hit rather than an estimate from a previous
    screen.
    """

    items: tuple[Item, ...]
    memories: int
    chunks: int
    embedded_chunks: int
    mentions: int
    # Entities whose *last* mention is in what is being deleted. They are not
    # deleted — nothing cascades to `entities` — but every read of an entity joins
    # through its mentions, so after this they are rows no query reaches. Named
    # separately from `mentions` because "9 mentions removed" and "2 entities the
    # corpus will no longer know about" are different facts and the second is the
    # one somebody minds.
    orphaned_entities: int
    tags: int
    turns: int
    attachments: int
    # Evidence rows in the decision tables that cite what is being deleted. These
    # cascade, and a citation to a document that no longer exists is a citation to
    # nothing — but a decision losing its evidence is worth being told about
    # before rather than after.
    evidence: int
    # Artifacts whose bytes will be shredded, and those kept because something
    # else still points at them. Two uploads of one document share an artifact.
    blobs: int
    shared_blobs: int
    # A first line of each item, for a dialog that has to say *which* memory.
    previews: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.items


@dataclass(frozen=True, slots=True)
class PurgeReport:
    """What a permanent deletion actually removed."""

    items: int
    memories: int
    chunks: int
    mentions: int
    tags: int
    turns: int
    blobs_shredded: int
    # Named rather than counted: the operator's next move is to look for these
    # files, and a count cannot be grepped for.
    blobs_surviving: tuple[str, ...] = field(default_factory=tuple)
    events_appended: int = 0


# --------------------------------------------------------------------------
# Resolving what was asked for
# --------------------------------------------------------------------------


async def item_of_memory(
    session: AsyncSession, memory_id: UUID
) -> Item:
    """The item one version belongs to.

    Every entry point here takes a memory id — that is what a UI has, and what a
    message links to — and immediately widens it to the item, for the reason
    `Item` gives.
    """
    row = (
        await session.execute(
            select(
                models.Memory.source_id, models.Memory.external_key, models.Source.name
            )
            .join(models.Source, models.Source.id == models.Memory.source_id)
            .where(models.Memory.id == memory_id)
        )
    ).first()
    if row is None:
        raise NoSuchMemory(f"no memory with id {memory_id}")
    return Item(source_id=row[0], source_name=row[2], external_key=row[1])


async def _items_of_source(session: AsyncSession, source_id: UUID) -> list[Item]:
    """Every distinct item a source has produced, tombstoned ones included.

    Tombstoned ones included, and that is the point of asking here rather than
    reusing a retrieval read: deleting a source has to remove what it hid as well
    as what it shows, or the "remove from view" level becomes a way to survive the
    "delete everything" one.
    """
    name = (
        await session.execute(
            select(models.Source.name).where(models.Source.id == source_id)
        )
    ).scalar_one_or_none()
    if name is None:
        raise NoSuchMemory(f"no source with id {source_id}")

    keys = (
        await session.execute(
            select(models.Memory.external_key)
            .where(models.Memory.source_id == source_id)
            .distinct()
            .order_by(models.Memory.external_key)
        )
    ).scalars().all()
    return [
        Item(source_id=source_id, source_name=name, external_key=key) for key in keys
    ]


# --------------------------------------------------------------------------
# Level one: remove from view
# --------------------------------------------------------------------------


async def tombstone(
    session_factory: async_sessionmaker[AsyncSession],
    memory_id: UUID,
) -> Item:
    """Exclude a memory from retrieval, reversibly.

    The same two writes the sync makes when a file disappears — an `item_deleted`
    event and a `deleted_at` stamped from that event rather than from the clock, so
    a replay months later lands on the same value — plus the graph sync that prunes
    the node. Sharing the path is what makes this level trustworthy: it is not a
    new kind of hiding, it is the one the corpus has filtered on for nine
    milestones.

    Idempotent. Tombstoning something already tombstoned appends no event and
    changes nothing, because the second event would claim a second deletion that
    did not happen.
    """
    async with session_factory.begin() as session:
        item = await item_of_memory(session, memory_id)
        current = await SqlAlchemyMemoryRepository(session).get_current(
            item.source_id, item.external_key
        )
        if current is None:
            raise NoSuchMemory(
                f"{item.external_key!r} has no current version to remove from view"
            )
        if current.deleted_at is not None:
            logger.info("deletion.already_tombstoned", key=item.external_key)
            return item

        event = await SqlAlchemyEventLog(session).append(
            IngestionEvent(
                id=new_id(),
                event_type=EventType.ITEM_DELETED,
                source_id=item.source_id,
                external_key=item.external_key,
                # We know when we were told, which is `recorded_at`. We do not know
                # when the thought stopped being true, and claiming otherwise is
                # the fabrication the provenance column exists to prevent.
                occurred_at=None,
                occurred_at_source=TimeProvenance.UNKNOWN,
                content_hash=None,
            )
        )
        await SqlAlchemyMemoryRepository(session).tombstone(
            current.id, recorded_at_of(event)
        )
        await enqueue_in(session, graph_sync_spec(memory_ids=[current.id]))

    logger.info(
        "deletion.tombstoned", key=item.external_key, source=item.source_name
    )
    return item


async def restore(
    session_factory: async_sessionmaker[AsyncSession],
    blobs: BlobStore,
    memory_id: UUID,
    *,
    now: datetime | None = None,
) -> Item:
    """Bring a tombstoned memory back, through the ordinary version path.

    **Not an `UPDATE` clearing `deleted_at`.** The corpus is a projection of the
    log, so un-deleting has to be something the log says: this re-observes the same
    bytes at the same key, which `ingest_item` records as a new version and which a
    replay reproduces without knowing anything about restoration. Clearing the
    column directly would produce a corpus that no replay could arrive at — the
    exact divergence M1.7 exists to make impossible.

    The bytes have to still be there, which they are: this is the level that keeps
    them. A purged item cannot be restored, and says so rather than failing on a
    missing blob.
    """
    at = now or datetime.now(UTC)
    async with session_factory.begin() as session:
        item = await item_of_memory(session, memory_id)
        current = await SqlAlchemyMemoryRepository(session).get_current(
            item.source_id, item.external_key
        )
        if current is None:
            raise NoSuchMemory(f"{item.external_key!r} has no version to restore")
        if current.deleted_at is None:
            raise NotDeletable(
                f"{item.external_key!r} is not removed from view, so there is "
                f"nothing to restore"
            )
        # Already a `ContentHash`: `current` is the domain entity, not the row.
        content_hash = current.content_hash
        if not await blobs.exists(content_hash):
            raise NotDeletable(
                f"the bytes for {item.external_key!r} are not in the blob store, "
                f"so it cannot be restored. A permanent deletion shreds them "
                f"deliberately and is not reversible; anything else here is a "
                f"blob store that has lost a write."
            )

        artifact = await session.get(models.RawArtifact, content_hash.value)
        recorded = await ingest_item(
            session,
            blobs,
            # A `Source` entity is what `ingest_item` takes and the id is all it
            # reads off it, so this is loaded rather than reconstructed.
            await _source_entity(session, item.source_id),
            ObservedItem(
                external_key=item.external_key,
                content_hash=content_hash,
                byte_size=artifact.byte_size if artifact else 0,
                media_type=artifact.media_type if artifact else None,
                occurred_at=current.occurred_at,
                occurred_at_source=TimeProvenance(current.occurred_at_source),
                # Already stored — this is the same artifact, which is the whole
                # reason a restore is cheap. `None` is what M10.2's optional reader
                # means: do not fetch, it is there.
                read_bytes=None,
                fingerprint=None,
            ),
        )
        if recorded is None:
            # Unreachable: `ingest_item` only returns None for an identical
            # *undeleted* current version, and this one is tombstoned. Named
            # rather than asserted, because a silent None would report a restore
            # that restored nothing.
            raise NotDeletable(
                f"{item.external_key!r} was not restored; the ingest path treated "
                f"it as unchanged, which cannot happen for a tombstoned item"
            )

    logger.info("deletion.restored", key=item.external_key, at=at.isoformat())
    return item


async def _source_entity(session: AsyncSession, source_id: UUID) -> Source:
    """The source as `ingest_item` takes it.

    Loaded rather than reconstructed from the item: `ingest_item` reads the id off
    it, and building one with a made-up config would work today and be wrong the
    first time anything downstream looked at the rest of the entity.
    """
    row = await session.get(models.Source, source_id)
    if row is None:
        raise NoSuchMemory(f"no source with id {source_id}")
    return to_source(row)


# --------------------------------------------------------------------------
# Level two: delete permanently
# --------------------------------------------------------------------------


async def scope_of(
    session_factory: async_sessionmaker[AsyncSession], items: Sequence[Item]
) -> PurgeScope:
    """Count what purging these items would remove.

    One read, before anything is written. Every count is scoped to the items rather
    than derived from a total, because the number that matters in a confirmation is
    what *this* operation will take.
    """
    if not items:
        return PurgeScope(
            items=(),
            memories=0,
            chunks=0,
            embedded_chunks=0,
            mentions=0,
            orphaned_entities=0,
            tags=0,
            turns=0,
            attachments=0,
            evidence=0,
            blobs=0,
            shared_blobs=0,
        )

    async with session_factory() as session:
        memory_ids, hashes = await _versions_of(session, items)
        keys = [item.external_key for item in items]

        chunks, embedded = 0, 0
        if memory_ids:
            row = (
                await session.execute(
                    select(
                        func.count(models.MemoryChunk.id),
                        func.count(models.MemoryChunk.embedding),
                    ).where(models.MemoryChunk.memory_id.in_(memory_ids))
                )
            ).one()
            chunks, embedded = int(row[0]), int(row[1])

        mentions = 0
        orphaned = 0
        if memory_ids:
            mentions = int(
                (
                    await session.execute(
                        select(func.count(models.EntityMention.id)).where(
                            models.EntityMention.memory_id.in_(memory_ids)
                        )
                    )
                ).scalar_one()
            )
            orphaned = await _entities_left_unmentioned(session, memory_ids)

        tags = int(
            (
                await session.execute(
                    select(func.count(models.MemoryTag.id)).where(
                        models.MemoryTag.external_key.in_(keys),
                        models.MemoryTag.source_id.in_(
                            {item.source_id for item in items}
                        ),
                    )
                )
            ).scalar_one()
        )
        turn_ids = (
            await session.execute(
                select(models.ChatMessage.id).where(
                    models.ChatMessage.external_key.in_(keys)
                )
            )
        ).scalars().all()
        attachments = 0
        if turn_ids:
            attachments = int(
                (
                    await session.execute(
                        select(func.count(models.ChatAttachment.id)).where(
                            models.ChatAttachment.message_id.in_(turn_ids)
                        )
                    )
                ).scalar_one()
            )
        evidence = await _evidence_citing(session, memory_ids)
        shreddable, shared = await _blob_split(session, hashes, memory_ids)
        previews = await _previews(session, items)

    return PurgeScope(
        items=tuple(items),
        memories=len(memory_ids),
        chunks=chunks,
        embedded_chunks=embedded,
        mentions=mentions,
        orphaned_entities=orphaned,
        tags=tags,
        turns=len(turn_ids),
        attachments=attachments,
        evidence=evidence,
        blobs=len(shreddable),
        shared_blobs=len(shared),
        previews=previews,
    )


async def scope_of_memory(
    session_factory: async_sessionmaker[AsyncSession], memory_id: UUID
) -> PurgeScope:
    """What purging the item this version belongs to would remove."""
    async with session_factory() as session:
        item = await item_of_memory(session, memory_id)
    return await scope_of(session_factory, [item])


async def scope_of_source(
    session_factory: async_sessionmaker[AsyncSession], source_id: UUID
) -> PurgeScope:
    """What deleting a whole source would remove.

    The most destructive operation in the product, so the counts are the same
    counts a single item's dialog shows — computed the same way, by the same
    function — rather than a separate estimate that could disagree with the
    operation it describes.
    """
    async with session_factory() as session:
        items = await _items_of_source(session, source_id)
    return await scope_of(session_factory, items)


async def purge(
    session_factory: async_sessionmaker[AsyncSession],
    blobs: BlobStore,
    items: Sequence[Item],
) -> PurgeReport:
    """Remove the content of these items, keeping the log's record that it existed.

    **One transaction per item**, for the reason the sync and the replay both use
    one per item: purging a source of eight thousand files must not hold locks for
    minutes, and a failure at file seven thousand must leave the first six thousand
    nine hundred and ninety-nine genuinely deleted rather than rolled back into
    existence.

    Blobs are shredded *after* the transaction that removed their rows, and the
    ordering is a choice between two imperfect failures. Shredding first can
    destroy bytes for a memory that survives a rollback, and no rollback puts a
    file back. Shredding second can leave a file with nothing pointing at it. The
    second is recoverable and the first is not, so the second is what happens —
    and `BlobsSurvived` names the files rather than logging them, because the
    operator's next move is to go and look.
    """
    report = PurgeReport(
        items=0,
        memories=0,
        chunks=0,
        mentions=0,
        tags=0,
        turns=0,
        blobs_shredded=0,
    )
    shred: list[ContentHash] = []
    surviving: list[str] = []

    for item in items:
        async with session_factory.begin() as session:
            counted, hashes = await _purge_one(session, item)
        report = PurgeReport(
            items=report.items + 1,
            memories=report.memories + counted["memories"],
            chunks=report.chunks + counted["chunks"],
            mentions=report.mentions + counted["mentions"],
            tags=report.tags + counted["tags"],
            turns=report.turns + counted["turns"],
            blobs_shredded=0,
            events_appended=report.events_appended + 1,
        )
        shred.extend(hashes)

    # After every row is gone, so that "is anything still referencing these bytes"
    # is asked of the corpus as it now is. Asked once rather than per item, because
    # two purged items may share an artifact and each would otherwise see the
    # other as a reason to keep it.
    shredded = 0
    async with session_factory() as session:
        for content_hash in dict.fromkeys(shred):
            if await _still_referenced(session, content_hash):
                continue
            try:
                await blobs.delete(content_hash)
            except Exception:
                logger.warning(
                    "deletion.blob_survived", digest=content_hash.value, exc_info=True
                )
                surviving.append(content_hash.value)
                continue
            shredded += 1

    report = PurgeReport(
        items=report.items,
        memories=report.memories,
        chunks=report.chunks,
        mentions=report.mentions,
        tags=report.tags,
        turns=report.turns,
        blobs_shredded=shredded,
        blobs_surviving=tuple(surviving),
        events_appended=report.events_appended,
    )
    logger.info(
        "deletion.purged",
        items=report.items,
        memories=report.memories,
        chunks=report.chunks,
        blobs=report.blobs_shredded,
    )
    if surviving:
        raise BlobsSurvived(
            f"the corpus is purged and {len(surviving)} blob(s) could not be "
            f"removed: {', '.join(surviving)}. The content is unreachable through "
            f"every read path and the bytes are still in the blob store; delete "
            f"them by hand, or run the purge again once the store is writable."
        )
    return report


async def _purge_one(
    session: AsyncSession, item: Item
) -> tuple[dict[str, int], list[ContentHash]]:
    """One item's rows, in the order their dependencies allow.

    The event first, because it is the record that this happened and everything
    after it is the consequence. If the transaction fails at any later point,
    nothing is written at all — including the claim.
    """
    memory_ids, hashes = await _versions_of(session, [item])
    if not memory_ids:
        raise NoSuchMemory(
            f"{item.external_key!r} has no versions in {item.source_name!r}"
        )

    await SqlAlchemyEventLog(session).append(
        IngestionEvent(
            id=new_id(),
            event_type=EventType.ITEM_PURGED,
            source_id=item.source_id,
            external_key=item.external_key,
            # A purge is not an observation, so there is nothing it happened *at*
            # in the world. `recorded_at` is when we were told, which is all that
            # is true.
            occurred_at=None,
            occurred_at_source=TimeProvenance.UNKNOWN,
            # Deliberately null even though the versions had hashes. The event's
            # foreign key into `raw_artifacts` is what keeps that row alive, and a
            # purge naming one artifact out of several versions would be an
            # arbitrary choice recorded as a fact. The observations before it
            # already say which bytes were seen.
            content_hash=None,
        )
    )

    mentions = int(
        (
            await session.execute(
                select(func.count(models.EntityMention.id)).where(
                    models.EntityMention.memory_id.in_(memory_ids)
                )
            )
        ).scalar_one()
    )
    chunks = int(
        (
            await session.execute(
                select(func.count(models.MemoryChunk.id)).where(
                    models.MemoryChunk.memory_id.in_(memory_ids)
                )
            )
        ).scalar_one()
    )

    # The transcript, explicitly. It carries the message text a second time, so
    # leaving it would keep the content in the most visible place in the product —
    # and it is keyed on `external_key` rather than by a foreign key into
    # `memories`, precisely so that a replay cannot take it, which means a cascade
    # cannot either. Attachments cascade from the turn.
    turns = len(
        (
            await session.execute(
                delete(models.ChatMessage)
                .where(models.ChatMessage.external_key == item.external_key)
                .returning(models.ChatMessage.id)
            )
        )
        .scalars()
        .all()
    )

    tags = len(
        (
            await session.execute(
                delete(models.MemoryTag)
                .where(
                    models.MemoryTag.source_id == item.source_id,
                    models.MemoryTag.external_key == item.external_key,
                )
                .returning(models.MemoryTag.id)
            )
        )
        .scalars()
        .all()
    )

    # Every version, not the current one. A purge of the current version alone
    # would leave the pre-correction text in the corpus, which is the failure that
    # makes a deletion guarantee worthless. The cascade takes chunks — and
    # therefore vectors — mentions, relationships, change summaries and evidence.
    await session.execute(
        delete(models.Memory).where(models.Memory.id.in_(memory_ids))
    )

    # Queued in the same transaction as the deletion, so there is no window in
    # which the rows are gone and the job that removes their graph nodes does not
    # exist. The sync prunes the neighbourhood and re-projects it from Postgres,
    # which no longer implies these nodes — so "delete, then project" removes them
    # without needing to be told they were deleted.
    await enqueue_in(session, graph_sync_spec(memory_ids=list(memory_ids)))

    return (
        {
            "memories": len(memory_ids),
            "chunks": chunks,
            "mentions": mentions,
            "tags": tags,
            "turns": turns,
        },
        hashes,
    )


async def purge_memory(
    session_factory: async_sessionmaker[AsyncSession],
    blobs: BlobStore,
    memory_id: UUID,
) -> PurgeReport:
    """Permanently delete the item this version belongs to."""
    async with session_factory() as session:
        item = await item_of_memory(session, memory_id)
    return await purge(session_factory, blobs, [item])


async def purge_source(
    session_factory: async_sessionmaker[AsyncSession],
    blobs: BlobStore,
    source_id: UUID,
    *,
    drop_source: bool = True,
) -> PurgeReport:
    """Permanently delete everything a source produced, and the source itself.

    `drop_source=False` leaves the registration in place, which is what a
    re-index-from-scratch wants: the same root, the same config, no memories, and
    the next sync walks it again. Dropping it is the default because "delete this
    source" said without qualification means the source too.

    The source row is deleted last and only if nothing references it. Its
    `ingestion_events` do — every event holds a foreign key to it — so in practice
    a source with any history keeps its registration and its config, and says so.
    That is the log's append-only rule reaching one table further than it looks
    like it should, and it is the same rule: the events are the record that this
    source existed and observed things.
    """
    async with session_factory() as session:
        items = await _items_of_source(session, source_id)
    report = await purge(session_factory, blobs, items)

    if drop_source:
        async with session_factory.begin() as session:
            events = int(
                (
                    await session.execute(
                        select(func.count(models.IngestionEvent.id)).where(
                            models.IngestionEvent.source_id == source_id
                        )
                    )
                ).scalar_one()
            )
            if events:
                logger.info(
                    "deletion.source_registration_kept",
                    source_id=str(source_id),
                    events=events,
                )
            else:
                await session.execute(
                    delete(models.Source).where(models.Source.id == source_id)
                )
    return report


# --------------------------------------------------------------------------
# Reads the two levels share
# --------------------------------------------------------------------------


async def _versions_of(
    session: AsyncSession, items: Sequence[Item]
) -> tuple[list[UUID], list[ContentHash]]:
    """Every version id of every item, and the artifacts they reference."""
    if not items:
        return [], []
    rows = (
        await session.execute(
            select(models.Memory.id, models.Memory.content_hash).where(
                _matches_any(items)
            )
        )
    ).all()
    return [row[0] for row in rows], [ContentHash(row[1]) for row in rows]


def _matches_any(items: Sequence[Item]) -> ColumnElement[bool]:
    """`(source_id, external_key) IN ((…), (…))`, as one predicate.

    A tuple `IN` rather than an `OR` chain of pairs, because a source deletion
    passes thousands of items and Postgres plans the first as a hash lookup and
    the second as thousands of clauses.
    """
    return tuple_(models.Memory.source_id, models.Memory.external_key).in_(
        [(item.source_id, item.external_key) for item in items]
    )


async def _entities_left_unmentioned(
    session: AsyncSession, memory_ids: Sequence[UUID]
) -> int:
    """Entities whose every mention is inside what is being deleted.

    A left-anti-join rather than two counts subtracted: an entity mentioned in
    both a deleted memory and a surviving one must not appear here, and
    subtracting totals cannot tell those apart.
    """
    if not memory_ids:
        return 0
    doomed = (
        select(models.EntityMention.entity_id)
        .where(models.EntityMention.memory_id.in_(memory_ids))
        .distinct()
        .subquery()
    )
    surviving = (
        select(models.EntityMention.entity_id)
        .where(models.EntityMention.memory_id.not_in(memory_ids))
        .distinct()
        .subquery()
    )
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(doomed)
                .outerjoin(
                    surviving, surviving.c.entity_id == doomed.c.entity_id
                )
                .where(surviving.c.entity_id.is_(None))
            )
        ).scalar_one()
    )


async def _evidence_citing(
    session: AsyncSession, memory_ids: Sequence[UUID]
) -> int:
    """Rows in the decision tables that cite what is being deleted.

    Three tables rather than a generic sweep of every inbound foreign key,
    because these are the three whose loss a person would want to hear about
    before agreeing: a decision, an assumption or an outcome losing the document
    it rested on. The rest of the cascade is derived data being rederived.
    """
    if not memory_ids:
        return 0
    total = 0
    for table in (
        models.DecisionEvidence,
        models.AssumptionEvidence,
        models.OutcomeEvidence,
    ):
        total += int(
            (
                await session.execute(
                    select(func.count(table.id)).where(
                        table.memory_id.in_(memory_ids)
                    )
                )
            ).scalar_one()
        )
    return total


async def _blob_split(
    session: AsyncSession,
    hashes: Sequence[ContentHash],
    doomed: Sequence[UUID],
) -> tuple[list[ContentHash], list[ContentHash]]:
    """Which artifacts lose their last reference, and which are shared.

    Asked before the deletion, so the surviving references have to exclude the
    versions about to go — which is what `doomed` is for. `_still_referenced` asks
    the same question afterwards, when the exclusion is unnecessary because the
    rows are gone.
    """
    shreddable: list[ContentHash] = []
    shared: list[ContentHash] = []
    for content_hash in dict.fromkeys(hashes):
        others = int(
            (
                await session.execute(
                    select(func.count(models.Memory.id)).where(
                        models.Memory.content_hash == content_hash.value,
                        models.Memory.id.not_in(doomed),
                    )
                )
            ).scalar_one()
        )
        attached = int(
            (
                await session.execute(
                    select(func.count(models.ChatAttachment.id)).where(
                        models.ChatAttachment.content_hash == content_hash.value
                    )
                )
            ).scalar_one()
        )
        (shared if others or attached else shreddable).append(content_hash)
    return shreddable, shared


async def _still_referenced(
    session: AsyncSession, content_hash: ContentHash
) -> bool:
    """Whether anything surviving still needs these bytes.

    `memories` and `chat_attachments`, and deliberately not `ingestion_events` or
    `raw_artifacts`. Those two are the log and the log's metadata: they record that
    the bytes were once observed, which is exactly the record a purge keeps, and
    treating them as a reason to keep the file would make permanent deletion
    impossible for anything that was ever ingested — which is everything.
    """
    memories = int(
        (
            await session.execute(
                select(func.count(models.Memory.id)).where(
                    models.Memory.content_hash == content_hash.value
                )
            )
        ).scalar_one()
    )
    if memories:
        return True
    attachments = int(
        (
            await session.execute(
                select(func.count(models.ChatAttachment.id)).where(
                    models.ChatAttachment.content_hash == content_hash.value
                )
            )
        ).scalar_one()
    )
    return bool(attachments)


async def _previews(
    session: AsyncSession, items: Sequence[Item]
) -> tuple[str, ...]:
    """A first line per item, so a dialog can say which memory this is.

    Capped, because a source deletion covers thousands and a confirmation that
    listed them all would be a wall nobody reads — which is the same as no
    confirmation. The counts carry the scale; these carry the identity.
    """
    limit = 5
    rows = (
        await session.execute(
            select(models.Memory.title, models.Memory.content, models.Memory.external_key)
            .where(_matches_any(items[:limit]), models.Memory.is_current.is_(True))
            .order_by(models.Memory.external_key)
        )
    ).all()
    previews: list[str] = []
    for title, content, key in rows:
        first = (content or "").strip().splitlines()
        text = title or (first[0] if first else key)
        previews.append(text[:120])
    return tuple(previews)
