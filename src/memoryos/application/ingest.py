"""Recording one observed item: blob, artifact, event, memory, jobs.

**Lifted out of `SyncSource` by M10.0, and the lifting is the point.** Chat is
the first thing to enter this corpus that nobody walks. There is no directory to
enumerate, no cursor to keep and no second observation of the same message — a
typed line is pushed once and is never seen again — so `Connector.observe` has
nothing to offer it and `SyncSource.__call__` has nothing to drive.

What chat needs is everything *downstream* of the walk, which was the back half
of one private method. The milestone was explicit that there must be no parallel
path, and the two ways to satisfy that are to make chat pretend to be a
filesystem or to make the shared half callable. A fake connector yielding one
item, driven by a sync that then writes a cursor and a `last_sync_at` for a
source that cannot be synced, would be a parallel path wearing the sync's
clothes: the same code would run and every one of its assumptions would be
false.

So this is the function both writers call, and `ObservedItem` turns out to
describe a typed message without a single field bent to fit — an external key,
some bytes, a content hash, a media type and a time with its provenance is what
an item *is*, independently of whether anything walked to find it.

**The session is the caller's.** `SyncSource` opens one transaction per item so
that a crash at file eight thousand keeps the first seven thousand nine hundred
and ninety-nine; the chat path opens one per message so that the reply can be
sent the moment it commits. Both want the same unit of work — memory and jobs,
together or not at all — and neither wants the other's loop, so the transaction
is opened outside and the invariant is stated here.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.adapters.db.repositories import (
    SqlAlchemyArtifactRepository,
    SqlAlchemyEventLog,
    SqlAlchemyMemoryRepository,
)
from memoryos.application.graph_sync import graph_sync_spec
from memoryos.application.ports import BlobStore, ObservedItem
from memoryos.application.projection import memory_from_event
from memoryos.domain.entities import IngestionEvent, RawArtifact, Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobSpec, JobType
from memoryos.domain.values import EventType


class UnreadableItem(RuntimeError):
    """An item whose bytes are neither readable nor already stored."""


@dataclass(frozen=True, slots=True)
class Ingested:
    """What one recorded item became.

    `memory_id` is here because the chat path needs it before the transaction it
    was created in has even committed — the reply says "stored" and links to the
    memory, and a caller that had to go and find the row it just wrote by its
    external key would be re-deriving something this function already knew.

    `superseded` distinguishes a new version of a known item from a new item. The
    sync counts them separately, and a chat message is always the second thing:
    its key carries the message id, so nothing can ever supersede it.
    """

    memory_id: UUID
    superseded: bool


async def ingest_item(
    session: AsyncSession, blobs: BlobStore, source: Source, item: ObservedItem
) -> Ingested | None:
    """Record one item in the caller's transaction.

    Returns None when nothing changed — the item is byte-identical to the current
    version, so there is no blob write, no artifact, no event, no version and no
    job. That path is what makes re-running a sync free, which is what makes
    running it often a reasonable thing to do.
    """
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
        return None

    if not artifact_known:
        # Bytes before the row that promises them: an artifact referencing a blob
        # nobody stored is a dangling promise. A connector that hashes by
        # streaming into the blob store has already done this; one that cannot
        # stream leaves it to us.
        if not await blobs.exists(item.content_hash):
            if item.read_bytes is None:
                # A pushed source promised its bytes were already stored and they
                # are not. Named rather than left as a `TypeError` on `None()`,
                # because the two causes are very different: a connector that
                # forgot to supply a reader, or a blob store that lost a write.
                raise UnreadableItem(
                    f"{item.external_key!r} carries no reader and its bytes are "
                    f"not in the blob store; a source that supplies no "
                    f"`read_bytes` must have stored them first"
                )
            await blobs.put(item.content_hash, await item.read_bytes())
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
    await memories.add_version(memory_from_event(event, memory_id=memory_id))

    # In the same transaction as the memory it refers to. This is the entire
    # reason the queue is a table rather than a broker: there is no window in
    # which the memory exists and the job that processes it does not, or the
    # reverse.
    await enqueue_in(
        session,
        JobSpec(
            job_type=JobType.NORMALIZE_MEMORY,
            payload={"memory_id": str(memory_id)},
            # Per memory version, so a re-run cannot queue the same work twice
            # while the first attempt is still in flight.
            dedupe_key=str(memory_id),
        ),
    )
    # And the graph, in the same transaction and for the same reason.
    #
    # The graph projects *every* current memory, not only the ones extraction has
    # reached, so a new memory is a projection change on its own — before
    # anything has parsed it, and whether or not this deployment has an API key
    # to extract entities with. Without this enqueue the only thing that ever
    # announced a memory to the graph was extraction, and `graph verify` would
    # report a permanent divergence for every file nobody had extracted yet.
    #
    # It also handles the superseded version implicitly: the projection reads
    # `is_current`, so re-projecting this memory's neighbourhood removes the node
    # for the version this one has just replaced.
    await enqueue_in(session, graph_sync_spec(memory_ids=[memory_id]))

    return Ingested(memory_id=memory_id, superseded=current is not None)
