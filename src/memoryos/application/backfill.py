"""Finding work for the embedder, and reporting what it has done.

`reembed` is the rehearsal for the model swap Phase 2 will demand. Making that
operation a query rather than a rebuild is most of the reason the model id is
in the cache key and on every chunk row.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Select, func, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.domain.jobs import JobSpec, JobType


@dataclass(frozen=True, slots=True)
class PendingMemory:
    id: UUID
    external_key: str
    chunks: int


@dataclass(frozen=True, slots=True)
class Stats:
    memories: int
    current_memories: int
    chunks: int
    embedded_chunks: int
    cache_entries: int
    models: dict[str, int]
    # The graph layer, counted here so one call answers "how big is this corpus"
    # across both halves of it. Defaulted rather than required because every
    # existing caller predates Phase 3 and none of them constructs a `Stats`.
    entities: int = 0
    relationships: int = 0

    @property
    def coverage(self) -> float:
        return self.embedded_chunks / self.chunks if self.chunks else 0.0


async def find_unembedded(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    model_id: str,
    source: str | None = None,
    stale_only: bool = False,
) -> list[PendingMemory]:
    """Memories with chunks this model has not embedded.

    `stale_only` narrows to chunks embedded by a *different* model, which is
    the re-embed case; otherwise a null embedding counts too, which is the
    ordinary backfill.
    """
    if stale_only:
        chunk_filter = models.MemoryChunk.embedding_model.is_distinct_from(model_id)
        chunk_filter = chunk_filter & models.MemoryChunk.embedding.is_not(None)
    else:
        chunk_filter = or_(
            models.MemoryChunk.embedding.is_(None),
            models.MemoryChunk.embedding_model.is_distinct_from(model_id),
        )

    stmt = (
        select(
            models.Memory.id,
            models.Memory.external_key,
            func.count(models.MemoryChunk.id),
        )
        .join(models.MemoryChunk, models.MemoryChunk.memory_id == models.Memory.id)
        .where(
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
            chunk_filter,
        )
        .group_by(models.Memory.id, models.Memory.external_key)
        .order_by(models.Memory.external_key)
    )
    if source is not None:
        stmt = stmt.where(
            models.Memory.source_id.in_(
                select(models.Source.id).where(models.Source.name == source)
            )
        )

    async with session_factory() as session:
        rows = await session.execute(stmt)
        return [PendingMemory(id=row[0], external_key=row[1], chunks=row[2]) for row in rows]


async def enqueue_embedding(
    session_factory: async_sessionmaker[AsyncSession], pending: list[PendingMemory]
) -> int:
    enqueued = 0
    for memory in pending:
        async with session_factory.begin() as session:
            job_id = await enqueue_in(
                session,
                JobSpec(
                    job_type=JobType.EMBED_MEMORY,
                    payload={"memory_id": str(memory.id)},
                    dedupe_key=f"embed:{memory.id}",
                ),
            )
        enqueued += job_id is not None
    return enqueued


async def gather_stats(session_factory: async_sessionmaker[AsyncSession]) -> Stats:
    async with session_factory() as session:
        memories = (
            await session.execute(select(func.count()).select_from(models.Memory))
        ).scalar_one()
        current = (
            await session.execute(
                select(func.count())
                .select_from(models.Memory)
                .where(models.Memory.is_current.is_(True), models.Memory.deleted_at.is_(None))
            )
        ).scalar_one()
        chunks = (
            await session.execute(select(func.count()).select_from(models.MemoryChunk))
        ).scalar_one()
        embedded = (
            await session.execute(
                select(func.count())
                .select_from(models.MemoryChunk)
                .where(models.MemoryChunk.embedding.is_not(None))
            )
        ).scalar_one()
        cache_entries = (
            await session.execute(
                select(func.count()).select_from(models.EmbeddingCacheEntry)
            )
        ).scalar_one()
        by_model: dict[str, int] = {
            str(model or ""): count
            for model, count in (
                await session.execute(
                    select(models.MemoryChunk.embedding_model, func.count())
                    .where(models.MemoryChunk.embedding.is_not(None))
                    .group_by(models.MemoryChunk.embedding_model)
                )
            ).all()
        }

        # Merged-away entities excluded, which is not a refinement but the
        # predicate `models.Entity` requires of every read that counts them:
        # a loser survives only so its merge can be undone, and counting it
        # would report the duplicate M3.2 already resolved. Same `WHERE` as
        # `entity_stats.gather_entity_stats`, so the two agree.
        entities = (
            await session.execute(
                select(func.count())
                .select_from(models.Entity)
                .where(models.Entity.merged_into_id.is_(None))
            )
        ).scalar_one()
        relationships = (
            await session.execute(_distinct_edges())
        ).scalar_one()

    return Stats(
        memories=memories,
        current_memories=current,
        chunks=chunks,
        embedded_chunks=embedded,
        cache_entries=cache_entries,
        models=by_model,
        entities=entities,
        relationships=relationships,
    )


def _distinct_edges() -> Select[tuple[int]]:
    """How many relationships the corpus asserts, counted as edges not rows.

    **The row count is the wrong number and would overstate this by a lot.**
    `models.EntityRelationship` stores one row per *assertion* — the same claim
    made in five chunks is five rows, deliberately, because M3.5 weights edges
    by how often the corpus repeats them. What a reader means by "relationships"
    is the distinct claims, so this counts distinct `(subject, predicate,
    object)` after resolving merged endpoints to their winners.

    The predicates mirror `graph_projection._relationship_edges`, so this agrees
    with what the graph holds: merges resolved one hop with `coalesce`,
    self-loops dropped because after a merge "X uses X" is two names for one
    thing, and superseded or deleted memories excluded because their claims are
    no longer part of the current corpus.

    One deliberate difference: that function additionally narrows to entities it
    is writing nodes for, so the graph can hold slightly fewer edges than this
    reports. Reproducing that here would mean building the projection to count
    it.
    """
    subject = models.Entity.__table__.alias("subject")
    obj = models.Entity.__table__.alias("object")
    subject_id = func.coalesce(subject.c.merged_into_id, subject.c.id)
    object_id = func.coalesce(obj.c.merged_into_id, obj.c.id)

    return select(
        func.count(
            func.distinct(
                tuple_(subject_id, models.EntityRelationship.predicate, object_id)
            )
        )
    ).select_from(
        models.EntityRelationship.__table__.join(
            subject, subject.c.id == models.EntityRelationship.subject_id
        )
        .join(obj, obj.c.id == models.EntityRelationship.object_id)
        .join(
            models.Memory.__table__,
            models.Memory.id == models.EntityRelationship.memory_id,
        )
    ).where(
        models.Memory.is_current.is_(True),
        models.Memory.deleted_at.is_(None),
        subject_id != object_id,
    )


@dataclass(frozen=True, slots=True)
class ExtractionTarget:
    memory_id: UUID
    external_key: str
    chunks: int


async def find_extraction_targets(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    extractor_version: str,
    source: str | None = None,
    limit: int | None = None,
) -> list[ExtractionTarget]:
    """Current, undeleted memories with chunks and no mentions at this version.

    The version predicate is the whole idempotency story, and it is a `NOT
    EXISTS` rather than a join so that a memory with *some* mentions at the
    current version counts as done. Extraction writes all of a memory's mentions
    in one transaction, so partial state is not reachable — and treating it as
    done anyway is what makes re-running the command free rather than a second
    full spend.

    Ordered by `external_key` so a `--limit` run is reproducible: the same
    twenty memories every time, rather than whatever the planner returned.
    """
    current = (
        select(
            models.Memory.id,
            models.Memory.external_key,
            func.count(models.MemoryChunk.id).label("chunks"),
        )
        .join(models.MemoryChunk, models.MemoryChunk.memory_id == models.Memory.id)
        .where(
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
            # The marker, not a probe for mentions. A memory that legitimately
            # contains no entities writes no mention rows, so a probe never
            # marks it done and every run pays to extract it again.
            models.Memory.entity_extractor_version.is_distinct_from(extractor_version),
        )
        .group_by(models.Memory.id, models.Memory.external_key)
        .order_by(models.Memory.external_key)
    )

    if source is not None:
        current = current.join(
            models.Source, models.Source.id == models.Memory.source_id
        ).where(models.Source.name == source)
    if limit is not None:
        current = current.limit(limit)

    async with session_factory() as session:
        return [
            ExtractionTarget(memory_id=row[0], external_key=row[1], chunks=row[2])
            for row in await session.execute(current)
        ]


async def find_relationship_targets(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    extractor_version: str,
    source: str | None = None,
    limit: int | None = None,
) -> list[ExtractionTarget]:
    """Memories with resolved entity mentions and no relationships at this version.

    The join to `entity_mentions` is the useful half: a memory nobody has
    extracted entities from has nothing to relate, and offering it to the model
    would spend a request to be told so. On a rate-limited tier that filter is
    the difference between finishing a corpus and stopping partway.
    """
    current = (
        select(
            models.Memory.id,
            models.Memory.external_key,
            func.count(func.distinct(models.EntityMention.chunk_id)).label("chunks"),
        )
        .join(
            models.EntityMention, models.EntityMention.memory_id == models.Memory.id
        )
        .where(
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
            models.Memory.relationship_extractor_version.is_distinct_from(
                extractor_version
            ),
        )
        .group_by(models.Memory.id, models.Memory.external_key)
        .order_by(models.Memory.external_key)
    )

    if source is not None:
        current = current.join(
            models.Source, models.Source.id == models.Memory.source_id
        ).where(models.Source.name == source)
    if limit is not None:
        current = current.limit(limit)

    async with session_factory() as session:
        return [
            ExtractionTarget(memory_id=row[0], external_key=row[1], chunks=row[2])
            for row in await session.execute(current)
        ]
