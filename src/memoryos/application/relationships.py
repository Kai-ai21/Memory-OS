"""Extract relationships for a memory, store them, and project typed edges.

The candidates handed to the model are the memory's *already-resolved* entities,
and that ordering is load-bearing in both directions. Extracting before
resolution would attach edges to entities that are about to be merged away, and
every one would need re-pointing by hand. Offering the model a fixed list rather
than letting it name entities is what makes an edge to a non-existent entity
structurally impossible rather than merely unlikely — an edge to a fabricated
node looks exactly like a real one until somebody follows it.

Transaction boundary as everywhere else in Phase 3: Postgres commits, then the
projection follows. A crash between them leaves a correct corpus and a graph
that is behind, which the next run repairs because every graph write is a MERGE.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum, auto
from uuid import UUID

import structlog
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import (
    EntityExtractor,
    EntityRef,
    ExtractedRelationship,
    GraphEdge,
    GraphNode,
    GraphStore,
)
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import PermanentError
from memoryos.domain.values import EdgeType, EntityType, GraphLabel

logger = structlog.get_logger(__name__)


class RelationshipOutcome(StrEnum):
    SKIPPED = auto()
    EXTRACTED = auto()
    DELETED = auto()
    EMPTY = auto()


@dataclass(slots=True)
class RelationshipReport:
    outcome: RelationshipOutcome
    chunks_considered: int = 0
    relationships: int = 0
    projected: int = 0
    duration_ms: int = 0

    def as_dict(self) -> dict[str, str | int]:
        return {
            "outcome": self.outcome.value,
            "chunks_considered": self.chunks_considered,
            "relationships": self.relationships,
            "projected": self.projected,
            "duration_ms": self.duration_ms,
        }


class ExtractRelationships:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        extractor: EntityExtractor,
        graph: GraphStore | None = None,
        *,
        version: str | None = None,
    ) -> None:
        self._sessions = session_factory
        self._extractor = extractor
        self._graph = graph
        # The relationship extractor's own version, distinct from the entity
        # one so the two prompts can be improved independently.
        fallback = getattr(extractor, "relationship_version", None)
        self._version = version or str(fallback or extractor.version)

    @property
    def version(self) -> str:
        return self._version

    async def __call__(self, memory_id: UUID) -> RelationshipReport:
        started = time.monotonic()
        log = logger.bind(memory_id=str(memory_id), extractor=self._version)

        memory = await self._load(memory_id)
        if memory is None:
            raise PermanentError(f"no such memory: {memory_id}")
        if memory.deleted_at is not None:
            return _finish(RelationshipReport(RelationshipOutcome.DELETED), started)
        if memory.relationship_extractor_version == self._version:
            log.info("relationships.skipped_current")
            return _finish(RelationshipReport(RelationshipOutcome.SKIPPED), started)

        # Only chunks that mention at least two entities. A chunk with one
        # entity has nothing to relate it to, and asking is a request spent to
        # be told so — which on a rate-limited tier is the difference between
        # extracting a corpus and running out of quota partway through.
        per_chunk = await self._entities_by_chunk(memory_id)
        eligible = {
            chunk_id: refs for chunk_id, refs in per_chunk.items() if len(refs) >= 2
        }
        if not eligible:
            await self._mark(memory_id)
            log.info("relationships.no_eligible_chunks")
            return _finish(RelationshipReport(RelationshipOutcome.EMPTY), started)

        texts = await self._chunk_texts(list(eligible))
        found: dict[UUID, list[ExtractedRelationship]] = {}
        for chunk_id, refs in eligible.items():
            text = texts.get(chunk_id)
            if text is None:
                continue
            found[chunk_id] = await self._extractor.extract_relationships(text, refs)

        report = await self._store(memory_id, found)
        report.chunks_considered = len(eligible)
        report.projected = await self._project(memory_id)
        log.info("relationships.stored", **report.as_dict())
        return _finish(report, started)

    async def _load(self, memory_id: UUID) -> models.Memory | None:
        async with self._sessions() as session:
            return await session.get(models.Memory, memory_id)

    async def _entities_by_chunk(
        self, memory_id: UUID
    ) -> dict[UUID, list[EntityRef]]:
        """The resolved entities each chunk mentions.

        `merged_into_id IS NULL` is the whole reason this runs after M3.2: a
        relationship attached to an entity that has been merged away points at a
        node no traversal reaches, and the edge would have to be re-pointed by
        hand later.
        """
        stmt = (
            select(
                models.EntityMention.chunk_id,
                models.Entity.id,
                models.Entity.name,
                models.Entity.type,
            )
            .join(models.Entity, models.Entity.id == models.EntityMention.entity_id)
            .where(
                models.EntityMention.memory_id == memory_id,
                models.Entity.merged_into_id.is_(None),
            )
        )
        grouped: dict[UUID, dict[UUID, EntityRef]] = defaultdict(dict)
        async with self._sessions() as session:
            for chunk_id, entity_id, name, kind in await session.execute(stmt):
                # Deduplicated per chunk: an entity named three times in one
                # chunk is one candidate, and offering it three times would
                # spend prompt tokens inviting the model to relate it to itself.
                grouped[chunk_id][entity_id] = EntityRef(
                    entity_id=entity_id, name=name, type=EntityType(kind)
                )
        return {chunk_id: list(refs.values()) for chunk_id, refs in grouped.items()}

    async def _chunk_texts(self, chunk_ids: list[UUID]) -> dict[UUID, str]:
        async with self._sessions() as session:
            return {
                row[0]: row[1]
                for row in await session.execute(
                    select(models.MemoryChunk.id, models.MemoryChunk.content).where(
                        models.MemoryChunk.id.in_(chunk_ids)
                    )
                )
            }

    async def _store(
        self, memory_id: UUID, found: dict[UUID, list[ExtractedRelationship]]
    ) -> RelationshipReport:
        """Replace this memory's relationships, in one transaction.

        Every row carries the chunk that asserted it, so the same claim made in
        five chunks is five rows. That is the evidence rather than duplication:
        M3.5 weights edges by assertion count, and collapsing here would discard
        the only signal separating a claim from a pattern.
        """
        stored = 0
        async with self._sessions.begin() as session:
            await session.execute(
                delete(models.EntityRelationship).where(
                    models.EntityRelationship.memory_id == memory_id
                )
            )
            for chunk_id, claims in found.items():
                for claim in claims:
                    result = await session.execute(
                        insert(models.EntityRelationship)
                        .values(
                            id=new_id(),
                            subject_id=claim.subject_id,
                            object_id=claim.object_id,
                            predicate=claim.predicate.value,
                            confidence=claim.confidence,
                            memory_id=memory_id,
                            chunk_id=chunk_id,
                            char_start=claim.char_start,
                            char_end=claim.char_end,
                            extractor_version=self._version,
                        )
                        .on_conflict_do_nothing(
                            constraint="uq_entity_relationships_assertion"
                        )
                        .returning(models.EntityRelationship.id)
                    )
                    stored += int(result.scalar_one_or_none() is not None)

            await session.execute(
                update(models.Memory)
                .where(models.Memory.id == memory_id)
                .values(relationship_extractor_version=self._version)
            )

        return RelationshipReport(
            outcome=RelationshipOutcome.EXTRACTED, relationships=stored
        )

    async def _mark(self, memory_id: UUID) -> None:
        """Record the attempt for a memory with nothing to extract.

        The same lesson as M3.1's entity marker: "produced no rows" and "was
        never tried" have to be distinguishable, or every run pays to rediscover
        that a memory has no relatable chunks.
        """
        async with self._sessions.begin() as session:
            await session.execute(
                update(models.Memory)
                .where(models.Memory.id == memory_id)
                .values(relationship_extractor_version=self._version)
            )

    async def _project(self, memory_id: UUID) -> int:
        """Mirror this memory's relationships into Neo4j as typed edges.

        Read back from Postgres rather than reusing the in-memory claims,
        because `assertion_count` is a property of the whole corpus rather than
        of this memory: the same pair may have been asserted by chunks belonging
        to memories extracted weeks apart, and an edge weighted by only what
        this run saw would under-report every repeated claim.
        """
        if self._graph is None:
            return 0
        try:
            edges = await self._edges_for(memory_id)
            for edge in edges:
                await self._graph.link(edge)
        except Exception as exc:
            logger.warning(
                "relationships.projection_failed",
                memory_id=str(memory_id),
                error=str(exc),
            )
            return 0
        return len(edges)

    async def _edges_for(self, memory_id: UUID) -> list[GraphEdge]:
        """Corpus-wide edge weights for the pairs this memory touched."""
        pairs = (
            select(
                models.EntityRelationship.subject_id,
                models.EntityRelationship.object_id,
                models.EntityRelationship.predicate,
            )
            .where(models.EntityRelationship.memory_id == memory_id)
            .distinct()
            .subquery()
        )
        stmt = (
            select(
                models.EntityRelationship.subject_id,
                models.EntityRelationship.object_id,
                models.EntityRelationship.predicate,
                func.count().label("assertions"),
                func.max(models.EntityRelationship.confidence).label("confidence"),
            )
            .join(
                pairs,
                (models.EntityRelationship.subject_id == pairs.c.subject_id)
                & (models.EntityRelationship.object_id == pairs.c.object_id)
                & (models.EntityRelationship.predicate == pairs.c.predicate),
            )
            .group_by(
                models.EntityRelationship.subject_id,
                models.EntityRelationship.object_id,
                models.EntityRelationship.predicate,
            )
        )
        async with self._sessions() as session:
            rows = list(await session.execute(stmt))

        return [
            GraphEdge(
                type=ENTITY_EDGE_TYPE,
                start=GraphNode(GraphLabel.ENTITY, str(subject_id)),
                end=GraphNode(GraphLabel.ENTITY, str(object_id)),
                properties={
                    "predicate": predicate,
                    "assertion_count": assertions,
                    "confidence": float(confidence) if confidence is not None else 0.0,
                },
            )
            for subject_id, object_id, predicate, assertions, confidence in rows
        ]


# Every entity-to-entity claim becomes a `RELATES_TO` edge carrying its
# predicate as a property, rather than one Cypher relationship type per
# predicate.
#
# M3.0 declared three relationship types and traversals are written against
# them; promoting seven predicates to seven types would mean every existing
# pattern has to enumerate them, and `[:RELATES_TO {predicate: 'uses'}]` filters
# exactly as precisely. `MENTIONS` deliberately stays what M3.1 made it —
# memory to entity — and is not reused for a `mentions` predicate between two
# entities, because one edge type carrying two different meanings is a
# traversal that cannot tell them apart.
ENTITY_EDGE_TYPE = EdgeType.RELATES_TO


def _finish(report: RelationshipReport, started: float) -> RelationshipReport:
    report.duration_ms = int((time.monotonic() - started) * 1000)
    return report
