"""What Postgres implies the graph should contain.

One definition, four callers: the full rebuild, the incremental sync, the
divergence check, and a replay that has just discarded everything derived. That
is the whole reason this module exists, and it is the same argument
`application/projection.py` makes about memories — two implementations of "what
the graph should look like" would agree until somebody changed one of them, and
the disagreement would be invisible because both would keep producing plausible
nodes.

The shape of the answer is deliberately *data* rather than a sequence of writes.
`read` returns a `GraphProjection`; `write` puts one into a store; `verify`
hashes one and compares it against what the store actually holds. A projector
that wrote as it read could not be verified against anything except itself.

## What is projected, and what is not

    (:Source)<-[:FROM_SOURCE]-(:Memory)-[:MENTIONS]->(:Entity)-[:RELATES_TO]->(:Entity)

* **Every current, undeleted memory**, not only the ones extraction has reached.
  M3.1 and M3.2 projected a `Memory` node only when it had mentions, which made
  the graph's memory count a fact about how far extraction had got rather than
  about the corpus, and left `verify` unable to say whether a missing node was a
  defect or a memory nobody had extracted yet.
* **Its source**, which M3.0 declared a label and an edge type for and nothing
  ever wrote. `(source_name, external_key)` is the durable identity of an item
  everywhere else in this system, and the graph knew half of it.
* **Active entities with at least one mention.** A merged-away entity is not
  projected at all: its mentions belong to the winner now, and a node for it
  would be a path through a name the corpus has stopped using.
* **No content.** `Memory` carries identity, kind and date; the text stays in
  Postgres. See `ports.MemoryNode`.

## Merges are followed on read, not repointed on write

`entity_relationships` rows keep pointing at whichever entity the extractor
named, including one that has since been merged away, and this module resolves
those endpoints to the surviving entity as it reads.

The alternative — repointing the rows when a merge is applied — is worse in two
ways. It destroys the evidence: the row records what the model actually claimed
about the name it actually saw, which is the same reason `entities.name` keeps
the first surface form rather than being overwritten. And it makes `unmerge`
owe a second restoration, which would need a second `moved_*_ids` column to be
exact rather than approximate.

Resolving on read costs one join and makes the graph correct by construction
after any merge, which is what M3.3 shipped without: a relationship whose
subject had been merged away projected an `Entity` node carrying an id and
nothing else — because `link` merges its endpoints — and every traversal through
it walked into a node with no name.

A relationship whose two endpoints resolve to the *same* entity is dropped. "X
uses X" is not a claim the corpus made; it is two names for one thing, and the
merge is what revealed that.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from memoryos.adapters.db import models
from memoryos.application.ports import (
    EntityNode,
    GraphEdge,
    GraphNode,
    GraphStore,
    MemoryNode,
    SourceNode,
)
from memoryos.domain.values import EdgeType, GraphLabel, MemoryKind

logger = structlog.get_logger(__name__)

# Every entity-to-entity claim becomes a `RELATES_TO` edge carrying its predicate
# as a property, rather than one Cypher relationship type per predicate. M3.3
# decided this; it lives here rather than beside the extractor because the shape
# of the projection is now one module's business.
#
# M3.0 declared three relationship types and traversals are written against them;
# promoting seven predicates to seven types would mean every existing pattern has
# to enumerate them, and `[:RELATES_TO {predicate: 'uses'}]` filters exactly as
# precisely. `MENTIONS` deliberately stays what M3.1 made it — memory to entity —
# and is not reused for a `mentions` predicate between two entities, because one
# edge type carrying two different meanings is a traversal that cannot tell them
# apart.
ENTITY_EDGE_TYPE = EdgeType.RELATES_TO


@dataclass(frozen=True, slots=True)
class GraphProjection:
    """The nodes and edges Postgres says the graph should hold.

    Ordered tuples rather than sets, and sorted by identity as they are read, so
    that two reads of an unchanged corpus produce byte-identical content hashes.
    An unordered structure would make the hash depend on whatever order Postgres
    returned the rows in, and a verification that fails intermittently is worse
    than none.
    """

    sources: tuple[SourceNode, ...] = ()
    memories: tuple[MemoryNode, ...] = ()
    entities: tuple[EntityNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        """Node and edge totals, by label and by relationship type."""
        counts = {
            "Source": len(self.sources),
            "Memory": len(self.memories),
            "Entity": len(self.entities),
        }
        for edge_type in EdgeType:
            counts[edge_type.value] = sum(
                1 for edge in self.edges if edge.type is edge_type
            )
        return counts

    @property
    def nodes(self) -> int:
        return len(self.sources) + len(self.memories) + len(self.entities)


# --------------------------------------------------------------------------
# Reading the projection out of Postgres
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scope:
    """Which part of the projection to read.

    An empty scope means the whole corpus, which is what the rebuild wants. A
    populated one names memories and entities, which is what the sync wants — and
    the two go through the same queries with a `WHERE` added, so a scoped
    projection cannot disagree with the whole one about what a node looks like.
    """

    memory_ids: frozenset[UUID] = field(default_factory=frozenset)
    entity_ids: frozenset[UUID] = field(default_factory=frozenset)

    @property
    def is_whole_corpus(self) -> bool:
        return not self.memory_ids and not self.entity_ids

    def describe(self) -> str:
        if self.is_whole_corpus:
            return "whole corpus"
        return f"{len(self.memory_ids)} memories, {len(self.entity_ids)} entities"


async def read(
    session_factory: async_sessionmaker[AsyncSession], scope: Scope | None = None
) -> GraphProjection:
    """The projection the current contents of Postgres imply."""
    resolved = scope or Scope()
    async with session_factory() as session:
        memories, source_of = await _memories(session, resolved)
        sources = await _sources(session, set(source_of.values()))
        entities, entity_ids = await _entities(session, resolved)
        mentions = await _mention_edges(session, resolved)
        relates = await _relationship_edges(session, entity_ids or None)

    return GraphProjection(
        sources=sources,
        memories=memories,
        entities=entities,
        edges=_from_source_edges(source_of) + mentions + relates,
    )


async def _memories(
    session: AsyncSession, scope: Scope
) -> tuple[tuple[MemoryNode, ...], dict[UUID, UUID]]:
    """Current, undeleted memories in scope, and the sources they came from.

    Superseded versions are excluded, and so is the tombstone of a deleted item.
    Both are the same rule as every other read of the corpus: the graph answers
    questions about what is there now, and a node for version 3 of a file that is
    on version 7 would put four copies of one document into every traversal.
    """
    stmt = select(
        models.Memory.id,
        models.Memory.source_id,
        models.Memory.external_key,
        models.Memory.kind,
        models.Memory.occurred_at,
    ).where(
        models.Memory.is_current.is_(True),
        models.Memory.deleted_at.is_(None),
    )
    if not scope.is_whole_corpus:
        stmt = stmt.where(models.Memory.id.in_(scope.memory_ids))
    stmt = stmt.order_by(models.Memory.id)

    rows = list(await session.execute(stmt))
    memories = tuple(
        MemoryNode(
            memory_id=row[0],
            external_key=row[2],
            kind=MemoryKind(row[3]),
            occurred_at=row[4],
        )
        for row in rows
    )
    return memories, {row[0]: row[1] for row in rows}


async def _sources(
    session: AsyncSession, source_ids: set[UUID]
) -> tuple[SourceNode, ...]:
    if not source_ids:
        return ()
    stmt = (
        select(models.Source.id, models.Source.name, models.Source.kind)
        .where(models.Source.id.in_(source_ids))
        .order_by(models.Source.id)
    )
    return tuple(
        SourceNode(source_id=row[0], name=row[1], kind=row[2])
        for row in await session.execute(stmt)
    )


def _from_source_edges(source_of: dict[UUID, UUID]) -> tuple[GraphEdge, ...]:
    """One edge per memory, built from what `_memories` already read.

    The source id is deliberately not a field on `MemoryNode` — that node carries
    identity and nothing that duplicates a join — so the pairing is carried out of
    the query rather than re-fetched. Direction is from the memory to its source,
    matching `MENTIONS`, which also points away from the thing doing the
    referring.
    """
    return tuple(
        GraphEdge(
            type=EdgeType.FROM_SOURCE,
            start=GraphNode(GraphLabel.MEMORY, str(memory_id)),
            end=GraphNode(GraphLabel.SOURCE, str(source_id)),
        )
        for memory_id, source_id in sorted(source_of.items(), key=lambda pair: pair[0])
    )


async def _entities(
    session: AsyncSession, scope: Scope
) -> tuple[tuple[EntityNode, ...], set[UUID]]:
    """Active entities that something in scope mentions.

    The mention join is not a filter for tidiness. An entity with no mentions is
    either a row a merge left behind or one whose memory has been deleted, and in
    both cases projecting it would put a node in the graph that no path can
    reach — indistinguishable, to `verify`, from a node somebody wrote by hand.
    """
    stmt = (
        select(
            models.Entity.id,
            models.Entity.name,
            models.Entity.canonical_name,
            models.Entity.type,
            models.Entity.confidence,
        )
        .join(models.EntityMention, models.EntityMention.entity_id == models.Entity.id)
        .join(models.Memory, models.Memory.id == models.EntityMention.memory_id)
        .where(
            models.Entity.merged_into_id.is_(None),
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
        )
        .group_by(
            models.Entity.id,
            models.Entity.name,
            models.Entity.canonical_name,
            models.Entity.type,
            models.Entity.confidence,
        )
        .order_by(models.Entity.id)
    )
    if not scope.is_whole_corpus:
        stmt = stmt.where(_in_scope(scope))

    rows = list(await session.execute(stmt))
    entities = tuple(
        EntityNode(
            entity_id=row[0],
            name=row[1],
            canonical_name=row[2],
            type=row[3],
            confidence=row[4] if row[4] is not None else 0.0,
        )
        for row in rows
    )
    return entities, {row[0] for row in rows}


async def _mention_edges(session: AsyncSession, scope: Scope) -> tuple[GraphEdge, ...]:
    """One `MENTIONS` edge per (memory, entity) pair, not per mention row.

    Several mentions of one entity in one memory are one edge, because a
    relationship in a graph has no multiplicity — `MERGE` collapses them
    whatever this returns. So the count is aggregated here rather than
    discovered later: `mentions` on the edge is how many times the corpus named
    the entity in that memory, and `chunk_id` is the first chunk that did, which
    is what gives a traversal somewhere to cite.
    """
    stmt = (
        select(
            models.EntityMention.memory_id,
            models.EntityMention.entity_id,
            func.count().label("mentions"),
            func.min(models.MemoryChunk.ordinal).label("ordinal"),
        )
        .join(models.Entity, models.Entity.id == models.EntityMention.entity_id)
        .join(models.Memory, models.Memory.id == models.EntityMention.memory_id)
        .join(
            models.MemoryChunk, models.MemoryChunk.id == models.EntityMention.chunk_id
        )
        .where(
            models.Entity.merged_into_id.is_(None),
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
        )
        .group_by(models.EntityMention.memory_id, models.EntityMention.entity_id)
        .order_by(models.EntityMention.memory_id, models.EntityMention.entity_id)
    )
    if not scope.is_whole_corpus:
        stmt = stmt.where(_in_scope(scope))

    return tuple(
        GraphEdge(
            type=EdgeType.MENTIONS,
            start=GraphNode(GraphLabel.MEMORY, str(memory_id)),
            end=GraphNode(GraphLabel.ENTITY, str(entity_id)),
            # The chunk is named by *ordinal*, not by id. A chunk id is minted per
            # write and a rebuild produces a different one, so an id here would
            # make every replay report the whole projection as changed; the
            # ordinal is the chunk's natural key and survives.
            properties={"mentions": int(mentions), "chunk_ordinal": int(ordinal)},
        )
        for memory_id, entity_id, mentions, ordinal in await session.execute(stmt)
    )


async def _relationship_edges(
    session: AsyncSession, entity_ids: set[UUID] | None
) -> tuple[GraphEdge, ...]:
    """`RELATES_TO` edges, with merged-away endpoints resolved to their winners.

    Aggregated over the whole corpus rather than per memory: `assertion_count` is
    how many chunks anywhere made this claim, and M3.3's own model docstring says
    one assertion is a claim while five are a pattern. A count of what a single
    memory asserted would under-report every repeated claim.

    Two entity aliases are joined in so the resolution happens in SQL. Doing it
    in Python would mean reading every relationship row and then discovering
    which endpoints moved, which is the same work with a second place to be
    wrong.
    """
    subject = models.Entity.__table__.alias("subject")
    obj = models.Entity.__table__.alias("object")
    # `coalesce(merged_into_id, id)` *is* the resolution: one hop, because the
    # resolver never merges a loser, so a chain of length two is data corruption
    # rather than a state to support. `_follow_merge` in extraction.py walks
    # deeper because it is writing; this only has to read what is there.
    subject_id = func.coalesce(subject.c.merged_into_id, subject.c.id)
    object_id = func.coalesce(obj.c.merged_into_id, obj.c.id)

    stmt = (
        select(
            subject_id.label("subject_id"),
            object_id.label("object_id"),
            models.EntityRelationship.predicate,
            func.count().label("assertions"),
            func.max(models.EntityRelationship.confidence).label("confidence"),
        )
        .join(subject, subject.c.id == models.EntityRelationship.subject_id)
        .join(obj, obj.c.id == models.EntityRelationship.object_id)
        .join(models.Memory, models.Memory.id == models.EntityRelationship.memory_id)
        .where(
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
            # Dropped rather than projected as a self-loop. See the module
            # docstring: after a merge, "X uses X" is two names for one thing.
            subject_id != object_id,
        )
        .group_by(subject_id, object_id, models.EntityRelationship.predicate)
        .order_by(subject_id, object_id, models.EntityRelationship.predicate)
    )
    if entity_ids is not None:
        # Either endpoint, so an edge is projected whenever the sync is
        # responsible for one of the entities it connects.
        stmt = stmt.where(subject_id.in_(entity_ids) | object_id.in_(entity_ids))

    return tuple(
        GraphEdge(
            type=ENTITY_EDGE_TYPE,
            start=GraphNode(GraphLabel.ENTITY, str(row[0])),
            end=GraphNode(GraphLabel.ENTITY, str(row[1])),
            properties={
                "predicate": row[2],
                "assertion_count": int(row[3]),
                "confidence": float(row[4]) if row[4] is not None else 0.0,
            },
        )
        for row in await session.execute(stmt)
    )


def _in_scope(scope: Scope) -> ColumnElement[bool]:
    """The scope predicate shared by the entity and mention queries.

    Either side of the mention: a memory in scope brings its entities, and an
    entity in scope brings the memories that mention it. That is what makes a
    scoped projection a *neighbourhood* rather than a slice — see
    `graph_sync.expand`, which is what widens a payload into a closed one.
    """
    return models.EntityMention.memory_id.in_(scope.memory_ids) | (
        models.EntityMention.entity_id.in_(scope.entity_ids)
    )


# --------------------------------------------------------------------------
# Writing it to the graph
# --------------------------------------------------------------------------


async def write(graph: GraphStore, projection: GraphProjection) -> None:
    """Upsert every node, then every edge. Idempotent, by `MERGE`.

    Nodes first, and not only for tidiness: `link` merges its endpoints, so an
    edge written before its endpoint's own upsert creates a node carrying an id
    and nothing else. That is recoverable — the upsert fills it in — but only if
    the upsert happens, and ordering it first is cheaper than a diagnostic that
    finds out it did not.
    """
    for source in projection.sources:
        await graph.upsert_source(source)
    for memory in projection.memories:
        await graph.upsert_memory(memory)
    for entity in projection.entities:
        await graph.upsert_entity(entity)
    for edge in projection.edges:
        await graph.link(edge)


async def rebuild(
    session_factory: async_sessionmaker[AsyncSession], graph: GraphStore
) -> GraphProjection:
    """Throw the whole projection away and build it again from Postgres.

    Cleared rather than upserted over, and that is the difference between a
    rebuild and a repair. Upserting converges on *containing* everything Postgres
    implies; it never removes a node Postgres has stopped implying — a merged-away
    entity, a deleted memory, an edge whose relationship row is gone. Those are
    precisely the divergences worth catching, so the operation that exists to end
    divergence has to be able to remove.

    Returns what it wrote, so the caller can report counts without reading the
    graph back.
    """
    projection = await read(session_factory)
    await graph.clear()
    await write(graph, projection)
    logger.info("graph.rebuilt", **projection.counts)
    return projection


# --------------------------------------------------------------------------
# Content hashing
# --------------------------------------------------------------------------


def content_hash(projection: GraphProjection) -> str:
    """A digest of the whole projection, for comparing two of them in one line.

    Every property of every node and edge goes in, which is the point: a count
    check passes while a node carries the wrong name, and M1.6.1 is the standing
    proof that count checks pass through real defects. Floats are formatted with
    `repr` so a confidence of 0.9 and one of 0.9000000000000001 do not compare
    equal — a projection is derived data and "close enough" is how a rebuild that
    changed something gets reported as clean.
    """
    return hashlib.blake2b(
        json.dumps(_canonical(projection), sort_keys=True, separators=(",", ":")).encode(),
        digest_size=16,
    ).hexdigest()


def hashes_by_type(projection: GraphProjection) -> dict[str, str]:
    """One digest per node label and edge type.

    Reported alongside the whole-projection hash because the whole one only says
    *whether* something differs. Per type, it says where to look, which is the
    difference between a check that fails and a check that diagnoses.
    """
    canonical = _canonical(projection)
    digests: dict[str, str] = {}
    for label in ("Source", "Memory", "Entity"):
        digests[label] = _digest(canonical[label.lower() + "s"])
    for edge_type in EdgeType:
        digests[edge_type.value] = _digest(
            [edge for edge in canonical["edges"] if edge[0] == edge_type.value]
        )
    return digests


CanonicalRows = dict[str, list[list[str | None]]]


def _canonical(projection: GraphProjection) -> CanonicalRows:
    """The projection as sorted, JSON-safe primitives."""
    return {
        "sources": sorted(
            [str(node.source_id), node.name, node.kind] for node in projection.sources
        ),
        "memories": sorted(
            [
                str(node.memory_id),
                node.external_key,
                node.kind.value,
                # `str`, so a `datetime` and the same instant read back from the
                # driver compare as text rather than across two temporal types.
                None if node.occurred_at is None else str(node.occurred_at),
            ]
            for node in projection.memories
        ),
        "entities": sorted(
            [
                str(node.entity_id),
                node.name,
                node.canonical_name,
                node.type,
                repr(round(float(node.confidence), 6)),
            ]
            for node in projection.entities
        ),
        "edges": sorted(
            [
                edge.type.value,
                edge.start.key,
                edge.end.key,
                json.dumps(_canonical_properties(edge.properties), sort_keys=True),
            ]
            for edge in projection.edges
        ),
    }


def _canonical_properties(properties: Mapping[str, Any]) -> dict[str, object]:
    """Edge properties, with floats pinned to a fixed precision.

    Neo4j stores a float as a double and the driver returns it as one, so a
    confidence that went in as a 32-bit `REAL` from Postgres comes back with
    digits Postgres never had. Rounding both sides to six places is what makes
    the comparison a statement about the data rather than about float widths.
    """
    return {
        key: (repr(round(float(value), 6)) if isinstance(value, float) else value)
        for key, value in sorted(properties.items())
    }


def _digest(rows: list[list[str | None]]) -> str:
    return hashlib.blake2b(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode(), digest_size=8
    ).hexdigest()
