"""Keeping the projection current without rebuilding all of it.

**Nothing writes to Neo4j from a use case any more.** Extraction and resolution
enqueue a `SYNC_GRAPH` job naming what changed, and this is the only thing that
projects. That is not tidiness: a use case writing to both stores would have made
the graph a second source of truth in everything but name, and its writes would
survive exactly until the next rebuild — which is the worst failure available,
because it is silent and time-delayed.

## Why incremental at all, when the rebuild is cheap

It is cheap *now*. A rebuild reads five tables and writes 500 nodes, which is
under a second on this corpus, and the honest thing to say is that at this scale
`graph rebuild` after every change would work. It stops working for a reason that
has nothing to do with speed: a rebuild `clear()`s, and the clear is a window in
which the graph answers no questions at all. Every retrieval that reaches the
graph during it degrades, and the window grows linearly with the corpus while the
change that triggered it stays one memory.

So the sync exists to make the *cost of a change* proportional to the change
rather than to the corpus. `graph rebuild` remains the answer to divergence.

## Delete, then project. Never patch.

A scoped sync is a rebuild of a neighbourhood, and it works exactly as the full
rebuild does: prune what the scope covers, then write what Postgres says that
scope should be. An additive sync — upsert what is there now — converges on the
graph *containing* everything Postgres implies and never removes anything it has
stopped implying. A mention deleted by a re-extraction, an entity merged away, a
relationship whose row is gone: all of them survive an additive sync forever, and
none of them is visible in a node count.

Which makes idempotence structural rather than tested-for. Running the same sync
twice prunes the same nodes and writes the same nodes, so the second run cannot
differ from the first — and the test that asserts it is checking the *scope*
arithmetic below, not the arithmetic of `MERGE`.

## The scope has to be closed before anything is pruned

This is the whole difficulty, and it is all in `expand`. Pruning is detaching, so
removing an entity node takes every `MENTIONS` edge into it — including edges from
memories the payload never named. Those memories then have to be re-projected, or
the sync has quietly deleted a relationship it was not asked to touch.

So a payload is widened until it is closed under one step of the mention
relation, in *both* directions and against *both* stores:

* a memory in scope brings the entities it mentions;
* an entity in scope brings every memory that mentions it — according to
  Postgres, which knows what is true now, **and** according to the graph, which
  knows what it currently claims. The second is not redundant. After a merge,
  Postgres no longer associates the loser with any memory at all; the graph still
  does, and those stale edges are exactly what the sync exists to remove.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application import graph_projection
from memoryos.application.graph_projection import GraphProjection, Scope
from memoryos.application.ports import GraphStore, JobQueue
from memoryos.domain.jobs import JobSpec, JobType, PermanentError

logger = structlog.get_logger(__name__)

# How many ids one job's payload may name before it is worth rebuilding instead.
#
# Not a memory limit — the payload is JSONB and would hold thousands. It is the
# point where "prune this neighbourhood and re-project it" stops being cheaper
# than "prune everything and re-project everything", because the neighbourhood
# has become most of the graph. Expansion through a hub entity reaches that
# quickly: one entity mentioned in a third of the corpus drags a third of the
# corpus into scope, and pruning it a node at a time is strictly more work than
# one `DETACH DELETE`.
WIDE_SCOPE_MEMORIES = 200


@dataclass(slots=True)
class SyncReport:
    memories: int = 0
    entities: int = 0
    edges: int = 0
    pruned_memories: int = 0
    pruned_entities: int = 0
    # True when the scope grew wide enough that a full rebuild was cheaper. The
    # outcome is the same graph either way; it is reported because a sync that
    # silently became a rebuild is a latency spike nobody can explain.
    escalated: bool = False
    duration_ms: int = 0

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "memories": self.memories,
            "entities": self.entities,
            "edges": self.edges,
            "pruned_memories": self.pruned_memories,
            "pruned_entities": self.pruned_entities,
            "escalated": self.escalated,
            "duration_ms": self.duration_ms,
        }


# --------------------------------------------------------------------------
# Enqueuing
# --------------------------------------------------------------------------


def payload_for(
    *, memory_ids: list[UUID] | None = None, entity_ids: list[UUID] | None = None
) -> dict[str, Any]:
    """The job payload naming what changed.

    Sorted, so that two enqueues describing the same change produce the same
    `dedupe_key` and collapse into one job rather than two doing identical work.
    """
    return {
        "memory_ids": sorted(str(value) for value in memory_ids or []),
        "entity_ids": sorted(str(value) for value in entity_ids or []),
    }


def graph_sync_spec(
    *, memory_ids: list[UUID] | None = None, entity_ids: list[UUID] | None = None
) -> JobSpec:
    """The job that asks for a neighbourhood to be re-projected.

    A spec rather than an enqueue, because two callers need it at two different
    transaction boundaries. `SyncSource` enqueues it *inside* the transaction that
    writes the memory — the whole reason this queue is a table rather than a broker
    — while extraction and resolution enqueue it after their own commit, so a
    worker cannot claim the job before the rows it describes are visible.
    """
    payload = payload_for(memory_ids=memory_ids, entity_ids=entity_ids)
    return JobSpec(
        job_type=JobType.SYNC_GRAPH,
        payload=payload,
        # Two changes to the same neighbourhood while a sync is still pending are
        # one sync: it reads the current state of Postgres when it runs, so the
        # second job would re-derive exactly what the first will.
        dedupe_key=f"sync-graph:{_fingerprint(payload)}",
        # Below the pipeline stages. The graph is a projection, and a sync that
        # jumped the queue would delay ingestion to keep a derived store current.
        priority=-1,
    )


async def enqueue_sync(
    queue: JobQueue,
    *,
    memory_ids: list[UUID] | None = None,
    entity_ids: list[UUID] | None = None,
) -> UUID | None:
    """Queue a projection update. Returns None when there is nothing to name.

    Fire-and-forget from the caller's point of view, which is what makes the
    graph's availability stop being extraction's problem. An unreachable Neo4j
    now fails the sync job and retries with the worker's existing backoff,
    instead of being caught and logged inside a use case that had already
    committed.
    """
    if not memory_ids and not entity_ids:
        return None
    return await queue.enqueue(
        graph_sync_spec(memory_ids=memory_ids, entity_ids=entity_ids)
    )


def _fingerprint(payload: dict[str, Any]) -> str:
    ids = [str(value) for value in [*payload["memory_ids"], *payload["entity_ids"]]]
    if len(ids) == 1:
        # The common case, and worth keeping legible: one memory's sync shows the
        # memory's id in the queue rather than a digest of it.
        return ids[0]
    return hashlib.blake2b("|".join(ids).encode(), digest_size=8).hexdigest()


# --------------------------------------------------------------------------
# The use case
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpandedScope:
    """A scope closed under one step of the mention relation. See the docstring."""

    memory_ids: frozenset[UUID] = field(default_factory=frozenset)
    entity_ids: frozenset[UUID] = field(default_factory=frozenset)

    @property
    def is_empty(self) -> bool:
        return not self.memory_ids and not self.entity_ids

    @property
    def is_wide(self) -> bool:
        return len(self.memory_ids) >= WIDE_SCOPE_MEMORIES

    def as_projection_scope(self) -> Scope:
        return Scope(memory_ids=self.memory_ids, entity_ids=self.entity_ids)


class SyncGraph:
    """Make a neighbourhood of the graph equal what Postgres says it should be."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        graph: GraphStore,
    ) -> None:
        self._sessions = session_factory
        self._graph = graph

    async def __call__(self, payload: dict[str, Any]) -> SyncReport:
        started = time.monotonic()
        requested = _ids_from(payload)
        if requested.is_empty:
            # A job that names nothing cannot be made valid by retrying it.
            raise PermanentError("sync_graph job names no memories and no entities")

        scope = await self.expand(requested)
        log = logger.bind(
            asked_memories=len(requested.memory_ids),
            asked_entities=len(requested.entity_ids),
            scope_memories=len(scope.memory_ids),
            scope_entities=len(scope.entity_ids),
        )

        if scope.is_wide:
            # Escalated rather than refused: the caller asked for the graph to
            # match Postgres, and a rebuild is that answer at a different cost.
            projection = await graph_projection.rebuild(self._sessions, self._graph)
            report = _report_of(projection, escalated=True)
            log.info("graph.sync_escalated", **report.as_dict())
            return _finish(report, started)

        # Read before pruning. The projection is a pure function of Postgres, so
        # the order does not affect the result — but reading first means a failure
        # in the read leaves the graph as it was, rather than emptied.
        projection = await graph_projection.read(
            self._sessions, scope.as_projection_scope()
        )
        pruned_memories = await self._graph.prune_memories(sorted(scope.memory_ids))
        pruned_entities = await self._graph.prune_entities(sorted(scope.entity_ids))
        await graph_projection.write(self._graph, projection)

        report = _report_of(projection)
        report.pruned_memories = pruned_memories
        report.pruned_entities = pruned_entities
        log.info("graph.synced", **report.as_dict())
        return _finish(report, started)

    async def expand(self, requested: ExpandedScope) -> ExpandedScope:
        """Close the requested scope under one step of the mention relation.

        One step, not a transitive closure. Two steps would reach the entities
        co-mentioned with the entities of the named memory — which is a large part
        of the corpus through any hub, and none of it can have changed, because
        the projection of a memory depends only on that memory's own rows.

        Both stores are asked, and the graph's answer is not redundant. Postgres
        says what should be true; the graph says what it currently claims, and the
        difference between them is exactly what a sync exists to remove. Two cases
        make it concrete, and each is invisible in the other store:

        * A re-extraction that dropped an entity. Postgres no longer associates it
          with any memory, so no Postgres query reaches it — but its node is still
          in the graph with an edge from a memory in scope, and if the scope does
          not name it, that node survives as an orphan forever.
        * A merge. The loser is unreachable from Postgres in the same way, and only
          the graph can say which memories need re-projecting once it is pruned.
        """
        memory_ids = set(requested.memory_ids)
        entity_ids = set(requested.entity_ids)

        # What the graph claims, before anything widens the scope against Postgres.
        # Asked with the *requested* ids rather than the growing ones, because the
        # question is about the change, not about its neighbourhood.
        claimed = await self._graph.mention_edges(
            memory_ids=sorted(requested.memory_ids),
            entity_ids=sorted(requested.entity_ids),
        )
        for memory_id, entity_id in claimed:
            memory_ids.add(memory_id)
            entity_ids.add(entity_id)

        async with self._sessions() as session:
            if memory_ids:
                entity_ids |= await _entities_of(session, memory_ids)
            if entity_ids:
                memory_ids |= await _memories_of(session, entity_ids)

        return ExpandedScope(
            memory_ids=frozenset(memory_ids), entity_ids=frozenset(entity_ids)
        )


async def _entities_of(session: AsyncSession, memory_ids: set[UUID]) -> set[UUID]:
    """Entities these memories mention, merges followed.

    The winner rather than the loser, because the winner is the node the
    projection will write and therefore the node the prune has to cover.
    """
    loser = models.Entity.__table__.alias("loser")
    stmt = (
        select(models.EntityMention.entity_id, loser.c.merged_into_id)
        .join(loser, loser.c.id == models.EntityMention.entity_id)
        .where(models.EntityMention.memory_id.in_(memory_ids))
        .distinct()
    )
    found: set[UUID] = set()
    for entity_id, merged_into in await session.execute(stmt):
        found.add(merged_into if merged_into is not None else entity_id)
    return found


async def _memories_of(session: AsyncSession, entity_ids: set[UUID]) -> set[UUID]:
    stmt = (
        select(models.EntityMention.memory_id)
        .where(models.EntityMention.entity_id.in_(entity_ids))
        .distinct()
    )
    return {row[0] for row in await session.execute(stmt)}


def _ids_from(payload: dict[str, Any]) -> ExpandedScope:
    """Parse a payload, refusing a malformed id rather than skipping it.

    A payload is written by this process and read by another, possibly older one.
    An unparseable id there means the two disagree about the format, and a sync
    that skipped it would report success for a neighbourhood it never touched.
    """
    try:
        return ExpandedScope(
            memory_ids=frozenset(
                UUID(str(value)) for value in payload.get("memory_ids") or []
            ),
            entity_ids=frozenset(
                UUID(str(value)) for value in payload.get("entity_ids") or []
            ),
        )
    except ValueError as exc:
        raise PermanentError(f"sync_graph payload holds an unparseable id: {exc}") from exc


def _report_of(projection: GraphProjection, *, escalated: bool = False) -> SyncReport:
    return SyncReport(
        memories=len(projection.memories),
        entities=len(projection.entities),
        edges=len(projection.edges),
        escalated=escalated,
    )


def _finish(report: SyncReport, started: float) -> SyncReport:
    report.duration_ms = int((time.monotonic() - started) * 1000)
    return report
