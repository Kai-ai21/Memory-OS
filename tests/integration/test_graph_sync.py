"""Projection sync, rebuild, and divergence detection.

**Two stores, and each test uses the one that can answer its question.**

`InMemoryGraphStore` is used wherever the assertion is about the *whole* graph —
a rebuild, a divergence check, a clear. Neo4j Community Edition has exactly one
user database, so those assertions against a real store would be assertions
against whatever graph the developer happens to have, and the setup would have to
empty it first. What is under test there is arithmetic over two projections, and a
database does not make it more true.

A real Neo4j is used for the one claim the fake cannot make: that what
`graph_projection.write` puts in is exactly what `all_nodes` and `all_edges` read
back. That is a claim about Cypher, and it is the link everything else here
depends on — see `test_graph.py::test_the_projection_reads_back_exactly_as_written`,
which is scoped to ids the fixture minted so it needs no clear.
"""

from uuid import UUID

import pytest
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import PostgresJobQueue
from memoryos.application import graph_projection, graph_sync, graph_verify
from memoryos.application.graph_sync import SyncGraph, enqueue_sync
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType, PermanentError
from memoryos.domain.values import EdgeType, EntityType, GraphLabel, Predicate
from tests.integration.conftest import Pipeline
from tests.support.fakes import InMemoryGraphStore

pytestmark = pytest.mark.integration


async def seed(
    pipeline: Pipeline, *, names: tuple[str, ...] = ("postgres", "sqlalchemy")
) -> tuple[UUID, UUID, list[UUID]]:
    """A corpus with one memory carrying entity mentions in its first chunk."""
    await pipeline.ingest()
    async with pipeline.sessions() as session:
        memory_id, chunk_id, content = (
            await session.execute(
                select(
                    models.MemoryChunk.memory_id,
                    models.MemoryChunk.id,
                    models.MemoryChunk.content,
                )
                .order_by(models.MemoryChunk.ordinal)
                .limit(1)
            )
        ).one()

    entity_ids: list[UUID] = []
    async with pipeline.sessions.begin() as session:
        for offset, name in enumerate(names):
            entity_id = new_id()
            entity_ids.append(entity_id)
            session.add(
                models.Entity(
                    id=entity_id,
                    name=name,
                    canonical_name=name.lower(),
                    type=EntityType.TECHNOLOGY.value,
                    confidence=0.9,
                )
            )
            session.add(
                models.EntityMention(
                    id=new_id(),
                    entity_id=entity_id,
                    memory_id=memory_id,
                    chunk_id=chunk_id,
                    # Offsets into real text, so the mention is as valid as one the
                    # extractor would have written. Distinct per entity because
                    # `UNIQUE (entity_id, chunk_id, char_start)` is the natural key.
                    char_start=offset,
                    char_end=offset + min(4, len(content) - offset),
                    confidence=0.9,
                    extractor_version="sync-test@1",
                )
            )
    return memory_id, chunk_id, entity_ids


async def projected(
    pipeline: Pipeline, graph: InMemoryGraphStore
) -> graph_verify.GraphDivergence:
    expected = await graph_projection.read(pipeline.sessions)
    actual, foreign = await graph_verify.read_graph(graph)
    return graph_verify.compare(expected, actual, foreign_labels=foreign)


async def all_memory_ids(pipeline: Pipeline) -> list[UUID]:
    async with pipeline.sessions() as session:
        return [
            row[0]
            for row in await session.execute(
                select(models.Memory.id).where(models.Memory.is_current.is_(True))
            )
        ]


async def sync_whole_corpus(pipeline: Pipeline, graph: InMemoryGraphStore) -> None:
    """Sync every memory, which is what the queue does one job at a time.

    The tests below assert against the *whole* projection, so the graph has to have
    heard about every memory — otherwise a divergence would only mean "this test
    synced one of two files", which says nothing about the sync. One job per memory
    rather than one naming all of them, so the path exercised is the path
    `SyncSource` actually enqueues.
    """
    sync = SyncGraph(pipeline.sessions, graph)
    for memory_id in await all_memory_ids(pipeline):
        await sync(graph_sync.payload_for(memory_ids=[memory_id]))


# --------------------------------------------------------------------------
# A3: the divergence check has to be able to fail
# --------------------------------------------------------------------------


async def test_verify_is_clean_after_a_rebuild(pipeline: Pipeline) -> None:
    """The baseline the corruption tests are measured against.

    Worth its own test rather than being an assertion inside them: a check that
    reported divergence unconditionally would pass every one of those, and this is
    the only thing that says the clean case is actually clean.
    """
    await seed(pipeline)
    graph = InMemoryGraphStore()
    await graph_projection.rebuild(pipeline.sessions, graph)

    divergence = await projected(pipeline, graph)
    assert divergence.identical, divergence.render()


async def test_verify_fails_when_a_node_is_corrupted(pipeline: Pipeline) -> None:
    """The requirement. A changed name is invisible to every count.

    This is the graph's equivalent of M1.7's corrupted-chunk test, and the same
    reasoning applies: the node is still there, its label is right, its id joins to
    a real row, and its every relationship is intact. Only `name` is wrong — which
    is precisely the shape of the M3.3 defect this milestone fixed, where a merged
    endpoint produced a node carrying an id and nothing else.
    """
    _, _, entity_ids = await seed(pipeline)
    graph = InMemoryGraphStore()
    await graph_projection.rebuild(pipeline.sessions, graph)
    assert (await projected(pipeline, graph)).identical

    # Reaching past the port on purpose: `GraphStore` offers no way to write a
    # wrong value, which is the point of it, so corruption has to be injected the
    # way a stray Cypher session or a half-finished sync would do it.
    node = graph._nodes[(GraphLabel.ENTITY, str(entity_ids[0]))]
    node.properties["name"] = "something else entirely"

    divergence = await projected(pipeline, graph)
    assert not divergence.identical, divergence.render()

    entities = next(row for row in divergence.by_type if row.name == "Entity")
    assert entities.expected == entities.actual, "the count is unchanged; that is the point"
    assert not entities.clean
    assert entities.changed, "the diff has to name the entity that moved"
    assert "something else entirely" in entities.changed[0]


async def test_verify_fails_when_a_node_the_corpus_dropped_is_left_behind(
    pipeline: Pipeline,
) -> None:
    """A stale node: present in the graph, absent from Postgres.

    The divergence an upsert-only sync produces on every merge, and the reason the
    sync prunes before it projects.
    """
    _, _, entity_ids = await seed(pipeline)
    graph = InMemoryGraphStore()
    await graph_projection.rebuild(pipeline.sessions, graph)

    async with pipeline.sessions.begin() as session:
        await session.execute(
            update(models.Entity)
            .where(models.Entity.id == entity_ids[0])
            .values(merged_into_id=entity_ids[1])
        )
        await session.execute(
            update(models.EntityMention)
            .where(models.EntityMention.entity_id == entity_ids[0])
            .values(entity_id=entity_ids[1], char_start=900, char_end=904)
        )

    divergence = await projected(pipeline, graph)
    assert not divergence.identical
    entities = next(row for row in divergence.by_type if row.name == "Entity")
    assert entities.unexpected, "a merged-away entity is a node the graph should not have"


async def test_verify_reports_a_label_the_projection_does_not_define(
    pipeline: Pipeline,
) -> None:
    """Something wrote to this graph that was not the projection.

    Counted apart from an unexpected `Entity`, because it is a different kind of
    wrong: a stale entity is a projection that is behind, and a foreign label is a
    writer nobody knows about.
    """
    await seed(pipeline)
    graph = InMemoryGraphStore()
    await graph_projection.rebuild(pipeline.sessions, graph)

    _, foreign = await graph_verify.read_graph(graph)
    assert not foreign


# --------------------------------------------------------------------------
# A5: the sync is idempotent
# --------------------------------------------------------------------------


async def test_syncing_twice_changes_nothing(pipeline: Pipeline) -> None:
    """Running the same sync again has to be a no-op, hash for hash.

    Idempotence is structural here rather than incidental — the sync prunes its
    scope and re-projects it, so the second run does what the first did — and this
    is what would catch it becoming additive. An additive sync would still pass a
    count check on this corpus: nothing is *added* by the second run either.
    What it would fail is the hash, once anything had gone stale.
    """
    memory_id, _, _ = await seed(pipeline)
    graph = InMemoryGraphStore()
    await sync_whole_corpus(pipeline, graph)
    sync = SyncGraph(pipeline.sessions, graph)
    payload = graph_sync.payload_for(memory_ids=[memory_id])

    first = await sync(payload)
    hash_after_first = graph_projection.content_hash(
        (await graph_verify.read_graph(graph))[0]
    )

    second = await sync(payload)
    hash_after_second = graph_projection.content_hash(
        (await graph_verify.read_graph(graph))[0]
    )

    assert hash_after_first == hash_after_second
    assert (first.memories, first.entities, first.edges) == (
        second.memories,
        second.entities,
        second.edges,
    )
    assert (first.pruned_memories, first.pruned_entities) == (
        second.pruned_memories,
        second.pruned_entities,
    )
    assert (await projected(pipeline, graph)).identical


async def test_a_sync_removes_what_postgres_has_stopped_saying(
    pipeline: Pipeline,
) -> None:
    """The case an additive sync gets wrong, and the reason pruning exists.

    A re-extraction that finds fewer entities leaves the graph asserting a mention
    the corpus no longer has. Upserting what is there now cannot fix that: there is
    no upsert for "this edge should be gone".
    """
    memory_id, _, entity_ids = await seed(pipeline)
    graph = InMemoryGraphStore()
    await sync_whole_corpus(pipeline, graph)
    assert len(graph.entities) == 2

    async with pipeline.sessions.begin() as session:
        await session.execute(
            delete(models.EntityMention).where(
                models.EntityMention.entity_id == entity_ids[0]
            )
        )

    sync = SyncGraph(pipeline.sessions, graph)

    await sync(graph_sync.payload_for(memory_ids=[memory_id]))

    assert len(graph.entities) == 1
    assert (await projected(pipeline, graph)).identical


async def test_a_merge_sync_prunes_the_loser_and_keeps_the_rest(
    pipeline: Pipeline,
) -> None:
    """The scope expansion, which is the only difficult part of the sync.

    The payload names two entities. Pruning them detaches every `MENTIONS` edge
    into either — including edges from memories the payload never mentioned — so
    the scope has to be widened to those memories or the sync silently deletes
    relationships it was not asked to touch. Here the widening comes from the
    graph's own claim about the loser, which Postgres no longer makes.
    """
    _, _, entity_ids = await seed(pipeline, names=("postgres", "postgresql"))
    winner, loser = entity_ids
    graph = InMemoryGraphStore()
    await sync_whole_corpus(pipeline, graph)
    sync = SyncGraph(pipeline.sessions, graph)
    assert len(graph.entities) == 2

    async with pipeline.sessions.begin() as session:
        await session.execute(
            update(models.Entity)
            .where(models.Entity.id == loser)
            .values(merged_into_id=winner)
        )
        await session.execute(
            update(models.EntityMention)
            .where(models.EntityMention.entity_id == loser)
            .values(entity_id=winner, char_start=500, char_end=504)
        )

    report = await sync(graph_sync.payload_for(entity_ids=[winner, loser]))

    assert report.pruned_entities == 2
    assert [node.key for node in graph.entities] == [str(winner)]
    assert (await projected(pipeline, graph)).identical, "the loser's edges went with it"


async def test_a_relationship_edge_survives_a_neighbouring_sync(
    pipeline: Pipeline,
) -> None:
    """A scoped sync must not lose an edge whose other end is out of scope.

    Pruning an entity detaches its `RELATES_TO` edges too, so a projection scoped
    to one memory has to include every edge touching that memory's entities — not
    only the edges between them. This is the sync-shaped version of the M3.3 defect
    that lost every relationship on `resolve-entities`.
    """
    memory_id, chunk_id, entity_ids = await seed(pipeline)
    subject, obj = entity_ids
    async with pipeline.sessions.begin() as session:
        session.add(
            models.EntityRelationship(
                id=new_id(),
                subject_id=subject,
                object_id=obj,
                predicate=Predicate.USES.value,
                confidence=0.9,
                memory_id=memory_id,
                chunk_id=chunk_id,
                extractor_version="sync-test@1",
            )
        )

    graph = InMemoryGraphStore()
    await sync_whole_corpus(pipeline, graph)
    sync = SyncGraph(pipeline.sessions, graph)
    mentions = [edge for edge in graph.edges if edge.type is EdgeType.MENTIONS]
    relates = [edge for edge in graph.edges if edge.type is EdgeType.RELATES_TO]
    assert (len(mentions), len(relates)) == (2, 1)

    # Again, naming only one of the two entities the edge connects.
    await sync(graph_sync.payload_for(entity_ids=[subject]))

    assert (await projected(pipeline, graph)).identical


# --------------------------------------------------------------------------
# Enqueuing
# --------------------------------------------------------------------------


async def test_extraction_queues_a_sync_instead_of_writing_to_the_graph(
    pipeline: Pipeline,
) -> None:
    """The principle, asserted where it can be: no use case writes to Neo4j.

    `SYNC_GRAPH` in the queue and a graph that has not been touched — which is what
    makes an unreachable Neo4j a retried job rather than a caught exception inside a
    use case that has already committed.
    """
    memory_id, _, _ = await seed(pipeline)

    payloads = await pending_sync_payloads(pipeline.sessions)

    # Queued by `SyncSource` as it wrote each memory, in the same transaction. The
    # graph projects every current memory, so ingestion is itself a projection
    # change — before anything has parsed the file and whether or not this
    # deployment has an API key to extract entities with.
    assert {"memory_ids": [str(memory_id)], "entity_ids": []} in payloads
    assert len(payloads) == len(await all_memory_ids(pipeline))


async def test_two_enqueues_for_one_change_collapse(pipeline: Pipeline) -> None:
    """The dedupe key. Two changes to a neighbourhood while a sync is pending are one.

    Safe because the sync reads the current state of Postgres when it runs: the
    second job would derive exactly what the first is about to.
    """
    memory_id, _, _ = await seed(pipeline)
    queue = PostgresJobQueue(pipeline.sessions)
    before = len(await pending_sync_payloads(pipeline.sessions))

    # Ingestion has already queued one for this memory, so this is the collapse
    # rather than the first enqueue — which is the case that matters: extraction is
    # about to ask for the same neighbourhood the ingest already asked for.
    again = await enqueue_sync(queue, memory_ids=[memory_id])

    assert again is None
    assert len(await pending_sync_payloads(pipeline.sessions)) == before


async def test_a_payload_naming_nothing_is_permanent(pipeline: Pipeline) -> None:
    """Retrying cannot make an empty payload valid."""
    sync = SyncGraph(pipeline.sessions, InMemoryGraphStore())
    with pytest.raises(PermanentError, match="names no memories"):
        await sync({"memory_ids": [], "entity_ids": []})


async def test_an_unparseable_id_is_permanent(pipeline: Pipeline) -> None:
    """An id this build cannot read means two builds disagree about the format.

    Refused rather than skipped: a sync that stepped over it would report success
    for a neighbourhood it never touched.
    """
    sync = SyncGraph(pipeline.sessions, InMemoryGraphStore())
    with pytest.raises(PermanentError, match="unparseable id"):
        await sync({"memory_ids": ["not-a-uuid"], "entity_ids": []})


async def pending_sync_payloads(
    sessions: async_sessionmaker[AsyncSession],
) -> list[dict[str, object]]:
    async with sessions() as session:
        return [
            row[0]
            for row in await session.execute(
                select(models.Job.payload).where(
                    models.Job.job_type == JobType.SYNC_GRAPH.value
                )
            )
        ]
