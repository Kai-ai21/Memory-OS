"""Relationship extraction against a real database.

The four required tests. A fake extractor for the storage paths, a
`FakeLanguageModel` driving the real adapter where the guard being tested lives
in the adapter, and a real Neo4j for the direction test — because direction
surviving the projection is a claim about Neo4j, and a fake graph store would
only establish that the fake preserves it.
"""

from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.extraction.llm import LlmEntityExtractor
from memoryos.application import graph_projection
from memoryos.application.ports import EntityNode, EntityRef, GraphEdge, GraphNode
from memoryos.application.relationships import ExtractRelationships
from memoryos.domain.ids import new_id
from memoryos.domain.values import (
    ContentHash,
    EdgeType,
    EntityType,
    GraphLabel,
    Predicate,
)
from tests.integration.conftest import GraphFixture, Pipeline
from tests.support.fakes import FakeEntityExtractor, FakeLanguageModel

pytestmark = pytest.mark.integration


async def seed(
    pipeline: Pipeline, *, names: tuple[str, ...] = ("postgres", "sqlalchemy")
) -> tuple[UUID, UUID, list[UUID]]:
    """A memory, its first chunk, and entities mentioned in that chunk."""
    await pipeline.ingest()
    async with pipeline.sessions() as session:
        memory_id, chunk_id = (
            await session.execute(
                select(models.MemoryChunk.memory_id, models.MemoryChunk.id)
                .order_by(models.MemoryChunk.ordinal)
                .limit(1)
            )
        ).one()

    entity_ids: list[UUID] = []
    for offset, name in enumerate(names):
        entity_id = new_id()
        entity_ids.append(entity_id)
        async with pipeline.sessions.begin() as session:
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
                    char_start=offset * 10,
                    char_end=offset * 10 + 5,
                    confidence=0.9,
                    extractor_version="test@1",
                )
            )
    return memory_id, chunk_id, entity_ids


async def add_chunk(pipeline: Pipeline, memory_id: UUID, *, ordinal: int) -> UUID:
    """A second chunk on an existing memory, for the repetition test."""
    chunk_id = new_id()
    text = "postgres and sqlalchemy appear together here as well."
    async with pipeline.sessions.begin() as session:
        session.add(
            models.MemoryChunk(
                id=chunk_id,
                memory_id=memory_id,
                ordinal=ordinal,
                content=text,
                token_count=12,
                char_start=1000,
                char_end=1000 + len(text),
                chunker_version="test@1",
                content_hash=ContentHash.of(text.encode("utf-8")).value,
            )
        )
    return chunk_id


async def relationship_count(sessions: async_sessionmaker[AsyncSession]) -> int:
    async with sessions() as session:
        return int(
            (
                await session.execute(
                    select(func.count()).select_from(models.EntityRelationship)
                )
            ).scalar_one()
        )


# --------------------------------------------------------------------------
# An unknown endpoint is rejected
# --------------------------------------------------------------------------


async def test_a_relationship_naming_an_unknown_entity_is_rejected() -> None:
    """The guardrail that makes the candidate list worth passing.

    An edge to an entity that does not exist looks exactly like an edge to one
    that does, until somebody follows it and finds nothing. The model links by
    *number* into a supplied list precisely so this is structurally hard, and an
    out-of-range number is the residual case.
    """
    entities = [
        EntityRef(uuid4(), "postgres", EntityType.TECHNOLOGY),
        EntityRef(uuid4(), "sqlalchemy", EntityType.TECHNOLOGY),
    ]
    payload = (
        '{"relationships": ['
        '{"subject": 0, "predicate": "uses", "object": 1, '
        '"confidence": 0.9, "evidence": "sqlalchemy talks to postgres"}, '
        '{"subject": 0, "predicate": "uses", "object": 7, '
        '"confidence": 0.95, "evidence": "invented"}'
        "]}"
    )
    extractor = LlmEntityExtractor(FakeLanguageModel(payload))

    found = await extractor.extract_relationships(
        "sqlalchemy talks to postgres", entities
    )

    assert len(found) == 1
    assert found[0].subject_id == entities[0].entity_id
    assert found[0].object_id == entities[1].entity_id
    assert extractor.stats.dropped_unknown_entity == 1


async def test_an_unknown_endpoint_never_reaches_the_table(
    pipeline: Pipeline,
) -> None:
    """The same rule at the storage layer.

    The foreign key would refuse a fabricated id anyway — with an integrity
    error that names a constraint rather than the behaviour, mid-transaction,
    taking the memory's real relationships down with it.
    """
    memory_id, _, _ = await seed(pipeline)
    extract = ExtractRelationships(
        pipeline.sessions, FakeEntityExtractor(phantom_relationship=True)
    )

    with pytest.raises(Exception):  # noqa: B017 — the FK's own error type
        await extract(memory_id)


# --------------------------------------------------------------------------
# Repetition is evidence
# --------------------------------------------------------------------------


async def test_the_same_relationship_in_two_chunks_is_two_rows_and_one_edge(
    pipeline: Pipeline, graph: GraphFixture
) -> None:
    """Repetition is the signal M3.5 weights edges by.

    Two chunks asserting the same thing are two rows, because one claim and a
    twice-repeated claim have to stay distinguishable — and one edge, because
    the graph is about connectivity rather than about evidence. The count lives
    on the edge as `assertion_count`.
    """
    memory_id, first_chunk, entity_ids = await seed(pipeline)
    subject, obj = entity_ids

    # A second chunk of the same memory, created here rather than hunted for in
    # the fixture. The fixture chunks each of its files to one chunk, and a test
    # that skipped when it could not find a second would be a required test
    # silently not running — which looks exactly like one that passed.
    second_chunk = await add_chunk(pipeline, memory_id, ordinal=1)

    async with pipeline.sessions.begin() as session:
        for chunk_id in (first_chunk, second_chunk):
            session.add(
                models.EntityRelationship(
                    id=new_id(),
                    subject_id=subject,
                    object_id=obj,
                    predicate=Predicate.USES.value,
                    confidence=0.9,
                    memory_id=memory_id,
                    chunk_id=chunk_id,
                    extractor_version="rel-test@1",
                )
            )

    assert await relationship_count(pipeline.sessions) == 2

    projection = await graph_projection.read(
        pipeline.sessions, graph_projection.Scope(memory_ids=frozenset({memory_id}))
    )
    edges = [
        edge for edge in projection.edges if edge.type is graph_projection.ENTITY_EDGE_TYPE
    ]

    assert len(edges) == 1, "two assertions of one pair are one edge"
    assert edges[0].properties["assertion_count"] == 2
    assert edges[0].properties["predicate"] == Predicate.USES.value


async def test_a_relationship_on_a_merged_away_entity_projects_onto_the_winner(
    pipeline: Pipeline,
) -> None:
    """The M3.3 defect: an edge to an entity resolution had already removed.

    A relationship row keeps naming whichever entity the extractor saw, so after a
    merge its endpoint is an entity that takes no further part in the graph. M3.3
    projected the edge anyway — and because `link` merges its endpoints, that
    *created* the node, carrying an id and no name. Nothing failed, nothing
    logged, and every traversal through the pair walked into a node with no name.

    Invisible to a count check: the edge existed, its properties were right, and
    the node was there. Only its `name` was missing.
    """
    memory_id, chunk_id, entity_ids = await seed(
        pipeline, names=("postgres", "sqlalchemy", "postgresql")
    )
    subject, obj, duplicate = entity_ids

    async with pipeline.sessions.begin() as session:
        session.add(
            models.EntityRelationship(
                id=new_id(),
                subject_id=duplicate,
                object_id=obj,
                predicate=Predicate.USES.value,
                confidence=0.9,
                memory_id=memory_id,
                chunk_id=chunk_id,
                extractor_version="rel-test@1",
            )
        )
        # `postgresql` is merged into `postgres`, which is what M3.2 does to this
        # exact pair, and its mention moves with it.
        await session.execute(
            update(models.Entity)
            .where(models.Entity.id == duplicate)
            .values(merged_into_id=subject)
        )
        await session.execute(
            update(models.EntityMention)
            .where(models.EntityMention.entity_id == duplicate)
            .values(entity_id=subject)
        )

    projection = await graph_projection.read(pipeline.sessions)
    edges = [
        edge for edge in projection.edges if edge.type is graph_projection.ENTITY_EDGE_TYPE
    ]
    projected = {str(node.entity_id) for node in projection.entities}

    assert len(edges) == 1
    assert edges[0].start.key == str(subject), "the endpoint follows the merge"
    assert str(duplicate) not in projected, "a merged-away entity is not a node"
    assert edges[0].start.key in projected, "every endpoint is a projected entity"


async def test_a_relationship_between_two_names_of_one_thing_is_dropped(
    pipeline: Pipeline,
) -> None:
    """Both endpoints resolving to the same entity is not a claim.

    "postgresql uses postgres" was a real assertion about two names. Once the
    merge says they are one thing, projecting it would put a self-loop in the
    graph — a path from an entity to itself that every traversal has to walk and
    no reader can interpret.
    """
    memory_id, chunk_id, entity_ids = await seed(
        pipeline, names=("postgres", "postgresql")
    )
    winner, loser = entity_ids

    async with pipeline.sessions.begin() as session:
        session.add(
            models.EntityRelationship(
                id=new_id(),
                subject_id=loser,
                object_id=winner,
                predicate=Predicate.USES.value,
                confidence=0.9,
                memory_id=memory_id,
                chunk_id=chunk_id,
                extractor_version="rel-test@1",
            )
        )
        await session.execute(
            update(models.Entity)
            .where(models.Entity.id == loser)
            .values(merged_into_id=winner)
        )
        await session.execute(
            update(models.EntityMention)
            .where(models.EntityMention.entity_id == loser)
            .values(entity_id=winner)
        )

    projection = await graph_projection.read(pipeline.sessions)

    assert not [
        edge for edge in projection.edges if edge.type is graph_projection.ENTITY_EDGE_TYPE
    ], "a self-loop is not a relationship"


# --------------------------------------------------------------------------
# Cascade
# --------------------------------------------------------------------------


async def test_deleting_a_memory_cascades_to_its_relationships(
    pipeline: Pipeline,
) -> None:
    """A relationship whose chunk is gone has no evidence left.

    Enforced by the database, because the application is not the only writer: a
    replay's `DELETE FROM memories` has to leave the same invariant standing.
    """
    memory_id, chunk_id, entity_ids = await seed(pipeline)
    async with pipeline.sessions.begin() as session:
        session.add(
            models.EntityRelationship(
                id=new_id(),
                subject_id=entity_ids[0],
                object_id=entity_ids[1],
                predicate=Predicate.USES.value,
                confidence=0.9,
                memory_id=memory_id,
                chunk_id=chunk_id,
                extractor_version="rel-test@1",
            )
        )
    assert await relationship_count(pipeline.sessions) == 1

    async with pipeline.sessions.begin() as session:
        await session.execute(
            delete(models.Memory).where(models.Memory.id == memory_id)
        )

    assert await relationship_count(pipeline.sessions) == 0
    # The entities survive: they are named elsewhere in the corpus, and a
    # relationship losing its evidence is not a reason to forget its endpoints.
    async with pipeline.sessions() as session:
        assert (
            await session.execute(select(func.count()).select_from(models.Entity))
        ).scalar_one() == 2


# --------------------------------------------------------------------------
# Direction survives the projection
# --------------------------------------------------------------------------


@pytest.mark.graph
async def test_direction_is_preserved_through_the_neo4j_projection(
    pipeline: Pipeline, graph: GraphFixture
) -> None:
    """"A supersedes B" must not come back as "B supersedes A".

    Asserted against a real Neo4j rather than a recording fake, because this is
    a claim about what the projection stores: a fake would only establish that
    the fake preserves direction. The traversal is undirected — that is what
    `neighbours` does — so the assertion reads the path's own node order.
    """
    subject_id = graph.new_id()
    object_id = graph.new_id()
    await graph.store.upsert_entity(_entity_node(subject_id, "new-design"))
    await graph.store.upsert_entity(_entity_node(object_id, "old-design"))

    await graph.store.link(
        GraphEdge(
            type=EdgeType.RELATES_TO,
            start=GraphNode(GraphLabel.ENTITY, str(subject_id)),
            end=GraphNode(GraphLabel.ENTITY, str(object_id)),
            properties={
                "predicate": Predicate.SUPERSEDES.value,
                "assertion_count": 1,
                "confidence": 0.9,
            },
        )
    )

    paths = await graph.store.neighbours(subject_id, depth=1, limit=10)

    assert len(paths) == 1
    # The path starts at the queried node and ends at the other, and the stored
    # edge runs subject -> object. Reading the far end is what tells us the
    # direction was not flipped on the way in.
    assert paths[0].nodes[0].key == str(subject_id)
    assert paths[0].nodes[-1].key == str(object_id)

    # And the reverse query walks the same edge backwards rather than finding a
    # second one, which is what "one directed edge" means.
    reverse = await graph.store.neighbours(object_id, depth=1, limit=10)
    assert len(reverse) == 1
    assert reverse[0].nodes[-1].key == str(subject_id)


def _entity_node(entity_id: UUID, name: str) -> EntityNode:
    return EntityNode(
        entity_id=entity_id,
        name=name,
        canonical_name=name,
        type=EntityType.CONCEPT.value,
        confidence=0.9,
    )


async def test_a_relationship_on_an_entity_with_no_mentions_is_not_projected(
    pipeline: Pipeline,
) -> None:
    """The projection has to be closed under its own edges.

    `entity_relationships` rows survive the deletion of their entities' mentions —
    the foreign keys cascade from chunks and memories, not from mentions — so a
    re-extraction that stops finding an entity leaves rows pointing at one the
    projection no longer writes a node for. Because `link` merges its endpoints,
    each of those rows then *created* an `Entity` node carrying an id and nothing
    else: the M3.3 defect again, through a different door.

    Found by `graph verify` on the live corpus reporting four nameless nodes, which
    is the check doing precisely what it was built for.
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
                extractor_version="rel-test@1",
            )
        )

    before = await graph_projection.read(pipeline.sessions)
    assert [
        edge for edge in before.edges if edge.type is graph_projection.ENTITY_EDGE_TYPE
    ], "the edge is projected while both endpoints have mentions"

    # The object keeps its row and loses its last mention, which is what a
    # re-extraction that no longer finds it leaves behind.
    async with pipeline.sessions.begin() as session:
        await session.execute(
            delete(models.EntityMention).where(models.EntityMention.entity_id == obj)
        )

    after = await graph_projection.read(pipeline.sessions)
    projected = {str(node.entity_id) for node in after.entities}

    assert not [
        edge for edge in after.edges if edge.type is graph_projection.ENTITY_EDGE_TYPE
    ], "an edge whose endpoint has no node must not be projected"
    assert str(obj) not in projected
    # And every edge that *is* projected names endpoints the projection covers,
    # which is the invariant rather than this one case.
    for edge in after.edges:
        if edge.type is graph_projection.ENTITY_EDGE_TYPE:
            assert edge.start.key in projected
            assert edge.end.key in projected
