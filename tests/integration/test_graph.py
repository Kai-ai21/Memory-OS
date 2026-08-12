"""The graph, against a real Neo4j.

Everything here needs a running database and none of it needs Postgres, which is
why these carry `graph` rather than `integration`. The fixture skips when Neo4j
is absent; see the note above `GraphFixture` in conftest for why isolation works
by identity here instead of by truncation.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest
from neo4j import AsyncGraphDatabase

from memoryos.adapters.graph.neo4j_store import (
    MAX_DEPTH,
    Neo4jGraphStore,
    UnknownGraphLabel,
)
from memoryos.adapters.graph.schema import (
    SCHEMA_VERSION,
    STATEMENTS,
    apply_schema,
    read_schema_version,
)
from memoryos.application import graph_projection, graph_verify
from memoryos.application.graph_projection import GraphProjection
from memoryos.application.ports import (
    EntityNode,
    GraphEdge,
    GraphNode,
    GraphReach,
    MemoryNode,
    SourceNode,
)
from memoryos.domain.values import EdgeType, GraphLabel, MemoryKind, Predicate
from tests.integration.conftest import GraphFixture
from tests.support.fakes import InMemoryGraphStore

pytestmark = pytest.mark.graph

OCCURRED_AT = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)


async def schema_object_names(graph: GraphFixture) -> set[str]:
    """The constraints and indexes this database has, by name.

    Read through a separate driver for the same reason the fixture's cleanup
    does: `SHOW CONSTRAINTS` is not something `GraphStore` offers, and it should
    not start offering it because a test wanted it.
    """
    driver = AsyncGraphDatabase.driver(graph.uri, auth=(graph.user, graph.password))
    try:
        constraints = await driver.execute_query("SHOW CONSTRAINTS YIELD name")
        indexes = await driver.execute_query("SHOW INDEXES YIELD name")
        return {record["name"] for record in constraints.records} | {
            record["name"] for record in indexes.records
        }
    finally:
        await driver.close()


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


async def test_the_schema_applies_twice_without_changing_anything(
    graph: GraphFixture,
) -> None:
    """The property that replaces having a migration framework.

    Neo4j has no Alembic, so the schema is applied on every connect rather than
    once by a migration somebody has to remember to run. That is only safe if
    applying it repeatedly is a no-op — otherwise every process start would be a
    schema change, and a second one would fail on a constraint that already
    exists.

    Asserted on the database's own view rather than on the absence of an
    exception: `CREATE CONSTRAINT ... IF NOT EXISTS` not raising proves only that
    it did not raise. What matters is that the second run left the same
    constraints, not a duplicate set under generated names.
    """
    await graph.store.ensure_schema()
    after_first = await schema_object_names(graph)

    # A second store, so the in-process `_schema_ready` flag cannot be what makes
    # this pass. This is the fresh-process case: the statements really do run
    # again, against a database that already has them.
    second = Neo4jGraphStore(graph.uri, graph.user, graph.password)
    try:
        await second.ensure_schema()
        after_second = await schema_object_names(graph)
        assert await second.schema_version() == SCHEMA_VERSION
    finally:
        await second.close()

    assert after_second == after_first
    # And what was applied is what the module declares, not a subset that
    # happened to survive.
    assert {"memory_id", "entity_id", "source_id", "entity_canonical"} <= after_first
    assert len(STATEMENTS) == 4


async def test_the_schema_version_is_recorded_for_drift_to_be_visible(
    graph: GraphFixture,
) -> None:
    """Without this, a database predating a constraint looks like one that has it."""
    driver = AsyncGraphDatabase.driver(graph.uri, auth=(graph.user, graph.password))
    try:
        await apply_schema(driver)
        assert await read_schema_version(driver) == SCHEMA_VERSION
    finally:
        await driver.close()


# --------------------------------------------------------------------------
# Idempotent writes
# --------------------------------------------------------------------------


async def test_upserting_the_same_memory_twice_leaves_one_node(
    graph: GraphFixture,
) -> None:
    """`MERGE`, not `CREATE`. The property the whole projection rests on.

    A rebuild that doubled its nodes on the second run would not be a rebuild,
    and nothing about it would fail — the graph would simply hold two of
    everything, and every traversal would return each answer twice.
    """
    memory_id = graph.new_id()
    entity_id = graph.new_id()

    first = MemoryNode(memory_id, "notes/decision.md", MemoryKind.NOTE, OCCURRED_AT)
    await graph.store.upsert_memory(first)
    await graph.store.upsert_memory(first)

    # The second write also carries different properties, because idempotent has
    # to mean "converges on the latest", not "ignores everything after the
    # first". A rebuild whose second run silently kept stale properties would be
    # the same defect wearing a different hat.
    await graph.store.upsert_memory(
        MemoryNode(memory_id, "notes/renamed.md", MemoryKind.DOCUMENT, OCCURRED_AT)
    )

    # Counted through a traversal, which is the only read the port offers: one
    # entity linked to this memory should find exactly one path to exactly one
    # memory node, however many times the memory was written.
    await graph.store.upsert_entity(
        EntityNode(entity_id, "Ada Lovelace", "ada lovelace", "person", 0.91)
    )
    await graph.store.link(
        GraphEdge(
            EdgeType.MENTIONS,
            GraphNode(GraphLabel.MEMORY, str(memory_id)),
            GraphNode(GraphLabel.ENTITY, str(entity_id)),
        )
    )

    paths = await graph.store.neighbours(entity_id, depth=1, limit=50)
    memories = [node for path in paths for node in path.nodes if node.label is GraphLabel.MEMORY]
    assert len(memories) == 1
    assert memories[0].key == str(memory_id)
    assert memories[0].properties["external_key"] == "notes/renamed.md"
    assert memories[0].properties["kind"] == MemoryKind.DOCUMENT.value


async def test_linking_twice_leaves_one_relationship(graph: GraphFixture) -> None:
    """The same argument as above, for edges.

    M3.1 re-extracts entities from a memory whenever it is re-normalized, so the
    same mention is written repeatedly by design. Duplicate relationships would
    make every path count the same connection several times.
    """
    memory_id = graph.new_id()
    entity_id = graph.new_id()
    await graph.store.upsert_memory(
        MemoryNode(memory_id, "notes/a.md", MemoryKind.NOTE, OCCURRED_AT)
    )
    await graph.store.upsert_entity(
        EntityNode(entity_id, "Chen", "chen", "person", 0.8)
    )
    edge = GraphEdge(
        EdgeType.MENTIONS,
        GraphNode(GraphLabel.MEMORY, str(memory_id)),
        GraphNode(GraphLabel.ENTITY, str(entity_id)),
        {"mentions": 3},
    )
    await graph.store.link(edge)
    await graph.store.link(edge)

    paths = await graph.store.neighbours(entity_id, depth=1, limit=50)
    assert len(paths) == 1
    assert paths[0].edges == (EdgeType.MENTIONS,)


# --------------------------------------------------------------------------
# Traversal
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Neighbourhood:
    entity_id: UUID
    source_id: UUID
    memory_ids: list[UUID]


@pytest.fixture
async def neighbourhood(graph: GraphFixture) -> Neighbourhood:
    """An entity with three memories at one hop and a source at two.

    Deliberately asymmetric: the depth assertions are only meaningful if the
    graph has something at depth 2 that is not at depth 1.
    """
    entity_id = graph.new_id()
    source_id = graph.new_id()
    memory_ids = [graph.new_id() for _ in range(3)]

    await graph.store.upsert_entity(
        EntityNode(entity_id, "Neo4j", "neo4j", "technology", 0.99)
    )
    for ordinal, memory_id in enumerate(memory_ids):
        await graph.store.upsert_memory(
            MemoryNode(memory_id, f"notes/{ordinal}.md", MemoryKind.NOTE, OCCURRED_AT)
        )
        await graph.store.link(
            GraphEdge(
                EdgeType.MENTIONS,
                GraphNode(GraphLabel.MEMORY, str(memory_id)),
                GraphNode(GraphLabel.ENTITY, str(entity_id)),
            )
        )
    # One memory also belongs to a source, which is the only depth-2 node.
    await graph.store.link(
        GraphEdge(
            EdgeType.FROM_SOURCE,
            GraphNode(GraphLabel.MEMORY, str(memory_ids[0])),
            GraphNode(GraphLabel.SOURCE, str(source_id)),
        )
    )
    return Neighbourhood(entity_id, source_id, memory_ids)


async def test_neighbours_returns_paths_at_the_requested_depth(
    graph: GraphFixture, neighbourhood: Neighbourhood
) -> None:
    """Depth bounds the walk, and the bound is the point of using a graph at all.

    Also the direction check. `MENTIONS` points from the memory to the entity, so
    a traversal that respected direction would leave an entity with no
    neighbours at all — and would look exactly like an empty graph.
    """
    entity_id = neighbourhood.entity_id

    shallow = await graph.store.neighbours(entity_id, depth=1, limit=50)
    assert [path.length for path in shallow] == [1, 1, 1]
    assert {node.label for path in shallow for node in path.nodes} == {
        GraphLabel.ENTITY,
        GraphLabel.MEMORY,
    }

    deeper = await graph.store.neighbours(entity_id, depth=2, limit=50)
    lengths = [path.length for path in deeper]
    # Three at one hop, and the source reached through the first memory.
    assert lengths == [1, 1, 1, 2]
    # Shortest first, which is what makes a limit a useful bound rather than an
    # arbitrary truncation.
    assert lengths == sorted(lengths)

    two_hop = next(path for path in deeper if path.length == 2)
    assert two_hop.edges == (EdgeType.MENTIONS, EdgeType.FROM_SOURCE)
    assert [node.label for node in two_hop.nodes] == [
        GraphLabel.ENTITY,
        GraphLabel.MEMORY,
        GraphLabel.SOURCE,
    ]
    # One more node than edge, always. The port promises the two interleave.
    assert all(len(path.nodes) == path.length + 1 for path in deeper)


async def test_neighbours_respects_the_limit(
    graph: GraphFixture, neighbourhood: Neighbourhood
) -> None:
    """The bound that stops a well-connected entity returning the whole graph."""
    entity_id = neighbourhood.entity_id

    assert len(await graph.store.neighbours(entity_id, depth=2, limit=2)) == 2
    assert len(await graph.store.neighbours(entity_id, depth=2, limit=1)) == 1
    # Truncation keeps the shortest, because the ordering is applied before the
    # limit rather than after it.
    assert [path.length for path in await graph.store.neighbours(entity_id, depth=2, limit=3)] == [
        1,
        1,
        1,
    ]
    assert await graph.store.neighbours(entity_id, depth=2, limit=0) == []


async def test_an_entity_with_no_neighbours_returns_nothing(
    graph: GraphFixture,
) -> None:
    """An ordinary answer, not an error — and not a missing node either."""
    entity_id = graph.new_id()
    await graph.store.upsert_entity(
        EntityNode(entity_id, "Solitary", "solitary", "concept", 0.5)
    )
    assert await graph.store.neighbours(entity_id, depth=3, limit=10) == []


@pytest.mark.parametrize("depth", [0, -1, MAX_DEPTH + 1])
async def test_an_out_of_range_depth_is_refused(
    graph: GraphFixture, depth: int
) -> None:
    """Refused rather than clamped.

    The bound exists because an undirected variable-length match fans out with
    the branching factor. Silently clamping a request for depth 20 would answer a
    question nobody asked and hide that the caller believed something false.
    """
    with pytest.raises(ValueError, match="depth must be between"):
        await graph.store.neighbours(graph.new_id(), depth=depth, limit=10)


# --------------------------------------------------------------------------
# Round trip: what the projection writes is what it reads back
#
# The one claim in M3.4 that the in-memory store cannot make, and the link every
# assertion in `test_graph_sync.py` rests on. Those tests compare a projection
# against what a `GraphStore` reports; if Cypher lost a property, rounded a float,
# or returned a `neo4j.time.DateTime` that compared unequal to a `datetime`, every
# one of them would still pass and `graph verify` would report divergence on a
# perfectly good graph.
#
# Scoped to ids the fixture minted, so this needs no `clear()` — see the note above
# `GraphFixture`. `all_nodes` and `all_edges` read the whole database, so the
# assertions filter to the minted set rather than comparing totals.
# --------------------------------------------------------------------------


async def test_the_projection_reads_back_exactly_as_written(graph: GraphFixture) -> None:
    """Write one of every node and edge type, then read them back and compare.

    Every property that `graph_verify` compares is checked here, including the two
    that cross a type boundary: `occurred_at`, which the driver returns as its own
    temporal type, and `confidence`, which Postgres stores as a 32-bit REAL and
    Neo4j as a double.
    """
    source_id, memory_id, entity_id, other_id = (graph.new_id() for _ in range(4))
    expected = GraphProjection(
        sources=(SourceNode(source_id=source_id, name="vault", kind="filesystem"),),
        memories=(
            MemoryNode(
                memory_id=memory_id,
                external_key="notes/design.md",
                kind=MemoryKind.NOTE,
                occurred_at=OCCURRED_AT,
            ),
        ),
        entities=(
            EntityNode(entity_id, "Postgres", "postgres", "technology", 0.9),
            EntityNode(other_id, "SQLAlchemy", "sqlalchemy", "technology", 0.75),
        ),
        edges=(
            GraphEdge(
                type=EdgeType.FROM_SOURCE,
                start=GraphNode(GraphLabel.MEMORY, str(memory_id)),
                end=GraphNode(GraphLabel.SOURCE, str(source_id)),
            ),
            GraphEdge(
                type=EdgeType.MENTIONS,
                start=GraphNode(GraphLabel.MEMORY, str(memory_id)),
                end=GraphNode(GraphLabel.ENTITY, str(entity_id)),
                properties={"mentions": 3, "chunk_ordinal": 0},
            ),
            GraphEdge(
                type=EdgeType.RELATES_TO,
                start=GraphNode(GraphLabel.ENTITY, str(other_id)),
                end=GraphNode(GraphLabel.ENTITY, str(entity_id)),
                properties={
                    "predicate": Predicate.USES.value,
                    "assertion_count": 2,
                    "confidence": 0.8,
                },
            ),
        ),
    )

    await graph_projection.write(graph.store, expected)
    actual = await _minted_projection(graph)

    divergence = graph_verify.compare(expected, actual)
    assert divergence.identical, divergence.render()


async def test_projecting_twice_leaves_one_of_everything(graph: GraphFixture) -> None:
    """`MERGE`, not `CREATE`, all the way down.

    Asserted against a real database because it is a claim about Cypher: the fake's
    dict cannot double a node however the projection is written, so this is the only
    place the property is actually under test.
    """
    memory_id, entity_id = graph.new_id(), graph.new_id()
    projection = GraphProjection(
        memories=(
            MemoryNode(memory_id, "notes/twice.md", MemoryKind.NOTE, OCCURRED_AT),
        ),
        entities=(EntityNode(entity_id, "Neo4j", "neo4j", "technology", 0.5),),
        edges=(
            GraphEdge(
                type=EdgeType.MENTIONS,
                start=GraphNode(GraphLabel.MEMORY, str(memory_id)),
                end=GraphNode(GraphLabel.ENTITY, str(entity_id)),
                properties={"mentions": 1, "chunk_ordinal": 0},
            ),
        ),
    )

    await graph_projection.write(graph.store, projection)
    once = await _minted_projection(graph)
    await graph_projection.write(graph.store, projection)
    twice = await _minted_projection(graph)

    assert graph_projection.content_hash(once) == graph_projection.content_hash(twice)


async def test_pruning_a_memory_takes_its_edges_with_it(graph: GraphFixture) -> None:
    """Detached, which is what makes a scoped sync a rebuild rather than a merge.

    An edge that should no longer exist cannot survive by not being mentioned — so
    the delete has to remove the relationships, not only the node.
    """
    memory_id, entity_id = graph.new_id(), graph.new_id()
    await graph_projection.write(
        graph.store,
        GraphProjection(
            memories=(
                MemoryNode(memory_id, "notes/gone.md", MemoryKind.NOTE, OCCURRED_AT),
            ),
            entities=(EntityNode(entity_id, "Kept", "kept", "concept", 0.5),),
            edges=(
                GraphEdge(
                    type=EdgeType.MENTIONS,
                    start=GraphNode(GraphLabel.MEMORY, str(memory_id)),
                    end=GraphNode(GraphLabel.ENTITY, str(entity_id)),
                    properties={"mentions": 1, "chunk_ordinal": 0},
                ),
            ),
        ),
    )
    assert len((await _minted_projection(graph)).edges) == 1

    removed = await graph.store.prune_memories([memory_id])
    after = await _minted_projection(graph)

    assert removed == 1
    assert not after.memories
    assert not after.edges, "the MENTIONS edge went with the node"
    assert len(after.entities) == 1, "the entity at the other end did not"
    # Idempotent: pruning what is already gone is not an error, and reports zero.
    assert await graph.store.prune_memories([memory_id]) == 0


async def test_the_graph_reports_what_it_claims_a_memory_mentions(
    graph: GraphFixture,
) -> None:
    """The read the sync's scope expansion depends on, in both directions.

    Postgres cannot answer either question once the rows have moved: an entity that
    lost its last mention, and a merged-away loser, are both unreachable there. The
    graph still holds the edge, and this is how the sync finds it.
    """
    memory_id, entity_id = graph.new_id(), graph.new_id()
    await graph.store.link(
        GraphEdge(
            type=EdgeType.MENTIONS,
            start=GraphNode(GraphLabel.MEMORY, str(memory_id)),
            end=GraphNode(GraphLabel.ENTITY, str(entity_id)),
        )
    )

    by_memory = await graph.store.mention_edges(memory_ids=[memory_id])
    by_entity = await graph.store.mention_edges(entity_ids=[entity_id])

    assert by_memory == [(memory_id, entity_id)]
    assert by_entity == [(memory_id, entity_id)]
    assert await graph.store.mention_edges() == []


async def _minted_projection(graph: GraphFixture) -> GraphProjection:
    """Everything this test wrote, read back through the port.

    Filtered to the fixture's own ids, because `all_nodes` reads the whole database
    and Community Edition has only the one.
    """
    whole, foreign = await graph_verify.read_graph(graph.store)
    assert not foreign, f"a label the projection does not define: {foreign}"
    # `GraphFixture.minted` records ids as strings, for the Cypher its teardown
    # runs. Parsed back, because everything below compares UUIDs.
    minted = {UUID(value) for value in graph.minted}
    return GraphProjection(
        sources=tuple(
            node for node in whole.sources if node.source_id in minted
        ),
        memories=tuple(
            node for node in whole.memories if node.memory_id in minted
        ),
        entities=tuple(
            node for node in whole.entities if node.entity_id in minted
        ),
        edges=tuple(
            edge
            for edge in whole.edges
            if UUID(edge.start.key) in minted and UUID(edge.end.key) in minted
        ),
    )


async def test_two_predicates_between_one_pair_are_two_edges(
    graph: GraphFixture,
) -> None:
    """The predicate is part of a relationship's identity, not a label on it.

    Neo4j merges one relationship per (type, start, end), so before
    `GraphEdge.identity` existed, "a uses b" and "a depends_on b" — both real claims
    in the corpus — collapsed into a single `RELATES_TO` carrying whichever
    predicate the projection happened to write last. Nothing failed. The projection
    reported 25 edges and the graph held 24, and which of the two survived depended
    on the order Postgres returned the rows in.

    Against a real Neo4j because it is a claim about what `MERGE` treats as one
    relationship, which no fake can establish.
    """
    subject, obj = graph.new_id(), graph.new_id()
    for predicate in (Predicate.USES, Predicate.DEPENDS_ON):
        await graph.store.link(
            GraphEdge(
                type=EdgeType.RELATES_TO,
                start=GraphNode(GraphLabel.ENTITY, str(subject)),
                end=GraphNode(GraphLabel.ENTITY, str(obj)),
                properties={
                    "predicate": predicate.value,
                    "assertion_count": 1,
                    "confidence": 0.9,
                },
                identity=("predicate",),
            )
        )

    edges = (await _minted_projection(graph)).edges

    assert len(edges) == 2
    assert {edge.properties["predicate"] for edge in edges} == {
        Predicate.USES.value,
        Predicate.DEPENDS_ON.value,
    }


async def test_an_identity_property_outside_the_allowlist_is_refused(
    graph: GraphFixture,
) -> None:
    """These names are interpolated into Cypher, so the set has to be closed.

    A property name cannot be a bound parameter any more than a label can, which is
    what makes `EDGE_IDENTITY_PROPERTIES` a guard rather than a convention.
    """
    edge = GraphEdge(
        type=EdgeType.RELATES_TO,
        start=GraphNode(GraphLabel.ENTITY, str(graph.new_id())),
        end=GraphNode(GraphLabel.ENTITY, str(graph.new_id())),
        properties={"confidence": 0.5},
        identity=("confidence",),
    )
    with pytest.raises(UnknownGraphLabel, match="not an edge identity property"):
        await graph.store.link(edge)


async def test_an_identity_property_the_edge_does_not_carry_is_refused(
    graph: GraphFixture,
) -> None:
    """Merging on a null would collapse every edge that omitted it into one."""
    edge = GraphEdge(
        type=EdgeType.RELATES_TO,
        start=GraphNode(GraphLabel.ENTITY, str(graph.new_id())),
        end=GraphNode(GraphLabel.ENTITY, str(graph.new_id())),
        properties={"assertion_count": 1},
        identity=("predicate",),
    )
    with pytest.raises(UnknownGraphLabel, match="does not carry it"):
        await graph.store.link(edge)


# --------------------------------------------------------------------------
# The traversal M3.5 expands along
#
# `test_graph_expand.py` asserts what the expansion *does* with a traversal
# against `InMemoryGraphStore`, whose walk is a breadth-first reimplementation of
# the Cypher below. That is only legitimate if the two agree, and this is where
# that is checked: same graph, same seeds, same depth, same reached set.
# --------------------------------------------------------------------------


async def test_the_traversal_reaches_what_the_fake_reaches(graph: GraphFixture) -> None:
    """Cypher and the test double, over one graph, must return the same memories.

    Not the same *rows* — Neo4j returns one row per path and the fake returns one
    per newly-visited entity, so a densely-connected pair legitimately produces
    different counts. What has to match is the answer the expansion consumes: which
    memories were reached, through which entity, at what hop distance.
    """
    fixture = _co_mention_fixture(graph)
    await graph_projection.write(graph.store, fixture.projection)
    fake = InMemoryGraphStore()
    await graph_projection.write(fake, fixture.projection)

    minted = {UUID(value) for value in graph.minted}
    real = await graph.store.reach([fixture.seed_entity], depth=2, limit=200)
    faked = await fake.reach([fixture.seed_entity], depth=2, limit=200)

    def summarise(rows: list[GraphReach]) -> set[tuple[str, str, int]]:
        return {
            (str(row.memory_id), str(row.entity_id), row.hops)
            for row in rows
            # `all_nodes` is the whole database, and so is a traversal: filtered to
            # this test's own ids for the reason `_minted_projection` is.
            if row.memory_id in minted and row.entity_id in minted
        }

    assert summarise(real) == summarise(faked)
    assert summarise(real), "a traversal that reached nothing would prove nothing"


async def test_a_hub_in_the_middle_of_a_path_is_excluded(graph: GraphFixture) -> None:
    """Suppression applies to every node a path crosses, not to its endpoints.

    A hub excluded only as a destination is still a bridge: at depth 2 a path
    through it connects everything that mentions it to everything else that does.
    The Cypher says `all(node IN nodes(path) ...)` for exactly this, and the null
    guard in that predicate is load-bearing — `Memory` nodes carry no `entity_id`,
    and `null IN $list` is null, which would make the whole clause null and drop
    every path.
    """
    fixture = _co_mention_fixture(graph)
    await graph_projection.write(graph.store, fixture.projection)
    minted = {UUID(value) for value in graph.minted}

    without = await graph.store.reach([fixture.seed_entity], depth=2, limit=200)
    suppressed = await graph.store.reach(
        [fixture.seed_entity],
        depth=2,
        exclude_entity_ids=[fixture.bridge_entity],
        limit=200,
    )

    reached_before = {row.memory_id for row in without if row.memory_id in minted}
    reached_after = {row.memory_id for row in suppressed if row.memory_id in minted}

    assert fixture.far_memory in reached_before
    assert fixture.far_memory not in reached_after, "the bridge was the only route"
    assert fixture.near_memory in reached_after, "and the direct route survives"


@dataclass(frozen=True, slots=True)
class _CoMentionFixture:
    projection: GraphProjection
    seed_entity: UUID
    bridge_entity: UUID
    near_memory: UUID
    far_memory: UUID


def _co_mention_fixture(graph: GraphFixture) -> _CoMentionFixture:
    """Three memories chained by two shared entities.

        near --[seed, bridge]--> ...   far --[bridge]--> ...

    `near` mentions both entities, so it is reachable from the seed in one hop;
    `far` mentions only the bridge, so the only route to it runs through a node
    that hub suppression can remove.
    """
    seed_entity, bridge_entity = graph.new_id(), graph.new_id()
    origin, near, far = graph.new_id(), graph.new_id(), graph.new_id()

    def mention(memory_id: UUID, entity_id: UUID) -> GraphEdge:
        return GraphEdge(
            type=EdgeType.MENTIONS,
            start=GraphNode(GraphLabel.MEMORY, str(memory_id)),
            end=GraphNode(GraphLabel.ENTITY, str(entity_id)),
            properties={"mentions": 1, "chunk_ordinal": 0},
        )

    projection = GraphProjection(
        memories=tuple(
            MemoryNode(memory_id, f"notes/{name}.md", MemoryKind.NOTE, OCCURRED_AT)
            for memory_id, name in ((origin, "origin"), (near, "near"), (far, "far"))
        ),
        entities=(
            EntityNode(seed_entity, "SKIP LOCKED", "skip locked", "concept", 0.9),
            EntityNode(bridge_entity, "worker", "worker", "concept", 0.9),
        ),
        edges=(
            mention(origin, seed_entity),
            mention(near, seed_entity),
            mention(near, bridge_entity),
            mention(far, bridge_entity),
        ),
    )
    return _CoMentionFixture(
        projection=projection,
        seed_entity=seed_entity,
        bridge_entity=bridge_entity,
        near_memory=near,
        far_memory=far,
    )
