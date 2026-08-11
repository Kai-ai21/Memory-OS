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

from memoryos.adapters.graph.neo4j_store import MAX_DEPTH, Neo4jGraphStore
from memoryos.adapters.graph.schema import (
    SCHEMA_VERSION,
    STATEMENTS,
    apply_schema,
    read_schema_version,
)
from memoryos.application.ports import EntityNode, GraphEdge, GraphNode, MemoryNode
from memoryos.domain.values import EdgeType, GraphLabel, MemoryKind
from tests.integration.conftest import GraphFixture

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
