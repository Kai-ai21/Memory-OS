"""The Neo4j adapter behind `GraphStore`.

Three things here are load-bearing and none of them are obvious:

**Every write is a `MERGE`.** `CREATE` would make the projection un-rebuildable —
the second run of a rebuild would double every node — and the graph exists on the
premise that it can be thrown away and reconstructed. `MERGE` matches on the
identity property alone and `SET`s the rest, so re-running a rebuild converges
rather than accumulates.

**Nothing connects until something is asked of it.** The driver object is cheap
and does no I/O, and the schema is applied on first use rather than at
construction. That is what lets `Container.build` wire a graph store on a machine
with no Neo4j running and leave every Phase 1 and Phase 2 code path working — the
failure surfaces when somebody touches the graph, and nowhere else.

**The three queries that are not constants are checked at runtime.** Cypher
cannot parameterise a variable-length bound, a label, or a relationship type, so
those are formatted in. The driver's signatures ask for `LiteralString`, which
looks like it makes exactly this a type error — but mypy does not enforce
`LiteralString` against an f-string built from `str` parts, so leaning on it
here would be leaning on a check that does not run. The guarantee is therefore
made where it actually holds: `_checked_label`, `_checked_edge` and
`_checked_depth` reject anything that is not an enum member or an int inside a
fixed range, so the set of statements this module can emit is finite and no
caller value ever reaches a query string.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.graph import Node, Path, Relationship

from memoryos.adapters.graph.schema import (
    VERSION_LABEL,
    apply_schema,
    identity_property,
    read_schema_version,
)
from memoryos.application.ports import (
    EntityNode,
    GraphEdge,
    GraphNode,
    GraphPath,
    MemoryNode,
    SourceNode,
)
from memoryos.domain.values import EdgeType, GraphLabel

logger = structlog.get_logger(__name__)

# The ceiling on `neighbours(depth=...)`. Not a tidiness limit: an undirected
# variable-length match fans out with the branching factor, so depth 8 on a
# well-connected entity is not eight times the work of depth 1, it is
# potentially the whole graph. A bound that is checked is also what lets the
# depth be formatted into the query at all.
MAX_DEPTH = 6

# Every label a path may legitimately walk through, by name. `SchemaVersion` is
# not among them — it is disconnected, so a traversal cannot reach it, and it is
# not part of the projection.
_LABELS_BY_NAME: dict[str, GraphLabel] = {label.value: label for label in GraphLabel}
_EDGES_BY_NAME: dict[str, EdgeType] = {edge.value: edge for edge in EdgeType}

# The driver defaults to 30 seconds to open a connection and 60 to take one from
# the pool, which are reasonable for a batch job and wrong for the readiness
# probe that now calls `verify`. A refused connection fails instantly either way;
# these bound the case that does not — a host that accepts the packet and never
# answers, where the default would hang a health check for half a minute and an
# orchestrator would call it a timeout rather than a degraded graph.
CONNECTION_TIMEOUT_SECONDS = 5.0
ACQUISITION_TIMEOUT_SECONDS = 10.0
# Retries inside a transaction function, which is what `execute_query` runs.
# Kept well above the connect timeout so a brief leader re-election is ridden
# out rather than surfaced.
MAX_RETRY_SECONDS = 15.0


class UnknownGraphLabel(RuntimeError):
    """A traversal walked through a node this schema does not define.

    Loud rather than skipped, and the reasoning is the same as
    `UnreplayableEvent`'s: the graph is written only by the rebuild, through this
    adapter, using the labels in `GraphLabel`. A node outside that set means
    something wrote to Neo4j that was not supposed to, and a traversal quietly
    dropping it would hide exactly the thing worth knowing.
    """


class Neo4jGraphStore:
    """`GraphStore` over the official async driver.

    One driver per instance, holding its own connection pool — which is why the
    container builds one and hands it round rather than letting each caller make
    its own. A driver per use case would mean a pool per use case.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str | None = None,
    ) -> None:
        # Constructing a driver validates the URI and allocates a pool. It opens
        # nothing: the first connection is made by the first query, which is what
        # keeps an unreachable Neo4j out of everything that does not use it.
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_timeout=CONNECTION_TIMEOUT_SECONDS,
            connection_acquisition_timeout=ACQUISITION_TIMEOUT_SECONDS,
            max_transaction_retry_time=MAX_RETRY_SECONDS,
            # The schema is re-applied on first use in every process, and every
            # `IF NOT EXISTS` that finds its constraint already there emits an
            # INFORMATION notification. That is four multi-line notices per run
            # reporting that idempotency worked, which drowns the output of any
            # command that touches the graph. Warnings and above still surface.
            notifications_min_severity="WARNING",
        )
        self._database = database
        self._uri = uri
        self._schema_ready = False
        # Without the lock, N coroutines racing to the first query all apply the
        # schema. The statements are idempotent so the result would be correct,
        # but it is N round trips and N chances to hit a concurrent-schema-change
        # error from the server.
        self._schema_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection and schema
    # ------------------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Apply the schema once per store instance.

        Called at the head of every operation rather than at construction, so
        "on connect" means the first real use. Cached on success only — a failed
        attempt leaves the flag down so the next call retries, which is what
        makes a Neo4j that was down at startup usable once it comes back without
        a restart of this process.
        """
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            await apply_schema(self._driver, database=self._database)
            self._schema_ready = True

    async def verify(self) -> None:
        """Raise if the database cannot be reached. Does not apply the schema.

        Separate from `ensure_schema` because readiness and diagnostics want to
        report on a database without writing to it — a health check that applied
        a schema as a side effect would be a health check that changes the thing
        it is measuring.
        """
        await self._driver.verify_connectivity()

    async def schema_version(self) -> int | None:
        return await read_schema_version(self._driver, database=self._database)

    async def close(self) -> None:
        await self._driver.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def upsert_memory(self, node: MemoryNode) -> None:
        """Project a memory into the graph. Identity only; content stays in Postgres."""
        await self.ensure_schema()
        await self._driver.execute_query(
            "MERGE (m:Memory {memory_id: $memory_id}) "
            "SET m.external_key = $external_key, m.kind = $kind, "
            "    m.occurred_at = $occurred_at",
            {
                # Neo4j has no UUID type. Stringified rather than stored as bytes
                # so that a person reading the browser at :7474 sees the same id
                # they would see in Postgres.
                "memory_id": str(node.memory_id),
                "external_key": node.external_key,
                "kind": node.kind.value,
                "occurred_at": node.occurred_at,
            },
            database_=self._database,
        )

    async def upsert_entity(self, node: EntityNode) -> None:
        await self.ensure_schema()
        await self._driver.execute_query(
            "MERGE (e:Entity {entity_id: $entity_id}) "
            "SET e.name = $name, e.canonical_name = $canonical_name, "
            "    e.type = $type, e.confidence = $confidence",
            {
                "entity_id": str(node.entity_id),
                "name": node.name,
                "canonical_name": node.canonical_name,
                "type": node.type,
                "confidence": node.confidence,
            },
            database_=self._database,
        )

    async def upsert_source(self, node: SourceNode) -> None:
        await self.ensure_schema()
        await self._driver.execute_query(
            "MERGE (s:Source {source_id: $source_id}) "
            "SET s.name = $name, s.kind = $kind",
            {
                "source_id": str(node.source_id),
                "name": node.name,
                "kind": node.kind,
            },
            database_=self._database,
        )

    async def link(self, edge: GraphEdge) -> None:
        """Relate two nodes, creating either endpoint if it is not there yet.

        The endpoints are merged rather than matched, and that is deliberate: it
        makes `link` independent of the order the rebuild happens to run in. A
        `MATCH` would silently do nothing when an endpoint has not been written
        yet — the worst failure available, because it produces a graph that is
        merely missing edges and reports success.

        The cost is that an endpoint may exist carrying only its identity
        property until its own upsert lands. `doctor`'s per-label counts are
        where that shows up if it never does.
        """
        await self.ensure_schema()
        await self._driver.execute_query(
            self._link_statement(edge),
            {
                "start_key": edge.start.key,
                "end_key": edge.end.key,
                "properties": dict(edge.properties),
            },
            database_=self._database,
        )

    @staticmethod
    def _link_statement(edge: GraphEdge) -> str:
        """Assemble the `MERGE`, whose labels and relationship type cannot be bound.

        Cypher parameters can carry values but not structure: `MERGE (n:$label)`
        and `-[:$type]->` are both syntax errors before Neo4j 5.26 and remain a
        different feature after it. So these three fragments are formatted in.

        That is safe here for a reason worth stating rather than assuming: every
        interpolated fragment is the `.value` of a `GraphLabel` or `EdgeType`
        member — checked, at runtime, immediately below — so the set of
        statements this function can produce is finite and enumerable. No caller
        input reaches the string. Caller input is `$start_key`, `$end_key` and
        `$properties`, all of which are bound.
        """
        start_label = _checked_label(edge.start.label)
        end_label = _checked_label(edge.end.label)
        return (
            f"MERGE (a:{start_label} "
            f"{{{identity_property(edge.start.label)}: $start_key}}) "
            f"MERGE (b:{end_label} "
            f"{{{identity_property(edge.end.label)}: $end_key}}) "
            f"MERGE (a)-[r:{_checked_edge(edge.type)}]->(b) "
            f"SET r += $properties"
        )

    async def clear(self) -> None:
        """Delete the projection, keeping the constraints and the version marker.

        The constraints survive on their own — they are schema, and `DETACH
        DELETE` touches data. The version node is data and would not, so it is
        excluded explicitly: wiping it would leave a rebuilt graph reporting "no
        schema ever applied", which is the one state `doctor` uses to tell a
        fresh database from a stale one.

        One transaction, which is correct for a graph that fits in memory and
        will need `CALL { ... } IN TRANSACTIONS` when it does not. At M3.0 the
        graph is empty by construction; the note is here so the limit is found
        by reading rather than by running out of heap.
        """
        await self.ensure_schema()
        await self._driver.execute_query(
            f"MATCH (n) WHERE NOT n:{VERSION_LABEL} DETACH DELETE n",
            database_=self._database,
        )
        logger.info("graph.cleared", uri=self._uri)

    async def prune_memories(self, memory_ids: Sequence[UUID]) -> int:
        """See the port. One statement for the whole batch, not one per id."""
        return await self._prune(GraphLabel.MEMORY, memory_ids)

    async def prune_entities(self, entity_ids: Sequence[UUID]) -> int:
        return await self._prune(GraphLabel.ENTITY, entity_ids)

    async def _prune(self, label: GraphLabel, ids: Sequence[UUID]) -> int:
        if not ids:
            # An empty `IN []` is a legal query that matches nothing, so this is
            # a round trip saved rather than a correctness guard.
            return 0
        await self.ensure_schema()
        checked = _checked_label(label)
        key = identity_property(label)
        result = await self._driver.execute_query(
            # `count` before the delete: after `DETACH DELETE` there is nothing
            # left to count, and the caller needs to know whether the graph
            # actually held what it was asked to remove.
            f"MATCH (n:{checked}) WHERE n.{key} IN $ids "
            f"WITH collect(n) AS found "
            f"CALL {{ WITH found UNWIND found AS n DETACH DELETE n }} "
            f"RETURN size(found) AS removed",
            {"ids": [str(value) for value in ids]},
            database_=self._database,
        )
        removed = int(result.records[0]["removed"]) if result.records else 0
        logger.debug("graph.pruned", label=checked, asked=len(ids), removed=removed)
        return removed

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def memories_mentioning(
        self, entity_ids: Sequence[UUID]
    ) -> list[tuple[UUID, UUID]]:
        if not entity_ids:
            return []
        await self.ensure_schema()
        result = await self._driver.execute_query(
            "MATCH (m:Memory)-[:MENTIONS]->(e:Entity) "
            "WHERE e.entity_id IN $ids "
            "RETURN DISTINCT m.memory_id AS memory_id, e.entity_id AS entity_id",
            {"ids": [str(value) for value in entity_ids]},
            database_=self._database,
        )
        return [
            (UUID(record["memory_id"]), UUID(record["entity_id"]))
            for record in result.records
        ]

    async def all_nodes(self) -> list[GraphNode]:
        await self.ensure_schema()
        result = await self._driver.execute_query(
            f"MATCH (n) WHERE NOT n:{VERSION_LABEL} RETURN n",
            database_=self._database,
        )
        return [_node_of(record["n"]) for record in result.records]

    async def all_edges(self) -> list[GraphEdge]:
        await self.ensure_schema()
        result = await self._driver.execute_query(
            f"MATCH (a)-[r]->(b) "
            f"WHERE NOT a:{VERSION_LABEL} AND NOT b:{VERSION_LABEL} "
            f"RETURN a, r, b",
            database_=self._database,
        )
        return [
            GraphEdge(
                type=_edge_of(record["r"]),
                # Identity only, deliberately: see the port. `_node_of` would
                # attach every property of both endpoints to every edge.
                start=_identity_of(record["a"]),
                end=_identity_of(record["b"]),
                properties=_properties_of(record["r"]),
            )
            for record in result.records
        ]

    async def neighbours(
        self, entity_id: UUID, *, depth: int = 2, limit: int = 50
    ) -> list[GraphPath]:
        """Paths out from an entity, shortest first. See the port for semantics."""
        await self.ensure_schema()
        bounded = _checked_depth(depth)
        if limit <= 0:
            return []

        result = await self._driver.execute_query(
            # `$entity_id` and `$limit` are bound; only the depth bound is
            # formatted, because Cypher does not accept a parameter inside
            # `[*1..n]` — the planner needs the bound at plan time. It is an int
            # checked against `MAX_DEPTH` by `_checked_depth` immediately above.
            "MATCH path = (start:Entity {entity_id: $entity_id})"
            f"-[*1..{bounded}]-(other) "
            "RETURN path ORDER BY length(path) LIMIT $limit",
            {"entity_id": str(entity_id), "limit": limit},
            database_=self._database,
        )
        return [_path_of(record["path"]) for record in result.records]

    async def counts_by_label(self) -> dict[str, int]:
        """Node counts per label, for `doctor`.

        A scan, and knowingly so. The cheap form counts each *known* label
        through its index, which would report nothing at all for a label that is
        not supposed to be there — and a node with an unexpected label is
        precisely what a diagnostic should surface, since the only writer of this
        graph is meant to be the rebuild.
        """
        await self.verify()
        result = await self._driver.execute_query(
            f"MATCH (n) WHERE NOT n:{VERSION_LABEL} "
            "UNWIND labels(n) AS label "
            "RETURN label, count(*) AS count ORDER BY label",
            database_=self._database,
        )
        return {record["label"]: int(record["count"]) for record in result.records}


# ----------------------------------------------------------------------
# Translation between the driver's types and the port's
# ----------------------------------------------------------------------


def _checked_label(label: GraphLabel) -> str:
    """The label's Cypher name, having confirmed it is a member of the enum.

    A redundant check under a type checker and not under one at runtime, where
    `GraphLabel` is a `StrEnum` and any string could be passed in its place.
    This is the guard that keeps `_link_statement`'s claim — that only enum
    values are interpolated — true rather than merely intended.
    """
    if not isinstance(label, GraphLabel):
        raise UnknownGraphLabel(f"{label!r} is not a GraphLabel")
    return label.value


def _checked_edge(edge: EdgeType) -> str:
    if not isinstance(edge, EdgeType):
        raise UnknownGraphLabel(f"{edge!r} is not an EdgeType")
    return edge.value


def _checked_depth(depth: int) -> int:
    if not isinstance(depth, int) or isinstance(depth, bool):
        raise TypeError(f"depth must be an int, got {depth!r}")
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(
            f"depth must be between 1 and {MAX_DEPTH}, got {depth}. An undirected "
            f"variable-length traversal fans out with the branching factor, so a "
            f"deeper bound is not a longer query, it is potentially the whole graph."
        )
    return depth


def _path_of(path: Path) -> GraphPath:
    return GraphPath(
        nodes=tuple(_node_of(node) for node in path.nodes),
        edges=tuple(_edge_of(relationship) for relationship in path.relationships),
    )


def _node_of(node: Node) -> GraphNode:
    label = _sole_label(node)
    key = node.get(identity_property(label))
    return GraphNode(
        label=label,
        key=str(key) if key is not None else "",
        properties=_properties_of(node),
    )


def _identity_of(node: Node) -> GraphNode:
    """The node's label and key, without reading its properties."""
    label = _sole_label(node)
    key = node.get(identity_property(label))
    return GraphNode(label=label, key=str(key) if key is not None else "")


def _sole_label(node: Node) -> GraphLabel:
    known = [_LABELS_BY_NAME[name] for name in node.labels if name in _LABELS_BY_NAME]
    if not known:
        raise UnknownGraphLabel(
            f"node {node.element_id} carries labels {sorted(node.labels)}, none of "
            f"which are in GraphLabel. Nothing but the rebuild should be writing "
            f"to this graph; a node outside the schema means something did."
        )
    # More than one known label is not currently produced by anything here, and
    # picking the first would be a silent choice. Sorted, so at least the choice
    # is the same one every time rather than dependent on set ordering.
    return sorted(known)[0]


def _edge_of(relationship: Relationship) -> EdgeType:
    edge = _EDGES_BY_NAME.get(relationship.type)
    if edge is None:
        raise UnknownGraphLabel(
            f"relationship {relationship.element_id} has type "
            f"{relationship.type!r}, which is not in EdgeType"
        )
    return edge


def _properties_of(node: Mapping[str, Any]) -> dict[str, Any]:
    """Node properties as plain Python.

    The driver returns its own temporal types, which carry nanosecond precision
    Python's `datetime` does not have. Converting at the boundary means nothing
    above this module has to know a `neo4j.time.DateTime` from a `datetime` —
    the alternative being that it finds out when one fails to compare against
    the other.
    """
    return {key: _native(value) for key, value in node.items()}


def _native(value: Any) -> Any:
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value
