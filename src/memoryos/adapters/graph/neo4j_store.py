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

from memoryos.adapters.db.scoping import CURRENT_USER_ID
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
    GraphReach,
    MemoryNode,
    SourceNode,
)
from memoryos.domain.values import EDGE_IDENTITY_PROPERTIES, EdgeType, GraphLabel

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


def _scope() -> str:
    """The current user, as Neo4j will store it, or a value that matches nothing.

    **Neo4j Community Edition has no row-level security and no second
    database**, so isolation here is a property every query has to carry rather
    than something the server enforces. This is the value it carries.

    Unset resolves to a sentinel rather than to NULL or to an empty string,
    which matters: `MATCH (n {user_id: null})` in Cypher does not mean "no rows",
    it means the property is unconstrained. A sentinel that no account can ever
    have is what makes an unscoped connection see nothing instead of everything
    — the same fail-closed shape as the Postgres side, reached differently.
    """
    user_id = CURRENT_USER_ID.get()
    return str(user_id) if user_id is not None else "no-such-user"


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
            "MERGE (m:Memory {memory_id: $memory_id, user_id: $user_id}) "
            "SET m.external_key = $external_key, m.kind = $kind, "
            "    m.occurred_at = $occurred_at",
            {
                # Neo4j has no UUID type. Stringified rather than stored as bytes
                # so that a person reading the browser at :7474 sees the same id
                # they would see in Postgres.
                "memory_id": str(node.memory_id),
                "user_id": _scope(),
                "external_key": node.external_key,
                "kind": node.kind.value,
                "occurred_at": node.occurred_at,
            },
            database_=self._database,
        )

    async def upsert_entity(self, node: EntityNode) -> None:
        await self.ensure_schema()
        await self._driver.execute_query(
            "MERGE (e:Entity {entity_id: $entity_id, user_id: $user_id}) "
            "SET e.name = $name, e.canonical_name = $canonical_name, "
            "    e.type = $type, e.confidence = $confidence",
            {
                "entity_id": str(node.entity_id),
                "user_id": _scope(),
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
            "MERGE (s:Source {source_id: $source_id, user_id: $user_id}) "
            "SET s.name = $name, s.kind = $kind",
            {
                "source_id": str(node.source_id),
                "user_id": _scope(),
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
                "user_id": _scope(),
                "start_key": edge.start.key,
                "end_key": edge.end.key,
                "properties": dict(edge.properties),
                # Bound under a prefixed name, so an identity property called
                # `properties` could not shadow the map above.
                **{
                    f"identity_{name}": edge.properties.get(name)
                    for name in _checked_identity(edge)
                },
            },
            database_=self._database,
        )

    @staticmethod
    def _link_statement(edge: GraphEdge) -> str:
        """Assemble the `MERGE`, whose labels and relationship type cannot be bound.

        Cypher parameters can carry values but not structure: `MERGE (n:$label)`,
        `-[:$type]->` and `{$name: $value}` are all syntax errors before Neo4j 5.26
        and remain a different feature after it. So those fragments are formatted
        in.

        That is safe here for a reason worth stating rather than assuming: every
        interpolated fragment is the `.value` of a `GraphLabel` or `EdgeType`
        member, or a name from `EDGE_IDENTITY_PROPERTIES` — each checked, at
        runtime, immediately below — so the set of statements this function can
        produce is finite and enumerable. No caller input reaches the string.
        Caller input is `$start_key`, `$end_key`, `$properties` and the
        `$identity_*` bindings, all of which are bound.

        The identity properties go *inside* the relationship pattern. See
        `ports.GraphEdge.identity` for what that fixes.
        """
        start_label = _checked_label(edge.start.label)
        end_label = _checked_label(edge.end.label)
        names = _checked_identity(edge)
        # Omitted entirely rather than emitted as an empty map: `-[r:T {}]->` is
        # accepted by the parser and there is no reason to make a reader of the
        # query log wonder whether it means something.
        identity = (
            " {" + ", ".join(f"{name}: $identity_{name}" for name in names) + "}"
            if names
            else ""
        )
        # `user_id` is part of both endpoint identities, which is what makes a
        # cross-user edge unrepresentable rather than merely unwritten: `MERGE`
        # on an identity that includes the owner cannot attach to somebody
        # else's node, it creates this user's own.
        return (
            f"MERGE (a:{start_label} "
            f"{{{identity_property(edge.start.label)}: $start_key, user_id: $user_id}}) "
            f"MERGE (b:{end_label} "
            f"{{{identity_property(edge.end.label)}: $end_key, user_id: $user_id}}) "
            f"MERGE (a)-[r:{_checked_edge(edge.type)}{identity}]->(b) "
            f"SET r += $properties, r.user_id = $user_id"
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
            f"MATCH (n) WHERE NOT n:{VERSION_LABEL} AND n.user_id = $user_id DETACH DELETE n",
            {"user_id": _scope()},
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
            # Counted before the delete: after `DETACH DELETE` there is nothing
            # left to count, and the caller needs to know whether the graph
            # actually held what it was asked to remove.
            #
            # `FOREACH` rather than a `CALL` subquery, which is what this was and
            # which the server deprecates without a variable scope clause — a
            # clause that does not exist before 5.23. `FOREACH` has been the
            # spelling for a delete over a collected list throughout.
            f"MATCH (n:{checked}) WHERE n.{key} IN $ids AND n.user_id = $user_id "
            f"WITH collect(n) AS found, count(n) AS removed "
            f"FOREACH (node IN found | DETACH DELETE node) "
            f"RETURN removed",
            {"ids": [str(value) for value in ids], "user_id": _scope()},
            database_=self._database,
        )
        removed = int(result.records[0]["removed"]) if result.records else 0
        logger.debug("graph.pruned", label=checked, asked=len(ids), removed=removed)
        return removed

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def mention_edges(
        self,
        *,
        memory_ids: Sequence[UUID] = (),
        entity_ids: Sequence[UUID] = (),
    ) -> list[tuple[UUID, UUID]]:
        if not memory_ids and not entity_ids:
            return []
        await self.ensure_schema()
        result = await self._driver.execute_query(
            "MATCH (m:Memory)-[:MENTIONS]->(e:Entity) "
            "WHERE m.user_id = $user_id "
            "AND (m.memory_id IN $memories OR e.entity_id IN $entities) "
            "RETURN DISTINCT m.memory_id AS memory_id, e.entity_id AS entity_id",
            {
                "memories": [str(value) for value in memory_ids],
                "entities": [str(value) for value in entity_ids],
                "user_id": _scope(),
            },
            database_=self._database,
        )
        return [
            (UUID(record["memory_id"]), UUID(record["entity_id"]))
            for record in result.records
        ]

    async def all_nodes(self) -> list[GraphNode]:
        await self.ensure_schema()
        result = await self._driver.execute_query(
            f"MATCH (n) WHERE NOT n:{VERSION_LABEL} AND n.user_id = $user_id RETURN n",
            {"user_id": _scope()},
            database_=self._database,
        )
        return [_node_of(record["n"]) for record in result.records]

    async def all_edges(self) -> list[GraphEdge]:
        await self.ensure_schema()
        result = await self._driver.execute_query(
            f"MATCH (a)-[r]->(b) "
            f"WHERE NOT a:{VERSION_LABEL} AND NOT b:{VERSION_LABEL} "
            f"AND a.user_id = $user_id AND b.user_id = $user_id "
            f"RETURN a, r, b",
            {"user_id": _scope()},
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
            "MATCH path = (start:Entity {entity_id: $entity_id, user_id: $user_id})"
            f"-[*1..{bounded}]-(other) "
            "RETURN path ORDER BY length(path) LIMIT $limit",
            {"entity_id": str(entity_id), "limit": limit, "user_id": _scope()},
            database_=self._database,
        )
        return [_path_of(record["path"]) for record in result.records]

    async def reach(
        self,
        seed_entity_ids: Sequence[UUID],
        *,
        depth: int = 2,
        exclude_entity_ids: Sequence[UUID] = (),
        limit: int = 200,
    ) -> list[GraphReach]:
        """See the port. One variable-length traversal, hubs excluded inside it."""
        if not seed_entity_ids or limit <= 0:
            return []
        # Entity hops to graph hops. An entity reaches another in one graph hop
        # through `RELATES_TO` or two through a memory that mentions both, so the
        # bound is twice the requested depth — and `_checked_depth` is applied to
        # the graph bound, because that is what the planner will actually walk.
        bounded = _checked_depth(_checked_entity_depth(depth) * 2)

        # Two queries, and the first one is not an optimisation.
        #
        # **Cypher cannot express "a path from an entity back to itself" here, and
        # that is the most valuable expansion there is.** Neo4j's relationship
        # uniqueness rule forbids reusing a relationship within one path, so
        # `(seed)-[MENTIONS]-(m)-[MENTIONS]-(seed)` does not match: the two hops
        # would be the same edge. The route back only exists through a *different*
        # memory, which is a longer and weaker connection. So "another memory
        # mentions the same thing retrieval just found" — a direct co-mention, and
        # the strongest signal the graph has — is unreachable by traversal and is
        # matched directly below.
        #
        # Found by a test comparing this against the in-memory store's
        # breadth-first walk, which happily revisits its start.
        direct = await self._driver.execute_query(
            "MATCH (m:Memory)-[mention:MENTIONS]->(seed:Entity) "
            "WHERE m.user_id = $user_id AND seed.entity_id IN $seeds "
            "RETURN m.memory_id AS memory_id, "
            "       mention.chunk_ordinal AS chunk_ordinal, "
            "       seed.entity_id AS entity_id, "
            # Two, matching what the graph distance would be if the path existed:
            # entity to memory to entity. Reported the same way so that `_score`
            # weights a co-mention against a traversed hop on one scale.
            "       2 AS hops, "
            "       [seed.name] AS route "
            "ORDER BY memory_id, entity_id "
            "LIMIT $limit",
            {
                "seeds": [str(value) for value in seed_entity_ids],
                "user_id": _scope(),
                "limit": limit,
            },
            database_=self._database,
        )

        result = await self._driver.execute_query(
            # Only the depth bound is formatted in; everything else is a bound
            # parameter. Cypher will not accept a parameter inside `[*1..n]`
            # because the planner needs the bound at plan time.
            #
            # `MENTIONS|RELATES_TO` rather than an unrestricted walk: `FROM_SOURCE`
            # would make every memory of one source two hops from every other,
            # which is a connection nobody wrote down.
            "MATCH path = (seed:Entity {user_id: $user_id})"
            f"-[:MENTIONS|RELATES_TO*1..{bounded}]-(target:Entity) "
            "WHERE seed.entity_id IN $seeds "
            # The seed itself is handled by the query above, where it can be
            # matched at all. Excluded here so a longer route back to it does not
            # produce a second, worse-ranked copy of the same connection.
            "  AND target.entity_id <> seed.entity_id "
            # Hub suppression, applied to every node the path crosses rather than
            # to the endpoint alone: a hub in the middle is a bridge, and at this
            # depth a bridge connects everything to everything. The null guard is
            # load-bearing — `Memory` nodes carry no `entity_id`, and `null IN $x`
            # is null, which would make `all(...)` null and drop every path.
            "  AND all(node IN nodes(path) WHERE "
            "        node.entity_id IS NULL OR NOT node.entity_id IN $exclude) "
            "MATCH (m:Memory)-[mention:MENTIONS]->(target) "
            "WHERE m.user_id = $user_id "
            "RETURN m.memory_id AS memory_id, "
            "       mention.chunk_ordinal AS chunk_ordinal, "
            "       target.entity_id AS entity_id, "
            "       length(path) AS hops, "
            "       [node IN nodes(path) WHERE node:Entity | node.name] AS route "
            # Shortest first, which is what makes `limit` a bound on the *best*
            # routes rather than on an arbitrary set of them. Ties broken by id so
            # two runs of one query return the same rows in the same order.
            "ORDER BY hops, memory_id, entity_id "
            "LIMIT $limit",
            {
                "seeds": [str(value) for value in seed_entity_ids],
                "user_id": _scope(),
                "exclude": [str(value) for value in exclude_entity_ids],
                "limit": limit,
            },
            database_=self._database,
        )
        reached = [
            GraphReach(
                memory_id=UUID(record["memory_id"]),
                chunk_ordinal=int(record["chunk_ordinal"] or 0),
                entity_id=UUID(record["entity_id"]),
                hops=int(record["hops"]),
                route=_route_of(record["route"]),
            )
            for record in [*direct.records, *result.records]
        ]
        # Sorted across both queries, so `limit` keeps the shortest routes overall
        # rather than the first query's rows followed by the second's.
        reached.sort(key=lambda row: (row.hops, str(row.memory_id), str(row.entity_id)))
        return reached[:limit]

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
            f"MATCH (n) WHERE NOT n:{VERSION_LABEL} AND n.user_id = $user_id "
            "UNWIND labels(n) AS label "
            "RETURN label, count(*) AS count ORDER BY label",
            {"user_id": _scope()},
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


def _checked_identity(edge: GraphEdge) -> tuple[str, ...]:
    """The identity property names, having confirmed each is one this schema knows.

    The same guarantee `_checked_label` makes, for the same reason: these names are
    interpolated into Cypher because a property name cannot be bound, so the set of
    statements this adapter can emit has to stay finite and enumerable. An
    allowlist rather than a character check, because "looks like an identifier" is
    a weaker claim than "is one of two known names".
    """
    for name in edge.identity:
        if name not in EDGE_IDENTITY_PROPERTIES:
            raise UnknownGraphLabel(
                f"{name!r} is not an edge identity property; the projection may "
                f"only merge on {sorted(EDGE_IDENTITY_PROPERTIES)}"
            )
        if name not in edge.properties:
            raise UnknownGraphLabel(
                f"edge {edge.type.value} declares {name!r} as part of its identity "
                f"but does not carry it. Merging on a null would collapse every "
                f"edge that omitted it into one."
            )
    # Sorted, so two edges built with the same names in a different order produce
    # the same statement and share the driver's query plan cache.
    return tuple(sorted(edge.identity))


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


def _route_of(names: list[object]) -> tuple[str, ...]:
    """The entity names along a path, without a name repeated back to back.

    A route that returns to the entity it started from — the co-mention case — reads
    `SKIP LOCKED -> SKIP LOCKED` verbatim, which tells a reader nothing about why
    the result is there. Collapsed to one name, it says exactly what happened:
    another memory mentions this.
    """
    route: list[str] = []
    for name in names:
        text = str(name)
        if not route or route[-1] != text:
            route.append(text)
    return tuple(route)


def _checked_entity_depth(depth: int) -> int:
    """The entity-hop depth, bounded before it is doubled into a graph bound.

    Separate from `_checked_depth` because the two count different things and the
    error a caller needs to read is about the one they passed. `MAX_DEPTH // 2` is
    the ceiling that falls out of the mapping rather than a second limit to keep in
    step with the first.
    """
    if not isinstance(depth, int) or isinstance(depth, bool):
        raise TypeError(f"depth must be an int, got {depth!r}")
    if not 1 <= depth <= MAX_DEPTH // 2:
        raise ValueError(
            f"entity depth must be between 1 and {MAX_DEPTH // 2}, got {depth}. Each "
            f"entity hop is up to two graph hops — an entity reaches another through "
            f"a relationship or through a memory that mentions both — so the graph "
            f"bound is twice this and {MAX_DEPTH} is the ceiling that makes a "
            f"variable-length match affordable."
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

    **`user_id` is dropped on the way out.** M11.1 puts it on every node so that
    isolation is a property of the data rather than of the queries, but it is
    scoping metadata and not part of the projection: `graph verify` hashes these
    properties against what Postgres says the graph should contain, and Postgres
    describes a memory without saying twice who it belongs to. Leaving it in
    would make every node diverge from its own source of truth.
    """
    return {key: _native(value) for key, value in node.items() if key != "user_id"}


def _native(value: Any) -> Any:
    to_native = getattr(value, "to_native", None)
    return to_native() if callable(to_native) else value
