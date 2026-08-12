"""Does the graph still say what Postgres says it should?

The same requirement `verify-replay` carries, against a different pair of stores:
**it has to be able to fail.** A check that cannot distinguish a correct
projection from a corrupted one is worse than no check, because it teaches you
that the projection is fine.

Two rules, and they are not the same two `verification.py` uses — the difference
is worth stating because it looks like an inconsistency.

**Primary keys, not natural keys.** `verify-replay` compares on natural keys
because a rebuild mints new ids, so an id comparison would fail on every honest
replay. Here the opposite holds: the graph *copies* Postgres' ids by construction,
which is the entire mechanism by which a traversal can join back to a row. An
`Entity` node whose `entity_id` names no entity is therefore a real divergence and
not a rebuild artefact, and comparing on canonical names instead would hide it.

**Content, not counts** — that one is the same, and for the same reason. Counts
match while a node carries a stale name, a merged-away entity is still projected,
or an edge's `assertion_count` predates half its evidence. So every property of
every node and edge goes into a digest, and the digests are reported per type so a
failure says where to look rather than only that something is wrong.

## What a divergence actually means

Postgres wins, always. Nothing here repairs anything: it reports, and
`graph rebuild` is the repair. That is deliberate — a verification that fixed what
it found would leave nobody able to answer how often the projection diverges, and
that number is the only evidence for whether the sync works.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.application import graph_projection
from memoryos.application.graph_projection import GraphProjection
from memoryos.application.ports import (
    EntityNode,
    GraphEdge,
    GraphNode,
    GraphStore,
    MemoryNode,
    SourceNode,
)
from memoryos.domain.values import (
    EDGE_IDENTITY_PROPERTIES,
    EdgeType,
    GraphLabel,
    MemoryKind,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TypeDivergence:
    """What differs for one node label or one relationship type."""

    name: str
    expected: int
    actual: int
    expected_hash: str
    actual_hash: str
    # Present in Postgres and not in the graph, and the other way round, by
    # identity. Names rather than ids where a name exists: an operator reading
    # this needs to know which entity, and a UUID does not tell them.
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    # Present in both, carrying different properties.
    changed: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return self.expected_hash == self.actual_hash

    @property
    def total(self) -> int:
        return len(self.missing) + len(self.unexpected) + len(self.changed)


@dataclass(frozen=True, slots=True)
class GraphDivergence:
    by_type: tuple[TypeDivergence, ...] = ()
    expected_hash: str = ""
    actual_hash: str = ""
    # Nodes carrying a label the projection does not define. Counted separately
    # from `unexpected` because they are a different kind of wrong: an unexpected
    # `Entity` is a stale projection, whereas an unexpected label means something
    # wrote to this database that was never supposed to.
    foreign_labels: dict[str, int] = field(default_factory=dict)

    @property
    def identical(self) -> bool:
        return (
            self.expected_hash == self.actual_hash
            and not self.foreign_labels
        )

    def render(self, *, examples: int = 5) -> str:
        lines = [
            f"{'':<14}{'postgres':>10}{'graph':>10}   hash",
            "",
        ]
        for row in self.by_type:
            mark = "ok  " if row.clean else "FAIL"
            digest = (
                row.expected_hash[:12]
                if row.clean
                else f"{row.expected_hash[:12]} != {row.actual_hash[:12]}"
            )
            lines.append(
                f"[{mark}] {row.name:<14}{row.expected:>10}{row.actual:>10}   {digest}"
            )
            for label, rows in (
                ("only in postgres", row.missing),
                ("only in the graph", row.unexpected),
                ("differs", row.changed),
            ):
                for item in rows[:examples]:
                    lines.append(f"         {label}: {item}")
                if len(rows) > examples:
                    lines.append(f"         ... and {len(rows) - examples} more {label}")

        if self.foreign_labels:
            lines += ["", "labels the projection does not define:"]
            lines += [
                f"  {label}: {count}" for label, count in sorted(self.foreign_labels.items())
            ]
            lines.append(
                "  nothing but the projection should be writing to this graph; a node "
                "outside the schema means something did"
            )

        lines += [
            "",
            f"postgres {self.expected_hash}",
            f"graph    {self.actual_hash}",
        ]
        return "\n".join(lines)


async def verify(
    session_factory: async_sessionmaker[AsyncSession], graph: GraphStore
) -> GraphDivergence:
    """Compare what Postgres implies against what the graph holds."""
    expected = await graph_projection.read(session_factory)
    actual, foreign = await read_graph(graph)
    divergence = compare(expected, actual, foreign_labels=foreign)
    logger.info(
        "graph.verified",
        identical=divergence.identical,
        expected=expected.nodes,
        actual=actual.nodes,
    )
    return divergence


async def read_graph(graph: GraphStore) -> tuple[GraphProjection, dict[str, int]]:
    """The graph's contents, in the same shape Postgres' are read into.

    Translated into `GraphProjection` rather than compared as raw nodes, so that
    the comparison below is between two values of one type. A comparison written
    against two different shapes is a comparison with a translation step inside
    it, and the translation is where a divergence gets normalised away.

    A node whose properties are missing or the wrong type does not raise: it is
    read as whatever is there, so that the diff reports it rather than the read
    failing. A bare `Entity` node — the M3.3 defect — is a node with no `name`,
    and this has to be able to say so.
    """
    nodes = await graph.all_nodes()
    edges = await graph.all_edges()

    sources: list[SourceNode] = []
    memories: list[MemoryNode] = []
    entities: list[EntityNode] = []
    foreign: dict[str, int] = {}

    for node in nodes:
        if node.label is GraphLabel.SOURCE:
            sources.append(
                SourceNode(
                    source_id=_uuid(node, "source_id"),
                    name=_text(node, "name"),
                    kind=_text(node, "kind"),
                )
            )
        elif node.label is GraphLabel.MEMORY:
            memories.append(
                MemoryNode(
                    memory_id=_uuid(node, "memory_id"),
                    external_key=_text(node, "external_key"),
                    kind=_kind(node),
                    occurred_at=node.properties.get("occurred_at"),
                )
            )
        elif node.label is GraphLabel.ENTITY:
            entities.append(
                EntityNode(
                    entity_id=_uuid(node, "entity_id"),
                    name=_text(node, "name"),
                    canonical_name=_text(node, "canonical_name"),
                    type=_text(node, "type"),
                    confidence=float(node.properties.get("confidence") or 0.0),
                )
            )
        else:
            foreign[node.label.value] = foreign.get(node.label.value, 0) + 1

    return (
        GraphProjection(
            sources=tuple(sources),
            memories=tuple(memories),
            entities=tuple(entities),
            edges=tuple(edges),
        ),
        foreign,
    )


def compare(
    expected: GraphProjection,
    actual: GraphProjection,
    *,
    foreign_labels: dict[str, int] | None = None,
) -> GraphDivergence:
    """Every difference, by type, with example rows. Pure."""
    rows = [
        _diff_nodes(
            "Source",
            {str(node.source_id): (node.name, node.kind) for node in expected.sources},
            {str(node.source_id): (node.name, node.kind) for node in actual.sources},
            {str(node.source_id): node.name for node in expected.sources}
            | {str(node.source_id): node.name for node in actual.sources},
        ),
        _diff_nodes(
            "Memory",
            {str(node.memory_id): _memory_row(node) for node in expected.memories},
            {str(node.memory_id): _memory_row(node) for node in actual.memories},
            {str(node.memory_id): node.external_key for node in expected.memories}
            | {str(node.memory_id): node.external_key for node in actual.memories},
        ),
        _diff_nodes(
            "Entity",
            {str(node.entity_id): _entity_row(node) for node in expected.entities},
            {str(node.entity_id): _entity_row(node) for node in actual.entities},
            {str(node.entity_id): f"{node.name} ({node.type})" for node in expected.entities}
            | {str(node.entity_id): f"{node.name} ({node.type})" for node in actual.entities},
        ),
    ]
    rows += [
        _diff_edges(edge_type, expected.edges, actual.edges) for edge_type in EdgeType
    ]

    return GraphDivergence(
        by_type=tuple(rows),
        expected_hash=graph_projection.content_hash(expected),
        actual_hash=graph_projection.content_hash(actual),
        foreign_labels=dict(foreign_labels or {}),
    )


def _memory_row(node: MemoryNode) -> tuple[str, ...]:
    return (
        node.external_key,
        node.kind.value,
        # As text, because the driver returns its own temporal type and Postgres
        # returns a `datetime`. Compared as strings, two equal instants compare
        # equal without either side having to know about the other's clock type.
        "" if node.occurred_at is None else str(node.occurred_at),
    )


def _entity_row(node: EntityNode) -> tuple[str, ...]:
    return (
        node.name,
        node.canonical_name,
        node.type,
        # Six places: Postgres stores a 32-bit REAL and Neo4j a double, so the
        # round trip adds digits that were never in the data. Comparing raw
        # doubles would report every entity as changed.
        f"{round(float(node.confidence), 6):.6f}",
    )


def _diff_nodes(
    name: str,
    expected: dict[str, tuple[str, ...]],
    actual: dict[str, tuple[str, ...]],
    labels: dict[str, str],
) -> TypeDivergence:
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(key for key in expected.keys() & actual.keys() if expected[key] != actual[key])

    return TypeDivergence(
        name=name,
        expected=len(expected),
        actual=len(actual),
        expected_hash=_digest_rows((key, *row) for key, row in sorted(expected.items())),
        actual_hash=_digest_rows((key, *row) for key, row in sorted(actual.items())),
        missing=tuple(_describe(key, labels) for key in missing),
        unexpected=tuple(_describe(key, labels) for key in unexpected),
        changed=tuple(
            f"{_describe(key, labels)}: {expected[key]} -> {actual[key]}" for key in changed
        ),
    )


def _diff_edges(
    edge_type: EdgeType,
    expected: tuple[GraphEdge, ...],
    actual: tuple[GraphEdge, ...],
) -> TypeDivergence:
    left = _edges_by_key(edge_type, expected)
    right = _edges_by_key(edge_type, actual)
    missing = sorted(left.keys() - right.keys())
    unexpected = sorted(right.keys() - left.keys())
    changed = sorted(key for key in left.keys() & right.keys() if left[key] != right[key])

    return TypeDivergence(
        name=edge_type.value,
        expected=len(left),
        actual=len(right),
        expected_hash=_digest_rows((*key, str(row)) for key, row in sorted(left.items())),
        actual_hash=_digest_rows((*key, str(row)) for key, row in sorted(right.items())),
        missing=tuple(_describe_edge(key) for key in missing),
        unexpected=tuple(_describe_edge(key) for key in unexpected),
        changed=tuple(
            f"{_describe_edge(key)}: {left[key]} -> {right[key]}" for key in changed
        ),
    )


def _edges_by_key(
    edge_type: EdgeType, edges: tuple[GraphEdge, ...]
) -> dict[tuple[str, ...], tuple[tuple[str, object], ...]]:
    """Edges of one type, keyed by identity, properties canonicalised.

    Keyed by the endpoints and *not* by the rest of the properties, so that a
    changed `assertion_count` is reported as one edge that differs rather than as
    one missing edge and one unexpected one.

    The identity properties are in the key, because they are what the graph merges
    on: two `RELATES_TO` edges between one pair with different predicates are two
    relationships, and keying on the endpoints alone would compare one of them
    against the other and report both as changed.

    Read from the properties rather than from `GraphEdge.identity`, because an edge
    read back out of the graph does not carry that field — the store returns what a
    relationship *is*, not how it was merged.
    """
    return {
        _edge_key(edge): tuple(
            (key, _comparable(value)) for key, value in sorted(edge.properties.items())
        )
        for edge in edges
        if edge.type is edge_type
    }


def _describe_edge(key: tuple[str, ...]) -> str:
    start, end, *identity = key
    suffix = f" [{', '.join(identity)}]" if identity else ""
    return f"{start} -> {end}{suffix}"


def _edge_key(edge: GraphEdge) -> tuple[str, ...]:
    return (
        edge.start.key,
        edge.end.key,
        *(
            str(edge.properties[name])
            for name in sorted(EDGE_IDENTITY_PROPERTIES)
            if name in edge.properties
        ),
    )


def _comparable(value: object) -> object:
    """A property value in a form both stores agree about.

    Floats to six places for the reason `_entity_row` explains; ints stay ints,
    because `assertion_count` is a count and rounding one would be nonsense.
    """
    if isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return f"{round(value, 6):.6f}"
    return value


def _describe(key: str, labels: dict[str, str]) -> str:
    label = labels.get(key)
    return f"{key} ({label})" if label else key


def _digest_rows(rows: Iterable[tuple[object, ...]]) -> str:
    joined = "\n".join("\x1f".join(str(field) for field in row) for row in rows)
    return hashlib.blake2b(joined.encode(), digest_size=8).hexdigest()


def _uuid(node: GraphNode, name: str) -> UUID:
    value = node.properties.get(name)
    try:
        return UUID(str(value))
    except ValueError:
        # A node whose identity property is not a UUID is a divergence, not a
        # crash. Read as the nil UUID so the diff reports one unexpected node
        # rather than the whole verification failing to run.
        return UUID(int=0)


def _text(node: GraphNode, name: str) -> str:
    value = node.properties.get(name)
    # The empty string, not "None": a node missing a property is what the M3.3
    # bare-node defect looked like, and it has to compare unequal to a real name
    # while still being printable in a diff.
    return "" if value is None else str(value)


def _kind(node: GraphNode) -> MemoryKind:
    try:
        return MemoryKind(str(node.properties.get("kind")))
    except ValueError:
        return MemoryKind.OTHER
