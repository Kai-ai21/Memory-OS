"""Prove a rebuilt corpus is the same corpus.

Two rules decide everything about how this is written.

**Compare content, not counts.** Row counts match while `is_current` flags are on
the wrong versions, versions are numbered in the wrong order, or the vectors came
from a different model. The M1.6.1 defect — 89% of chunks truncated before
embedding — passed every count-based check anyone had written, and would pass a
count-based replay check too.

**Compare natural keys, not primary keys.** Ids are UUIDv7, minted at write time;
a rebuild legitimately produces different ones. A comparison that included them
would fail on every honest replay, which is worse than useless because it teaches
you to ignore the check. A memory is identified by
`(source, external_key, version)` and a chunk by `(memory, ordinal)`.

The embedding is compared as a digest of its bytes rather than element-wise. It
is 384 floats produced by the same deterministic model over the same text, so
equality is the right assertion, and a hash makes the diff readable when it
fails.
"""

import hashlib
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application import graph_projection
from memoryos.application.ports import MemoryNode
from memoryos.domain.values import EdgeType

# --------------------------------------------------------------------------
# Snapshot rows
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryRow:
    """One memory version, described without reference to its id."""

    source_name: str
    external_key: str
    version: int
    is_current: bool
    normalized_hash: str | None
    content_hash: str
    kind: str
    occurred_at: str | None
    # Derived from the causing event's `recorded_at`, so a replay reproduces it.
    # Included beyond the milestone's stated column list precisely because it is
    # the value the "no clock in the replay path" rule exists to protect: if
    # anything ever reads `now()` there again, this is the column that catches it.
    ingested_at: str | None
    deleted_at: str | None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.source_name, self.external_key, self.version)


@dataclass(frozen=True, slots=True)
class ChunkRow:
    """One chunk, described without reference to its id or its memory's id."""

    source_name: str
    external_key: str
    memory_version: int
    ordinal: int
    content_hash: str
    chunker_version: str
    embedding_model: str | None
    token_count: int
    char_start: int
    char_end: int
    prefix_chars: int
    # A digest, not the vector. Element-wise comparison of 384 floats produces a
    # diff nobody can read; this reduces "the vectors differ" to one line, and
    # the model is deterministic so exact equality is the right bar.
    embedding_digest: str | None
    metadata: str

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.source_name, self.external_key, self.memory_version, self.ordinal)


@dataclass(frozen=True, slots=True)
class GraphMemoryRow:
    """One `Memory` node and its source edge, keyed the way a replay can compare.

    **Only the part of the projection a replay rebuilds is compared, and that is
    the same rule this snapshot already applies to the tables.** `entities`,
    `entity_mentions` and `entity_relationships` are truncated by a full replay and
    deliberately not rebuilt — an LLM call per chunk, against offsets that no
    longer point anywhere — which is exactly why `Snapshot` has never held an
    entity row either. The graph's entity half inherits that: it is *reported* by
    `ComparisonResult.graph_notes` and not diffed, because a diff of a section that
    is empty by design on one side is a check that fails every time and teaches
    nobody anything.

    What is left is a real check. Memory nodes and their `FROM_SOURCE` edges are a
    pure function of `memories` and `sources`, both of which a replay rebuilds, so
    these have to match exactly — and they are keyed on `(source_name,
    external_key)` rather than on `memory_id`, because a rebuild mints new ids.
    """

    source_name: str
    external_key: str
    kind: str
    occurred_at: str | None

    @property
    def key(self) -> tuple[str, str]:
        return (self.source_name, self.external_key)


@dataclass(frozen=True, slots=True)
class Snapshot:
    memories: tuple[MemoryRow, ...] = ()
    chunks: tuple[ChunkRow, ...] = ()
    graph_memories: tuple[GraphMemoryRow, ...] = ()
    # Counts of the entity-derived projection: `Entity` nodes, `MENTIONS` and
    # `RELATES_TO` edges. Reported, never diffed. See `GraphMemoryRow`.
    graph_entity_counts: dict[str, int] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "memories": len(self.memories),
            "chunks": len(self.chunks),
            "embedded_chunks": sum(
                1 for chunk in self.chunks if chunk.embedding_digest is not None
            ),
            "graph_memory_nodes": len(self.graph_memories),
        }


# --------------------------------------------------------------------------
# Differences
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TableDiff:
    table: str
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not (self.missing or self.unexpected or self.changed)

    @property
    def total(self) -> int:
        return len(self.missing) + len(self.unexpected) + len(self.changed)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    diffs: tuple[TableDiff, ...] = ()
    before: dict[str, int] = field(default_factory=dict)
    after: dict[str, int] = field(default_factory=dict)
    # The entity-derived graph counts on both sides, reported rather than
    # compared. See `GraphMemoryRow`.
    graph_notes: tuple[tuple[str, int, int], ...] = ()

    @property
    def identical(self) -> bool:
        return all(diff.clean for diff in self.diffs)

    def render(self, *, examples: int = 5) -> str:
        lines: list[str] = []
        for name in sorted(set(self.before) | set(self.after)):
            before = self.before.get(name, 0)
            after = self.after.get(name, 0)
            mark = "ok  " if before == after else "DIFF"
            lines.append(f"[{mark}] {name}: before {before}  after {after}")

        for diff in self.diffs:
            if diff.clean:
                lines.append(f"[ok  ] {diff.table}: identical")
                continue
            lines.append(
                f"[FAIL] {diff.table}: {len(diff.missing)} missing, "
                f"{len(diff.unexpected)} unexpected, {len(diff.changed)} changed"
            )
            for label, rows in (
                ("only before", diff.missing),
                ("only after", diff.unexpected),
                ("changed", diff.changed),
            ):
                for row in rows[:examples]:
                    lines.append(f"         {label}: {row}")
                if len(rows) > examples:
                    lines.append(f"         ... and {len(rows) - examples} more {label}")

        if self.graph_notes:
            lines += ["", "graph projection, entity half (reported, not compared):"]
            for name, before, after in self.graph_notes:
                lines.append(f"         {name}: before {before}  after {after}")
            lines.append(
                "         a full replay truncates entities, mentions and "
                "relationships and does not rebuild them — an LLM call per chunk, "
                "against offsets that no longer point anywhere. Re-run "
                "extract-entities, resolve-entities and extract-relationships "
                "after a replay; `graph verify` is the check that the projection "
                "matches whatever Postgres holds."
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Taking a snapshot
# --------------------------------------------------------------------------


async def snapshot(
    session_factory: async_sessionmaker[AsyncSession], *, sample: int | None = None
) -> Snapshot:
    """Read the derived state through natural keys.

    `sample` caps how many memories are read, for a quick check on a corpus too
    large to compare whole. It is a sample of *memories* and takes all of their
    chunks, because comparing an arbitrary subset of one document's chunks would
    make an ordinal mismatch look like a missing row.
    """
    async with session_factory() as session:
        memories = await _memory_rows(session, sample=sample)
        keys = {(row.source_name, row.external_key) for row in memories}
        chunks = await _chunk_rows(session)
        if sample is not None:
            chunks = tuple(
                chunk for chunk in chunks if (chunk.source_name, chunk.external_key) in keys
            )

    # From the same Postgres the rest of this snapshot reads, not from Neo4j. The
    # projection is a pure function of these tables, so this is what the graph
    # *owes* — which is the only thing a replay can be held to, since the live
    # graph is never touched by a shadow rebuild and there is nowhere to build a
    # second one. `graph verify` is where owed and held are compared.
    projection = await graph_projection.read(session_factory)
    graph_memories = tuple(
        GraphMemoryRow(
            source_name=source_name,
            external_key=node.external_key,
            kind=node.kind.value,
            occurred_at=_stamp(node.occurred_at),
        )
        for node, source_name in _memory_nodes_with_sources(projection)
        if sample is None or (source_name, node.external_key) in keys
    )
    counts = projection.counts

    return Snapshot(
        memories=memories,
        chunks=chunks,
        graph_memories=graph_memories,
        graph_entity_counts={
            "graph_entity_nodes": counts["Entity"],
            "graph_mentions_edges": counts[EdgeType.MENTIONS.value],
            "graph_relates_to_edges": counts[EdgeType.RELATES_TO.value],
        },
    )


def _memory_nodes_with_sources(
    projection: graph_projection.GraphProjection,
) -> list[tuple[MemoryNode, str]]:
    """Pair each projected memory with its source's *name*.

    The name rather than the id, because the id is minted per write and this row
    has to survive a rebuild. The pairing comes from the `FROM_SOURCE` edges, which
    is the projection's own statement of which source a memory belongs to —
    re-deriving it from a join here would be a second opinion about the same fact.
    """
    source_names = {str(node.source_id): node.name for node in projection.sources}
    source_of = {
        edge.start.key: source_names.get(edge.end.key, "")
        for edge in projection.edges
        if edge.type is EdgeType.FROM_SOURCE
    }
    return [
        (node, source_of.get(str(node.memory_id), "")) for node in projection.memories
    ]


async def _memory_rows(
    session: AsyncSession, *, sample: int | None
) -> tuple[MemoryRow, ...]:
    stmt = (
        select(
            models.Source.name,
            models.Memory.external_key,
            models.Memory.version,
            models.Memory.is_current,
            models.Memory.normalized_hash,
            models.Memory.content_hash,
            models.Memory.kind,
            models.Memory.occurred_at,
            models.Memory.ingested_at,
            models.Memory.deleted_at,
        )
        .join(models.Source, models.Source.id == models.Memory.source_id)
        # Ordered by the natural key, so two snapshots line up without sorting
        # and the rendered diff reads in a stable order.
        .order_by(models.Source.name, models.Memory.external_key, models.Memory.version)
    )
    if sample is not None:
        stmt = stmt.limit(sample)

    return tuple(
        MemoryRow(
            source_name=row[0],
            external_key=row[1],
            version=row[2],
            is_current=row[3],
            normalized_hash=row[4],
            content_hash=row[5],
            kind=row[6],
            occurred_at=_stamp(row[7]),
            ingested_at=_stamp(row[8]),
            deleted_at=_stamp(row[9]),
        )
        for row in await session.execute(stmt)
    )


async def _chunk_rows(session: AsyncSession) -> tuple[ChunkRow, ...]:
    stmt = (
        select(
            models.Source.name,
            models.Memory.external_key,
            models.Memory.version,
            models.MemoryChunk.ordinal,
            models.MemoryChunk.content_hash,
            models.MemoryChunk.chunker_version,
            models.MemoryChunk.embedding_model,
            models.MemoryChunk.token_count,
            models.MemoryChunk.char_start,
            models.MemoryChunk.char_end,
            models.MemoryChunk.prefix_chars,
            models.MemoryChunk.embedding,
            models.MemoryChunk.meta,
        )
        .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
        .join(models.Source, models.Source.id == models.Memory.source_id)
        .order_by(
            models.Source.name,
            models.Memory.external_key,
            models.Memory.version,
            models.MemoryChunk.ordinal,
        )
    )
    return tuple(
        ChunkRow(
            source_name=row[0],
            external_key=row[1],
            memory_version=row[2],
            ordinal=row[3],
            content_hash=row[4],
            chunker_version=row[5],
            embedding_model=row[6],
            token_count=row[7],
            char_start=row[8],
            char_end=row[9],
            prefix_chars=row[10],
            embedding_digest=digest_of(row[11]),
            # Serialised with sorted keys so that two equal mappings cannot
            # compare unequal because Postgres returned them in a different order.
            metadata=_canonical(row[12]),
        )
        for row in await session.execute(stmt)
    )


def digest_of(vector: Sequence[float] | None) -> str | None:
    """A short, stable fingerprint of one embedding.

    Packed as big-endian float64 rather than formatted as text: a text rendering
    would round, and two vectors that differ below the printed precision would
    compare equal — which is exactly the kind of "close enough" that hides a
    model swap.
    """
    if vector is None:
        return None
    packed = struct.pack(f">{len(vector)}d", *(float(value) for value in vector))
    return hashlib.blake2b(packed, digest_size=8).hexdigest()


def _stamp(value: object) -> str | None:
    return None if value is None else str(value)


def _canonical(value: object) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------
# Comparing two snapshots
# --------------------------------------------------------------------------


def compare(before: Snapshot, after: Snapshot) -> ComparisonResult:
    """Every difference, by table, with example rows.

    Reported rather than raised. A caller that wants an exit code asks
    `identical`; a caller that wants to know what moved reads `render()`.
    """
    return ComparisonResult(
        diffs=(
            _diff("memories", before.memories, after.memories),
            _diff("memory_chunks", before.chunks, after.chunks),
            _diff("graph_memory_nodes", before.graph_memories, after.graph_memories),
        ),
        before=before.counts,
        after=after.counts,
        graph_notes=tuple(
            (name, before.graph_entity_counts.get(name, 0), value)
            for name, value in sorted(after.graph_entity_counts.items())
        ),
    )


SnapshotRow = MemoryRow | ChunkRow | GraphMemoryRow


def _diff(
    table: str,
    before: Sequence[SnapshotRow],
    after: Sequence[SnapshotRow],
) -> TableDiff:
    left = {row.key: row for row in before}
    right = {row.key: row for row in after}

    missing = tuple(_describe(left[key]) for key in sorted(left.keys() - right.keys()))
    unexpected = tuple(_describe(right[key]) for key in sorted(right.keys() - left.keys()))

    changed: list[str] = []
    for key in sorted(left.keys() & right.keys()):
        if left[key] != right[key]:
            changed.append(_describe_change(left[key], right[key]))

    return TableDiff(
        table=table, missing=missing, unexpected=unexpected, changed=tuple(changed)
    )


def _describe(row: SnapshotRow) -> str:
    return " ".join(str(part) for part in row.key)


def _describe_change(left: SnapshotRow, right: SnapshotRow) -> str:
    """Name the fields that moved, not the whole row.

    A row-versus-row dump of two 11-field records leaves the reader diffing by
    eye. The point of a comparison is to say *which column* is wrong, because
    that is what identifies the defect: `is_current` says the version logic is
    broken, `embedding_digest` says the model changed.
    """
    fields = [
        name
        for name in left.__slots__
        if getattr(left, name) != getattr(right, name)
    ]
    parts = [
        f"{name}: {getattr(left, name)!r} -> {getattr(right, name)!r}" for name in fields
    ]
    return f"{_describe(left)}  [" + "; ".join(parts) + "]"
