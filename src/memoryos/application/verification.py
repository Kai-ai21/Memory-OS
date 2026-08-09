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
    # A digest, not the vector. Element-wise comparison of 384 floats produces a
    # diff nobody can read; this reduces "the vectors differ" to one line, and
    # the model is deterministic so exact equality is the right bar.
    embedding_digest: str | None
    metadata: str

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.source_name, self.external_key, self.memory_version, self.ordinal)


@dataclass(frozen=True, slots=True)
class Snapshot:
    memories: tuple[MemoryRow, ...] = ()
    chunks: tuple[ChunkRow, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        return {
            "memories": len(self.memories),
            "chunks": len(self.chunks),
            "embedded_chunks": sum(
                1 for chunk in self.chunks if chunk.embedding_digest is not None
            ),
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
    return Snapshot(memories=memories, chunks=chunks)


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
            embedding_digest=digest_of(row[10]),
            # Serialised with sorted keys so that two equal mappings cannot
            # compare unequal because Postgres returned them in a different order.
            metadata=_canonical(row[11]),
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
        ),
        before=before.counts,
        after=after.counts,
    )


def _diff(
    table: str,
    before: Sequence[MemoryRow] | Sequence[ChunkRow],
    after: Sequence[MemoryRow] | Sequence[ChunkRow],
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


def _describe(row: MemoryRow | ChunkRow) -> str:
    return " ".join(str(part) for part in row.key)


def _describe_change(left: MemoryRow | ChunkRow, right: MemoryRow | ChunkRow) -> str:
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
