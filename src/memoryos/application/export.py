"""Everything you wrote, in a form that outlives this program.

**Local-first means you can leave, and an export is the only thing that makes that
claim checkable.** Every other guarantee in this system is about what happens while
it is running. This one is about what you have if you stop.

## Version history is included, and that is the whole design decision

An export that flattened to current state would be smaller, simpler and would lose
the thing this system was built for. A corpus of current values is a database dump;
what makes this corpus worth keeping is that it knows what you thought last month
and that you changed your mind — the evolution view, the change summaries, the
`as-of` queries and every reflection Phase 5 generates all rest on superseded
versions still being there. Exporting only the current text would produce a file
that could reconstruct the search index and could not reconstruct a single thing
Phase 4 or 5 says.

So the unit of export is the *item*, and an item is a list of versions, oldest
first, with the current one marked. A tombstoned item is exported with its deletion
recorded rather than omitted: "this was here and I removed it from view" is a fact
about the corpus, and an export that silently dropped it would disagree with the
timeline.

A **purged** item is absent, and completely. There is no version of it, no text and
no key — that is what permanent deletion means, and an export that reported the
purge would leak exactly the thing the purge removed. The log still holds the
event; the export is of the corpus, not of the log.

## Two formats, for two readers

`json` is the round-trippable one: every field, every version, in a shape a program
can read. It is what the round-trip test asserts against, and the reason each
version carries its `content_hash` — a consumer can verify what it received without
trusting this writer.

`markdown` is for a person, and for the case that matters most: the day you stop
running this. It is one document per item with its versions as sections, readable in
any editor, with the metadata in front matter rather than in prose. Lossier by
construction — a chunk boundary has no markdown — and it says so in its own header
rather than implying completeness.

Both stream. A corpus does not fit in memory and an export that worked on a fixture
and died on a real corpus would fail exactly when somebody was trying to leave.
"""

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application import tags as tags_module

logger = structlog.get_logger(__name__)

# The export's own version, distinct from the schema's. A consumer reading a file
# written today has to be able to tell what shape it is in, and "whatever Alembic
# revision the writer happened to be on" is not a contract — the schema changes for
# reasons that do not change this file's layout, and this layout could change
# without a migration.
FORMAT_VERSION = 1

# Items per batch. The point of streaming is that the corpus does not have to fit
# in memory; the point of batching is that one query per item does not either.
BATCH = 100


class UnknownSource(LookupError):
    """A `--source` naming nothing."""


@dataclass(frozen=True, slots=True)
class Version:
    """One version of one item, as exported."""

    version: int
    is_current: bool
    content_hash: str
    # Null before normalization has run. Exported as null rather than omitted,
    # because "not yet parsed" and "empty" are different states and a consumer
    # rebuilding an index needs to tell them apart.
    content: str | None
    title: str | None
    kind: str
    occurred_at: str | None
    # Beside the timestamp it qualifies, always. An mtime and a date somebody
    # declared are different claims, and a consumer that received only the
    # timestamps could not render them differently however much it wanted to.
    occurred_at_source: str
    ingested_at: str
    # When this version was removed from view, if it was. Only the current version
    # of a tombstoned item carries one.
    deleted_at: str | None = None
    chunks: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "is_current": self.is_current,
            "content_hash": self.content_hash,
            "content": self.content,
            "title": self.title,
            "kind": self.kind,
            "occurred_at": self.occurred_at,
            "occurred_at_source": self.occurred_at_source,
            "ingested_at": self.ingested_at,
            "deleted_at": self.deleted_at,
            "chunks": self.chunks,
        }


@dataclass(frozen=True, slots=True)
class ExportedItem:
    """One item and its whole history."""

    source: str
    source_kind: str
    external_key: str
    versions: tuple[Version, ...]
    tags: tuple[str, ...] = ()

    @property
    def current(self) -> Version | None:
        for version in self.versions:
            if version.is_current:
                return version
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_kind": self.source_kind,
            "external_key": self.external_key,
            "tags": list(self.tags),
            "versions": [version.as_dict() for version in self.versions],
        }


@dataclass(slots=True)
class ExportStats:
    """What an export covered, for the caller to report.

    Mutable, unlike everything else here, because it is an accumulator handed *into*
    a generator. The alternative is for the generator to yield counts alongside the
    items and for every consumer to fold them, which puts arithmetic in the CLI and
    in the API and in the test — three places to get it wrong.
    """

    items: int = 0
    versions: int = 0
    tagged: int = 0
    tombstoned: int = 0
    sources: set[str] = field(default_factory=set)


async def items(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    source: str | None = None,
    stats: ExportStats | None = None,
) -> AsyncIterator[ExportedItem]:
    """Every item, with every version, oldest key first.

    Ordered by `(source, external_key)` rather than by date, and deliberately: an
    export is diffed against the previous one far more often than it is read
    top-to-bottom, and a stable order is what makes that diff mean something. A
    date order would reshuffle the whole file whenever anything was corrected.

    Batched by item, not by row, so an item's versions are never split across two
    batches — a consumer streaming this must never see a partial history and be
    unable to tell it from a complete one.
    """
    async with session_factory() as session:
        source_id = await _resolve_source(session, source)
        keys, sources = await _keys(session, source_id)

        for start in range(0, len(keys), BATCH):
            batch = keys[start : start + BATCH]
            grouped = await _versions_for(session, batch)
            tagged = await tags_module.for_items(session, batch)
            for pair in batch:
                versions = grouped.get(pair, ())
                if not versions:
                    continue
                name, kind = sources[pair[0]]
                item = ExportedItem(
                    source=name,
                    source_kind=kind,
                    external_key=pair[1],
                    versions=versions,
                    tags=tuple(tag.display for tag in tagged.get(pair, [])),
                )
                if stats is not None:
                    stats.items += 1
                    stats.versions += len(versions)
                    stats.sources.add(item.source)
                    if item.tags:
                        stats.tagged += 1
                    current = item.current
                    if current is not None and current.deleted_at is not None:
                        stats.tombstoned += 1
                yield item


async def _resolve_source(
    session: AsyncSession, source: str | None
) -> UUID | None:
    """The id of a named source, or None for the whole corpus.

    An unknown name is refused rather than treated as "no filter". Exporting the
    entire corpus because a name was misspelled is the wrong direction to fail in
    for a command somebody redirects to a file.
    """
    if source is None:
        return None
    found = (
        await session.execute(
            select(models.Source.id).where(models.Source.name == source)
        )
    ).scalar_one_or_none()
    if found is None:
        raise UnknownSource(f"no source named {source!r}")
    return found


async def _keys(
    session: AsyncSession, source_id: UUID | None
) -> tuple[list[tuple[UUID, str]], dict[UUID, tuple[str, str]]]:
    """Every `(source_id, external_key)` in scope, and the sources they belong to.

    The source names come back alongside rather than being looked up per item: the
    set is tiny and unchanging during an export, and a join per batch to re-fetch a
    name is the query-per-row shape this module exists to avoid. Returned rather
    than cached on the module, because module state shared by two concurrent
    exports is a bug waiting for the second caller.

    Read up front rather than streamed, and this is the one place that holds
    something proportional to the corpus: one UUID and one path per item. A million
    items is tens of megabytes, and it buys the guarantee above — that an item's
    versions are never split across batches — which a server-side cursor over
    `memories` ordered by key could also give but only while nothing wrote to the
    table during the export.
    """
    stmt = select(
        models.Memory.source_id, models.Memory.external_key, models.Source.name, models.Source.kind
    ).join(models.Source, models.Source.id == models.Memory.source_id)
    if source_id is not None:
        stmt = stmt.where(models.Memory.source_id == source_id)
    rows = (
        await session.execute(
            stmt.distinct().order_by(models.Source.name, models.Memory.external_key)
        )
    ).all()
    sources = {row[0]: (str(row[2]), str(row[3])) for row in rows}
    return [(row[0], str(row[1])) for row in rows], sources


async def _versions_for(
    session: AsyncSession, keys: Sequence[tuple[UUID, str]]
) -> dict[tuple[UUID, str], tuple[Version, ...]]:
    """Every version of each of these items, oldest first, with chunk counts."""
    if not keys:
        return {}
    rows = (
        await session.execute(
            select(
                models.Memory.source_id,
                models.Memory.external_key,
                models.Memory.version,
                models.Memory.is_current,
                models.Memory.content_hash,
                models.Memory.content,
                models.Memory.title,
                models.Memory.kind,
                models.Memory.occurred_at,
                models.Memory.occurred_at_source,
                models.Memory.ingested_at,
                models.Memory.deleted_at,
                func.count(models.MemoryChunk.id),
            )
            .outerjoin(
                models.MemoryChunk, models.MemoryChunk.memory_id == models.Memory.id
            )
            .where(
                tuple_(models.Memory.source_id, models.Memory.external_key).in_(
                    list(keys)
                )
            )
            .group_by(models.Memory.id)
            .order_by(models.Memory.external_key, models.Memory.version)
        )
    ).all()

    grouped: dict[tuple[UUID, str], list[Version]] = {}
    for row in rows:
        grouped.setdefault((row[0], str(row[1])), []).append(
            Version(
                version=int(row[2]),
                is_current=bool(row[3]),
                content_hash=str(row[4]),
                content=row[5],
                title=row[6],
                kind=str(row[7]),
                occurred_at=_stamp(row[8]),
                occurred_at_source=str(row[9]),
                ingested_at=_stamp(row[10]) or "",
                deleted_at=_stamp(row[11]),
                chunks=int(row[12]),
            )
        )
    return {key: tuple(value) for key, value in grouped.items()}


def _stamp(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


async def to_json(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    source: str | None = None,
    stats: ExportStats | None = None,
) -> AsyncIterator[str]:
    """The whole corpus as one JSON document, emitted a line at a time.

    **Hand-assembled rather than `json.dumps` of a list**, because the list is the
    thing that must not exist: a corpus-sized `dumps` builds the entire document as
    one string before the first byte reaches the pipe. Each *item* is dumped
    individually — bounded by one item's size — and the brackets and commas are
    written around them.

    Newline-delimited inside the array, so the file is both valid JSON and greppable
    line by line, which is what somebody actually does with an export.
    """
    header = {
        "format": "memoryos-export",
        "format_version": FORMAT_VERSION,
        "exported_at": datetime.now(UTC).isoformat(),
        "source": source,
        # Said in the file rather than in a README, because the file is what
        # survives. A consumer that finds only current versions in an export
        # written by some later program can tell that this one was different.
        "includes_version_history": True,
    }
    yield "{\n"
    yield f'"meta": {json.dumps(header)},\n'
    yield '"items": [\n'
    first = True
    async for item in items(session_factory, source=source, stats=stats):
        yield ("" if first else ",\n") + json.dumps(item.as_dict())
        first = False
    yield "\n]\n}\n"


async def to_markdown(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    source: str | None = None,
    stats: ExportStats | None = None,
) -> AsyncIterator[str]:
    """The corpus as documents a person can read without this program.

    One section per item, one subsection per version, front matter for the
    metadata. The header states what markdown cannot carry rather than leaving a
    reader to infer completeness from a file that looks complete — chunk boundaries,
    embeddings and the entity graph are all derived and all absent, and the JSON
    format is named as the one to use for a round trip.
    """
    yield "# Memory OS export\n\n"
    yield f"Exported {datetime.now(UTC).isoformat()}"
    yield f" — source `{source}`\n\n" if source else " — whole corpus\n\n"
    yield (
        "Every version of every item is here, oldest first, because what you "
        "thought before you corrected it is part of the record.\n\n"
        "This format is for reading. Chunk boundaries, embeddings and the entity "
        "graph are derived from the text below and are not reproduced; "
        "`--format json` is the one to use to move this corpus somewhere else.\n\n"
        "---\n\n"
    )
    async for item in items(session_factory, source=source, stats=stats):
        current = item.current
        heading = (current.title if current else None) or item.external_key
        yield f"## {heading}\n\n"
        yield "```yaml\n"
        yield f"source: {item.source}\n"
        yield f"external_key: {item.external_key}\n"
        yield f"kind: {current.kind if current else 'unknown'}\n"
        yield f"versions: {len(item.versions)}\n"
        if item.tags:
            yield f"tags: [{', '.join(item.tags)}]\n"
        if current and current.deleted_at:
            # Named in the export, because an item removed from view is still part
            # of the history and a reader has to know this one is not current.
            yield f"removed_from_view: {current.deleted_at}\n"
        yield "```\n\n"

        for version in item.versions:
            marker = " (current)" if version.is_current else " (superseded)"
            when = version.occurred_at or version.ingested_at
            yield f"### Version {version.version}{marker} — {when}\n\n"
            if version.content is None:
                yield "_Not yet normalized; no text to export._\n\n"
            else:
                yield version.content.rstrip() + "\n\n"
        yield "---\n\n"
