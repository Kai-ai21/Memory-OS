"""The golden set: labelled queries, resolved against the corpus they will score.

A golden query is a query text, the filters it was judged under, and a set of
relevant items. An item is a triple — `(source_name, external_key,
chunk_ordinal)` — where a null ordinal means "this memory, whichever chunk
matched" and a number means "this chunk of it, and no other".

Three decisions carry the weight here.

**`missing` counts as relevant.** It is the only verdict that names something the
ranking failed to return, so excluding it would define recall as "of the things
we found, how many did we find" and quietly report 1.0 forever.

**A query with no relevant items is excluded, loudly.** It can only ever score
zero, so leaving it in drags every mean down by a constant that has nothing to
do with retrieval quality. Dropping it silently would be worse than either: the
harness would report a healthier number every time somebody judged a query
entirely `not_relevant`.

**Triples are resolved against the current corpus at load time, and failures are
reported rather than dropped.** Files move and get renamed; a golden set that
quietly shrinks as that happens reports a rising score for a corpus that is
losing its answer key. Unresolved items are excluded from scoring — nothing can
retrieve them — but they are named in the output and in the JSON, so a shrinking
answer key is visible instead of flattering.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.values import Verdict

logger = structlog.get_logger(__name__)

# Verdicts that put an item in the answer key. `not_relevant` is the explicit
# opposite and `missing` is the whole reason recall is measurable at all.
RELEVANT_VERDICTS = frozenset({Verdict.RELEVANT.value, Verdict.MISSING.value})


@dataclass(frozen=True, slots=True, order=True)
class Item:
    """One thing a query should return, at memory or chunk granularity."""

    source_name: str
    external_key: str
    chunk_ordinal: int | None = None

    @property
    def key(self) -> str:
        """The string the metrics compare on.

        `source::key` and `source::key#4` are different tokens, which is the
        entire mechanism by which "right file, wrong chunk" scores as a miss.
        """
        base = f"{self.source_name}::{self.external_key}"
        return base if self.chunk_ordinal is None else f"{base}#{self.chunk_ordinal}"

    @property
    def memory_key(self) -> str:
        return f"{self.source_name}::{self.external_key}"


@dataclass(frozen=True, slots=True)
class Unresolved:
    """A judged item that no longer names anything in the corpus."""

    query_text: str
    item: Item
    reason: str

    def describe(self) -> str:
        return f"{self.item.key}: {self.reason}"


@dataclass(frozen=True, slots=True)
class Excluded:
    query_text: str
    reason: str


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    query_text: str
    filters: dict[str, Any]
    relevant: frozenset[Item]

    @property
    def relevant_keys(self) -> set[str]:
        return {item.key for item in self.relevant}

    @property
    def pinned(self) -> dict[str, frozenset[int]]:
        """Per memory, the ordinals this query insists on.

        Empty for the ordinary case. When it is not empty for a memory, that
        memory only counts as retrieved if one of these ordinals came back.
        """
        pins: dict[str, set[int]] = {}
        for item in self.relevant:
            if item.chunk_ordinal is not None:
                pins.setdefault(item.memory_key, set()).add(item.chunk_ordinal)
        return {memory: frozenset(ordinals) for memory, ordinals in pins.items()}

    def project(self, source_name: str, external_key: str, ordinals: Iterable[int]) -> str:
        """The key a retrieved memory occupies in this query's scoring space.

        The projection is where memory-level and chunk-level judgements are
        reconciled. A hit is normally just its memory. But if this query pinned
        chunks of that memory, the hit is only credited when one of them is
        among the chunks that actually matched — otherwise it projects to the
        bare memory key, which is not in the relevant set, and the query
        correctly scores a miss for returning the right file on the wrong chunk.
        """
        memory_key = f"{source_name}::{external_key}"
        wanted = self.pinned.get(memory_key)
        if not wanted:
            return memory_key
        matched = sorted(ordinal for ordinal in ordinals if ordinal in wanted)
        return f"{memory_key}#{matched[0]}" if matched else memory_key


@dataclass(frozen=True, slots=True)
class GoldenSet:
    queries: list[GoldenQuery]
    excluded: list[Excluded] = field(default_factory=list)
    unresolved: list[Unresolved] = field(default_factory=list)
    generated_at: str | None = None

    def select(self, query_text: str) -> "GoldenSet":
        """The one query whose text matches, for `evaluate --query`."""
        wanted = query_text.strip().casefold()
        return GoldenSet(
            queries=[q for q in self.queries if q.query_text.casefold() == wanted],
            excluded=self.excluded,
            unresolved=self.unresolved,
            generated_at=self.generated_at,
        )


@dataclass(frozen=True, slots=True)
class CorpusIndex:
    """What the corpus currently contains, keyed the way judgements are.

    A mapping rather than a session, so the resolution rules can be tested
    without a database — they are rules about identity, not about SQL.
    """

    ordinals: Mapping[tuple[str, str], frozenset[int]]

    def has(self, item: Item) -> bool:
        return (item.source_name, item.external_key) in self.ordinals

    def chunk_count(self, item: Item) -> int:
        return len(self.ordinals.get((item.source_name, item.external_key), ()))

    def resolve(self, item: Item) -> str | None:
        """None when the item resolves; otherwise why it does not."""
        if not self.has(item):
            return "no current memory with this source and external key"
        if item.chunk_ordinal is None:
            return None
        present = self.ordinals[(item.source_name, item.external_key)]
        if item.chunk_ordinal in present:
            return None
        return f"chunk {item.chunk_ordinal} does not exist; the memory has {len(present)}"


async def load_corpus_index(
    session_factory: async_sessionmaker[AsyncSession],
) -> CorpusIndex:
    """Every current memory's `(source, key)` and the ordinals it actually has.

    Ordinals are read rather than inferred from a count. They should be a
    contiguous 0..n-1 range and the schema very nearly guarantees it, but "very
    nearly" is how a golden set ends up pinned to a chunk that is not there.
    """
    stmt = (
        select(
            models.Source.name,
            models.Memory.external_key,
            models.MemoryChunk.ordinal,
        )
        .join(models.Source, models.Source.id == models.Memory.source_id)
        .join(models.MemoryChunk, models.MemoryChunk.memory_id == models.Memory.id)
        .where(
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
        )
    )
    ordinals: dict[tuple[str, str], set[int]] = {}
    async with session_factory() as session:
        for source_name, external_key, ordinal in await session.execute(stmt):
            ordinals.setdefault((source_name, external_key), set()).add(ordinal)
    return CorpusIndex(
        ordinals={key: frozenset(values) for key, values in ordinals.items()}
    )


def build_golden_set(payload: Mapping[str, Any], corpus: CorpusIndex) -> GoldenSet:
    """Turn an export into scoreable queries. Pure: the corpus is passed in."""
    queries: list[GoldenQuery] = []
    excluded: list[Excluded] = []
    unresolved: list[Unresolved] = []

    for entry in payload.get("queries", []):
        query_text = str(entry["query_text"])
        relevant, bad = _relevant_items(query_text, entry.get("items", []), corpus)
        unresolved.extend(bad)

        if not relevant:
            reason = (
                "every judged item is unresolved"
                if bad
                else "no relevant or missing judgements"
            )
            excluded.append(Excluded(query_text=query_text, reason=reason))
            # Loud, because the alternative — a mean that quietly improves when
            # somebody judges a whole query irrelevant — is a metric that lies.
            logger.warning(
                "golden.query_excluded", query=query_text, reason=reason, unresolved=len(bad)
            )
            continue

        queries.append(
            GoldenQuery(
                query_text=query_text,
                filters=dict(entry.get("filters") or {}),
                relevant=frozenset(relevant),
            )
        )

    for item in unresolved:
        logger.warning(
            "golden.item_unresolved", query=item.query_text, item=item.item.key,
            reason=item.reason,
        )

    return GoldenSet(
        queries=queries,
        excluded=excluded,
        unresolved=unresolved,
        generated_at=payload.get("generated_at"),
    )


async def load_golden_set(
    path: Path, session_factory: async_sessionmaker[AsyncSession]
) -> GoldenSet:
    """Read the M2.0a export and resolve it against the corpus as it is now."""
    payload = json.loads(path.read_text())
    return build_golden_set(payload, await load_corpus_index(session_factory))


def _relevant_items(
    query_text: str, items: Sequence[Mapping[str, Any]], corpus: CorpusIndex
) -> tuple[set[Item], list[Unresolved]]:
    """The answer key for one query, minus anything that no longer exists.

    A chunk-level verdict overrides a memory-level one for the same file: the
    person who said "chunk 4 is the answer" has said something strictly more
    specific than "this file is the answer", and keeping both would let the
    wrong chunk score as a hit through the memory-level entry — which is the
    exact thing the ordinal was added to prevent.
    """
    relevant: set[Item] = set()
    unresolved: list[Unresolved] = []

    for raw in items:
        if raw.get("verdict") not in RELEVANT_VERDICTS:
            continue
        ordinal = raw.get("chunk_ordinal")
        item = Item(
            source_name=str(raw["source_name"]),
            external_key=str(raw["external_key"]),
            chunk_ordinal=None if ordinal is None else int(ordinal),
        )
        reason = corpus.resolve(item)
        if reason is not None:
            unresolved.append(Unresolved(query_text=query_text, item=item, reason=reason))
            continue
        relevant.add(item)

    pinned = {item.memory_key for item in relevant if item.chunk_ordinal is not None}
    return {
        item
        for item in relevant
        if item.chunk_ordinal is not None or item.memory_key not in pinned
    }, unresolved
