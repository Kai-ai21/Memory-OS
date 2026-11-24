"""`[[wiki links]]` as entity mentions, found without a model (M12.2).

A link is the author saying, in so many words, "this is a thing, and this is
its name". A language model asked to find entities in the same text is
guessing at exactly what the link states, so the link is the stronger evidence
and costs nothing to read: no call, no quota, no rate limit. That matters on
this project's free tier, where model extraction has reached 8 of 320 memories.

**Behind the `EntityExtractor` port, not beside it.** Mentions are stored by
`ExtractEntities`, one extractor version per memory — a second extractor
writing its own mentions would delete the first one's. So `LinkedExtractor`
wraps whatever extractor is configured and merges, and its version names both,
which is what makes a corpus extracted before links existed re-extract.

**Notes only.** `[[` is also how Python writes a list of lists, and how a
Markdown code block might quote one. Links are read from `MemoryKind.NOTE`
text, outside code fences, and only where the target looks like a name.

**The link does not say what kind of thing it names.** `[[Alice]]` may be a
person, a project, or just the title of another note. So a link's entity is
marked `type_is_guess`, and storage resolves it against the entities that
already exist — M3.2's resolved, merge-followed set — by canonical name across
every type, falling back to `concept` only when nothing by that name exists.
"""

import re
from collections.abc import Sequence

from memoryos.adapters.extraction.llm import ExtractionStats
from memoryos.application.ports import (
    EntityExtractor,
    EntityRef,
    ExtractedEntity,
    ExtractedRelationship,
)
from memoryos.domain.normalization import canonical_entity_name
from memoryos.domain.values import EntityType, MemoryKind

LINKS_VERSION = "links-v1"

# `[[Target]]`, `[[Target|shown text]]`, `[[Target#Heading]]`, `![[Embed]]`.
# The target is group 1: everything before a `|` or `#`, on one line.
_LINK = re.compile(r"!?\[\[([^\[\]\|#\n]+?)(?:[#|][^\[\]\n]*)?\]\]")
_FENCE = re.compile(r"^\s*(?:```|~~~)", re.MULTILINE)
_HAS_LETTER = re.compile(r"[^\W\d_]")

#: A link's confidence. The author wrote it, so it is not in doubt that the
#: name is there; what is in doubt is only the type, which is marked separately.
LINK_CONFIDENCE = 1.0

#: Longer than any name, shorter than a sentence pasted between brackets.
_MAX_TARGET_CHARS = 120


def links_in(text: str) -> list[ExtractedEntity]:
    """Every link target in `text`, with offsets that slice to exactly its name."""
    fenced = _fenced_spans(text)
    found: list[ExtractedEntity] = []
    for match in _LINK.finditer(text):
        if any(start <= match.start() < end for start, end in fenced):
            continue
        raw = match.group(1)
        name = raw.strip()
        if not name or len(name) > _MAX_TARGET_CHARS or not _HAS_LETTER.search(name):
            continue
        # The stripped name's own offsets, so `text[start:end] == name` holds.
        start = match.start(1) + (len(raw) - len(raw.lstrip()))
        found.append(
            ExtractedEntity(
                name=name,
                type=EntityType.CONCEPT,
                confidence=LINK_CONFIDENCE,
                char_start=start,
                char_end=start + len(name),
                type_is_guess=True,
            )
        )
    return found


class LinkedExtractor(EntityExtractor):
    """The configured extractor, plus every `[[link]]` in a note.

    `primary` may be None — no API key — and links are still found, so a
    keyless deployment gets the entities its author marked by hand.
    """

    def __init__(self, primary: EntityExtractor | None) -> None:
        self._primary = primary
        self._empty_stats: ExtractionStats | None = None

    @property
    def primary(self) -> EntityExtractor | None:
        return self._primary

    @property
    def stats(self) -> ExtractionStats:
        # The model's cost report, which the CLI prints. Links cost nothing, so
        # without a model there is nothing to report.
        stats = getattr(self._primary, "stats", None)
        if isinstance(stats, ExtractionStats):
            return stats
        if self._empty_stats is None:
            self._empty_stats = ExtractionStats()
        return self._empty_stats

    @property
    def version(self) -> str:
        if self._primary is None:
            return LINKS_VERSION
        return f"{self._primary.version}+{LINKS_VERSION}"

    @property
    def relationship_version(self) -> str:
        # Unchanged by links: relationships are the model's alone, so wrapping
        # must not invalidate relationships already extracted.
        primary = self._primary
        return str(
            getattr(primary, "relationship_version", None)
            or (primary.version if primary is not None else LINKS_VERSION)
        )

    async def extract(self, text: str, *, kind: MemoryKind) -> list[ExtractedEntity]:
        return (await self.extract_batch([text], kind=kind))[0]

    async def extract_relationships(
        self, text: str, entities: Sequence[EntityRef]
    ) -> list[ExtractedRelationship]:
        """The model's, unchanged. A link says a thing exists, not how two relate."""
        if self._primary is None:
            return []
        return await self._primary.extract_relationships(text, entities)

    async def extract_batch(
        self, texts: list[str], *, kind: MemoryKind
    ) -> list[list[ExtractedEntity]]:
        primary = await self._primary_batch(texts, kind)
        if kind is not MemoryKind.NOTE:
            return primary
        return [
            _merge(found, links_in(text))
            for text, found in zip(texts, primary, strict=True)
        ]

    async def _primary_batch(
        self, texts: list[str], kind: MemoryKind
    ) -> list[list[ExtractedEntity]]:
        if self._primary is None:
            return [[] for _ in texts]
        batch = getattr(self._primary, "extract_batch", None)
        if callable(batch):
            results: list[list[ExtractedEntity]] = await batch(texts, kind=kind)
            return results
        return [await self._primary.extract(text, kind=kind) for text in texts]


def _merge(
    found: Sequence[ExtractedEntity], links: Sequence[ExtractedEntity]
) -> list[ExtractedEntity]:
    """The model's entities, plus links at spans the model did not already claim.

    Where the model typed the same name elsewhere in this chunk, the link takes
    that type rather than guessing: the model read the sentence, the link
    pattern did not.
    """
    claimed = {(entity.char_start, entity.char_end) for entity in found}
    typed = {canonical_entity_name(entity.name): entity.type for entity in found}
    merged = list(found)
    for link in links:
        if (link.char_start, link.char_end) in claimed:
            continue
        known = typed.get(canonical_entity_name(link.name))
        merged.append(
            link
            if known is None
            else ExtractedEntity(
                name=link.name,
                type=known,
                confidence=link.confidence,
                char_start=link.char_start,
                char_end=link.char_end,
            )
        )
    return merged


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    fences = [match.start() for match in _FENCE.finditer(text)]
    spans = [(fences[i], fences[i + 1]) for i in range(0, len(fences) - 1, 2)]
    if len(fences) % 2 == 1:
        spans.append((fences[-1], len(text)))
    return spans
