"""Structure-aware chunking.

The ordering of the algorithm is the whole design: split on boundaries the
author already marked, only then fall back to sentence filling, then repair
sections too small to retrieve well, then add overlap. Going straight to
fixed-size windows would be simpler and would cut through every heading in the
corpus.
"""

import re
from dataclasses import dataclass
from typing import Any

from memoryos.adapters.chunking.sentences import split_sentences
from memoryos.application.ports import (
    Chunker,
    ParsedDocument,
    StructureMarker,
    TextChunk,
    TokenCounter,
)
from memoryos.domain.values import MemoryKind

Span = tuple[int, int]


# Room for [CLS], [SEP], and whatever else the tokenizer prepends. Cheaper to
# reserve a dozen tokens than to discover the encoder silently dropped the last
# sentence of every chunk.
SPECIAL_TOKEN_HEADROOM = 16


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """Sizes in *model* tokens, derived from the window rather than chosen.

    M1.4 picked 640 by judgement and counted words; the model read 256
    WordPieces. Deriving from `max_sequence_tokens` means a model swap
    re-derives correct sizes instead of relying on somebody remembering.
    """

    # The window the counter reports, kept so the version string can name it.
    model_window: int = 512
    # Hard ceiling. No longer a preference: exceeding it means text is
    # discarded before embedding, which is silent data loss.
    maximum: int = 496
    target: int = 396
    # Boundaries are arbitrary, and a concept that straddles one would
    # otherwise appear in neither chunk in full.
    overlap: int = 47
    # Below this a chunk is a fragment: a heading and one line retrieves on
    # keyword luck rather than meaning.
    minimum: int = 79

    @classmethod
    def for_window(cls, model_window: int) -> "ChunkerConfig":
        maximum = max(1, model_window - SPECIAL_TOKEN_HEADROOM)
        target = max(1, int(maximum * 0.8))
        return cls(
            model_window=model_window,
            maximum=maximum,
            target=target,
            overlap=int(target * 0.12),
            minimum=int(target * 0.2),
        )


class StructuralChunker(Chunker):
    """Splits text into spans the embedding model can actually read.

    Takes a `TokenCounter` rather than an `Embedder`: it needs to know how the
    model counts and where it stops reading, not how to produce vectors.
    """

    def __init__(
        self, counter: TokenCounter, config: ChunkerConfig | None = None
    ) -> None:
        self._counter = counter
        self._config = config or ChunkerConfig.for_window(counter.max_sequence_tokens)

    @property
    def config(self) -> ChunkerConfig:
        return self._config

    @property
    def max_tokens(self) -> int:
        return self._config.maximum

    @property
    def version(self) -> str:
        """Algorithm and parameters, together.

        The parameters are in the string because that is what makes improving
        the chunker a query — find every chunk carrying the old stamp and
        re-chunk only those — instead of a corpus-wide rebuild. It is also what
        answers "what produced this bad chunk?" six months from now.

        `model_window` is in there because a model with a different window
        produces differently-sized chunks. Chunks sized for the old window are
        stale even when every other parameter looks unchanged.
        """
        config = self._config
        return (
            f"structural-v2:model_window={config.model_window}"
            f":target={config.target}:overlap={config.overlap}"
            f":min={config.minimum}:max={config.maximum}"
        )

    def _count(self, text: str) -> int:
        return self._counter.count_tokens(text)

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        text = doc.text
        if not text.strip():
            return []

        sections = self._sections(doc)

        if doc.kind is MemoryKind.CODE:
            spans = self._chunk_code(sections, doc)
        else:
            spans = self._merge_undersized(
                [span for section in sections for span in self._fill(section, text)], text
            )

        definitions = [m for m in doc.structure if m.kind == "definition"]
        return self._to_chunks(spans, text, definitions)

    def _chunk_code(self, sections: list[Span], doc: ParsedDocument) -> list[Span]:
        """Definition boundaries first, then split anything still too large.

        M1.4 kept whole functions intact at any size, on the grounds that half
        a function embeds badly. That was right about halves and wrong about
        the alternative: an oversized function was not kept whole, it was
        truncated — the model read its first 256 tokens and discarded the rest.
        Two complete halves beat one silently-truncated whole.

        So the ceiling is now a hard invariant, and an oversized definition is
        split at the outermost boundary inside it: blank lines first, then
        statement starts at the lowest indentation the body uses.
        """
        spans: list[Span] = []
        for section in sections:
            if self._count(doc.text[section[0] : section[1]]) <= self._config.maximum:
                spans.append(section)
            else:
                spans.extend(self._split_definition(section, doc.text))
        return self._merge_undersized(spans, doc.text)

    def _split_definition(self, section: Span, text: str) -> list[Span]:
        """Break one oversized definition at the safest boundaries it offers."""
        for boundaries in (
            _blank_line_offsets(text, section),
            _outermost_statement_offsets(text, section),
        ):
            if not boundaries:
                continue
            spans = self._fill_between(section, boundaries, text)
            if all(
                self._count(text[start:end]) <= self._config.maximum
                for start, end in spans
            ):
                return spans

        # Neither offered anything usable — one enormous unbroken line.
        return self._hard_split(section, text)

    def _fill_between(
        self, section: Span, boundaries: list[int], text: str
    ) -> list[Span]:
        """Group candidate boundaries into spans that fit under the ceiling."""
        start, end = section
        spans: list[Span] = []
        cursor = start

        for boundary in [*boundaries, end]:
            if boundary <= cursor:
                continue
            if self._count(text[cursor:boundary]) > self._config.target and spans:
                continue
            if self._count(text[cursor:boundary]) > self._config.maximum:
                spans.append((cursor, boundary))
                cursor = boundary
                continue
            if self._count(text[cursor:boundary]) >= self._config.target:
                spans.append((cursor, boundary))
                cursor = boundary

        if cursor < end:
            spans.append((cursor, end))
        return spans or [section]

    def _sections(self, doc: ParsedDocument) -> list[Span]:
        """Cut the text at the offsets the parser marked.

        A markdown document splits at headings because the author already said
        where the topic changes; code splits at definitions for the same
        reason. Nothing here has to infer a boundary that was stated.
        """
        length = len(doc.text)
        offsets = sorted(
            {marker.char_offset for marker in doc.structure if 0 < marker.char_offset < length}
        )
        bounds = [0, *offsets, length]
        return [
            (bounds[index], bounds[index + 1])
            for index in range(len(bounds) - 1)
            if bounds[index] < bounds[index + 1]
        ]

    def _fill(self, section: Span, text: str) -> list[Span]:
        """Break an oversized section at sentence boundaries."""
        start, end = section
        if self._count(text[start:end]) <= self._config.target:
            return [section]

        sentences = split_sentences(text[start:end])
        if not sentences:
            return self._hard_split(section, text)

        spans: list[Span] = []
        cursor = start
        accumulated = 0

        for relative_start, relative_end in sentences:
            absolute_start = start + relative_start
            tokens = self._count(text[absolute_start : start + relative_end])

            if accumulated and accumulated + tokens > self._config.target:
                # Break *before* this sentence, so the whitespace between the
                # two stays with the chunk that precedes it and the spans
                # remain contiguous.
                spans.append((cursor, absolute_start))
                cursor = absolute_start
                accumulated = 0

            accumulated += tokens

        spans.append((cursor, end))

        # A single sentence longer than the ceiling — minified text, a giant
        # table row — cannot be fixed by sentence splitting.
        return [
            piece
            for span in spans
            for piece in (
                [span]
                if self._count(text[span[0] : span[1]]) <= self._config.maximum
                else self._hard_split(span, text)
            )
        ]

    def _hard_split(self, section: Span, text: str) -> list[Span]:
        """Last resort: break on whitespace to stay under the ceiling."""
        start, end = section
        spans: list[Span] = []
        cursor = start
        accumulated = 0
        index = start

        while index < end:
            while index < end and not text[index].isspace():
                index += 1
            while index < end and text[index].isspace():
                index += 1
            accumulated = self._count(text[cursor:index])
            if accumulated >= self._config.target and index < end:
                spans.append((cursor, index))
                cursor = index

        if cursor < end:
            spans.append((cursor, end))

        # Whitespace is the preferred cut, but text can simply not have any:
        # a minified bundle, a base64 blob, an unbroken run of CJK or emoji.
        # The ceiling is a hard invariant, so something has to give, and a
        # mid-word cut is a far smaller loss than a discarded half.
        if any(
            self._count(text[start:stop]) > self._config.maximum
            for start, stop in spans
        ):
            return [
                piece
                for span in spans
                for piece in (
                    [span]
                    if self._count(text[span[0] : span[1]]) <= self._config.maximum
                    else self._character_split(span, text)
                )
            ]
        return spans

    def _character_split(self, section: Span, text: str) -> list[Span]:
        """Last resort: cut on character counts.

        The token count is not a linear function of characters, so each cut is
        estimated proportionally and then halved until it fits — a handful of
        counter calls per span rather than a scan.
        """
        start, end = section
        spans: list[Span] = []
        cursor = start

        while cursor < end:
            tokens = self._count(text[cursor:end])
            if tokens <= self._config.maximum:
                spans.append((cursor, end))
                break

            remaining = end - cursor
            stop = min(end, cursor + max(1, remaining * self._config.target // tokens))
            while stop - cursor > 1 and self._count(text[cursor:stop]) > self._config.maximum:
                stop = cursor + max(1, (stop - cursor) // 2)

            spans.append((cursor, stop))
            cursor = stop

        return spans

    def _merge_undersized(self, spans: list[Span], text: str) -> list[Span]:
        """Absorb fragments into their neighbour.

        A twelve-token section holding a heading and one line is not a useful
        retrieval unit: it matches on keyword luck and carries no context. It
        only merges while the result stays within the target, so repairing
        small sections never manufactures oversized ones.
        """
        if not spans:
            return []

        merged: list[Span] = []
        index = 0
        while index < len(spans):
            start, end = spans[index]
            while (
                self._count(text[start:end]) < self._config.minimum
                and index + 1 < len(spans)
                and self._count(text[start : spans[index + 1][1]]) <= self._config.target
            ):
                index += 1
                end = spans[index][1]
            merged.append((start, end))
            index += 1

        # The last section has nothing following it, so it merges backwards or
        # not at all.
        if len(merged) > 1:
            start, end = merged[-1]
            previous_start, _ = merged[-2]
            if (
                self._count(text[start:end]) < self._config.minimum
                and self._count(text[previous_start:end]) <= self._config.target
            ):
                merged[-2] = (previous_start, end)
                merged.pop()

        return merged

    def _to_chunks(
        self,
        spans: list[Span],
        text: str,
        definitions: list[StructureMarker] | None = None,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        previous: Span | None = None

        for start, end in spans:
            body_start = start
            if previous is not None:
                body_start = self._overlap_start(text, previous, end)

            # Spans are contiguous, so the overlap prefix and the body are one
            # continuous slice: `text[body_start:end]`. char_start/char_end
            # still name the body alone, which is what a citation should
            # highlight — the overlap belongs to the chunk before this one.
            content = text[body_start:end]
            tokens = self._count(content)
            if tokens == 0 or not content.strip():
                # The schema requires token_count > 0 and char_end > char_start,
                # and a whitespace-only chunk retrieves for nothing anyway.
                continue

            chunks.append(
                TextChunk(
                    ordinal=len(chunks),
                    text=content,
                    char_start=start,
                    char_end=end,
                    token_count=tokens,
                    metadata=_definition_metadata(definitions, start),
                )
            )
            previous = (start, end)

        return chunks

    def _overlap_start(self, text: str, previous: Span, end: int) -> int:
        """Where the overlap prefix begins, at a sentence boundary."""
        if self._config.overlap <= 0:
            return previous[1]

        previous_start, previous_end = previous
        sentences = split_sentences(text[previous_start:previous_end])
        if not sentences:
            return previous_end

        chosen = previous_end
        for relative_start, _ in reversed(sentences):
            candidate = previous_start + relative_start
            chosen = candidate
            if self._count(text[candidate:previous_end]) >= self._config.overlap:
                break

        # Overlap must not push a chunk over the ceiling; context is worth less
        # than staying inside the model's window.
        if self._count(text[chosen:end]) > self._config.maximum:
            return previous_end
        return chosen


def _blank_line_offsets(text: str, section: Span) -> list[int]:
    """Offsets just after a blank line inside the section.

    The safest place to cut a function: a blank line is where its author
    already grouped things.
    """
    start, end = section
    offsets: list[int] = []
    for match in re.finditer(r"\n[ \t]*\n", text[start:end]):
        offset = start + match.end()
        if start < offset < end:
            offsets.append(offset)
    return offsets


def _outermost_statement_offsets(text: str, section: Span) -> list[int]:
    """Line starts at the shallowest indentation the body actually uses.

    The definition's own line is skipped, so this finds the top level *inside*
    the function rather than the function header itself. Cutting there keeps
    whole statements together.
    """
    start, end = section
    body = text[start:end]
    lines: list[tuple[int, int]] = []
    offset = 0
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            lines.append((offset, len(line) - len(line.lstrip())))
        offset += len(line)

    if len(lines) < 2:
        return []

    # Skip the header line; the rest of the body defines the inner top level.
    indents = [indent for _, indent in lines[1:]]
    if not indents:
        return []
    shallowest = min(indents)

    return [
        start + line_offset
        for line_offset, indent in lines[1:]
        if indent == shallowest and start + line_offset > start
    ]


def _definition_metadata(
    definitions: list[StructureMarker] | None, start: int
) -> dict[str, Any]:
    """Name the definition a span falls inside, if any."""
    if not definitions:
        return {}
    enclosing = [marker for marker in definitions if marker.char_offset <= start]
    if not enclosing:
        return {}
    label = max(enclosing, key=lambda marker: marker.char_offset).label
    return {"definition": label} if label else {}
