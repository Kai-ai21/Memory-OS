"""Structure-aware chunking.

The ordering of the algorithm is the whole design: split on boundaries the
author already marked, only then fall back to sentence filling, then repair
sections too small to retrieve well, then add overlap. Going straight to
fixed-size windows would be simpler and would cut through every heading in the
corpus.
"""

from dataclasses import dataclass

from memoryos.adapters.chunking.sentences import split_sentences
from memoryos.adapters.chunking.tokens import count_tokens
from memoryos.application.ports import Chunker, ParsedDocument, TextChunk
from memoryos.domain.values import MemoryKind

Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    # Comfortably inside a sentence-transformer's window, with room for the
    # overlap prefix on top.
    target: int = 640
    # Boundaries are arbitrary, and a concept that straddles one would
    # otherwise appear in neither chunk in full.
    overlap: int = 80
    # Below this a chunk is a fragment: a heading and one line retrieves on
    # keyword luck rather than meaning.
    minimum: int = 120
    # Hard ceiling for prose. Code is exempt; see `chunk`.
    maximum: int = 1024


class StructuralChunker(Chunker):
    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._config = config or ChunkerConfig()

    @property
    def config(self) -> ChunkerConfig:
        return self._config

    @property
    def version(self) -> str:
        """Algorithm and parameters, together.

        The parameters are in the string because that is what makes improving
        the chunker a query — find every chunk carrying the old stamp and
        re-chunk only those — instead of a corpus-wide rebuild. It is also what
        answers "what produced this bad chunk?" six months from now.
        """
        config = self._config
        return (
            f"structural-v1:target={config.target}:overlap={config.overlap}"
            f":min={config.minimum}:max={config.maximum}"
        )

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        text = doc.text
        if not text.strip():
            return []

        sections = self._sections(doc)

        if doc.kind is MemoryKind.CODE:
            # Never split mid-function, even when the function is larger than
            # the target. Half a function embeds as neither a complete function
            # nor a coherent statement; size variance is the lesser cost.
            spans = self._merge_undersized(sections, text)
        else:
            spans = self._merge_undersized(
                [span for section in sections for span in self._fill(section, text)], text
            )

        return self._to_chunks(spans, text)

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
        if count_tokens(text[start:end]) <= self._config.target:
            return [section]

        sentences = split_sentences(text[start:end])
        if not sentences:
            return self._hard_split(section, text)

        spans: list[Span] = []
        cursor = start
        accumulated = 0

        for relative_start, relative_end in sentences:
            absolute_start = start + relative_start
            tokens = count_tokens(text[absolute_start : start + relative_end])

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
                if count_tokens(text[span[0] : span[1]]) <= self._config.maximum
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
            accumulated = count_tokens(text[cursor:index])
            if accumulated >= self._config.target and index < end:
                spans.append((cursor, index))
                cursor = index

        if cursor < end:
            spans.append((cursor, end))
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
                count_tokens(text[start:end]) < self._config.minimum
                and index + 1 < len(spans)
                and count_tokens(text[start : spans[index + 1][1]]) <= self._config.target
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
                count_tokens(text[start:end]) < self._config.minimum
                and count_tokens(text[previous_start:end]) <= self._config.target
            ):
                merged[-2] = (previous_start, end)
                merged.pop()

        return merged

    def _to_chunks(self, spans: list[Span], text: str) -> list[TextChunk]:
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
            tokens = count_tokens(content)
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
            if count_tokens(text[candidate:previous_end]) >= self._config.overlap:
                break

        # Overlap must not push a chunk over the ceiling; context is worth less
        # than staying inside the model's window.
        if count_tokens(text[chosen:end]) > self._config.maximum:
            return previous_end
        return chosen
