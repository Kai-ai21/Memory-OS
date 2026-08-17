"""Markdown: front matter, headings, and the source kept intact."""

import re
from typing import Any

import frontmatter

from memoryos.adapters.parsers.text import decode
from memoryos.application.ports import ParsedDocument, Parser, StructureMarker
from memoryos.domain.normalization import normalize_text
from memoryos.domain.values import MemoryKind

SUFFIXES = (".md", ".markdown", ".mdown", ".mkd")
MEDIA_TYPES = {"text/markdown", "text/x-markdown"}

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.MULTILINE)
# Fences are tracked so that a `#` inside a code block is not mistaken for a
# heading — a comment in a shell snippet is not a section boundary.
_FENCE = re.compile(r"^\s*(?:```|~~~)", re.MULTILINE)


class MarkdownParser(Parser):
    def can_parse(self, media_type: str | None, external_key: str) -> bool:
        return external_key.lower().endswith(SUFFIXES) or (media_type or "") in MEDIA_TYPES

    def parse(
        self, data: bytes, *, media_type: str | None, external_key: str
    ) -> ParsedDocument:
        raw, codec = decode(data)
        post = frontmatter.loads(raw)

        metadata: dict[str, Any] = {"codec": codec}
        if post.metadata:
            metadata["front_matter"] = _jsonable(dict(post.metadata))

        # The markdown source is kept as-is. Stripping it to plain prose would
        # discard heading syntax, list markers, and code fences — all of which
        # an embedding model reads as signal about what kind of text this is.
        text = normalize_text(post.content)

        title = _title_from(post.metadata) or _first_h1(text)

        return ParsedDocument(
            text=text,
            title=title,
            metadata=metadata,
            structure=headings(text),
            kind=MemoryKind.NOTE,
        )


def headings(text: str) -> list[StructureMarker]:
    """Every ATX heading outside a code fence, with its offset into `text`."""
    fence_spans = _fenced_spans(text)
    markers: list[StructureMarker] = []

    for match in _ATX_HEADING.finditer(text):
        offset = match.start()
        if any(start <= offset < end for start, end in fence_spans):
            continue
        markers.append(
            StructureMarker(
                kind="heading",
                level=len(match.group(1)),
                char_offset=offset,
                label=match.group(2).strip() or None,
            )
        )
    return markers


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    fences = [match.start() for match in _FENCE.finditer(text)]
    # Fences pair up. An unclosed final fence runs to the end of the document,
    # which is also how a renderer treats it.
    spans = [(fences[i], fences[i + 1]) for i in range(0, len(fences) - 1, 2)]
    if len(fences) % 2 == 1:
        spans.append((fences[-1], len(text)))
    return spans


def _title_from(metadata: dict[str, Any]) -> str | None:
    title = metadata.get("title")
    return str(title).strip() or None if title is not None else None


def _first_h1(text: str) -> str | None:
    for marker in headings(text):
        if marker.level == 1:
            return marker.label
    return None


def _jsonable(value: Any) -> Any:
    """Front matter goes into a JSONB column, so dates and the like must go.

    Coerced rather than dropped: a `date:` field is worth keeping as a string,
    and the alternative is losing it because YAML happened to type it.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
