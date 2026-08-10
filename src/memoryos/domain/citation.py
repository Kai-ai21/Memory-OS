"""What a result points at, precisely enough to check.

Pure Python. Two guardrails converge here — *every answer cites its memories*
and *retrieval stays explainable* — and both are unimplementable without
provenance carried end to end. Phase 1 built that; this surfaces it.

**The offsets are the whole thing, and they are subtle.** `char_start` and
`char_end` index the parent memory's *normalized* text, and they bound the
chunk's own span — not the text stored on the chunk. Every chunk after the first
carries an overlap head borrowed from its predecessor, so:

    chunk.content[prefix_chars:] == memory.content[char_start:char_end]

M1.4a made that relationship exact after an earlier version documented it
wrongly. A citation built on the old meaning highlights text near the answer
rather than the answer, and looks entirely plausible while doing it — which is
why `verify-citations` asserts the identity above rather than trusting this
docstring.

`version` is on the citation for the same class of reason. A memory is a
versioned row; a citation to "README.md" that does not say *which* README.md is
a citation to whatever it says today, which may no longer contain the sentence
that was quoted.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

# Where a context window is allowed to start or stop. Ordered by preference:
# a paragraph break reads as a clean boundary, a line break next, then the end
# of a sentence. Falling back to a word boundary is handled separately, because
# it is a different quality of answer — acceptable, but not a real boundary.
_BOUNDARY = re.compile(r"\n\n|\n|(?<=[.!?])\s")

# How far to look for a real boundary before giving up and using a word break.
# Roughly a sentence: beyond this the "context" stops being about the span.
_BOUNDARY_SEARCH = 120


@dataclass(frozen=True, slots=True)
class Excerpt:
    """A quotable window of a memory, with the matched span located inside it.

    `text` is what a reader sees. `span_start`/`span_end` index *into `text`*,
    not into the memory, so a UI can highlight without redoing the arithmetic —
    which is exactly the arithmetic that gets it wrong.
    """

    text: str
    span_start: int
    span_end: int
    # True when context was cut on that side, so a UI can show an ellipsis and a
    # reader can tell a quote from a complete passage.
    truncated_start: bool
    truncated_end: bool

    @property
    def span(self) -> str:
        """The matched text alone. The identity a citation is checked on."""
        return self.text[self.span_start : self.span_end]


@dataclass(frozen=True, slots=True)
class Citation:
    """One retrievable span, traceable to the exact version it came from."""

    memory_id: UUID
    source_name: str
    external_key: str
    chunk_ordinal: int
    char_start: int
    char_end: int
    prefix_chars: int
    # The chunk's own text with the borrowed overlap head stripped: exactly
    # `memory.content[char_start:char_end]`, and nothing a neighbour contributed.
    excerpt: str
    # What the chunker recorded about the enclosing definition, for code. The
    # difference between "somewhere in sync.py" and a reference somebody can
    # check.
    definition: str | None
    occurred_at: datetime | None
    version: int
    # The excerpt widened to surrounding context, when the memory's text was
    # available to widen it from. None means the caller asked for citations
    # without the extra fetch.
    context: Excerpt | None = None

    @property
    def locator(self) -> str:
        """A short human-readable pointer: `self::src/x.py#3 @1234-1456 (v2)`."""
        where = f"{self.source_name}::{self.external_key}#{self.chunk_ordinal}"
        return f"{where} @{self.char_start}-{self.char_end} (v{self.version})"


def build_excerpt(
    memory_text: str,
    char_start: int,
    char_end: int,
    *,
    context_chars: int = 200,
) -> Excerpt:
    """The matched span plus readable context, with the span located inside it.

    A bare chunk is often uninterpretable on its own — "the second approach"
    means nothing without the first — so the span is widened by up to
    `context_chars` on each side.

    **Boundaries snap to real breaks.** A window that starts mid-word looks
    broken and undermines the citation it is supposed to support, so the edges
    move to the nearest paragraph, line or sentence break within reach, and fall
    back to a word boundary when there is none. The span itself is never
    trimmed: context can be cut, the quote cannot.
    """
    if char_start < 0 or char_end < char_start:
        raise ValueError(f"invalid span {char_start}..{char_end}")

    length = len(memory_text)
    start = max(0, min(char_start, length))
    end = max(start, min(char_end, length))

    window_start = max(0, start - context_chars)
    window_end = min(length, end + context_chars)

    window_start = _snap_start(memory_text, window_start, start)
    window_end = _snap_end(memory_text, end, window_end)

    return Excerpt(
        text=memory_text[window_start:window_end],
        span_start=start - window_start,
        span_end=end - window_start,
        truncated_start=window_start > 0,
        truncated_end=window_end < length,
    )


def _snap_start(text: str, window_start: int, span_start: int) -> int:
    """Move the left edge to a clean boundary *near where the window begins*.

    Near the window edge, not near the span. Snapping to whichever boundary sits
    closest to the quote collapses the context to nothing whenever a sentence
    ends immediately before it — which is most of the time, because chunks are
    split on exactly those boundaries, and a citation with no context is the
    thing this function exists to prevent.

    So a boundary only counts if it falls in the first half of the window.
    Beyond that it is a boundary belonging to the span rather than to the
    context, and a word break at the window edge preserves more while still not
    starting mid-token.
    """
    if window_start <= 0:
        return 0

    region = text[window_start:span_start]
    reach = max(1, len(region) // 2)
    match = _BOUNDARY.search(region[:reach])
    if match is not None:
        return window_start + match.end()

    space = region.find(" ")
    if space != -1 and space <= _BOUNDARY_SEARCH:
        return window_start + space + 1
    return window_start


def _snap_end(text: str, span_end: int, window_end: int) -> int:
    """Move the right edge to a clean boundary near where the window ends.

    The mirror of `_snap_start`, for the same reason: a boundary immediately
    after the span would truncate the trailing context to nothing.
    """
    if window_end >= len(text):
        return len(text)

    region = text[span_end:window_end]
    reach = max(1, len(region) // 2)
    matches = list(_BOUNDARY.finditer(region, reach))
    if matches:
        return span_end + matches[-1].start()

    space = region.rfind(" ")
    if space != -1:
        return span_end + space
    return window_end
