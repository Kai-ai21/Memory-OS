"""Text normalization, and the second hash level built on it.

Pure Python. This is the piece that makes a cosmetic edit cost nothing: a file
saved with different line endings is genuinely different bytes and therefore a
genuinely new artifact, but its normalized text is identical, so chunking and
embedding are skipped entirely.
"""

import re
import unicodedata

from memoryos.domain.values import ContentHash

_BOM = "﻿"
_THREE_OR_MORE_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Canonical form: NFC, LF line endings, no trailing whitespace.

    Every rule here removes a difference that is invisible to a reader and
    meaningless to an embedding model, while leaving every difference that is
    not. Indentation survives; a trailing space does not.
    """
    # NFC first, so that composed and decomposed spellings of the same
    # character stop being different bytes. "é" written two ways is one word.
    text = unicodedata.normalize("NFC", text)

    # A BOM is an encoding artifact, not content. Only a leading one is
    # stripped — the same code point mid-document is a zero-width no-break
    # space that somebody may have meant.
    if text.startswith(_BOM):
        text = text[len(_BOM) :]

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Two blank lines is a paragraph break in every convention worth honouring;
    # nine is a formatting accident.
    text = _THREE_OR_MORE_BLANK_LINES.sub("\n\n", text)

    return text.strip("\n")


def normalized_hash(text: str) -> ContentHash:
    """The hash of the normalized text.

    The second level of the two-level scheme. `raw_artifacts.content_hash`
    answers "are these the same bytes"; this answers "is this the same text",
    which is the question that decides whether any downstream work is needed.
    """
    return ContentHash.of(text.encode("utf-8"))


# --------------------------------------------------------------------------
# Entity names (M3.1)
# --------------------------------------------------------------------------

# Whitespace runs, including the newlines a chunk boundary leaves in a name.
_WHITESPACE = re.compile(r"\s+")


def canonical_entity_name(name: str) -> str:
    """The form two mentions of the same surface string agree on.

    Deliberately minimal: casefold and collapse whitespace, nothing else. This
    is *not* resolution — it does not know that "Dr. Chen" and "Chen" are one
    person, or that "Postgres" and "PostgreSQL" are one technology. That is
    M3.2's problem, and M3.1's job is to state the size of it rather than to
    pre-empt it.

    Doing more here would hide the measurement. Stripping punctuation and
    suffixes at write time would collapse "neo4j" and "Neo4j." into one row and
    make the duplicate count M3.2 is scoped against look smaller than it is —
    the number would improve because the ruler shrank.

    Casefold rather than lower: it folds ß to ss and handles non-ASCII case
    pairs that `lower()` leaves alone, and an entity name is arbitrary text from
    an arbitrary corpus.
    """
    return _WHITESPACE.sub(" ", name).strip().casefold()
