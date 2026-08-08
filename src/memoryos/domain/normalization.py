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
