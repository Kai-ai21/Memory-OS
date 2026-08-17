"""The fallback parser: decode bytes, claim no structure."""

from typing import Any

from memoryos.application.ports import ParsedDocument, Parser
from memoryos.domain.normalization import normalize_text
from memoryos.domain.values import MemoryKind

# Tried in order. latin-1 is last because it cannot fail — every byte sequence
# is valid latin-1 — which makes it a guaranteed answer rather than a good one.
_CODECS = ("utf-8", "utf-8-sig", "latin-1")


def decode(data: bytes) -> tuple[str, str]:
    """Decode bytes, returning the text and the codec that worked.

    Never raises. A file that cannot be decoded cleanly is still worth storing
    imperfectly — refusing it would mean one mis-encoded file in a corpus stops
    being searchable at all — but the codec is recorded so that anyone puzzled
    by mojibake later can see what happened.
    """
    for codec in _CODECS:
        try:
            return data.decode(codec), codec
        except UnicodeDecodeError:
            continue
    # Unreachable: latin-1 accepts any byte. Kept so the function is total by
    # construction rather than by argument.
    return data.decode("utf-8", errors="replace"), "utf-8/replace"


# What this parser is *asked for* by name, as opposed to what it accepts — which
# is everything, because it is the registry's catch-all. The upload allow-list
# reads this one: a `.txt` is a file somebody meant to send, and "anything at all"
# is not a policy a door can have.
SUFFIXES = (".txt", ".text", ".log")
MEDIA_TYPES = frozenset({"text/plain"})


class TextParser(Parser):
    """Matches anything. Registered last, so it only sees what nothing else took."""

    def can_parse(self, media_type: str | None, external_key: str) -> bool:
        return True

    def parse(
        self, data: bytes, *, media_type: str | None, external_key: str
    ) -> ParsedDocument:
        text, codec = decode(data)
        metadata: dict[str, Any] = {"codec": codec}
        if codec != "utf-8":
            metadata["decode_fallback"] = True

        return ParsedDocument(
            text=normalize_text(text),
            title=None,
            metadata=metadata,
            structure=[],
            kind=MemoryKind.NOTE if external_key.endswith(".txt") else MemoryKind.OTHER,
        )
