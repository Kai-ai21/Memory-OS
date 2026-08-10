"""Value objects and enumerations.

Pure Python. Nothing here may import SQLAlchemy, FastAPI, or perform I/O.
"""

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum, auto

# Shared by ContentHash and by the CHECK constraint on the SQL columns that store
# these digests, so the two definitions of "well formed" cannot drift apart. The
# syntax is valid for both `re` and Postgres' `~` operator.
HEX64_PATTERN = r"^[0-9a-f]{64}$"

_HEX64 = re.compile(HEX64_PATTERN)

# BLAKE2b truncated to 256 bits: 32 bytes, 64 hex characters.
_DIGEST_SIZE = 32


@dataclass(frozen=True, slots=True)
class ContentHash:
    """A BLAKE2b-256 digest of some content, as lowercase hex.

    Content addressing is what makes ingestion idempotent: re-reading an
    unchanged file produces the same hash and therefore the same row.
    """

    value: str

    def __post_init__(self) -> None:
        if _HEX64.fullmatch(self.value) is None:
            raise ValueError(
                f"content hash must be 64 lowercase hex characters, got {self.value!r}"
            )

    @classmethod
    def of(cls, data: bytes) -> "ContentHash":
        return cls(hashlib.blake2b(data, digest_size=_DIGEST_SIZE).hexdigest())

    def __str__(self) -> str:
        return self.value


class TimeProvenance(StrEnum):
    """How confidently `occurred_at` is known.

    Retrieval that ranks or filters by time needs to know whether a timestamp was
    stated by the source, extracted from text, taken from file metadata, or guessed.
    """

    DECLARED = auto()
    PARSED = auto()
    FILESYSTEM = auto()
    INFERRED = auto()
    UNKNOWN = auto()


class SourceKind(StrEnum):
    FILESYSTEM = auto()


class MemoryKind(StrEnum):
    NOTE = auto()
    DOCUMENT = auto()
    CODE = auto()
    EMAIL = auto()
    COMMIT = auto()
    MEETING = auto()
    BOOKMARK = auto()
    OTHER = auto()


class EventType(StrEnum):
    ARTIFACT_OBSERVED = auto()
    ITEM_DELETED = auto()


class Verdict(StrEnum):
    """What a human said about one result for one query.

    `MISSING` is the one that carries the most information and the only one that
    cannot be inferred from the ranking: it names a memory that *should* have
    been returned and was not. Precision can be measured from the first two;
    recall cannot be measured from anything else.
    """

    RELEVANT = auto()
    NOT_RELEVANT = auto()
    MISSING = auto()


class SearchMode(StrEnum):
    """Which retriever answers a query.

    They fail in opposite directions — the vector half is blind to opaque exact
    tokens, the keyword half is blind to paraphrase — and M2.1 measured that
    complement before M2.2 fused it, so that it was possible to tell which half
    carries which query. `SKIP LOCKED` goes from recall 0.000 to 1.000 between
    them; the paraphrase queries go the other way, to exactly 0.000.

    `HYBRID` is the product. The other two remain because the only way to know
    what fusion is doing is to be able to run each half alone.
    """

    VECTOR = auto()
    KEYWORD = auto()
    HYBRID = auto()


# One definition of "the default", used by the use case, the CLI, the API and
# the evaluation harness. Two places that must agree about a default is how a
# CLI and a test end up measuring different systems.
DEFAULT_SEARCH_MODE = SearchMode.HYBRID


class EmbeddingRole(StrEnum):
    """Which side of a retrieval a piece of text is on.

    Asymmetric models are trained to embed the two differently — bge prepends an
    instruction to queries and nothing to passages — so a vector is only
    meaningful for the role it was produced in. The role is part of the cache
    key for the same reason the model id is: without it, a query and a passage
    with identical text collide and one silently receives the other's vector.
    """

    PASSAGE = auto()
    QUERY = auto()
