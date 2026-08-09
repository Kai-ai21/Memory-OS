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
