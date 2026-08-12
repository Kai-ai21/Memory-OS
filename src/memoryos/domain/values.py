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


class EntityType(StrEnum):
    """What kind of thing an entity is.

    Lowercase values from `auto()`, matching `MemoryKind` and every other enum
    here; the CHECK constraint on `entities.type` is generated from this, so the
    database and Python cannot drift without the migration diff showing it.

    Seven types, and the set is deliberately closed. An open vocabulary is what
    an LLM produces if you let it — `person`, `people`, `human`, `individual`
    — and a type that means the same as another type is a filter that silently
    returns half its rows. The extractor is given this list and anything outside
    it is dropped rather than coerced, because a model that invented a type has
    also probably invented the entity.
    """

    PERSON = auto()
    TECHNOLOGY = auto()
    PROJECT = auto()
    ORGANIZATION = auto()
    CONCEPT = auto()
    FILE = auto()
    DECISION = auto()


class Predicate(StrEnum):
    """What one entity does to another.

    **Typed and directed, and both halves earn their place.** An untyped edge
    says two things are related, which is close to useless for traversal: at
    depth three every entity relates to every other, so a query that follows
    "related" returns the corpus. `USES` and `AUTHORED_BY` let a question be
    asked instead of a neighbourhood being dumped.

    Direction is not symmetric and cannot be flattened. "A supersedes B" and "B
    supersedes A" are contradictory claims about the same pair, and an edge that
    could not tell them apart would make the graph unable to answer the one
    question `SUPERSEDES` exists for.

    Closed, for the reason `EntityType` is: an open predicate vocabulary is what
    a language model produces when allowed, and `USES` beside `USED_BY` beside
    `UTILIZES` is three names for one traversal that each return a third of it.
    """

    USES = auto()
    DEPENDS_ON = auto()
    PART_OF = auto()
    AUTHORED_BY = auto()
    MENTIONS = auto()
    SUPERSEDES = auto()
    RELATES_TO = auto()


class MergeStrategy(StrEnum):
    """How a merge candidate was found.

    Recorded per merge because the strategies have very different precision, and
    a wrong merge is only diagnosable if you can tell which rule produced it.
    `EXACT` is near-certain; `EMBEDDING` is the one that needs a threshold and
    produces the errors worth reading; `MANUAL` is a person and outranks all of
    them.
    """

    EXACT = auto()
    EMBEDDING = auto()
    ALIAS = auto()
    LLM = auto()
    MANUAL = auto()


class MergeStatus(StrEnum):
    """Whether a merge is a proposal, in force, or undone.

    `PENDING` is the review queue: a candidate the resolver found and refused to
    apply on its own. It is a first-class state rather than an absence, because
    "we looked at this pair and were not sure" is information, and a system that
    discarded it would re-propose the same pair on every run forever.
    """

    PENDING = auto()
    APPLIED = auto()
    REVERTED = auto()


class GraphLabel(StrEnum):
    """The node kinds the graph projection knows about.

    Explicit values rather than `auto()`, which lowercases. Cypher labels are
    conventionally PascalCase and are case-sensitive, so `Memory` and `memory`
    are two different labels — a mismatch between a write and a read would not
    error, it would simply match nothing.

    `Memory` is deliberately a *projection*: `memory_id`, `external_key`, `kind`
    and `occurred_at`, and no content. Postgres stays the system of record, and a
    copy of the text here would be a second thing to keep correct and a second
    answer to give when the two disagree.
    """

    MEMORY = "Memory"
    ENTITY = "Entity"
    SOURCE = "Source"


class EdgeType(StrEnum):
    """The relationship types, defined now and populated in M3.1 and M3.3.

    Declared ahead of the code that writes them because the schema is the
    milestone: a traversal cannot be written against relationship types that are
    invented one at a time by whatever happens to need them, and a typo in a
    relationship type is invisible in Cypher — the pattern just matches nothing.
    """

    MENTIONS = "MENTIONS"
    RELATES_TO = "RELATES_TO"
    FROM_SOURCE = "FROM_SOURCE"


# The property that identifies a node of each label, which is also the property
# each label's uniqueness constraint is declared on. One mapping, because a
# `MERGE` that keyed on a different property than the constraint would create
# duplicates rather than fail.
IDENTITY_PROPERTY: dict[GraphLabel, str] = {
    GraphLabel.MEMORY: "memory_id",
    GraphLabel.ENTITY: "entity_id",
    GraphLabel.SOURCE: "source_id",
}

# Properties that identify a *relationship*, as opposed to describing one. The
# counterpart of `IDENTITY_PROPERTY` for edges, and here for the same reason: the
# projection merges on these, the divergence check keys on these, and two places
# that disagreed about the set would compare one edge against a different one.
#
# `predicate` is the only member, and it earns it. Neo4j merges one relationship
# per (type, start, end), so "sqlalchemy uses postgres" and "sqlalchemy depends_on
# postgres" collapsed into a single `RELATES_TO` whose predicate was whichever the
# projection happened to write last. Both claims are in the corpus.
#
# Nothing may be added here without a thought that does not apply to node
# properties: these names are interpolated into Cypher, because a property name
# cannot be a bound parameter any more than a label can.
EDGE_IDENTITY_PROPERTIES: frozenset[str] = frozenset({"predicate"})


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
