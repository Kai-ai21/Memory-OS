"""Test doubles for ports this project owns.

A fake embedder is legitimate precisely because the `Embedder` port is ours:
the contract it honours is one we wrote and can change. Mocking
sentence-transformers itself would be a different thing — it would assert our
beliefs about that library rather than its behaviour, and those beliefs are
exactly what the one slow test exists to check.
"""

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from itertools import pairwise
from uuid import UUID, uuid4

from memoryos.application.ports import (
    Embedder,
    EntityNode,
    EntityRef,
    ExtractedEntity,
    ExtractedRelationship,
    GraphEdge,
    GraphPath,
    LanguageModel,
    MemoryNode,
    Reranker,
)
from memoryos.domain.values import EntityType, MemoryKind, Predicate

FAKE_MODEL_ID = "fake/deterministic@1"

_TOKEN = re.compile(r"\w+|[^\w\s]")


class FakeEmbedder(Embedder):
    """Deterministic unit vectors derived from a hash of the text.

    Same text in, same vector out, so cache behaviour is testable. Different
    text gives an unrelated direction, which is all any test here needs — no
    test asserts that the geometry is *meaningful*, because a fake cannot
    establish that.
    """

    def __init__(
        self,
        model_id: str = FAKE_MODEL_ID,
        dimension: int = 384,
        *,
        broken_dimension: int | None = None,
        max_sequence_tokens: int = 512,
        query_prefix: str = "",
    ) -> None:
        self._model_id = model_id
        self._dimension = dimension
        # Configurable so tests can drive the window boundary — including a
        # deliberately tiny one — without loading a model.
        self._window = max_sequence_tokens
        # When set, `embed_passage` returns vectors of this width instead — for
        # the test that a mismatch is caught before anything is written.
        self._broken_dimension = broken_dimension
        # Empty by default, matching the port's symmetric default. A test that
        # needs the two roles to diverge sets one.
        self._query_prefix = query_prefix
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalizes(self) -> bool:
        return True

    @property
    def max_sequence_tokens(self) -> int:
        return self._window

    def count_tokens(self, text: str) -> int:
        """A deterministic stand-in for WordPiece.

        Words and punctuation, plus an extra token per long word, so that the
        fake overcounts dense identifiers roughly the way a real tokenizer
        does. It does not need to match any model; it needs to be stable and
        to punish long tokens.
        """
        pieces = _TOKEN.findall(text)
        return sum(1 + len(piece) // 8 for piece in pieces)

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)

    def embed_passage(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        width = self._broken_dimension or self._dimension
        return [self._vector(text, width) for text in texts]

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._query_prefix:
            return self.embed_passage(texts)
        return self.embed_passage([self._query_prefix + text for text in texts])

    def _vector(self, text: str, width: int) -> list[float]:
        seed = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
        # Stretch the digest to the required width, then normalise, so the
        # fake honours the port's `normalizes = True` claim.
        raw = [seed[index % len(seed)] / 255.0 - 0.5 for index in range(width)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]


class FakeReranker(Reranker):
    """Deterministic pair scores, so a test never loads a cross-encoder.

    Legitimate for the same reason `FakeEmbedder` is: `Reranker` is a port this
    project owns, and what the fake honours is a contract we wrote. It cannot
    establish that reranking *improves* anything — that is what the golden set
    and the one slow test are for — but it can establish that the pipeline
    reorders by whatever the reranker says, truncates before asking, and records
    the answer.

    The default scores by input position descending, which reverses the
    shortlist. A reversal is the strongest possible signal that the pipeline
    honours the reranker rather than quietly keeping the fused order: any
    partial ordering could be a coincidence, and the identity ordering would be
    indistinguishable from ignoring the model entirely.
    """

    def __init__(
        self,
        model_id: str = "fake/cross-encoder@1",
        *,
        max_length: int = 64,
        scorer: Callable[[str, str], float] | None = None,
    ) -> None:
        self._model_id = model_id
        self._max_length = max_length
        self._scorer = scorer
        self.calls: list[tuple[str, list[str]]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_length(self) -> int:
        return self._max_length

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        self.calls.append((query, list(documents)))
        if self._scorer is not None:
            return [self._scorer(query, document) for document in documents]
        # Descending in input position: the first candidate scores lowest, so a
        # pipeline that honours the reranker returns the shortlist reversed.
        return [float(index) for index in range(len(documents))]

    @property
    def pairs_scored(self) -> int:
        return sum(len(documents) for _, documents in self.calls)


class FakeLanguageModel(LanguageModel):
    """Canned completions, so a test never makes a network call.

    Legitimate for the same reason the other fakes are: `LanguageModel` is a
    port this project owns. What it cannot establish is whether a real model
    stays inside its evidence — that is what the guardrails and the one slow
    test are for. What it can establish is that the pipeline assembles the
    right context, verifies what comes back, and reports it honestly however
    badly the model behaves.

    `responses` are returned in order and the last one repeats, so a test can
    script a sequence without counting calls.
    """

    def __init__(
        self,
        *responses: str,
        model_id: str = "fake/llm@1",
        raises: Exception | None = None,
    ) -> None:
        self._responses = list(responses) or ["The passages do not cover this."]
        self._model_id = model_id
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    async def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        self.calls.append((system, user))
        if self._raises is not None:
            raise self._raises
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]

    @property
    def last_user_prompt(self) -> str:
        return self.calls[-1][1]


class RecordingGraphStore:
    """A `GraphStore` that records what was asked of it and stores nothing.

    Deliberately used where a real Neo4j would also work. What the replay tests
    need to establish is *which scopes clear the graph* — a decision made
    entirely in `ReplayCorpus`, and one that a real store would answer with a
    `DETACH DELETE` over the whole database. Neo4j Community Edition has one
    user database, so that assertion would be run against whatever graph the
    developer happens to have, to prove something about control flow that never
    needed a database to be proved.
    """

    def __init__(self) -> None:
        self.clears = 0
        self.memories: list[MemoryNode] = []
        self.entities: list[EntityNode] = []
        self.edges: list[GraphEdge] = []

    async def upsert_memory(self, node: MemoryNode) -> None:
        self.memories.append(node)

    async def upsert_entity(self, node: EntityNode) -> None:
        self.entities.append(node)

    async def link(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    async def neighbours(
        self, entity_id: UUID, *, depth: int = 2, limit: int = 50
    ) -> list[GraphPath]:
        return []

    async def clear(self) -> None:
        self.clears += 1


class UnreachableGraphStore(RecordingGraphStore):
    """A graph that raises on `clear`, for the case a replay must not swallow."""

    async def clear(self) -> None:
        raise ConnectionError("no route to the graph")


class FakeEntityExtractor:
    """Deterministic entities from a regex, so tests never call a model.

    Legitimate for the same reason the other fakes are: `EntityExtractor` is a
    port this project owns. What it cannot establish is whether a real model
    extracts anything *useful* — that is what M3.1's corpus measurement is for.
    What it can establish is that the pipeline stores what it is given, skips
    what it has already done, and drops what it cannot find in the text.

    Capitalised words by default, which is a terrible entity extractor and a
    perfectly good test double: it is reproducible, and every name it returns is
    by construction present in the text at a known offset.

    `phantom_names` is the interesting knob. It makes the fake return names that
    are *not* in the text, which is what a real model does when it paraphrases
    or invents — the exact case the offset verification exists to catch.
    """

    def __init__(
        self,
        *,
        version: str = "fake-extractor@1",
        pattern: str = r"\b[A-Z][a-zA-Z0-9]{2,}\b",
        entity_type: EntityType = EntityType.CONCEPT,
        confidence: float = 0.9,
        phantom_names: Sequence[str] = (),
        predicate: Predicate = Predicate.USES,
        phantom_relationship: bool = False,
    ) -> None:
        self._version = version
        self._pattern = re.compile(pattern)
        self._type = entity_type
        self._confidence = confidence
        self._phantoms = list(phantom_names)
        self._predicate = predicate
        self._phantom_relationship = phantom_relationship
        self.calls: list[str] = []
        self.relationship_calls: list[str] = []

    @property
    def version(self) -> str:
        return self._version

    async def extract(self, text: str, *, kind: MemoryKind) -> list[ExtractedEntity]:
        self.calls.append(text)

        found = [
            ExtractedEntity(
                name=match.group(0),
                type=self._type,
                confidence=self._confidence,
                char_start=match.start(),
                char_end=match.end(),
            )
            for match in self._pattern.finditer(text)
        ]

        # Names that are not in the text, at offsets that are therefore lies.
        # A real extractor verifies and drops these; this fake deliberately does
        # not, so the layer above can be tested on what it does with them.
        found.extend(
            ExtractedEntity(
                name=name,
                type=self._type,
                confidence=self._confidence,
                char_start=0,
                char_end=len(name),
            )
            for name in self._phantoms
        )
        return found

    async def extract_relationships(
        self, text: str, entities: Sequence[EntityRef]
    ) -> list[ExtractedRelationship]:
        """Chain the supplied entities with a fixed predicate.

        Deterministic and structurally valid: every endpoint comes from the
        supplied list, so the fake exercises the storage path without ever
        producing the invented endpoint the real adapter has to guard against.
        `phantom_relationship` is how a test asks for that failure explicitly.
        """
        self.relationship_calls.append(text)
        pairs = list(pairwise(entities))
        found = [
            ExtractedRelationship(
                subject_id=left.entity_id,
                object_id=right.entity_id,
                predicate=self._predicate,
                confidence=self._confidence,
                evidence=f"{left.name} -> {right.name}",
            )
            for left, right in pairs
        ]
        if self._phantom_relationship and entities:
            # An endpoint that is not in the supplied set, which is what a real
            # model occasionally invents.
            found.append(
                ExtractedRelationship(
                    subject_id=entities[0].entity_id,
                    object_id=uuid4(),
                    predicate=self._predicate,
                    confidence=0.99,
                    evidence="an entity nobody supplied",
                )
            )
        return found
