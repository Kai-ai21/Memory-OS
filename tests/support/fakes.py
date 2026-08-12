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
    GraphNode,
    GraphPath,
    GraphReach,
    LanguageModel,
    MemoryNode,
    Reranker,
    SourceNode,
)
from memoryos.domain.values import (
    EDGE_IDENTITY_PROPERTIES,
    IDENTITY_PROPERTY,
    EdgeType,
    EntityType,
    GraphLabel,
    MemoryKind,
    Predicate,
)

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


class InMemoryGraphStore:
    """A `GraphStore` with `MERGE` semantics and no database behind it.

    Deliberately used where a real Neo4j would also work, and the reason is the
    one M3.0 wrote into `tests/integration/conftest.py`: Community Edition has
    exactly one user database, so anything asserting about the *whole* graph — a
    clear, a rebuild, a divergence check — would assert it against whatever graph
    the developer happens to have.

    Two kinds of test use this rather than Neo4j:

    * **Control flow.** Which replay scopes clear the graph is a decision made
      entirely in `ReplayCorpus`, and a real store would answer it with a
      `DETACH DELETE` over that one database.
    * **Divergence detection.** `graph_verify` compares two projections and is
      pure once they are in hand. Corrupting a node and requiring a non-zero exit
      needs a whole graph whose every node is the test's own.

    What it deliberately does *not* stand in for is Cypher. The adapter's
    fidelity — that `write` puts in exactly what `all_nodes` and `all_edges` read
    back — is checked against a real Neo4j in `test_graph.py`, because a fake
    that agreed with itself would prove only that.

    Keys are `(label, key)` and writes overwrite, which is what `MERGE` does. A
    list of writes would let a double-projection look like twice the graph.
    """

    def __init__(self) -> None:
        self.clears = 0
        self._nodes: dict[tuple[GraphLabel, str], GraphNode] = {}
        # Keyed by (type, start, end) plus whatever the edge merges on: Neo4j's
        # `MERGE (a)-[r:T {k: v}]->(b)` is one relationship per that tuple, and
        # `SET r +=` overwrites the rest of its properties.
        self._edges: dict[tuple[str, ...], GraphEdge] = {}

    # -- writes --------------------------------------------------------

    async def upsert_memory(self, node: MemoryNode) -> None:
        self._nodes[(GraphLabel.MEMORY, str(node.memory_id))] = GraphNode(
            label=GraphLabel.MEMORY,
            key=str(node.memory_id),
            properties={
                "memory_id": str(node.memory_id),
                "external_key": node.external_key,
                "kind": node.kind.value,
                "occurred_at": node.occurred_at,
            },
        )

    async def upsert_entity(self, node: EntityNode) -> None:
        self._nodes[(GraphLabel.ENTITY, str(node.entity_id))] = GraphNode(
            label=GraphLabel.ENTITY,
            key=str(node.entity_id),
            properties={
                "entity_id": str(node.entity_id),
                "name": node.name,
                "canonical_name": node.canonical_name,
                "type": node.type,
                "confidence": node.confidence,
            },
        )

    async def upsert_source(self, node: SourceNode) -> None:
        self._nodes[(GraphLabel.SOURCE, str(node.source_id))] = GraphNode(
            label=GraphLabel.SOURCE,
            key=str(node.source_id),
            properties={
                "source_id": str(node.source_id),
                "name": node.name,
                "kind": node.kind,
            },
        )

    async def link(self, edge: GraphEdge) -> None:
        # Endpoints are merged, exactly as the adapter does, so an edge written
        # before its node's own upsert leaves a node carrying only its identity.
        # Reproduced rather than tidied away: it is the behaviour the ordering in
        # `graph_projection.write` exists to avoid, and a fake that created
        # complete nodes would make that ordering untestable.
        for node in (edge.start, edge.end):
            self._nodes.setdefault(
                (node.label, node.key),
                GraphNode(
                    label=node.label,
                    key=node.key,
                    properties={IDENTITY_PROPERTY[node.label]: node.key},
                ),
            )
        self._edges[_edge_key(edge)] = edge

    async def prune_memories(self, memory_ids: Sequence[UUID]) -> int:
        return self._prune(GraphLabel.MEMORY, memory_ids)

    async def prune_entities(self, entity_ids: Sequence[UUID]) -> int:
        return self._prune(GraphLabel.ENTITY, entity_ids)

    def _prune(self, label: GraphLabel, ids: Sequence[UUID]) -> int:
        keys = {str(value) for value in ids}
        removed = 0
        for key in keys:
            if self._nodes.pop((label, key), None) is not None:
                removed += 1
        # Detached, so every edge touching a removed node goes with it.
        for edge_key, edge in list(self._edges.items()):
            if edge.start.key in keys or edge.end.key in keys:
                del self._edges[edge_key]
        return removed

    async def clear(self) -> None:
        self.clears += 1
        self._nodes.clear()
        self._edges.clear()

    # -- reads ---------------------------------------------------------

    async def mention_edges(
        self,
        *,
        memory_ids: Sequence[UUID] = (),
        entity_ids: Sequence[UUID] = (),
    ) -> list[tuple[UUID, UUID]]:
        memories = {str(value) for value in memory_ids}
        entities = {str(value) for value in entity_ids}
        return [
            (UUID(edge.start.key), UUID(edge.end.key))
            for edge in self._edges.values()
            if edge.type is EdgeType.MENTIONS
            and (edge.start.key in memories or edge.end.key in entities)
        ]

    async def all_nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    async def all_edges(self) -> list[GraphEdge]:
        return list(self._edges.values())

    async def neighbours(
        self, entity_id: UUID, *, depth: int = 2, limit: int = 50
    ) -> list[GraphPath]:
        return []

    async def reach(
        self,
        seed_entity_ids: Sequence[UUID],
        *,
        depth: int = 2,
        exclude_entity_ids: Sequence[UUID] = (),
        limit: int = 200,
    ) -> list[GraphReach]:
        """Breadth-first over the same edges the Cypher walks, hubs excluded.

        Reimplemented rather than stubbed, because the tests it serves are about
        what the *expansion* does with a traversal — which routes it keeps, how it
        weights them, whether a hub can bridge a path — and a stub returning nothing
        would let all of that pass untested. What it deliberately does not
        establish is that Cypher agrees with it; that is
        `test_graph.py::test_the_traversal_reaches_what_the_fake_reaches`.

        Undirected and bounded at `2 * depth` graph hops, matching the adapter: an
        entity reaches another through a `RELATES_TO` edge or through a memory that
        mentions both.
        """
        excluded = {str(value) for value in exclude_entity_ids}
        adjacency: dict[str, list[tuple[str, GraphLabel]]] = {}
        for edge in self._edges.values():
            if edge.type is EdgeType.FROM_SOURCE:
                # Excluded from the walk for the reason the Cypher excludes it: it
                # would make every memory of one source two hops from every other.
                continue
            adjacency.setdefault(edge.start.key, []).append((edge.end.key, edge.end.label))
            adjacency.setdefault(edge.end.key, []).append((edge.start.key, edge.start.label))

        mentions: dict[str, list[GraphEdge]] = {}
        for edge in self._edges.values():
            if edge.type is EdgeType.MENTIONS:
                mentions.setdefault(edge.end.key, []).append(edge)

        found: list[GraphReach] = []
        for seed in sorted(str(value) for value in seed_entity_ids):
            if seed in excluded:
                continue

            def emit(entity: str, hops: int, route: tuple[str, ...]) -> None:
                for edge in mentions.get(entity, []):
                    found.append(
                        GraphReach(
                            memory_id=UUID(edge.start.key),
                            chunk_ordinal=int(edge.properties.get("chunk_ordinal", 0)),
                            entity_id=UUID(entity),
                            hops=hops,
                            route=route,
                        )
                    )

            # The seed itself, at two hops: entity-memory-entity is the shortest
            # route from an entity back to itself, and it is the case "another
            # memory mentions the same thing retrieval found" — the most valuable
            # expansion there is. The Cypher reaches it by not excluding
            # `target = seed`; here it has to be emitted before the walk, because a
            # breadth-first search does not revisit its own start.
            emit(seed, 2, (self._name_of(seed),))

            frontier: list[tuple[str, int, tuple[str, ...]]] = [
                (seed, 0, (self._name_of(seed),))
            ]
            visited = {seed}
            while frontier:
                key, hops, route = frontier.pop(0)
                if hops >= 2 * depth:
                    continue
                for neighbour, label in sorted(adjacency.get(key, [])):
                    if neighbour in visited or neighbour in excluded:
                        continue
                    visited.add(neighbour)
                    is_entity = label is GraphLabel.ENTITY
                    name = self._name_of(neighbour)
                    onward = (
                        (*route, name)
                        if is_entity and (not route or route[-1] != name)
                        else route
                    )
                    frontier.append((neighbour, hops + 1, onward))
                    if is_entity:
                        emit(neighbour, hops + 1, onward)
        found.sort(key=lambda reach: (reach.hops, str(reach.memory_id), str(reach.entity_id)))
        return found[:limit]

    def _name_of(self, key: str) -> str:
        node = self._nodes.get((GraphLabel.ENTITY, key))
        return "" if node is None else str(node.properties.get("name", ""))

    # -- for assertions ------------------------------------------------

    @property
    def memories(self) -> list[GraphNode]:
        return [node for node in self._nodes.values() if node.label is GraphLabel.MEMORY]

    @property
    def entities(self) -> list[GraphNode]:
        return [node for node in self._nodes.values() if node.label is GraphLabel.ENTITY]

    @property
    def edges(self) -> list[GraphEdge]:
        return list(self._edges.values())


def _edge_key(edge: GraphEdge) -> tuple[str, ...]:
    """What `MERGE` treats as one relationship: type, endpoints, identity properties.

    Reproduced rather than simplified, because the difference is a real defect this
    fake would otherwise hide: keying on the endpoints alone collapses two
    predicates between one pair into a single edge, which is exactly what Neo4j did
    before `GraphEdge.identity` existed.
    """
    return (
        edge.type.value,
        edge.start.key,
        edge.end.key,
        *(
            str(edge.properties[name])
            for name in sorted(EDGE_IDENTITY_PROPERTIES)
            if name in edge.properties
        ),
    )


class UnreachableGraphStore(InMemoryGraphStore):
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
