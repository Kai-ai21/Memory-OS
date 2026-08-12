"""Graph expansion, and the two guards that decide whether it works.

**Most of this is hermetic on purpose.** What M3.5 has to establish is not that a
graph traversal returns rows — `test_graph.py` does that against real Cypher — but
that the expansion *suppresses hubs*, *bounds depth*, and *contributes nothing at
weight zero*. Those are properties of arithmetic over a known graph, and a real
Neo4j makes none of them more true while making the graph a shared mutable
fixture. So the graph here is `InMemoryGraphStore`, whose traversal is the same
breadth-first walk over the same edge types, and whose agreement with Cypher is
asserted separately in `test_graph.py`.

The corpus is written directly into Postgres rather than ingested. Expansion reads
chunks and mention counts and nothing else, and going through the connector,
parser and chunker to obtain them would make the fixture's shape a fact about the
chunker.
"""

from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application import graph_projection
from memoryos.application.graph_expand import (
    EntityStats,
    ExpandThroughGraph,
    GraphCandidates,
)
from memoryos.application.ports import ScoredChunk
from memoryos.application.search import FusionWeights, fuse
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash, EntityType, MemoryKind, TimeProvenance
from tests.integration.conftest import OCCURRED_AT
from tests.support.fakes import InMemoryGraphStore

pytestmark = pytest.mark.integration

CHUNKER = "test-chunker@1"


class Corpus:
    """A corpus described by which entities each memory mentions.

    Deliberately not a fixture over real files: what every test here varies is the
    *mention structure*, and expressing that as `{"a.md": ["queue", "hub"]}` is the
    difference between a test whose setup states its hypothesis and one whose setup
    is four paragraphs of prose being chunked.
    """

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions
        self.memories: dict[str, UUID] = {}
        self.entities: dict[str, UUID] = {}
        self.chunks: dict[str, UUID] = {}

    async def create(self, mentions: dict[str, list[str]]) -> None:
        """One memory per key, one chunk each, one mention per named entity.

        Offsets are located in the text rather than invented, so every mention
        satisfies the invariant `entity_mentions` exists to carry —
        `content[char_start:char_end] == name` — and the fixture cannot pass a test
        that a real extraction would fail.
        """
        source_id = new_id()
        digest = ContentHash.of(f"corpus-{source_id}".encode())
        self.entities = {name: new_id() for name in _named(mentions)}

        async with self._sessions.begin() as session:
            session.add(
                models.Source(
                    id=source_id, kind="filesystem", name=f"corpus-{source_id}"
                )
            )
            session.add(
                models.RawArtifact(content_hash=digest.value, byte_size=len(digest.value))
            )
            for name, entity_id in self.entities.items():
                session.add(
                    models.Entity(
                        id=entity_id,
                        name=name,
                        canonical_name=name.lower(),
                        type=EntityType.TECHNOLOGY.value,
                        confidence=0.9,
                    )
                )

            # Flushed in stages, because these models carry foreign keys without
            # ORM relationships — so SQLAlchemy's unit of work has nothing to
            # topologically sort by and will happily insert a mention before the
            # chunk it points at.
            await session.flush()

            texts: dict[str, str] = {}
            for key, names in mentions.items():
                memory_id = new_id()
                chunk_id = new_id()
                self.memories[key] = memory_id
                self.chunks[key] = chunk_id
                text = f"{key} mentions " + " ".join(names)
                texts[key] = text
                session.add(
                    models.Memory(
                        id=memory_id,
                        source_id=source_id,
                        external_key=key,
                        content_hash=digest.value,
                        kind=MemoryKind.NOTE.value,
                        version=1,
                        is_current=True,
                        # A date and a provenance together: the CHECK constraint
                        # refuses `filesystem` provenance with no timestamp, which
                        # is the invariant that keeps an invented date out.
                        occurred_at=OCCURRED_AT,
                        occurred_at_source=TimeProvenance.FILESYSTEM.value,
                    )
                )
                session.add(
                    models.MemoryChunk(
                        id=chunk_id,
                        memory_id=memory_id,
                        ordinal=0,
                        content=text,
                        token_count=len(text.split()),
                        char_start=0,
                        char_end=len(text),
                        chunker_version=CHUNKER,
                        content_hash=ContentHash.of(text.encode()).value,
                    )
                )
            await session.flush()

            for key, names in mentions.items():
                text = texts[key]
                for name in names:
                    start = text.index(name)
                    session.add(
                        models.EntityMention(
                            id=new_id(),
                            entity_id=self.entities[name],
                            memory_id=self.memories[key],
                            chunk_id=self.chunks[key],
                            char_start=start,
                            char_end=start + len(name),
                            confidence=0.9,
                            extractor_version="expand-test@1",
                        )
                    )


def _named(mentions: dict[str, list[str]]) -> list[str]:
    """Every entity name, in first-seen order so ids are stable per run."""
    seen: list[str] = []
    for names in mentions.values():
        for name in names:
            if name not in seen:
                seen.append(name)
    return seen


@pytest.fixture
async def corpus(sessions: async_sessionmaker[AsyncSession]) -> Corpus:
    return Corpus(sessions)


async def graph_for(
    sessions: async_sessionmaker[AsyncSession],
) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    await graph_projection.rebuild(sessions, store)
    return store


# --------------------------------------------------------------------------
# B5: hub entities above the threshold are excluded
# --------------------------------------------------------------------------


async def test_a_hub_entity_cannot_bridge_an_expansion(
    corpus: Corpus, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The guard the milestone says decides whether any of this works.

    `hub` is mentioned in six of eight memories. Without suppression it connects
    all six to each other, so expanding from any one of them returns the rest and
    calls it evidence — which is not a ranking, it is the corpus in a different
    order. `rare` is mentioned twice, which is the connection worth having.

    `hub_ratio=0.4` rather than the shipped 0.10, so the arithmetic is visible in
    the test: eight reachable memories, threshold `ceil(8 * 0.4) = 4`, and the two
    entities sit either side of it by construction.
    """
    await corpus.create(
        {
            "seed.md": ["queue", "hub"],
            "connected.md": ["queue", "rare"],
            "reached-by-rare.md": ["rare"],
            "hub-only-a.md": ["hub"],
            "hub-only-b.md": ["hub"],
            "hub-only-c.md": ["hub"],
            "hub-only-d.md": ["hub"],
            "hub-only-e.md": ["hub"],
        }
    )
    graph = await graph_for(sessions)
    expand = ExpandThroughGraph(sessions, graph, hub_ratio=0.4, depth=2)

    stats = await expand.entity_stats()
    assert stats.documents == 8
    assert stats.hub_threshold == 4
    assert corpus.entities["hub"] in stats.hubs
    assert corpus.entities["rare"] not in stats.hubs
    assert corpus.entities["queue"] not in stats.hubs

    candidates = await expand([corpus.memories["seed.md"]])
    reached = {chunk.memory_id for chunk in candidates.chunks}

    assert corpus.memories["connected.md"] in reached, "queue is not a hub; follow it"
    assert corpus.memories["reached-by-rare.md"] in reached, "and one hop further"
    for key in ("hub-only-a.md", "hub-only-b.md", "hub-only-c.md"):
        assert corpus.memories[key] not in reached, f"{key} is only reachable via a hub"


async def test_without_suppression_the_hub_returns_the_corpus(
    corpus: Corpus, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The counterfactual, without which the test above proves nothing.

    A guard that excluded nothing would pass every assertion above if the graph
    simply could not reach those memories. Raising the ratio until the hub is not a
    hub has to bring them all back.
    """
    await corpus.create(
        {
            "seed.md": ["queue", "hub"],
            "hub-only-a.md": ["hub"],
            "hub-only-b.md": ["hub"],
            "hub-only-c.md": ["hub"],
            "hub-only-d.md": ["hub"],
            "hub-only-e.md": ["hub"],
        }
    )
    graph = await graph_for(sessions)
    # A ratio above 1.0 is a threshold no entity can reach, which is the cleanest
    # way to express "suppression disabled" without pretending a hub is not one.
    permissive = ExpandThroughGraph(sessions, graph, hub_ratio=2.0, depth=2)

    stats = await permissive.entity_stats()
    candidates = await permissive([corpus.memories["seed.md"]])
    reached = {chunk.memory_id for chunk in candidates.chunks}

    assert corpus.entities["hub"] not in stats.hubs
    assert len(reached) == 5, "every memory the hub touches, which is the failure mode"


async def test_a_rarer_entity_outranks_a_commoner_one(
    corpus: Corpus, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """IDF on graph nodes, which is the other half of hub suppression.

    Suppression is a cliff; this is the slope below it. A memory connected through
    an entity that appears twice is better evidence than one connected through an
    entity that appears in a third of the corpus, and the ranking has to say so
    even when neither is over the threshold.
    """
    await corpus.create(
        {
            "seed.md": ["queue", "common", "rare"],
            "via-rare.md": ["rare"],
            "via-common-a.md": ["common"],
            "via-common-b.md": ["common"],
            "filler-a.md": ["unrelated"],
            "filler-b.md": ["unrelated"],
            "filler-c.md": ["unrelated"],
            "filler-d.md": ["unrelated"],
            "filler-e.md": ["unrelated"],
            "filler-f.md": ["unrelated"],
        }
    )
    graph = await graph_for(sessions)
    expand = ExpandThroughGraph(sessions, graph, hub_ratio=0.5, depth=1)

    candidates = await expand([corpus.memories["seed.md"]])
    order = [chunk.memory_id for chunk in candidates.chunks]

    assert order[0] == corpus.memories["via-rare.md"], (
        "the rarer connection ranks first: "
        + str([(str(c.memory_id)[:8], round(c.score, 3)) for c in candidates.chunks])
    )


async def test_depth_bounds_how_far_the_expansion_reaches(
    corpus: Corpus, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """One entity hop, then two, and the third is out of reach.

    An *entity* hop is either a typed relationship or a shared memory, so from the
    seed's `alpha` the first hop reaches `beta` — co-mentioned with it in
    `shares-alpha.md` — and every memory that mentions `beta`. The second hop
    reaches `gamma` the same way. The third would reach `delta`, and does not.

    The bound is the second of the two guards, and the reason it is a bound rather
    than a preference: each hop multiplies by the branching factor, so depth 3 on a
    connected graph reaches most of the corpus and returns it as a ranking.
    """
    await corpus.create(
        {
            "seed.md": ["alpha"],
            "shares-alpha.md": ["alpha", "beta"],
            "shares-beta.md": ["beta", "gamma"],
            "shares-gamma.md": ["gamma", "delta"],
            "shares-delta.md": ["delta"],
            "filler-a.md": ["unrelated"],
            "filler-b.md": ["unrelated"],
            "filler-c.md": ["unrelated"],
            "filler-d.md": ["unrelated"],
            "filler-e.md": ["unrelated"],
        }
    )
    graph = await graph_for(sessions)

    shallow = await ExpandThroughGraph(sessions, graph, depth=1, hub_ratio=0.5)(
        [corpus.memories["seed.md"]]
    )
    deep = await ExpandThroughGraph(sessions, graph, depth=2, hub_ratio=0.5)(
        [corpus.memories["seed.md"]]
    )

    assert {chunk.memory_id for chunk in shallow.chunks} == {
        corpus.memories["shares-alpha.md"],
        corpus.memories["shares-beta.md"],
    }, "one entity hop: beta, and both memories that mention it"
    reached = {chunk.memory_id for chunk in deep.chunks}
    assert corpus.memories["shares-gamma.md"] in reached, "two hops reaches gamma"
    assert corpus.memories["shares-delta.md"] not in reached, (
        "depth 2 is two entity hops, not three"
    )


async def test_an_unreachable_graph_degrades_to_no_candidates(
    corpus: Corpus, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The graph is a projection; a search must not fail because it is down.

    Returning nothing is the same answer an empty projection gives, which is what
    makes this a degradation rather than a special case: fusion collapses to the
    hybrid ranking with no branch anywhere deciding that it should.
    """

    class Unreachable(InMemoryGraphStore):
        async def mention_edges(self, **_: object) -> list[tuple[UUID, UUID]]:
            raise ConnectionError("no route to the graph")

    await corpus.create({"seed.md": ["queue"], "other.md": ["queue"]})
    expand = ExpandThroughGraph(sessions, Unreachable())

    candidates = await expand([corpus.memories["seed.md"]])

    assert candidates.chunks == []
    assert candidates.routes == {}


# --------------------------------------------------------------------------
# B5: graph weight 0 reproduces M2.3 exactly
# --------------------------------------------------------------------------


def _chunk(ordinal: int, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=new_id(),
        memory_id=new_id(),
        ordinal=ordinal,
        text=f"chunk {ordinal}",
        score=score,
        char_start=0,
        char_end=8,
    )


def test_graph_weight_zero_reproduces_m2_3_exactly() -> None:
    """The requirement, and it is stronger than "the weighted term is zero".

    Graph expansion is the one ranking that *introduces* candidates. A chunk that
    reached the fused set with a zero-weighted contribution would score zero, sort
    below everything, and change the result — so at weight zero the candidates must
    not enter the fusion at all. Asserted against a `fuse` call that was never
    given an expansion, which is M2.3's fusion exactly.

    Hermetic, because what is under test is `fuse`. Going through the search use
    case would test the same arithmetic through a database.
    """
    vector = [_chunk(index, 0.9 - index / 100) for index in range(4)]
    keyword = [vector[1], vector[3], _chunk(9, 4.2)]
    introduced = [_chunk(20, 5.0), _chunk(21, 4.0)]
    expansion = GraphCandidates(
        chunks=introduced,
        routes={str(chunk.chunk_id): ("queue", "worker") for chunk in introduced},
    )

    m2_3 = fuse(vector, keyword, weights=FusionWeights(graph=0.0))
    with_expansion_off = fuse(
        vector, keyword, weights=FusionWeights(graph=0.0), graph=expansion
    )
    with_expansion_on = fuse(
        vector, keyword, weights=FusionWeights(graph=0.5), graph=expansion
    )

    assert [chunk.chunk_id for chunk in with_expansion_off] == [
        chunk.chunk_id for chunk in m2_3
    ]
    assert [chunk.score for chunk in with_expansion_off] == [
        chunk.score for chunk in m2_3
    ]
    assert all(
        chunk.breakdown is not None and chunk.breakdown.graph_rank is None
        for chunk in with_expansion_off
    )

    # And not inert when switched on, or the assertions above would hold against a
    # version that ignored the expansion entirely.
    introduced_ids = {chunk.chunk_id for chunk in introduced}
    assert introduced_ids <= {chunk.chunk_id for chunk in with_expansion_on}
    assert introduced_ids.isdisjoint({chunk.chunk_id for chunk in with_expansion_off})


def test_a_graph_only_candidate_cannot_outrank_agreement() -> None:
    """Why the default weight is 0.5 rather than 1.0, as arithmetic.

    A chunk both retrievers ranked first scores 1/61 + 1/61. A chunk only the graph
    found, ranked first, scores 0.5/61. So expansion can lift a connected memory
    into a list it would never have entered and cannot displace something both
    retrievers agree about — which is the whole safety argument for letting a
    ranking introduce rows.
    """
    agreed = _chunk(0, 0.9)
    introduced = _chunk(20, 99.0)
    expansion = GraphCandidates(
        chunks=[introduced], routes={str(introduced.chunk_id): ("queue",)}
    )

    fused = fuse([agreed], [agreed], weights=FusionWeights(graph=0.5), graph=expansion)

    assert [chunk.chunk_id for chunk in fused] == [agreed.chunk_id, introduced.chunk_id]


# --------------------------------------------------------------------------
# B5: the breakdown carries the entity path
# --------------------------------------------------------------------------


def test_the_breakdown_carries_the_entity_path() -> None:
    """M2.5's guardrail, applied to the ranking that needs it most.

    A graph-introduced result may share no word with the query, so it is the one
    contribution a reader cannot reconstruct from the text. `graph_rank` without
    `graph_path` would be "the graph says so" with a number attached.
    """
    vector = [_chunk(0, 0.9)]
    introduced = _chunk(20, 5.0)
    expansion = GraphCandidates(
        chunks=[introduced],
        routes={str(introduced.chunk_id): ("job queue", "SKIP LOCKED", "worker")},
    )

    fused = fuse(vector, [], weights=FusionWeights(graph=0.5), graph=expansion)
    promoted = next(
        chunk for chunk in fused if chunk.chunk_id == introduced.chunk_id
    )

    assert promoted.breakdown is not None
    assert promoted.breakdown.graph_rank == 1
    assert promoted.breakdown.graph_score == 5.0
    assert promoted.breakdown.graph_path == ("job queue", "SKIP LOCKED", "worker")
    assert promoted.breakdown.as_dict()["graph_path"] == (
        "job queue -> SKIP LOCKED -> worker"
    )
    # A chunk the graph did not reach carries none of it, which is the distinction
    # that makes the field readable: null means "not reached", not "no route".
    untouched = next(chunk for chunk in fused if chunk.chunk_id == vector[0].chunk_id)
    assert untouched.breakdown is not None
    assert untouched.breakdown.graph_rank is None
    assert untouched.breakdown.graph_path is None


async def test_the_route_is_the_shortest_one_that_reached_the_chunk(
    corpus: Corpus, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Several routes reach one memory; the one shown is the one worth reading.

    A graph returns every path, and the last one the traversal happened to produce
    is not an explanation. The shortest is, tie-broken by the rarest endpoint.
    """
    await corpus.create(
        {
            "seed.md": ["alpha", "beta"],
            "target.md": ["alpha", "gamma"],
            "detour.md": ["beta", "gamma"],
            "filler-a.md": ["unrelated"],
            "filler-b.md": ["unrelated"],
            "filler-c.md": ["unrelated"],
            "filler-d.md": ["unrelated"],
            "filler-e.md": ["unrelated"],
        }
    )
    graph = await graph_for(sessions)
    candidates = await ExpandThroughGraph(sessions, graph, depth=2, hub_ratio=0.5)(
        [corpus.memories["seed.md"]]
    )

    target = next(
        chunk
        for chunk in candidates.chunks
        if chunk.memory_id == corpus.memories["target.md"]
    )
    route = candidates.routes[str(target.chunk_id)]

    # `target.md` shares `alpha` with the seed directly and `gamma` with `detour.md`
    # one hop further. The direct co-mention is the route worth showing, and it is
    # one name long: "another memory mentions this".
    assert route == ("alpha",), f"the shortest route, not a detour: {route}"


# --------------------------------------------------------------------------
# The arithmetic underneath, stated on its own
# --------------------------------------------------------------------------


def test_the_hub_threshold_never_collapses_on_a_tiny_corpus() -> None:
    """Floored at 2, or a three-memory corpus has no non-hub entities at all.

    Without the floor, `ceil(3 * 0.10) = 1` makes every entity a hub, expansion
    reaches nothing, and the graph looks useless when what was wrong was the
    arithmetic.
    """
    assert EntityStats(documents=3, hub_ratio=0.10).hub_threshold == 2
    assert EntityStats(documents=100, hub_ratio=0.10).hub_threshold == 10
    assert EntityStats(documents=0, hub_ratio=0.10).hub_threshold == 2


def test_idf_stays_positive_for_an_entity_in_every_memory() -> None:
    """`log(1 + N/df)`, not `log(N/df)`, so a universal entity is weak not free.

    At `log(N/df)` an entity in every memory scores exactly zero, which would
    silently discard the path rather than rank it last — and "discarded" and
    "ranked last" are different claims about a connection that does exist.
    """
    stats = EntityStats(documents=10, frequency={(entity := new_id()): 10})
    assert stats.idf(entity) > 0.0
    rarer = new_id()
    assert stats.idf(rarer) > stats.idf(entity)
