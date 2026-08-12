"""Graph expansion: candidates that are connected rather than similar.

## What this is for, and why it might not work

Vector retrieval finds text that means the same thing. Keyword retrieval finds
text that says the same thing. Both are similarity relations over one document at
a time, and both are blind to the same class of answer: the memory that shares no
vocabulary and no paraphrase with the query, and is relevant because it is *about
the same things* — the person in the discussion, the file the decision touched,
the note written the same week about the same project.

That is what a graph can offer and an index cannot. It can also offer noise, and
the mechanism is not subtle: an entity mentioned in fifty memories connects all
fifty, so expanding along it produces fifty weakly-related candidates and calls
them evidence. Two guards decide which of the two happens, and they are the whole
milestone:

**Hub suppression.** An entity that appears in a large fraction of the corpus
carries almost no information about any particular memory — the same argument IDF
makes about the word "the", applied to graph nodes. Hubs above
`hub_ratio` are excluded from the traversal entirely, and the rest are weighted by
inverse document frequency, so a shared mention of `SKIP LOCKED` counts for far
more than a shared mention of `postgres`.

**Depth.** Two entity hops by default. Depth 3 on a graph this connected reaches
most of the corpus, and a ranking that contains everything is not a ranking.

## Introducing candidates, not reordering them

This is the one ranking in the fusion that *adds* rows. M2.3's recency and
importance signals deliberately only reorder what the retrievers found — a
document can be promoted for being recent but cannot appear for it — because
ranking the whole corpus by date and fusing that in would surface last week's
untouched notes for every query.

Graph expansion has to introduce, because the memory that shares no vocabulary
with the query is by construction not in either retriever's list. That is the
point, and it is also the risk, and it is why the weight defaults to 0.5 rather
than 1.0: at 0.5 a graph-only candidate contributes 0.5/61 against a retriever's
1/61, so the graph can bring a strongly-connected memory into the top ten and
cannot manufacture an answer out of a weak one.

## The seeds are the retrievers' own answers

Expansion starts from the top N memories hybrid retrieval already found, not from
the query. There is no reliable way to name the entities a natural-language
question is about — that is entity linking, which is a research problem and a
model call — and guessing wrong would expand from the wrong neighbourhood. Using
retrieval's own top results means the graph is answering "what else is connected
to what you already found", which is a question it can answer exactly.
"""

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import GraphReach, GraphStore, ScoredChunk

logger = structlog.get_logger(__name__)

# How many of hybrid retrieval's memories are used as seeds.
#
# Ten rather than the whole candidate list, and the bound is a precision decision
# rather than a cost one. Expanding from a memory retrieval ranked fortieth means
# expanding from something retrieval was not confident about, and the graph cannot
# repair a bad seed — it can only find things connected to it.
DEFAULT_SEED_MEMORIES = 10

# Entity hops. See `GraphStore.reach`.
DEFAULT_DEPTH = 2

# The fraction of reachable memories an entity may appear in before it is treated
# as carrying no information.
#
# 0.10 rather than something larger, because the failure is asymmetric. Excluding
# an entity that was not really a hub costs one path; keeping one that is turns
# every expansion into a list of everything it touches. On this corpus the top
# entities — `sqlalchemy`, `postgres`, `alembic` — are exactly the ones that appear
# everywhere and say nothing about which memory answers a question.
DEFAULT_HUB_RATIO = 0.10

# Rows the traversal may return. Bounds the fan-out of a depth-2 undirected walk;
# shortest-route-first ordering is what makes the bound keep the best ones.
DEFAULT_REACH_LIMIT = 400


@dataclass(frozen=True, slots=True)
class EntityStats:
    """Document frequency per entity, and the hub threshold derived from it.

    `documents` is the number of memories the graph can reach *at all* — memories
    with at least one mention — and not the size of the corpus. That distinction
    matters whenever extraction is incomplete: an entity in 20 of 34 extracted
    memories is a hub of everything the graph knows, and measuring it against 162
    memories would call it a 12% entity and let it bridge every path. The
    denominator is what the traversal can actually see.
    """

    documents: int
    frequency: dict[UUID, int] = field(default_factory=dict)
    hub_ratio: float = DEFAULT_HUB_RATIO

    @property
    def hub_threshold(self) -> int:
        """The mention count at which an entity stops being informative.

        Rounded up and floored at 2, so that on a tiny corpus the threshold cannot
        collapse to "every entity mentioned twice is a hub" — which would leave
        expansion with nothing to walk and make a graph look useless when what was
        actually wrong was the arithmetic.
        """
        return max(2, math.ceil(self.documents * self.hub_ratio))

    @property
    def hubs(self) -> frozenset[UUID]:
        return frozenset(
            entity_id
            for entity_id, count in self.frequency.items()
            if count >= self.hub_threshold
        )

    def idf(self, entity_id: UUID) -> float:
        """Inverse document frequency of one entity, for weighting a route.

        `log(1 + N / df)`, so the value stays positive for an entity in every
        document rather than going to zero and silently discarding the path. An
        unseen entity is treated as maximally rare, which is correct: it was
        reached through the graph, so it exists, and the only way its frequency is
        missing is that nothing else mentions it.
        """
        frequency = self.frequency.get(entity_id, 1)
        return math.log(1 + self.documents / max(1, frequency))


@dataclass(frozen=True, slots=True)
class GraphCandidates:
    """The expansion's ranking, and the route that justifies each row.

    `chunks` are ranked best first and carry the graph score in `.score`, which is
    thrown away by fusion — RRF keeps only the ordering. It is kept here anyway
    because a ranking whose scores are invisible cannot be debugged.

    `routes` is keyed by chunk id and holds the entity path that reached it. This
    is the explainability guardrail applied to the graph: M2.5 made every ranking
    contribution arguable, and "the graph promoted this" is not an argument.
    """

    chunks: list[ScoredChunk] = field(default_factory=list)
    routes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Diagnostics, reported by `search --explain` and by the tuner.
    seeds: int = 0
    hubs_excluded: int = 0
    reached: int = 0
    duration_ms: int = 0

    @property
    def ranking(self) -> list[str]:
        return [str(chunk.chunk_id) for chunk in self.chunks]

    def as_dict(self) -> dict[str, int]:
        return {
            "seeds": self.seeds,
            "hubs_excluded": self.hubs_excluded,
            "reached": self.reached,
            "candidates": len(self.chunks),
            "duration_ms": self.duration_ms,
        }


class ExpandThroughGraph:
    """The five steps of M3.5's expansion, in order."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        graph: GraphStore,
        *,
        depth: int = DEFAULT_DEPTH,
        hub_ratio: float = DEFAULT_HUB_RATIO,
        seed_memories: int = DEFAULT_SEED_MEMORIES,
        reach_limit: int = DEFAULT_REACH_LIMIT,
    ) -> None:
        self._sessions = session_factory
        self._graph = graph
        self._depth = depth
        self._hub_ratio = hub_ratio
        self._seed_memories = seed_memories
        self._reach_limit = reach_limit

    async def __call__(
        self, seed_memory_ids: Sequence[UUID], *, exclude: Sequence[UUID] = ()
    ) -> GraphCandidates:
        """Expand from these memories. Never raises; an empty result is an answer.

        **Degrades rather than fails, and that is deliberate.** The graph is a
        projection: it can be unreachable, empty, or behind, and none of those is a
        reason for a search to fail. A caller that got an exception here would have
        to decide what to do about it, and the only sensible decision — return the
        hybrid ranking — is the one an empty expansion produces on its own.
        """
        started = time.monotonic()
        seeds = list(seed_memory_ids)[: self._seed_memories]
        if not seeds:
            return GraphCandidates()

        try:
            return await self._expand(seeds, exclude, started)
        except Exception as exc:
            # Logged at warning, because a search that silently stopped using the
            # graph is a metric moving for a reason nobody can find.
            logger.warning("graph_expand.failed", error=str(exc), seeds=len(seeds))
            return GraphCandidates(duration_ms=_elapsed_ms(started))

    async def _expand(
        self, seeds: list[UUID], exclude: Sequence[UUID], started: float
    ) -> GraphCandidates:
        stats = await self.entity_stats()
        hubs = stats.hubs

        # (2) The entities those memories mention, from the graph rather than from
        # Postgres: this is the store whose reachability is being measured, and
        # seeding from Postgres would report a graph that works while the
        # projection is empty.
        mentioned = await self._graph.mention_edges(memory_ids=seeds)
        seed_entities = sorted({entity_id for _, entity_id in mentioned} - hubs)
        if not seed_entities:
            # Logged rather than returned quietly, because this is the state that
            # makes expansion contribute nothing and it has two very different
            # causes: the seeds have no entities at all (extraction has not reached
            # them) or every entity they have is a hub (the threshold is too low
            # for this corpus). Without the counts, both look like "the graph did
            # nothing" and neither is diagnosable.
            logger.info(
                "graph_expand.no_seed_entities",
                seeds=len(seeds),
                seed_mentions=len(mentioned),
                hubs_excluded=len(hubs),
                hub_threshold=stats.hub_threshold,
                reachable_memories=stats.documents,
            )
            return GraphCandidates(
                seeds=len(seeds), hubs_excluded=len(hubs), duration_ms=_elapsed_ms(started)
            )

        # (3) and (4), in one traversal. See `GraphStore.reach`.
        reached = await self._graph.reach(
            seed_entities,
            depth=self._depth,
            exclude_entity_ids=sorted(hubs),
            limit=self._reach_limit,
        )

        # Memories retrieval already found are not candidates the graph introduced.
        # Dropped here rather than left to fusion, because leaving them in would
        # let the graph ranking *reorder* the retrievers' own results — a second,
        # differently-shaped mechanism doing what RRF is already doing.
        known = set(seeds) | set(exclude)
        useful = [reach for reach in reached if reach.memory_id not in known]

        scored = _score(useful, stats)
        chunks, routes = await self._as_chunks(scored)

        candidates = GraphCandidates(
            chunks=chunks,
            routes=routes,
            seeds=len(seeds),
            hubs_excluded=len(hubs),
            reached=len({reach.memory_id for reach in useful}),
            duration_ms=_elapsed_ms(started),
        )
        # Info rather than debug: this is the number that says whether M3.5 is doing
        # anything on a given corpus, and it is the first thing to look at when a
        # metric moves. `search.finished` reports the fused total; this reports how
        # much of it the graph put there.
        logger.info("graph_expand.finished", **candidates.as_dict())
        return candidates

    async def entity_stats(self) -> EntityStats:
        """How many memories mention each entity, and how many are reachable at all.

        One aggregate over `entity_mentions`, read from Postgres rather than counted
        in the graph. That is not a contradiction of "seed from the graph": document
        frequency is a property of the corpus, the graph is a projection of the
        corpus, and Postgres can answer it in one grouped query where the graph
        needs a scan of every `MENTIONS` edge.
        """
        stmt = (
            select(
                models.EntityMention.entity_id,
                func.count(func.distinct(models.EntityMention.memory_id)),
            )
            .join(models.Entity, models.Entity.id == models.EntityMention.entity_id)
            .join(models.Memory, models.Memory.id == models.EntityMention.memory_id)
            .where(
                models.Entity.merged_into_id.is_(None),
                models.Memory.is_current.is_(True),
                models.Memory.deleted_at.is_(None),
            )
            .group_by(models.EntityMention.entity_id)
        )
        async with self._sessions() as session:
            frequency = {row[0]: int(row[1]) for row in await session.execute(stmt)}
            documents = int(
                (
                    await session.execute(
                        select(func.count(func.distinct(models.EntityMention.memory_id)))
                        .join(
                            models.Memory,
                            models.Memory.id == models.EntityMention.memory_id,
                        )
                        .where(
                            models.Memory.is_current.is_(True),
                            models.Memory.deleted_at.is_(None),
                        )
                    )
                ).scalar_one()
            )
        return EntityStats(
            documents=documents, frequency=frequency, hub_ratio=self._hub_ratio
        )

    async def _as_chunks(
        self, scored: list[tuple[tuple[UUID, int], float, tuple[str, ...]]]
    ) -> tuple[list[ScoredChunk], dict[str, tuple[str, ...]]]:
        """Turn `(memory, ordinal)` pairs into the chunks fusion works on.

        The graph stores a chunk's *ordinal*, deliberately, so that the projection
        survives a rebuild that mints new chunk ids. Fusion works on chunk ids,
        because that is what the retrievers return. This is the one place the two
        identities meet, and it is a join rather than a lookup table because the
        answer changes every time the corpus is rechunked.

        A pair that resolves to nothing is dropped silently: it means the corpus was
        rechunked since the projection was written, which `graph verify` reports and
        `graph rebuild` fixes. Failing a search over it would be the wrong end of
        the system to complain.
        """
        if not scored:
            return [], {}

        wanted = [pair for pair, _, _ in scored]
        stmt = select(
            models.MemoryChunk.id,
            models.MemoryChunk.memory_id,
            models.MemoryChunk.ordinal,
            models.MemoryChunk.content,
            models.MemoryChunk.char_start,
            models.MemoryChunk.char_end,
            models.MemoryChunk.prefix_chars,
            models.MemoryChunk.meta,
        ).where(
            models.MemoryChunk.memory_id.in_({memory_id for memory_id, _ in wanted})
        )
        async with self._sessions() as session:
            rows = {
                (row[1], row[2]): row for row in await session.execute(stmt)
            }

        chunks: list[ScoredChunk] = []
        routes: dict[str, tuple[str, ...]] = {}
        for pair, score, route in scored:
            row = rows.get(pair)
            if row is None:
                continue
            chunks.append(
                ScoredChunk(
                    chunk_id=row[0],
                    memory_id=row[1],
                    ordinal=row[2],
                    text=row[3],
                    score=score,
                    char_start=row[4],
                    char_end=row[5],
                    prefix_chars=row[6],
                    metadata=dict(row[7] or {}),
                )
            )
            routes[str(row[0])] = route
        return chunks, routes


def _score(
    reached: Sequence[GraphReach], stats: EntityStats
) -> list[tuple[tuple[UUID, int], float, tuple[str, ...]]]:
    """Rank the reached chunks, and keep the route that scored highest for each.

    The score sums `idf(entity) / hops` over the distinct entities that reached a
    chunk, which encodes three claims, each of which could be wrong and each of
    which is at least stated:

    * **Rare entities are worth more.** Sharing `SKIP LOCKED` with the query's
      answer is evidence; sharing `postgres` is not, on a corpus about Postgres.
    * **Nearer is worth more.** A memory reached in one hop shares an entity with
      what retrieval found; one reached in four shares an entity with something
      that shares an entity.
    * **Several routes are worth more than one.** Two independent connections are
      better evidence than one, which is the same argument RRF makes about two
      retrievers agreeing.

    The route kept is the shortest one, tie-broken by the rarest endpoint, because
    that is the connection a reader would want shown — not the last one the
    traversal happened to return.
    """
    scores: dict[tuple[UUID, int], float] = {}
    best_route: dict[tuple[UUID, int], tuple[int, float, tuple[str, ...]]] = {}
    seen: set[tuple[UUID, int, UUID]] = set()

    for reach in reached:
        key = (reach.memory_id, reach.chunk_ordinal)
        # One contribution per entity per chunk. The traversal legitimately returns
        # several routes to the same pair — that is what a graph does — and counting
        # each of them would let a densely-connected neighbourhood outvote a rare
        # entity by sheer path count.
        marker = (reach.memory_id, reach.chunk_ordinal, reach.entity_id)
        if marker in seen:
            continue
        seen.add(marker)

        weight = stats.idf(reach.entity_id) / max(1, reach.hops)
        scores[key] = scores.get(key, 0.0) + weight

        candidate = (reach.hops, -stats.idf(reach.entity_id), reach.route)
        current = best_route.get(key)
        if current is None or candidate[:2] < current[:2]:
            best_route[key] = candidate

    return [
        (key, score, best_route[key][2])
        for key, score in sorted(
            scores.items(),
            # Descending by score, ties broken by memory id then ordinal, so a
            # rerun of one query cannot reorder the expansion.
            key=lambda item: (-item[1], str(item[0][0]), item[0][1]),
        )
    ]


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
