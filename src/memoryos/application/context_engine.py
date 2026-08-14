"""Given an event, assemble the context a person would want.

**Four sources, one fusion, and a budget that decides.** Every phase of this
project contributes a ranking here, and the milestone is not about any of them —
it is about what to do when they collectively return fifty relevant things and
eight fit.

    Phase 2   hybrid retrieval on the focus text
    Phase 3   the graph neighbourhood of what retrieval found
    Phase 4   recent activity, and other versions of the focused item
    Phase 5   decisions whose evidence cites what retrieval found

Fused with RRF, exactly as M2.2 fuses vector and keyword. **That is also how
deduplication happens**, and it is worth being explicit rather than treating it
as a side effect: a memory that arrives from retrieval *and* from the graph is
one item, and its arrival by two independent routes is evidence it belongs. RRF
gives it one entry whose score is the sum of both contributions, so it appears
once and ranks higher — the same "agreement outweighs enthusiasm" property that
made rank fusion the right choice in the first place, applied to four rankings
instead of two.

Selection is `domain/context.select`: MMR for diversity, a cap per category, and
a token budget counted with the real tokenizer, dropping whole items rather than
truncating any. None of that touches the database, which is why it is in
`domain/`.

**What this deliberately does not do is decide whether the context was any
good.** There is no relevance model here and no score to optimise. The
milestone's own acceptance test is a person reading the output for five real
focuses and saying whether it is what they wanted, and nothing in this module
substitutes for that.
"""

import asyncio
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum, auto
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func
from sqlalchemy import select as sql_select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.graph_expand import ExpandThroughGraph
from memoryos.application.ports import Embedder, TokenCounter
from memoryos.application.search import SearchMemories
from memoryos.domain.context import (
    DEFAULT_CATEGORY_SHARE,
    DEFAULT_LAMBDA,
    Candidate,
    ContextCategory,
    Rejection,
    category_of,
)
from memoryos.domain.context import select as select_items
from memoryos.domain.events import Event
from memoryos.domain.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from memoryos.domain.ids import new_id
from memoryos.domain.values import MemoryKind

logger = structlog.get_logger(__name__)

DEFAULT_TOKEN_BUDGET = 4000
DEFAULT_MAX_ITEMS = 12

# How many candidates each source may propose before fusion.
#
# Deliberately several times `max_items`. Fusion needs depth to have anything to
# agree about — a source that proposed exactly twelve would make "found by two
# routes" nearly impossible — and selection needs slack, because MMR and the
# category caps both reject candidates and a shallow pool would leave the budget
# unspent.
CANDIDATES_PER_SOURCE = 30

# How long a cached context stays valid, on top of the fingerprint check.
#
# The fingerprint already invalidates on any corpus change, so this is not about
# staleness of *content*. It is about staleness of *intent*: a context assembled
# for a meeting three days ago is answering a question nobody is asking now, and
# keeping it costs a row that will never be read.
DEFAULT_TTL = timedelta(hours=6)

# How much text each item contributes.
#
# One excerpt per item, not the whole memory. A 2,000-token file would eat half
# the budget on its own and the part that made it relevant is a paragraph of it.
EXCERPT_CHARS = 1200


class ContextSource(StrEnum):
    """Which phase proposed a candidate.

    Carried per item rather than summed, because `--explain` has to answer
    "which source proposed this" and the answer is frequently more than one —
    that being the point of fusing them.

    **`TEMPORAL` and `RECENCY` were one source until M6.3 measured what that
    cost.** Phase 4 contributes two different things — the focused item found by
    its name, and whatever changed lately — and M6.1 returned them as one
    concatenated ranking called `temporal`. They are not one signal. The first is
    *about the focus*; the second is about the calendar and would return the same
    list whatever you asked. Fusing them under one name made every file edited
    this week appear in every context at a high temporal rank, and M6.3's gate —
    which requires two independent routes to agree before it interrupts anybody —
    read "retrieval mentioned it and it is recent" as agreement. On a repository
    somebody is working in daily that is nearly everything, which is why the
    scores of fourteen consecutive focuses landed within ±10% of each other.

    Splitting them changes no item in any context. It changes what the fused
    score *means*, and it lets a caller ask about the routes that are actually
    about the focus. See `FOCUS_SPECIFIC` below.
    """

    RETRIEVAL = auto()
    GRAPH = auto()
    TEMPORAL = auto()
    RECENCY = auto()
    DECISIONS = auto()


# The sources whose ranking is a claim about *this focus*.
#
# `RECENCY` is deliberately outside it. "What changed lately" is a real and
# useful contribution to a context — a person looking at a file usually is in
# the middle of the week that produced it — but it is not evidence that this
# item is related to what they are looking at, because it would have said the
# same thing about any other focus at the same moment.
#
# Declared here, beside the enum, rather than in the one module that reads it.
# The next source added has to be placed on one side of the line, and a set
# literal in a caller is a decision nobody sees when they add a sixth.
FOCUS_SPECIFIC: frozenset[ContextSource] = frozenset(
    {
        ContextSource.RETRIEVAL,
        ContextSource.GRAPH,
        ContextSource.TEMPORAL,
        ContextSource.DECISIONS,
    }
)


@dataclass(frozen=True, slots=True)
class ContextRequest:
    """What to assemble, and how much of it fits.

    `trigger` is the event that caused this, or None when a person asked
    directly. Kept on the request rather than passed separately so that the
    cache key, the log line and the explain output all describe the same thing.
    """

    focus: str
    trigger: Event | None = None
    token_budget: int = DEFAULT_TOKEN_BUDGET
    max_items: int = DEFAULT_MAX_ITEMS
    lambda_: float = DEFAULT_LAMBDA
    category_share: float = DEFAULT_CATEGORY_SHARE


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One item in the assembled context, with everything `--explain` needs."""

    key: str
    title: str
    category: ContextCategory
    text: str
    tokens: int
    position: int
    # Rank in each ranking that proposed it, 1-based. Two entries means two
    # independent routes found it, which is why it is where it is.
    sources: dict[ContextSource, int]
    relevance: float
    redundancy: float
    # Set for memories, None for decisions.
    memory_id: UUID | None = None
    decision_id: UUID | None = None
    external_key: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "category": self.category.value,
            "text": self.text,
            "tokens": self.tokens,
            "position": self.position,
            "sources": {source.value: rank for source, rank in self.sources.items()},
            "relevance": self.relevance,
            "redundancy": self.redundancy,
            "memory_id": None if self.memory_id is None else str(self.memory_id),
            "decision_id": None if self.decision_id is None else str(self.decision_id),
            "external_key": self.external_key,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ContextItem":
        return cls(
            key=str(raw["key"]),
            title=str(raw["title"]),
            category=ContextCategory(raw["category"]),
            text=str(raw["text"]),
            tokens=int(raw["tokens"]),
            position=int(raw["position"]),
            sources={
                ContextSource(name): int(rank)
                for name, rank in dict(raw["sources"]).items()
            },
            relevance=float(raw["relevance"]),
            redundancy=float(raw["redundancy"]),
            memory_id=None if raw["memory_id"] is None else UUID(str(raw["memory_id"])),
            decision_id=(
                None if raw["decision_id"] is None else UUID(str(raw["decision_id"]))
            ),
            external_key=raw["external_key"],
        )


@dataclass(frozen=True, slots=True)
class SourceReport:
    source: ContextSource
    proposed: int
    duration_ms: int


@dataclass(slots=True)
class AssembledContext:
    """The context, plus how it was arrived at."""

    focus: str
    items: list[ContextItem] = field(default_factory=list)
    token_budget: int = DEFAULT_TOKEN_BUDGET
    tokens_used: int = 0
    # Every candidate that did not make it, with the reason. On a tight budget
    # what nearly fit is the useful diagnostic, and "eight items" alone does not
    # distinguish a rich corpus from an empty one.
    rejected: list[tuple[str, Rejection]] = field(default_factory=list)
    sources: list[SourceReport] = field(default_factory=list)
    candidates: int = 0
    duration_ms: int = 0
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "focus": self.focus,
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "items": [item.as_dict() for item in self.items],
        }


# --------------------------------------------------------------------------
# The corpus fingerprint
# --------------------------------------------------------------------------


async def corpus_fingerprint(sessions: async_sessionmaker[AsyncSession]) -> str:
    """A short digest that changes when anything context draws from changes.

    Counts and latest timestamps over the two tables that feed assembly, hashed.
    Cheap — four indexed aggregates — and it has the property that matters: a
    sync that ingests one file changes the digest, so every cached context built
    before it is invalid without anything having to know which focuses were
    affected.

    **Deliberately coarse.** A per-focus dependency set would invalidate less and
    would have to be maintained by every writer in the system; getting that wrong
    means serving context that silently omits the file somebody just changed,
    which is the failure this cache exists to avoid rather than to cause. Over-
    invalidation costs a re-assembly measured in hundreds of milliseconds.

    `updated_at` on decisions rather than `created_at`: an amended decision is a
    changed input, and M5.0's `edit` moves that column.
    """
    async with sessions() as session:
        memories = (
            await session.execute(
                sql_select(
                    func.count(),
                    func.max(models.Memory.ingested_at),
                ).where(
                    models.Memory.is_current.is_(True),
                    models.Memory.deleted_at.is_(None),
                )
            )
        ).one()
        decisions = (
            await session.execute(
                sql_select(func.count(), func.max(models.Decision.updated_at))
            )
        ).one()

    raw = f"{memories[0]}:{memories[1]}:{decisions[0]}:{decisions[1]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# What shape a cached payload is in.
#
# **The fingerprint covers the corpus and this covers the code.** A cached
# context is invalidated when the material it was built from changes; nothing
# invalidated it when the *meaning of its contents* changed, and M6.3's split of
# `temporal` into two sources is exactly that — a payload written before it
# labels a recency hit `temporal`, and the gate that reads those labels would
# count it as evidence about the focus. Rows written by two different builds
# cannot be told apart otherwise, because a hash of the focus and the corpus is
# identical across a deploy.
#
# Bumped whenever the payload's fields or their meaning change. Cheap: every key
# changes, and the cost is one re-assembly per focus.
CONTEXT_SCHEMA_VERSION = 2


def cache_key_for(request: ContextRequest, fingerprint: str) -> str:
    """The identity of one assembled context.

    The budget and the item cap are part of it, because a 4,000-token context is
    not a truncation of a 1,000-token one — different items were selected, not
    fewer. Keying on focus alone would serve the wrong shape to whichever caller
    asked second.
    """
    raw = (
        f"v{CONTEXT_SCHEMA_VERSION}\x00{request.focus}\x00{request.token_budget}"
        f"\x00{request.max_items}\x00{request.lambda_}"
        f"\x00{request.category_share}\x00{fingerprint}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


async def read_cached(
    sessions: async_sessionmaker[AsyncSession], key: str
) -> AssembledContext | None:
    """A cached context, or None. **Constructs nothing that loads a model.**

    A module-level function rather than a method, and that is the whole point of
    it: `AssembleContext` holds a search use case, an embedder and a
    cross-encoder, so building one to read a row would load two models into the
    API process — which is exactly the cost the cache exists to avoid, paid on
    the path that exists to avoid it.

    The hit is counted in the same transaction as the read. A hit rate assembled
    from log lines is a hit rate nobody can query afterwards, and it is the
    number that decides whether any of this precomputation earns its cost.
    """
    now = datetime.now(UTC)
    async with sessions.begin() as session:
        row = (
            await session.execute(
                sql_select(models.ContextCache).where(
                    models.ContextCache.cache_key == key,
                    models.ContextCache.expires_at > now,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        row.hit_count += 1
        payload = dict(row.payload)

    return AssembledContext(
        focus=str(payload["focus"]),
        items=[ContextItem.from_dict(item) for item in payload["items"]],
        token_budget=int(payload["token_budget"]),
        tokens_used=int(payload["tokens_used"]),
        cached=True,
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


class AssembleContext:
    """The whole assembly path, as one use case."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        search: SearchMemories,
        counter: TokenCounter,
        embedder: Embedder,
        *,
        expand: ExpandThroughGraph | None = None,
        candidates_per_source: int = CANDIDATES_PER_SOURCE,
        ttl: timedelta = DEFAULT_TTL,
    ) -> None:
        self._sessions = sessions
        self._search = search
        # The same object in every real wiring, and two parameters anyway. The
        # ports are separate because M1.5 made them separate on purpose — the
        # chunker sizes text and has no business producing vectors — and
        # collapsing them here to save an argument would be this module deciding
        # that distinction does not matter.
        self._counter = counter
        self._embedder = embedder
        # None means the graph is not wired at all, which is different from a
        # graph that returns nothing: one is a deployment without Neo4j, the
        # other is a corpus whose entities have not been extracted. Both produce
        # a context, and the distinction is visible in the source report.
        self._expand = expand
        self._per_source = candidates_per_source
        self._ttl = ttl

    async def __call__(
        self, request: ContextRequest, *, use_cache: bool = True
    ) -> AssembledContext:
        started = time.monotonic()
        fingerprint = await corpus_fingerprint(self._sessions)
        key = cache_key_for(request, fingerprint)

        if use_cache:
            cached = await self._read_cache(key)
            if cached is not None:
                cached.duration_ms = _ms(started)
                logger.info(
                    "context.cache_hit", focus=request.focus, duration_ms=cached.duration_ms
                )
                return cached

        assembled = await self._assemble(request, started)
        if use_cache:
            await self._write_cache(key, request, assembled)
        return assembled

    async def _assemble(
        self, request: ContextRequest, started: float
    ) -> AssembledContext:
        rankings: dict[ContextSource, list[str]] = {}
        reports: list[SourceReport] = []
        memory_ids: dict[UUID, None] = {}
        decision_ids: dict[UUID, None] = {}

        # (1) Phase 2. Everything else seeds from this, so it runs first and
        # alone: the graph expands from what retrieval found, and the decision
        # source looks for decisions citing it.
        retrieval_started = time.monotonic()
        result = await self._search(request.focus, k=self._per_source)
        retrieved = [hit.memory_id for hit in result.hits]
        # The chunk that put each memory where it is, kept for rendering. A
        # context item showing the *head* of a file shows its import block, and
        # the paragraph that made it relevant is three screens down — which is
        # the same defect as truncating an item, arrived at from the other end.
        excerpts: dict[UUID, str] = {
            hit.memory_id: _best_chunk_text(hit.matched_chunks)
            for hit in result.hits
            if hit.matched_chunks
        }
        rankings[ContextSource.RETRIEVAL] = [f"memory:{item}" for item in retrieved]
        for memory_id in retrieved:
            memory_ids.setdefault(memory_id, None)
        reports.append(
            SourceReport(
                ContextSource.RETRIEVAL, len(retrieved), _ms(retrieval_started)
            )
        )

        # (2) Phase 3, seeded from those hits. `ExpandThroughGraph` never raises
        # — an unreachable graph is an empty ranking, which fusion handles as
        # "this retriever found nothing" rather than as a failure.
        graph_started = time.monotonic()
        graph_ids: list[UUID] = []
        if self._expand is not None and retrieved:
            candidates = await self._expand(retrieved)
            seen: set[UUID] = set()
            for chunk in candidates.chunks:
                if chunk.memory_id not in seen:
                    seen.add(chunk.memory_id)
                    graph_ids.append(chunk.memory_id)
                    # Only when retrieval did not already supply one. Retrieval
                    # scored its chunk against the focus; the graph scored this
                    # one against a path through entities, and the first is the
                    # better excerpt when both exist.
                    excerpts.setdefault(
                        chunk.memory_id, chunk.text[chunk.prefix_chars :].strip()
                    )
            rankings[ContextSource.GRAPH] = [f"memory:{item}" for item in graph_ids]
            for memory_id in graph_ids:
                memory_ids.setdefault(memory_id, None)
        reports.append(
            SourceReport(ContextSource.GRAPH, len(graph_ids), _ms(graph_started))
        )

        # (3) Phase 4, as two rankings rather than one. The focused item by name
        # is about the focus; what changed lately is not, and fusing them under
        # one name let recency manufacture the agreement M6.3's gate depends on.
        # See `ContextSource`.
        temporal_started = time.monotonic()
        named_ids = await self._by_name(request.focus, limit=self._per_source)
        rankings[ContextSource.TEMPORAL] = [f"memory:{item}" for item in named_ids]
        for memory_id in named_ids:
            memory_ids.setdefault(memory_id, None)
        reports.append(
            SourceReport(ContextSource.TEMPORAL, len(named_ids), _ms(temporal_started))
        )

        recency_started = time.monotonic()
        recent_ids = await self._recent(limit=self._per_source)
        rankings[ContextSource.RECENCY] = [f"memory:{item}" for item in recent_ids]
        for memory_id in recent_ids:
            memory_ids.setdefault(memory_id, None)
        reports.append(
            SourceReport(ContextSource.RECENCY, len(recent_ids), _ms(recency_started))
        )

        # (4) Phase 5: decisions whose evidence cites anything the other sources
        # found. Not a text search over decisions — the link is the evidence
        # table, which is a fact somebody recorded rather than a similarity.
        decisions_started = time.monotonic()
        # Ranked by *which* memories they cite rather than how many. See
        # `_decisions`: the memories are passed with their position, and a
        # decision citing the top hit outranks one citing five also-rans.
        decided = await self._decisions(
            {memory_id: rank for rank, memory_id in enumerate(memory_ids, 1)},
            limit=self._per_source,
        )
        rankings[ContextSource.DECISIONS] = [f"decision:{item}" for item in decided]
        for decision_id in decided:
            decision_ids.setdefault(decision_id, None)
        reports.append(
            SourceReport(
                ContextSource.DECISIONS, len(decided), _ms(decisions_started)
            )
        )

        # Fusion, which is also the deduplication. A memory found by retrieval
        # and by the graph is one key appearing in two rankings: one entry, with
        # both contributions summed.
        ordered_sources = list(rankings)
        fused = reciprocal_rank_fusion(
            [rankings[source] for source in ordered_sources], k=DEFAULT_RRF_K
        )
        ranks = {
            source: {key: position for position, key in enumerate(rankings[source], 1)}
            for source in ordered_sources
        }

        texts = await self._render(list(memory_ids), list(decision_ids), excerpts)
        vectors = await self._vectors(list(memory_ids))
        vectors.update(await self._embed_unvectored(texts, vectors))

        candidates_for_selection: list[Candidate] = []
        for key, score in fused:
            rendered = texts.get(key)
            if rendered is None:
                continue
            candidates_for_selection.append(
                Candidate(
                    key=key,
                    category=rendered.category,
                    tokens=self._counter.count_tokens(rendered.text),
                    relevance=score,
                    vector=vectors.get(key),
                )
            )

        selection = select_items(
            candidates_for_selection,
            token_budget=request.token_budget,
            max_items=request.max_items,
            lambda_=request.lambda_,
            category_share=request.category_share,
        )

        items: list[ContextItem] = []
        for chosen in selection.chosen:
            rendered = texts[chosen.key]
            items.append(
                ContextItem(
                    key=chosen.key,
                    title=rendered.title,
                    category=rendered.category,
                    text=rendered.text,
                    tokens=chosen.tokens,
                    position=chosen.position,
                    sources={
                        source: ranks[source][chosen.key]
                        for source in ordered_sources
                        if chosen.key in ranks[source]
                    },
                    relevance=chosen.relevance,
                    redundancy=chosen.redundancy,
                    memory_id=rendered.memory_id,
                    decision_id=rendered.decision_id,
                    external_key=rendered.external_key,
                )
            )

        assembled = AssembledContext(
            focus=request.focus,
            items=items,
            token_budget=request.token_budget,
            tokens_used=selection.tokens_used,
            rejected=selection.rejected,
            sources=reports,
            candidates=len(candidates_for_selection),
            duration_ms=_ms(started),
        )
        logger.info(
            "context.assembled",
            focus=request.focus,
            # Which event caused this, or None when a person asked directly.
            # Recorded rather than carried unread: M6.2's whole question is
            # whether context built for a trigger is ever looked at, and that is
            # unanswerable if the assembly does not say what triggered it.
            trigger_kind=(
                None if request.trigger is None else request.trigger.kind.value
            ),
            trigger_id=(
                None if request.trigger is None else str(request.trigger.id)
            ),
            candidates=assembled.candidates,
            items=len(items),
            tokens_used=assembled.tokens_used,
            token_budget=request.token_budget,
            duration_ms=assembled.duration_ms,
        )
        return assembled

    # ----------------------------------------------------------------------
    # Sources
    # ----------------------------------------------------------------------

    async def _by_name(self, focus: str, *, limit: int) -> list[UUID]:
        """The focused item itself, found by its key rather than its content.

        Half of what M6.1 called the temporal source, and now its own ranking.
        The other half is `_recent`, and the reason they are separate is in
        `ContextSource`: one is about the focus and one is about the clock.

        Fires only when the focus looks like something in the corpus — an
        `external_key` suffix match. A focus of "chunking strategy" names no
        file, so this contributes nothing, which is the honest outcome rather
        than a miss.

        **Current versions only, and superseded ones are excluded rather than
        ranked below.** The milestone asks for "versions of the focused item" and
        offering them is actively worse than not: M1.4 deletes the chunks of
        every superseded version, so an old version has *no embedding* — which
        means MMR cannot measure its redundancy against anything, scores it a
        flat zero penalty, and systematically promotes it over the live text it
        is a stale copy of. That is exactly what this corpus produced: the
        current `search.py` was crowded out of its own context by its previous
        version. Version history is M4.2's view, where it can be read as history
        instead of being mistaken for the file.

        **This is also the only source that can find the focused item by name**,
        and on a path focus it is frequently the one that does. Retrieval embeds
        the focus as text, and `src/memoryos/application/search.py` as a *query*
        does not sit near that file's own prose — so the file a person is
        looking at reaches the context through its key rather than through its
        content. Worth knowing when reading the source report: a low retrieval
        rank for the focused file is not a retrieval failure, it is a path being
        a poor embedding.
        """
        if not focus:
            return []
        async with self._sessions() as session:
            return list(
                (
                    await session.execute(
                        sql_select(models.Memory.id)
                        .where(
                            models.Memory.external_key.like(
                                f"%{_like_literal(focus)}", escape="\\"
                            ),
                            models.Memory.is_current.is_(True),
                            models.Memory.deleted_at.is_(None),
                        )
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )

    async def _recent(self, *, limit: int) -> list[UUID]:
        """What changed lately, which is not a claim about the focus at all.

        **It takes no focus argument, and that is the whole point of the split.**
        This ranking is identical for every question asked at the same moment. It
        earns its place in a context — somebody looking at a file is usually in
        the middle of the week that produced it, and M6.1 measured it finding
        things retrieval missed — but it is not evidence that any of it is
        *related* to the focus, and for two milestones it was fused under a name
        that implied it was.

        Superseded versions stay excluded here for the reason `_by_name` gives:
        M1.4 deletes their chunks, so MMR cannot measure an old version's
        redundancy and systematically promotes it over the live text.
        """
        async with self._sessions() as session:
            return list(
                (
                    await session.execute(
                        sql_select(models.Memory.id)
                        .where(
                            models.Memory.is_current.is_(True),
                            models.Memory.deleted_at.is_(None),
                        )
                        .order_by(models.Memory.occurred_at.desc().nullslast())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )

    async def _decisions(
        self, ranked: dict[UUID, int], *, limit: int
    ) -> list[UUID]:
        """Decisions whose recorded evidence cites the memories in front of you.

        **Ranked by which memories a decision cites, not how many.** Counting
        citations sounds right and is not: it selects for decisions that cite
        *popular* memories rather than relevant ones. On this corpus the README,
        `cli.py` and `models.py` are cited by nearly every decision, so a plain
        count returned the same four decisions — the embedding model, the vector
        store, the LLM provider, the dependency rule — for every focus tried,
        whatever the focus was. Five different questions, one answer.

        So each cited memory contributes `1 / (k + its rank)`, the same
        reciprocal-rank shape M2.2 fuses with and for the same reason: a
        decision citing the top hit should outrank one citing five also-rans,
        and the flattening stops a single high rank from dominating outright.

        Matched on `(source_name, external_key)` rather than on `memory_id`, and
        that is not a stylistic preference. A memory id names one *version* of a
        file: re-sync it and every id changes, so evidence linked by id points at
        a superseded row and the decision silently stops being found. This
        corpus demonstrated that too — a re-ingest made the RRF decision vanish
        from the context of the file it is about. `decision_evidence` carries the
        natural key beside the id for exactly this, the same fix M1.7 wrote down
        for `query_judgements`, and this is the first thing outside a replay to
        read those columns.
        """
        if not ranked:
            return []
        async with self._sessions() as session:
            natural = (
                sql_select(
                    models.Source.name.label("source_name"),
                    models.Memory.external_key,
                    models.Memory.id.label("memory_id"),
                )
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Memory.id.in_(list(ranked)))
                .subquery()
            )
            rows = await session.execute(
                sql_select(
                    models.DecisionEvidence.decision_id,
                    natural.c.memory_id,
                )
                .join(
                    natural,
                    (models.DecisionEvidence.source_name == natural.c.source_name)
                    & (models.DecisionEvidence.external_key == natural.c.external_key),
                )
            )
            scored: dict[UUID, float] = {}
            for decision_id, memory_id in rows:
                rank = ranked.get(memory_id)
                if rank is None:
                    continue
                scored[decision_id] = scored.get(decision_id, 0.0) + 1.0 / (
                    DEFAULT_RRF_K + rank
                )

        # Ties break on the id, as `reciprocal_rank_fusion` does, so two
        # decisions citing the same evidence come back in the same order twice.
        ordered = sorted(scored.items(), key=lambda pair: (-pair[1], str(pair[0])))
        return [decision_id for decision_id, _ in ordered[:limit]]


    # ----------------------------------------------------------------------
    # Text and vectors
    # ----------------------------------------------------------------------

    async def _render(
        self,
        memory_ids: Sequence[UUID],
        decision_ids: Sequence[UUID],
        excerpts: dict[UUID, str],
    ) -> dict[str, "_Rendered"]:
        """The text each candidate contributes.

        **The matched chunk where there is one, the head of the memory where
        there is not**, and the difference is most of this milestone's output
        quality. A code file rendered from its head is a screen of imports; the
        function that made it relevant is further down, and a reader shown the
        imports concludes the retrieval was wrong. Only the temporal source
        proposes memories with no matched chunk — it ranks by recency, so there
        is no passage that matched — and for those the head is the honest
        default rather than a guess at which paragraph mattered.
        """
        rendered: dict[str, _Rendered] = {}
        async with self._sessions() as session:
            if memory_ids:
                rows = await session.execute(
                    sql_select(
                        models.Memory.id,
                        models.Memory.external_key,
                        models.Memory.kind,
                        models.Memory.content,
                        models.Source.name,
                    )
                    .join(models.Source, models.Source.id == models.Memory.source_id)
                    .where(models.Memory.id.in_(list(memory_ids)))
                )
                for memory_id, external_key, kind, content, source in rows:
                    excerpt = excerpts.get(memory_id)
                    text = (excerpt or (content or "").strip())[:EXCERPT_CHARS]
                    rendered[f"memory:{memory_id}"] = _Rendered(
                        title=f"{source}::{external_key}",
                        category=category_of(MemoryKind(kind)),
                        text=text,
                        memory_id=memory_id,
                        external_key=external_key,
                    )

            if decision_ids:
                rows = await session.execute(
                    sql_select(models.Decision).where(
                        models.Decision.id.in_(list(decision_ids))
                    )
                )
                for decision in rows.scalars():
                    body = f"Chose: {decision.chosen}"
                    if decision.reasoning:
                        body += f"\nBecause: {decision.reasoning}"
                    if decision.confidence is not None:
                        # The horizon and not just the number. A confidence
                        # reconstructed after the fact means something different
                        # from one recorded at the time, and a reader looking at
                        # this in an editor cannot see the column.
                        body += (
                            f"\nConfidence at the time: {decision.confidence:.2f}"
                            f" ({decision.confidence_horizon})"
                        )
                    rendered[f"decision:{decision.id}"] = _Rendered(
                        title=f"decision: {decision.question}",
                        category=ContextCategory.DECISION,
                        text=body,
                        decision_id=decision.id,
                    )
        return rendered

    async def _vectors(
        self, memory_ids: Sequence[UUID]
    ) -> dict[str, tuple[float, ...]]:
        """One vector per memory, for MMR.

        The first chunk's, not an average of all of them. Averaging chunk vectors
        produces a point that is near none of them — the centroid of a file that
        discusses two unrelated things sits between the two, and MMR would judge
        redundancy against a passage that does not exist.
        """
        if not memory_ids:
            return {}
        async with self._sessions() as session:
            rows = await session.execute(
                sql_select(models.MemoryChunk.memory_id, models.MemoryChunk.embedding)
                .where(
                    models.MemoryChunk.memory_id.in_(list(memory_ids)),
                    models.MemoryChunk.ordinal == 0,
                    models.MemoryChunk.embedding.is_not(None),
                )
            )
            return {
                f"memory:{memory_id}": tuple(float(value) for value in embedding)
                for memory_id, embedding in rows
                if embedding is not None
            }

    async def _embed_unvectored(
        self, texts: dict[str, "_Rendered"], have: dict[str, tuple[float, ...]]
    ) -> dict[str, tuple[float, ...]]:
        """Vectors for the candidates that have none — in practice, decisions.

        **Without this, a decision is free novelty and takes the context.** A
        candidate with no vector gets a redundancy of 0.0, which at any lambda
        below 1 is the best possible diversity score — so decisions win round
        after round on the strength of being unmeasurable rather than on being
        different. This corpus made that concrete: five of twelve items were
        decisions, several about subjects the focus never touched.

        Treating the missing vector as a zero vector would be worse, not better:
        it is the same free novelty with an extra step. The fix is to stop the
        gap existing, and a decision has text — its question and what was chosen
        — that embeds like anything else.

        `embed_passage` rather than `embed_query`: these are things being ranked,
        not the thing being ranked *with*, and M1.5's port makes the role a
        required part of the call precisely so this is answered rather than
        defaulted.

        Pushed onto a thread because embedding is CPU-bound matrix work, and
        awaiting it on the event loop would stall every other coroutine in the
        process — including the health checks.
        """
        missing = [key for key in texts if key not in have]
        if not missing:
            return {}
        payload = [f"{texts[key].title}\n{texts[key].text}" for key in missing]
        vectors = await asyncio.to_thread(self._embedder.embed_passage, payload)
        return {
            key: tuple(float(value) for value in vector)
            for key, vector in zip(missing, vectors, strict=True)
        }

    # ----------------------------------------------------------------------
    # Cache
    # ----------------------------------------------------------------------

    async def _read_cache(self, key: str) -> AssembledContext | None:
        return await read_cached(self._sessions, key)

    async def _write_cache(
        self, key: str, request: ContextRequest, assembled: AssembledContext
    ) -> None:
        """Store it, replacing any row under the same key.

        `ON CONFLICT DO UPDATE` rather than a delete and an insert: two triggers
        for the same focus can race, and the loser of that race should refresh
        the row rather than fail. The hit count is deliberately *not* reset — it
        counts reads of this key, and a rebuild under the same key is the same
        question being asked again.
        """
        now = datetime.now(UTC)
        stmt = (
            insert(models.ContextCache)
            .values(
                id=new_id(),
                cache_key=key,
                focus=request.focus,
                payload=assembled.as_dict(),
                token_count=assembled.tokens_used,
                built_at=now,
                expires_at=now + self._ttl,
            )
            .on_conflict_do_update(
                index_elements=["cache_key"],
                set_={
                    "payload": assembled.as_dict(),
                    "token_count": assembled.tokens_used,
                    "built_at": now,
                    "expires_at": now + self._ttl,
                },
            )
        )
        async with self._sessions.begin() as session:
            await session.execute(stmt)


@dataclass(frozen=True, slots=True)
class _Rendered:
    title: str
    category: ContextCategory
    text: str
    memory_id: UUID | None = None
    decision_id: UUID | None = None
    external_key: str | None = None


# --------------------------------------------------------------------------
# Cache statistics
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CacheStats:
    entries: int
    live: int
    hits: int
    tokens: int

    @property
    def hit_rate(self) -> float | None:
        """Hits over reads, where a read is a hit or the build that replaced it.

        Every cached entry was built by exactly one miss, so `entries + hits` is
        the number of times somebody asked. None when nothing has been cached —
        a rate over zero requests reported as 0.0 would read as "the cache never
        works".
        """
        requests = self.entries + self.hits
        if requests == 0:
            return None
        return self.hits / requests


async def cache_stats(sessions: async_sessionmaker[AsyncSession]) -> CacheStats:
    now = datetime.now(UTC)
    async with sessions() as session:
        row = (
            await session.execute(
                sql_select(
                    func.count(),
                    func.count(models.ContextCache.id).filter(
                        models.ContextCache.expires_at > now
                    ),
                    func.coalesce(func.sum(models.ContextCache.hit_count), 0),
                    func.coalesce(func.sum(models.ContextCache.token_count), 0),
                ).select_from(models.ContextCache)
            )
        ).one()
    return CacheStats(
        entries=int(row[0]), live=int(row[1]), hits=int(row[2]), tokens=int(row[3])
    )


def _like_literal(value: str) -> str:
    """A focus, escaped so `LIKE` reads it as text rather than as a pattern.

    **A focus of `%` matched all 239 memories in this corpus**, and the by-name
    source silently became "the entire corpus, in arbitrary order, ranked as
    though each row were the file you are looking at". Not SQL injection — the
    value is parameterised — but the wrong query, which is the harder kind to
    notice because nothing errors.

    Live rather than theoretical from M6.2 onwards: the watcher and the editor
    extension send *file paths* as the focus, and a path may legitimately
    contain `%` or `_`. The underscore is the one that would have gone unnoticed
    longest: `a_b.py` quietly also matches `aXb.py`.

    The backslash is escaped first, or escaping the metacharacters would
    introduce ones this pass then re-escapes.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _best_chunk_text(chunks: Sequence[Any]) -> str:
    """The highest-scoring chunk, with its borrowed lead-in removed.

    The overlap prefix is a copy of the previous chunk's tail. Including it
    spends budget on text the reader may already have above, and M2.5 carries
    `prefix_chars` precisely so this arithmetic is not done by hand in two
    places and got wrong in one.
    """
    best = max(chunks, key=lambda chunk: chunk.score)
    return str(best.text[best.prefix_chars :]).strip()


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
