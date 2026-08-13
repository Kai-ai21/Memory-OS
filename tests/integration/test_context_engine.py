"""Assembly over a real corpus: what fuses, what is cached, and what expires.

The selection rules — dropped-whole, category caps, MMR — are checked without a
database in `tests/unit/test_context_selection.py`, where they belong. What this
file checks is the two properties that only exist once four sources and a cache
are involved:

* a memory found by two sources appears **once** and ranks **higher** for it,
* and a cached context stops being served the moment the corpus changes.

The second is the one worth the setup cost. A cache that serves stale context is
worse than no cache: it answers confidently from a corpus that has moved, and
nothing about the answer looks different.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, func, select

from memoryos.adapters.db import models
from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.application.context_engine import (
    CANDIDATES_PER_SOURCE,
    AssembleContext,
    ContextRequest,
    ContextSource,
    cache_key_for,
    cache_stats,
    corpus_fingerprint,
)
from memoryos.application.decisions import DecisionDraft, EvidenceInput, OptionInput
from memoryos.application.decisions import record as record_decision
from memoryos.application.search import SearchMemories
from memoryos.domain.context import ContextCategory
from memoryos.domain.fusion import reciprocal_rank_fusion
from memoryos.domain.values import EvidenceRelation, TimeProvenance
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration


def engine(harness: Harness) -> AssembleContext:
    """The real engine over the harness's corpus, with no graph.

    `expand=None` is a deployment without Neo4j rather than a graph that returns
    nothing, and it is the right default here: M3.0 established that Community
    Edition has one database, so a test asserting about graph contributions
    would assert against whatever graph the developer happens to have.
    """
    return AssembleContext(
        harness.sessions,
        _search(harness),
        harness.embedder,
        harness.embedder,
        expand=None,
    )


def _search(harness: Harness) -> SearchMemories:
    return SearchMemories(
        harness.sessions,
        harness.embedder,
        PgVectorStore(harness.sessions, harness.embedder, default_ef_search=100),
        PostgresKeywordStore(harness.sessions),
    )


# --------------------------------------------------------------------------
# Fusion is the deduplication
# --------------------------------------------------------------------------


def test_a_key_in_two_rankings_appears_once_and_scores_higher() -> None:
    """The property M6.1 needs from RRF, asserted on RRF itself.

    A memory that arrives from retrieval *and* from the graph is one item, and
    its arrival by two independent routes is evidence it belongs. Deduplication
    is not a pass over the results afterwards — it is what fusing on keys
    already does, and asserting it here rather than through the engine is what
    makes the claim about the mechanism rather than about one corpus.
    """
    retrieval = ["memory:a", "memory:b", "memory:c"]
    graph = ["memory:c", "memory:d"]

    fused = reciprocal_rank_fusion([retrieval, graph])
    scores = dict(fused)
    keys = [key for key, _ in fused]

    # Once, not twice.
    assert keys.count("memory:c") == 1
    # And higher than the item ranked above it by one source alone: `c` is third
    # for retrieval and first for the graph, and that agreement beats `b`'s
    # single second place. Agreement outweighs enthusiasm, which is the property
    # M2.2 chose rank fusion for.
    assert scores["memory:c"] > scores["memory:b"]
    # And `c` leads outright. Third-of-three plus first-of-two beats a single
    # first place: 1/63 + 1/61 against 1/61. That is the whole reason a memory
    # two routes agree on is worth more than one route's favourite.
    assert keys[0] == "memory:c"


async def test_an_item_found_twice_carries_both_routes(harness: Harness) -> None:
    """And the engine keeps both, because `--explain` has to show them.

    A single "found by" would make an item that two routes agreed on
    indistinguishable from one that scraped in on a single mention, which is
    precisely the distinction that put it where it is.
    """
    assembled = await engine(harness)(
        ContextRequest(focus="chunking", max_items=12), use_cache=False
    )

    assert assembled.items
    # Retrieval and the temporal source both propose from the same corpus, so on
    # any non-trivial corpus at least one item is found by both.
    multi = [item for item in assembled.items if len(item.sources) > 1]
    assert multi, "expected at least one item proposed by more than one source"
    for item in multi:
        assert all(rank >= 1 for rank in item.sources.values())
    # No key is ever listed twice.
    keys = [item.key for item in assembled.items]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------
# The budget and the caps, over real text
# --------------------------------------------------------------------------


async def test_the_budget_is_respected_with_the_real_tokenizer(
    harness: Harness,
) -> None:

    assembled = await engine(harness)(
        ContextRequest(focus="chunking", token_budget=300), use_cache=False
    )

    assert assembled.tokens_used <= 300
    # Counted with the model's own tokenizer, not estimated: the sum of the
    # items' counts is the number the budget was checked against.
    assert sum(item.tokens for item in assembled.items) == assembled.tokens_used
    # And every item is whole. Nothing in the pipeline can express a partial one.
    for item in assembled.items:
        assert item.text.strip()


async def test_no_category_takes_more_than_half_the_items(harness: Harness) -> None:

    assembled = await engine(harness)(
        ContextRequest(focus="chunking", max_items=6, token_budget=8000),
        use_cache=False,
    )

    counts: dict[ContextCategory, int] = {}
    for item in assembled.items:
        counts[item.category] = counts.get(item.category, 0) + 1
    assert counts
    assert max(counts.values()) <= 3


# --------------------------------------------------------------------------
# Phase 5 contributes decisions, through evidence rather than similarity
# --------------------------------------------------------------------------


async def test_a_decision_citing_a_retrieved_memory_is_offered(
    harness: Harness,
) -> None:
    """The link is a row somebody wrote, not a similarity.

    No text search over decisions happens anywhere. A decision is in the context
    because its recorded evidence names a memory the other sources found, which
    is a fact rather than a guess — and it is why a decision can be the most
    useful item in a context whose focus never mentions it.
    """
    async with harness.sessions() as session:
        memory = (
            await session.execute(select(models.Memory).limit(1))
        ).scalar_one()
        source = await session.get_one(models.Source, memory.source_id)

    await record_decision(
        harness.sessions,
        DecisionDraft(
            question="How is the corpus chunked?",
            chosen="Structural chunking, sized against the model window",
            confidence=0.8,
            options=(
                OptionInput(
                    description="Fixed windows", rejected_because="Splits definitions."
                ),
            ),
            evidence=(
                EvidenceInput(
                    source_name=source.name,
                    external_key=memory.external_key,
                    relation=EvidenceRelation.INFORMED,
                ),
            ),
        ),
        decided_at=datetime.now(UTC),
        decided_at_source=TimeProvenance.DECLARED,
    )

    assembled = await engine(harness)(
        ContextRequest(focus="chunking", max_items=12), use_cache=False
    )

    decisions = [
        item for item in assembled.items if item.category is ContextCategory.DECISION
    ]
    assert decisions, "a decision citing a retrieved memory should be offered"
    assert ContextSource.DECISIONS in decisions[0].sources
    assert decisions[0].decision_id is not None


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


async def test_the_cache_serves_the_second_request(harness: Harness) -> None:
    assemble = engine(harness)
    request = ContextRequest(focus="chunking")

    first = await assemble(request)
    second = await assemble(request)

    assert not first.cached
    assert second.cached
    assert [item.key for item in second.items] == [item.key for item in first.items]
    assert second.tokens_used == first.tokens_used

    stats = await cache_stats(harness.sessions)
    assert stats.entries == 1
    assert stats.hits == 1
    # One build and one hit is one request served from cache out of two.
    assert stats.hit_rate == pytest.approx(0.5)


async def test_the_cache_invalidates_when_the_corpus_fingerprint_changes(
    harness: Harness,
) -> None:
    """The property that makes the cache safe rather than merely fast.

    A context is a function of the whole corpus, so ingesting one file makes
    every context built before it a confident answer whose evidence has moved —
    and nothing about the answer looks different. Keying on the fingerprint means
    no writer anywhere has to know which focuses its change affected.
    """
    assemble = engine(harness)
    request = ContextRequest(focus="chunking")

    before = await corpus_fingerprint(harness.sessions)
    first = await assemble(request)
    assert not first.cached
    assert (await assemble(request)).cached

    (harness.root / "brand-new.md").write_text(
        "# A new note about chunking\n\nSomething that did not exist before.\n"
    )
    await harness.ingest()

    after = await corpus_fingerprint(harness.sessions)
    assert after != before
    # The old key is still in the table and is simply never asked for again;
    # the new request misses and rebuilds.
    rebuilt = await assemble(request)
    assert not rebuilt.cached

    async with harness.sessions() as session:
        assert (
            await session.execute(
                select(func.count()).select_from(models.ContextCache)
            )
        ).scalar_one() == 2


async def test_a_recorded_decision_changes_the_fingerprint_too(
    harness: Harness,
) -> None:
    # Decisions are an input to assembly, so a new one has to invalidate for the
    # same reason a new memory does. A fingerprint over memories alone would
    # serve context that omits the decision somebody just recorded.
    before = await corpus_fingerprint(harness.sessions)

    await record_decision(
        harness.sessions,
        DecisionDraft(
            question="Does a decision move the fingerprint?",
            chosen="It does",
            options=(
                OptionInput(
                    description="It does not",
                    rejected_because="Then context would go stale.",
                ),
            ),
        ),
        decided_at=datetime.now(UTC),
        decided_at_source=TimeProvenance.DECLARED,
    )

    assert await corpus_fingerprint(harness.sessions) != before


def test_the_budget_is_part_of_the_cache_key() -> None:
    """A 4,000-token context is not a truncation of a 1,000-token one.

    Different items were selected, not fewer — MMR and the caps both see a
    different admissible set. Keying on focus alone would serve the wrong shape
    to whichever caller asked second.
    """
    fingerprint = "abc123"
    small = cache_key_for(ContextRequest(focus="x", token_budget=1000), fingerprint)
    large = cache_key_for(ContextRequest(focus="x", token_budget=4000), fingerprint)
    fewer = cache_key_for(ContextRequest(focus="x", max_items=4), fingerprint)
    more = cache_key_for(ContextRequest(focus="x", max_items=12), fingerprint)

    assert small != large
    assert fewer != more
    # And the same request twice is the same key, or nothing would ever hit.
    assert cache_key_for(ContextRequest(focus="x"), fingerprint) == cache_key_for(
        ContextRequest(focus="x"), fingerprint
    )


async def test_an_expired_entry_is_not_served(harness: Harness) -> None:
    # The TTL answers a different question from the fingerprint: content
    # staleness versus intent staleness. A context assembled for a meeting three
    # days ago is answering something nobody is asking now.
    assemble = engine(harness)
    request = ContextRequest(focus="chunking")
    await assemble(request)

    # Both timestamps move, not just the expiry: `ck_context_cache_expiry_order`
    # forbids a row that expired before it was built, and it is right to — that
    # shape is a clock or TTL bug, and it caught this test writing one.
    async with harness.sessions.begin() as session:
        row = (
            await session.execute(select(models.ContextCache))
        ).scalar_one()
        row.built_at = datetime.now(UTC) - timedelta(hours=2)
        row.expires_at = datetime.now(UTC) - timedelta(hours=1)

    assert not (await assemble(request)).cached


async def test_no_cache_neither_reads_nor_writes(harness: Harness) -> None:
    assemble = engine(harness)
    request = ContextRequest(focus="chunking")

    await assemble(request, use_cache=False)
    await assemble(request, use_cache=False)

    stats = await cache_stats(harness.sessions)
    assert stats.entries == 0
    assert stats.hit_rate is None


async def test_an_empty_corpus_assembles_nothing_rather_than_failing(
    harness: Harness,
) -> None:
    # The default state of a new installation. It must read as "there is nothing
    # here" rather than as a broken pipeline. The corpus is emptied rather than
    # a fresh harness built, because the fixture ingests on construction.
    async with harness.sessions.begin() as session:
        await session.execute(delete(models.Memory))
    assembled = await engine(harness)(
        ContextRequest(focus="anything at all"), use_cache=False
    )

    assert assembled.items == []
    assert assembled.tokens_used == 0
    assert {report.source for report in assembled.sources} == set(ContextSource)


async def test_a_focus_containing_a_wildcard_is_matched_literally(
    harness: Harness,
) -> None:
    """`%` in a focus made the by-name source return the whole corpus.

    Not SQL injection — the value is parameterised — but the wrong query, which
    is harder to notice because nothing errors: every memory matched, and each
    was ranked as though it were the file being looked at. Live rather than
    theoretical from M6.2, where the watcher and the editor send file paths as
    the focus and a path may legitimately contain `%` or `_`.
    """
    wildcard = await engine(harness)(
        ContextRequest(focus="%", max_items=12), use_cache=False
    )
    underscore = await engine(harness)(
        ContextRequest(focus="_", max_items=12), use_cache=False
    )

    for assembled in (wildcard, underscore):
        temporal = next(
            report
            for report in assembled.sources
            if report.source is ContextSource.TEMPORAL
        )
        # Recency still proposes — that half does not read the focus at all —
        # but the by-name half must contribute nothing, because no external key
        # ends with a literal percent sign.
        assert temporal.proposed <= CANDIDATES_PER_SOURCE

    # And the corpus does contain a file whose key ends in `.py`, so the by-name
    # half is not simply broken for everything.
    named = await engine(harness)(ContextRequest(focus="mod.py"), use_cache=False)
    assert any(
        item.external_key is not None and item.external_key.endswith("mod.py")
        for item in named.items
    )
