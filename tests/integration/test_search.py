"""Search against a real database and a real HNSW index, with a fake model."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.embed import EmbedMemory
from memoryos.application.evaluation import measure_recall
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.ports import SearchFilters
from memoryos.application.search import SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import MemoryKind, SourceKind
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog and keeps running onward. "
    "Every good boy deserves fudge and a reasonable amount of rest. "
)


@dataclass(slots=True)
class Corpus:
    root: Path
    source: Source
    sessions: async_sessionmaker[AsyncSession]
    embedder: FakeEmbedder
    store: PgVectorStore
    search: SearchMemories
    sync: SyncSource
    normalize: NormalizeMemory
    embed: EmbedMemory

    async def ingest(self) -> None:
        await self.sync(self.source.id, full=True)
        await self._drain(JobType.NORMALIZE_MEMORY, self.normalize)
        await self._drain(JobType.EMBED_MEMORY, self.embed)

    async def _drain(self, job_type: JobType, handler: object) -> None:
        async with self.sessions() as session:
            targets = [
                UUID(row[0]["memory_id"])
                for row in await session.execute(
                    select(models.Job.payload).where(models.Job.job_type == job_type.value)
                )
            ]
        for memory_id in targets:
            await handler(memory_id)  # type: ignore[operator]
        async with self.sessions.begin() as session:
            await session.execute(
                delete(models.Job).where(models.Job.job_type == job_type.value)
            )

    async def chunk_text(self, external_key: str, ordinal: int = 0) -> str:
        async with self.sessions() as session:
            return (
                await session.execute(
                    select(models.MemoryChunk.content)
                    .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
                    .where(
                        models.Memory.external_key == external_key,
                        models.Memory.is_current.is_(True),
                        models.MemoryChunk.ordinal == ordinal,
                    )
                )
            ).scalar_one()

    async def embed_text(self, value: str) -> list[float]:
        (vector,) = self.embedder.embed([value])
        return vector


def build_corpus(
    root: Path, blobs_root: Path, sessions: async_sessionmaker[AsyncSession], source: Source
) -> Corpus:
    blobs = FilesystemBlobStore(blobs_root)
    embedder = FakeEmbedder()
    cache = PostgresEmbeddingCache(sessions)
    store = PgVectorStore(sessions, embedder, default_ef_search=100)
    return Corpus(
        root=root,
        source=source,
        sessions=sessions,
        embedder=embedder,
        store=store,
        search=SearchMemories(sessions, embedder, store),
        sync=SyncSource(sessions, FilesystemConnector(blobs), blobs),
        normalize=NormalizeMemory(sessions, blobs, build_parsers(), StructuralChunker(embedder)),
        embed=EmbedMemory(sessions, embedder, cache),
    )


@pytest.fixture
async def corpus(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> Corpus:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.md").write_text("# Alpha\n\n" + PARAGRAPH * 5 + "\n")
    (root / "beta.md").write_text("# Beta\n\nsomething entirely different here. " * 20)
    (root / "gamma.txt").write_text("a third document about other matters. " * 25)

    source = Source(
        id=new_id(), kind=SourceKind.FILESYSTEM, name="corpus", config={"root": str(root)}
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    built = build_corpus(root, tmp_path / "blobs", sessions, source)
    await built.ingest()
    return built


# --------------------------------------------------------------------------
# The assertion that catches a wrong operator, a sign error, or a dead embedder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("exact", [False, True])
async def test_a_chunks_own_text_retrieves_itself_first(
    corpus: Corpus, exact: bool
) -> None:
    """One assertion, three failure modes.

    A wrong distance operator, a sign error on `<#>`, or an embedder returning
    nonsense would each break this and would each otherwise produce results
    that look entirely plausible in shape.
    """
    query = await corpus.chunk_text("alpha.md", ordinal=0)
    vector = await corpus.embed_text(query)

    filters = SearchFilters()
    if exact:
        found = await corpus.store.search_exact(vector, k=5, filters=filters)
    else:
        found = await corpus.store.search(vector, k=5, filters=filters)

    assert found
    assert found[0].text == query
    # Unit vectors, so a chunk against itself is inner product 1.
    assert found[0].score == pytest.approx(1.0, abs=1e-4)


async def test_scores_are_similarities_not_distances(corpus: Corpus) -> None:
    query = await corpus.chunk_text("alpha.md")
    found = await corpus.store.search(
        await corpus.embed_text(query), k=5, filters=SearchFilters()
    )

    scores = [chunk.score for chunk in found]
    # Descending, and the best is the self-match.
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]


async def test_ann_and_exact_agree_on_the_top_result(corpus: Corpus) -> None:
    # The corpus is small enough that HNSW is effectively exhaustive, so any
    # disagreement here is a bug rather than an approximation.
    query = await corpus.chunk_text("beta.md")
    vector = await corpus.embed_text(query)

    approximate = await corpus.store.search(vector, k=5, filters=SearchFilters())
    exhaustive = await corpus.store.search_exact(vector, k=5, filters=SearchFilters())

    assert approximate[0].chunk_id == exhaustive[0].chunk_id


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


async def test_a_source_filter_excludes_everything_else(
    corpus: Corpus, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "delta.md").write_text("# Delta\n\n" + PARAGRAPH * 5 + "\n")
    other = Source(
        id=new_id(),
        kind=SourceKind.FILESYSTEM,
        name="other",
        config={"root": str(other_root)},
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(other)
    second = build_corpus(other_root, tmp_path / "blobs2", sessions, other)
    await second.ingest()

    vector = await corpus.embed_text(await corpus.chunk_text("alpha.md"))

    everything = await corpus.store.search(vector, k=20, filters=SearchFilters())
    scoped = await corpus.store.search(
        vector, k=20, filters=SearchFilters(source_ids=[corpus.source.id])
    )

    assert len(everything) > len(scoped)
    async with sessions() as session:
        scoped_sources = set(
            (
                await session.execute(
                    select(models.Memory.source_id).where(
                        models.Memory.id.in_([chunk.memory_id for chunk in scoped])
                    )
                )
            ).scalars()
        )
    assert scoped_sources == {corpus.source.id}


async def test_a_kind_filter_narrows_results(corpus: Corpus) -> None:
    vector = await corpus.embed_text(await corpus.chunk_text("alpha.md"))

    notes = await corpus.store.search(
        vector, k=20, filters=SearchFilters(kinds=[MemoryKind.NOTE])
    )
    code = await corpus.store.search(
        vector, k=20, filters=SearchFilters(kinds=[MemoryKind.CODE])
    )

    assert notes
    assert code == []


async def test_a_tombstoned_memorys_chunks_do_not_surface(corpus: Corpus) -> None:
    query = await corpus.chunk_text("alpha.md")
    vector = await corpus.embed_text(query)
    assert any(chunk.text == query for chunk in await corpus.store.search(
        vector, k=20, filters=SearchFilters()
    ))

    (corpus.root / "alpha.md").unlink()
    await corpus.sync(corpus.source.id, full=True)

    found = await corpus.store.search(vector, k=20, filters=SearchFilters())
    assert not any(chunk.text == query for chunk in found)

    # But they are still there if you ask for them.
    including = await corpus.store.search(
        vector, k=20, filters=SearchFilters(include_deleted=True)
    )
    assert any(chunk.text == query for chunk in including)


async def test_a_superseded_versions_chunks_do_not_surface(corpus: Corpus) -> None:
    """A stale version describes text the item no longer says.

    Surfacing it would be a correctness bug rather than a ranking one, which is
    why `is_current` is unconditional in the predicates.
    """
    original = await corpus.chunk_text("beta.md")
    vector = await corpus.embed_text(original)

    (corpus.root / "beta.md").write_text("# Beta\n\ncompletely rewritten content. " * 25)
    await corpus.ingest()

    async with corpus.sessions() as session:
        versions = list(
            (
                await session.execute(
                    select(models.Memory.version).where(
                        models.Memory.external_key == "beta.md"
                    )
                )
            ).scalars()
        )
    assert sorted(versions) == [1, 2]

    found = await corpus.store.search(vector, k=20, filters=SearchFilters())
    assert not any(chunk.text == original for chunk in found)


async def test_requesting_more_than_exists_returns_what_there_is(
    corpus: Corpus,
) -> None:
    # Not an error: k is a ceiling, not a promise.
    found = await corpus.search(
        "anything at all", k=50, filters=SearchFilters(source_ids=[corpus.source.id])
    )
    assert 0 < len(found.hits) <= 3


async def test_chunks_without_embeddings_are_skipped(corpus: Corpus) -> None:
    async with corpus.sessions.begin() as session:
        await session.execute(
            update(models.MemoryChunk)
            .where(models.MemoryChunk.ordinal == 0)
            .values(embedding=None, embedding_model=None, embedded_at=None)
        )

    # Excluded rather than blowing up mid-scan.
    found = await corpus.store.search(
        await corpus.embed_text("anything"), k=20, filters=SearchFilters()
    )
    assert all(chunk.ordinal != 0 for chunk in found)


# --------------------------------------------------------------------------
# ef_search
# --------------------------------------------------------------------------


async def test_ef_search_does_not_leak_to_the_next_query(corpus: Corpus) -> None:
    """`SET LOCAL`, not `SET`.

    A session-level SET would ride the pooled connection into every later query
    on it, silently changing the recall/latency trade for unrelated callers.
    """
    vector = await corpus.embed_text(await corpus.chunk_text("alpha.md"))
    await corpus.store.search(vector, k=5, filters=SearchFilters(), ef_search=400)

    async with corpus.sessions() as session:
        leaked = (await session.execute(text("SHOW hnsw.ef_search"))).scalar_one()

    assert int(leaked) != 400


async def test_ef_search_is_applied_per_query(corpus: Corpus) -> None:
    vector = await corpus.embed_text(await corpus.chunk_text("alpha.md"))

    # Both settings work; on a corpus this small they agree, which is the
    # point — the setting must not change correctness, only recall.
    narrow = await corpus.store.search(vector, k=3, filters=SearchFilters(), ef_search=1)
    wide = await corpus.store.search(vector, k=3, filters=SearchFilters(), ef_search=400)

    assert narrow
    assert wide[0].chunk_id == narrow[0].chunk_id


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


async def test_the_search_query_uses_the_index(corpus: Corpus) -> None:
    """An index that exists but is never chosen is worse than no index.

    It costs write amplification and disk and buys nothing, and the plan is the
    only place that shows which it is.

    The corpus is padded first, and well past the crossover. Measured on this
    schema, Postgres switches from a sequential scan to the HNSW index
    somewhere between roughly 1,600 and 2,900 embedded chunks — below that it
    is *correct* to scan, because reading a few hundred rows beats descending a
    graph. Asserting on the plan against a fixture-sized table would only prove
    that Postgres can count.
    """
    await _pad_corpus(corpus, count=4000)

    vector = await corpus.embed_text("a query about anything at all")
    literal = "[" + ",".join(f"{value:.8f}" for value in vector) + "]"

    async with corpus.sessions() as session:
        rows = await session.execute(
            text(
                "EXPLAIN SELECT c.id FROM memory_chunks c "
                "JOIN memories m ON m.id = c.memory_id "
                "WHERE c.embedding IS NOT NULL AND m.is_current AND m.deleted_at IS NULL "
                f"ORDER BY c.embedding <#> '{literal}'::vector LIMIT 10"
            )
        )
        plan = "\n".join(row[0] for row in rows)

    assert "ix_memory_chunks_embedding_hnsw" in plan, plan
    assert "Index Scan" in plan, plan


async def _pad_corpus(corpus: Corpus, *, count: int) -> None:
    """Insert enough embedded chunks that the index becomes the cheaper plan.

    Vectors are generated in Postgres rather than round-tripped from Python:
    four thousand 384-float lists is a lot of bytes to push over the wire for a
    test that only cares about row counts.
    """
    async with corpus.sessions() as session:
        memory_id = (
            await session.execute(
                select(models.Memory.id)
                .where(models.Memory.is_current.is_(True))
                .limit(1)
            )
        ).scalar_one()
        base = (
            await session.execute(
                select(func.max(models.MemoryChunk.ordinal)).where(
                    models.MemoryChunk.memory_id == memory_id
                )
            )
        ).scalar_one()

    async with corpus.sessions.begin() as session:
        await session.execute(
            text(
                "CREATE OR REPLACE FUNCTION pg_temp.rand_vec() RETURNS vector AS "
                "$$ SELECT array_agg(random())::vector FROM generate_series(1, 384) $$ "
                "LANGUAGE sql VOLATILE"
            )
        )
        await session.execute(
            text(
                "INSERT INTO memory_chunks (id, memory_id, ordinal, content, token_count, "
                "  char_start, char_end, chunker_version, content_hash, embedding, "
                "  embedding_model) "
                "SELECT gen_random_uuid(), :memory_id, :base + g, 'padding ' || g, 4, "
                "  g, g + 5, 'padding-v1', md5(g::text) || md5((g + 1)::text), "
                "  pg_temp.rand_vec(), 'padding' "
                "FROM generate_series(1, :count) g"
            ).bindparams(memory_id=memory_id, base=base + 1, count=count)
        )
        # Both tables. Truncation between tests leaves stale statistics, and a
        # planner that believes `memories` holds one row estimates the join at
        # seventeen and sorts them without ever considering the index.
        await session.execute(text("ANALYZE memory_chunks"))
        await session.execute(text("ANALYZE memories"))


async def test_search_exact_deliberately_avoids_the_index(corpus: Corpus) -> None:
    # Ground truth has to be exhaustive by construction, not by hoping the
    # planner agrees.
    vector = await corpus.embed_text(await corpus.chunk_text("alpha.md"))
    exhaustive = await corpus.store.search_exact(vector, k=5, filters=SearchFilters())
    approximate = await corpus.store.search(vector, k=5, filters=SearchFilters())

    assert exhaustive
    assert exhaustive[0].chunk_id == approximate[0].chunk_id


# --------------------------------------------------------------------------
# The use case
# --------------------------------------------------------------------------


async def test_search_returns_memories_carrying_their_chunks(corpus: Corpus) -> None:
    result = await corpus.search(await corpus.chunk_text("alpha.md"), k=3)

    assert result.hits
    top = result.hits[0]
    assert top.external_key == "alpha.md"
    assert top.matched_chunks
    # Chunk-level evidence behind a memory-level answer: exactly what Phase 2's
    # citations need.
    assert top.score == max(chunk.score for chunk in top.matched_chunks)
    assert [c.ordinal for c in top.matched_chunks] == sorted(
        c.ordinal for c in top.matched_chunks
    )


async def test_search_reports_timings(corpus: Corpus) -> None:
    result = await corpus.search("a query", k=3)

    assert result.timing.total_ms >= 0
    assert result.timing.embed_ms >= 0
    assert result.timing.search_ms >= 0
    assert result.timing.total_ms >= result.timing.search_ms


async def test_search_on_an_empty_corpus_returns_nothing(
    sessions: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    embedder = FakeEmbedder()
    store = PgVectorStore(sessions, embedder)
    result = await SearchMemories(sessions, embedder, store)("anything", k=5)

    assert result.hits == []


# --------------------------------------------------------------------------
# Recall measurement
# --------------------------------------------------------------------------


async def test_recall_measurement_runs_and_reports_perfect_self_retrieval(
    corpus: Corpus,
) -> None:
    rows = await measure_recall(
        corpus.sessions,
        corpus.embedder,
        corpus.store,
        queries=5,
        k=5,
        ef_search_values=[40, 200],
    )

    assert [row.ef_search for row in rows] == [40, 200]
    for row in rows:
        # A corpus this small is effectively exhaustive under HNSW.
        assert row.mean_recall == pytest.approx(1.0)
        # And a chunk used as its own query must come back first — the
        # correctness check hiding inside the recall harness.
        assert row.self_retrieval == pytest.approx(1.0)
