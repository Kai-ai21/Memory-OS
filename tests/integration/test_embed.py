"""The embedding pipeline against a real database, with a fake model."""


import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import cache_key_for
from memoryos.application.backfill import enqueue_embedding, find_unembedded, gather_stats
from memoryos.application.embed import EmbedOutcome
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType, PermanentError
from tests.integration.conftest import PARAGRAPH, Pipeline
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


async def chunk_rows(
    sessions: async_sessionmaker[AsyncSession],
) -> list[models.MemoryChunk]:
    async with sessions() as session:
        return list(
            (
                await session.execute(
                    select(models.MemoryChunk).order_by(models.MemoryChunk.ordinal)
                )
            ).scalars()
        )


async def count_of(sessions: async_sessionmaker[AsyncSession], model: type) -> int:
    async with sessions() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


async def test_every_chunk_gets_a_vector_of_the_right_width(pipeline: Pipeline) -> None:
    await pipeline.ingest()

    assert all(chunk.embedding is None for chunk in await chunk_rows(pipeline.sessions))

    await pipeline.embed_all()

    chunks = await chunk_rows(pipeline.sessions)
    assert chunks
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384
        assert chunk.embedding_model == pipeline.embedder.model_id
        assert chunk.embedded_at is not None


async def test_vectors_are_cached_for_reuse(pipeline: Pipeline) -> None:
    await pipeline.ingest()
    await pipeline.embed_all()

    chunks = await chunk_rows(pipeline.sessions)
    assert await count_of(pipeline.sessions, models.EmbeddingCacheEntry) == len(chunks)

    async with pipeline.sessions() as session:
        entry = (
            await session.execute(select(models.EmbeddingCacheEntry).limit(1))
        ).scalar_one()
    assert entry.model_id == pipeline.embedder.model_id
    assert entry.dimension == 384


async def test_embedding_is_batched(pipeline: Pipeline) -> None:
    pipeline.batch_size = 2
    await pipeline.ingest()

    await pipeline.embed_all()

    assert all(len(batch) <= 2 for batch in pipeline.embedder.calls)
    assert pipeline.embedder.calls


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


async def test_re_embedding_an_embedded_memory_does_no_work(
    pipeline: Pipeline,
) -> None:
    await pipeline.ingest()
    await pipeline.embed_all()
    before = pipeline.embedder.texts_embedded

    fresh = FakeEmbedder()
    async with pipeline.sessions() as session:
        memory_ids = list(
            (await session.execute(select(models.Memory.id))).scalars()
        )
    reports = [await pipeline.embedder_for(fresh)(mid) for mid in memory_ids]

    assert {report.outcome for report in reports} == {EmbedOutcome.SKIPPED}
    # Not merely cached — never even looked up, because no chunk was pending.
    assert fresh.texts_embedded == 0
    assert pipeline.embedder.texts_embedded == before


async def test_identical_text_in_two_memories_is_embedded_once(
    pipeline: Pipeline,
) -> None:
    """Why the cache is its own table rather than a column on memory_chunks.

    Two files with the same content are two memories and two chunks — they are
    distinct items in the world — but one vector.
    """
    (pipeline.root / "copy.txt").write_text(PARAGRAPH * 3 + "\n")

    await pipeline.ingest()
    await pipeline.embed_all()

    async with pipeline.sessions() as session:
        chunk_contents = list(
            (await session.execute(select(models.MemoryChunk.content))).scalars()
        )
    duplicated = [text for text in chunk_contents if chunk_contents.count(text) > 1]
    assert duplicated, "fixture did not actually produce duplicate chunk text"

    keys = {cache_key_for(pipeline.embedder.model_id, text) for text in chunk_contents}
    assert await count_of(pipeline.sessions, models.EmbeddingCacheEntry) == len(keys)
    assert len(keys) < len(chunk_contents)

    # And both chunks are embedded, from the one cached vector.
    for chunk in await chunk_rows(pipeline.sessions):
        assert chunk.embedding is not None


async def test_a_line_ending_change_costs_no_embedding_calls(
    pipeline: Pipeline,
) -> None:
    """The payoff from M1.4's chunk adoption, measured.

    Different bytes, so a new artifact and a new memory version. The normalized
    text is identical, so M1.4 moves the chunk rows to the new version — and the
    vectors ride along on those rows. A counting embedder proves the model was
    never asked for anything.
    """
    await pipeline.ingest()
    await pipeline.embed_all()

    counting = FakeEmbedder()
    guide = pipeline.root / "guide.md"
    guide.write_bytes(guide.read_text().replace("\n", "\r\n").encode())

    await pipeline.sync(pipeline.source.id, full=True)
    for memory_id in await pipeline.job_targets(JobType.NORMALIZE_MEMORY):
        await pipeline.normalize(memory_id)
    await pipeline.clear_jobs(JobType.NORMALIZE_MEMORY)

    reports = await pipeline.embed_all(counting)

    assert counting.texts_embedded == 0
    assert counting.calls == []
    assert all(report.outcome is EmbedOutcome.SKIPPED for report in reports)

    for chunk in await chunk_rows(pipeline.sessions):
        assert chunk.embedding is not None
        assert chunk.embedding_model == counting.model_id


# --------------------------------------------------------------------------
# Model changes
# --------------------------------------------------------------------------


async def test_changing_the_model_re_embeds_everything(pipeline: Pipeline) -> None:
    await pipeline.ingest()
    await pipeline.embed_all()
    chunks = await count_of(pipeline.sessions, models.MemoryChunk)
    cache_after_first = await count_of(pipeline.sessions, models.EmbeddingCacheEntry)

    upgraded = FakeEmbedder(model_id="fake/deterministic@2")
    pending = await find_unembedded(pipeline.sessions, model_id=upgraded.model_id)
    assert pending, "a model change must make every memory pending again"
    assert await enqueue_embedding(pipeline.sessions, pending) == len(pending)

    await pipeline.embed_all(upgraded)

    for chunk in await chunk_rows(pipeline.sessions):
        assert chunk.embedding_model == upgraded.model_id
    assert upgraded.texts_embedded == chunks

    # Both coordinate systems are cached, keyed apart by model id.
    assert await count_of(pipeline.sessions, models.EmbeddingCacheEntry) == (
        cache_after_first * 2
    )
    async with pipeline.sessions() as session:
        models_cached = set(
            (
                await session.execute(select(models.EmbeddingCacheEntry.model_id).distinct())
            ).scalars()
        )
    assert models_cached == {pipeline.embedder.model_id, upgraded.model_id}


async def test_stale_selection_finds_chunks_from_another_model(
    pipeline: Pipeline,
) -> None:
    await pipeline.ingest()
    await pipeline.embed_all()

    same = await find_unembedded(
        pipeline.sessions, model_id=pipeline.embedder.model_id, stale_only=True
    )
    assert same == []

    other = await find_unembedded(
        pipeline.sessions, model_id="fake/deterministic@2", stale_only=True
    )
    assert other


async def test_unembedded_selection_covers_null_and_stale(pipeline: Pipeline) -> None:
    # Both conditions matter: null is the backfill case, mismatched model is
    # the upgrade case, and a query that only handled one would silently skip
    # the other.
    await pipeline.ingest()

    null_pending = await find_unembedded(
        pipeline.sessions, model_id=pipeline.embedder.model_id
    )
    assert null_pending

    await pipeline.embed_all()
    assert await find_unembedded(pipeline.sessions, model_id=pipeline.embedder.model_id) == []

    stale_pending = await find_unembedded(pipeline.sessions, model_id="fake/other@1")
    assert stale_pending


# --------------------------------------------------------------------------
# Failure
# --------------------------------------------------------------------------


async def test_a_dimension_mismatch_is_refused_before_any_write(
    pipeline: Pipeline,
) -> None:
    await pipeline.ingest()
    broken = FakeEmbedder(broken_dimension=128)

    with pytest.raises(PermanentError, match="128-dimensional"):
        await pipeline.embed_all(broken)

    # Nothing partial: no chunk updated, and no wrong-width vector cached.
    for chunk in await chunk_rows(pipeline.sessions):
        assert chunk.embedding is None
    assert await count_of(pipeline.sessions, models.EmbeddingCacheEntry) == 0


async def test_a_crash_after_embedding_leaves_the_retry_free(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is written before the chunk update, and separately.

    So a crash between the two costs the vectors nothing: the retried job finds
    every one of them in cache and never touches the model.
    """
    await pipeline.ingest()

    embed = pipeline.embedder_for()
    original_write = embed._write

    async def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("crash between embedding and the chunk update")

    monkeypatch.setattr(embed, "_write", explode)

    async with pipeline.sessions() as session:
        memory_ids = list((await session.execute(select(models.Memory.id))).scalars())

    for memory_id in memory_ids:
        with pytest.raises(RuntimeError, match="crash between"):
            await embed(memory_id)

    # No partial state: chunks untouched, but the vectors survived in cache.
    for chunk in await chunk_rows(pipeline.sessions):
        assert chunk.embedding is None
    assert await count_of(pipeline.sessions, models.EmbeddingCacheEntry) > 0

    monkeypatch.setattr(embed, "_write", original_write)
    retry = FakeEmbedder()
    reports = [await pipeline.embedder_for(retry)(mid) for mid in memory_ids]

    assert retry.texts_embedded == 0
    assert all(report.cache_hits > 0 for report in reports)
    for chunk in await chunk_rows(pipeline.sessions):
        assert chunk.embedding is not None


async def test_embedding_a_missing_memory_fails_permanently(pipeline: Pipeline) -> None:
    with pytest.raises(PermanentError, match="no such memory"):
        await pipeline.embedder_for()(new_id())


async def test_a_tombstoned_memory_is_not_embedded(pipeline: Pipeline) -> None:
    await pipeline.ingest()
    async with pipeline.sessions() as session:
        memory_id = (
            await session.execute(
                select(models.Memory.id).where(models.Memory.external_key == "guide.md")
            )
        ).scalar_one()

    (pipeline.root / "guide.md").unlink()
    await pipeline.sync(pipeline.source.id, full=True)

    report = await pipeline.embedder_for()(memory_id)
    assert report.outcome is EmbedOutcome.DELETED


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


async def test_stats_report_coverage(pipeline: Pipeline) -> None:
    await pipeline.ingest()

    before = await gather_stats(pipeline.sessions)
    assert before.chunks > 0
    assert before.embedded_chunks == 0
    assert before.coverage == 0.0

    await pipeline.embed_all()

    after = await gather_stats(pipeline.sessions)
    assert after.embedded_chunks == after.chunks
    assert after.coverage == 1.0
    assert after.cache_entries > 0
    assert after.models == {pipeline.embedder.model_id: after.chunks}
