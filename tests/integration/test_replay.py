"""Replay against a real database, with a fake model.

The milestone's whole claim is here: ingest a corpus, record exactly what the
derived tables say, destroy them, rebuild from the log and the blobs, and assert
the result is the same corpus — by content, on natural keys, not by row counts.

A fake embedder is right for all of this. Whether the vectors mean anything is
M1.5's question and the slow suite's job; what these tests establish is that the
same text lands in the same chunk of the same version of the same memory, and
that the vector attached to it is the same vector. A deterministic fake proves
that better than the real model would, because it is reproducible.
"""

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select, update
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.embed import EmbedMemory
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.replay import (
    MissingBlob,
    PartialShadowReplay,
    ReplayCorpus,
    ReplayScope,
    ReplayStage,
    truncate_derived,
)
from memoryos.application.verification import compare, snapshot
from memoryos.config import Settings
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import ContentHash
from tests.integration.conftest import (
    BREAD_TEXT,
    QUEUE_TEXT,
    Harness,
    add_source,
    build_harness,
    shadow_schemas,
)
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

# --------------------------------------------------------------------------
# The milestone's central claim
# --------------------------------------------------------------------------


async def test_a_truncated_corpus_rebuilds_identically(harness: Harness) -> None:
    """Snapshot, destroy, rebuild, compare. This is the milestone.

    The tables are truncated before the replay rather than relying on the
    replay's own truncation, so that nothing left behind can be mistaken for
    something rebuilt.
    """
    before = await harness.snapshot()
    assert before.memories and before.chunks

    await truncate_derived(harness.sessions, clear_cache=False)
    assert await harness.count(models.Memory) == 0
    assert await harness.count(models.MemoryChunk) == 0

    report = await harness.replay()

    after = await harness.snapshot()
    result = compare(before, after)
    assert result.identical, result.render()
    assert report.events == len(before.memories)
    assert report.chunks == len(before.chunks)


async def test_the_source_of_truth_is_untouched_by_a_replay(harness: Harness) -> None:
    """A replay that rewrote the log would be proving nothing about the log."""
    async with harness.sessions() as session:
        events = list(
            (
                await session.execute(
                    select(models.IngestionEvent.seq, models.IngestionEvent.id).order_by(
                        models.IngestionEvent.seq
                    )
                )
            ).all()
        )
        artifacts = await harness.count(models.RawArtifact)

    await harness.replay(clear_cache=True)

    async with harness.sessions() as session:
        after = list(
            (
                await session.execute(
                    select(models.IngestionEvent.seq, models.IngestionEvent.id).order_by(
                        models.IngestionEvent.seq
                    )
                )
            ).all()
        )
    assert after == events
    assert await harness.count(models.RawArtifact) == artifacts


async def test_clearing_the_cache_gives_the_same_result_and_refills_it(
    harness: Harness,
) -> None:
    """The honest replay: every vector recomputed, and the same vectors produced.

    Also the only version of this test that proves the embedding half of the
    pipeline runs at all. With the cache kept, identical output would be
    consistent with the embedder never being called.
    """
    before = await harness.snapshot()
    cached_before = await harness.count(models.EmbeddingCacheEntry)
    assert cached_before > 0

    calls_before = harness.embedder.texts_embedded
    report = await harness.replay(clear_cache=True)

    assert harness.embedder.texts_embedded > calls_before, "nothing was re-embedded"
    # Every distinct chunk text computed exactly once — no more, because the
    # cache refills as the run goes and the two identical fixture files share a
    # chunk; no fewer, because the cache started empty.
    distinct = len({chunk.content_hash for chunk in before.chunks})
    assert report.vectors_computed == distinct
    assert report.cache_hits == len(before.chunks) - distinct

    after = await harness.snapshot()
    assert compare(before, after).identical, compare(before, after).render()
    # Repopulated, and to the same size: the key is a pure function of model,
    # role and text, so the same corpus yields the same set of entries.
    assert await harness.count(models.EmbeddingCacheEntry) == cached_before


async def test_keeping_the_cache_recomputes_nothing(harness: Harness) -> None:
    calls_before = harness.embedder.texts_embedded

    report = await harness.replay(clear_cache=False)

    assert report.vectors_computed == 0
    assert report.cache_hits > 0
    assert harness.embedder.texts_embedded == calls_before


async def test_replaying_twice_gives_the_same_result(harness: Harness) -> None:
    # Idempotence. A replay that accumulated anything — a duplicate version, a
    # stray chunk — would show up on the second pass.
    first = await harness.replay()
    once = await harness.snapshot()

    second = await harness.replay()
    twice = await harness.snapshot()

    assert compare(once, twice).identical, compare(once, twice).render()
    assert first.events == second.events
    assert first.chunks == second.chunks


# --------------------------------------------------------------------------
# The test most likely to fail: versions and tombstones
# --------------------------------------------------------------------------


async def test_versions_and_tombstones_survive_a_rebuild(harness: Harness) -> None:
    """Three versions of one file and a deletion of another.

    This is the one the milestone flagged as the likeliest correctness failure,
    and it is invisible to counts: the wrong number of versions changes a count,
    but versions numbered in the wrong order, or `is_current` on the wrong row,
    does not.
    """
    (harness.root / "queue.md").write_text("# Queue\n\n" + QUEUE_TEXT * 5 + "\n")
    await harness.ingest()
    (harness.root / "queue.md").write_text("# Queue v3\n\n" + QUEUE_TEXT * 6 + "\n")
    await harness.ingest()
    (harness.root / "bread.txt").unlink()
    await harness.ingest()

    before = await harness.snapshot()
    versions = sorted(
        (row.version, row.is_current)
        for row in before.memories
        if row.external_key == "queue.md"
    )
    assert versions == [(1, False), (2, False), (3, True)], versions
    tombstoned = [row for row in before.memories if row.external_key == "bread.txt"]
    assert len(tombstoned) == 1
    assert tombstoned[0].deleted_at is not None

    await truncate_derived(harness.sessions, clear_cache=True)
    await harness.replay(clear_cache=True)

    after = await harness.snapshot()
    assert compare(before, after).identical, compare(before, after).render()


async def test_exactly_one_version_is_current_after_a_rebuild(
    harness: Harness,
) -> None:
    (harness.root / "queue.md").write_text("# Queue\n\n" + QUEUE_TEXT * 7 + "\n")
    await harness.ingest()

    await truncate_derived(harness.sessions, clear_cache=False)
    await harness.replay()

    async with harness.sessions() as session:
        rows = list(
            (
                await session.execute(
                    select(models.Memory.external_key, func.count())
                    .where(models.Memory.is_current.is_(True))
                    .group_by(models.Memory.external_key)
                )
            ).all()
        )
    assert all(count == 1 for _, count in rows), rows


async def test_a_deleted_file_that_returns_gets_a_new_version(
    harness: Harness,
) -> None:
    """Delete then restore, which is where a tombstone-aware rebuild goes wrong.

    The restored file must become a *new* version rather than clearing the
    tombstone in place, because the log records two separate facts and a replay
    has to reproduce both.
    """
    (harness.root / "bread.txt").unlink()
    await harness.ingest()
    (harness.root / "bread.txt").write_text(BREAD_TEXT * 4 + "\n")
    await harness.ingest()

    before = await harness.snapshot()
    rows = sorted(
        (row.version, row.is_current, row.deleted_at is not None)
        for row in before.memories
        if row.external_key == "bread.txt"
    )
    assert rows == [(1, False, True), (2, True, False)], rows

    await truncate_derived(harness.sessions, clear_cache=False)
    await harness.replay()

    after = await harness.snapshot()
    assert compare(before, after).identical, compare(before, after).render()


async def test_a_superseded_version_keeps_no_chunks(harness: Harness) -> None:
    # A version nobody can retrieve leaving chunks behind is the same failure as
    # a deleted memory leaving chunks behind.
    (harness.root / "queue.md").write_text("# Queue\n\n" + QUEUE_TEXT * 9 + "\n")
    await harness.ingest()

    await truncate_derived(harness.sessions, clear_cache=False)
    await harness.replay()

    after = await harness.snapshot()
    stale = [
        chunk
        for chunk in after.chunks
        if any(
            memory.external_key == chunk.external_key
            and memory.version == chunk.memory_version
            and not memory.is_current
            for memory in after.memories
        )
    ]
    assert stale == []


async def test_a_tombstone_is_stamped_from_the_event_not_the_clock(
    harness: Harness,
) -> None:
    """`deleted_at` has to equal the deletion event's `recorded_at`.

    It used to come from `now()`, which was within milliseconds of the event on
    the original write and therefore looked fine. A rebuild months later would
    stamp a completely different value, and nothing but a column-level comparison
    would ever say so. This asserts equality on both sides of a replay.
    """
    (harness.root / "bread.txt").unlink()
    await harness.ingest()

    async with harness.sessions() as session:
        recorded_at = (
            await session.execute(
                select(models.IngestionEvent.recorded_at).where(
                    models.IngestionEvent.external_key == "bread.txt",
                    models.IngestionEvent.event_type == "item_deleted",
                )
            )
        ).scalar_one()
        deleted_at = (
            await session.execute(
                select(models.Memory.deleted_at).where(
                    models.Memory.external_key == "bread.txt"
                )
            )
        ).scalar_one()
    assert deleted_at == recorded_at

    await truncate_derived(harness.sessions, clear_cache=False)
    await harness.replay()

    async with harness.sessions() as session:
        rebuilt = (
            await session.execute(
                select(models.Memory.deleted_at).where(
                    models.Memory.external_key == "bread.txt"
                )
            )
        ).scalar_one()
    assert rebuilt == recorded_at


async def test_a_tombstoned_version_keeps_exactly_the_chunk_it_had(
    harness: Harness,
) -> None:
    """It was chunked before it was deleted, so the chunk is part of the history.

    Worth pinning because it is counter-intuitive: a deleted memory still owns
    chunks, and they still carry vectors. They are invisible to search — the
    vector store filters on `deleted_at` — but they exist, and a rebuild that
    dropped them would not be a rebuild of this corpus. The first version of the
    replay did exactly that.
    """
    before = await harness.snapshot()
    original = [chunk for chunk in before.chunks if chunk.external_key == "bread.txt"]
    assert original, "the fixture never chunked this file"

    (harness.root / "bread.txt").unlink()
    await harness.ingest()

    await truncate_derived(harness.sessions, clear_cache=False)
    await harness.replay()

    after = await harness.snapshot()
    kept = [chunk for chunk in after.chunks if chunk.external_key == "bread.txt"]
    assert kept == original


# --------------------------------------------------------------------------
# Scopes
# --------------------------------------------------------------------------


async def test_stage_embed_keeps_the_chunk_rows_and_only_changes_vectors(
    harness: Harness,
) -> None:
    """The scope that will actually get used: swap the model, keep the chunks."""
    ids_before = await harness.chunk_ids()
    before = await harness.snapshot()

    upgraded = FakeEmbedder(model_id="fake/upgraded@1")
    replay = ReplayCorpus(
        harness.sessions,
        make_normalize=lambda factory: NormalizeMemory(
            factory,
            harness.blobs,
            build_parsers(),
            StructuralChunker(harness.embedder),
            enqueue_followup=False,
        ),
        make_embed=lambda factory: EmbedMemory(
            factory, upgraded, PostgresEmbeddingCache(factory)
        ),
    )
    report = await replay(ReplayScope(stage=ReplayStage.EMBED))

    assert await harness.chunk_ids() == ids_before, "chunk identity was not preserved"
    assert report.events == 0, "an embed-stage replay should read no events"

    after = await harness.snapshot()
    # Same rows, same text, different vectors and a different model stamp.
    assert [chunk.key for chunk in after.chunks] == [
        chunk.key for chunk in before.chunks
    ]
    assert all(chunk.embedding_model == "fake/upgraded@1" for chunk in after.chunks)
    changed = compare(before, after).diffs[1]
    assert len(changed.changed) == len(before.chunks)
    assert not changed.missing and not changed.unexpected


@pytest.mark.parametrize("stage", [ReplayStage.EMBED, ReplayStage.NORMALIZE])
async def test_a_partial_stage_does_not_destroy_what_it_will_not_rebuild(
    harness: Harness, stage: ReplayStage
) -> None:
    """A tombstoned memory's chunks must survive a downstream-only replay.

    `_targets` will not re-chunk or re-embed a tombstone — correctly, since
    re-deriving one would mean re-reading a file that is gone. So clearing its
    chunks or its vectors leaves nothing to restore them. The first version of
    this cleared every chunk in scope, which meant a routine `--stage embed` model
    swap permanently stripped the vectors behind every deleted file, and left a
    corpus that `doctor` would report as partly unembedded forever.
    """
    (harness.root / "bread.txt").unlink()
    await harness.ingest()

    before = await harness.snapshot()
    tombstoned = [chunk for chunk in before.chunks if chunk.external_key == "bread.txt"]
    assert tombstoned, "the fixture has no tombstoned chunk to protect"
    assert all(chunk.embedding_digest is not None for chunk in tombstoned)

    await harness.replay(ReplayScope(stage=stage), clear_cache=True)

    after = await harness.snapshot()
    kept = [chunk for chunk in after.chunks if chunk.external_key == "bread.txt"]
    assert kept == tombstoned, "a tombstoned memory's chunks were altered or lost"


async def test_stage_normalize_rebuilds_chunks_without_touching_memories(
    harness: Harness,
) -> None:
    async with harness.sessions() as session:
        memory_ids = sorted(
            str(row[0]) for row in await session.execute(select(models.Memory.id))
        )

    report = await harness.replay(ReplayScope(stage=ReplayStage.NORMALIZE))

    async with harness.sessions() as session:
        after_ids = sorted(
            str(row[0]) for row in await session.execute(select(models.Memory.id))
        )
    # Upstream artifacts kept: the memory rows, and therefore their ids, survive.
    assert after_ids == memory_ids
    assert report.events == 0
    assert report.normalized > 0


async def test_replaying_one_source_leaves_another_untouched(
    harness: Harness, tmp_path: Path, sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "elsewhere.md").write_text("# Elsewhere\n\n" + BREAD_TEXT * 5 + "\n")
    other_source = await add_source(sessions, "other", other_root)
    other = build_harness(
        other_root, tmp_path / "other-blobs", sessions, other_source, settings
    )
    await other.ingest()

    before = await snapshot(sessions)
    other_rows = [row for row in before.memories if row.source_name == "other"]
    other_chunks = [row for row in before.chunks if row.source_name == "other"]
    assert other_rows and other_chunks

    await harness.replay(ReplayScope(source_name="corpus"), clear_cache=True)

    after = await snapshot(sessions)
    # The other source's rows are not merely present, they are unchanged —
    # including their ids, which a rebuild would have replaced.
    assert [row for row in after.memories if row.source_name == "other"] == other_rows
    assert [row for row in after.chunks if row.source_name == "other"] == other_chunks
    # And the replayed source did get rebuilt.
    assert compare(before, after).identical, compare(before, after).render()


async def test_a_scoped_replay_only_clears_the_jobs_it_invalidates(
    harness: Harness, tmp_path: Path, sessions: async_sessionmaker[AsyncSession],
    settings: Settings,
) -> None:
    """Queued work for a source nobody asked to replay must survive.

    A scoped replay changes the memory ids of the source it rebuilds, so that
    source's pending jobs are genuinely dead. Every other source's are fine, and
    clearing the whole queue — which is what this did first — would silently throw
    away work that was going to happen.
    """
    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "elsewhere.md").write_text("# Elsewhere\n\n" + BREAD_TEXT * 5 + "\n")
    other_source = await add_source(sessions, "other", other_root)
    other = build_harness(
        other_root, tmp_path / "other-blobs", sessions, other_source, settings
    )
    await other.ingest()

    # A pending job for each source, of the kind sync leaves behind.
    async with sessions() as session:
        one_per_source: dict[str, UUID] = {
            row[0]: row[1]
            for row in await session.execute(
                select(models.Source.name, models.Memory.id)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Memory.is_current.is_(True))
            )
        }
    async with sessions.begin() as session:
        for name, memory_id in one_per_source.items():
            session.add(
                models.Job(
                    id=new_id(),
                    job_type=JobType.EMBED_MEMORY.value,
                    payload={"memory_id": str(memory_id), "source": name},
                )
            )

    await harness.replay(ReplayScope(source_name="corpus"))

    async with sessions() as session:
        surviving = {
            row[0]["source"]
            for row in await session.execute(select(models.Job.payload))
        }
    assert surviving == {"other"}, surviving


async def test_replaying_since_a_sequence_only_applies_later_events(
    harness: Harness,
) -> None:
    async with harness.sessions() as session:
        midpoint = int(
            (
                await session.execute(
                    select(func.min(models.IngestionEvent.seq))
                )
            ).scalar_one()
        )

    report = await harness.replay(ReplayScope(after_seq=midpoint))

    async with harness.sessions() as session:
        total = int(
            (
                await session.execute(select(func.count()).select_from(models.IngestionEvent))
            ).scalar_one()
        )
    # Strictly after: the event at `midpoint` is excluded.
    assert report.events == total - 1


async def test_an_unknown_source_name_is_reported_rather_than_ignored(
    harness: Harness,
) -> None:
    # Silently replaying everything would be far worse than an error.
    with pytest.raises(LookupError, match="no source named"):
        await harness.replay(ReplayScope(source_name="does-not-exist"))


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


async def test_a_missing_blob_is_refused_before_anything_is_destroyed(
    harness: Harness,
) -> None:
    """The check runs before the truncate, not after it.

    The rebuild rests entirely on the blobs being there, and skipping a document
    would produce a corpus quietly missing a file — a memory row with no chunks,
    which is also what an empty file looks like. So it fails, loudly, naming the
    file and the whole hash.

    That it fails *first* was learned the hard way: running `replay` from a
    subdirectory resolved the default relative `blob_root` to an empty path, and
    the run truncated 119 memories before failing on the first document. The
    corpus was rebuildable from the same log once the command was run from the
    right place — the system working as designed — but "destroys your corpus,
    then explains why" is the wrong failure mode for a destructive operation when
    the check costs one stat per artifact.
    """
    before = await harness.snapshot()
    async with harness.sessions() as session:
        content_hash = (
            await session.execute(
                select(models.Memory.content_hash).where(
                    models.Memory.external_key == "queue.md",
                    models.Memory.is_current.is_(True),
                )
            )
        ).scalar_one()

    await harness.blobs.delete(ContentHash(content_hash))

    with pytest.raises(MissingBlob) as caught:
        await harness.replay()

    message = str(caught.value)
    # Names the file and the whole hash, so the next step is a search rather
    # than a guess.
    assert "queue.md" in message
    assert content_hash in message
    assert "Nothing has been changed" in message
    # Untouched: not merely present, byte-identical.
    assert compare(before, await harness.snapshot()).identical


async def test_an_unreplayable_event_type_is_refused(harness: Harness) -> None:
    """A future event kind must not be silently skipped.

    Ignoring it would mean the projection quietly stops reflecting part of the
    log — a corpus that disagrees with its own source of truth, with nothing
    reporting it. The CHECK constraint is dropped for the duration because the
    scenario being simulated is precisely a schema that has learned a new event
    kind which this code has not.
    """
    async with harness.sessions.begin() as session:
        await session.execute(
            sa_text(
                "ALTER TABLE ingestion_events "
                "DROP CONSTRAINT ck_ingestion_events_event_type"
            )
        )
        await session.execute(
            update(models.IngestionEvent)
            .where(
                models.IngestionEvent.seq
                == select(func.min(models.IngestionEvent.seq)).scalar_subquery()
            )
            .values(event_type="item_archived")
        )

    try:
        # Refused at the enum boundary, before the projection sees it — which is
        # the right layer, and loud either way. What matters is that the replay
        # stops rather than skipping the event and reporting success.
        with pytest.raises(ValueError, match="item_archived"):
            await harness.replay()
    finally:
        async with harness.sessions.begin() as session:
            await session.execute(
                sa_text(
                    "ALTER TABLE ingestion_events ADD CONSTRAINT "
                    "ck_ingestion_events_event_type CHECK "
                    "(event_type IN ('artifact_observed', 'item_deleted')) NOT VALID"
                )
            )


# --------------------------------------------------------------------------
# Verification, which has to be able to fail
# --------------------------------------------------------------------------


async def test_verification_passes_on_an_honest_corpus(harness: Harness) -> None:
    before = await harness.snapshot()

    async with harness.replay.rebuild_into_shadow() as (report, shadow_sessions):
        after = await snapshot(shadow_sessions)

    result = compare(before, after)
    assert result.identical, result.render()
    assert report.into_shadow is True


async def test_verification_notices_a_corrupted_chunk(harness: Harness) -> None:
    """A verification that cannot fail is not a verification.

    The corruption is chosen to be invisible to counts: one chunk's text is
    replaced, so the number of memories, chunks and embedded chunks all stay
    exactly the same.
    """
    async with harness.sessions.begin() as session:
        chunk_id = (
            await session.execute(select(models.MemoryChunk.id).limit(1))
        ).scalar_one()
        await session.execute(
            update(models.MemoryChunk)
            .where(models.MemoryChunk.id == chunk_id)
            .values(content_hash=ContentHash.of(b"tampered").value)
        )

    before = await harness.snapshot()
    async with harness.replay.rebuild_into_shadow() as (_, shadow_sessions):
        after = await snapshot(shadow_sessions)

    result = compare(before, after)
    assert not result.identical
    # Counts agree, which is the point.
    assert result.before == result.after
    chunks = next(diff for diff in result.diffs if diff.table == "memory_chunks")
    assert len(chunks.changed) == 1
    assert "content_hash" in chunks.changed[0]


async def test_verification_notices_a_wrong_is_current_flag(
    harness: Harness,
) -> None:
    """The M1.6.1-shaped failure: right counts, wrong current version."""
    (harness.root / "queue.md").write_text("# Queue\n\n" + QUEUE_TEXT * 5 + "\n")
    await harness.ingest()

    async with harness.sessions.begin() as session:
        # Move the flag to the superseded version, keeping exactly one current
        # row per item so the partial unique index stays satisfied.
        rows = list(
            (
                await session.execute(
                    select(models.Memory.id, models.Memory.version)
                    .where(models.Memory.external_key == "queue.md")
                    .order_by(models.Memory.version)
                )
            ).all()
        )
        old_id, _ = rows[0]
        new_id_, _ = rows[-1]
        await session.execute(
            update(models.Memory).where(models.Memory.id == new_id_).values(is_current=False)
        )
        await session.execute(
            update(models.Memory).where(models.Memory.id == old_id).values(is_current=True)
        )

    before = await harness.snapshot()
    async with harness.replay.rebuild_into_shadow() as (_, shadow_sessions):
        after = await snapshot(shadow_sessions)

    result = compare(before, after)
    assert not result.identical
    assert result.before["memories"] == result.after["memories"]
    memories = next(diff for diff in result.diffs if diff.table == "memories")
    assert any("is_current" in change for change in memories.changed), memories.changed


async def test_a_verification_run_does_not_disturb_the_live_corpus(
    harness: Harness,
) -> None:
    before = await harness.snapshot()
    async with harness.sessions() as session:
        ids = sorted(str(row[0]) for row in await session.execute(select(models.Memory.id)))

    async with harness.replay.rebuild_into_shadow() as (_, _sessions):
        pass

    async with harness.sessions() as session:
        after_ids = sorted(
            str(row[0]) for row in await session.execute(select(models.Memory.id))
        )
    # Not just equivalent — the very same rows, ids included.
    assert after_ids == ids
    assert compare(before, await harness.snapshot()).identical


# --------------------------------------------------------------------------
# The shadow workspace
# --------------------------------------------------------------------------


async def test_a_shadow_replay_swaps_in_an_identical_corpus(harness: Harness) -> None:
    before = await harness.snapshot()

    report = await harness.replay(into_shadow=True)

    after = await harness.snapshot()
    assert compare(before, after).identical, compare(before, after).render()
    assert report.into_shadow is True


async def test_the_swapped_in_tables_keep_their_constraints_and_indexes(
    harness: Harness,
) -> None:
    """The half of the swap that a row comparison cannot see.

    `SET SCHEMA` carries constraints and indexes with the table, and they can keep
    canonical names because names are unique per schema. If they came across
    renamed, the schema would no longer match the models and the next migration
    check would break — long after the replay that caused it.
    """
    await harness.replay(into_shadow=True)

    async with harness.sessions() as session:
        indexes = {
            row[0]
            for row in await session.execute(
                sa_text("SELECT indexname FROM pg_indexes WHERE tablename='memory_chunks'")
            )
        }
        constraints = {
            row[0]
            for row in await session.execute(
                sa_text(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conrelid = 'memory_chunks'::regclass"
                )
            )
        }

    assert "pk_memory_chunks" in constraints
    assert "fk_memory_chunks_memory_id" in constraints
    assert "uq_memory_chunks_memory_ordinal" in indexes
    assert "ix_memory_chunks_embedding_hnsw" in indexes
    assert await shadow_schemas(harness.sessions) == set(), "workspace left behind"


async def test_a_swapped_in_chunk_still_points_at_a_real_memory(
    harness: Harness,
) -> None:
    """The failure the whole-schema rename would have produced.

    A foreign key surviving as a *name* while pointing at a retired table is the
    outcome that looks fine until something reads it. Every chunk resolving to a
    memory that resolves to a source is the check that would have caught it.
    """
    await harness.replay(into_shadow=True)

    async with harness.sessions() as session:
        orphans = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(models.MemoryChunk)
                    .outerjoin(
                        models.Memory, models.Memory.id == models.MemoryChunk.memory_id
                    )
                    .where(models.Memory.id.is_(None))
                )
            ).scalar_one()
        )
        detached = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(models.Memory)
                    .outerjoin(models.Source, models.Source.id == models.Memory.source_id)
                    .where(models.Source.id.is_(None))
                )
            ).scalar_one()
        )
    assert orphans == 0
    assert detached == 0


@pytest.mark.parametrize(
    "scope",
    [
        ReplayScope(source_name="corpus"),
        ReplayScope(after_seq=1),
        ReplayScope(stage=ReplayStage.EMBED),
        ReplayScope(stage=ReplayStage.NORMALIZE),
    ],
)
async def test_a_partial_scope_cannot_be_swapped_in(
    harness: Harness, scope: ReplayScope
) -> None:
    """The guardrail on a genuinely destructive combination.

    Each of these would have swapped in a workspace containing less than the whole
    corpus and silently deleted the difference — `--stage embed --into-shadow`
    would have replaced everything with nothing, and reported success. The corpus
    must be untouched afterwards.
    """
    before = await harness.snapshot()

    with pytest.raises(PartialShadowReplay, match="partial"):
        await harness.replay(scope, into_shadow=True)

    assert compare(before, await harness.snapshot()).identical
    assert await shadow_schemas(harness.sessions) == set()


async def test_a_failed_shadow_replay_leaves_the_live_corpus_alone(
    harness: Harness, tmp_path: Path, settings: Settings
) -> None:
    """The reason to build in a workspace at all.

    The blob is removed so the rebuild fails partway through normalization. The
    live tables must be exactly as they were, which an in-place replay could not
    promise.
    """
    before = await harness.snapshot()
    async with harness.sessions() as session:
        content_hash = (
            await session.execute(
                select(models.Memory.content_hash).where(
                    models.Memory.external_key == "queue.md",
                    models.Memory.is_current.is_(True),
                )
            )
        ).scalar_one()
    await harness.blobs.delete(ContentHash(content_hash))

    with pytest.raises(MissingBlob):
        await harness.replay(into_shadow=True)

    assert compare(before, await harness.snapshot()).identical
    # And the workspace was cleaned up rather than left holding a partial rebuild
    # for the next run to continue into.
    assert await shadow_schemas(harness.sessions) == set()
