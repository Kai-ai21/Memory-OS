"""The sync use case against a real database and a real directory.

`tmp_path` throughout. The filesystem is not mocked — we own neither `pathlib`
nor the OS, and mocking them would only assert that our beliefs about them are
self-consistent.
"""

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.application.sync import SyncReport, SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobSpec
from memoryos.domain.values import ContentHash, EventType, SourceKind

pytestmark = pytest.mark.integration


@dataclass(slots=True)
class Fixture:
    root: Path
    source: Source
    sync: SyncSource
    blobs: FilesystemBlobStore
    sessions: async_sessionmaker[AsyncSession]

    async def run(self, *, full: bool = False) -> SyncReport:
        return await self.sync(self.source.id, full=full)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small fixture directory, including things that must be ignored."""
    root = tmp_path / "corpus"
    (root / "notes").mkdir(parents=True)
    (root / ".git").mkdir()

    (root / "readme.md").write_text("the readme\n")
    (root / "notes" / "alpha.md").write_text("alpha\n")
    (root / "notes" / "beta.txt").write_text("beta\n")
    (root / "script.py").write_text("print('hi')\n")

    # Neither of these may ever be observed.
    (root / ".git" / "config").write_text("[core]\n")
    (root / "image.bin").write_bytes(b"\x00\x01")

    return root


@pytest.fixture
async def fixture(
    tree: Path, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> Fixture:
    source = Source(
        id=new_id(),
        kind=SourceKind.FILESYSTEM,
        name="corpus",
        config={"root": str(tree)},
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    return Fixture(
        root=tree,
        source=source,
        sync=SyncSource(sessions, FilesystemConnector(), blobs),
        blobs=blobs,
        sessions=sessions,
    )


async def count(sessions: async_sessionmaker[AsyncSession], model: type) -> int:
    async with sessions() as session:
        return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def counts(sessions: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    return {
        "artifacts": await count(sessions, models.RawArtifact),
        "events": await count(sessions, models.IngestionEvent),
        "memories": await count(sessions, models.Memory),
        "jobs": await count(sessions, models.Job),
    }


async def memories_for(
    sessions: async_sessionmaker[AsyncSession], external_key: str
) -> list[models.Memory]:
    async with sessions() as session:
        result = await session.execute(
            select(models.Memory)
            .where(models.Memory.external_key == external_key)
            .order_by(models.Memory.version)
        )
        return list(result.scalars())


def blob_files(blobs: FilesystemBlobStore) -> Iterator[Path]:
    return (path for path in blobs.root.rglob("*") if path.is_file())


# --------------------------------------------------------------------------
# First sync
# --------------------------------------------------------------------------


async def test_a_first_sync_records_everything_it_observed(fixture: Fixture) -> None:
    report = await fixture.run(full=True)

    # Four included files; .git/config and image.bin are excluded.
    assert report.observed == 4
    assert report.new == 4
    assert report.modified == 0
    assert report.skipped == 0
    assert report.deleted == 0
    assert report.errors == 0

    assert await counts(fixture.sessions) == {
        "artifacts": 4,
        "events": 4,
        "memories": 4,
        "jobs": 4,
    }


async def test_a_first_sync_stores_the_bytes(fixture: Fixture) -> None:
    await fixture.run(full=True)

    digest = ContentHash.of(b"alpha\n")
    assert await fixture.blobs.exists(digest) is True
    assert await fixture.blobs.get(digest) == b"alpha\n"
    assert len(list(blob_files(fixture.blobs))) == 4


async def test_the_events_are_artifact_observed_with_filesystem_provenance(
    fixture: Fixture,
) -> None:
    await fixture.run(full=True)

    async with fixture.sessions() as session:
        rows = list(
            (await session.execute(select(models.IngestionEvent))).scalars()
        )

    assert {row.event_type for row in rows} == {EventType.ARTIFACT_OBSERVED.value}
    assert {row.occurred_at_source for row in rows} == {"filesystem"}
    assert all(row.occurred_at is not None for row in rows)
    assert all(row.content_hash is not None for row in rows)


async def test_every_new_memory_enqueues_exactly_one_normalize_job(
    fixture: Fixture,
) -> None:
    await fixture.run(full=True)

    async with fixture.sessions() as session:
        jobs = list((await session.execute(select(models.Job))).scalars())
        memory_ids = {
            str(row)
            for row in (await session.execute(select(models.Memory.id))).scalars()
        }

    assert {job.job_type for job in jobs} == {"normalize_memory"}
    assert {job.payload["memory_id"] for job in jobs} == memory_ids
    # dedupe_key is the memory id, so a re-run cannot queue the same work twice.
    assert {job.dedupe_key for job in jobs} == memory_ids


# --------------------------------------------------------------------------
# The central assertion: re-running is free
# --------------------------------------------------------------------------


@pytest.mark.parametrize("full", [True, False])
async def test_re_syncing_an_unchanged_tree_changes_nothing(
    fixture: Fixture, full: bool
) -> None:
    """The milestone's central claim.

    Nothing changed on disk, so nothing may be written: no artifact, no event,
    no memory version, no job. This is what makes running a sync often a
    reasonable thing to do rather than an expensive one.
    """
    first = await fixture.run(full=True)
    before = await counts(fixture.sessions)

    second = await fixture.run(full=full)

    assert await counts(fixture.sessions) == before
    assert second.new == 0
    assert second.modified == 0
    assert second.deleted == 0
    assert second.errors == 0

    if full:
        # A full walk looks at everything and finds all of it unchanged.
        assert second.observed == first.observed
        assert second.skipped == second.observed
    else:
        # An incremental walk does not even reach the hash: the (mtime, size)
        # filter answered the question more cheaply.
        assert second.observed == 0
        assert second.skipped == second.observed


async def test_a_third_sync_is_still_free(fixture: Fixture) -> None:
    await fixture.run(full=True)
    await fixture.run(full=True)
    before = await counts(fixture.sessions)

    await fixture.run(full=True)

    assert await counts(fixture.sessions) == before


# --------------------------------------------------------------------------
# Change
# --------------------------------------------------------------------------


async def test_modifying_a_file_versions_the_memory(fixture: Fixture) -> None:
    await fixture.run(full=True)

    (fixture.root / "notes" / "alpha.md").write_text("alpha, revised\n")
    report = await fixture.run(full=True)

    assert report.modified == 1
    assert report.new == 0
    assert report.skipped == 3

    versions = await memories_for(fixture.sessions, "notes/alpha.md")
    assert [(row.version, row.is_current) for row in versions] == [(1, False), (2, True)]
    assert versions[1].content_hash == ContentHash.of(b"alpha, revised\n").value

    assert await count(fixture.sessions, models.RawArtifact) == 5
    assert await count(fixture.sessions, models.IngestionEvent) == 5


async def test_changing_only_line_endings_is_a_real_change(fixture: Fixture) -> None:
    """CRLF is different bytes, so it is a different artifact.

    The memory versions because the content hash is the authority and the bytes
    genuinely differ. Whether the *text* is meaningfully different is a
    normalization question, and normalization is M1.4's business, not this
    milestone's.
    """
    await fixture.run(full=True)
    before = await count(fixture.sessions, models.RawArtifact)

    (fixture.root / "notes" / "alpha.md").write_bytes(b"alpha\r\n")
    report = await fixture.run(full=True)

    assert report.modified == 1
    assert await count(fixture.sessions, models.RawArtifact) == before + 1

    versions = await memories_for(fixture.sessions, "notes/alpha.md")
    assert [(row.version, row.is_current) for row in versions] == [(1, False), (2, True)]


async def test_identical_content_at_two_paths_shares_one_artifact_and_one_blob(
    fixture: Fixture,
) -> None:
    # Identity is the content, not the path. Two paths holding the same bytes
    # are one artifact and one blob, but two distinct items in the world.
    (fixture.root / "notes" / "copy.md").write_text("alpha\n")

    report = await fixture.run(full=True)

    assert report.observed == 5
    assert report.new == 5
    assert await count(fixture.sessions, models.RawArtifact) == 4
    assert await count(fixture.sessions, models.Memory) == 5
    assert len(list(blob_files(fixture.blobs))) == 4

    alpha = await memories_for(fixture.sessions, "notes/alpha.md")
    copy = await memories_for(fixture.sessions, "notes/copy.md")
    assert alpha[0].content_hash == copy[0].content_hash


async def test_a_new_file_is_new_and_the_rest_are_skipped(fixture: Fixture) -> None:
    await fixture.run(full=True)

    (fixture.root / "notes" / "gamma.md").write_text("gamma\n")
    report = await fixture.run(full=True)

    assert report.new == 1
    assert report.modified == 0
    assert report.skipped == 4


# --------------------------------------------------------------------------
# Deletion
# --------------------------------------------------------------------------


async def test_a_full_sync_detects_a_deletion(fixture: Fixture) -> None:
    await fixture.run(full=True)
    (fixture.root / "notes" / "alpha.md").unlink()

    report = await fixture.run(full=True)

    assert report.deleted == 1
    assert report.observed == 3

    async with fixture.sessions() as session:
        deletions = list(
            (
                await session.execute(
                    select(models.IngestionEvent).where(
                        models.IngestionEvent.event_type == EventType.ITEM_DELETED.value
                    )
                )
            ).scalars()
        )

    assert len(deletions) == 1
    assert deletions[0].external_key == "notes/alpha.md"
    assert deletions[0].content_hash is None
    # We know when we noticed the absence — recorded_at — not when the deletion
    # happened. Saying otherwise would be a fabrication.
    assert deletions[0].occurred_at is None
    assert deletions[0].occurred_at_source == "unknown"
    assert deletions[0].recorded_at is not None

    (memory,) = await memories_for(fixture.sessions, "notes/alpha.md")
    assert memory.deleted_at is not None


async def test_an_incremental_sync_cannot_detect_a_deletion(fixture: Fixture) -> None:
    """The negative, asserted explicitly.

    A deleted file produces no observation at all — absence is not an event —
    so an incremental walk has nothing to notice. Only comparing the complete
    observed set against the complete known set reveals it, which is exactly
    why `sources` carries `last_full_sync_at` separately.
    """
    await fixture.run(full=True)
    (fixture.root / "notes" / "alpha.md").unlink()

    report = await fixture.run(full=False)

    assert report.deleted == 0

    async with fixture.sessions() as session:
        deletions = (
            await session.execute(
                select(func.count())
                .select_from(models.IngestionEvent)
                .where(models.IngestionEvent.event_type == EventType.ITEM_DELETED.value)
            )
        ).scalar_one()
    assert deletions == 0

    (memory,) = await memories_for(fixture.sessions, "notes/alpha.md")
    assert memory.deleted_at is None

    # And the very next full sync does find it.
    assert (await fixture.run(full=True)).deleted == 1


async def test_a_tombstone_leaves_chunks_alone(fixture: Fixture) -> None:
    # Tombstoning is not deleting. The row survives, so the ON DELETE CASCADE
    # on memory_chunks is never triggered and nothing is lost.
    await fixture.run(full=True)
    (memory,) = await memories_for(fixture.sessions, "notes/alpha.md")

    async with fixture.sessions.begin() as session:
        session.add(
            models.MemoryChunk(
                id=new_id(),
                memory_id=memory.id,
                ordinal=0,
                content="alpha",
                token_count=1,
                char_start=0,
                char_end=5,
                chunker_version="test-v1",
                content_hash=ContentHash.of(b"alpha").value,
            )
        )

    (fixture.root / "notes" / "alpha.md").unlink()
    await fixture.run(full=True)

    assert await count(fixture.sessions, models.MemoryChunk) == 1


async def test_a_deleted_file_that_returns_is_ingested_again(fixture: Fixture) -> None:
    await fixture.run(full=True)
    (fixture.root / "notes" / "alpha.md").unlink()
    await fixture.run(full=True)

    (fixture.root / "notes" / "alpha.md").write_text("alpha\n")
    report = await fixture.run(full=True)

    assert report.modified == 1
    versions = await memories_for(fixture.sessions, "notes/alpha.md")
    assert [(row.version, row.is_current) for row in versions] == [(1, False), (2, True)]
    assert versions[1].deleted_at is None


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


async def test_an_oversized_file_is_skipped_without_aborting_the_sync(
    fixture: Fixture, tmp_path: Path
) -> None:
    (fixture.root / "huge.md").write_text("y" * 10_000)

    async with fixture.sessions.begin() as session:
        source_row = await session.get(models.Source, fixture.source.id)
        assert source_row is not None
        source_row.config = {**source_row.config, "max_file_bytes": 1000}

    report = await fixture.run(full=True)

    assert report.observed == 4
    assert report.errors == 0
    assert await count(fixture.sessions, models.Memory) == 4


@pytest.mark.skipif(os.getuid() == 0, reason="root can read anything")
async def test_an_unreadable_file_is_skipped_without_aborting_the_sync(
    fixture: Fixture,
) -> None:
    locked = fixture.root / "locked.md"
    locked.write_text("secret\n")
    os.chmod(locked, 0o000)
    try:
        report = await fixture.run(full=True)
    finally:
        os.chmod(locked, 0o644)

    # One bad file must not end a sync of the other four.
    assert report.observed == 4
    assert report.errors == 0
    assert await count(fixture.sessions, models.Memory) == 4


async def test_the_cursor_and_sync_timestamps_are_updated(fixture: Fixture) -> None:
    await fixture.run(full=False)

    async with fixture.sessions() as session:
        row = await session.get(models.Source, fixture.source.id)
        assert row is not None
        assert set(row.cursor["seen"]) == {
            "readme.md",
            "script.py",
            "notes/alpha.md",
            "notes/beta.txt",
        }
        assert row.last_sync_at is not None
        # An incremental sync did not reconcile deletions, so it has no business
        # claiming it did.
        assert row.last_full_sync_at is None

    await fixture.run(full=True)

    async with fixture.sessions() as session:
        row = await session.get(models.Source, fixture.source.id)
        assert row is not None
        assert row.last_full_sync_at is not None


async def test_a_failed_transaction_takes_its_job_with_it(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory and job commit together or not at all.

    This is the whole reason the queue is a table rather than a broker. With a
    broker there is a window where one landed and the other did not, and no
    amount of care closes it.
    """
    from memoryos.adapters.db.job_queue import enqueue_in

    async def enqueue_then_fail(session: AsyncSession, spec: JobSpec) -> UUID | None:
        result = await enqueue_in(session, spec)
        raise RuntimeError("crash after enqueue, before commit")
        return result

    monkeypatch.setattr("memoryos.application.sync.enqueue_in", enqueue_then_fail)

    report = await fixture.run(full=True)

    # Every item failed, and every item failed cleanly.
    assert report.errors == 4
    assert report.new == 0
    assert await counts(fixture.sessions) == {
        "artifacts": 0,
        "events": 0,
        "memories": 0,
        "jobs": 0,
    }
