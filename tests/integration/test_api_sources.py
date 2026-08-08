"""The HTTP surface for sources, syncs, and memories."""

from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "readme.md").write_text("hello\n")
    return root


async def test_creating_and_listing_a_source(
    client: AsyncClient, corpus: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    created = await client.post(
        "/sources", json={"kind": "filesystem", "name": "corpus", "root": str(corpus)}
    )
    assert created.status_code == 201
    body = created.json()
    assert body["name"] == "corpus"
    assert body["config"]["root"] == str(corpus.resolve())
    assert body["last_sync_at"] is None

    listed = await client.get("/sources")
    assert listed.status_code == 200
    assert [row["name"] for row in listed.json()] == ["corpus"]


async def test_creating_a_duplicate_source_conflicts(
    client: AsyncClient, corpus: Path
) -> None:
    payload = {"kind": "filesystem", "name": "corpus", "root": str(corpus)}
    assert (await client.post("/sources", json=payload)).status_code == 201
    assert (await client.post("/sources", json=payload)).status_code == 409


async def test_triggering_a_sync_enqueues_rather_than_running_it(
    client: AsyncClient, corpus: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """202 and a job id, never the work itself.

    A sync of a large directory takes minutes. Doing it in the request would
    blow the HTTP timeout, and whatever it managed would have no retry, no
    progress, and no way to resume.
    """
    source_id = (
        await client.post(
            "/sources", json={"kind": "filesystem", "name": "corpus", "root": str(corpus)}
        )
    ).json()["id"]

    response = await client.post(f"/sources/{source_id}/sync", params={"full": True})

    assert response.status_code == 202
    assert response.json()["job_id"] is not None
    assert response.json()["full"] is True

    async with sessions() as session:
        jobs = list((await session.execute(select(models.Job))).scalars())
        # Enqueued, not run: no memory exists yet.
        memories = list((await session.execute(select(models.Memory))).scalars())

    assert [job.job_type for job in jobs] == ["sync_source"]
    assert jobs[0].payload == {"source_id": source_id, "full": True}
    assert memories == []


async def test_a_second_sync_request_does_not_queue_a_duplicate_walk(
    client: AsyncClient, corpus: Path
) -> None:
    source_id = (
        await client.post(
            "/sources", json={"kind": "filesystem", "name": "corpus", "root": str(corpus)}
        )
    ).json()["id"]

    first = await client.post(f"/sources/{source_id}/sync")
    second = await client.post(f"/sources/{source_id}/sync")

    assert first.json()["job_id"] is not None
    # Still 202 — the request was accepted — but the dedupe key collapsed it
    # into the walk that is already queued.
    assert second.status_code == 202
    assert second.json()["job_id"] is None


async def test_syncing_an_unknown_source_is_a_404(client: AsyncClient) -> None:
    response = await client.post(
        "/sources/00000000-0000-0000-0000-000000000000/sync"
    )
    assert response.status_code == 404


async def test_listing_memories_is_empty_before_any_sync(client: AsyncClient) -> None:
    response = await client.get("/memories")
    assert response.status_code == 200
    assert response.json() == []


async def test_listing_memories_filters_by_source_and_paginates(
    client: AsyncClient, corpus: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
    from memoryos.adapters.connectors.filesystem import FilesystemConnector
    from memoryos.application.sync import SyncSource

    (corpus / "second.md").write_text("second\n")
    (corpus / "third.md").write_text("third\n")

    source_id = (
        await client.post(
            "/sources", json={"kind": "filesystem", "name": "corpus", "root": str(corpus)}
        )
    ).json()["id"]

    from uuid import UUID

    blobs = FilesystemBlobStore(corpus.parent / "blobs")
    sync = SyncSource(sessions, FilesystemConnector(blobs), blobs)
    await sync(UUID(source_id), full=True)

    everything = await client.get("/memories", params={"source_id": source_id})
    assert everything.status_code == 200
    assert len(everything.json()) == 3
    assert all(row["is_current"] for row in everything.json())

    page = await client.get(
        "/memories", params={"source_id": source_id, "limit": 2, "offset": 0}
    )
    assert len(page.json()) == 2

    other_source = await client.get(
        "/memories", params={"source_id": "00000000-0000-0000-0000-000000000000"}
    )
    assert other_source.json() == []
