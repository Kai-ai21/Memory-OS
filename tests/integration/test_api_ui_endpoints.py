"""The HTTP surface M2.0a's frontend needs.

The UI is a client of these and nothing else, so what they promise is what it can
render. Two things get asserted more carefully than the rest: that chunks come
back in ordinal order — the UI widens a hit into its neighbours by ordinal, so an
arbitrary order would show the wrong surrounding text — and that `/stats` and
`/doctor` agree with the CLI, because a dashboard that reassures you while
`memoryos doctor` disagrees is worse than no dashboard.
"""

from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.api.app import WildcardOrigin, create_app
from memoryos.application.backfill import gather_stats
from memoryos.config import Settings
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------
# GET /memories/{id}
# --------------------------------------------------------------------------


async def test_a_memory_comes_back_with_its_chunks_in_ordinal_order(
    client: AsyncClient, harness: Harness
) -> None:
    await harness.ingest()
    async with harness.sessions() as session:
        memory_id = (
            await session.execute(
                select(models.Memory.id).where(
                    models.Memory.external_key == "mod.py",
                    models.Memory.is_current.is_(True),
                )
            )
        ).scalar_one()

    response = await client.get(f"/memories/{memory_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["external_key"] == "mod.py"
    assert body["source_name"] == "corpus"
    assert body["content"], "the UI draws boundaries against this text"
    ordinals = [chunk["ordinal"] for chunk in body["chunks"]]
    # Ordinal order, and contiguous from zero: the UI reads neighbours by
    # ordinal, so a gap or a shuffle would silently show the wrong context.
    assert ordinals == sorted(ordinals)
    assert ordinals == list(range(len(ordinals)))


async def test_chunk_offsets_index_into_the_memory_content(
    client: AsyncClient, harness: Harness
) -> None:
    """The property the whole highlight feature rests on.

    `char_start`/`char_end` are offsets into the parent's normalized text. If they
    did not line up, the UI would highlight the wrong span and look entirely
    plausible doing it.
    """
    await harness.ingest()
    async with harness.sessions() as session:
        memory_id = (
            await session.execute(
                select(models.Memory.id).where(
                    models.Memory.external_key == "queue.md",
                    models.Memory.is_current.is_(True),
                )
            )
        ).scalar_one()

    body = (await client.get(f"/memories/{memory_id}")).json()
    content = body["content"]

    for chunk in body["chunks"]:
        assert 0 <= chunk["char_start"] < chunk["char_end"] <= len(content)
        # The chunk carries an overlap prefix from its neighbour, so its text is
        # not always the slice — but the slice must be real text, not past the end.
        assert content[chunk["char_start"] : chunk["char_end"]].strip()


async def test_a_memory_carries_its_version_history(
    client: AsyncClient, harness: Harness
) -> None:
    from tests.integration.conftest import QUEUE_TEXT

    await harness.ingest()
    (harness.root / "queue.md").write_text("# Queue v2\n\n" + QUEUE_TEXT * 6 + "\n")
    await harness.ingest()

    async with harness.sessions() as session:
        memory_id = (
            await session.execute(
                select(models.Memory.id).where(
                    models.Memory.external_key == "queue.md",
                    models.Memory.is_current.is_(True),
                )
            )
        ).scalar_one()

    body = (await client.get(f"/memories/{memory_id}")).json()

    versions = body["versions"]
    assert [row["version"] for row in versions] == [2, 1], "newest first"
    assert [row["is_current"] for row in versions] == [True, False]


async def test_an_unknown_memory_is_a_404(client: AsyncClient) -> None:
    response = await client.get(f"/memories/{uuid4()}")
    assert response.status_code == 404


# --------------------------------------------------------------------------
# GET /sources, /stats, /doctor
# --------------------------------------------------------------------------


async def test_sources_report_their_own_counts(
    client: AsyncClient, harness: Harness
) -> None:
    await harness.ingest()

    (source,) = (await client.get("/sources")).json()

    assert source["name"] == "corpus"
    assert source["memories"] == 5
    assert source["chunks"] > 0


async def test_a_source_with_nothing_ingested_reports_zero(
    client: AsyncClient, tmp_path: Path
) -> None:
    # The interesting value. An outer join is what makes it zero rather than
    # omitting the row entirely.
    empty = tmp_path / "empty"
    empty.mkdir()
    await client.post("/sources", json={"name": "empty", "root": str(empty)})

    (source,) = (await client.get("/sources")).json()
    assert (source["memories"], source["chunks"]) == (0, 0)


async def test_stats_matches_what_the_cli_reports(
    client: AsyncClient, harness: Harness
) -> None:
    """Same numbers, same source. Two implementations would eventually disagree."""
    await harness.ingest()

    body = (await client.get("/stats")).json()
    expected = await gather_stats(harness.sessions)

    assert body["memories"] == expected.memories
    assert body["chunks"] == expected.chunks
    assert body["embedded_chunks"] == expected.embedded_chunks
    assert body["cache_entries"] == expected.cache_entries
    assert body["coverage"] == pytest.approx(expected.coverage)
    # And what produced them, so a number on screen can be attributed.
    assert body["chunker_version"]
    assert body["embedding_model"]


async def test_doctor_reports_its_findings(client: AsyncClient, harness: Harness) -> None:
    await harness.ingest()

    body = (await client.get("/doctor")).json()

    checks = {finding["check"] for finding in body["findings"]}
    assert "chunks_over_model_window" in checks
    assert "chunks_without_embeddings" in checks
    assert isinstance(body["healthy"], bool)


async def test_doctor_reports_unhealthy_when_something_is_wrong(
    client: AsyncClient, harness: Harness
) -> None:
    """The check has to be able to fail, or rendering it means nothing."""
    from sqlalchemy import update

    await harness.ingest()
    async with harness.sessions.begin() as session:
        await session.execute(
            update(models.MemoryChunk).values(
                embedding=None, embedding_model=None, embedded_at=None
            )
        )

    body = (await client.get("/doctor")).json()

    assert body["healthy"] is False
    unembedded = next(
        f for f in body["findings"] if f["check"] == "chunks_without_embeddings"
    )
    assert unembedded["count"] > 0


# --------------------------------------------------------------------------
# Judgements over HTTP
# --------------------------------------------------------------------------


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "query_text": "how does the job queue claim work",
        "source_name": "corpus",
        "external_key": "queue.md",
        "verdict": "relevant",
        "rank_at_judgement": 1,
        "score_at_judgement": 0.81,
        "filters": {"k": 10},
    }
    body.update(overrides)
    return body


async def test_posting_a_judgement_and_listing_it(
    client: AsyncClient, sessions: async_sessionmaker[AsyncSession]
) -> None:
    created = await client.post("/judgements", json=payload())
    assert created.status_code == 201
    assert created.json()["id"]

    (summary,) = (await client.get("/judgements")).json()
    assert summary["query_text"] == "how does the job queue claim work"
    assert summary["relevant"] == 1
    assert summary["total"] == 1


async def test_rejudging_over_http_replaces(client: AsyncClient) -> None:
    first = await client.post("/judgements", json=payload(verdict="relevant"))
    second = await client.post("/judgements", json=payload(verdict="not_relevant"))

    assert first.json()["id"] == second.json()["id"]
    (summary,) = (await client.get("/judgements")).json()
    assert (summary["relevant"], summary["not_relevant"]) == (0, 1)


async def test_a_missing_verdict_with_a_rank_is_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/judgements", json=payload(verdict="missing", rank_at_judgement=3)
    )
    assert response.status_code == 422
    assert "rank" in response.text


async def test_an_unknown_verdict_is_rejected(client: AsyncClient) -> None:
    response = await client.post("/judgements", json=payload(verdict="maybe"))
    assert response.status_code == 422


async def test_the_export_endpoint_returns_the_golden_set(client: AsyncClient) -> None:
    await client.post("/judgements", json=payload(external_key="a.md"))
    await client.post(
        "/judgements", json=payload(external_key="b.md", verdict="not_relevant")
    )

    body = (await client.get("/judgements/export")).json()

    assert body["totals"]["queries"] == 1
    assert body["totals"]["judgements"] == 2
    (query,) = body["queries"]
    assert query["relevant_keys"] == ["a.md"]
    assert body["generated_at"]


# --------------------------------------------------------------------------
# CORS
# --------------------------------------------------------------------------


async def build_client(settings: Settings) -> AsyncClient:
    app = create_app(settings)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_no_cors_headers_when_no_origins_are_configured(
    clean_database: None,
) -> None:
    """The default. No middleware at all, so there is no surface to reason about."""
    settings = Settings(cors_origins=[])
    async with await build_client(settings) as client:
        response = await client.get(
            "/health/live", headers={"Origin": "http://localhost:5173"}
        )
    assert "access-control-allow-origin" not in response.headers


async def test_a_configured_origin_is_allowed(clean_database: None) -> None:
    settings = Settings(cors_origins=["http://localhost:5173"])
    async with await build_client(settings) as client:
        response = await client.get(
            "/health/live", headers={"Origin": "http://localhost:5173"}
        )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


async def test_an_unconfigured_origin_is_not_allowed(clean_database: None) -> None:
    settings = Settings(cors_origins=["http://localhost:5173"])
    async with await build_client(settings) as client:
        response = await client.get(
            "/health/live", headers={"Origin": "http://evil.example"}
        )
    assert "access-control-allow-origin" not in response.headers


def test_a_wildcard_origin_refuses_to_start() -> None:
    """Loud at startup rather than documented against.

    This API answers questions about a private corpus, so `*` means any page the
    operator happens to visit can read it.
    """
    with pytest.raises(WildcardOrigin, match=r"private corpus"):
        create_app(Settings(cors_origins=["*"]))
