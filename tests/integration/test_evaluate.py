"""The harness end to end, against a real index, twice.

The property under test is repeatability, not quality. A number that moves
between two runs over an unchanged corpus is worthless as a baseline — every
later Phase 2 milestone reports its diff against this file's output, and a
harness with its own noise floor would make small real improvements
indistinguishable from that noise.

So: ingest a fixture corpus, score a golden set that mixes memory-level and
chunk-level judgements, and require the two runs to agree on every number and
every ranking. The fake embedder makes the *scores* arbitrary; it does not make
them unstable, which is exactly the distinction being checked.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, select
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
from memoryos.application.evaluate import evaluate
from memoryos.application.golden import load_golden_set
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.search import SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import SourceKind
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

QUEUE = (
    "The worker claims a task from the queue and holds a lease while the handler "
    "runs. Renewing the lease is how a long task keeps its hold on the work. "
)
TIMES = (
    "A memory carries two timestamps. One says when the thing happened at the "
    "source, the other says when this system first saw it. "
)
BREAD = (
    "A wild yeast starter is fed flour and water until it doubles, then folded "
    "gently and given a long cold rest. "
)


@pytest.fixture
async def golden_path(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "queue.md").write_text("# Queue\n\n" + QUEUE * 6 + "\n")
    (root / "times.md").write_text("# Times\n\n" + TIMES * 6 + "\n")
    (root / "bread.md").write_text("# Bread\n\n" + BREAD * 6 + "\n")

    source = Source(
        id=new_id(), kind=SourceKind.FILESYSTEM, name="fixture", config={"root": str(root)}
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    embedder = FakeEmbedder()
    await SyncSource(sessions, FilesystemConnector(blobs), blobs)(source.id, full=True)
    await _drain(
        sessions,
        JobType.NORMALIZE_MEMORY,
        NormalizeMemory(sessions, blobs, build_parsers(), StructuralChunker(embedder)),
    )
    await _drain(
        sessions,
        JobType.EMBED_MEMORY,
        EmbedMemory(sessions, embedder, PostgresEmbeddingCache(sessions)),
    )

    path = tmp_path / "golden-set.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-08-10T00:00:00+00:00",
                "queries": [
                    {
                        "query_text": "how does the worker hold a lease",
                        "filters": {},
                        "items": [
                            _item("queue.md", "relevant"),
                            _item("bread.md", "not_relevant"),
                        ],
                    },
                    {
                        # Pinned to a chunk, so the harness has to project a hit
                        # through `GoldenQuery.project` rather than treating the
                        # file as the unit.
                        "query_text": "why are there two timestamps",
                        "filters": {},
                        "items": [
                            _item("times.md", "relevant", ordinal=0),
                            _item("queue.md", "missing"),
                        ],
                    },
                    {
                        # Excluded before any search happens, and its absence is
                        # part of what the two runs must agree on.
                        "query_text": "nothing here answers this",
                        "filters": {},
                        "items": [_item("bread.md", "not_relevant")],
                    },
                ],
            }
        )
    )
    return path


async def test_evaluate_is_repeatable_over_an_unchanged_corpus(
    golden_path: Path, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    search = SearchMemories(
        sessions, FakeEmbedder(), PgVectorStore(sessions, FakeEmbedder(), default_ef_search=100)
    )

    first, second = [
        (
            await evaluate(
                await load_golden_set(golden_path, sessions),
                search,
                sessions,
                k=3,
                # Fixed, so `ran_at` cannot be the reason two dicts differ — the
                # comparison below is meant to catch retrieval drift, not clocks.
                now=datetime(2026, 8, 10, tzinfo=UTC),
            )
        ).as_dict()
        for _ in range(2)
    ]

    assert first == second

    # Not vacuous: something was scored, something was retrieved, and the
    # unscoreable query left by the front door.
    assert first["queries"] == 2
    assert all(result["retrieved"] for result in first["results"])
    assert [entry["query_text"] for entry in first["excluded"]] == [
        "nothing here answers this"
    ]
    assert first["unresolved"] == []

    # The chunk-pinned query scores against a chunk key, which is the whole
    # reason `chunk_ordinal` exists — and `missing` sits in the answer key
    # beside it at memory granularity.
    pinned = next(
        result
        for result in first["results"]
        if result["query_text"] == "why are there two timestamps"
    )
    assert pinned["relevant"] == ["fixture::queue.md", "fixture::times.md#0"]


def _item(external_key: str, verdict: str, ordinal: int | None = None) -> dict[str, object]:
    return {
        "source_name": "fixture",
        "external_key": external_key,
        "chunk_ordinal": ordinal,
        "verdict": verdict,
    }


async def _drain(
    sessions: async_sessionmaker[AsyncSession], job_type: JobType, handler: object
) -> None:
    async with sessions() as session:
        targets = [
            UUID(row[0]["memory_id"])
            for row in await session.execute(
                select(models.Job.payload).where(models.Job.job_type == job_type.value)
            )
        ]
    for memory_id in targets:
        await handler(memory_id)  # type: ignore[operator]
    async with sessions.begin() as session:
        await session.execute(delete(models.Job).where(models.Job.job_type == job_type.value))
