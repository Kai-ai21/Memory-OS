"""`GET /context`: serve the cache, enqueue on a miss, never assemble.

The endpoint the editor panel reads, and the one property that matters is what
it does *not* do. M6.1 measured assembly at 0.9-1.8s warm and 18-21s cold,
because a cold process loads an embedder and a cross-encoder. The caller is a
sidebar redrawn on every tab change, so an endpoint that assembled would be an
endpoint somebody turns off.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select

from memoryos.adapters.db import models
from memoryos.application.context_engine import (
    ContextRequest,
    cache_key_for,
    corpus_fingerprint,
)
from memoryos.domain.jobs import JobType
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration


async def jobs_of(harness: Harness, job_type: JobType) -> list[dict[str, Any]]:
    async with harness.sessions() as session:
        return [
            dict(row.payload)
            for row in (
                await session.execute(
                    select(models.Job).where(models.Job.job_type == job_type.value)
                )
            ).scalars()
        ]


async def test_a_miss_enqueues_and_answers_202(
    client: AsyncClient, harness: Harness
) -> None:
    """Cached-or-loading, and this is the loading half.

    `202` rather than `404`: nothing is wrong, the answer is being built. The
    client renders that as "Assembling context…" and polls, which is a different
    face from the one it shows for a failure — and collapsing the two would make
    a working system look broken for its first second.
    """
    response = await client.get("/context", params={"focus": "chunking"})

    assert response.status_code == 202
    body = response.json()
    assert body["ready"] is False
    assert body["items"] == []

    queued = await jobs_of(harness, JobType.ASSEMBLE_CONTEXT)
    assert len(queued) == 1
    assert queued[0]["focus"] == "chunking"


async def test_asking_twice_queues_one_build(
    client: AsyncClient, harness: Harness
) -> None:
    # A panel polling four times while the worker builds must not queue four
    # builds. Deduped on the cache key, which already encodes everything that
    # changes the answer.
    for _ in range(4):
        assert (await client.get("/context", params={"focus": "chunking"})).status_code == 202

    assert len(await jobs_of(harness, JobType.ASSEMBLE_CONTEXT)) == 1


async def test_different_budgets_are_different_questions(
    client: AsyncClient, harness: Harness
) -> None:
    # A 4,000-token context is not a truncation of a 1,000-token one: different
    # items were selected, not fewer. Two panels at two budgets are two builds.
    await client.get("/context", params={"focus": "chunking", "token_budget": 1000})
    await client.get("/context", params={"focus": "chunking", "token_budget": 4000})

    assert len(await jobs_of(harness, JobType.ASSEMBLE_CONTEXT)) == 2


async def test_a_cached_context_is_served_without_enqueueing(
    client: AsyncClient, harness: Harness
) -> None:
    """The hit path, which is the one with a latency requirement on it.

    Written into the cache directly rather than by running the engine, because
    what is being tested is that the *endpoint* reads it — a test that assembled
    first would pass on an endpoint that assembled again.
    """
    request = ContextRequest(focus="chunking")
    key = cache_key_for(request, await corpus_fingerprint(harness.sessions))
    async with harness.sessions.begin() as session:
        session.add(
            models.ContextCache(
                id=uuid4(),
                cache_key=key,
                focus="chunking",
                payload={
                    "focus": "chunking",
                    "token_budget": 4000,
                    "tokens_used": 42,
                    "items": [
                        {
                            "key": "memory:11111111-1111-7111-8111-111111111111",
                            "title": "self::src/a.py",
                            "category": "code",
                            "text": "def handler():",
                            "tokens": 42,
                            "position": 1,
                            "sources": {"retrieval": 3, "temporal": 1},
                            "relevance": 0.03,
                            "redundancy": 0.0,
                            "memory_id": "11111111-1111-7111-8111-111111111111",
                            "decision_id": None,
                            "external_key": "src/a.py",
                        }
                    ],
                },
                token_count=42,
                expires_at=datetime.now(UTC) + timedelta(hours=1),
            )
        )

    response = await client.get("/context", params={"focus": "chunking"})

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert len(body["items"]) == 1
    assert body["items"][0]["sources"] == {"retrieval": 3, "temporal": 1}
    assert body["items"][0]["memory_id"] == "11111111-1111-7111-8111-111111111111"
    # Nothing was queued: the answer was already there.
    assert await jobs_of(harness, JobType.ASSEMBLE_CONTEXT) == []
    # And the hit was counted, because that number is what decides whether any
    # of this precomputation earns its cost.
    async with harness.sessions() as session:
        assert (
            await session.execute(select(func.sum(models.ContextCache.hit_count)))
        ).scalar_one() == 1


async def test_an_empty_focus_is_refused_by_the_schema(client: AsyncClient) -> None:
    # A context assembled about "" is a context about the whole corpus: the
    # least useful possible answer and the most expensive to compute.
    assert (await client.get("/context", params={"focus": ""})).status_code == 422
    assert (await client.get("/context")).status_code == 422
