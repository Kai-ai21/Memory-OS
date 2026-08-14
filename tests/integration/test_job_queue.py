"""The queue against a real Postgres.

These drive a real session factory rather than a wrapped one: the queue owns
its own transactions, and a claim that never commits proves nothing about
whether two workers can take the same row. Isolation comes from the shared
`clean_database` fixture, which truncates before every test.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import PostgresJobQueue, enqueue_in
from memoryos.domain.jobs import JobSpec, JobStatus, JobType

pytestmark = pytest.mark.integration

LEASE = timedelta(seconds=30)


@pytest.fixture
def queue(sessions: async_sessionmaker[AsyncSession]) -> PostgresJobQueue:
    return PostgresJobQueue(sessions)


async def job_row(sessions: async_sessionmaker[AsyncSession], job_id: object) -> models.Job:
    async with sessions() as session:
        row = await session.get(models.Job, job_id)
        assert row is not None
        return row


async def status_counts(sessions: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    async with sessions() as session:
        result = await session.execute(
            select(models.Job.status, text("count(*)")).group_by(models.Job.status)
        )
        return {status: count for status, count in result}


# --------------------------------------------------------------------------
# Enqueue and dedupe
# --------------------------------------------------------------------------


async def test_enqueue_returns_the_new_job_id(queue: PostgresJobQueue) -> None:
    job_id = await queue.enqueue(JobSpec(job_type=JobType.NOOP, payload={"n": 1}))
    assert job_id is not None


async def test_enqueue_without_a_dedupe_key_never_collapses(queue: PostgresJobQueue) -> None:
    first = await queue.enqueue(JobSpec(job_type=JobType.NOOP))
    second = await queue.enqueue(JobSpec(job_type=JobType.NOOP))
    assert first is not None
    assert second is not None
    assert first != second


async def test_the_same_dedupe_key_collapses_while_pending(queue: PostgresJobQueue) -> None:
    spec = JobSpec(job_type=JobType.NOOP, dedupe_key="notes/a.md")

    assert await queue.enqueue(spec) is not None
    assert await queue.enqueue(spec) is None


async def test_the_same_dedupe_key_collapses_while_running(queue: PostgresJobQueue) -> None:
    spec = JobSpec(job_type=JobType.NOOP, dedupe_key="notes/a.md")
    await queue.enqueue(spec)
    claimed = await queue.claim("worker-a", LEASE)
    assert claimed is not None

    assert await queue.enqueue(spec) is None


async def test_a_dedupe_key_is_reusable_once_the_job_finishes(
    queue: PostgresJobQueue,
) -> None:
    # The index is partial over pending/running for exactly this reason: the
    # same file changing twice must be enqueueable twice.
    spec = JobSpec(job_type=JobType.NOOP, dedupe_key="notes/a.md")
    await queue.enqueue(spec)
    claimed = await queue.claim("worker-a", LEASE)
    assert claimed is not None
    assert await queue.complete(claimed.id, "worker-a") is True

    assert await queue.enqueue(spec) is not None


async def test_the_same_key_under_a_different_job_type_is_a_different_job(
    queue: PostgresJobQueue,
) -> None:
    assert await queue.enqueue(JobSpec(JobType.NOOP, dedupe_key="k")) is not None
    assert await queue.enqueue(JobSpec(JobType.FAIL_TRANSIENT, dedupe_key="k")) is not None


async def test_enqueue_in_joins_the_callers_transaction(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    # The reason the queue lives in the database: a use case can write its data
    # and enqueue the job that processes it atomically. If the transaction rolls
    # back, the job is not left behind pointing at data that never existed.
    async with sessions() as session:
        await enqueue_in(session, JobSpec(job_type=JobType.NOOP, dedupe_key="rolled-back"))
        await session.rollback()

    assert await status_counts(sessions) == {}


# --------------------------------------------------------------------------
# Claim ordering and readiness
# --------------------------------------------------------------------------


async def test_higher_priority_is_claimed_first(queue: PostgresJobQueue) -> None:
    await queue.enqueue(JobSpec(JobType.NOOP, payload={"tag": "low"}, priority=0))
    await queue.enqueue(JobSpec(JobType.NOOP, payload={"tag": "high"}, priority=10))
    await queue.enqueue(JobSpec(JobType.NOOP, payload={"tag": "mid"}, priority=5))

    order = []
    while (job := await queue.claim("worker-a", LEASE)) is not None:
        order.append(job.payload["tag"])

    assert order == ["high", "mid", "low"]


async def test_equal_priority_is_claimed_oldest_first(queue: PostgresJobQueue) -> None:
    now = datetime.now(UTC)
    for tag, offset in [("third", -10), ("first", -300), ("second", -60)]:
        await queue.enqueue(
            JobSpec(
                JobType.NOOP,
                payload={"tag": tag},
                run_after=now + timedelta(seconds=offset),
            )
        )

    order = []
    while (job := await queue.claim("worker-a", LEASE)) is not None:
        order.append(job.payload["tag"])

    assert order == ["first", "second", "third"]


async def test_a_future_run_after_is_not_claimed_until_it_passes(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    job_id = await queue.enqueue(
        JobSpec(JobType.NOOP, run_after=datetime.now(UTC) + timedelta(hours=1))
    )
    assert job_id is not None

    assert await queue.claim("worker-a", LEASE) is None

    async with sessions.begin() as session:
        await session.execute(
            update(models.Job)
            .where(models.Job.id == job_id)
            .values(run_after=datetime.now(UTC) - timedelta(seconds=1))
        )

    claimed = await queue.claim("worker-a", LEASE)
    assert claimed is not None
    assert claimed.id == job_id


async def test_claim_returns_none_on_an_empty_queue(queue: PostgresJobQueue) -> None:
    assert await queue.claim("worker-a", LEASE) is None


async def test_only_restricts_the_claim_and_leaves_the_rest_pending(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """**A filtered drain must not consume the jobs it declines.**

    The queue is shared across phases and the checks are not: `make
    phase1-check` drains it to prove ingestion works, and embedding enqueues an
    entity extraction per memory, which is a live model call. Filtering happens
    in the candidate select rather than after the claim, so a declined job keeps
    its `pending` status *and* its attempt budget — a filter applied afterwards
    would take each job, release it, and burn an attempt on work nothing was
    wrong with, dead-lettering the queue in a few passes.
    """
    await queue.enqueue(JobSpec(JobType.EXTRACT_ENTITIES, payload={"tag": "llm"}))
    await queue.enqueue(JobSpec(JobType.EMBED_MEMORY, payload={"tag": "local"}))

    wanted = frozenset({JobType.EMBED_MEMORY})
    claimed = []
    while (job := await queue.claim("worker-a", LEASE, only=wanted)) is not None:
        claimed.append(job.payload["tag"])

    assert claimed == ["local"]

    # The excluded job is untouched: still pending, still on zero attempts, so a
    # later unrestricted drain runs it with its full budget.
    async with sessions() as session:
        row = (
            await session.execute(
                select(models.Job).where(
                    models.Job.job_type == JobType.EXTRACT_ENTITIES.value
                )
            )
        ).scalar_one()
        assert row.status == JobStatus.PENDING.value
        assert row.attempts == 0

    # And an unrestricted claim still finds it.
    later = await queue.claim("worker-b", LEASE)
    assert later is not None
    assert later.payload["tag"] == "llm"


async def test_only_with_no_matching_type_claims_nothing(
    queue: PostgresJobQueue,
) -> None:
    """The other half, so the filter is a filter rather than a no-op that
    happens to pass the test above."""
    await queue.enqueue(JobSpec(JobType.EXTRACT_ENTITIES, payload={"tag": "llm"}))

    assert await queue.claim("w", LEASE, only=frozenset({JobType.NOOP})) is None
    # An empty filter is not "claim nothing" — the worker passes None for that,
    # and conflating the two would make a typo in `--only` look like an empty
    # queue.
    assert await queue.claim("w", LEASE) is not None


# --------------------------------------------------------------------------
# Attempts, leases, fencing
# --------------------------------------------------------------------------


async def test_attempts_increment_on_claim_not_on_failure(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # A worker that segfaults never reaches its failure handler. Counting the
    # attempt at claim time is what stops such a job retrying forever.
    job_id = await queue.enqueue(JobSpec(JobType.NOOP))
    assert job_id is not None

    claimed = await queue.claim("worker-a", LEASE)
    assert claimed is not None
    assert claimed.attempts == 1
    assert (await job_row(sessions, job_id)).attempts == 1

    # Nothing completes it; the lease simply expires and it comes back.
    await expire_lease(sessions, job_id)
    assert await queue.reclaim_expired() == 1

    reclaimed = await queue.claim("worker-a", LEASE)
    assert reclaimed is not None
    assert reclaimed.attempts == 2


async def test_a_claimed_job_carries_a_worker_and_a_lease(queue: PostgresJobQueue) -> None:
    await queue.enqueue(JobSpec(JobType.NOOP))
    claimed = await queue.claim("worker-a", LEASE)

    assert claimed is not None
    assert claimed.status is JobStatus.RUNNING
    assert claimed.locked_by == "worker-a"
    assert claimed.lease_expires_at is not None


async def test_heartbeat_extends_the_lease(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await queue.enqueue(JobSpec(JobType.NOOP))
    claimed = await queue.claim("worker-a", timedelta(seconds=5))
    assert claimed is not None
    before = (await job_row(sessions, claimed.id)).lease_expires_at
    assert before is not None

    assert await queue.heartbeat(claimed.id, "worker-a", timedelta(seconds=600)) is True

    after = (await job_row(sessions, claimed.id)).lease_expires_at
    assert after is not None
    assert after > before


async def test_a_fenced_out_worker_cannot_complete_a_job(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # Worker A claims. Worker B takes the job over — what reclaim_expired plus a
    # second claim amounts to. A's late completion must not land.
    await queue.enqueue(JobSpec(JobType.NOOP))
    claimed = await queue.claim("worker-a", LEASE)
    assert claimed is not None

    async with sessions.begin() as session:
        await session.execute(
            update(models.Job)
            .where(models.Job.id == claimed.id)
            .values(locked_by="worker-b")
        )

    assert await queue.complete(claimed.id, "worker-a") is False
    assert await queue.heartbeat(claimed.id, "worker-a", LEASE) is False
    assert await queue.dead_letter(claimed.id, "worker-a", "e", "tb") is False
    assert (
        await queue.reschedule(claimed.id, "worker-a", datetime.now(UTC), "e", "tb") is False
    )

    # And the row is untouched: still running, still owned by B.
    row = await job_row(sessions, claimed.id)
    assert row.status == JobStatus.RUNNING.value
    assert row.locked_by == "worker-b"
    assert row.completed_at is None
    assert row.last_error is None

    assert await queue.complete(claimed.id, "worker-b") is True


async def test_a_lease_that_expires_returns_the_job_to_pending(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # Without the sweeper, a worker killed mid-job leaves this row running
    # forever and nothing ever picks the work back up.
    await queue.enqueue(JobSpec(JobType.NOOP, max_attempts=5))
    claimed = await queue.claim("worker-a", LEASE)
    assert claimed is not None
    await expire_lease(sessions, claimed.id)

    assert await queue.reclaim_expired() == 1

    row = await job_row(sessions, claimed.id)
    assert row.status == JobStatus.PENDING.value
    assert row.locked_by is None
    assert row.lease_expires_at is None
    assert row.last_error == "lease expired"


async def test_an_expired_lease_with_no_attempts_left_fails_instead(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await queue.enqueue(JobSpec(JobType.NOOP, max_attempts=1))
    claimed = await queue.claim("worker-a", LEASE)
    assert claimed is not None
    assert claimed.attempts == 1
    await expire_lease(sessions, claimed.id)

    assert await queue.reclaim_expired() == 1

    row = await job_row(sessions, claimed.id)
    assert row.status == JobStatus.FAILED.value
    assert row.completed_at is not None


async def test_reclaim_leaves_live_leases_alone(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await queue.enqueue(JobSpec(JobType.NOOP))
    claimed = await queue.claim("worker-a", timedelta(minutes=10))
    assert claimed is not None

    assert await queue.reclaim_expired() == 0
    assert (await job_row(sessions, claimed.id)).status == JobStatus.RUNNING.value


async def test_reclaim_respects_its_limit(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    for _ in range(5):
        await queue.enqueue(JobSpec(JobType.NOOP))
    for _ in range(5):
        claimed = await queue.claim("worker-a", LEASE)
        assert claimed is not None
        await expire_lease(sessions, claimed.id)

    assert await queue.reclaim_expired(limit=2) == 2
    assert await queue.reclaim_expired(limit=100) == 3


async def test_terminal_transitions_record_their_error(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await queue.enqueue(JobSpec(JobType.NOOP))
    claimed = await queue.claim("worker-a", LEASE)
    assert claimed is not None

    assert await queue.dead_letter(claimed.id, "worker-a", "boom", "Traceback: ...") is True

    row = await job_row(sessions, claimed.id)
    assert row.status == JobStatus.FAILED.value
    assert row.last_error == "boom"
    assert row.last_traceback == "Traceback: ..."
    assert row.completed_at is not None
    assert row.locked_by is None


async def expire_lease(
    sessions: async_sessionmaker[AsyncSession], job_id: object
) -> None:
    """Push a lease into the past, using the database's clock.

    Deliberately not `datetime.now(UTC) - 1s` computed here: the sweeper
    compares against Postgres' `now()`, and on a machine whose clock has
    drifted from the container's, a host-computed timestamp is not reliably in
    the past.
    """
    async with sessions.begin() as session:
        await session.execute(
            update(models.Job)
            .where(models.Job.id == job_id)
            .values(lease_expires_at=func.now() - timedelta(seconds=1))
        )
