"""The worker end to end, against a real queue."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import PostgresJobQueue
from memoryos.application.jobs.handlers import build_default_registry
from memoryos.application.jobs.registry import HandlerRegistry, JobContext
from memoryos.application.worker import Worker, WorkerConfig
from memoryos.domain.jobs import JobSpec, JobStatus, JobType

pytestmark = pytest.mark.integration

FAST = WorkerConfig(
    lease=timedelta(seconds=30),
    poll_min_seconds=0.01,
    poll_max_seconds=0.05,
    idle_polls_before_drain_stop=2,
)


@pytest.fixture
def queue(sessions: async_sessionmaker[AsyncSession]) -> PostgresJobQueue:
    return PostgresJobQueue(sessions)


def make_worker(
    queue: PostgresJobQueue,
    sessions: async_sessionmaker[AsyncSession],
    worker_id: str,
    registry: HandlerRegistry | None = None,
    config: WorkerConfig = FAST,
) -> Worker:
    return Worker(
        queue=queue,
        registry=registry or build_default_registry(),
        session_factory=sessions,
        config=config,
        worker_id=worker_id,
    )


async def all_jobs(sessions: async_sessionmaker[AsyncSession]) -> list[models.Job]:
    async with sessions() as session:
        result = await session.execute(select(models.Job).order_by(models.Job.created_at))
        return list(result.scalars())


async def status_counts(sessions: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    async with sessions() as session:
        result = await session.execute(
            select(models.Job.status, func.count()).group_by(models.Job.status)
        )
        return {status: count for status, count in result.all()}


async def test_five_workers_drain_fifty_jobs_without_claiming_any_twice(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The test that proves SKIP LOCKED.

    Without it every worker's claim selects the same top row, four of the five
    block on that row's lock, and the queue drains serially however many workers
    are running. The evidence that it did not happen is `attempts`: a job
    claimed twice would have been counted twice.
    """
    for n in range(50):
        assert await queue.enqueue(JobSpec(JobType.NOOP, payload={"n": n})) is not None

    workers = [make_worker(queue, sessions, f"worker-{i}") for i in range(5)]
    await asyncio.gather(*(w.run(drain=True, handle_signals=False) for w in workers))

    jobs = await all_jobs(sessions)
    assert len(jobs) == 50
    assert {job.status for job in jobs} == {JobStatus.SUCCEEDED.value}

    # attempts increments once per claim, so anything above 1 means two workers
    # got the same row.
    assert {job.attempts for job in jobs} == {1}

    # Every payload arrived exactly once — no job silently dropped either.
    assert sorted(job.payload["n"] for job in jobs) == list(range(50))


async def test_concurrent_workers_share_the_work(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Not just correct, but actually parallel.

    Each handler sleeps briefly, so a serialised queue would leave one worker
    holding every job. Asserting that more than one worker's name appears is
    what distinguishes SKIP LOCKED from blocking.
    """
    claimed_by: dict[str, int] = {}
    registry = HandlerRegistry()

    async def record(ctx: JobContext) -> None:
        worker = ctx.job.locked_by or "?"
        claimed_by[worker] = claimed_by.get(worker, 0) + 1
        await asyncio.sleep(0.02)

    registry.register(JobType.NOOP, record)

    for _ in range(30):
        await queue.enqueue(JobSpec(JobType.NOOP))

    workers = [make_worker(queue, sessions, f"worker-{i}", registry) for i in range(5)]
    await asyncio.gather(*(w.run(drain=True, handle_signals=False) for w in workers))

    assert sum(claimed_by.values()) == 30
    assert len(claimed_by) > 1, f"only one worker ever claimed: {claimed_by}"


async def test_a_transient_failure_retries_with_growing_delay_then_fails(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    job_id = await queue.enqueue(JobSpec(JobType.FAIL_TRANSIENT, max_attempts=3))
    assert job_id is not None
    worker = make_worker(queue, sessions, "worker-a")

    # compute_backoff(n) is base 2s doubled per attempt, jittered into
    # [0.5x, 1.0x]: attempt 1 lands in [2s, 4s], attempt 2 in [4s, 8s]. The
    # windows are asserted rather than the raw values, so the test does not
    # depend on the random draw — and the fact that they do not overlap is
    # itself the evidence that the delay grows.
    for expected_attempts, low, high in [(1, 2.0, 4.0), (2, 4.0, 8.0)]:
        job = await queue.claim(worker.worker_id, FAST.lease)
        assert job is not None
        assert job.attempts == expected_attempts

        before = datetime.now(UTC)
        await worker._run_job(job)

        async with sessions() as session:
            row = await session.get(models.Job, job_id)
            assert row is not None
            assert row.status == JobStatus.PENDING.value
            assert row.last_error is not None
            assert "transient failure" in row.last_error
            assert row.last_traceback is not None
            assert "TransientError" in row.last_traceback

            delay = (row.run_after - before).total_seconds()
            assert low <= delay <= high + 1.0, f"attempt {expected_attempts}: {delay}s"

        # The retry really is deferred: nothing is claimable until it is due.
        assert await queue.claim(worker.worker_id, FAST.lease) is None
        await make_ready(sessions, job_id)

    # Third attempt exhausts the budget.
    job = await queue.claim(worker.worker_id, FAST.lease)
    assert job is not None
    assert job.attempts == 3
    await worker._run_job(job)

    async with sessions() as session:
        row = await session.get(models.Job, job_id)
        assert row is not None
        assert row.status == JobStatus.FAILED.value
        assert row.attempts == 3
        assert row.completed_at is not None
        assert row.last_error is not None
        assert row.last_traceback is not None


async def test_a_permanent_failure_fails_on_the_first_attempt(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    job_id = await queue.enqueue(JobSpec(JobType.FAIL_PERMANENT, max_attempts=5))
    assert job_id is not None

    await make_worker(queue, sessions, "worker-a").run(drain=True, handle_signals=False)

    async with sessions() as session:
        row = await session.get(models.Job, job_id)
        assert row is not None
        # Four attempts still available and it is done anyway: no amount of
        # retrying turns a permanent failure into a success.
        assert row.status == JobStatus.FAILED.value
        assert row.attempts == 1
        assert row.last_error is not None
        assert row.last_traceback is not None


async def test_an_unregistered_job_type_fails_permanently(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    job_id = await queue.enqueue(JobSpec(JobType.NOOP, max_attempts=5))
    assert job_id is not None

    empty = HandlerRegistry()
    await make_worker(queue, sessions, "worker-a", empty).run(
        drain=True, handle_signals=False
    )

    async with sessions() as session:
        row = await session.get(models.Job, job_id)
        assert row is not None
        assert row.status == JobStatus.FAILED.value
        assert row.attempts == 1
        assert row.last_error is not None
        assert "no handler registered" in row.last_error


async def test_the_worker_reclaims_a_dead_workers_job_and_finishes_it(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    # A worker claims and then dies. Nothing else would ever move this row.
    job_id = await queue.enqueue(JobSpec(JobType.NOOP, max_attempts=5))
    assert job_id is not None
    abandoned = await queue.claim("worker-that-died", FAST.lease)
    assert abandoned is not None

    from tests.integration.test_job_queue import expire_lease

    await expire_lease(sessions, abandoned.id)

    await make_worker(queue, sessions, "worker-b").run(drain=True, handle_signals=False)

    async with sessions() as session:
        row = await session.get(models.Job, job_id)
        assert row is not None
        assert row.status == JobStatus.SUCCEEDED.value
        # One claim from the dead worker, one from the live one.
        assert row.attempts == 2


async def test_a_worker_that_lost_its_lease_writes_no_terminal_state(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The fencing path, end to end.

    Worker A is mid-handler when worker B takes the job over. A must cancel and
    write nothing: a `succeeded` from A would erase the fact that B is still
    working on it.
    """
    started = asyncio.Event()
    cancelled = asyncio.Event()
    registry = HandlerRegistry()

    async def slow(ctx: JobContext) -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    registry.register(JobType.NOOP, slow)

    job_id = await queue.enqueue(JobSpec(JobType.NOOP))
    assert job_id is not None

    # A very short lease so the heartbeat runs almost immediately.
    config = WorkerConfig(lease=timedelta(seconds=0.3), poll_min_seconds=0.01)
    worker_a = make_worker(queue, sessions, "worker-a", registry, config)

    job = await queue.claim("worker-a", config.lease)
    assert job is not None
    running = asyncio.create_task(worker_a._run_job(job))
    await asyncio.wait_for(started.wait(), timeout=5)

    # Worker B takes it over; A's next heartbeat finds it no longer owns the job.
    async with sessions.begin() as session:
        await session.execute(
            update(models.Job)
            .where(models.Job.id == job_id)
            .values(locked_by="worker-b")
        )

    await asyncio.wait_for(running, timeout=10)
    assert cancelled.is_set()

    async with sessions() as session:
        row = await session.get(models.Job, job_id)
        assert row is not None
        # Untouched by A: still running, still B's.
        assert row.status == JobStatus.RUNNING.value
        assert row.locked_by == "worker-b"
        assert row.completed_at is None
        assert row.last_error is None


async def test_a_drain_run_on_an_empty_queue_exits(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    await asyncio.wait_for(
        make_worker(queue, sessions, "worker-a").run(drain=True, handle_signals=False),
        timeout=5,
    )
    assert await status_counts(sessions) == {}


async def test_request_stop_ends_the_loop(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    worker = make_worker(queue, sessions, "worker-a")
    running = asyncio.create_task(worker.run(handle_signals=False))

    await asyncio.sleep(0.1)
    worker.request_stop()

    # The idle sleep wakes on the stop event rather than running its timer out.
    await asyncio.wait_for(running, timeout=5)


async def test_priority_is_honoured_by_a_running_worker(
    queue: PostgresJobQueue, sessions: async_sessionmaker[AsyncSession]
) -> None:
    order: list[int] = []
    registry = HandlerRegistry()

    async def record(ctx: JobContext) -> None:
        order.append(ctx.job.priority)

    registry.register(JobType.NOOP, record)

    for priority in (0, 9, 3):
        await queue.enqueue(JobSpec(JobType.NOOP, priority=priority))

    await make_worker(queue, sessions, "worker-a", registry).run(
        drain=True, handle_signals=False
    )

    assert order == [9, 3, 0]


async def make_ready(sessions: async_sessionmaker[AsyncSession], job_id: object) -> None:
    """Pull a deferred retry forward, using the database's clock."""
    async with sessions.begin() as session:
        await session.execute(
            update(models.Job)
            .where(models.Job.id == job_id)
            .values(run_after=func.now() - timedelta(seconds=1))
        )
