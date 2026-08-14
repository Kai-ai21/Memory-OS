"""The worker's decisions, exercised without a database.

The queue is faked; what is under test is which call the worker makes, not what
Postgres does with it.
"""

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID

import pytest
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.application.jobs.handlers import build_default_registry
from memoryos.application.jobs.registry import HandlerRegistry
from memoryos.application.worker import Worker, WorkerConfig, default_worker_id
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import (
    Job,
    JobSpec,
    JobStatus,
    JobType,
    TransientError,
)


class FakeQueue:
    """Records what the worker asked for."""

    def __init__(self) -> None:
        self.completed: list[UUID] = []
        self.rescheduled: list[tuple[UUID, datetime, str]] = []
        self.dead_lettered: list[tuple[UUID, str]] = []
        self.fence: bool = True

    async def enqueue(self, spec: JobSpec) -> UUID | None:
        return new_id()

    async def claim(
        self,
        worker_id: str,
        lease: timedelta,
        *,
        only: frozenset[JobType] | None = None,
    ) -> Job | None:
        return None

    async def heartbeat(self, job_id: UUID, worker_id: str, lease: timedelta) -> bool:
        return self.fence

    async def complete(self, job_id: UUID, worker_id: str) -> bool:
        self.completed.append(job_id)
        return self.fence

    async def reschedule(
        self, job_id: UUID, worker_id: str, run_after: datetime, error: str, tb: str
    ) -> bool:
        self.rescheduled.append((job_id, run_after, error))
        return self.fence

    async def dead_letter(self, job_id: UUID, worker_id: str, error: str, tb: str) -> bool:
        self.dead_lettered.append((job_id, error))
        return self.fence

    async def reclaim_expired(self, limit: int = 100) -> int:
        return 0


def make_job(attempts: int, max_attempts: int = 5, job_type: JobType = JobType.NOOP) -> Job:
    return Job(
        id=new_id(),
        job_type=job_type.value,
        payload={},
        status=JobStatus.RUNNING,
        attempts=attempts,
        max_attempts=max_attempts,
        run_after=datetime.now(UTC),
    )


class FakeSession:
    """Enough of AsyncSession for handlers that do not touch the database."""

    def __init__(self) -> None:
        self.committed = False

    async def commit(self) -> None:
        self.committed = True

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def make_worker(queue: FakeQueue, registry: HandlerRegistry | None = None) -> Worker:
    sessions = cast("async_sessionmaker[AsyncSession]", lambda: FakeSession())
    return Worker(
        queue=cast("Any", queue),
        registry=registry or build_default_registry(),
        session_factory=sessions,
        config=WorkerConfig(lease=timedelta(seconds=30)),
        worker_id="test-worker",
    )


LOG = structlog.get_logger("test")


@pytest.mark.parametrize(
    ("attempts", "max_attempts"),
    [(1, 5), (2, 5), (4, 5), (1, 2)],
)
async def test_retries_while_attempts_remain(attempts: int, max_attempts: int) -> None:
    queue = FakeQueue()
    job = make_job(attempts, max_attempts)

    await make_worker(queue)._retry_or_dead_letter(job, TransientError("boom"), LOG)

    assert [entry[0] for entry in queue.rescheduled] == [job.id]
    assert queue.dead_lettered == []


@pytest.mark.parametrize(
    ("attempts", "max_attempts"),
    [(5, 5), (6, 5), (1, 1), (2, 1)],
)
async def test_dead_letters_once_attempts_are_exhausted(
    attempts: int, max_attempts: int
) -> None:
    queue = FakeQueue()
    job = make_job(attempts, max_attempts)

    await make_worker(queue)._retry_or_dead_letter(job, TransientError("boom"), LOG)

    assert [entry[0] for entry in queue.dead_lettered] == [job.id]
    assert queue.rescheduled == []


async def test_the_boundary_is_attempts_greater_or_equal_max_attempts() -> None:
    # attempts is incremented at claim, so the attempt that just failed is
    # already counted: 5 of 5 means the budget is spent, not that one remains.
    assert make_job(4, 5).attempts_exhausted is False
    assert make_job(5, 5).attempts_exhausted is True


async def test_reschedule_is_pushed_into_the_future() -> None:
    queue = FakeQueue()
    job = make_job(1)
    before = datetime.now(UTC)

    await make_worker(queue)._retry_or_dead_letter(job, TransientError("boom"), LOG)

    _, run_after, _ = queue.rescheduled[0]
    assert run_after > before


async def test_the_recorded_error_is_never_empty() -> None:
    queue = FakeQueue()

    # str(TransientError()) is '', which would leave last_error blank in the
    # table someone is querying at 3am.
    await make_worker(queue)._retry_or_dead_letter(make_job(1), TransientError(), LOG)

    assert queue.rescheduled[0][2] == "TransientError"


async def test_an_unregistered_job_type_is_dead_lettered_not_retried() -> None:
    queue = FakeQueue()
    job = make_job(1, job_type=JobType.NOOP)
    empty = HandlerRegistry()

    await make_worker(queue, empty)._run_job(job)

    assert [entry[0] for entry in queue.dead_lettered] == [job.id]
    assert "no handler registered" in queue.dead_lettered[0][1]
    assert queue.rescheduled == []


async def test_a_permanent_failure_skips_the_attempt_budget_entirely() -> None:
    queue = FakeQueue()
    # Four attempts still available, and it is dead-lettered anyway: retrying
    # cannot turn a permanent failure into a success.
    job = make_job(1, max_attempts=5, job_type=JobType.FAIL_PERMANENT)

    await make_worker(queue)._run_job(job)

    assert [entry[0] for entry in queue.dead_lettered] == [job.id]
    assert queue.rescheduled == []


async def test_a_transient_failure_with_budget_left_is_rescheduled() -> None:
    queue = FakeQueue()
    job = make_job(1, max_attempts=5, job_type=JobType.FAIL_TRANSIENT)

    await make_worker(queue)._run_job(job)

    assert [entry[0] for entry in queue.rescheduled] == [job.id]
    assert queue.dead_lettered == []


async def test_a_successful_handler_completes_the_job() -> None:
    queue = FakeQueue()
    job = make_job(1, job_type=JobType.NOOP)

    await make_worker(queue)._run_job(job)

    assert queue.completed == [job.id]
    assert queue.rescheduled == []
    assert queue.dead_lettered == []


async def test_an_unclassified_exception_is_retried_under_the_cap() -> None:
    # Unknown failures are treated as transient but bounded: an unrecognised
    # transient one recovers, an unrecognised permanent one still terminates.
    queue = FakeQueue()
    registry = HandlerRegistry()

    async def explode(ctx: object) -> None:
        raise RuntimeError("something nobody classified")

    registry.register(JobType.NOOP, cast("Any", explode))
    job = make_job(1, max_attempts=5, job_type=JobType.NOOP)

    await make_worker(queue, registry)._run_job(job)

    assert [entry[0] for entry in queue.rescheduled] == [job.id]


async def test_an_unclassified_exception_still_dead_letters_at_the_cap() -> None:
    queue = FakeQueue()
    registry = HandlerRegistry()

    async def explode(ctx: object) -> None:
        raise RuntimeError("something nobody classified")

    registry.register(JobType.NOOP, cast("Any", explode))
    job = make_job(5, max_attempts=5, job_type=JobType.NOOP)

    await make_worker(queue, registry)._run_job(job)

    assert [entry[0] for entry in queue.dead_lettered] == [job.id]


async def test_a_fenced_out_worker_writes_nothing_further() -> None:
    # complete() returning False means another worker owns this job now. The
    # worker must not follow up with a retry or a dead-letter on top of it.
    queue = FakeQueue()
    queue.fence = False
    job = make_job(1, job_type=JobType.NOOP)

    await make_worker(queue)._run_job(job)

    assert queue.completed == [job.id]
    assert queue.rescheduled == []
    assert queue.dead_lettered == []


async def test_default_worker_ids_are_unique_per_process() -> None:
    # Host and pid alone repeat after a restart, and a recycled identity would
    # pass the fencing check on a job the previous process still holds.
    assert default_worker_id() != default_worker_id()
