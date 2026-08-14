"""Postgres-backed job queue.

Unlike the M1.1 repositories, this adapter owns its transactions. A queue
operation must be short and must commit on its own: a worker that held one
transaction open from claim through to completion would pin a connection for
the whole handler and block vacuum on a table that churns constantly.

The one exception is `enqueue_in`, which runs inside a transaction the caller
already owns. That is the point of putting the queue in the database: a use
case can write its data and enqueue the job that processes it in a single
transaction, so there is no window where one committed and the other did not.
`PostgresJobQueue.enqueue` is the same statement with a transaction wrapped
around it, for callers that have nothing else to commit.
"""

from datetime import datetime, timedelta
from typing import Any, cast
from uuid import UUID

from sqlalchemy import (
    ColumnElement,
    CursorResult,
    RowMapping,
    Select,
    case,
    func,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import JobQueue
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import Job, JobSpec, JobStatus, JobType

# The partial unique index that makes enqueue idempotent. Naming its columns and
# predicate here targets that index specifically, so a primary-key collision
# still raises instead of being silently swallowed as a duplicate.
_DEDUPE_INDEX_WHERE = text("dedupe_key IS NOT NULL AND status IN ('pending', 'running')")
_DEDUPE_INDEX_ELEMENTS = ["job_type", "dedupe_key"]

_JOB_COLUMNS = models.Job.__table__.c


def _to_job(row: RowMapping) -> Job:
    return Job(
        id=row["id"],
        job_type=row["job_type"],
        payload=dict(row["payload"]),
        status=JobStatus(row["status"]),
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        run_after=row["run_after"],
        priority=row["priority"],
        dedupe_key=row["dedupe_key"],
        locked_by=row["locked_by"],
        locked_at=row["locked_at"],
        lease_expires_at=row["lease_expires_at"],
        last_error=row["last_error"],
        last_traceback=row["last_traceback"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
    )


def _lease_expiry(lease: timedelta) -> ColumnElement[datetime]:
    """`now() + lease`, evaluated against the database clock.

    Deliberately not `datetime.now() + lease` computed here: leases are compared
    against `now()` inside Postgres, and a worker whose clock runs a few seconds
    fast would otherwise hand itself a lease that is already expiring.
    """
    return func.now() + lease


def _owned_by(job_id: UUID, worker_id: str) -> ColumnElement[bool]:
    """The fencing predicate every mutating statement carries.

    A worker whose lease expired mid-handler no longer matches `locked_by`, so
    its late write finds no row and reports failure rather than trampling the
    state of whichever worker owns the job now.
    """
    return (
        (models.Job.id == job_id)
        & (models.Job.locked_by == worker_id)
        & (models.Job.status == JobStatus.RUNNING.value)
    )


async def enqueue_in(session: AsyncSession, spec: JobSpec) -> UUID | None:
    """Enqueue inside a transaction the caller owns.

    Returns None when a job with the same (job_type, dedupe_key) is already
    pending or running.
    """
    values: dict[str, Any] = {
        "id": new_id(),
        "job_type": spec.job_type.value,
        "payload": spec.payload,
        "dedupe_key": spec.dedupe_key,
        "status": JobStatus.PENDING.value,
        "priority": spec.priority,
        "max_attempts": spec.max_attempts,
    }
    if spec.run_after is not None:
        values["run_after"] = spec.run_after

    stmt = (
        insert(models.Job)
        .values(**values)
        .on_conflict_do_nothing(
            index_elements=_DEDUPE_INDEX_ELEMENTS, index_where=_DEDUPE_INDEX_WHERE
        )
        .returning(models.Job.id)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


class PostgresJobQueue(JobQueue):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def enqueue(self, spec: JobSpec) -> UUID | None:
        async with self._sessions.begin() as session:
            return await enqueue_in(session, spec)

    async def claim(
        self,
        worker_id: str,
        lease: timedelta,
        *,
        only: frozenset[JobType] | None = None,
    ) -> Job | None:
        """Take the highest-priority ready job, or None if there is none.

        `FOR UPDATE SKIP LOCKED` on the inner select is the clause that makes
        more than one worker useful. Without it every worker picks the same top
        row, all but one block on the lock, and the queue drains serially no
        matter how many workers are running. With it, each worker steps over
        the rows its peers have already taken.

        `attempts` increments here rather than in the failure path. A worker
        that segfaults or is SIGKILLed never reaches its failure handler, so a
        job that reliably kills the process would otherwise retry forever with
        `attempts` stuck at zero — a poison message nothing can drain. Counting
        the attempt at claim time means every job exhausts its budget, however
        it dies.
        """
        candidate: Select[tuple[UUID]] = (
            select(models.Job.id)
            .where(
                models.Job.status == JobStatus.PENDING.value,
                models.Job.run_after <= func.now(),
            )
            .order_by(models.Job.priority.desc(), models.Job.run_after)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if only is not None:
            # **In the candidate select, not after the claim.** A filter applied
            # after claiming would take the job, find it unwanted, and have to
            # release it — which increments `attempts` on a job nothing was
            # wrong with, and a few passes of that dead-letters the queue.
            candidate = candidate.where(
                models.Job.job_type.in_([item.value for item in only])
            )

        stmt = (
            update(models.Job)
            .where(models.Job.id == candidate.scalar_subquery())
            .values(
                status=JobStatus.RUNNING.value,
                locked_by=worker_id,
                locked_at=func.now(),
                lease_expires_at=_lease_expiry(lease),
                attempts=models.Job.attempts + 1,
                updated_at=func.now(),
            )
            .returning(*_JOB_COLUMNS)
        )

        # Its own transaction, committed on the way out. The handler runs
        # afterwards, on a different session, with nothing held open here.
        async with self._sessions.begin() as session:
            row = (await session.execute(stmt)).mappings().one_or_none()
            return None if row is None else _to_job(row)

    async def heartbeat(self, job_id: UUID, worker_id: str, lease: timedelta) -> bool:
        """Extend the lease. False means it was lost while we were working."""
        return await self._fenced_update(
            job_id,
            worker_id,
            lease_expires_at=_lease_expiry(lease),
            updated_at=func.now(),
        )

    async def complete(self, job_id: UUID, worker_id: str) -> bool:
        return await self._fenced_update(
            job_id,
            worker_id,
            status=JobStatus.SUCCEEDED.value,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            completed_at=func.now(),
            updated_at=func.now(),
        )

    async def reschedule(
        self, job_id: UUID, worker_id: str, run_after: datetime, error: str, tb: str
    ) -> bool:
        return await self._fenced_update(
            job_id,
            worker_id,
            status=JobStatus.PENDING.value,
            run_after=run_after,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            last_error=error,
            last_traceback=tb,
            updated_at=func.now(),
        )

    async def dead_letter(self, job_id: UUID, worker_id: str, error: str, tb: str) -> bool:
        return await self._fenced_update(
            job_id,
            worker_id,
            status=JobStatus.FAILED.value,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            last_error=error,
            last_traceback=tb,
            completed_at=func.now(),
            updated_at=func.now(),
        )

    async def reclaim_expired(self, limit: int = 100) -> int:
        """Return jobs whose worker died back to the queue.

        A worker killed mid-job leaves its row in `running` with a lease that
        stops being extended. Nothing else would ever move it, so without this
        sweep the job is stuck forever.

        `attempts` was already incremented at claim, so a job that has burned
        through its budget goes straight to `failed` rather than looping.
        """
        expired: Select[tuple[UUID]] = (
            select(models.Job.id)
            .where(
                models.Job.status == JobStatus.RUNNING.value,
                models.Job.lease_expires_at < func.now(),
            )
            .order_by(models.Job.lease_expires_at)
            .with_for_update(skip_locked=True)
            .limit(limit)
        )
        exhausted = models.Job.attempts >= models.Job.max_attempts

        stmt = (
            update(models.Job)
            .where(models.Job.id.in_(expired.scalar_subquery()))
            .values(
                status=case(
                    (exhausted, JobStatus.FAILED.value), else_=JobStatus.PENDING.value
                ),
                locked_by=None,
                locked_at=None,
                lease_expires_at=None,
                last_error=func.coalesce(models.Job.last_error, "lease expired"),
                updated_at=func.now(),
                completed_at=case((exhausted, func.now()), else_=None),
            )
        )

        async with self._sessions.begin() as session:
            result = cast("CursorResult[Any]", await session.execute(stmt))
            return result.rowcount

    async def _fenced_update(self, job_id: UUID, worker_id: str, **values: Any) -> bool:
        stmt = update(models.Job).where(_owned_by(job_id, worker_id)).values(**values)
        async with self._sessions.begin() as session:
            result = cast("CursorResult[Any]", await session.execute(stmt))
            return result.rowcount == 1
