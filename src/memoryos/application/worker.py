"""The worker process: claim a job, run it, record what happened.

The invariants worth keeping in mind while reading this:

* A job that is claimed is always accounted for. It succeeds, it is rescheduled,
  it is dead-lettered, or its lease expires and the sweeper returns it. There is
  no fourth outcome.
* A worker that has lost its lease writes nothing. Another worker owns that job
  now, and a late write from the previous owner would corrupt its state. Every
  mutating call returns a bool for exactly this reason.
* Nothing here holds a database transaction open across the handler.
"""

import asyncio
import contextlib
import os
import signal
import socket
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.application.jobs.registry import Handler, HandlerRegistry, JobContext
from memoryos.application.live import memory_ready, notify_sql
from memoryos.application.ports import JobQueue
from memoryos.domain.backoff import compute_backoff
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import Job, JobType, PermanentError, TransientError

logger = structlog.get_logger(__name__)


def default_worker_id() -> str:
    """Host, pid, and a random suffix.

    The suffix matters: host and pid alone repeat after a restart, and a
    recycled identity would let a new process pass the fencing check on a job
    its predecessor still nominally holds.
    """
    return f"{socket.gethostname()}/{os.getpid()}/{str(new_id())[-12:]}"


@dataclass(slots=True, frozen=True)
class WorkerConfig:
    # How long a claim is good for without a heartbeat. Long enough to survive a
    # slow handler, short enough that a dead worker's jobs come back promptly.
    lease: timedelta = timedelta(seconds=30)
    # Adaptive poll bounds. A fixed tight poll burns queries on an idle queue; a
    # fixed slow poll adds its whole interval to the latency of every job.
    poll_min_seconds: float = 0.1
    poll_max_seconds: float = 2.0
    # The sweep is a full-table-ish write; it does not belong on every tick.
    reclaim_every_iterations: int = 50
    reclaim_limit: int = 100
    # Grace period for an in-flight job once a stop is requested.
    shutdown_timeout_seconds: float = 30.0
    # Consecutive empty polls before a --drain worker exits.
    idle_polls_before_drain_stop: int = 3
    concurrency: int = 1
    # Job types this worker will claim. Empty means every type, which is the
    # ordinary case and the default.
    #
    # It exists for the drains, not for the long-running worker. The queue is
    # shared across phases and the checks are not: `make phase1-check` drains it
    # to prove Phase 1's pipeline works, and embedding enqueues a Phase 3 entity
    # extraction per memory — so on a machine with an API key configured, a check
    # about ingestion blocks on hundreds of live model calls and never finishes.
    # Restricting the *claim* leaves the excluded jobs pending rather than
    # burning an attempt on each, so a later unrestricted drain still runs them.
    only: frozenset[JobType] = frozenset()

    def __post_init__(self) -> None:
        if self.concurrency != 1:
            # The knob exists because the loop will grow slots once there is
            # real work to overlap. Until then, accepting a value it ignores
            # would be worse than refusing it.
            raise ValueError("concurrency > 1 is not implemented yet (M1.2 runs one job)")
        if self.poll_min_seconds <= 0 or self.poll_max_seconds < self.poll_min_seconds:
            raise ValueError("poll bounds must satisfy 0 < poll_min_seconds <= poll_max_seconds")


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        registry: HandlerRegistry,
        session_factory: async_sessionmaker[AsyncSession],
        config: WorkerConfig | None = None,
        worker_id: str | None = None,
    ) -> None:
        self._queue = queue
        self._registry = registry
        self._sessions = session_factory
        self._config = config or WorkerConfig()
        self._worker_id = worker_id or default_worker_id()
        self._stopping = asyncio.Event()
        self._poll_delay = self._config.poll_min_seconds
        self._current_handler: asyncio.Task[None] | None = None
        self._forced = False

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def request_stop(self) -> None:
        """Stop claiming new work; let the job in flight finish.

        Killing a worker mid-job is survivable — that is what the lease is for —
        but a clean stop skips the reclaim delay entirely, so the job is
        finished rather than merely recoverable.
        """
        if self._stopping.is_set():
            return
        self._stopping.set()
        logger.info("worker.stopping", worker_id=self._worker_id)
        loop = asyncio.get_running_loop()
        loop.call_later(self._config.shutdown_timeout_seconds, self._force_stop)

    def _force_stop(self) -> None:
        handler = self._current_handler
        if handler is not None and not handler.done():
            logger.warning("worker.shutdown_timeout", worker_id=self._worker_id)
            self._forced = True
            handler.cancel()

    async def run(self, *, drain: bool = False, handle_signals: bool = True) -> None:
        """Drain the queue until stopped.

        `drain=True` exits once the queue has been empty for a few consecutive
        polls, which is what a one-shot run wants. Otherwise this runs until a
        signal arrives.
        """
        if handle_signals:
            self._install_signal_handlers()

        logger.info("worker.started", worker_id=self._worker_id, drain=drain)
        iterations = 0
        idle_polls = 0

        try:
            while not self._stopping.is_set():
                if iterations % self._config.reclaim_every_iterations == 0:
                    await self._reclaim()
                iterations += 1

                job = await self._queue.claim(
                    self._worker_id,
                    self._config.lease,
                    only=self._config.only or None,
                )

                if job is None:
                    idle_polls += 1
                    if drain and idle_polls >= self._config.idle_polls_before_drain_stop:
                        break
                    await self._idle_sleep()
                    continue

                idle_polls = 0
                self._poll_delay = self._config.poll_min_seconds
                await self._run_job(job)
        except asyncio.CancelledError:
            # A forced shutdown cancelled the handler and the cancellation
            # reached us. The job stays 'running' with a lease that will expire,
            # so the sweeper returns it; nothing is lost.
            if not self._forced:
                raise
        finally:
            logger.info("worker.stopped", worker_id=self._worker_id)

    async def _reclaim(self) -> None:
        reclaimed = await self._queue.reclaim_expired(self._config.reclaim_limit)
        if reclaimed:
            logger.info("worker.reclaimed_expired", worker_id=self._worker_id, count=reclaimed)

    async def _idle_sleep(self) -> None:
        """Sleep, but wake immediately if a stop is requested."""
        delay = self._poll_delay
        self._poll_delay = min(delay * 2, self._config.poll_max_seconds)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=delay)

    async def _run_job(self, job: Job) -> None:
        log = logger.bind(
            worker_id=self._worker_id,
            job_id=str(job.id),
            job_type=job.job_type,
            attempts=job.attempts,
        )
        log.info("job.claimed")

        handler = self._registry.get(job.job_type)
        if handler is None:
            # Not a transient condition: no amount of retrying makes a handler
            # appear in a process that was not built with one.
            await self._dead_letter(
                job, PermanentError(f"no handler registered for job_type {job.job_type!r}"), log
            )
            return

        lease_lost = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(job, lease_lost))
        handler_task = asyncio.create_task(self._invoke(handler, job))
        self._current_handler = handler_task
        lost_watcher = asyncio.create_task(lease_lost.wait())

        try:
            await asyncio.wait(
                {handler_task, lost_watcher}, return_when=asyncio.FIRST_COMPLETED
            )

            if not handler_task.done():
                # The lease went. Cancel the handler and write nothing at all:
                # another worker owns this job now, and a terminal write from
                # here would overwrite its state.
                handler_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await handler_task
                log.warning("job.lease_lost")
                return

            await self._settle(job, handler_task, log)
        finally:
            self._current_handler = None
            for task in (heartbeat, lost_watcher):
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(heartbeat, lost_watcher, return_exceptions=True)

    async def _settle(
        self, job: Job, handler_task: asyncio.Task[None], log: structlog.BoundLogger
    ) -> None:
        """Classify the handler's outcome and record it.

        Unknown exceptions are treated as transient, under the attempt cap. That
        default is safe in both directions: an unrecognised transient failure
        recovers on its own, and an unrecognised permanent one is bounded rather
        than eternal.
        """
        try:
            await handler_task
            if await self._queue.complete(job.id, self._worker_id):
                log.info("job.succeeded")
                await self._announce(job)
            else:
                log.warning("job.lease_lost")
        except PermanentError as exc:
            await self._dead_letter(job, exc, log)
        except TransientError as exc:
            await self._retry_or_dead_letter(job, exc, log)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._retry_or_dead_letter(job, exc, log)

    async def _invoke(self, handler: Handler, job: Job) -> None:
        """Run the handler in a session of its own, committed only on success.

        A handler that raises leaves the session unclosed on this path, so the
        context manager rolls it back: a failed job writes nothing.
        """
        async with self._sessions() as session:
            await handler(JobContext(job=job, session=session))
            await session.commit()

    async def _heartbeat(self, job: Job, lease_lost: asyncio.Event) -> None:
        """Extend the lease while the handler works.

        A third of the lease gives two chances to renew before it lapses, so a
        single slow or failed round trip does not cost the job.
        """
        interval = self._config.lease.total_seconds() / 3
        while True:
            await asyncio.sleep(interval)
            if not await self._queue.heartbeat(job.id, self._worker_id, self._config.lease):
                lease_lost.set()
                return

    async def _retry_or_dead_letter(
        self, job: Job, exc: BaseException, log: structlog.BoundLogger
    ) -> None:
        tb = traceback.format_exc()
        if job.attempts_exhausted:
            await self._dead_letter(job, exc, log, tb=tb)
            return

        delay = compute_backoff(job.attempts)
        run_after = datetime.now(UTC) + timedelta(seconds=delay)
        if await self._queue.reschedule(
            job.id, self._worker_id, run_after, _describe(exc), tb
        ):
            log.info("job.retrying", delay_seconds=round(delay, 3), run_after=str(run_after))
        else:
            log.warning("job.lease_lost")

    async def _announce(self, job: Job) -> None:
        """Tell any open page that this memory moved.

        `NOTIFY` through Postgres, which is the boundary the worker and the API
        already share — the queue is a table for the same reason. Nothing is
        computed here: the payload is the memory id and what finished, and the API
        re-reads the status when it arrives, so what reaches a browser is the
        state at delivery rather than at publication.

        **A failure to announce is logged and swallowed**, and that is deliberate.
        The job succeeded; its work is committed. Letting a notification failure
        turn a completed job into a retried one would redo real work — an
        extraction costs money — to fix a stream that reconnects on its own and
        whose client polls as a fallback anyway.
        """
        memory_id = job.payload.get("memory_id")
        if not memory_id:
            return
        channel, payload = notify_sql(
            "memory_ready", memory_ready(UUID(str(memory_id)), str(job.job_type))
        )
        try:
            async with self._sessions.begin() as session:
                await session.execute(
                    text("SELECT pg_notify(:channel, :payload)"),
                    {"channel": channel, "payload": payload},
                )
        except Exception:
            logger.warning("worker.announce_failed", job_id=str(job.id), exc_info=True)

    async def _dead_letter(
        self,
        job: Job,
        exc: BaseException,
        log: structlog.BoundLogger,
        tb: str | None = None,
    ) -> None:
        detail = tb if tb is not None else traceback.format_exc()
        if await self._queue.dead_letter(job.id, self._worker_id, _describe(exc), detail):
            log.warning("job.dead_lettered", error=_describe(exc))
        else:
            log.warning("job.lease_lost")

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, self.request_stop)


def _describe(exc: BaseException) -> str:
    """A message that is never empty.

    `str(SomeError())` is `''`, and an empty `last_error` column tells whoever
    is debugging at 3am nothing at all.
    """
    return str(exc) or type(exc).__name__
