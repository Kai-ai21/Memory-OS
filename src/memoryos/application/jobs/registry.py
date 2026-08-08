"""Mapping from job type to the coroutine that performs it."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from memoryos.domain.jobs import Job, JobType


@dataclass(slots=True)
class JobContext:
    """What a handler is given: the job, and a session to do its work in.

    The session is created fresh for each job by the worker and committed only
    if the handler returns without raising.
    """

    job: Job
    session: AsyncSession


Handler = Callable[[JobContext], Awaitable[None]]


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}

    def register(self, job_type: JobType, handler: Handler) -> None:
        self._handlers[job_type.value] = handler

    def get(self, job_type: str) -> Handler | None:
        """The handler for `job_type`, or None if nothing is registered.

        The lookup takes a plain `str` rather than a `JobType`, because the
        value comes out of the database and may name a type this build does not
        know about — an older worker draining a queue a newer one wrote to. The
        worker treats that as permanent: retrying cannot make a handler appear.
        """
        return self._handlers.get(job_type)

    def registered_types(self) -> frozenset[str]:
        return frozenset(self._handlers)
