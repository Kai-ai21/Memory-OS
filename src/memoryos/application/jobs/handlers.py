"""Test handlers.

These exist so the worker's success, retry, and dead-letter paths can be
exercised end to end before there is any real work to do. The first real job
types arrive with the connector in M1.3.
"""

from memoryos.application.jobs.registry import HandlerRegistry, JobContext
from memoryos.domain.jobs import JobType, PermanentError, TransientError


async def handle_noop(ctx: JobContext) -> None:
    """Succeed, writing nothing."""


async def handle_fail_transient(ctx: JobContext) -> None:
    raise TransientError(f"transient failure on attempt {ctx.job.attempts}")


async def handle_fail_permanent(ctx: JobContext) -> None:
    raise PermanentError("permanent failure; retrying cannot help")


def build_default_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    registry.register(JobType.NOOP, handle_noop)
    registry.register(JobType.FAIL_TRANSIENT, handle_fail_transient)
    registry.register(JobType.FAIL_PERMANENT, handle_fail_permanent)
    return registry
