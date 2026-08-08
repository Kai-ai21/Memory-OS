import pytest

from memoryos.application.jobs.handlers import build_default_registry
from memoryos.application.jobs.registry import HandlerRegistry, JobContext
from memoryos.domain.jobs import JobType, PermanentError, TransientError

from .test_worker_decisions import make_job


def test_unknown_job_type_returns_none() -> None:
    assert HandlerRegistry().get("no_such_type") is None


def test_unknown_job_type_returns_none_even_with_handlers_registered() -> None:
    # The lookup takes a str because the value comes from the database and may
    # name a type this build has never heard of.
    assert build_default_registry().get("some_type_from_a_newer_build") is None


def test_registered_handlers_are_returned() -> None:
    registry = build_default_registry()
    for job_type in (JobType.NOOP, JobType.FAIL_TRANSIENT, JobType.FAIL_PERMANENT):
        assert registry.get(job_type.value) is not None


def test_registering_the_same_type_twice_replaces_the_handler() -> None:
    registry = HandlerRegistry()

    async def first(ctx: JobContext) -> None: ...

    async def second(ctx: JobContext) -> None: ...

    registry.register(JobType.NOOP, first)
    registry.register(JobType.NOOP, second)

    assert registry.get(JobType.NOOP.value) is second


def test_registered_types_reports_what_is_known() -> None:
    assert build_default_registry().registered_types() == {
        "noop",
        "fail_transient",
        "fail_permanent",
    }


@pytest.mark.parametrize(
    "job_type",
    [JobType.SYNC_SOURCE, JobType.NORMALIZE_MEMORY, JobType.EMBED_MEMORY],
)
def test_real_job_types_are_absent_without_their_collaborators(job_type: JobType) -> None:
    # These need a session factory, a blob store, and (for sync) a connector. A
    # registry built without them simply has no handler, and the worker already
    # dead-letters an unregistered type rather than retrying forever.
    assert build_default_registry().get(job_type.value) is None


async def test_the_noop_handler_succeeds_and_writes_nothing() -> None:
    handler = build_default_registry().get(JobType.NOOP.value)
    assert handler is not None
    # A None session is safe precisely because this handler touches nothing.
    await handler(JobContext(job=make_job(1), session=None))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("job_type", "expected"),
    [
        (JobType.FAIL_TRANSIENT, TransientError),
        (JobType.FAIL_PERMANENT, PermanentError),
    ],
)
async def test_the_failing_handlers_raise_their_classified_errors(
    job_type: JobType, expected: type[Exception]
) -> None:
    handler = build_default_registry().get(job_type.value)
    assert handler is not None
    with pytest.raises(expected):
        await handler(JobContext(job=make_job(1), session=None))  # type: ignore[arg-type]
