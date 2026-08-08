"""Job handlers, and the registry the worker looks them up in."""

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import BlobNotFound
from memoryos.adapters.parsers.registry import build_default_registry as build_parser_registry
from memoryos.application.jobs.registry import Handler, HandlerRegistry, JobContext
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.ports import BlobStore, Connector
from memoryos.application.sync import SyncSource
from memoryos.domain.jobs import JobType, PermanentError, TransientError

logger = structlog.get_logger(__name__)


async def handle_embed_memory(ctx: JobContext) -> None:
    """A stub until M1.5.

    Chunks land with `embedding IS NULL`; this is what will fill them. It is
    enqueued now so the normalize-to-embed wiring is exercised end to end
    without pulling embeddings into this milestone.
    """
    logger.info(
        "embed.stub", job_id=str(ctx.job.id), memory_id=ctx.job.payload.get("memory_id")
    )


def make_normalize_handler(
    session_factory: async_sessionmaker[AsyncSession], blob_store: BlobStore
) -> Handler:
    """Build the `NORMALIZE_MEMORY` handler."""

    normalize = NormalizeMemory(session_factory, blob_store, build_parser_registry())

    async def handle_normalize_memory(ctx: JobContext) -> None:
        raw_id = ctx.job.payload.get("memory_id")
        if not raw_id:
            raise PermanentError("normalize_memory job has no memory_id in its payload")

        try:
            report = await normalize(UUID(str(raw_id)))
        except BlobNotFound as exc:
            # The artifact row promises bytes that are not in the store. That
            # is a broken invariant, not a passing failure.
            raise PermanentError(f"blob missing for memory {raw_id}: {exc}") from exc

        logger.info("normalize.job_finished", job_id=str(ctx.job.id), **report.as_dict())

    return handle_normalize_memory


def make_sync_handler(
    session_factory: async_sessionmaker[AsyncSession],
    connector: Connector,
    blob_store: BlobStore,
) -> Handler:
    """Build the `SYNC_SOURCE` handler.

    Syncs are themselves queued work, which is what lets the HTTP endpoint
    return immediately and lets a failed sync retry with backoff like anything
    else.
    """

    async def handle_sync_source(ctx: JobContext) -> None:
        raw_id = ctx.job.payload.get("source_id")
        if not raw_id:
            # Nothing to retry towards: this job can never become valid.
            raise PermanentError("sync_source job has no source_id in its payload")

        sync = SyncSource(session_factory, connector, blob_store)
        try:
            report = await sync(UUID(str(raw_id)), full=bool(ctx.job.payload.get("full")))
        except LookupError as exc:
            raise PermanentError(str(exc)) from exc
        except OSError as exc:
            # An unmounted volume or a disappearing root is exactly the kind of
            # thing that is fixed by the time the retry runs.
            raise TransientError(str(exc)) from exc

        logger.info("sync.job_finished", job_id=str(ctx.job.id), **report.as_dict())

    return handle_sync_source


async def handle_noop(ctx: JobContext) -> None:
    """Succeed, writing nothing."""


async def handle_fail_transient(ctx: JobContext) -> None:
    raise TransientError(f"transient failure on attempt {ctx.job.attempts}")


async def handle_fail_permanent(ctx: JobContext) -> None:
    raise PermanentError("permanent failure; retrying cannot help")


def build_default_registry(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    connector: Connector | None = None,
    blob_store: BlobStore | None = None,
) -> HandlerRegistry:
    """The registry a worker runs with.

    `SYNC_SOURCE` and `NORMALIZE_MEMORY` need collaborators the test types do
    not, so they are only registered when those are supplied. A worker built
    without them simply has no handler for that type, and the worker already
    treats an unregistered type as permanent rather than retrying forever.
    """
    registry = HandlerRegistry()
    registry.register(JobType.NOOP, handle_noop)
    registry.register(JobType.FAIL_TRANSIENT, handle_fail_transient)
    registry.register(JobType.FAIL_PERMANENT, handle_fail_permanent)
    registry.register(JobType.EMBED_MEMORY, handle_embed_memory)

    if session_factory is not None and blob_store is not None:
        registry.register(
            JobType.NORMALIZE_MEMORY, make_normalize_handler(session_factory, blob_store)
        )

    if session_factory is not None and connector is not None and blob_store is not None:
        registry.register(
            JobType.SYNC_SOURCE,
            make_sync_handler(session_factory, connector, blob_store),
        )
    return registry
