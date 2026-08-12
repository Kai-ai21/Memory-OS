"""Job handlers, and the registry the worker looks them up in."""

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import BlobNotFound
from memoryos.adapters.parsers.registry import build_default_registry as build_parser_registry
from memoryos.application.embed import EmbedMemory
from memoryos.application.extraction import ExtractEntities
from memoryos.application.graph_sync import SyncGraph
from memoryos.application.jobs.registry import Handler, HandlerRegistry, JobContext
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.ports import (
    BlobStore,
    Chunker,
    Connector,
    Embedder,
    EmbeddingCache,
    EntityExtractor,
    GraphStore,
    JobQueue,
)
from memoryos.application.sync import SyncSource
from memoryos.domain.jobs import JobType, PermanentError, TransientError

logger = structlog.get_logger(__name__)


def make_embed_handler(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    cache: EmbeddingCache,
    batch_size: int,
) -> Handler:
    """Build the `EMBED_MEMORY` handler."""

    embed = EmbedMemory(session_factory, embedder, cache, batch_size)

    async def handle_embed_memory(ctx: JobContext) -> None:
        raw_id = ctx.job.payload.get("memory_id")
        if not raw_id:
            raise PermanentError("embed_memory job has no memory_id in its payload")

        report = await embed(UUID(str(raw_id)))
        logger.info("embed.job_finished", job_id=str(ctx.job.id), **report.as_dict())

    return handle_embed_memory


def make_extract_handler(
    session_factory: async_sessionmaker[AsyncSession],
    extractor: EntityExtractor,
    queue: JobQueue | None = None,
) -> Handler:
    """Build the `EXTRACT_ENTITIES` handler.

    Nothing here catches provider failures. The adapters already classify them —
    a rate limit is `TransientError` and reaches the worker's existing backoff
    unchanged, which is exactly what M2.6 built that taxonomy for, and what makes
    a free tier's quota a pause rather than a dead-lettered corpus.

    The queue is passed rather than the graph: extraction queues a `SYNC_GRAPH`
    job and writes nothing to Neo4j itself.
    """

    extract = ExtractEntities(session_factory, extractor, queue)

    async def handle_extract_entities(ctx: JobContext) -> None:
        raw_id = ctx.job.payload.get("memory_id")
        if not raw_id:
            raise PermanentError("extract_entities job has no memory_id in its payload")

        report = await extract(UUID(str(raw_id)))
        logger.info("extract.job_finished", job_id=str(ctx.job.id), **report.as_dict())

    return handle_extract_entities


def make_graph_sync_handler(
    session_factory: async_sessionmaker[AsyncSession], graph: GraphStore
) -> Handler:
    """Build the `SYNC_GRAPH` handler.

    Nothing is caught. Every other stage in this pipeline treats an unreachable
    graph as degradation and logs past it, because their own work is already
    committed and correct — and that is exactly why the sync must not. It has no
    other work: if it cannot reach the graph, it has done nothing, and reporting
    success would leave a projection permanently behind with no record that it is.
    A raise reaches the worker's backoff, and the retry is what makes a Neo4j
    that was down for a minute a pause rather than a divergence.
    """

    sync = SyncGraph(session_factory, graph)

    async def handle_sync_graph(ctx: JobContext) -> None:
        report = await sync(ctx.job.payload)
        logger.info("graph.sync_job_finished", job_id=str(ctx.job.id), **report.as_dict())

    return handle_sync_graph


def make_normalize_handler(
    session_factory: async_sessionmaker[AsyncSession],
    blob_store: BlobStore,
    chunker: Chunker,
) -> Handler:
    """Build the `NORMALIZE_MEMORY` handler."""

    normalize = NormalizeMemory(
        session_factory, blob_store, build_parser_registry(), chunker
    )

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
    embedder: Embedder | None = None,
    cache: EmbeddingCache | None = None,
    chunker: Chunker | None = None,
    batch_size: int = 32,
    extractor: EntityExtractor | None = None,
    graph: GraphStore | None = None,
    queue: JobQueue | None = None,
) -> HandlerRegistry:
    """The registry a worker runs with.

    The real job types need collaborators the test types do not, so each is
    registered only when its own are supplied. A worker built without them
    simply has no handler for that type, and the worker already treats an
    unregistered type as permanent rather than retrying forever.
    """
    registry = HandlerRegistry()
    registry.register(JobType.NOOP, handle_noop)
    registry.register(JobType.FAIL_TRANSIENT, handle_fail_transient)
    registry.register(JobType.FAIL_PERMANENT, handle_fail_permanent)
    if session_factory is not None and embedder is not None and cache is not None:
        registry.register(
            JobType.EMBED_MEMORY,
            make_embed_handler(session_factory, embedder, cache, batch_size),
        )

    if session_factory is not None and blob_store is not None and chunker is not None:
        registry.register(
            JobType.NORMALIZE_MEMORY,
            make_normalize_handler(session_factory, blob_store, chunker),
        )

    if session_factory is not None and extractor is not None:
        registry.register(
            JobType.EXTRACT_ENTITIES,
            make_extract_handler(session_factory, extractor, queue),
        )

    if session_factory is not None and graph is not None:
        registry.register(
            JobType.SYNC_GRAPH,
            make_graph_sync_handler(session_factory, graph),
        )

    if session_factory is not None and connector is not None and blob_store is not None:
        registry.register(
            JobType.SYNC_SOURCE,
            make_sync_handler(session_factory, connector, blob_store),
        )
    return registry
