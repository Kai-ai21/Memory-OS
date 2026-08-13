"""Job queue domain types.

Pure Python. Nothing here imports SQLAlchemy or performs I/O.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any
from uuid import UUID

DEFAULT_MAX_ATTEMPTS = 5


class JobStatus(StrEnum):
    PENDING = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()


class JobType(StrEnum):
    SYNC_SOURCE = auto()
    # A stub in M1.3: it logs and returns. M1.4 gives it a body. Enqueuing it
    # now is what proves the connector-to-queue wiring end to end without
    # pulling normalization forward into this milestone.
    NORMALIZE_MEMORY = auto()
    # A stub in M1.4: it logs and returns. M1.5 fills the vectors. Enqueuing it
    # now proves the normalize-to-embed wiring without pulling embeddings
    # forward into this milestone.
    EMBED_MEMORY = auto()
    # Enqueued when a memory finishes embedding. Last in the chain because it is
    # the only stage that costs money per run — putting it behind the free
    # stages means a memory that fails to parse or chunk never reaches the
    # model at all.
    EXTRACT_ENTITIES = auto()
    # M3.4. Enqueued when extraction or resolution changes anything the graph
    # projects, and the *only* thing that writes to Neo4j. Its payload names the
    # memories and entities that changed, so the job updates a neighbourhood
    # rather than rebuilding the whole projection.
    #
    # Last in the chain and lowest priority, because the graph is a projection:
    # a sync that jumped the queue would delay ingestion to keep derived state
    # current, and derived state that is a few seconds behind is the normal
    # condition of this system rather than a fault.
    SYNC_GRAPH = auto()
    # M6.0, and the first job type enqueued by something outside this system.
    #
    # One job per subscribed handler rather than one per event, which is what
    # makes a failing handler a failing *job*: it retries on its own schedule,
    # dead-letters on its own, and cannot take another handler's work down with
    # it. The payload names the event and the handler by name, because a worker
    # draining this queue may be a different process on a different build from
    # the one that dispatched.
    HANDLE_EVENT = auto()
    # M6.2. Assembling context takes about a second warm and twenty cold, and
    # the process that wants it is an editor panel — so the API enqueues this
    # and answers immediately rather than holding an HTTP request open while a
    # cross-encoder loads. Deduped on the focus, because two panels open on one
    # file is one piece of work.
    ASSEMBLE_CONTEXT = auto()

    # Test types. The worker's success, retry, and dead-letter paths are
    # exercised through these rather than through real work.
    NOOP = auto()
    FAIL_TRANSIENT = auto()
    FAIL_PERMANENT = auto()


class TransientError(Exception):
    """A failure worth retrying: a timeout, a rate limit, a restarting service."""


class PermanentError(Exception):
    """A failure retrying cannot fix: bad input, a handler that does not exist."""


@dataclass(frozen=True, slots=True)
class JobSpec:
    """What an enqueuer supplies. The queue fills in everything else."""

    job_type: JobType
    payload: dict[str, Any] = field(default_factory=dict)
    # Two enqueues of the same logical work collapse into one while it is still
    # pending or running. None opts out.
    dedupe_key: str | None = None
    # Higher runs first.
    priority: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    # Earliest time this may be claimed. None means immediately.
    run_after: datetime | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.dedupe_key is not None and not self.dedupe_key:
            raise ValueError("dedupe_key must be non-empty or None")


@dataclass(frozen=True, slots=True)
class Job:
    """A claimed unit of work, as the worker sees it."""

    id: UUID
    job_type: str
    payload: dict[str, Any]
    status: JobStatus
    attempts: int
    max_attempts: int
    run_after: datetime
    priority: int = 0
    dedupe_key: str | None = None
    # Worker identity and lease. A running job always has both; the database
    # enforces it, because a running job without a lease is unreclaimable.
    locked_by: str | None = None
    locked_at: datetime | None = None
    lease_expires_at: datetime | None = None
    last_error: str | None = None
    last_traceback: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def attempts_exhausted(self) -> bool:
        """Whether this job has used its budget.

        `attempts` is incremented at claim time, so by the time a handler
        fails, the attempt it just consumed is already counted.
        """
        return self.attempts >= self.max_attempts

    def __post_init__(self) -> None:
        if self.attempts < 0:
            raise ValueError(f"attempts must be >= 0, got {self.attempts}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
