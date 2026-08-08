"""Ports: what the application layer needs from the outside world.

Protocols only. No implementations, no imports from `memoryos.adapters`. The
dependency arrow points inward: adapters implement these, never the reverse.
"""

from collections.abc import Sequence
from typing import Any, Protocol
from uuid import UUID

from memoryos.domain.entities import IngestionEvent, Memory, RawArtifact, Source
from memoryos.domain.values import ContentHash, SourceKind


class SourceRepository(Protocol):
    async def get(self, source_id: UUID) -> Source | None: ...

    async def get_by_name(self, kind: SourceKind, name: str) -> Source | None: ...

    async def add(self, source: Source) -> None: ...

    async def update_cursor(self, source_id: UUID, cursor: dict[str, Any]) -> None: ...


class ArtifactRepository(Protocol):
    async def exists(self, content_hash: ContentHash) -> bool: ...

    async def add(self, artifact: RawArtifact) -> None: ...


class EventLog(Protocol):
    """The append-only ingestion log.

    There is no `update` and no `delete`, and their absence is the point: the
    log is what `memories` and `memory_chunks` are replayed from, so a rewritten
    event would silently change history that has already been projected.
    """

    async def append(self, event: IngestionEvent) -> None: ...

    async def replay(self, after_seq: int = 0, limit: int = 1000) -> Sequence[IngestionEvent]: ...


class MemoryRepository(Protocol):
    async def get_current(self, source_id: UUID, external_key: str) -> Memory | None: ...

    async def add_version(self, memory: Memory) -> None: ...

    async def tombstone(self, memory_id: UUID) -> None: ...
