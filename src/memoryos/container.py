"""Composition root.

One place assembles the object graph, so the CLI, the API, and the worker all
run against the same wiring rather than three slightly different versions of it.
"""

from dataclasses import dataclass

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db.engine import Database
from memoryos.adapters.db.job_queue import PostgresJobQueue
from memoryos.application.jobs.handlers import build_default_registry
from memoryos.application.jobs.registry import HandlerRegistry
from memoryos.application.sync import SyncSource
from memoryos.config import Settings


@dataclass(slots=True)
class Container:
    settings: Settings
    database: Database
    blobs: FilesystemBlobStore
    connector: FilesystemConnector
    queue: PostgresJobQueue

    @classmethod
    def build(cls, settings: Settings) -> "Container":
        database = Database.from_url(settings.database_url, echo=settings.db_echo)
        blobs = FilesystemBlobStore(settings.blob_root)
        return cls(
            settings=settings,
            database=database,
            blobs=blobs,
            connector=FilesystemConnector(blobs),
            queue=PostgresJobQueue(database.session_factory),
        )

    def registry(self) -> HandlerRegistry:
        return build_default_registry(
            session_factory=self.database.session_factory,
            connector=self.connector,
            blob_store=self.blobs,
        )

    def sync(self) -> SyncSource:
        return SyncSource(self.database.session_factory, self.connector, self.blobs)

    async def dispose(self) -> None:
        await self.database.dispose()
