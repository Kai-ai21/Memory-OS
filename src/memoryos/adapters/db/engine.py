from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Imported for its side effect: the module registers the `after_begin` listener
# that sets `app.current_user_id` on every transaction. Imported here because
# every session in this application is built from this module, so there is no
# way to obtain one without having loaded the thing that scopes it.
from memoryos.adapters.db import scoping as _scoping  # noqa: F401


@dataclass(slots=True)
class Database:
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]

    @classmethod
    def from_url(cls, url: str, *, echo: bool = False) -> "Database":
        engine = create_async_engine(url, echo=echo, pool_pre_ping=True)
        return cls(
            engine=engine,
            session_factory=async_sessionmaker(engine, expire_on_commit=False),
        )

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()
