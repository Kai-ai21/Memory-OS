from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from memoryos.api.app import create_app
from memoryos.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def alembic_config() -> Config:
    # env.py reads the URL from Settings, so nothing needs to be injected here.
    return Config(str(PROJECT_ROOT / "alembic.ini"))


def run_alembic(action: Callable[[Config, str], None], revision: str) -> None:
    """Run an Alembic command off the current thread.

    Alembic's async env.py calls `asyncio.run`, which refuses to start inside a
    thread that already has a running loop. Tests do, so the command gets its
    own thread.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(action, alembic_config(), revision).result()


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
    ):
        yield http_client


@pytest.fixture(scope="session")
def migrated_database() -> None:
    """Bring the test database up to head once per session."""
    run_alembic(command.upgrade, "head")


@pytest.fixture
async def engine(migrated_database: None, settings: Settings) -> AsyncIterator[AsyncEngine]:
    async_engine = create_async_engine(settings.database_url)
    try:
        yield async_engine
    finally:
        await async_engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """A session whose work is rolled back when the test ends.

    The outer transaction is never committed, so tests leave no rows behind and
    do not depend on the order they run in. `create_savepoint` lets a repository
    or a test commit without escaping that outer transaction.
    """
    async with engine.connect() as connection:
        transaction = await connection.begin()
        db_session = AsyncSession(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        try:
            yield db_session
        finally:
            await db_session.close()
            await transaction.rollback()
