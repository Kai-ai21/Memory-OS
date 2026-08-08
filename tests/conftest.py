from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memoryos.adapters.db.models import Base
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
async def client(settings: Settings, clean_database: None) -> AsyncIterator[AsyncClient]:
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
async def clean_database(engine: AsyncEngine) -> AsyncIterator[None]:
    """Empty every table before the test runs.

    One isolation strategy for the whole suite, on purpose. Until M1.3 there
    were two: repository tests rolled back an outer transaction while queue
    tests truncated, because the queue commits. Sync commits too, and a suite
    with two strategies produces order-dependent failures whose cause is
    invisible in the failing test — the damage was done by whatever ran before
    it.

    Truncation is the strategy that survives code under test committing, which
    is why it is the one that generalises.
    """
    await truncate_all(engine)
    yield


async def truncate_all(engine: AsyncEngine) -> None:
    tables = ", ".join(f'"{table.name}"' for table in Base.metadata.sorted_tables)
    async with engine.begin() as connection:
        # RESTART IDENTITY so the event log's seq starts from 1 in every test
        # that asserts on ordering; CASCADE so foreign keys do not dictate the
        # order of a statement that is emptying everything anyway.
        await connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
async def sessions(
    engine: AsyncEngine, clean_database: None
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory over a freshly emptied database.

    Code under test owns its own transactions; nothing here wraps them.
    """
    yield async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def session(
    sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """A single session for tests that just need somewhere to write."""
    async with sessions() as db_session:
        yield db_session
