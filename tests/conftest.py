import os
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
from memoryos.application.auth import CreateUser
from memoryos.config import Settings, get_settings

# The account the signed-in `client` fixture uses. Not a secret and not
# pretending to be one: it exists only inside a database that is truncated
# between tests, and the suite has to be able to log in without a human.
TEST_ACCOUNT_EMAIL = "tester@example.invalid"
TEST_ACCOUNT_PASSWORD = "correct horse battery staple"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `clean_database` truncates every table, because truncation is the only
# isolation strategy that survives code under test committing its own
# transactions. So the one thing that must never happen is the suite reaching the
# development database — and it did, three times during M2.0a, each time costing
# a full re-ingest and re-embed of the corpus.
#
# Set here, in the rootdir conftest, which pytest imports before any test module
# and before anything constructs `Settings`. Settings resolves `database_url` to
# `test_database_url` under this value, so every consumer — the fixtures below,
# the app factory, Alembic's env.py — lands on `memos_test` without being told
# separately. `setdefault`, so an explicit environment still wins.
os.environ.setdefault("MEMOS_ENVIRONMENT", "test")
get_settings.cache_clear()


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


async def sign_in(app: object, http_client: AsyncClient) -> None:
    """Create the account on `app` and log `http_client` into it.

    Exported because two tests build their own application — they need
    non-default settings — and a signed-in client is now a precondition for
    reaching any route. Without this they would each grow their own copy of the
    login, which is three places for the fixture's account to drift from the
    suite's.
    """
    container = app.state.container  # type: ignore[attr-defined]
    async with container.database.session_factory.begin() as db:
        await CreateUser(db, container.password_hasher)(
            TEST_ACCOUNT_EMAIL, TEST_ACCOUNT_PASSWORD
        )
    response = await http_client.post(
        "/auth/login",
        json={"email": TEST_ACCOUNT_EMAIL, "password": TEST_ACCOUNT_PASSWORD},
    )
    assert response.status_code == 200, response.text


@pytest.fixture
async def anonymous_client(
    settings: Settings, clean_database: None
) -> AsyncIterator[AsyncClient]:
    """The app with no session. What an unauthenticated caller sees.

    M11.0 made every route but `/health/*` and `/auth/*` require a session, so
    this is the fixture that can still observe a 401 — and the one the auth
    tests drive, because they are about obtaining a session rather than having
    one.
    """
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
    ):
        yield http_client


@pytest.fixture
async def client(
    settings: Settings, clean_database: None
) -> AsyncIterator[AsyncClient]:
    """The app, signed in.

    **M11.0 changed what this fixture means and deliberately not what it is
    called.** Every API test in this suite was written against an open API; auth
    is now global, so without a session all of them would fail with 401 and none
    of them would be testing what they were written to test. Creating an account
    and logging in here keeps every existing assertion honest — and makes them
    stronger, because they now also demonstrate that a signed-in caller reaches
    each route.

    The login goes through `/auth/login` rather than forging a cookie, so the
    fixture exercises the same path a browser does. A test that wants to see the
    gate refuse somebody uses `anonymous_client`.
    """
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client,
    ):
        container = app.state.container
        async with container.database.session_factory.begin() as db:
            await CreateUser(db, container.password_hasher)(
                TEST_ACCOUNT_EMAIL, TEST_ACCOUNT_PASSWORD
            )
        response = await http_client.post(
            "/auth/login",
            json={"email": TEST_ACCOUNT_EMAIL, "password": TEST_ACCOUNT_PASSWORD},
        )
        assert response.status_code == 200, response.text
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
