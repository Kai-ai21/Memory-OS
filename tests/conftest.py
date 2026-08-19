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
from memoryos.adapters.db.scoping import CURRENT_USER_ID
from memoryos.api.app import create_app
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


async def give_the_fixture_account_a_password(app: object) -> None:
    """Make `clean_database`'s account loggable-in.

    The account itself is created by `clean_database` with a placeholder hash,
    because M11.1 needs *every* test to have a user in context and hashing a
    password with argon2 twelve hundred times to produce one that is never
    verified costs about two minutes of suite time. The API tests are the ones
    that actually log in, so they are the ones that pay for a real hash.
    """
    container = app.state.container  # type: ignore[attr-defined]
    async with container.database.session_factory.begin() as db:
        await db.execute(
            text("UPDATE users SET password_hash = :hash WHERE email = :email"),
            {
                "hash": container.password_hasher.hash(TEST_ACCOUNT_PASSWORD),
                "email": TEST_ACCOUNT_EMAIL,
            },
        )


async def sign_in(app: object, http_client: AsyncClient) -> None:
    """Create the account on `app` and log `http_client` into it.

    Exported because two tests build their own application — they need
    non-default settings — and a signed-in client is now a precondition for
    reaching any route. Without this they would each grow their own copy of the
    login, which is three places for the fixture's account to drift from the
    suite's.
    """
    await give_the_fixture_account_a_password(app)
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
        await give_the_fixture_account_a_password(app)
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
    """The application role, which is the point.

    M11.1 pointed `database_url` at `memos_app`, a role with DML and nothing
    else, because row-level security is skipped for superusers — so a suite
    that connected as the owner would pass every isolation test by not having
    any policies applied to it.
    """
    async_engine = create_async_engine(settings.database_url)
    try:
        yield async_engine
    finally:
        await async_engine.dispose()


@pytest.fixture
async def admin_engine(
    migrated_database: None, settings: Settings
) -> AsyncIterator[AsyncEngine]:
    """The owner, for the two things the application role cannot do.

    Truncating with `RESTART IDENTITY` needs ownership of the sequences, and
    creating an account writes a table the application is not the owner of.
    Both are test-harness operations rather than application ones, and giving
    `memos_app` the privileges to do them would weaken the role this milestone
    exists to introduce.
    """
    async_engine = create_async_engine(settings.database_admin_url)
    try:
        yield async_engine
    finally:
        await async_engine.dispose()


@pytest.fixture
async def clean_database(admin_engine: AsyncEngine) -> AsyncIterator[None]:
    """Empty every table before the test runs, then run as one account.

    **M11.1 added the second half and it is not optional.** Every scoped table
    now has a row-level policy comparing `user_id` against
    `app.current_user_id`, and a connection that never sets it sees nothing and
    can insert nothing. Without an account in context every one of the twelve
    hundred tests below would fail on an empty result or a NOT NULL violation —
    not because scoping is wrong, but because a test is a user too.

    Creating it here rather than in each test keeps the change to one fixture.
    Tests that need *two* users ask for `second_user`, which is the whole of
    `test_user_scoping.py`.

    One isolation strategy for the whole suite, on purpose. Until M1.3 there
    were two: repository tests rolled back an outer transaction while queue
    tests truncated, because the queue commits. Sync commits too, and a suite
    with two strategies produces order-dependent failures whose cause is
    invisible in the failing test — the damage was done by whatever ran before
    it.

    Truncation is the strategy that survives code under test committing, which
    is why it is the one that generalises.
    """
    await truncate_all(admin_engine)

    # Made directly rather than through `CreateUser`, which hashes a password
    # with argon2 — a hundred milliseconds per test, twelve hundred times, to
    # produce a hash nothing in most of these tests will ever verify.
    async with admin_engine.begin() as connection:
        owner = (
            await connection.execute(
                text(
                    "INSERT INTO users (id, email, password_hash) "
                    "VALUES (gen_random_uuid(), :email, 'x') RETURNING id"
                ),
                {"email": TEST_ACCOUNT_EMAIL},
            )
        ).scalar_one()

    token = CURRENT_USER_ID.set(owner)
    try:
        yield
    finally:
        CURRENT_USER_ID.reset(token)


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
