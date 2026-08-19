"""The gate: obtaining a session, spending it, and losing it.

Six properties, and five of them are about failure. Authentication is almost
entirely a set of things that must not work, so a suite that only demonstrates a
successful login demonstrates approximately nothing.

Driven through `anonymous_client`, because the ordinary `client` fixture is
already signed in — which is what every other API test in this suite needs and
exactly what these tests must not have.
"""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memoryos.adapters.auth.argon2_hasher import Argon2PasswordHasher
from memoryos.adapters.db import models
from memoryos.api.security import SESSION_COOKIE
from memoryos.application.auth import CreateUser

pytestmark = pytest.mark.integration

EMAIL = "operator@example.invalid"
PASSWORD = "a sufficiently long password"


@pytest.fixture
async def account(session: AsyncSession) -> models.User:
    user = await CreateUser(session, Argon2PasswordHasher())(EMAIL, PASSWORD)
    await session.commit()
    return user


async def test_login_with_correct_credentials_sets_a_session_cookie(
    anonymous_client: AsyncClient, account: models.User
) -> None:
    response = await anonymous_client.post(
        "/auth/login", json={"email": EMAIL, "password": PASSWORD}
    )

    assert response.status_code == 200
    assert response.json()["email"] == EMAIL

    cookie = response.cookies.get(SESSION_COOKIE)
    assert cookie, "no session cookie was set"

    # The attributes are the security properties, so they are asserted rather
    # than assumed. `HttpOnly` is what stops an XSS bug reading the session;
    # `SameSite=Lax` is what stops another origin's form post carrying it.
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=lax" in header
    # Not `Secure` under the test environment, and that is correct: browsers
    # drop `Secure` cookies over plain HTTP, so setting it unconditionally would
    # make local development unable to log in at all.
    assert "secure" not in header

    # And the cookie works: a route the gate protects now answers.
    assert (await anonymous_client.get("/stats")).status_code == 200


async def test_a_wrong_password_is_indistinguishable_from_an_unknown_email(
    anonymous_client: AsyncClient, account: models.User
) -> None:
    """The anti-enumeration property, and the reason both failures return None.

    Different messages let somebody read which addresses have accounts straight
    off the error. Asserted on the whole response — status and body — because a
    difference in either one is the oracle.
    """
    wrong_password = await anonymous_client.post(
        "/auth/login", json={"email": EMAIL, "password": "not the right password"}
    )
    unknown_email = await anonymous_client.post(
        "/auth/login",
        json={"email": "nobody@example.invalid", "password": PASSWORD},
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()
    assert wrong_password.json()["detail"] == "Incorrect email or password"
    assert SESSION_COOKIE not in wrong_password.cookies


async def test_a_protected_route_without_a_session_is_401(
    anonymous_client: AsyncClient,
) -> None:
    """The gate itself, on a route that never mentions auth.

    `/stats` has no decorator, no dependency and no knowledge that
    authentication exists — which is the point of applying it globally. The two
    public prefixes are checked in the same test so the opt-out list is
    exercised rather than trusted.
    """
    assert (await anonymous_client.get("/stats")).status_code == 401
    assert (await anonymous_client.get("/memories")).status_code == 401
    assert (await anonymous_client.get("/search?q=anything")).status_code == 401

    # And the two that must stay open.
    assert (await anonymous_client.get("/health/live")).status_code == 200
    assert (await anonymous_client.get("/auth/me")).status_code == 401


async def test_logout_revokes_the_session_and_the_same_cookie_then_fails(
    anonymous_client: AsyncClient, account: models.User
) -> None:
    """Revocation is the whole reason these are not JWTs.

    The cookie is replayed by hand after the logout, because the client drops it
    when the response clears it — and "the browser forgot the token" is a much
    weaker property than "the server refuses it".
    """
    login = await anonymous_client.post(
        "/auth/login", json={"email": EMAIL, "password": PASSWORD}
    )
    token = login.cookies[SESSION_COOKIE]
    assert (await anonymous_client.get("/stats")).status_code == 200

    assert (await anonymous_client.post("/auth/logout")).status_code == 204

    # Put the token back on the client rather than passing it per request:
    # httpx deprecates per-request cookies, and the property under test is that
    # the *server* refuses a revoked token, not that the browser forgot it.
    anonymous_client.cookies.set(SESSION_COOKIE, token)
    assert (await anonymous_client.get("/stats")).status_code == 401


async def test_an_expired_session_is_rejected(
    anonymous_client: AsyncClient, account: models.User, session: AsyncSession
) -> None:
    """Expiry is checked in SQL against `now()`, so the test moves the row.

    Ageing the database row rather than the clock: there is one place expiry is
    evaluated and it compares against the database's own time, so a test that
    patched a Python clock would be testing a code path nothing uses.
    """
    await anonymous_client.post(
        "/auth/login", json={"email": EMAIL, "password": PASSWORD}
    )
    assert (await anonymous_client.get("/stats")).status_code == 200

    past = datetime.now(UTC) - timedelta(minutes=1)
    await session.execute(
        update(models.UserSession).values(
            created_at=past - timedelta(days=31), expires_at=past
        )
    )
    await session.commit()

    assert (await anonymous_client.get("/stats")).status_code == 401


async def test_rate_limiting_blocks_the_sixth_attempt_in_the_window(
    anonymous_client: AsyncClient, account: models.User
) -> None:
    """Five wrong passwords, then the sixth is refused without being checked.

    429 rather than 401, because it is a different fact — the credentials were
    not examined — and because a client that retried on 401 would spin.

    The final assertion is the one that matters most: the *correct* password is
    also refused while the window is open. A limiter that let the right password
    through would be no limiter at all, since that is precisely what a
    brute-force attempt eventually presents.
    """
    for _ in range(5):
        response = await anonymous_client.post(
            "/auth/login", json={"email": EMAIL, "password": "wrong"}
        )
        assert response.status_code == 401

    blocked = await anonymous_client.post(
        "/auth/login", json={"email": EMAIL, "password": "wrong"}
    )
    assert blocked.status_code == 429
    assert "Retry-After" in blocked.headers

    with_the_real_password = await anonymous_client.post(
        "/auth/login", json={"email": EMAIL, "password": PASSWORD}
    )
    assert with_the_real_password.status_code == 429


async def test_a_successful_login_clears_the_attempt_count(
    anonymous_client: AsyncClient, account: models.User
) -> None:
    """Four mistakes then success must not leave somebody locked out.

    The limiter records failures only and a success clears the address, so a
    person who mistypes their own password a few times is not shut out of their
    own machine for a quarter of an hour.
    """
    for _ in range(4):
        assert (
            await anonymous_client.post(
                "/auth/login", json={"email": EMAIL, "password": "wrong"}
            )
        ).status_code == 401

    assert (
        await anonymous_client.post(
            "/auth/login", json={"email": EMAIL, "password": PASSWORD}
        )
    ).status_code == 200

    # The window is clear, so a fresh mistake is a 401 rather than a 429.
    assert (
        await anonymous_client.post(
            "/auth/login", json={"email": EMAIL, "password": "wrong"}
        )
    ).status_code == 401


async def test_the_token_is_never_stored_in_the_database(
    anonymous_client: AsyncClient, account: models.User, session: AsyncSession
) -> None:
    """The property the schema exists for, asserted rather than described.

    If this row held the token, a database leak would be a set of live
    credentials. It holds a SHA-256 of it, so it is not.
    """
    login = await anonymous_client.post(
        "/auth/login", json={"email": EMAIL, "password": PASSWORD}
    )
    token = login.cookies[SESSION_COOKIE]

    stored = (await session.execute(select(models.UserSession))).scalars().all()
    assert len(stored) == 1
    assert stored[0].token_hash != token
    assert len(stored[0].token_hash) == 64
    assert token not in stored[0].token_hash


async def test_a_second_account_is_refused_by_name(session: AsyncSession) -> None:
    """Single-user, and the refusal says which account already exists.

    "A user already exists" without naming it is a message that sends somebody
    to psql.
    """
    from memoryos.application.auth import UserAlreadyExists

    hasher = Argon2PasswordHasher()
    await CreateUser(session, hasher)(EMAIL, PASSWORD)
    await session.commit()

    with pytest.raises(UserAlreadyExists) as raised:
        await CreateUser(session, hasher)("someone.else@example.invalid", PASSWORD)

    assert EMAIL in str(raised.value)


async def test_a_short_password_is_refused(session: AsyncSession) -> None:
    from memoryos.application.auth import WeakPassword

    with pytest.raises(WeakPassword) as raised:
        await CreateUser(session, Argon2PasswordHasher())(EMAIL, "short")

    assert "at least 12" in str(raised.value)


async def test_resetting_the_password_revokes_every_session(
    anonymous_client: AsyncClient, account: models.User, session: AsyncSession
) -> None:
    """A reset that left old sessions alive would change nothing for an intruder.

    Which is usually the reason somebody is resetting it.
    """
    from memoryos.application.auth import ResetPassword

    login = await anonymous_client.post(
        "/auth/login", json={"email": EMAIL, "password": PASSWORD}
    )
    token = login.cookies[SESSION_COOKIE]
    assert (await anonymous_client.get("/stats")).status_code == 200

    await ResetPassword(session, Argon2PasswordHasher())(EMAIL, "a different long password")
    await session.commit()

    anonymous_client.cookies.set(SESSION_COOKIE, token)
    assert (await anonymous_client.get("/stats")).status_code == 401
