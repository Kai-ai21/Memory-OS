"""Accounts and sessions: the use cases, with no HTTP and no argon2 in them.

**Single-user, and the honesty about what that means lives here.** This module
authenticates one account against one password and issues opaque session tokens.
It does not scope any data — every query in the rest of this system still reads
the whole corpus, because there is only ever one person's corpus in it. That is
M11.1's job and it is not started.

What this buys is worth naming precisely: it stops somebody who is sitting at an
unlocked machine, or who can reach the port, from reading the corpus through the
UI. It is not multi-tenant isolation and nothing here should be read as if it
were.

Three rules that are easy to get wrong and are all in one place because of it:

  * **The same failure for an unknown email and a wrong password.** Different
    messages are an account-enumeration oracle: an attacker learns which
    addresses exist by reading the error. `authenticate` returns `None` for
    both, and the route turns that into one sentence.
  * **A password check runs even when the email is unknown.** Otherwise the
    response time answers the question the error message refused to.
  * **Expiry and revocation are checked together**, in `active_session`, so no
    caller can honour one and forget the other.
"""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memoryos.adapters.db import models
from memoryos.application.attachments import UPLOAD_SOURCE_NAME
from memoryos.application.chat import CHAT_SOURCE_NAME
from memoryos.application.ports import PasswordHasher
from memoryos.domain.ids import new_id
from memoryos.domain.values import SourceKind

# Long enough that a login is a rare event on a machine somebody uses daily, and
# short enough that a laptop lost and never noticed stops working.
SESSION_TTL = timedelta(days=30)

# Twelve, and the number is a floor rather than a policy. Composition rules
# ("one capital, one symbol") measurably push people towards `Password1!`; length
# is the only requirement that reliably buys entropy.
MIN_PASSWORD_LENGTH = 12

# The argon2 hash of nothing in particular, used to spend the same time on an
# unknown email as on a known one. Computed once at import rather than per
# request: the point is that the *verify* runs, not that the hash does.
_DECOY_PASSWORD = "a password that is not anybody's"


class WeakPassword(ValueError):
    """The password is shorter than `MIN_PASSWORD_LENGTH`."""


class EmailAlreadyRegistered(RuntimeError):
    """That address already has an account.

    **M11.1 replaced `UserAlreadyExists` with this**, and the change is the
    milestone in one class. M11.0 refused a *second account of any kind*,
    because nothing was scoped and a second account would have read the first
    one's corpus. Every row now has an owner and a policy that enforces it, so
    more than one account is safe — and the only thing left to refuse is two
    accounts claiming the same address.
    """

    def __init__(self, email: str) -> None:
        super().__init__(
            f"An account already exists for {email}. Use "
            f"`memoryos auth reset-password --email {email}` to change its password."
        )
        self.email = email


class NoSuchUser(RuntimeError):
    """A reset was asked for an address with no account."""


@dataclass(frozen=True, slots=True)
class IssuedSession:
    """What a login produces: the token to hand the browser, and when it dies.

    The token is in memory here and nowhere else — it is written to the response
    cookie and never to the database, which stores `sha256(token)`. This
    dataclass is the only place the plaintext exists after `secrets` returns it.
    """

    token: str
    session_id: UUID
    expires_at: datetime


def normalise_email(email: str) -> str:
    """Lowercased and stripped. The one spelling the database ever sees."""
    return email.strip().lower()


def hash_token(token: str) -> str:
    """SHA-256 of a session token, hex.

    Fast on purpose, and this is the one place in this milestone where fast is
    right: a 256-bit random token has no guessable structure, so the slow hash
    that protects a password would defend against nothing and cost tens of
    milliseconds on every authenticated request.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class CreateUser:
    """Make the account. Refuses to make a second one.

    The refusal names the address that exists, because "a user already exists"
    without saying which is a message that sends somebody to the database.
    """

    def __init__(self, session: AsyncSession, hasher: PasswordHasher) -> None:
        self._session = session
        self._hasher = hasher

    async def __call__(self, email: str, password: str) -> models.User:
        if len(password) < MIN_PASSWORD_LENGTH:
            raise WeakPassword(
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters; "
                f"that one is {len(password)}."
            )

        address = normalise_email(email)
        existing = (
            await self._session.execute(
                select(models.User).where(models.User.email == address)
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise EmailAlreadyRegistered(existing.email)

        user = models.User(
            id=new_id(),
            email=address,
            password_hash=self._hasher.hash(password),
        )
        self._session.add(user)
        await self._session.flush()
        return user


class ResetPassword:
    """Change the one account's password. Every session it had is revoked.

    Revoking is not optional politeness. A password is reset because it may be
    known to somebody else, and leaving the sessions it created alive means the
    reset changed nothing for whoever is already logged in.
    """

    def __init__(self, session: AsyncSession, hasher: PasswordHasher) -> None:
        self._session = session
        self._hasher = hasher

    async def __call__(self, email: str, password: str) -> models.User:
        if len(password) < MIN_PASSWORD_LENGTH:
            raise WeakPassword(
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters; "
                f"that one is {len(password)}."
            )

        address = normalise_email(email)
        user = (
            await self._session.execute(
                select(models.User).where(models.User.email == address)
            )
        ).scalar_one_or_none()
        if user is None:
            raise NoSuchUser(f"No account for {address}.")

        user.password_hash = self._hasher.hash(password)
        await self._session.execute(
            update(models.UserSession)
            .where(
                models.UserSession.user_id == user.id,
                models.UserSession.revoked_at.is_(None),
            )
            .values(revoked_at=datetime.now(UTC))
        )
        await self._session.flush()
        return user


class AuthenticateUser:
    """Email and password in, a user or `None` out.

    **`None` covers both failures on purpose.** An unknown address and a wrong
    password are indistinguishable to the caller, which is what stops the login
    route from becoming a way to enumerate accounts.
    """

    def __init__(self, session: AsyncSession, hasher: PasswordHasher) -> None:
        self._session = session
        self._hasher = hasher

    async def __call__(self, email: str, password: str) -> models.User | None:
        user = (
            await self._session.execute(
                select(models.User).where(models.User.email == normalise_email(email))
            )
        ).scalar_one_or_none()

        if user is None:
            # Spend the time anyway. Without this the response is fast for an
            # unknown address and slow for a known one, and the timing answers
            # the question the identical error message refused to.
            self._hasher.verify(self._decoy_hash(), password)
            return None

        if not self._hasher.verify(user.password_hash, password):
            return None

        user.last_login_at = datetime.now(UTC)
        await self._session.flush()
        return user

    def _decoy_hash(self) -> str:
        # Hashed per call rather than cached, because caching it would make the
        # unknown-email path cheaper than the known one again — a hash and a
        # verify is closer to the real cost than a verify alone.
        return self._hasher.hash(_DECOY_PASSWORD)


class IssueSession:
    """Mint a session token for a user and store only its hash."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __call__(
        self, user: models.User, *, user_agent: str | None = None
    ) -> IssuedSession:
        # 32 bytes from the OS CSPRNG, URL-safe so it survives a cookie value
        # without encoding. Not a JWT: this system has one server and a
        # database, so a stateless token buys nothing and costs the ability to
        # revoke — which is the whole of what logout is.
        token = secrets.token_urlsafe(32)
        now = datetime.now(UTC)
        record = models.UserSession(
            id=new_id(),
            user_id=user.id,
            token_hash=hash_token(token),
            created_at=now,
            expires_at=now + SESSION_TTL,
            # Truncated rather than validated. It is a client-controlled string
            # kept for a human to read, and a very long one is a way to write a
            # lot of rows into a table nobody watches.
            user_agent=(user_agent or None) and user_agent[:400],
        )
        self._session.add(record)
        await self._session.flush()
        return IssuedSession(token=token, session_id=record.id, expires_at=record.expires_at)


async def active_session(
    session: AsyncSession, token: str
) -> tuple[models.UserSession, models.User] | None:
    """The session and its user, if the token names one that is still good.

    **The one place expiry and revocation are checked**, so no caller can honour
    one and forget the other. Both are compared in SQL against `now()` rather
    than in Python against a clock the caller passed, because two clocks is two
    answers.
    """
    now = datetime.now(UTC)
    row = (
        await session.execute(
            select(models.UserSession, models.User)
            .join(models.User, models.User.id == models.UserSession.user_id)
            .where(
                models.UserSession.token_hash == hash_token(token),
                models.UserSession.revoked_at.is_(None),
                models.UserSession.expires_at > now,
            )
        )
    ).first()
    if row is None:
        return None
    return row[0], row[1]


async def revoke_session(session: AsyncSession, token: str) -> bool:
    """Mark the session revoked. True if there was a live one to revoke.

    Idempotent: logging out twice, or with a token that expired in between, is
    not an error. The caller clears the cookie either way.
    """
    result = await session.execute(
        update(models.UserSession)
        .where(
            models.UserSession.token_hash == hash_token(token),
            models.UserSession.revoked_at.is_(None),
        )
        .values(revoked_at=datetime.now(UTC))
        # Returning the id rather than reading `rowcount`: the attribute exists
        # on the cursor result at runtime but not on the type SQLAlchemy
        # declares for `execute`, and a `cast` to reach it would be asserting
        # something about the driver that this can simply ask for instead.
        .returning(models.UserSession.id)
    )
    return result.first() is not None


async def create_default_sources(session: AsyncSession, user_id: UUID) -> None:
    """Give a new account the sources it needs to be usable.

    **A new user sees an empty system, not somebody else's**, and "empty" here
    has to mean *usable and empty* rather than *broken*. Two singletons: `chat`,
    which every message you type is filed under, and `uploads`, which every
    attachment is. Both are created lazily elsewhere with `ON CONFLICT DO
    NOTHING` and would appear on first use anyway — doing it here means a new
    account's `sources` page is correct before it is touched rather than after.

    Written inside `scoped_to(user_id)` by the caller, so the row lands with the
    right owner through the same column default every other insert uses. The
    explicit `user_id` here is belt and braces: this is the one insert in the
    codebase that runs while creating the user it belongs to, and being wrong
    about it is a row nobody can see.
    """
    defaults = (
        (SourceKind.CHAT, CHAT_SOURCE_NAME),
        (SourceKind.UPLOAD, UPLOAD_SOURCE_NAME),
    )
    for kind, name in defaults:
        session.add(
            models.Source(
                id=new_id(),
                user_id=user_id,
                kind=kind.value,
                name=name,
                config={},
                cursor={},
            )
        )
    await session.flush()
