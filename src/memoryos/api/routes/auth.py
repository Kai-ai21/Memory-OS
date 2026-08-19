"""Login, logout, and who am I.

Three endpoints, and most of the care is in what they refuse to say.

**One error for both login failures.** An unknown address and a wrong password
return the same 401 with the same sentence. Anything else is an oracle: an
attacker reads which addresses have accounts straight off the error, and on a
single-user system that is most of the secret.

**Rate limited per IP.** Five attempts in fifteen minutes. Without it a password
is brute-forceable at whatever rate the network allows, and argon2's slowness is
the only thing standing in the way — which is a defence measured in attempts per
second rather than in attempts.

The limiter is in-process. That is the correct scope for this deployment and the
wrong one for any other: two workers means two independent counters and ten
attempts per window, so this is documented in the README rather than silently
assumed. A system with one uvicorn process and one operator does not need Redis
to hold five integers.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from memoryos.api.security import (
    SESSION_COOKIE,
    clear_session_cookie,
    set_session_cookie,
)
from memoryos.application import auth
from memoryos.container import Container

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

MAX_ATTEMPTS = 5
WINDOW_SECONDS = 15 * 60


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


@dataclass
class LoginRateLimiter:
    """Five attempts per fifteen minutes, per client address.

    A sliding window of timestamps rather than a counter with a reset, because a
    fixed window lets ten attempts through across a boundary — five at 14:59 and
    five at 15:01 — which is exactly the number the limit was chosen to prevent.

    Only *failures* are recorded. Succeeding clears the address, so somebody who
    mistypes four times and then gets it right is not locked out of their own
    machine for a quarter of an hour.
    """

    attempts: dict[str, deque[float]] = field(default_factory=dict)

    def _prune(self, key: str, now: float) -> deque[float]:
        window = self.attempts.setdefault(key, deque())
        while window and now - window[0] > WINDOW_SECONDS:
            window.popleft()
        return window

    def blocked(self, key: str, *, now: float | None = None) -> bool:
        return len(self._prune(key, now or time.monotonic())) >= MAX_ATTEMPTS

    def record_failure(self, key: str, *, now: float | None = None) -> None:
        moment = now or time.monotonic()
        self._prune(key, moment).append(moment)

    def clear(self, key: str) -> None:
        self.attempts.pop(key, None)

    def retry_after(self, key: str, *, now: float | None = None) -> int:
        moment = now or time.monotonic()
        window = self._prune(key, moment)
        if not window:
            return 0
        return max(1, int(WINDOW_SECONDS - (moment - window[0])))


def get_limiter(request: Request) -> LoginRateLimiter:
    limiter: LoginRateLimiter = request.app.state.login_limiter
    return limiter


LimiterDep = Annotated[LoginRateLimiter, Depends(get_limiter)]


def client_key(request: Request) -> str:
    """Which client an attempt is counted against.

    `request.client.host` and nothing else. `X-Forwarded-For` is deliberately
    not read: this API is not behind a proxy in the deployment it is written
    for, and trusting a client-settable header would let anybody reset their own
    limit by varying one string — a rate limiter that is worse than none,
    because it looks like one.
    """
    return request.client.host if request.client else "unknown"


class LoginIn(BaseModel):
    # `str`, not `EmailStr`. Two reasons and neither is laziness: `EmailStr`
    # needs the `email-validator` package for one form field on a single-user
    # system, and it answers 422 for a malformed address where every other
    # login failure answers 401 — a third response shape on the one endpoint
    # whose whole design is that it has two.
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    created_at: datetime
    last_login_at: datetime | None


@router.post("/login", response_model=UserOut)
async def login(
    body: LoginIn,
    request: Request,
    response: Response,
    container: ContainerDep,
    limiter: LimiterDep,
) -> UserOut:
    key = client_key(request)
    if limiter.blocked(key):
        # 429 rather than 401. It is a different fact — the credentials were not
        # examined — and a client that retried on 401 would spin forever.
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many sign-in attempts. Try again later.",
            headers={"Retry-After": str(limiter.retry_after(key))},
        )

    async with container.database.session_factory() as db:
        user = await auth.AuthenticateUser(db, container.password_hasher)(
            body.email, body.password
        )
        if user is None:
            limiter.record_failure(key)
            await db.rollback()
            logger.info("auth.login.failed", client=key)
            # The same sentence for an unknown address and a wrong password.
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
            )

        issued = await auth.IssueSession(db)(
            user, user_agent=request.headers.get("user-agent")
        )
        out = UserOut(
            id=str(user.id),
            email=user.email,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )
        await db.commit()

    limiter.clear(key)
    set_session_cookie(
        response,
        issued.token,
        secure=container.settings.session_cookie_secure,
        max_age_seconds=int(auth.SESSION_TTL.total_seconds()),
    )
    logger.info("auth.login.ok", user_id=str(out.id))
    return out


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request, response: Response, container: ContainerDep
) -> Response:
    """Revoke the session and clear the cookie.

    Unauthenticated on purpose — it is under `/auth`, so the global gate does not
    run on it. Logging out with a cookie that already expired has to succeed:
    the client's goal is to end up signed out, and answering 401 would leave the
    cookie in the browser and the user staring at a login page that says they
    are still signed in.
    """
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        async with container.database.session_factory() as db:
            await auth.revoke_session(db, token)
            await db.commit()

    clear_session_cookie(response, secure=container.settings.session_cookie_secure)
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers=response.headers)


@router.get("/me", response_model=UserOut)
async def me(request: Request, container: ContainerDep) -> UserOut:
    """The signed-in account, or 401.

    Under `/auth` and therefore outside the global gate, so it does its own
    check — which is what makes it usable as the frontend's "am I signed in?"
    probe without a 401 from the gate being indistinguishable from a 401 from
    here.
    """
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not signed in."
        )

    async with container.database.session_factory() as db:
        found = await auth.active_session(db, token)
        if found is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Not signed in."
            )
        _, user = found
        return UserOut(
            id=str(user.id),
            email=user.email,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )
