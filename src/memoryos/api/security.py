"""The gate: a valid session on every route, unless a route says otherwise.

**Applied globally and opted out of, never opted in to.** The dependency is
attached to the `FastAPI` application, so a route added tomorrow is protected
before anybody thinks about it. Opt-in auth fails in one direction — the
endpoint somebody forgets is the endpoint that leaks — and this fails in the
other, where the symptom is a 401 on something that should be public and you
find out in the first minute.

The opt-outs are two prefixes and they are listed here rather than scattered as
decorators, so "what is public" is one list you can read:

  * `/health/*` — an orchestrator has no session and needs to know whether this
    instance is alive. It reveals whether Postgres and Neo4j are reachable and
    nothing about the corpus.
  * `/auth/*` — you cannot require a session to obtain one.

Everything FastAPI mounts for itself is also open: `/docs`, `/redoc`,
`/openapi.json` and the OPTIONS preflights CORS depends on. The schema is the
shape of the API, not its contents, and `make types` reads it without a session.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status

from memoryos.adapters.db import models
from memoryos.application import auth
from memoryos.container import Container

# The cookie's name. Underscore-prefixed and `__Host-`-less deliberately: the
# `__Host-` prefix requires `Secure`, which is unavailable over plain HTTP on
# localhost, and a name that only works in one of the two deployments is a name
# that gets changed under pressure.
SESSION_COOKIE = "memos_session"

PUBLIC_PREFIXES: tuple[str, ...] = ("/health", "/auth")

# FastAPI's own, plus the trailing-slash and preflight cases.
_OPEN_PATHS: frozenset[str] = frozenset(
    {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}
)


def is_public(path: str) -> bool:
    """Whether a path is reachable without a session.

    Prefix matching is on a path *segment*, not on a string: `startswith` alone
    would make `/healthy-corpus` public because it happens to begin with
    `/health`, which is exactly the class of mistake this file exists to avoid.
    """
    if path in _OPEN_PATHS:
        return True
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in PUBLIC_PREFIXES)


class Unauthenticated(HTTPException):
    """401 with the one message every failure here uses."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not signed in.",
        )


async def require_session(request: Request) -> models.User | None:
    """Reject anything without a live session, except on a public path.

    Returns the user so a route can depend on this directly and be handed one;
    returns `None` on public paths, where there may not be one. The return value
    is not what does the work — raising is — which is why the type is optional
    rather than the signature lying about what a public route gets.
    """
    if is_public(request.url.path) or request.method == "OPTIONS":
        return None

    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise Unauthenticated()

    container: Container = request.app.state.container
    async with container.database.session_factory() as db:
        found = await auth.active_session(db, token)
        if found is None:
            raise Unauthenticated()
        _, user = found
        # Stashed so a route that wants the user does not repeat the lookup.
        request.state.user = user
        return user


def current_user(request: Request) -> models.User:
    """The authenticated user, for a route that needs to name them.

    Reads what `require_session` already resolved rather than querying again.
    Raising here would mean the global gate had not run, which is a wiring bug
    rather than an authentication failure — but it raises 401 anyway, because
    the alternative is a 500 that leaks that the route exists.
    """
    user: models.User | None = getattr(request.state, "user", None)
    if user is None:
        raise Unauthenticated()
    return user


CurrentUser = Annotated[models.User, Depends(current_user)]


def set_session_cookie(
    response: Response, token: str, *, secure: bool, max_age_seconds: int
) -> None:
    """Write the session cookie.

    `HttpOnly` is the one that matters: JavaScript cannot read the value, so an
    XSS bug on any page of this application cannot exfiltrate the session. That
    is also why nothing in the frontend stores a token anywhere — there is
    nothing for it to store.

    `SameSite=Lax` stops another origin's form post from carrying the cookie,
    which is CSRF for every state-changing route at once. Lax rather than Strict
    so that following a link back into the app arrives signed in.

    `Secure` is set whenever this is not a local deployment. It cannot be set
    unconditionally: browsers drop `Secure` cookies on plain HTTP, so a
    hardcoded `True` would make localhost unable to log in at all.
    """
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=secure,
        max_age=max_age_seconds,
        path="/",
    )


def clear_session_cookie(response: Response, *, secure: bool) -> None:
    """Delete the cookie, with the attributes it was set with.

    The attributes have to match or the browser treats it as a different cookie
    and keeps the original — a logout that appears to work and does not.
    """
    response.delete_cookie(
        SESSION_COOKIE, httponly=True, samesite="lax", secure=secure, path="/"
    )
