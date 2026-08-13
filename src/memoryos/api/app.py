from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from memoryos.api.routes import (
    answer,
    decisions,
    events,
    evolution,
    health,
    judgements,
    memories,
    search,
    sources,
    stats,
    timeline,
)
from memoryos.config import Settings, get_settings
from memoryos.container import Container
from memoryos.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = Container.build(resolved)
        app.state.settings = resolved
        app.state.container = container
        # Kept for the health routes, which predate the container.
        app.state.db = container.database
        try:
            yield
        finally:
            await container.dispose()

    app = FastAPI(title="Memory Intelligence OS", version="0.1.0", lifespan=lifespan)
    _install_cors(app, resolved)
    app.include_router(health.router)
    app.include_router(sources.router)
    app.include_router(search.router)
    # Before `memories`, and it has to be. Routes are matched in registration
    # order, so `/memories/{memory_id}` would claim `/memories/at` first and
    # answer 422 for a UUID that was never meant to be one. A literal path
    # segment must be registered ahead of the parameterised one that shadows it.
    app.include_router(timeline.router)
    # Also before `memories`: `/memories/{memory_id}/evolution` is more specific
    # than `/memories/{memory_id}` but FastAPI matches in registration order, not
    # by specificity, and the two do not collide only because their path lengths
    # differ. Registered here so the grouping stays obvious rather than lucky.
    app.include_router(evolution.router)
    app.include_router(memories.router)
    app.include_router(stats.router)
    app.include_router(judgements.router)
    # `/decisions/suggestions` is registered inside this router *before*
    # `/decisions/{decision_id}`, for the same registration-order reason
    # `timeline` sits above `memories`.
    app.include_router(decisions.router)
    app.include_router(answer.router)
    # Phase 6's one endpoint, and the only route in this API that something
    # other than a person calls. No path collision with anything above it.
    app.include_router(events.router)
    return app


class WildcardOrigin(RuntimeError):
    """CORS was configured to allow any origin."""


def _install_cors(app: FastAPI, settings: Settings) -> None:
    """Allow the configured browser origins, and nothing if there are none.

    No middleware at all when the list is empty, which is the default. That is
    deliberately different from "middleware allowing nothing": a deployment that
    never sets this has no CORS surface to reason about.

    A wildcard is refused rather than documented against. This API answers
    questions about a private corpus, so `*` means any page the operator happens
    to visit can read it — and `allow_credentials` with `*` is exactly the
    combination browsers reject anyway. Failing at startup makes that a
    five-second problem instead of an incident.
    """
    if not settings.cors_origins:
        return

    if any(origin.strip() == "*" for origin in settings.cors_origins):
        raise WildcardOrigin(
            "MEMOS_CORS_ORIGINS contains '*'. This API reads a private corpus; "
            "name the origins that may call it, e.g. http://localhost:5173."
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        # The UI sends no cookies and no Authorization header — there is no auth
        # in this milestone — so credentials stay off. Turning them on later is a
        # decision to make alongside whatever introduces auth.
        allow_credentials=False,
        # PATCH joins the list in M5.0: editing a decision is the first write in
        # this API that amends a row rather than creating one, and a browser
        # preflight for a method not named here fails before the request is made.
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type"],
    )
