from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from memoryos.api.routes import health, search, sources
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
    app.include_router(health.router)
    app.include_router(sources.router)
    app.include_router(search.router)
    return app
