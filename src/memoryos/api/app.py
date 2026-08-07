from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from memoryos.adapters.db.engine import Database
from memoryos.api.routes import health
from memoryos.config import Settings, get_settings
from memoryos.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database.from_url(resolved.database_url, echo=resolved.db_echo)
        app.state.settings = resolved
        app.state.db = database
        try:
            yield
        finally:
            await database.dispose()

    app = FastAPI(title="Memory Intelligence OS", version="0.1.0", lifespan=lifespan)
    app.include_router(health.router)
    return app
