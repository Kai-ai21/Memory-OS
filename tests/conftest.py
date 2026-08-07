from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from memoryos.api.app import create_app
from memoryos.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as http_client:
            yield http_client
