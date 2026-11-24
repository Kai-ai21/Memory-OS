"""What deployment relies on, checked exhaustively rather than by example (M12.3).

`test_api_auth.py` shows one protected route refusing an anonymous caller. A
deployment relies on *every* route doing so, including the one added next
month, so this walks the application's own route table.
"""

import re

import pytest
from httpx import ASGITransport, AsyncClient

from memoryos.api.app import create_app
from memoryos.api.security import is_public
from memoryos.config import Settings

pytestmark = pytest.mark.integration

#: Path parameters filled with something syntactically valid for any of them.
_PARAM = re.compile(r"\{[^}]+\}")
_FILLER = "01a00000-0000-7000-8000-000000000000"


async def test_every_non_public_route_refuses_an_anonymous_caller(
    anonymous_client: AsyncClient, settings: Settings
) -> None:
    # From the app's own description of itself: every path and method it
    # serves, however the routers were composed.
    paths = create_app(settings).openapi()["paths"]
    checked = 0
    for path, operations in paths.items():
        if is_public(path):
            continue
        concrete = _PARAM.sub(_FILLER, path)
        for method in operations:
            response = await anonymous_client.request(method.upper(), concrete)
            assert response.status_code == 401, f"{method.upper()} {path} -> {response.status_code}"
            checked += 1
    # The number is not the point; that it is not zero is.
    assert checked > 50


async def test_registration_is_not_there_unless_enabled(
    anonymous_client: AsyncClient,
) -> None:
    response = await anonymous_client.post(
        "/auth/register", json={"email": "someone@example.com", "password": "x" * 16}
    )
    # 404, not 403: a closed door should look like a wall.
    assert response.status_code == 404


async def test_production_publishes_no_api_description(settings: Settings) -> None:
    production = settings.model_copy(
        update={"environment": "production", "cors_origins": ["https://memo.example.com"]}
    )
    app = create_app(production)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as http:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert (await http.get(path)).status_code in (401, 404), path
