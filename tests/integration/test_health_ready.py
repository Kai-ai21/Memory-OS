from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from memoryos.api.app import create_app
from memoryos.config import Settings

pytestmark = pytest.mark.integration

# Nothing listens here. Port 1 is privileged and unused, so the connection is
# refused immediately rather than timing out — which keeps this test fast and
# keeps it testing the degraded path rather than the driver's timeout.
NOWHERE = "bolt://localhost:1"


@pytest.fixture
async def client_without_graph(
    settings: Settings, clean_database: None
) -> AsyncIterator[AsyncClient]:
    """An app whose Neo4j does not exist."""
    broken = settings.model_copy(update={"neo4j_uri": NOWHERE})
    app = create_app(broken)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        yield http


async def test_readiness_reports_pgvector(client: AsyncClient) -> None:
    response = await client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] is True
    assert body["pgvector_version"] is not None


async def test_an_unreachable_graph_is_degraded_and_not_a_failure(
    client_without_graph: AsyncClient,
) -> None:
    """The milestone's operational rule, asserted on the status code.

    A 503 here would be the wrong kind of correct. The body would accurately say
    the graph is down, and an orchestrator reading the code would take the
    instance out of rotation — so an outage of a projection that only M3.1
    onwards reads would stop ingestion, search and answering, none of which
    touch it. The status code answers "should traffic come here?", and the
    answer is still yes.

    Hence 200 with `status: degraded`. The two together are what "degraded, not
    failed" has to mean for it to be worth anything.
    """
    response = await client_without_graph.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["graph"] is False
    # And the half that must keep working is reported as working.
    assert body["database"] is True
    assert body["pgvector_version"] is not None


@pytest.mark.graph
async def test_readiness_is_ok_when_the_graph_is_reachable(client: AsyncClient) -> None:
    """The other side of the same check.

    Without this, `graph: false` in every response would pass the test above
    forever — including if the probe were wired to a function that always
    returned False.
    """
    response = await client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["graph"] is True
    assert body["status"] == "ok"
