import pytest

pytestmark = pytest.mark.integration


async def test_readiness_reports_pgvector(client):
    response = await client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["database"] is True
    assert body["pgvector_version"] is not None
