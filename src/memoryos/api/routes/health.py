from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from memoryos.adapters.db.engine import Database
from memoryos.adapters.graph.neo4j_store import Neo4jGraphStore

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


def get_db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


def get_graph(request: Request) -> Neo4jGraphStore:
    graph: Neo4jGraphStore = request.app.state.container.graph
    return graph


class Liveness(BaseModel):
    status: Literal["ok"]


class Readiness(BaseModel):
    """What this instance can currently do.

    `status` and the HTTP code answer two different questions, and conflating
    them is what makes a graph outage into an availability incident. The status
    code answers "should traffic come here?"; the body answers "what works?".

    So Postgres unreachable is a 503 — nothing works without it — while Neo4j
    unreachable is a 200 with `status: degraded`. The graph is a projection that
    only M3.1 onwards reads, and returning 503 for it would have an orchestrator
    remove an instance that can still serve every Phase 1 and Phase 2 request:
    ingestion, search, and answering would all go down to protect a feature none
    of them use.
    """

    status: Literal["ok", "degraded"]
    database: bool
    pgvector_version: str | None
    # False when the graph is unreachable. Not fatal; see above.
    graph: bool


@router.get("/live", response_model=Liveness)
async def live() -> Liveness:
    return Liveness(status="ok")


@router.get("/ready", response_model=Readiness)
async def ready(
    response: Response,
    db: Annotated[Database, Depends(get_db)],
    graph_store: Annotated[Neo4jGraphStore, Depends(get_graph)],
) -> Readiness:
    graph_up = await _graph_reachable(graph_store)

    try:
        async with db.session_factory() as session:
            result = await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            version: str | None = result.scalar_one_or_none()
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Readiness(
            status="degraded", database=False, pgvector_version=None, graph=graph_up
        )

    if version is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Readiness(
            status="degraded", database=True, pgvector_version=None, graph=graph_up
        )

    # The one place a degraded status is returned with a 200. Everything Phase 1
    # and Phase 2 serve is working; the graph is not.
    return Readiness(
        status="ok" if graph_up else "degraded",
        database=True,
        pgvector_version=version,
        graph=graph_up,
    )


async def _graph_reachable(graph_store: Neo4jGraphStore) -> bool:
    """Connectivity only, and no schema write.

    A readiness probe runs on a timer, so it must not be the thing that applies
    a schema — it would turn a repeated health check into repeated writes, and
    make the check's result depend on whether it had run before.
    """
    try:
        await graph_store.verify()
    except Exception as exc:
        logger.info("health.graph_unreachable", error=str(exc))
        return False
    return True
