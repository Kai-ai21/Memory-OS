from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from memoryos.adapters.db.engine import Database

router = APIRouter(prefix="/health", tags=["health"])


def get_db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


class Liveness(BaseModel):
    status: Literal["ok"]


class Readiness(BaseModel):
    status: Literal["ok", "degraded"]
    database: bool
    pgvector_version: str | None


@router.get("/live", response_model=Liveness)
async def live() -> Liveness:
    return Liveness(status="ok")


@router.get("/ready", response_model=Readiness)
async def ready(
    response: Response, db: Annotated[Database, Depends(get_db)]
) -> Readiness:
    try:
        async with db.session_factory() as session:
            result = await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            version: str | None = result.scalar_one_or_none()
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Readiness(status="degraded", database=False, pgvector_version=None)

    if version is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return Readiness(status="degraded", database=True, pgvector_version=None)

    return Readiness(status="ok", database=True, pgvector_version=version)
