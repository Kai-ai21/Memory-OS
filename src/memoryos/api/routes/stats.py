"""Corpus health over HTTP.

Both endpoints call the same functions the CLI does — `gather_stats` and
`run_doctor` — rather than reimplementing the queries. Two implementations of
"how many chunks are embedded" is how a dashboard ends up reassuring you while
`memoryos doctor` says otherwise.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from memoryos.application.backfill import gather_stats
from memoryos.application.doctor import run_doctor
from memoryos.container import Container

router = APIRouter(tags=["stats"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class StatsOut(BaseModel):
    memories: int
    current_memories: int
    chunks: int
    embedded_chunks: int
    cache_entries: int
    coverage: float
    models: dict[str, int]
    # Phase 3's half of the corpus. Here rather than on a graph endpoint because
    # the overview asks one question — how big is this thing — and answering it
    # from two requests is how the two halves end up measured at different
    # instants and quietly disagreeing.
    entities: int
    relationships: int
    # What the corpus was built with, so a number on screen can be attributed to
    # a model and a chunker rather than floating free.
    embedding_model: str
    chunker_version: str
    model_window: int


class FindingOut(BaseModel):
    check: str
    count: int
    detail: str
    examples: list[str]
    healthy: bool
    # A non-zero advisory is neither a pass nor a failure — it names a
    # capability nobody has exercised. Carried to the client so the corpus page
    # can render it as a note rather than as a green tick beside a real gap.
    advisory: bool = False


class DoctorOut(BaseModel):
    healthy: bool
    findings: list[FindingOut]
    embedding_model: str
    chunker_version: str
    model_window: int


@router.get("/stats", response_model=StatsOut)
async def get_stats(container: ContainerDep) -> StatsOut:
    # Per-source counts live on `GET /sources`, which already had to group by
    # source to report them. Computing them here too would be a second
    # implementation of the same number, which is how a dashboard ends up
    # disagreeing with the endpoint next to it.
    stats = await gather_stats(container.database.session_factory)

    return StatsOut(
        memories=stats.memories,
        current_memories=stats.current_memories,
        chunks=stats.chunks,
        embedded_chunks=stats.embedded_chunks,
        cache_entries=stats.cache_entries,
        coverage=stats.coverage,
        models=stats.models,
        entities=stats.entities,
        relationships=stats.relationships,
        embedding_model=container.embedder.model_id,
        chunker_version=container.chunker.version,
        model_window=container.embedder.max_sequence_tokens,
    )


@router.get("/doctor", response_model=DoctorOut)
async def get_doctor(container: ContainerDep) -> DoctorOut:
    """The standing checks, rendered instead of read from a terminal.

    Deliberately the same `run_doctor` the CLI calls. It tokenizes candidate
    chunks with the real tokenizer, so this is not a free request — it is a
    diagnostic somebody asked for, not something to poll.
    """
    report = await run_doctor(container.database.session_factory, container.embedder)
    return DoctorOut(
        healthy=report.healthy,
        findings=[
            FindingOut(
                check=finding.check,
                count=finding.count,
                detail=finding.detail,
                examples=finding.examples,
                healthy=finding.healthy,
                advisory=finding.advisory,
            )
            for finding in report.findings
        ],
        embedding_model=container.embedder.model_id,
        chunker_version=container.chunker.version,
        model_window=container.embedder.max_sequence_tokens,
    )
