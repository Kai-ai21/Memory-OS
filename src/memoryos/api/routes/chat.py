"""The front door, over HTTP.

Three endpoints and one asymmetry worth naming. `POST /chat` returns as soon as
a statement has *committed* — the memory, its normalization job and the
transcript row, in one transaction — and everything after that happens in a
worker. That is what makes sending instant, and it is also what makes the
response a promise it can keep: the message is durable, and it is not yet
searchable. `GET /chat/{memory_id}/status` is where the second half arrives, and
it says `searchable: false` until it is true rather than implying otherwise.

A question is different and cannot be made instant: it retrieves, assembles,
generates and verifies, which is seconds. It costs what `/answer` costs, because
it *is* `/answer` — the same use case, the same grounding checks, the same
refusal.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.application import chat as chat_use_case
from memoryos.application.chat import ChatTurn, EmptyMessage, NoLanguageModel
from memoryos.container import Container
from memoryos.domain.jobs import PermanentError, TransientError
from memoryos.domain.message_intent import MessageIntent

router = APIRouter(tags=["chat"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class MessageIn(BaseModel):
    text: str = Field(min_length=1)


class CitationOut(BaseModel):
    memory_id: UUID | None
    locator: str
    excerpt: str


class TurnOut(BaseModel):
    id: UUID
    text: str
    # `statement`, `question` or `both`. Sent on every turn so the interface can
    # show how the message was read and let a misreading be corrected — a
    # classification the user cannot see is one they cannot argue with.
    intent: MessageIntent
    created_at: datetime
    # What the log calls the memory this became, and the memory itself. The key
    # is null exactly when the turn was a question; the id is additionally null
    # when the memory has since been deleted, which the key outlives.
    external_key: str | None = None
    memory_id: UUID | None = None
    answer: str | None = None
    answer_model: str | None = None
    # Null when there is no answer, which is not the same as false. A UI that
    # conflated them would render every stored statement as a question that was
    # answered without refusing.
    refused: bool | None = None
    # Every factual sentence cited, nothing invented. Live only: null on a turn
    # read back from the transcript, because grounding is a property of the
    # verification run rather than of the text it checked.
    grounded: bool | None = None
    citations: list[CitationOut] = Field(default_factory=list)


class ConnectionOut(BaseModel):
    entity_id: UUID
    name: str
    memories: int


class StatusOut(BaseModel):
    memory_id: UUID
    chunks: int
    embedded_chunks: int
    extracted: bool
    searchable: bool
    connections: list[ConnectionOut] = Field(default_factory=list)
    connected_memories: int = 0


@router.post("/chat", response_model=TurnOut, status_code=status.HTTP_201_CREATED)
async def send(body: MessageIn, container: ContainerDep) -> TurnOut:
    try:
        turn = await container.chat()(body.text)
    except EmptyMessage as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except (NoLanguageModel, MissingApiKey) as exc:
        # 501 rather than 500, the same as `/answer`: no model is a configuration
        # state rather than a fault, and storing still works in this deployment.
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from exc
    except TransientError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except PermanentError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return _turn_out(turn)


@router.get("/chat", response_model=list[TurnOut])
async def transcript(container: ContainerDep, limit: int = 100) -> list[TurnOut]:
    """The conversation, oldest first — newest at the bottom, as it is drawn."""
    turns = await chat_use_case.history(container.database.session_factory, limit=limit)
    return [_turn_out(turn) for turn in turns]


@router.get("/chat/{memory_id}/status", response_model=StatusOut)
async def message_status(memory_id: UUID, container: ContainerDep) -> StatusOut:
    """How far a stored message has got, and what it turned out to connect to.

    Keyed by memory id rather than message id, because what it reports is a
    property of the memory: the same question is worth asking about a file, and
    an endpoint that only worked for things that were typed would be the first
    chat-shaped special case in this system.
    """
    found = await chat_use_case.status(container.database.session_factory, memory_id)
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such memory")
    return StatusOut(
        memory_id=found.memory_id,
        chunks=found.chunks,
        embedded_chunks=found.embedded_chunks,
        extracted=found.extracted,
        searchable=found.searchable,
        connections=[
            ConnectionOut(
                entity_id=connection.entity_id,
                name=connection.name,
                memories=connection.memories,
            )
            for connection in found.connections
        ],
        connected_memories=found.connected_memories,
    )


def _turn_out(turn: ChatTurn) -> TurnOut:
    return TurnOut(
        id=turn.id,
        text=turn.text,
        intent=turn.intent,
        created_at=turn.created_at,
        external_key=turn.external_key,
        memory_id=turn.memory_id,
        answer=turn.answer,
        answer_model=turn.answer_model,
        refused=turn.refused,
        grounded=turn.grounded,
        citations=[
            CitationOut(
                memory_id=citation.memory_id,
                locator=citation.locator,
                excerpt=citation.excerpt,
            )
            for citation in turn.citations
        ],
    )
