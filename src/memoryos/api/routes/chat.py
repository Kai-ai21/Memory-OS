"""The front door, over HTTP, with conversations in it.

The asymmetry M10.0 named is unchanged. `POST /chat` returns as soon as a
statement has *committed* — the session, the memory, its normalization job and
the turn, in one transaction — and everything after that happens in a worker.
That is what makes sending instant, and it is what makes the response a promise it
can keep: the message is durable and it is not yet searchable.
`GET /chat/messages/{memory_id}/status` is where the second half arrives.

A question still costs what `/answer` costs, because it *is* `/answer`: the same
use case, the same grounding checks, the same refusal, plus up to three turns of
this session's conversation folded into the question slot.

**Sessions are a view.** Every route here is navigation: which conversations
exist, what is in one, and which one a new message lands in. Nothing under this
prefix creates a memory for a session or changes what a memory means, and the
route that matters most for keeping that legible is the `memory_id` on each
message — a message links out to its memory detail view, where it sits beside
everything it connects to regardless of which conversation it was typed in.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.application import chat as chat_use_case
from memoryos.application.chat import (
    EmptyMessage,
    Exchange,
    Message,
    NoLanguageModel,
    NoSuchSession,
)
from memoryos.container import Container
from memoryos.domain.jobs import PermanentError, TransientError
from memoryos.domain.message_intent import ChatRole, MessageIntent

router = APIRouter(tags=["chat"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class MessageIn(BaseModel):
    text: str = Field(min_length=1)
    # Where to put it. Omitted means "continue the latest conversation, or start
    # one if it has gone quiet for thirty minutes" — the behaviour somebody typing
    # into a fresh tab wants without having to choose.
    session_id: UUID | None = None
    # The new-conversation button. Ignored when `session_id` is given, because an
    # explicit session is a person having clicked one and outranks a flag.
    new_session: bool = False


class CitationOut(BaseModel):
    memory_id: UUID | None
    locator: str
    excerpt: str


class MessageOut(BaseModel):
    id: UUID
    session_id: UUID
    role: ChatRole
    content: str
    ordinal: int
    created_at: datetime
    # `statement`, `question` or `both` on a user message; null on an answer.
    # Sent on every turn so the interface can show how the message was read and
    # let a misreading be corrected — a classification the user cannot see is one
    # they cannot argue with.
    intent: MessageIntent | None = None
    # What the log calls the memory this became, and the memory itself. The key is
    # null when nothing was stored; the id is additionally null when the memory has
    # been deleted or a replay has not yet rebuilt it, which the key outlives.
    external_key: str | None = None
    memory_id: UUID | None = None
    answer_model: str | None = None
    # Null when this is not an answer, which is not the same as false.
    refused: bool | None = None
    # Every factual sentence cited, nothing invented. Live only: null on a message
    # read back from the transcript, because grounding is a property of the
    # verification run rather than of the text it checked.
    grounded: bool | None = None
    citations: list[CitationOut] = Field(default_factory=list)


class ExchangeOut(BaseModel):
    session_id: UUID
    # One message for a statement, two for anything answered. A list rather than
    # two named fields, because the client renders them in order and a shape that
    # forced it to know which half exists is a shape it would get wrong.
    messages: list[MessageOut]


class SessionOut(BaseModel):
    id: UUID
    title: str | None
    started_at: datetime
    last_activity: datetime
    message_count: int
    archived_at: datetime | None = None


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


@router.post("/chat", response_model=ExchangeOut, status_code=status.HTTP_201_CREATED)
async def send(body: MessageIn, container: ContainerDep) -> ExchangeOut:
    try:
        exchange = await container.chat()(
            body.text, session_id=body.session_id, new_session=body.new_session
        )
    except EmptyMessage as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except NoSuchSession as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (NoLanguageModel, MissingApiKey) as exc:
        # 501 rather than 500, the same as `/answer`: no model is a configuration
        # state rather than a fault, and storing still works in this deployment.
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from exc
    except TransientError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except PermanentError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return _exchange_out(exchange)


# Registered before `/chat/{session_id}`, and it has to be. FastAPI matches in
# registration order, so the parameterised route would claim `/chat/sessions`
# first and answer 422 for a path that was never meant to be a UUID.
@router.get("/chat/sessions", response_model=list[SessionOut])
async def list_sessions(
    container: ContainerDep,
    include_archived: Annotated[bool, Query()] = False,
) -> list[SessionOut]:
    """Conversations, newest activity first."""
    found = await chat_use_case.sessions(
        container.database.session_factory, include_archived=include_archived
    )
    return [
        SessionOut(
            id=row.id,
            title=row.title,
            started_at=row.started_at,
            last_activity=row.last_activity,
            message_count=row.message_count,
            archived_at=row.archived_at,
        )
        for row in found
    ]


@router.post(
    "/chat/sessions/{session_id}/archive", status_code=status.HTTP_204_NO_CONTENT
)
async def archive_session(
    session_id: UUID,
    container: ContainerDep,
    archived: Annotated[bool, Query()] = True,
) -> None:
    """Hide a conversation, or with `archived=false`, bring it back.

    Deletes nothing. `archived=false` on the same route rather than a second
    endpoint, because unarchiving is the identical write with the opposite value —
    two routes would be two places for the "and nothing is deleted" guarantee to
    be forgotten.
    """
    if not await chat_use_case.archive(
        container.database.session_factory, session_id, archived=archived
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such session")


# Also before `/chat/{session_id}`: a literal segment must be registered ahead of
# the parameterised route that would shadow it.
@router.get("/chat/messages/{memory_id}/status", response_model=StatusOut)
async def message_status(memory_id: UUID, container: ContainerDep) -> StatusOut:
    """How far a stored message has got, and what it turned out to connect to.

    Keyed by memory id rather than message id, because what it reports is a
    property of the memory: the same question is worth asking about a file, and an
    endpoint that only worked for things that were typed would be the first
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


@router.get("/chat/{session_id}", response_model=list[MessageOut])
async def session_messages(
    session_id: UUID,
    container: ContainerDep,
    q: Annotated[str | None, Query()] = None,
) -> list[MessageOut]:
    """One conversation's turns, in order, oldest first.

    `q` filters within this session by plain substring, and that is deliberately
    all it does. **Search within a session is not corpus search**: it answers
    "where in this conversation did I say that" over rows the reader has already
    seen, and running it through the embedder would return semantic neighbours
    from a conversation they can see all of. `/search` is the other question.
    """
    found = await chat_use_case.messages(
        container.database.session_factory, session_id, query=q
    )
    return [_message_out(message) for message in found]


def _exchange_out(exchange: Exchange) -> ExchangeOut:
    return ExchangeOut(
        session_id=exchange.session_id,
        messages=[_message_out(message) for message in exchange.messages],
    )


def _message_out(message: Message) -> MessageOut:
    return MessageOut(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        ordinal=message.ordinal,
        created_at=message.created_at,
        intent=message.intent,
        external_key=message.external_key,
        memory_id=message.memory_id,
        answer_model=message.answer_model,
        refused=message.refused,
        grounded=message.grounded,
        citations=[
            CitationOut(
                memory_id=citation.memory_id,
                locator=citation.locator,
                excerpt=citation.excerpt,
            )
            for citation in message.citations
        ],
    )
