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

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.application import chat as chat_use_case
from memoryos.application.attachments import (
    ALLOWED_SUFFIXES,
    CHUNK_BYTES,
    MAX_FILE_BYTES,
    EmptyFile,
    FileTooLarge,
    UnsupportedMediaType,
    Upload,
)
from memoryos.application.chat import (
    EmptyMessage,
    Exchange,
    Message,
    NoLanguageModel,
    NoSuchSession,
    Stage,
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


class AttachmentOut(BaseModel):
    id: UUID
    ordinal: int
    filename: str
    byte_size: int
    media_type: str | None
    external_key: str
    content_hash: str
    # True when these bytes were already in the corpus, so this upload linked to
    # an existing artifact rather than storing new ones. "Already in memory,
    # linked" is worth saying; a silent success looks identical to a no-op.
    deduplicated: bool
    memory_id: UUID | None = None


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
    # Files dropped with this message. Several is normal: three files dropped
    # together are three memories and one turn.
    attachments: list[AttachmentOut] = Field(default_factory=list)


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
    # `stored`, `parsing`, `chunking`, `indexed` or `failed`. One word for where
    # this is, because a PDF is not searchable the moment its upload finishes and
    # an interface that said "done" would look broken thirty seconds later.
    stage: Stage
    # Why normalization gave up, in the parser's own words. Null unless the stage
    # is `failed`. A dead-lettered attachment nobody hears about is the worst
    # outcome — they think it worked.
    failure: str | None = None
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


@router.post(
    "/chat/attach", response_model=ExchangeOut, status_code=status.HTTP_201_CREATED
)
async def attach(
    container: ContainerDep,
    files: Annotated[list[UploadFile], File()],
    note: Annotated[str | None, Form()] = None,
    session_id: Annotated[UUID | None, Form()] = None,
    new_session: Annotated[bool, Form()] = False,
) -> ExchangeOut:
    """Files dropped into the chat, optionally with a note about them.

    **Streamed, never buffered.** Each file is read in `CHUNK_BYTES` pieces and
    handed to `BlobStore.put_stream`, which hashes as it writes — the single-pass
    pattern from M1.4. A 50MB PDF costs 64KiB of this process at a time, and the
    ceiling is enforced mid-stream rather than from `Content-Length`, which a
    client states and can be wrong about.

    Registered above `/chat/{session_id}` for the reason `/chat/sessions` is:
    FastAPI matches in registration order, so the parameterised route would claim
    `/chat/attach` and answer 422 for a path that was never a UUID.
    """
    uploads = [
        Upload(
            filename=item.filename or "file",
            media_type=item.content_type,
            stream=_chunks(item),
        )
        for item in files
    ]
    try:
        exchange = await container.chat().attach(
            uploads, note=note, session_id=session_id, new_session=new_session
        )
    except UnsupportedMediaType as exc:
        # 415, and the body names what *is* supported. A generic 400 makes somebody
        # guess which of the three things they did wrong.
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)
        ) from exc
    except FileTooLarge as exc:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(exc)
        ) from exc
    except (EmptyFile, EmptyMessage) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except NoSuchSession as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (NoLanguageModel, MissingApiKey) as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from exc
    except TransientError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except PermanentError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return _exchange_out(exchange)


class AttachLimitsOut(BaseModel):
    """What the door accepts, so the interface can say it before a rejection."""

    max_file_bytes: int
    suffixes: list[str]


@router.get("/chat/attach/limits", response_model=AttachLimitsOut)
async def attach_limits() -> AttachLimitsOut:
    """The upload limits, served rather than duplicated in the client.

    The allow-list is composed from the parsers that handle each format, so a
    client that restated it would eventually disagree with the pipeline — and the
    symptom is a file the interface refuses that the system can read, or worse the
    reverse.
    """
    return AttachLimitsOut(
        max_file_bytes=MAX_FILE_BYTES, suffixes=sorted(ALLOWED_SUFFIXES)
    )


async def _chunks(item: UploadFile) -> AsyncIterator[bytes]:
    """One file, in pieces, from the multipart body.

    Starlette has already spooled the part to a temp file for anything large, so
    this is a read from disk rather than from the socket — but it is still a read
    that must not become one `bytes` object. `read(-1)` here would undo the whole
    reason `put_stream` exists.
    """
    while chunk := await item.read(CHUNK_BYTES):
        yield chunk


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
        stage=found.stage,
        failure=found.failure,
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
        attachments=[
            AttachmentOut(
                id=attachment.id,
                ordinal=attachment.ordinal,
                filename=attachment.filename,
                byte_size=attachment.byte_size,
                media_type=attachment.media_type,
                external_key=attachment.external_key,
                content_hash=attachment.content_hash,
                deduplicated=attachment.deduplicated,
                memory_id=attachment.memory_id,
            )
            for attachment in message.attachments
        ],
    )
