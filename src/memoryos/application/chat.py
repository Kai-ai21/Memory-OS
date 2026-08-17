"""The front door: one box that stores what you type and answers what you ask.

**There is no chat pipeline.** A typed message goes through the same stages a
file does — content hash, artifact, ingestion event, memory, chunks, embedding,
entity extraction — via the same `ingest_item` the sync calls, and everything
downstream of that call is unaware it was typed. What this module adds is
narrow: which of the two things to do with a message, a source to hang the
memories off, and the transcript that makes a conversation survive a reload.

**One memory per message, never one per conversation.** A conversation is an
arbitrary container. A thought from Tuesday relates to one from last month
because they are about the same thing, not because they were typed in the same
sitting, and grouping by session would bury exactly that: the entity graph is
what connects them, and it connects memories. A session-shaped memory would also
grow with every turn, so its content hash would change on every message and its
chunks would be rebuilt each time — the pipeline would work and would do a
quadratic amount of it.

**The dates here are the first reliable ones in this corpus.** `occurred_at` is
when somebody pressed enter and `occurred_at_source` is `DECLARED`, which
nothing else in the system can honestly claim: a filesystem mtime says when a
file was last written to this disk, which is not when the thought happened, and
Phase 4's timeline has been drawing that noise for eight milestones. Anything
weighting recency should weight these differently, and the provenance column is
already the place that says so.

**Answers are not stored as memories.** They are derived from what is already
here, and a generated sentence that can be retrieved becomes evidence for the
next generated sentence — a corpus that quietly starts citing itself, with
nothing in any answer distinguishing a claim that came from a document from one
that came from a previous answer about the document. The answer is written to
`chat_messages`, which no retriever reads, and that is the difference between a
transcript and a source.

**M10.1's sessions are a view over all of that, and change none of it.** A
message belongs to a session for navigation and for the three turns of
conversational context; it belongs to the graph for meaning. Nothing here merges
messages into a session-level memory, and nothing that decides meaning reads
`chat_sessions` at all — which is what makes a thirty-minute clock an acceptable
way to draw the boundary.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.application.answering import (
    DEFAULT_HISTORY_TURNS,
    AnswerQuestion,
    ConversationTurn,
    GroundedAnswer,
)
from memoryos.application.ingest import ingest_item
from memoryos.application.ports import BlobStore, ObservedItem
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.message_intent import ChatRole, MessageIntent, classify
from memoryos.domain.sessions import starts_new_session, title_for
from memoryos.domain.values import ContentHash, SourceKind, TimeProvenance

logger = structlog.get_logger(__name__)

# The one chat source. A singleton by name rather than by constraint, because
# `sources` is already unique on `(kind, name)` and a second uniqueness rule
# expressing the same thing is a second rule to keep true.
CHAT_SOURCE_NAME = "chat"

# Markdown rather than plain text, and the suffix is doing real work: it is what
# `projection.kind_for` reads to call the memory a `note`, and what the parser
# registry reads to pick `MarkdownParser` — which keeps headings and code fences
# as structure the chunker can split on. Somebody pasting a block of notes into a
# chat box is pasting markdown far more often than not, and a plain-text parser
# would throw the boundaries away.
MESSAGE_MEDIA_TYPE = "text/markdown"
MESSAGE_SUFFIX = ".md"

# How many entities a connection line names. Two or three is a sentence; ten is a
# tag cloud, and a line nobody reads is the same as no line.
CONNECTION_ENTITIES = 3


class EmptyMessage(ValueError):
    """Nothing but whitespace. There is no thought here to store or answer."""


@dataclass(frozen=True, slots=True)
class Citation:
    """One passage an answer cited, flattened for the transcript.

    Locator and excerpt only. The full M2.5 citation is reconstructible from the
    memory it points at, and a copy of the chunk's text stored here would be a
    second version of that text which no re-chunk could ever update.
    """

    memory_id: UUID | None
    locator: str
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_id": None if self.memory_id is None else str(self.memory_id),
            "locator": self.locator,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True, slots=True)
class Message:
    """One turn as a reader sees it.

    `memory_id` is resolved from `external_key` rather than stored, and the
    difference is what lets this row outlive a replay: the key is what the
    ingestion log records, the id is minted fresh every rebuild. A null id beside
    a non-null key means the memory was deleted, which is a fact rather than a
    fault.
    """

    id: UUID
    session_id: UUID
    role: ChatRole
    content: str
    ordinal: int
    created_at: datetime
    intent: MessageIntent | None = None
    external_key: str | None = None
    memory_id: UUID | None = None
    answer_model: str | None = None
    refused: bool | None = None
    # Whether every factual sentence carried a citation and none was invented.
    #
    # Live only — null on a message read back from the transcript, and
    # deliberately not a column. It is a property of the *verification run*, and a
    # stored copy would be a claim about grounding that no later check could
    # re-derive or contradict.
    grounded: bool | None = None
    citations: list[Citation] = field(default_factory=list)

    @property
    def stored(self) -> bool:
        return self.external_key is not None


@dataclass(frozen=True, slots=True)
class Exchange:
    """What one send produced: the message, and the answer if there was one."""

    session_id: UUID
    user: Message
    assistant: Message | None = None

    @property
    def messages(self) -> list[Message]:
        return [self.user] if self.assistant is None else [self.user, self.assistant]


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """One conversation, for the rail."""

    id: UUID
    title: str | None
    started_at: datetime
    last_activity: datetime
    message_count: int
    archived_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Connection:
    """One entity this message shares with something already in the corpus."""

    entity_id: UUID
    name: str
    memories: int


@dataclass(frozen=True, slots=True)
class MessageStatus:
    """How far a stored message has got, and what it turned out to connect to.

    Polled rather than pushed, and it carries the pipeline state as well as the
    connections because they answer the same question at two different moments.
    A message is stored instantly and is not searchable for a second or two after
    that; saying "stored" and leaving it there implies a searchability that does
    not exist yet, and `searchable` is what lets the interface say "indexing…"
    instead of lying briefly.
    """

    memory_id: UUID
    chunks: int
    embedded_chunks: int
    # Extraction has *run*, whatever it found. The distinction matters for the
    # same reason `memories.entity_extractor_version` exists: a message with no
    # entities in it is finished, not pending, and treating the two the same is
    # how a connection line spins forever on a one-line thought about nothing.
    extracted: bool
    connections: list[Connection] = field(default_factory=list)
    # Distinct earlier memories reached through any shared entity. Not the sum of
    # the per-entity counts, which double-counts every memory sharing two.
    connected_memories: int = 0

    @property
    def searchable(self) -> bool:
        """Retrievable by both halves of hybrid search.

        Chunks alone make it findable by keyword; the vector half needs the
        embedding. Reporting "searchable" at the chunk stage would be true of one
        retriever and false of the one people notice.
        """
        return self.chunks > 0 and self.embedded_chunks == self.chunks


class NoSuchSession(LookupError):
    """A session id nothing has."""


class Chat:
    """Classify, then store or answer, inside a session."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        blob_store: BlobStore,
        answer: AnswerQuestion | None = None,
        *,
        k: int = 10,
        history_turns: int = DEFAULT_HISTORY_TURNS,
    ) -> None:
        self._sessions = session_factory
        self._blobs = blob_store
        # Optional, because storing does not need a language model and a
        # deployment without an API key must still have a working front door.
        # A question in that deployment raises whatever the container raises,
        # which is the same 501 `/answer` has always produced.
        self._answer = answer
        self._k = k
        self._history_turns = history_turns

    async def __call__(
        self,
        text: str,
        *,
        now: datetime | None = None,
        session_id: UUID | None = None,
        new_session: bool = False,
    ) -> Exchange:
        message = text.strip()
        if not message:
            raise EmptyMessage("a message must contain something other than whitespace")

        at = now or datetime.now(UTC)
        intent = classify(message)

        source = None if intent is MessageIntent.QUESTION else await self._source()
        user = await self._append_user(
            message, intent, at, source, session_id=session_id, new_session=new_session
        )

        if intent is MessageIntent.STATEMENT:
            logger.info(
                "chat.stored",
                session_id=str(user.session_id),
                ordinal=user.ordinal,
                external_key=user.external_key,
            )
            return Exchange(session_id=user.session_id, user=user)

        return await self._answer_exchange(user)

    # ----------------------------------------------------------------------
    # Writing a turn
    # ----------------------------------------------------------------------

    async def _append_user(
        self,
        message: str,
        intent: MessageIntent,
        at: datetime,
        source: Source | None,
        *,
        session_id: UUID | None,
        new_session: bool,
    ) -> Message:
        """The session, the memory, its normalization job and the turn, together.

        One transaction for all four, and the reason is M10.0's: the queue is a
        table precisely so a memory cannot exist without the job that processes
        it, and the transcript row joins that guarantee rather than sitting outside
        it. A turn recorded as stored whose memory was rolled back would be a
        message the interface claims to have kept and search cannot find.

        The session's `message_count` is both the ordinal source and the row being
        incremented, which is why it is read `FOR UPDATE`: two sends racing on one
        session would otherwise both read the same count and collide on
        `uq_chat_messages_session_ordinal` — a correct outcome arrived at by
        constraint violation rather than by ordering.
        """
        item = None if source is None else _message_item(new_id(), message, at)

        async with self._sessions.begin() as session:
            record = await self._resolve_session(
                session, at, session_id=session_id, new_session=new_session
            )

            external_key: str | None = None
            memory_id: UUID | None = None
            if source is not None and item is not None:
                recorded = await ingest_item(session, self._blobs, source, item)
                if recorded is None:
                    # Unreachable: the external key carries a fresh id, so there
                    # is never a current version for it to be identical to. Named
                    # rather than asserted, because a silent `None` here would
                    # write a turn claiming to be stored that stored nothing.
                    raise UnstorableMessage(
                        f"message produced no memory; its external key "
                        f"{item.external_key!r} was already current, which cannot happen"
                    )
                external_key = item.external_key
                memory_id = recorded.memory_id

            ordinal = record.message_count
            row = models.ChatMessage(
                id=new_id(),
                session_id=record.id,
                role=ChatRole.USER.value,
                content=message,
                ordinal=ordinal,
                external_key=external_key,
                intent=intent.value,
                created_at=at,
            )
            session.add(row)

            record.message_count = ordinal + 1
            record.last_activity = at
            if ordinal == 0:
                # From the first user message only. A title that tracked the
                # latest one would rename a conversation out from under somebody
                # halfway through reading the list.
                record.title = title_for(message)

            return Message(
                id=row.id,
                session_id=record.id,
                role=ChatRole.USER,
                content=message,
                ordinal=ordinal,
                created_at=at,
                intent=intent,
                external_key=external_key,
                # Known here without a lookup, because `ingest_item` just minted
                # it. Reads go the other way — key to id — for the replay reason
                # the model docstring gives.
                memory_id=memory_id,
            )

    async def _resolve_session(
        self,
        session: AsyncSession,
        at: datetime,
        *,
        session_id: UUID | None,
        new_session: bool,
    ) -> models.ChatSession:
        """The session this message belongs to, creating one where it should.

        Three ways in, in priority order, and the order is the point: an explicit
        id is a person having clicked a conversation and must never be overridden
        by a clock; an explicit new session is the button; and only in the absence
        of both does the thirty-minute rule get to guess.
        """
        if session_id is not None:
            found = await session.get(
                models.ChatSession, session_id, with_for_update=True
            )
            if found is None:
                raise NoSuchSession(f"no such session: {session_id}")
            return found

        if not new_session:
            latest = (
                await session.execute(
                    select(models.ChatSession)
                    .where(models.ChatSession.archived_at.is_(None))
                    .order_by(models.ChatSession.last_activity.desc())
                    .limit(1)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if latest is not None and not starts_new_session(at, latest.last_activity):
                return latest

        record = models.ChatSession(
            id=new_id(),
            title=None,
            started_at=at,
            last_activity=at,
            message_count=0,
        )
        session.add(record)
        # Flushed so the message inserted next has a row to point at, and so the
        # `FOR UPDATE` above sees it on the next send rather than a moment later.
        await session.flush()
        logger.info("chat.session_started", session_id=str(record.id))
        return record

    async def _answer_exchange(self, user: Message) -> Exchange:
        if self._answer is None:
            raise NoLanguageModel(
                "this deployment has no language model configured, so a question "
                "cannot be answered. Statements are stored as usual."
            )

        history = await self._history(user.session_id, before=user.ordinal)
        result = await self._answer(user.content, k=self._k, history=history)
        citations = _citations_of(result)

        async with self._sessions.begin() as session:
            record = await session.get(
                models.ChatSession, user.session_id, with_for_update=True
            )
            if record is None:
                raise NoSuchSession(f"no such session: {user.session_id}")
            ordinal = record.message_count
            row = models.ChatMessage(
                id=new_id(),
                session_id=user.session_id,
                role=ChatRole.ASSISTANT.value,
                content=result.answer,
                ordinal=ordinal,
                answer_model=result.model_id,
                answer_refused=result.refused,
                citations=[citation.as_dict() for citation in citations],
                # The question's clock, not the wall clock.
                #
                # One clock per exchange, and the reason is not only that `Chat`
                # takes an injectable `now`. An answer is *of* the question that
                # produced it: stamping it from `datetime.now()` would order a
                # conversation by two different clocks, and any caller that passed
                # a time — a test, a backfill, an importer — would produce a
                # transcript whose answers sort before their own questions.
                # `ordinal` already separates them.
                created_at=user.created_at,
            )
            session.add(row)
            record.message_count = ordinal + 1
            record.last_activity = user.created_at
            row_id = row.id

        logger.info(
            "chat.answered",
            session_id=str(user.session_id),
            intent=user.intent.value if user.intent else None,
            refused=result.refused,
            citations=len(citations),
        )
        return Exchange(
            session_id=user.session_id,
            user=user,
            assistant=Message(
                id=row_id,
                session_id=user.session_id,
                role=ChatRole.ASSISTANT,
                content=result.answer,
                ordinal=ordinal,
                created_at=user.created_at,
                answer_model=result.model_id,
                refused=result.refused,
                grounded=result.verification.grounded,
                citations=citations,
            ),
        )

    async def _history(
        self, session_id: UUID, *, before: int
    ) -> list[ConversationTurn]:
        """The last few turns of *this* session, oldest first.

        A turn is a user message and whatever answer followed it, which is why
        this reads twice the row limit and folds pairs back together: M10.0's
        single-row shape made "three turns" and "three rows" the same number, and
        M10.1's one-row-per-turn shape does not.

        **Capped at three, and the cap is the milestone's own instruction as well
        as a measured one.** Past three turns the model starts answering the
        conversation rather than the question, and retrieval degrades because the
        query is diluted by topics the question has moved on from — M10.0 measured
        exactly that and narrowed which questions borrow the conversation at all.
        Resuming an old session loads its last three turns, never all of it.

        Scoped to the session, which is the one thing sessions are load-bearing
        for. Carrying context across a boundary the interface drew would make the
        boundary a lie.
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(models.ChatMessage)
                    .where(
                        models.ChatMessage.session_id == session_id,
                        models.ChatMessage.ordinal < before,
                    )
                    .order_by(models.ChatMessage.ordinal.desc())
                    .limit(self._history_turns * 2)
                )
            ).scalars().all()

        turns: list[ConversationTurn] = []
        answer: str | None = None
        for row in rows:
            if row.role == ChatRole.ASSISTANT.value:
                answer = row.content
                continue
            turns.append(ConversationTurn(text=row.content, answer=answer))
            answer = None
        return list(reversed(turns))[-self._history_turns :]

    async def _source(self) -> Source:
        """The chat source, created on first use.

        Not cached on the instance. It is one indexed lookup on a unique index,
        and a cached id outlives the row it names in exactly the situation where
        that hurts most — a test that truncated between messages, or a replay
        that rebuilt the corpus underneath a long-lived process.
        """
        async with self._sessions.begin() as session:
            # ON CONFLICT DO NOTHING rather than check-then-insert: two messages
            # typed at once are two requests racing to create the same singleton,
            # and the loser of that race should get the winner's row rather than
            # a unique-violation traceback on somebody's first ever message.
            await session.execute(
                pg_insert(models.Source)
                .values(
                    id=new_id(),
                    kind=SourceKind.CHAT.value,
                    name=CHAT_SOURCE_NAME,
                    config={},
                    cursor={},
                )
                .on_conflict_do_nothing(constraint="uq_sources_kind_name")
            )
            source = await SqlAlchemySourceRepository(session).get_by_name(
                SourceKind.CHAT, CHAT_SOURCE_NAME
            )
        assert source is not None  # just inserted, or already there
        return source


class UnstorableMessage(RuntimeError):
    """A message the ingest path declined to record."""


class NoLanguageModel(RuntimeError):
    """A question arrived at a deployment with no model to answer it.

    Its own type so a transport can tell it from a malformed request: this is a
    configuration state, the corpus is intact, and storing still works.
    """


def _message_item(turn_id: UUID, message: str, at: datetime) -> ObservedItem:
    """A typed message, described the way a walked file is.

    Nothing is bent to fit. An external key, bytes, a hash, a media type and a
    time with its provenance is what an item *is*; that the shape holds for
    something nobody walked to find is the argument for `ingest_item` existing
    rather than a chat-shaped copy of it.

    The key is dated for the reader and unique by construction: `2026-08-17/
    0198….md` sorts by day in any listing of external keys, and the id is what
    makes it collide with nothing. A key derived from the text would make two
    identical thoughts on two days one memory with two versions, which is the
    wrong answer — the second one was a thought you had again.

    Deliberately carries no session in it. The key is the memory's durable
    identity and a session is a view; putting one inside the other would make
    moving a conversation a corpus migration.
    """
    data = message.encode("utf-8")

    async def read() -> bytes:
        return data

    return ObservedItem(
        external_key=f"{at:%Y-%m-%d}/{turn_id}{MESSAGE_SUFFIX}",
        content_hash=ContentHash.of(data),
        byte_size=len(data),
        media_type=MESSAGE_MEDIA_TYPE,
        occurred_at=at,
        # The one place in this system that can say this honestly. Everything
        # filesystem-shaped guesses from an mtime.
        occurred_at_source=TimeProvenance.DECLARED,
        read_bytes=read,
        # Nothing will ever look at this message again, so there is nothing for a
        # next pass to cheaply skip.
        fingerprint=None,
    )


def _citations_of(result: GroundedAnswer) -> list[Citation]:
    return [
        Citation(
            memory_id=citation.memory_id,
            locator=citation.locator,
            excerpt=" ".join(citation.excerpt.split()),
        )
        for explained in result.citations
        for citation in explained.citations[:1]
    ]


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


async def sessions(
    session_factory: async_sessionmaker[AsyncSession], *, include_archived: bool = False
) -> list[SessionSummary]:
    """Conversations, newest activity first.

    Archived ones are excluded by default and available by asking, which is the
    same shape every other "dismissed" read in this system has: hiding is not
    deleting, and a list that could not show you what you archived would make it
    deleting in everything but name.
    """
    stmt = select(models.ChatSession).order_by(models.ChatSession.last_activity.desc())
    if not include_archived:
        stmt = stmt.where(models.ChatSession.archived_at.is_(None))
    async with session_factory() as session:
        rows = (await session.execute(stmt)).scalars().all()
    return [
        SessionSummary(
            id=row.id,
            title=row.title,
            started_at=row.started_at,
            last_activity=row.last_activity,
            message_count=row.message_count,
            archived_at=row.archived_at,
        )
        for row in rows
    ]


async def latest_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> UUID | None:
    """The conversation a client with no opinion should open. None when there are none."""
    async with session_factory() as session:
        return (
            await session.execute(
                select(models.ChatSession.id)
                .where(models.ChatSession.archived_at.is_(None))
                .order_by(models.ChatSession.last_activity.desc())
                .limit(1)
            )
        ).scalars().first()


async def messages(
    session_factory: async_sessionmaker[AsyncSession],
    session_id: UUID,
    *,
    query: str | None = None,
) -> list[Message]:
    """One session's turns in order, optionally filtered.

    `query` is a plain substring match, case-insensitive, and that is the whole
    feature. **Search within a session is deliberately not corpus search**: this
    answers "where in this conversation did I say that", which is a scroll
    replacement over a hundred rows somebody has already read, and running it
    through the embedder would return semantic neighbours from a conversation the
    reader can see all of. Corpus search is a different question with its own
    page, and blurring them would leave neither answerable.
    """
    stmt = (
        select(models.ChatMessage)
        .where(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.ordinal)
    )
    if query and query.strip():
        stmt = stmt.where(models.ChatMessage.content.ilike(f"%{query.strip()}%"))

    async with session_factory() as session:
        rows = (await session.execute(stmt)).scalars().all()
        resolved = await _memories_by_key(
            session, [row.external_key for row in rows if row.external_key]
        )
    return [_message_of(row, resolved) for row in rows]


async def archive(
    session_factory: async_sessionmaker[AsyncSession],
    session_id: UUID,
    *,
    at: datetime | None = None,
    archived: bool = True,
) -> bool:
    """Hide a conversation, or bring it back. Deletes nothing, ever.

    Returns False when there is no such session, which a transport renders as a
    404 rather than as a silent success — a button that reports having archived
    something that does not exist is a button nobody can trust.
    """
    async with session_factory.begin() as session:
        record = await session.get(models.ChatSession, session_id)
        if record is None:
            return False
        record.archived_at = (at or datetime.now(UTC)) if archived else None
    return True


async def _memories_by_key(
    session: AsyncSession, keys: Sequence[str]
) -> dict[str, UUID]:
    """Current, undeleted chat memories for these external keys.

    One query for the whole page rather than one per turn. A key that is absent
    from the result is a memory that has been deleted — or one whose replay has
    not finished — which the caller renders as a turn with no link rather than as
    an error.
    """
    if not keys:
        return {}
    rows = await session.execute(
        select(models.Memory.external_key, models.Memory.id)
        .join(models.Source, models.Source.id == models.Memory.source_id)
        .where(
            models.Source.kind == SourceKind.CHAT.value,
            models.Memory.external_key.in_(list(keys)),
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
        )
    )
    return {str(key): memory_id for key, memory_id in rows}


async def status(
    sessions_factory: async_sessionmaker[AsyncSession], memory_id: UUID
) -> MessageStatus | None:
    """How far this memory has got, and what it connects to.

    Returns None when there is no such memory, which a caller renders as a 404
    rather than as an empty connection line — "nothing connects to this" and
    "this does not exist" are different sentences.
    """
    async with sessions_factory() as session:
        memory = await session.get(models.Memory, memory_id)
        if memory is None:
            return None

        counts = (
            await session.execute(
                select(
                    func.count(models.MemoryChunk.id),
                    func.count(models.MemoryChunk.embedding),
                ).where(models.MemoryChunk.memory_id == memory_id)
            )
        ).one()

        connections = await _connections(session, memory)
        reached = await _connected_memories(session, memory, connections)

    return MessageStatus(
        memory_id=memory_id,
        chunks=int(counts[0]),
        embedded_chunks=int(counts[1]),
        extracted=memory.entity_extractor_version is not None,
        connections=connections,
        connected_memories=reached,
    )


# The entity a mention resolves to after M3.2. `merged_into_id` is followed one
# hop rather than transitively, which is what resolution itself guarantees:
# `_upsert_entity` re-points mentions at the winner, so a chain of losers cannot
# form. Following it further would cost a recursive CTE to reach rows that do not
# exist.
def _resolved() -> Any:
    return models.Entity.__table__.alias("winner")


async def _connections(
    session: AsyncSession, memory: models.Memory
) -> list[Connection]:
    """Entities this memory shares with earlier ones, most connecting first.

    Ordered by how many earlier memories each entity reaches, because the line
    this produces names two or three of them and the two or three worth naming
    are the ones doing the connecting. Alphabetical order would name whichever
    entity happened to start with an `a`.
    """
    winner = _resolved()
    mine = (
        select(
            func.coalesce(winner.c.id, models.Entity.id).label("entity_id"),
            func.coalesce(winner.c.name, models.Entity.name).label("name"),
        )
        .select_from(models.EntityMention)
        .join(models.Entity, models.Entity.id == models.EntityMention.entity_id)
        .outerjoin(winner, winner.c.id == models.Entity.merged_into_id)
        .where(models.EntityMention.memory_id == memory.id)
        .distinct()
        .subquery()
    )

    theirs = (
        select(
            func.coalesce(winner.c.id, models.Entity.id).label("entity_id"),
            models.EntityMention.memory_id.label("memory_id"),
        )
        .select_from(models.EntityMention)
        .join(models.Entity, models.Entity.id == models.EntityMention.entity_id)
        .outerjoin(winner, winner.c.id == models.Entity.merged_into_id)
        .join(models.Memory, models.Memory.id == models.EntityMention.memory_id)
        .where(
            models.EntityMention.memory_id != memory.id,
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
            # Earlier, and meant literally. A message connects backwards to what
            # was already here; counting memories ingested after it would make
            # the line change every time anything else arrived.
            models.Memory.ingested_at <= memory.ingested_at,
        )
        .distinct()
        .subquery()
    )

    rows = (
        await session.execute(
            select(
                mine.c.entity_id,
                mine.c.name,
                func.count(theirs.c.memory_id).label("memories"),
            )
            .select_from(mine)
            .join(theirs, theirs.c.entity_id == mine.c.entity_id)
            .group_by(mine.c.entity_id, mine.c.name)
            .order_by(func.count(theirs.c.memory_id).desc(), mine.c.name)
        )
    ).all()

    return [
        Connection(entity_id=row[0], name=row[1], memories=int(row[2])) for row in rows
    ]


async def _connected_memories(
    session: AsyncSession, memory: models.Memory, connections: Sequence[Connection]
) -> int:
    """Distinct earlier memories reached through any of these entities.

    Not the sum of the per-entity counts. A memory sharing two entities with this
    one is one memory, and adding the columns up would report it twice — which is
    how a connection line ends up claiming more than the corpus contains.
    """
    if not connections:
        return 0

    winner = _resolved()
    resolved = func.coalesce(winner.c.id, models.Entity.id)
    return int(
        (
            await session.execute(
                select(func.count(func.distinct(models.EntityMention.memory_id)))
                .select_from(models.EntityMention)
                .join(models.Entity, models.Entity.id == models.EntityMention.entity_id)
                .outerjoin(winner, winner.c.id == models.Entity.merged_into_id)
                .join(models.Memory, models.Memory.id == models.EntityMention.memory_id)
                .where(
                    resolved.in_([connection.entity_id for connection in connections]),
                    models.EntityMention.memory_id != memory.id,
                    models.Memory.is_current.is_(True),
                    models.Memory.deleted_at.is_(None),
                    models.Memory.ingested_at <= memory.ingested_at,
                )
            )
        ).scalar_one()
    )


def _message_of(row: models.ChatMessage, resolved: dict[str, UUID]) -> Message:
    return Message(
        id=row.id,
        session_id=row.session_id,
        role=ChatRole(row.role),
        content=row.content,
        ordinal=row.ordinal,
        created_at=row.created_at,
        intent=None if row.intent is None else MessageIntent(row.intent),
        external_key=row.external_key,
        memory_id=None if row.external_key is None else resolved.get(row.external_key),
        answer_model=row.answer_model,
        refused=row.answer_refused,
        citations=[
            Citation(
                memory_id=(
                    None
                    if citation.get("memory_id") is None
                    else UUID(str(citation["memory_id"]))
                ),
                locator=str(citation.get("locator", "")),
                excerpt=str(citation.get("excerpt", "")),
            )
            for citation in row.citations
        ],
    )
