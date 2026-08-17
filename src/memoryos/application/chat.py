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
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import func, select, update
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
from memoryos.domain.message_intent import MessageIntent, classify
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
class ChatTurn:
    """One message and what became of it."""

    id: UUID
    text: str
    intent: MessageIntent
    created_at: datetime
    # What the log calls the memory this message became. Null exactly when the
    # turn was a question, and enforced that way by the schema.
    external_key: str | None = None
    # The memory itself, resolved from that key. Null for a question, and null
    # for a statement whose memory has since been deleted — the key survives
    # either way, which is why it rather than this is the stored column.
    memory_id: UUID | None = None
    answer: str | None = None
    answer_model: str | None = None
    # Null when there is no answer. False and None are different states and a
    # UI that conflated them would render a statement as an answered question.
    refused: bool | None = None
    # Whether every factual sentence carried a citation and none was invented.
    #
    # Live only — null on a turn read back from the transcript, and deliberately
    # not a column. It is a property of the *verification run*, and a stored copy
    # would be a claim about grounding that no later check could re-derive or
    # contradict. The transcript keeps what was said and whether the corpus
    # declined; how well the model cited is a measurement, and measurements
    # belong to the run that made them.
    grounded: bool | None = None
    citations: list[Citation] = field(default_factory=list)

    @property
    def stored(self) -> bool:
        return self.intent is not MessageIntent.QUESTION


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


class Chat:
    """Classify, then store or answer. The only thing that decides which."""

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

    async def __call__(self, text: str, *, now: datetime | None = None) -> ChatTurn:
        message = text.strip()
        if not message:
            raise EmptyMessage("a message must contain something other than whitespace")

        at = now or datetime.now(UTC)
        intent = classify(message)
        turn_id = new_id()

        memory_id: UUID | None = None
        external_key: str | None = None
        if intent is not MessageIntent.QUESTION:
            external_key, memory_id = await self._store(turn_id, message, intent, at)
        else:
            await self._record_question(turn_id, message, at)

        turn = ChatTurn(
            id=turn_id,
            text=message,
            intent=intent,
            created_at=at,
            external_key=external_key,
            memory_id=memory_id,
        )
        if intent is MessageIntent.STATEMENT:
            logger.info("chat.stored", message_id=str(turn_id), memory_id=str(memory_id))
            return turn

        return await self._answer_turn(turn)

    async def _store(
        self, turn_id: UUID, message: str, intent: MessageIntent, at: datetime
    ) -> tuple[str, UUID]:
        """The memory, its normalization job and the transcript row, together.

        One transaction for all three. The queue is a table precisely so that a
        memory cannot exist without the job that processes it, and the transcript
        row joins that guarantee rather than sitting outside it: a turn recorded
        as stored whose memory was rolled back would be a message the interface
        claims to have kept and search cannot find.
        """
        source = await self._source()
        item = _message_item(turn_id, message, at)

        async with self._sessions.begin() as session:
            recorded = await ingest_item(session, self._blobs, source, item)
            if recorded is None:
                # Unreachable: the external key carries this turn's id, so there
                # is never a current version for it to be identical to. Named
                # rather than asserted, because a silent `None` here would insert
                # a transcript row with no memory and look like a question.
                raise UnstorableMessage(
                    f"message {turn_id} produced no memory; its external key "
                    f"{item.external_key!r} was already current, which cannot happen"
                )
            session.add(
                models.ChatMessage(
                    id=turn_id,
                    text=message,
                    intent=intent.value,
                    external_key=item.external_key,
                    created_at=at,
                )
            )
            return item.external_key, recorded.memory_id

    async def _record_question(self, turn_id: UUID, message: str, at: datetime) -> None:
        """The turn, before the model has been asked.

        Written first so that a model failure leaves a transcript saying a
        question was asked and not answered, rather than leaving no trace of
        having been asked at all. The answer is an update.
        """
        async with self._sessions.begin() as session:
            session.add(
                models.ChatMessage(
                    id=turn_id,
                    text=message,
                    intent=MessageIntent.QUESTION.value,
                    created_at=at,
                )
            )

    async def _answer_turn(self, turn: ChatTurn) -> ChatTurn:
        if self._answer is None:
            raise NoLanguageModel(
                "this deployment has no language model configured, so a question "
                "cannot be answered. Statements are stored as usual."
            )

        history = await self._history(turn.id)
        result = await self._answer(turn.text, k=self._k, history=history)
        citations = _citations_of(result)

        async with self._sessions.begin() as session:
            await session.execute(
                update(models.ChatMessage)
                .where(models.ChatMessage.id == turn.id)
                .values(
                    answer=result.answer,
                    answer_model=result.model_id,
                    answer_refused=result.refused,
                    citations=[citation.as_dict() for citation in citations],
                )
            )

        logger.info(
            "chat.answered",
            message_id=str(turn.id),
            intent=turn.intent.value,
            refused=result.refused,
            citations=len(citations),
        )
        return ChatTurn(
            id=turn.id,
            text=turn.text,
            intent=turn.intent,
            created_at=turn.created_at,
            external_key=turn.external_key,
            memory_id=turn.memory_id,
            answer=result.answer,
            answer_model=result.model_id,
            refused=result.refused,
            grounded=result.verification.grounded,
            citations=citations,
        )

    async def _history(self, before: UUID) -> list[ConversationTurn]:
        """The last few turns, oldest first, excluding this one.

        Capped, and the cap is the point: "what about the other one?" needs the
        turn that named the first one, and the turn before that adds referents
        the question was not about. More context is not more resolution — past
        three turns it is drift, and drift in a retrieval query is invisible
        because the results still look like results.
        """
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(models.ChatMessage)
                    .where(models.ChatMessage.id != before)
                    .order_by(models.ChatMessage.created_at.desc())
                    .limit(self._history_turns)
                )
            ).scalars().all()
        return [
            ConversationTurn(text=row.text, answer=row.answer) for row in reversed(rows)
        ]

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


async def history(
    sessions: async_sessionmaker[AsyncSession], *, limit: int = 100
) -> list[ChatTurn]:
    """The transcript, oldest first, with each stored turn's memory resolved.

    Ordered ascending for the reader — newest at the bottom is where a
    conversation puts its newest line — but *selected* descending, so a long
    history returns its tail rather than its head.

    The join is what an id column would have saved and is the price of surviving
    a replay. It is outer, and a null on the right is meaningful rather than a
    fault: the memory was deleted, and the turn still happened.
    """
    async with sessions() as session:
        rows = (
            await session.execute(
                select(models.ChatMessage)
                .order_by(models.ChatMessage.created_at.desc())
                .limit(max(1, min(limit, 500)))
            )
        ).scalars().all()
        resolved = await _memories_by_key(
            session, [row.external_key for row in rows if row.external_key]
        )
    return [_turn_of(row, resolved) for row in reversed(rows)]


async def _memories_by_key(
    session: AsyncSession, keys: Sequence[str]
) -> dict[str, UUID]:
    """Current, undeleted chat memories for these external keys.

    One query for the whole page rather than one per turn. A key that is absent
    from the result is a memory that has been deleted, which the caller renders
    as a turn with no link rather than as an error.
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
    sessions: async_sessionmaker[AsyncSession], memory_id: UUID
) -> MessageStatus | None:
    """How far this memory has got, and what it connects to.

    Returns None when there is no such memory, which a caller renders as a 404
    rather than as an empty connection line — "nothing connects to this" and
    "this does not exist" are different sentences.
    """
    async with sessions() as session:
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
    winner = models.Entity.__table__.alias("winner")
    return winner


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


def _turn_of(row: models.ChatMessage, resolved: dict[str, UUID]) -> ChatTurn:
    return ChatTurn(
        id=row.id,
        text=row.text,
        intent=MessageIntent(row.intent),
        created_at=row.created_at,
        external_key=row.external_key,
        memory_id=None if row.external_key is None else resolved.get(row.external_key),
        answer=row.answer,
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
