"""The front door, end to end.

Five claims, and the shape of them is the milestone's own argument. Two are
about the split — a statement is stored and a question is not — one is about the
pipeline being genuinely shared rather than reimplemented, one is about the
transaction that makes "stored" mean something, and one is about the connection
that makes this a memory system rather than a notes app.

The language model is a fake throughout. `LanguageModel` is a port this project
owns, and what these tests can establish is that a question reaches the grounded
path unchanged and that a refusal survives being put behind a conversational
interface. Whether a real model refuses when it should is what
`tests/slow/test_real_answer.py` is for.
"""

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application import chat as chat_use_case
from memoryos.application import graph_projection
from memoryos.application.answering import AnswerQuestion
from memoryos.application.chat import CHAT_SOURCE_NAME, Chat, _message_item
from memoryos.application.embed import EmbedMemory
from memoryos.application.ingest import ingest_item
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.search import SearchMemories
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.message_intent import ChatRole, MessageIntent
from memoryos.domain.values import EntityType, SourceKind, TimeProvenance
from tests.support.fakes import FakeEmbedder, FakeLanguageModel, InMemoryGraphStore

pytestmark = pytest.mark.integration

STATEMENT = "postgres full-text search is faster than I expected on this corpus"
QUESTION = "what have I said about postgres?"
UNANSWERABLE = "what did I say about sourdough starters?"


def build(
    tmp_path: Path,
    sessions: async_sessionmaker[AsyncSession],
    model: FakeLanguageModel | None = None,
) -> Chat:
    """A chat wired to the same objects the container wires."""
    blobs = FilesystemBlobStore(tmp_path / "blobs")
    embedder = FakeEmbedder()
    answer = (
        None
        if model is None
        else AnswerQuestion(
            sessions,
            SearchMemories(
                sessions,
                embedder,
                PgVectorStore(sessions, embedder, default_ef_search=100),
                PostgresKeywordStore(sessions),
            ),
            model,
            embedder,
        )
    )
    return Chat(sessions, blobs, answer)


async def drain(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Normalize and embed everything the chat queued.

    The real worker, minus the process. What matters is that the jobs it runs
    are the jobs the chat path enqueued, and that they are the same handlers a
    file's jobs go through — no chat branch anywhere in here.
    """
    blobs = FilesystemBlobStore(tmp_path / "blobs")
    embedder = FakeEmbedder()
    normalize = NormalizeMemory(sessions, blobs, build_parsers(), StructuralChunker(embedder))
    embed = EmbedMemory(sessions, embedder, PostgresEmbeddingCache(sessions))

    for job_type, run in ((JobType.NORMALIZE_MEMORY, normalize), (JobType.EMBED_MEMORY, embed)):
        async with sessions() as session:
            payloads = (
                await session.execute(
                    select(models.Job.payload).where(models.Job.job_type == job_type.value)
                )
            ).all()
        for (payload,) in payloads:
            await run(UUID(payload["memory_id"]))
        async with sessions.begin() as session:
            await session.execute(
                delete(models.Job).where(models.Job.job_type == job_type.value)
            )


async def chat_source(sessions: async_sessionmaker[AsyncSession]) -> Source:
    async with sessions() as session:
        found = (
            await session.execute(
                select(models.Source).where(models.Source.kind == SourceKind.CHAT.value)
            )
        ).scalars().one()
    return Source(
        id=found.id, kind=SourceKind.CHAT, name=found.name, config=found.config
    )


async def counts(sessions: async_sessionmaker[AsyncSession]) -> tuple[int, int, int]:
    """Memories, pending normalization jobs, transcript rows.

    Three numbers rather than one, because the invariant is that they move
    together: a rollback that left any one of them behind is the failure.
    """
    async with sessions() as session:
        memories = (
            await session.execute(select(func.count(models.Memory.id)))
        ).scalar_one()
        jobs = (
            await session.execute(
                select(func.count(models.Job.id)).where(
                    models.Job.job_type == JobType.NORMALIZE_MEMORY.value
                )
            )
        ).scalar_one()
        turns = (
            await session.execute(select(func.count(models.ChatMessage.id)))
        ).scalar_one()
    return int(memories), int(jobs), int(turns)


async def test_a_statement_is_stored_and_a_question_is_not(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The split, asserted from both sides.

    The question half is the one that matters. Storing an answer's question
    would put the question's own words into the retrieval set, and every later
    answer would be retrieving over a corpus that includes what people asked
    about it — the corpus starting to describe itself.
    """
    chat = build(tmp_path, sessions, FakeLanguageModel("Nothing retrieved covers this."))

    stored = (await chat(STATEMENT)).user
    assert stored.intent is MessageIntent.STATEMENT
    assert stored.memory_id is not None

    asked = await chat(QUESTION)
    assert asked.user.intent is MessageIntent.QUESTION
    assert asked.user.memory_id is None
    assert asked.assistant is not None

    async with sessions() as session:
        keys = (
            await session.execute(
                select(models.Memory.external_key, models.Memory.content_hash)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Source.kind == SourceKind.CHAT.value)
            )
        ).all()
    assert len(keys) == 1, "the question left no memory behind"

    # Both turns are in the transcript, which is not the corpus. That is what
    # lets the message list survive a reload without the question becoming
    # retrievable.
    async with sessions() as session:
        transcript = (
            await session.execute(select(func.count(models.ChatMessage.id)))
        ).scalar_one()
    # Three rows for two sends: one row per *turn* as of M10.1, so the question
    # and its answer are two messages with two ordinals rather than one row with
    # an answer column. The statement is one row and is answered by nothing.
    assert transcript == 3


async def test_a_message_flows_through_the_whole_pipeline_and_becomes_searchable(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Content hash, artifact, event, memory, chunks, embedding — no chat branch.

    Every row asserted here is written by code that has never heard of chat. If
    this passes, the milestone's "no parallel path" requirement holds at the only
    level that matters: the same tables, filled by the same handlers.
    """
    chat = build(tmp_path, sessions)
    stored = (await chat(STATEMENT)).user
    assert stored.memory_id is not None

    async with sessions() as session:
        memory = await session.get(models.Memory, stored.memory_id)
        assert memory is not None
        artifact = await session.get(models.RawArtifact, memory.content_hash)
        events = (
            await session.execute(
                select(models.IngestionEvent).where(
                    models.IngestionEvent.external_key == memory.external_key
                )
            )
        ).scalars().all()

    assert artifact is not None, "the bytes were addressed and recorded"
    assert len(events) == 1, "one artifact_observed event, as for any file"
    # The first genuinely reliable date in this corpus. A file's `occurred_at` is
    # an mtime and is `filesystem`; this is when somebody pressed enter.
    assert memory.occurred_at_source == TimeProvenance.DECLARED.value
    assert memory.occurred_at is not None

    before = await chat_use_case.status(sessions, stored.memory_id)
    assert before is not None
    assert not before.searchable, "nothing is searchable before the worker runs"

    await drain(tmp_path, sessions)

    after = await chat_use_case.status(sessions, stored.memory_id)
    assert after is not None
    assert after.chunks >= 1
    assert after.searchable

    # And retrievable, which is the claim `searchable` is making.
    embedder = FakeEmbedder()
    search = SearchMemories(
        sessions,
        embedder,
        PgVectorStore(sessions, embedder, default_ef_search=100),
        PostgresKeywordStore(sessions),
    )
    hits = await search("postgres full-text search", k=5)
    assert any(hit.memory_id == stored.memory_id for hit in hits.hits)


class DeliberateFailure(RuntimeError):
    """Something going wrong after the memory was written and before it committed."""


async def test_the_memory_and_its_normalization_job_commit_together(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """A rollback leaves neither, which is what makes "stored" mean something.

    The queue is a table precisely so that there is no window in which a memory
    exists and the job that would chunk it does not. This asserts the window is
    closed in the chat path as well — a message committed without its job would
    be one the interface says it kept and search can never find, and a job
    committed without its memory would be one the worker fails permanently.

    Driven through `ingest_item` and the transcript row in one transaction,
    which is exactly what `Chat._store` opens, with a failure injected after both
    writes. Going through `Chat` itself and contriving a database error would
    test the same boundary less legibly.
    """
    chat = build(tmp_path, sessions)
    first = (await chat(STATEMENT)).user
    assert first.memory_id is not None

    source = await chat_source(sessions)
    before = await counts(sessions)

    doomed = "a thought that will not survive"
    blobs = FilesystemBlobStore(tmp_path / "blobs")

    with pytest.raises(DeliberateFailure):
        async with sessions.begin() as session:
            item = _message_item(new_id(), doomed, first.created_at)
            recorded = await ingest_item(session, blobs, source, item)
            assert recorded is not None
            session.add(
                models.ChatMessage(
                    id=new_id(),
                    session_id=first.session_id,
                    role=ChatRole.USER.value,
                    content=doomed,
                    ordinal=99,
                    intent=MessageIntent.STATEMENT.value,
                    external_key=item.external_key,
                )
            )
            await session.flush()
            raise DeliberateFailure("after the memory, the job and the turn were written")

    after = await counts(sessions)
    assert after == before, (
        "the memory, its normalization job and the transcript row went together"
    )


async def test_an_unanswerable_question_still_refuses_in_the_chat_path(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The guardrail, behind a conversational interface.

    This is the test the milestone is most worried about. A chat box is where a
    refusal feels wrong — "Hmm, I'm not sure, but maybe…" is what a conversation
    sounds like — and softening it is how the guardrail stops being one. Nothing
    in the chat path may add a word to what the answering path produced.

    Two halves, because there are two ways a refusal happens and both have to
    survive the trip. The first is the stronger: with nothing retrievable, the
    refusal is reached *without a model call at all*, so there is nothing that
    could have invented an answer. The second is the ordinary one, where the
    model declines and the chat path carries its words through unchanged.
    """
    model = FakeLanguageModel("Sourdough is made with a wild yeast starter.")
    chat = build(tmp_path, sessions, model)

    empty = (await chat(UNANSWERABLE)).assistant
    assert empty is not None

    assert empty.refused is True
    assert empty.citations == []
    assert model.calls == [], "nothing retrieved, so the model was never asked"
    # The fake would have answered from its training data given the chance.
    assert "Sourdough" not in empty.content

    # Now with a corpus that has something in it and nothing about this.
    await chat(STATEMENT)
    await drain(tmp_path, sessions)

    declining = "The passages do not cover sourdough starters."
    chat = build(tmp_path, sessions, FakeLanguageModel(declining))
    turn = (await chat(UNANSWERABLE)).assistant
    assert turn is not None

    assert turn.refused is True
    # Verbatim. Not prefixed, not hedged, not wrapped in an apology.
    assert turn.content == declining
    assert turn.citations == []

    # And the verdict is on the record, so the refusal rate over a chat log is a
    # number rather than a pattern match over prose read back later.
    async with sessions() as session:
        row = await session.get(models.ChatMessage, turn.id)
    assert row is not None
    assert row.role == ChatRole.ASSISTANT.value
    assert row.answer_refused is True
    assert row.content == declining


async def test_two_messages_sharing_an_entity_are_linked_in_the_graph(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The connection, all the way through to a traversal.

    This is the mechanism behind "connects to 3 earlier memories via postgres",
    and it is the reason a message is its own memory rather than a line in a
    conversation-shaped one. Two thoughts typed minutes apart connect because
    they are about the same thing; the graph is where that stops being a claim.

    The entities are written by hand rather than extracted, for the reason every
    graph test here gives: what is under test is the projection and the
    connection query, and a model call would make the test measure the model.
    """
    chat = build(tmp_path, sessions)
    first = (await chat("the postgres keyword half finds SKIP LOCKED immediately")).user
    second = (await chat("postgres full-text search is cheaper than I assumed")).user
    await drain(tmp_path, sessions)

    assert first.memory_id is not None and second.memory_id is not None

    entity_id = new_id()
    async with sessions.begin() as session:
        session.add(
            models.Entity(
                id=entity_id,
                name="postgres",
                canonical_name="postgres",
                type=EntityType.TECHNOLOGY.value,
                confidence=0.9,
            )
        )
        for memory_id in (first.memory_id, second.memory_id):
            chunk_id, content = (
                await session.execute(
                    select(models.MemoryChunk.id, models.MemoryChunk.content)
                    .where(models.MemoryChunk.memory_id == memory_id)
                    .order_by(models.MemoryChunk.ordinal)
                    .limit(1)
                )
            ).one()
            start = content.index("postgres")
            session.add(
                models.EntityMention(
                    id=new_id(),
                    entity_id=entity_id,
                    memory_id=memory_id,
                    chunk_id=chunk_id,
                    char_start=start,
                    char_end=start + len("postgres"),
                    confidence=0.9,
                    extractor_version="test@1",
                )
            )
        # Extraction has run, whatever it found — the flag the connection line
        # stops waiting on.
        for memory_id in (first.memory_id, second.memory_id):
            memory = await session.get(models.Memory, memory_id)
            assert memory is not None
            memory.entity_extractor_version = "test@1"

    graph = InMemoryGraphStore()
    await graph_projection.rebuild(sessions, graph)

    reached = await graph.reach([entity_id], depth=2)
    linked = {found.memory_id for found in reached}
    assert {first.memory_id, second.memory_id} <= linked, (
        "both messages are reachable from the entity they share"
    )

    # And the connection line says so, in the words the interface renders.
    status = await chat_use_case.status(sessions, second.memory_id)
    assert status is not None
    assert status.extracted
    assert [connection.name for connection in status.connections] == ["postgres"]
    assert status.connected_memories == 1


async def test_the_chat_source_is_created_once_and_is_not_walkable(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """One source, however many messages, and no connector behind it."""
    chat = build(tmp_path, sessions)
    await chat("a first thought about the queue")
    await chat("a second thought about the queue")

    async with sessions() as session:
        sources = (
            await session.execute(
                select(models.Source).where(models.Source.kind == SourceKind.CHAT.value)
            )
        ).scalars().all()

    assert len(sources) == 1
    assert sources[0].name == CHAT_SOURCE_NAME
    # No root, no include globs, no cursor. There is nothing to walk and nothing
    # to resume from, which is exactly what a pushed source looks like.
    assert sources[0].config == {}
    assert sources[0].cursor == {}
    assert sources[0].last_sync_at is None
