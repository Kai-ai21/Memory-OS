"""Conversations that persist, resume, and survive a rebuild.

Four claims. Two are about the boundary — when a conversation ends and what it
gets called — one is about the cap that keeps a follow-up from becoming a
different question, and the last is the one the schema was designed around: a
session outlives a replay while every memory it points at is thrown away and
built again.

**A session is a view, and every test here is written to keep that true.** None
of them assert that a session holds anything. What they assert is that it can be
navigated to, drawn with a name, resumed with three turns of context, and found
again after the corpus underneath it has been rebuilt from the log.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application import chat as chat_use_case
from memoryos.application.embed import EmbedMemory
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.replay import ReplayCorpus
from memoryos.domain.message_intent import ChatRole
from memoryos.domain.sessions import SESSION_GAP, title_for
from tests.integration.test_chat import build, drain
from tests.support.fakes import FakeEmbedder, FakeLanguageModel, InMemoryGraphStore

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)


async def test_a_message_after_thirty_minutes_of_silence_starts_a_new_session(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The boundary, from both sides of it.

    Time is passed in rather than slept, which is the only way this is testable at
    all — and it is passed in because `Chat` takes a `now`, which it takes because
    a rule about clocks that cannot be exercised is a rule nobody knows the shape
    of.

    The near miss is asserted as well as the hit. A threshold tested only from the
    far side passes for any threshold at all.
    """
    chat = build(tmp_path, sessions)

    # The gap is measured from the last *activity*, not from the session's start:
    # a conversation that keeps going keeps going, however long it runs. So the
    # third message has to be a full gap after the second one.
    just_inside = NOW + SESSION_GAP - timedelta(seconds=1)
    first = await chat("the queue is a table so the job commits with it", now=NOW)
    inside = await chat("and that is why there is no broker", now=just_inside)
    after = await chat("back the next morning, on the graph", now=just_inside + SESSION_GAP)

    assert inside.session_id == first.session_id, "a pause shorter than the gap continues"
    assert after.session_id != first.session_id, "silence of the full gap opens a new one"

    listed = await chat_use_case.sessions(sessions)
    assert len(listed) == 2
    # Newest activity first, which is the order the rail draws.
    assert listed[0].id == after.session_id
    assert listed[1].id == first.session_id
    assert [row.message_count for row in listed] == [1, 2]


async def test_an_explicit_session_outranks_the_clock(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Clicking a conversation is a decision; thirty minutes is a guess.

    Both directions. `new_session` opens one inside the window, and a
    `session_id` resumes one long outside it — a conversation from last week that
    somebody opened and typed into is not a new conversation, and a clock that
    overrode them would make the rail unusable.
    """
    chat = build(tmp_path, sessions)

    first = await chat("a thought about chunk overlap", now=NOW)
    split = await chat(
        "a different subject entirely",
        now=NOW + timedelta(minutes=1),
        new_session=True,
    )
    assert split.session_id != first.session_id

    resumed = await chat(
        "one more on chunk overlap, a week later",
        now=NOW + timedelta(days=7),
        session_id=first.session_id,
    )
    assert resumed.session_id == first.session_id
    assert resumed.user.ordinal == 1


async def test_the_title_comes_from_the_first_user_message(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Derived, from the first message, and not renamed by later ones.

    The second half is the part worth pinning. A title that tracked the latest
    message would rename a conversation out from under somebody halfway through
    reading the list, which is the kind of interface bug that reads as data loss.
    """
    chat = build(tmp_path, sessions)
    opening = (
        "declared dates are the first honest ones here. everything "
        "filesystem-shaped is an mtime, which is a different fact"
    )

    await chat(opening, now=NOW)
    await chat("and the timeline should weight them", now=NOW + timedelta(minutes=1))

    listed = await chat_use_case.sessions(sessions)
    assert len(listed) == 1
    assert listed[0].title == title_for(opening)
    # Cut at the sentence boundary inside the budget, so the title is a clause
    # rather than a truncation — no ellipsis, and nothing mid-word.
    assert listed[0].title == "declared dates are the first honest ones here"
    assert listed[0].message_count == 2


async def test_only_the_last_three_turns_reach_the_answering_path(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The cap, measured on the prompt the model actually received.

    A turn is a user message and whatever answer followed it, so three turns is up
    to six rows. Asserted against the prompt rather than against the history list,
    because the list is an implementation detail and the prompt is the thing that
    changes the answer.

    The question is referential on purpose — `refers_back` gates whether the
    conversation reaches the *retrieval query* at all, and a self-contained
    question would carry none of it and prove nothing about the cap.
    """
    model = FakeLanguageModel("Nothing here covers it.")
    chat = build(tmp_path, sessions, model)

    marks = [f"turn number {index} about the reranker" for index in range(5)]
    for index, text in enumerate(marks):
        await chat(text, now=NOW + timedelta(minutes=index))
    await drain(tmp_path, sessions)

    await chat("what about the other one?", now=NOW + timedelta(minutes=6))

    prompt = model.last_user_prompt
    # Asserted against the `- typed:` lines rather than against the raw text,
    # because every one of these messages is also *in the corpus* and can come
    # back as a retrieved passage. A bare substring check would pass on a prompt
    # that carried no conversation at all — which is exactly the bug this test
    # exists to catch.
    quoted = [line for line in prompt.splitlines() if line.startswith("- typed:")]
    assert len(quoted) == 3, quoted
    for text in marks[2:]:
        assert any(text in line for line in quoted), text
    for text in marks[:2]:
        assert not any(text in line for line in quoted), text

    # And the conversation is labelled as not being evidence, so the grounding
    # rule still governs it: only the numbered passages may be cited.
    assert "It is not evidence" in prompt


async def test_context_does_not_cross_a_session_boundary(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Which is the one thing sessions are load-bearing for.

    Carrying three turns across a boundary the interface drew would make the
    boundary a lie — the rail would show two conversations and the model would be
    reading one.
    """
    model = FakeLanguageModel("Nothing here covers it.")
    chat = build(tmp_path, sessions, model)

    await chat("the hub ratio suppresses well-connected entities", now=NOW)
    await drain(tmp_path, sessions)
    await chat("what about the other one?", now=NOW + SESSION_GAP)

    # No conversation block at all: the new session has nothing before this
    # question in it. Asserted on the block rather than on the words, because the
    # earlier message is in the corpus and may legitimately be retrieved as a
    # passage — which is a different thing from being carried as context.
    assert "- typed:" not in model.last_user_prompt


async def test_sessions_survive_a_replay_while_their_memories_are_rebuilt(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The claim the schema was designed around, and the reason for `external_key`.

    A replay truncates `memories` and rebuilds every row from the ingestion log,
    minting a fresh id for each. So a transcript that pointed at a memory *id*
    would have been taken by `TRUNCATE memories CASCADE` whatever set it was
    classified in, and any id that somehow survived would dangle.

    The external key is what the log records. It comes back on the rebuilt memory
    unchanged, so the link resolves again having had nothing preserved — which is
    what "key on nothing that changes" means, and what makes this test pass
    without a single line of snapshot machinery.
    """
    chat = build(tmp_path, sessions)
    first = await chat("the replay rebuilds every derived row from the log", now=NOW)
    second = await chat("and the transcript is not derived", now=NOW + timedelta(minutes=1))
    await drain(tmp_path, sessions)

    before = await chat_use_case.messages(sessions, first.session_id)
    assert [message.memory_id for message in before] == [
        first.user.memory_id,
        second.user.memory_id,
    ]
    assert all(message.memory_id is not None for message in before)

    embedder = FakeEmbedder()
    blobs = FilesystemBlobStore(tmp_path / "blobs")
    replay = ReplayCorpus(
        sessions,
        make_normalize=lambda factory: NormalizeMemory(
            factory, blobs, build_parsers(), StructuralChunker(embedder), enqueue_followup=False
        ),
        make_embed=lambda factory: EmbedMemory(
            factory, embedder, PostgresEmbeddingCache(factory), enqueue_followup=False
        ),
        make_shadow=None,
        blobs=blobs,
        graph=InMemoryGraphStore(),
    )
    report = await replay()
    assert report.memories == 2

    async with sessions() as session:
        rebuilt = {
            key: memory_id
            for key, memory_id in await session.execute(
                select(models.Memory.external_key, models.Memory.id)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Source.kind == "chat", models.Memory.is_current.is_(True))
            )
        }

    listed = await chat_use_case.sessions(sessions)
    assert len(listed) == 1, "the session itself was never touched"
    assert listed[0].id == first.session_id
    assert listed[0].title == title_for("the replay rebuilds every derived row from the log")

    after = await chat_use_case.messages(sessions, first.session_id)
    assert [message.content for message in after] == [
        message.content for message in before
    ]
    assert [message.role for message in after] == [ChatRole.USER, ChatRole.USER]

    # Every link resolves, and every id is a *different* id than before — which is
    # the whole point. A test where the ids happened to match would be passing for
    # the wrong reason.
    assert all(message.memory_id is not None for message in after)
    assert [message.memory_id for message in after] == [
        rebuilt[str(message.external_key)] for message in after
    ]
    assert {message.memory_id for message in after}.isdisjoint(
        {message.memory_id for message in before}
    )


def test_the_gap_and_the_title_are_pure(tmp_path: Path) -> None:
    """No database, no clock of its own. Both rules are functions of their input.

    Here rather than in `tests/unit` only because the rest of this file is, and a
    reader looking for what decides a boundary should find both halves together.
    """
    from memoryos.domain.sessions import TITLE_CHARS, starts_new_session

    assert starts_new_session(NOW, None) is True
    assert starts_new_session(NOW + SESSION_GAP, NOW) is True
    assert starts_new_session(NOW + SESSION_GAP - timedelta(seconds=1), NOW) is False

    assert title_for("") is None
    assert title_for("   \n ") is None
    assert title_for("short enough") == "short enough"
    # Cut at a word boundary with an ellipsis when there is no clause to cut at.
    long = "a" * 20 + " " + "b" * 60
    cut = title_for(long)
    assert cut is not None
    assert cut.endswith("…")
    assert len(cut) <= TITLE_CHARS + 1


def test_a_session_id_is_not_in_the_memory_key(tmp_path: Path) -> None:
    """A session is a view, asserted at the level where it would leak.

    The external key is the memory's durable identity. Putting a session in it
    would make moving a conversation a corpus migration, and would make the graph
    able to tell which messages were typed together — which is precisely the
    grouping M10.0 argued should not exist.
    """
    from memoryos.application.chat import _message_item

    item = _message_item(UUID("11111111-1111-7111-8111-111111111111"), "a thought", NOW)
    assert item.external_key == "2026-08-17/11111111-1111-7111-8111-111111111111.md"
