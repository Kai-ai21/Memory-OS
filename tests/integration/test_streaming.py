"""Answers that arrive in pieces, and the guarantees that must survive that.

Streaming is the first thing in this system that shows text *before* it has been
checked. Everything here is about that gap: a provider that cannot stream still
works, a draft that verification rejects is replaced rather than quietly kept, a
dropped connection resumes where it left off, and a stream that died mid-sentence
is marked rather than left looking finished.

The last two are the ones a demo never exercises and a laptop lid does.
"""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.application.answering import EventKind, StreamEvent
from memoryos.application.live import LiveBus, stream
from memoryos.application.ports import LanguageModel
from memoryos.domain.jobs import TransientError
from tests.integration.test_chat import build, drain
from tests.support.fakes import FakeLanguageModel

pytestmark = pytest.mark.integration

CORPUS = "postgres full-text search is faster than I expected on this corpus"


class Unstreamable(LanguageModel):
    """A provider with no `stream` of its own. The default fallback under test.

    Subclasses the port and overrides nothing but `complete`, which is exactly the
    shape of a provider that predates M10.3 — the inherited default is what makes
    it work.
    """

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "unstreamable@1"

    async def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        self.calls += 1
        return self.reply


class DiesMidSentence(LanguageModel):
    """A provider whose socket drops after a few chunks.

    The failure this milestone exists to handle honestly: text is already on
    screen, and the answer will never finish.
    """

    def __init__(self, *pieces: str) -> None:
        self.pieces = pieces

    @property
    def model_id(self) -> str:
        return "flaky@1"

    async def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        raise TransientError("the connection dropped")

    async def stream(
        self, system: str, user: str, *, max_tokens: int = 1024
    ) -> AsyncIterator[str]:
        for piece in self.pieces:
            yield piece
        raise TransientError("the connection dropped")


async def collect(
    answer: object, question: str
) -> tuple[list[StreamEvent], dict[EventKind, list[StreamEvent]]]:
    events: list[StreamEvent] = []
    async for event in answer.stream(question):  # type: ignore[attr-defined]
        events.append(event)
    grouped: dict[EventKind, list[StreamEvent]] = {}
    for event in events:
        grouped.setdefault(event.kind, []).append(event)
    return events, grouped


async def seeded(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession], model: LanguageModel
) -> object:
    chat = build(tmp_path, sessions, model)  # type: ignore[arg-type]
    await chat(CORPUS)
    await drain(tmp_path, sessions)
    return chat._answer


async def test_a_provider_that_cannot_stream_still_works(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The default fallback, which is why `stream` could be added at all.

    A provider that predates this milestone yields one chunk containing the
    finished answer, so every caller is written against the streaming shape and
    nothing branches on whether the configured provider supports it. The
    difference between the two is *when* text appears, never whether it does.
    """
    model = Unstreamable("Postgres full-text search is fast [1].")
    answer = await seeded(tmp_path, sessions, model)

    events, grouped = await collect(answer, "what did I say about postgres?")

    assert model.calls == 1, "the fallback went through `complete`, exactly once"
    assert [event.kind for event in events[:2]] == [
        EventKind.RETRIEVAL_STARTED,
        EventKind.RETRIEVAL_DONE,
    ], "the retrieval events arrive before any token, streaming or not"
    # One chunk carrying the whole answer, which is the honest shape of a
    # provider that cannot stream — not zero, and not a simulated drip.
    assert len(grouped[EventKind.TOKEN]) == 1
    assert grouped[EventKind.TOKEN][0].data["text"] == model.reply
    done = grouped[EventKind.DONE][0]
    assert done.data["answer"] == model.reply
    assert done.data["replacement"] is None
    assert done.data["grounded"] is True


async def test_verification_failure_replaces_the_streamed_answer(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The whole reason tokens are streamed as a *draft*.

    Verification cannot run per chunk — a citation marker can arrive split across
    two, so a per-chunk check would see `[` and `1]` and find neither — so it runs
    on the joined text after the stream ends. Which means text reaches the screen
    before it has been checked, and the only acceptable resolution when the check
    fails is to say so.

    `[9]` is the unambiguous signal: an index that was never supplied cannot be a
    matter of degree. Every softer verification result is reported as a mark on
    the sentence instead, because withdrawing a whole answer over one uncited
    clause would train somebody to ignore the withdrawal.
    """
    model = FakeLanguageModel("Postgres was benchmarked against Elasticsearch [9].")
    answer = await seeded(tmp_path, sessions, model)

    _, grouped = await collect(answer, "what did I say about postgres?")

    assert grouped[EventKind.TOKEN], "the draft was streamed before it was checked"
    done = grouped[EventKind.DONE][0]

    replacement = done.data["replacement"]
    assert isinstance(replacement, str)
    assert "withdrawn" in replacement
    # Names the fabricated index rather than saying "verification failed", so a
    # reader can see what the model did.
    assert "9" in replacement
    assert done.data["hallucinated_indices"] == [9]
    assert done.data["grounded"] is False
    # The draft is still carried, because hiding it would leave a reader unable to
    # see what was withdrawn — the interface swaps what it draws, it does not
    # pretend nothing happened.
    assert done.data["answer"] == model._responses[0]


async def test_an_interrupted_stream_is_marked_not_left_finished(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """A dropped socket after three chunks is not an answer.

    The failure arrives as an `error` event rather than an exception, and it
    carries the partial text — because by this point a 200 has been sent and half
    a body is on the wire, so raising would truncate the response and leave the
    client unable to tell a finished answer from a broken pipe.

    No `done` event, which is what the interface keys on: an answer without one is
    an answer that never finished.
    """
    model = DiesMidSentence("Postgres full-text ", "search is ", "faster than")
    answer = await seeded(tmp_path, sessions, model)

    events, grouped = await collect(answer, "what did I say about postgres?")

    assert len(grouped[EventKind.TOKEN]) == 3, "everything that arrived was shown"
    assert EventKind.DONE not in grouped, "an interrupted answer never completes"

    error = grouped[EventKind.ERROR][0]
    assert error is events[-1], "the error is the last thing on the stream"
    assert error.data["stage"] == "generation"
    # The partial text travels with the error, so an interface that lost the
    # tokens can still mark what there was.
    assert error.data["partial"] == "Postgres full-text search is faster than"
    assert "dropped" in str(error.data["message"])


async def test_the_event_stream_resumes_from_the_last_event_id() -> None:
    """Reconnect and catch up, or be told plainly that you cannot.

    Both halves matter and only one is obvious. Replaying what was missed is the
    feature; saying `gap` when the buffer no longer reaches back that far is what
    stops a client believing it has seen everything — an empty replay and a lost
    one look identical otherwise, and the second one silently drops a connection
    line that is the whole product.
    """
    bus = LiveBus(buffer_size=4)
    for index in range(3):
        bus.publish("memory_ready", {"memory_id": f"m{index}"})

    # A client that saw the first event asks for the rest.
    frames = await _read(stream(bus, last_event_id=1), until=2)
    assert "m1" in frames[0]
    assert "id: 2" in frames[0]
    assert "m2" in frames[1]

    # A client that saw everything gets nothing replayed and no gap: it is up to
    # date, which is different from having fallen behind.
    assert bus.since(3) == ([], True)

    # Past the buffer. Four more events push the first three out.
    for index in range(3, 8):
        bus.publish("memory_ready", {"memory_id": f"m{index}"})
    missed, resumable = bus.since(1)
    assert missed == []
    assert resumable is False, "the client must refetch rather than assume"

    opening = await _read(stream(bus, last_event_id=1), until=1)
    assert "event: gap" in opening[0]


async def test_a_live_event_reaches_an_already_open_stream() -> None:
    """The point of the bus: no refresh, no poll.

    Subscribing happens before the buffer is read, which is the ordering that
    makes the handover lossless — an event published between "read the buffer" and
    "start listening" would otherwise fall through the gap, and it would fall
    through it exactly when the system is busy.
    """
    bus = LiveBus()
    frames = stream(bus)
    # Nothing buffered, so the first read blocks until something is published.
    async def first() -> str:
        async for frame in frames:
            return frame
        raise AssertionError("the stream ended without publishing anything")

    reader: asyncio.Task[str] = asyncio.create_task(first())
    await asyncio.sleep(0)
    assert bus.subscribers == 1

    bus.publish("memory_ready", {"memory_id": "later", "job_type": "extract_entities"})
    frame = await asyncio.wait_for(reader, timeout=2)

    assert "event: memory_ready" in frame
    assert "later" in frame


async def _read(frames: AsyncIterator[str], *, until: int) -> list[str]:
    """The first `until` frames, ignoring keep-alive comments."""
    collected: list[str] = []
    async for frame in frames:
        if frame.startswith(":"):
            continue
        collected.append(frame)
        if len(collected) == until:
            break
    return collected
