"""When the system speaks unasked, and the four rules that stop it.

The milestone asks for exactly four properties, and each is one test here:

* below-threshold context is not surfaced, **and the reason is recorded**,
* the same context is not surfaced twice inside the window,
* dismissed context is suppressed for very much longer,
* and repeated dismissals raise *that focus's* threshold.

The arithmetic behind all four is `tests/unit/test_surfacing_gate.py`. What needs
a database is the half that makes them true over time: a log that remembers what
was shown, and a threshold derived from feedback rather than stored beside it.

Contexts are built by hand rather than assembled from a corpus, and that is
deliberate rather than a shortcut. The gate's input is an `AssembledContext` and
its rules are about scores and repetition; driving it through retrieval would
make every assertion here depend on what the fake embedder happens to rank
first, and a test that fails when the corpus changes is a test nobody trusts.
Assembly over a real corpus is `test_context_engine.py`'s job.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application import surfacing
from memoryos.application.context_engine import (
    AssembledContext,
    ContextItem,
    ContextRequest,
    ContextSource,
)
from memoryos.domain.context import ContextCategory
from memoryos.domain.events import Event, EventKind
from memoryos.domain.ids import new_id
from memoryos.domain.surfacing import (
    SINGLE_ROUTE_BEST,
    SurfaceReason,
    threshold_for,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)

# Over the default bar of 1.8, and over the 2.4 one dismissal produces, and
# under the 3.0 that two produce. Chosen to make the adaptation test able to
# watch a fixed context stop clearing a moving bar.
STRONG = 2.9 * SINGLE_ROUTE_BEST
# What one route ranking something first scores, which is the most a
# single-source item can ever reach.
WEAK = SINGLE_ROUTE_BEST


def context(
    focus: str,
    *,
    score: float = STRONG,
    keys: tuple[str, ...] = ("memory:a", "memory:b", "memory:c"),
) -> AssembledContext:
    """A context whose top item is not the focused file.

    `external_key` is set to something other than the focus on every item, so
    "the reader plausibly does not already have this open" is satisfied and the
    test is about the score. The one test that needs the opposite says so.
    """
    return AssembledContext(
        focus=focus,
        items=[
            ContextItem(
                key=key,
                title=f"self::{key}",
                category=ContextCategory.CODE,
                text="…",
                tokens=10,
                position=position,
                sources={ContextSource.RETRIEVAL: position, ContextSource.TEMPORAL: 1},
                # Only the top item's score is consulted, and it is the one that
                # has to clear the bar; the rest descend so `max` has something
                # to choose.
                relevance=score / position,
                redundancy=0.0,
                external_key=f"other/{key}.py",
            )
            for position, key in enumerate(keys, 1)
        ],
    )


def event(kind: EventKind, **payload: str) -> Event:
    return Event(
        id=new_id(),
        kind=kind,
        source="test",
        payload=payload,
        occurred_at=NOW,
        received_at=NOW,
    )


async def rows(
    sessions: async_sessionmaker[AsyncSession],
) -> list[models.SurfacingLog]:
    async with sessions() as session:
        return list(
            (
                await session.execute(
                    select(models.SurfacingLog).order_by(
                        models.SurfacingLog.decided_at
                    )
                )
            ).scalars()
        )


# --------------------------------------------------------------------------
# 1. Below the bar, and the reason is on record
# --------------------------------------------------------------------------


async def test_below_threshold_context_is_not_surfaced_and_says_why(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """**"Why didn't it show me anything?" is answerable, or this is a black box.**

    The refusal is written to the same table as a surfacing, with the score it
    reached and the bar it did not. A system that logged only what it did could
    not tell a gate that refused from a handler that never ran, and both look
    identical from outside: nothing happened.
    """
    outcome = await surfacing.surface(sessions, context("a.py", score=WEAK))

    assert outcome.decision.surface is False
    assert outcome.decision.reason is SurfaceReason.BELOW_THRESHOLD

    (row,) = await rows(sessions)
    assert row.surfaced_at is None
    assert row.reason == SurfaceReason.BELOW_THRESHOLD.value
    assert row.score == pytest.approx(WEAK)
    assert row.threshold == pytest.approx(threshold_for(dismissed=0, acted_on=0))
    # And the near miss is legible without re-running anything.
    assert row.score < row.threshold
    assert row.top_title is not None


async def test_a_context_of_only_the_focused_file_is_refused(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The second half of Step 1's gate: something the reader does not have open.

    A context whose only item is the file already in front of them clears no bar
    worth clearing, however high it scores — the whole content of the
    interruption would be "you are looking at this file".
    """
    assembled = context("src/app/search.py")
    assembled.items = [
        ContextItem(
            key="memory:self",
            title="self::src/app/search.py",
            category=ContextCategory.CODE,
            text="…",
            tokens=10,
            position=1,
            sources={ContextSource.RETRIEVAL: 1, ContextSource.TEMPORAL: 1},
            relevance=STRONG * 10,
            redundancy=0.0,
            external_key="src/app/search.py",
        )
    ]

    outcome = await surfacing.surface(sessions, assembled)

    assert outcome.decision.surface is False
    assert outcome.decision.reason is SurfaceReason.NOTHING_NEW


# --------------------------------------------------------------------------
# 2. Not twice
# --------------------------------------------------------------------------


async def test_the_same_context_is_not_surfaced_twice_within_the_window(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Once is context. Twice is a notification, and nobody asked for one."""
    first = await surfacing.surface(sessions, context("a.py"), now=NOW)
    assert first.decision.surface is True

    again = await surfacing.surface(
        sessions, context("a.py"), now=NOW + timedelta(minutes=30)
    )

    assert again.decision.surface is False
    assert again.decision.reason is SurfaceReason.ALREADY_SURFACED

    # Both decisions are recorded; only one of them was an interruption.
    assert [row.surfaced_at is not None for row in await rows(sessions)] == [True, False]


async def test_a_different_focus_is_a_different_question(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Suppression is per focus, so the same items about a different file are
    a new thing to say rather than a repeat."""
    await surfacing.surface(sessions, context("a.py"), now=NOW)
    other = await surfacing.surface(sessions, context("b.py"), now=NOW)

    assert other.decision.surface is True


async def test_the_window_ends(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The same file tomorrow is a genuinely new question — the same argument
    M6.0's partial index makes about re-focusing a file the next morning."""
    await surfacing.surface(sessions, context("a.py"), now=NOW)
    later = await surfacing.surface(
        sessions, context("a.py"), now=NOW + timedelta(hours=5)
    )

    assert later.decision.surface is True


# --------------------------------------------------------------------------
# 3. Dismissed means dismissed
# --------------------------------------------------------------------------


async def test_dismissed_context_is_suppressed_for_far_longer(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """**Repeating something somebody refused is the fastest way to be muted.**

    The interesting instant is the one *after* the ordinary repeat window has
    passed: five hours later an undismissed context would be offered again, and
    a dismissed one must not be. Thirty days later it is a new question.
    """
    first = await surfacing.surface(sessions, context("a.py"), now=NOW)
    assert await surfacing.dismiss(sessions, first.id) is True

    past_the_repeat_window = await surfacing.surface(
        sessions, context("a.py"), now=NOW + timedelta(hours=5)
    )
    assert past_the_repeat_window.decision.surface is False
    assert past_the_repeat_window.decision.reason is SurfaceReason.DISMISSED

    still_quiet = await surfacing.surface(
        sessions, context("a.py"), now=NOW + timedelta(days=20)
    )
    assert still_quiet.decision.reason is SurfaceReason.DISMISSED


async def test_a_verdict_cannot_be_overwritten(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The dismissal rate is the number this milestone is judged on, and a rate
    that depends on which click landed last is not a measurement."""
    first = await surfacing.surface(sessions, context("a.py"), now=NOW)
    await surfacing.dismiss(sessions, first.id)

    with pytest.raises(surfacing.AlreadyRated):
        await surfacing.mark_useful(sessions, first.id)


async def test_feedback_on_something_never_shown_is_refused(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    refused = await surfacing.surface(sessions, context("a.py", score=WEAK))

    assert await surfacing.dismiss(sessions, refused.id) is False
    assert await surfacing.dismiss(sessions, UUID(int=0)) is False


# --------------------------------------------------------------------------
# 4. The threshold learns, per focus
# --------------------------------------------------------------------------


async def test_repeated_dismissals_raise_that_focuss_threshold(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """**Per focus, not global — one noisy file must not silence the system.**

    The same context, unchanged, is offered three times a day apart. It clears
    the default bar and is dismissed; it clears the raised bar and is dismissed;
    the third time the bar has passed it and the same context is refused. The
    identical context about `quiet.py` is surfaced in the same breath, which is
    the half of this property a global counter gets wrong.

    Each round uses different item keys, or *suppression* would refuse the second
    round and this would be testing the wrong rule.
    """
    start = threshold_for(dismissed=0, acted_on=0)
    assert await surfacing.threshold_for_focus(sessions, "noisy.py") == start

    for round_number in range(2):
        keys = (f"memory:{round_number}a", f"memory:{round_number}b")
        shown = await surfacing.surface(
            sessions,
            context("noisy.py", keys=keys),
            now=NOW + timedelta(days=round_number),
        )
        assert shown.decision.surface is True, f"round {round_number} should surface"
        await surfacing.dismiss(sessions, shown.id)
        raised = await surfacing.threshold_for_focus(sessions, "noisy.py")
        assert raised > start, "each dismissal moves the bar"

    third = await surfacing.surface(
        sessions,
        context("noisy.py", keys=("memory:xa", "memory:xb")),
        now=NOW + timedelta(days=9),
    )
    assert third.decision.surface is False
    assert third.decision.reason is SurfaceReason.BELOW_THRESHOLD
    # Refused by the bar having moved, not by the score having changed.
    assert third.decision.score == pytest.approx(STRONG)

    # And the rest of the system is untouched.
    assert await surfacing.threshold_for_focus(sessions, "quiet.py") == start
    elsewhere = await surfacing.surface(
        sessions, context("quiet.py"), now=NOW + timedelta(days=9)
    )
    assert elsewhere.decision.surface is True


async def test_useful_context_lowers_the_bar_for_that_focus(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The other direction, and it moves more slowly on purpose."""
    shown = await surfacing.surface(sessions, context("a.py"), now=NOW)
    await surfacing.mark_useful(sessions, shown.id)

    lowered = await surfacing.threshold_for_focus(sessions, "a.py")
    assert lowered < threshold_for(dismissed=0, acted_on=0)


# --------------------------------------------------------------------------
# The trigger path, and the data source that is absent
# --------------------------------------------------------------------------


class FakeAssembler:
    """Stands in for the context engine, and counts being called.

    A real one would load an embedder and a cross-encoder, which is twenty
    seconds and most of a gigabyte to answer a question about *whether* it was
    called.
    """

    def __init__(self, result: AssembledContext | None = None) -> None:
        self.calls: list[str] = []
        self._result = result

    async def __call__(self, request: ContextRequest) -> AssembledContext:
        self.calls.append(request.focus)
        return self._result or context(request.focus)


async def test_a_meeting_event_drives_the_whole_path(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """**Step 4, and the data source is absent rather than faked.**

    No calendar is connected to this system and no `meeting_upcoming` has ever
    arrived from one. What is asserted here is that the handler works against a
    manually emitted event — so the day a connector exists, the path it feeds is
    already built and tested rather than being written under time pressure.
    """
    assembler = FakeAssembler()
    handler = surfacing.build_handler(
        sessions, lambda: assembler, lambda incoming: str(incoming.payload["title"])
    )

    await handler.handle(event(EventKind.MEETING_UPCOMING, title="Retro: phase 6"))

    assert assembler.calls == ["Retro: phase 6"]
    (row,) = await rows(sessions)
    assert row.focus == "Retro: phase 6"
    assert row.trigger_kind == EventKind.MEETING_UPCOMING.value
    assert row.surfaced_at is not None


async def test_a_focused_file_never_causes_an_assembly(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """M6.1's precompute policy, unchanged and now load-bearing twice.

    `FILE_FOCUSED` fires on every file glanced at. Assembling for each to decide
    whether to interrupt would burn a second of compute per glance to produce,
    in the overwhelming majority of cases, silence — which is the push-system
    failure this phase opened by naming, arriving through the door marked
    "restraint".

    So the handler reads the cache and stops. Nothing is cached here, so nothing
    is assembled and nothing is decided.
    """
    assembler = FakeAssembler()
    handler = surfacing.build_handler(
        sessions, lambda: assembler, lambda incoming: str(incoming.payload["path"])
    )

    await handler.handle(event(EventKind.FILE_FOCUSED, path="a.py"))

    assert assembler.calls == []
    assert await rows(sessions) == []


# --------------------------------------------------------------------------
# What it reports
# --------------------------------------------------------------------------


async def test_the_dismissal_rate_counts_every_interruption(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The harsher denominator is the headline, and both are reported.

    An interruption nobody bothered to rate was still an interruption. Counting
    only the rated ones would let a feature that is ignored score exactly like
    one that is valued.
    """
    for index, focus in enumerate(["a.py", "b.py", "c.py"]):
        shown = await surfacing.surface(
            sessions, context(focus), now=NOW + timedelta(minutes=index)
        )
        if focus == "a.py":
            await surfacing.dismiss(sessions, shown.id)
        elif focus == "b.py":
            await surfacing.mark_useful(sessions, shown.id)

    report = await surfacing.stats(sessions)

    assert report.surfaced == 3
    assert report.dismissed == 1
    assert report.acted_on == 1
    assert report.unrated == 1
    assert report.dismissal_rate == pytest.approx(1 / 3)
    assert report.rated_dismissal_rate == pytest.approx(1 / 2)
    # Per-focus thresholds, which is what "threshold after adaptation" means.
    thresholds = {row.focus: row.threshold for row in report.per_focus}
    assert thresholds["a.py"] > thresholds["c.py"] > thresholds["b.py"]


async def test_suppression_is_counted_apart_from_refusal(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Otherwise "the windows are working" and "the bar is high" are one number."""
    await surfacing.surface(sessions, context("a.py"), now=NOW)
    await surfacing.surface(sessions, context("a.py"), now=NOW + timedelta(minutes=1))
    await surfacing.surface(sessions, context("b.py", score=WEAK), now=NOW)

    report = await surfacing.stats(sessions)

    assert report.suppressed == 1
    assert report.by_reason[SurfaceReason.BELOW_THRESHOLD] == 1
    assert report.by_reason[SurfaceReason.ALREADY_SURFACED] == 1


# --------------------------------------------------------------------------
# The two clicks
# --------------------------------------------------------------------------


async def test_the_api_takes_both_verdicts_and_refuses_a_second_one(
    sessions: async_sessionmaker[AsyncSession], client: AsyncClient
) -> None:
    """The panel and the web UI both post to these, so they are asserted here
    once rather than twice in two languages."""
    shown = await surfacing.surface(sessions, context("a.py"), now=NOW)
    listed = await client.get("/surfacing", params={"focus": "a.py"})
    assert listed.status_code == 200
    assert [row["id"] for row in listed.json()] == [str(shown.id)]
    # The sentence travels with the row, so three clients cannot phrase one
    # rule three ways.
    assert listed.json()[0]["explanation"]

    assert (await client.post(f"/surfacing/{shown.id}/dismiss")).status_code == 204
    conflict = await client.post(f"/surfacing/{shown.id}/useful")
    assert conflict.status_code == 409

    # Rated, so the panel stops offering buttons for it.
    assert (await client.get("/surfacing", params={"focus": "a.py"})).json() == []
    # But it is still in the log, and still in the dismissal rate.
    history = await client.get(
        "/surfacing", params={"focus": "a.py", "include_refused": True}
    )
    assert history.json()[0]["verdict"] == "dismissed"

    missing = await client.post(f"/surfacing/{new_id()}/dismiss")
    assert missing.status_code == 404
