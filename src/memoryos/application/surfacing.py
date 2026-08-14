"""Volunteering context, and keeping a record of every time it did not.

M6.3 is the milestone where this system is finally allowed to interrupt, and it
is mostly about restraint. `domain/surfacing` holds the arithmetic; this holds
the four things that need a database:

* the per-focus threshold, derived from that focus's own feedback,
* the recent surfacings a new one is checked against,
* the row every decision writes, refusals included,
* and the counting that says whether any of it is working.

**Default to silence.** A false positive costs trust, and trust does not come
back; a false negative costs nothing, because `memoryos context` and the panel
are both still there. Every path here that cannot decide returns a refusal with
a reason rather than a guess with a shrug.

### It never assembles for a high-volume trigger

M6.1 refused to precompute context for `FILE_FOCUSED` because it fires on every
file somebody glances at, and assembling for each burns compute continuously to
produce output nobody reads. Surfacing does not overturn that: on a
`FILE_FOCUSED` it reads the cache and stops if there is nothing there. A context
somebody's panel already caused to be built is free to judge; one nobody has
asked for stays unbuilt.

So the policy is one line in `SurfaceOnTrigger.assembles`, and the reason it is
a set rather than an `if` is that the next person to add a trigger kind has to
decide which side of the line it falls on.

### The pre-meeting case

`MEETING_UPCOMING` is the trigger this milestone is really for: something with a
deadline, where context five minutes early is worth having and five minutes late
is worth nothing. The handler works — a manually emitted event drives it end to
end, and the tests do exactly that.

**There is no calendar connector, and nothing here pretends otherwise.** No
calendar has been connected to this system, no meeting event has ever arrived
from one, and every `meeting_upcoming` in this corpus was posted by hand. The
trigger path is built and waiting; the data source is absent, and that sentence
is the honest version of a demo.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.context_engine import (
    FOCUS_SPECIFIC,
    AssembledContext,
    ContextItem,
    ContextRequest,
    cache_key_for,
    corpus_fingerprint,
    read_cached,
)
from memoryos.domain.events import Event, EventKind
from memoryos.domain.fusion import contribution
from memoryos.domain.ids import new_id
from memoryos.domain.surfacing import (
    DISMISSAL_WINDOW,
    REPEAT_WINDOW,
    PriorSurfacing,
    SurfaceDecision,
    SurfaceReason,
    TopItem,
    decide,
    names_the_focus,
    threshold_for,
)

logger = structlog.get_logger(__name__)

# How far back the suppression query looks.
#
# The longer of the two windows, because a row older than that cannot suppress
# anything — checking against it would be work whose answer is known.
LOOKBACK = max(REPEAT_WINDOW, DISMISSAL_WINDOW)


class AssemblesContext(Protocol):
    """What the trigger handler needs from the context engine, and no more.

    A Protocol rather than `AssembleContext` itself, for the reason the event bus
    declares one: this module is about a decision, and typing it against the
    concrete engine would make every test of the gate construct an embedder and a
    cross-encoder to answer a question about arithmetic.
    """

    async def __call__(self, request: ContextRequest) -> AssembledContext: ...


class AlreadyRated(ValueError):
    """Feedback on a row that already has some. Refused rather than overwritten.

    A second click is either a double-submit or somebody changing their mind, and
    the two want different things — but silently replacing the first verdict
    makes the dismissal rate depend on click order, so both are refused and the
    caller is told which verdict is already there.
    """

    def __init__(self, row_id: UUID, verdict: str) -> None:
        super().__init__(f"surfacing {row_id} was already marked {verdict}")
        self.row_id = row_id
        self.verdict = verdict


@dataclass(frozen=True, slots=True)
class Surfaced:
    """A decision, and the row that recorded it."""

    id: UUID
    focus: str
    decision: SurfaceDecision


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


async def should_surface(
    sessions: async_sessionmaker[AsyncSession],
    context: AssembledContext,
    trigger: Event | None = None,
    *,
    now: datetime | None = None,
) -> SurfaceDecision:
    """Whether this context is worth interrupting for, and why either way.

    Reads two things and decides with neither: this focus's feedback, which sets
    the bar, and what has been surfaced for it lately, which is what stops the
    same thing being said twice. The judgement itself is `domain/surfacing.decide`
    so it can be checked against a worked example rather than against a corpus.

    `trigger` is accepted and unused by the arithmetic, which is deliberate. What
    caused a decision belongs in the row — the whole question of whether
    proactive context works is a question about which triggers produce anything
    worth reading — but letting the kind change the *threshold* would mean four
    gates with one name, and the first one somebody tuned would quietly become
    the only one that mattered.
    """
    moment = now or datetime.now(UTC)
    threshold = await threshold_for_focus(sessions, context.focus)
    recent = await recent_surfacings(sessions, context.focus, now=moment)

    return decide(
        top=_top_item(context),
        keys=[item.key for item in context.items],
        threshold=threshold,
        recent=recent,
        now=moment,
    )


def focus_specific_score(item: ContextItem) -> float:
    """The part of an item's fused score that is a claim about the focus.

    **The threshold is meaningless without this, and for one milestone it was.**
    M6.3 shipped comparing the whole fused score against a bar chosen so that no
    single route could clear it — a structural guarantee that two independent
    rankings had agreed. One of the rankings was global recency, which says the
    same thing about every focus at a given moment, so "retrieval found it and it
    is recent" read as agreement. On a repository somebody is working in daily
    that is nearly every file, and it is why fourteen consecutive focuses scored
    within ±10% of the bar.

    So the score the gate compares is recomputed from the routes in
    `FOCUS_SPECIFIC`, using the same term the fusion itself sums. Recency still
    ranks items, still puts things in the context, and still cannot vote on
    whether to interrupt anybody.
    """
    return sum(
        contribution(rank)
        for source, rank in item.sources.items()
        if source in FOCUS_SPECIFIC
    )


def _top_item(context: AssembledContext) -> TopItem | None:
    """The best item that is not the file already open.

    **Two of Step 1's three conditions collapse into this one function**, and
    they should: "the top item's score exceeds a high threshold" and "the context
    contains something the reader plausibly does not already have open" are the
    same requirement applied to the same item. Scoring the top item and then
    separately asking whether *anything* is novel would let a context be
    surfaced on the strength of an item nobody would be shown.

    Ranked by focus-specific score rather than by position or by the whole fused
    score. Position is MMR's output — it has already traded relevance for
    coverage — and the fused score includes a route that would have said the
    same about any focus. See `focus_specific_score`.
    """
    candidates = [
        item
        for item in context.items
        if not names_the_focus(item.external_key, context.focus)
    ]
    if not candidates:
        return None
    best = max(candidates, key=focus_specific_score)
    return TopItem(
        key=best.key,
        title=best.title,
        score=focus_specific_score(best),
        # Counted the same way, so "found by 2 of 4 sources" in the CLI and the
        # score in the log can never disagree about which routes were involved.
        routes=len([source for source in best.sources if source in FOCUS_SPECIFIC]),
    )


async def threshold_for_focus(
    sessions: async_sessionmaker[AsyncSession], focus: str
) -> float:
    """This focus's bar, after its own feedback.

    **Per focus, never global.** One noisy file must not be able to silence the
    whole system, and a global counter has that failure by construction: a
    fortnight of dismissals on a generated file would raise the bar everywhere,
    including on the focuses that were working.

    Derived from the log on every call rather than stored in a column. The
    counts are two integers behind an index, and a stored threshold is a second
    copy of a fact that can drift from the rows it was computed from — which for
    an adaptive number means drifting silently and in the direction nobody
    checks.
    """
    async with sessions() as session:
        row = (
            await session.execute(
                select(
                    func.count(models.SurfacingLog.dismissed_at),
                    func.count(models.SurfacingLog.acted_on_at),
                ).where(models.SurfacingLog.focus == focus)
            )
        ).one()
    return threshold_for(dismissed=int(row[0]), acted_on=int(row[1]))


async def recent_surfacings(
    sessions: async_sessionmaker[AsyncSession],
    focus: str,
    *,
    now: datetime | None = None,
) -> list[PriorSurfacing]:
    """What has actually been shown for this focus lately.

    Surfaced rows only, which the partial index is built for. A refusal
    suppresses nothing — it was never seen — and including refusals here would
    let one below-threshold assembly stop the same context being surfaced when it
    later earned it.
    """
    moment = now or datetime.now(UTC)
    async with sessions() as session:
        rows = list(
            (
                await session.execute(
                    select(models.SurfacingLog)
                    .where(
                        models.SurfacingLog.focus == focus,
                        models.SurfacingLog.surfaced_at.is_not(None),
                        models.SurfacingLog.surfaced_at > moment - LOOKBACK,
                    )
                    .order_by(models.SurfacingLog.surfaced_at.desc())
                )
            ).scalars()
        )
    return [_to_prior(row) for row in rows]


def _to_prior(row: models.SurfacingLog) -> PriorSurfacing:
    assert row.surfaced_at is not None  # the query selects on it
    return PriorSurfacing(
        context_hash=row.context_hash,
        keys=tuple(str(key) for key in row.item_keys),
        surfaced_at=row.surfaced_at,
        dismissed_at=row.dismissed_at,
        acted_on_at=row.acted_on_at,
    )


# --------------------------------------------------------------------------
# Deciding and recording
# --------------------------------------------------------------------------


async def surface(
    sessions: async_sessionmaker[AsyncSession],
    context: AssembledContext,
    trigger: Event | None = None,
    *,
    now: datetime | None = None,
) -> Surfaced:
    """Decide, write the row, and return both. The whole path, in one call.

    **Refusals are written too**, and that is the design rather than an
    accident of implementation. "Why didn't it show me anything?" has to be
    answerable, and it cannot be from a table of things that were shown: silence
    from a gate that refused looks exactly like silence from a handler that never
    ran. A refusal costs one row and buys the only diagnostic this feature has.
    """
    moment = now or datetime.now(UTC)
    decision = await should_surface(sessions, context, trigger, now=moment)
    row_id = new_id()

    row = models.SurfacingLog(
        id=row_id,
        focus=context.focus,
        context_hash=decision.context_hash,
        item_keys=[item.key for item in context.items],
        top_key=None if decision.top is None else decision.top.key,
        top_title=None if decision.top is None else decision.top.title,
        score=decision.score,
        threshold=decision.threshold,
        reason=decision.reason.value,
        trigger_kind=None if trigger is None else trigger.kind.value,
        trigger_id=None if trigger is None else trigger.id,
        decided_at=moment,
        surfaced_at=moment if decision.surface else None,
    )
    async with sessions.begin() as session:
        session.add(row)

    logger.info(
        "surfacing.decided",
        focus=context.focus,
        surfaced=decision.surface,
        reason=decision.reason.value,
        score=round(decision.score, 5),
        threshold=round(decision.threshold, 5),
        top=None if decision.top is None else decision.top.key,
        routes=None if decision.top is None else decision.top.routes,
        trigger_kind=None if trigger is None else trigger.kind.value,
    )
    return Surfaced(id=row_id, focus=context.focus, decision=decision)


async def dismiss(sessions: async_sessionmaker[AsyncSession], row_id: UUID) -> bool:
    """Mark one surfaced context as not worth having been shown."""
    return await _rate(sessions, row_id, dismissed=True)


async def mark_useful(sessions: async_sessionmaker[AsyncSession], row_id: UUID) -> bool:
    """Mark one surfaced context as worth having been shown."""
    return await _rate(sessions, row_id, dismissed=False)


async def _rate(
    sessions: async_sessionmaker[AsyncSession], row_id: UUID, *, dismissed: bool
) -> bool:
    """Record a verdict. False when there is no such surfaced row.

    Refuses a row that already carries a verdict rather than replacing it. The
    dismissal rate is the number this milestone is judged on, and a rate that
    depends on which click landed last is not a measurement.
    """
    now = datetime.now(UTC)
    async with sessions.begin() as session:
        row = await session.get(models.SurfacingLog, row_id)
        if row is None or row.surfaced_at is None:
            return False
        if row.dismissed_at is not None:
            raise AlreadyRated(row_id, "dismissed")
        if row.acted_on_at is not None:
            raise AlreadyRated(row_id, "useful")
        if dismissed:
            row.dismissed_at = now
        else:
            row.acted_on_at = now
    logger.info(
        "surfacing.rated",
        surfacing_id=str(row_id),
        verdict="dismissed" if dismissed else "useful",
    )
    return True


# --------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoggedDecision:
    """One row of the log, for the CLI, the API and the panel."""

    id: UUID
    focus: str
    reason: SurfaceReason
    score: float
    threshold: float
    top_key: str | None
    top_title: str | None
    item_count: int
    trigger_kind: str | None
    decided_at: datetime
    surfaced_at: datetime | None
    dismissed_at: datetime | None
    acted_on_at: datetime | None

    @property
    def rated(self) -> str | None:
        if self.dismissed_at is not None:
            return "dismissed"
        if self.acted_on_at is not None:
            return "useful"
        return None


def _to_logged(row: models.SurfacingLog) -> LoggedDecision:
    return LoggedDecision(
        id=row.id,
        focus=row.focus,
        reason=SurfaceReason(row.reason),
        score=row.score,
        threshold=row.threshold,
        top_key=row.top_key,
        top_title=row.top_title,
        item_count=len(row.item_keys),
        trigger_kind=row.trigger_kind,
        decided_at=row.decided_at,
        surfaced_at=row.surfaced_at,
        dismissed_at=row.dismissed_at,
        acted_on_at=row.acted_on_at,
    )


async def recent(
    sessions: async_sessionmaker[AsyncSession],
    *,
    focus: str | None = None,
    surfaced_only: bool = False,
    unrated_only: bool = False,
    within: timedelta | None = None,
    limit: int = 20,
) -> list[LoggedDecision]:
    """The tail of the log, newest first.

    `unrated_only` is what a panel asks for: things that were shown and have not
    yet been judged. Everything else is history, and a panel that re-offered
    "dismiss" on something already dismissed would be asking a question it
    already has the answer to.

    `within` bounds it by age, which the panel also wants and for a different
    reason: "was this useful?" about something from last week is a question whose
    answer would be noise. The row stays either way and is still counted in the
    dismissal rate — expiring it from the table would quietly improve the one
    number this milestone reports.
    """
    stmt = (
        select(models.SurfacingLog)
        .order_by(models.SurfacingLog.decided_at.desc())
        .limit(limit)
    )
    if focus is not None:
        stmt = stmt.where(models.SurfacingLog.focus == focus)
    if within is not None:
        stmt = stmt.where(models.SurfacingLog.decided_at > datetime.now(UTC) - within)
    if surfaced_only or unrated_only:
        stmt = stmt.where(models.SurfacingLog.surfaced_at.is_not(None))
    if unrated_only:
        stmt = stmt.where(
            models.SurfacingLog.dismissed_at.is_(None),
            models.SurfacingLog.acted_on_at.is_(None),
        )
    async with sessions() as session:
        rows = list((await session.execute(stmt)).scalars())
    return [_to_logged(row) for row in rows]


@dataclass(frozen=True, slots=True)
class FocusStats:
    """One focus, and what it has learned about itself."""

    focus: str
    decisions: int
    surfaced: int
    dismissed: int
    acted_on: int

    @property
    def threshold(self) -> float:
        """The bar this focus has adapted to. Recomputed, never stored."""
        return threshold_for(dismissed=self.dismissed, acted_on=self.acted_on)


@dataclass(frozen=True, slots=True)
class SurfacingStats:
    """The numbers that say whether volunteering context works.

    `suppressed` counts refusals that were *specifically* suppression — the same
    context again, or one somebody dismissed — rather than every refusal.
    Everything below the threshold was never a candidate for suppression, and
    folding the two together would make the number that says "the suppression is
    doing work" indistinguishable from the number that says "the bar is high".
    """

    decisions: int
    surfaced: int
    dismissed: int
    acted_on: int
    suppressed: int
    by_reason: dict[SurfaceReason, int] = field(default_factory=dict)
    per_focus: list[FocusStats] = field(default_factory=list)

    @property
    def unrated(self) -> int:
        return self.surfaced - self.dismissed - self.acted_on

    @property
    def dismissal_rate(self) -> float | None:
        """Dismissed over everything that was shown. **The number that matters.**

        The denominator is every surfaced item, not only the ones that got a
        verdict. That is the harsher of the two available readings and it is the
        right one: an interruption nobody bothered to rate was still an
        interruption, and counting only the rated ones would let a feature that
        is ignored score the same as one that is valued.

        `rated_dismissal_rate` is the other reading, reported beside it rather
        than instead of it — with a small `unrated` count the two are close, and
        with a large one the gap is itself the finding.

        None when nothing has been surfaced. A rate over zero interruptions
        reported as 0% would read as "never wrong" when it means "never spoke".
        """
        if self.surfaced == 0:
            return None
        return self.dismissed / self.surfaced

    @property
    def rated_dismissal_rate(self) -> float | None:
        rated = self.dismissed + self.acted_on
        if rated == 0:
            return None
        return self.dismissed / rated


async def stats(sessions: async_sessionmaker[AsyncSession]) -> SurfacingStats:
    """Counts over the whole log, computed in the database.

    Three queries rather than one: the totals, the reason breakdown, and the
    per-focus rows. They group differently, and a single statement doing all
    three would be a set of `FILTER` clauses nobody could read for the sake of
    two round trips against a table with an index on every column it touches.
    """
    suppressing = (SurfaceReason.DISMISSED.value, SurfaceReason.ALREADY_SURFACED.value)
    async with sessions() as session:
        totals = (
            await session.execute(
                select(
                    func.count(),
                    func.count(models.SurfacingLog.surfaced_at),
                    func.count(models.SurfacingLog.dismissed_at),
                    func.count(models.SurfacingLog.acted_on_at),
                    func.count().filter(models.SurfacingLog.reason.in_(suppressing)),
                ).select_from(models.SurfacingLog)
            )
        ).one()

        reasons = list(
            await session.execute(
                select(models.SurfacingLog.reason, func.count())
                .group_by(models.SurfacingLog.reason)
                .order_by(func.count().desc())
            )
        )

        focuses = list(
            await session.execute(
                select(
                    models.SurfacingLog.focus,
                    func.count(),
                    func.count(models.SurfacingLog.surfaced_at),
                    func.count(models.SurfacingLog.dismissed_at),
                    func.count(models.SurfacingLog.acted_on_at),
                )
                .group_by(models.SurfacingLog.focus)
                # Most-decided first, which is where the adaptation has had the
                # most to work with and where a wrong threshold does most harm.
                .order_by(func.count().desc(), models.SurfacingLog.focus)
            )
        )

    return SurfacingStats(
        decisions=int(totals[0]),
        surfaced=int(totals[1]),
        dismissed=int(totals[2]),
        acted_on=int(totals[3]),
        suppressed=int(totals[4]),
        by_reason={SurfaceReason(reason): int(count) for reason, count in reasons},
        per_focus=[
            FocusStats(
                focus=str(focus),
                decisions=int(decisions),
                surfaced=int(surfaced),
                dismissed=int(dismissed),
                acted_on=int(acted_on),
            )
            for focus, decisions, surfaced, dismissed, acted_on in focuses
        ],
    )


# --------------------------------------------------------------------------
# The trigger path
# --------------------------------------------------------------------------


class SurfaceOnTrigger:
    """The handler that lets an event produce an interruption.

    **Subscribed to three kinds and assembles for two**, which is M6.1's
    precompute policy unchanged rather than a new one. A `MEETING_UPCOMING` is
    scheduled and an `EDITOR_OPENED` predicts a session; both are worth the
    second of compute. A `FILE_FOCUSED` fires on every file glanced at, so this
    reads the cache and gives up when it is empty — a context somebody's panel
    already caused to be built is free to judge, and one nobody asked for stays
    unbuilt.

    That has a consequence worth stating plainly: **on this deployment, surfacing
    for a focused file only happens where the editor panel is running**, because
    the panel's `GET /context` is what fills the cache. A watcher on its own
    produces events that reach this handler and find nothing to judge. That is
    the correct behaviour under M6.1's policy and it is also why the surfacing
    numbers in this milestone are as small as they are.

    The assembler is built lazily, for the same reason `Container._LazyAssembler`
    exists: constructing it loads an embedder and a cross-encoder, and the API
    process subscribes handlers on every request that dispatches one.
    """

    name = "surfacing"
    kinds = frozenset(
        {EventKind.EDITOR_OPENED, EventKind.FILE_FOCUSED, EventKind.MEETING_UPCOMING}
    )
    # The kinds whose triggers justify building a context that does not exist.
    # A set rather than a condition, so the next kind added has to be placed on
    # one side of the line rather than inheriting whichever branch it fell into.
    assembles = frozenset({EventKind.EDITOR_OPENED, EventKind.MEETING_UPCOMING})

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        make_assembler: Callable[[], AssemblesContext],
        *,
        focus_of: Callable[[Event], str],
    ) -> None:
        self._sessions = sessions
        self._make_assembler = make_assembler
        self._focus_of = focus_of

    async def handle(self, event: Event) -> None:
        focus = self._focus_of(event)
        if not focus:
            # A payload with no focus. Logged rather than raised: it is a
            # client's mistake, and dead-lettering the job would make one
            # badly-formed plugin look like a broken queue.
            logger.info(
                "surfacing.no_focus", event_id=str(event.id), kind=event.kind.value
            )
            return

        context = await self._context_for(focus, event)
        if context is None:
            logger.info(
                "surfacing.nothing_cached",
                event_id=str(event.id),
                kind=event.kind.value,
                focus=focus,
                reason="this kind does not justify assembling one",
            )
            return

        await surface(self._sessions, context, event)

    async def _context_for(self, focus: str, event: Event) -> AssembledContext | None:
        """The context to judge: assembled for some kinds, read for the rest."""
        if event.kind in self.assembles:
            return await self._make_assembler()(
                ContextRequest(focus=focus, trigger=event)
            )
        request = ContextRequest(focus=focus)
        key = cache_key_for(request, await corpus_fingerprint(self._sessions))
        return await read_cached(self._sessions, key)


def build_handler(
    sessions: async_sessionmaker[AsyncSession],
    make_assembler: Callable[[], AssemblesContext],
    focus_of: Callable[[Event], str],
) -> SurfaceOnTrigger:
    """Constructed here so the container does not have to know the argument order."""
    return SurfaceOnTrigger(sessions, make_assembler, focus_of=focus_of)
