"""Deriving, asserting, superseding and dismissing facets.

**Rules-based, from data this system already holds. No language model.** The
reason is the one M5.3 gives and M7.2 gives again: a model asked to describe
somebody produces fluent, plausible, unfalsifiable sentences, and nothing in the
output distinguishes one drawn from five decisions from one drawn from the
question. Every statement here is a template with counts in it, and every count
is a row somebody can open.

### What each dimension is derived from, and what happens when it cannot be

| Dimension | Source | Bar |
| --- | --- | --- |
| `decision_patterns` | M5.3 patterns above their own threshold | 3 distinct decisions |
| `weaknesses` | assumption groups with a low hold rate | 3 evaluated members |
| `strengths` | assumption groups with a high hold rate | 3 evaluated members |
| `habits` | M4.0 activity periodicity | 3 occurrences of the cycle |
| `workflows` | entities recurring together in the graph | 3 shared memories |
| `goals` | **asserted only** | — |
| `learning_style` | **no deriver** | — |

A dimension with nothing above its bar produces an `Assessment` saying so, and
`model show` prints that instead of the section. This is the whole point of the
milestone on a corpus this size: the honest output is a page of stated gaps, and
a page of low-confidence sentences would be the same data pretending.

### Supersede, never update

A re-derivation that finds the same subject with a different statement inserts a
new row and points the old one at it. The history is the part that makes a model
of a person worth having — "you stopped doing this in March" needs March's
version to still exist — and an `UPDATE` would leave nothing to compare.

A re-derivation that finds the same subject with the *same* statement touches
nothing, so running `derive` twice is not two rows and not two timestamps.

### Dismissed facets are never re-derived

M5.3's rule, and the deriver checks it by subject rather than by id: a person who
rejected "you defer schema decisions" rejected the claim, not the row, and
re-proposing it under a new id on the next run would be a system arguing with its
user.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.ids import new_id
from memoryos.domain.user_model import (
    MIN_SUPPORT,
    UNDERIVABLE,
    Assessment,
    clears_bar,
    facet_confidence,
)
from memoryos.domain.values import (
    Dimension,
    FacetEvidenceKind,
    FacetOrigin,
    FacetRelation,
)

logger = structlog.get_logger(__name__)

# Hold rate at or below which a recurring assumption is a weakness, and at or
# above which it is a strength. The gap between them is deliberate: an assumption
# that holds two times in three is neither, and forcing every group into one of
# the two labels is how a model acquires claims it cannot support.
WEAK_HOLD_RATE = 0.5
STRONG_HOLD_RATE = 0.8


class UnknownFacet(LookupError):
    """No facet with that id."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """A facet a deriver is proposing, before the bar is applied."""

    dimension: Dimension
    detector: str
    subject_key: str
    statement: str
    supporting: int
    contradicting: int = 0
    evidence: tuple[tuple[FacetEvidenceKind, UUID, FacetRelation], ...] = ()
    first_observed: datetime | None = None
    last_observed: datetime | None = None

    @property
    def confidence(self) -> float:
        return facet_confidence(self.supporting, self.contradicting)

    @property
    def clears(self) -> bool:
        return clears_bar(self.supporting, self.contradicting)


@dataclass(frozen=True, slots=True)
class DeriveReport:
    """What one derivation did, and what it refused to do.

    `considered` and `below_bar` are reported rather than logged, for the reason
    `patterns discover` reports them: a run that emitted nothing has to be
    distinguishable from a run that found nothing to consider, and on this corpus
    almost every run is one of those two.
    """

    written: int = 0
    superseded: int = 0
    unchanged: int = 0
    considered: int = 0
    below_bar: int = 0
    skipped_dismissed: int = 0
    assessments: tuple[Assessment, ...] = ()
    rejected: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class FacetRow:
    """A facet as read back, with its evidence resolved."""

    id: UUID
    dimension: str
    statement: str
    confidence: float | None
    support_count: int
    contradiction_count: int
    origin: str
    detector: str | None
    subject_key: str | None
    first_observed: datetime | None
    last_observed: datetime | None
    superseded_by: UUID | None
    dismissed_at: datetime | None
    dismissed_reason: str | None
    created_at: datetime
    evidence: tuple[tuple[str, UUID, str], ...] = ()

    @property
    def live(self) -> bool:
        return self.superseded_by is None and self.dismissed_at is None


@dataclass(frozen=True, slots=True)
class ModelView:
    """Every dimension, with its facets or the reason it has none."""

    facets: dict[str, list[FacetRow]] = field(default_factory=dict)
    assessments: dict[str, Assessment] = field(default_factory=dict)
    dismissed: list[FacetRow] = field(default_factory=list)


# --------------------------------------------------------------------------
# Derivers
# --------------------------------------------------------------------------


async def _from_patterns(session: AsyncSession) -> list[Candidate]:
    """`decision_patterns`, one per M5.3 pattern that cleared its own bar.

    **The thinnest deriver here and the most defensible**, because the work was
    already done: a pattern has already been required to rest on three distinct
    decisions, already carries its supporting and contradicting counts, and has
    already been offered for dismissal. This restates it as a facet and carries
    the evidence across.

    Dismissed patterns are excluded at the source. A pattern somebody rejected
    must not reappear as a facet under a different name, which is the most
    obvious way this table could have been used to launder a rejected claim.
    """
    rows = list(
        await session.execute(
            select(models.Pattern).where(models.Pattern.dismissed_at.is_(None))
        )
    )
    candidates: list[Candidate] = []
    for (pattern,) in rows:
        evidence = list(
            (
                await session.execute(
                    select(models.PatternEvidence).where(
                        models.PatternEvidence.pattern_id == pattern.id
                    )
                )
            ).scalars()
        )
        candidates.append(
            Candidate(
                dimension=Dimension.DECISION_PATTERNS,
                detector="pattern",
                subject_key=f"{pattern.detector}:{pattern.subject_key}",
                statement=pattern.statement,
                supporting=pattern.support_count,
                contradicting=pattern.contradiction_count,
                evidence=(
                    (FacetEvidenceKind.PATTERN, pattern.id, FacetRelation.SUPPORTS),
                    *(
                        (
                            FacetEvidenceKind.DECISION,
                            row.decision_id,
                            FacetRelation.SUPPORTS
                            if row.relation == FacetRelation.SUPPORTS.value
                            else FacetRelation.CONTRADICTS,
                        )
                        for row in evidence
                        if row.decision_id is not None
                    ),
                ),
                first_observed=pattern.first_observed,
                last_observed=pattern.last_observed,
            )
        )
    return candidates


async def _from_assumption_groups(session: AsyncSession) -> list[Candidate]:
    """`strengths` and `weaknesses`, from how often a recurring belief held.

    A group is somebody's judgement that several assumptions are the same belief,
    which is exactly the unit this needs — and it is user-authored, so nothing
    here is inferring the grouping as well as the verdict.

    **Counted in distinct decisions, not in assumptions.** Four assumptions from
    two decisions is two observations of the belief, and counting rows would let
    one decision that stated the same thing four times clear a bar meant for
    three independent occasions.

    Unevaluated members are excluded from both counts rather than treated as
    holding. A belief nobody has checked is not evidence in either direction, and
    including it would make every group look strong the moment it was created.
    """
    rows = list(
        await session.execute(
            select(
                models.AssumptionGroup.id,
                models.AssumptionGroup.label,
                models.DecisionAssumption.decision_id,
                models.DecisionAssumption.held,
                models.DecisionAssumption.evaluated_at,
            ).join(
                models.DecisionAssumption,
                models.DecisionAssumption.group_id == models.AssumptionGroup.id,
            )
        )
    )
    grouped: dict[UUID, list[tuple[str, UUID, str | None, datetime | None]]] = {}
    for group_id, label, decision_id, held, evaluated_at in rows:
        grouped.setdefault(group_id, []).append((label, decision_id, held, evaluated_at))

    candidates: list[Candidate] = []
    for group_id, members in grouped.items():
        label = members[0][0]
        held_decisions = {
            decision_id for _, decision_id, held, _ in members if held == "held"
        }
        failed_decisions = {
            decision_id
            for _, decision_id, held, _ in members
            if held in ("failed", "partially")
        }
        evaluated = held_decisions | failed_decisions
        if not evaluated:
            continue
        rate = len(held_decisions) / len(evaluated)
        dates = [when for *_, when in members if when is not None]

        if rate <= WEAK_HOLD_RATE:
            dimension, detector = Dimension.WEAKNESSES, "assumption_group_weak"
            statement = (
                f"A belief you keep returning to has failed more often than it has "
                f"held: {label!r} held {len(held_decisions)} of "
                f"{len(evaluated)} times it was checked."
            )
            supporting, contradicting = len(failed_decisions), len(held_decisions)
        elif rate >= STRONG_HOLD_RATE:
            dimension, detector = Dimension.STRENGTHS, "assumption_group_strong"
            statement = (
                f"A belief you keep returning to has held every time it mattered: "
                f"{label!r} held {len(held_decisions)} of "
                f"{len(evaluated)} times it was checked."
            )
            supporting, contradicting = len(held_decisions), len(failed_decisions)
        else:
            # Between the two bars. Deliberately produces nothing: an assumption
            # that holds two times in three is neither a strength nor a weakness,
            # and forcing it into one is how a model acquires claims it cannot
            # support.
            continue

        candidates.append(
            Candidate(
                dimension=dimension,
                detector=detector,
                subject_key=str(group_id),
                statement=statement,
                supporting=supporting,
                contradicting=contradicting,
                evidence=tuple(
                    (
                        FacetEvidenceKind.DECISION,
                        decision_id,
                        FacetRelation.SUPPORTS
                        if (decision_id in held_decisions) == (dimension is Dimension.STRENGTHS)
                        else FacetRelation.CONTRADICTS,
                    )
                    for decision_id in evaluated
                ),
                first_observed=min(dates) if dates else None,
                last_observed=max(dates) if dates else None,
            )
        )
    return candidates


async def _from_activity(session: AsyncSession) -> list[Candidate]:
    """`habits`, from when work actually happens.

    Weekday-versus-weekend is the only periodicity a corpus can support without
    a year of it, and even that needs the cycle to have come round three times —
    which is the bar, expressed as three distinct calendar weeks.

    **What this can measure is bounded by where the dates come from.** A
    `filesystem` timestamp is when a file was last written, which a checkout, a
    bulk reformat or a clone resets for the whole tree at once. `TimeProvenance`
    exists to make that visible, and this deriver refuses to run on a corpus
    whose dates are entirely filesystem mtimes — a habit derived from those is a
    claim about `git clone`.
    """
    # Bound once and reused in both clauses. Two separate `func.date_trunc`
    # calls render two bind parameters, and Postgres will not match
    # `date_trunc($1, col)` in the projection against `date_trunc($3, col)` in
    # the GROUP BY — it reports the column as ungrouped, which is a confusing way
    # to be told the expressions are not literally identical.
    #
    # Three-argument form, as `temporal.activity_by_period` uses: the truncation
    # happens in UTC rather than in whatever the session's TimeZone is, so a
    # weekday boundary does not move with the client's locale.
    week = func.date_trunc("week", models.Memory.occurred_at, "UTC")
    weekday = func.extract("isodow", models.Memory.occurred_at)
    declared = list(
        await session.execute(
            select(week, weekday, func.count())
            .where(
                models.Memory.is_current.is_(True),
                models.Memory.occurred_at.is_not(None),
                # Anything but a bare file mtime. See the docstring.
                models.Memory.occurred_at_source != "filesystem",
            )
            .group_by(week, weekday)
        )
    )
    if not declared:
        return []

    weeks = {week for week, _, _ in declared}
    weekend = sum(count for _, dow, count in declared if int(dow) >= 6)
    weekday = sum(count for _, dow, count in declared if int(dow) < 6)
    total = weekend + weekday
    if not total:
        return []

    share = weekend / total
    if share >= 0.4:
        statement = (
            f"You work through weekends: {weekend} of {total} dated items fall on "
            f"a Saturday or Sunday, across {len(weeks)} weeks."
        )
        supporting, contradicting = weekend, weekday
    elif share <= 0.1:
        statement = (
            f"Your work keeps to weekdays: {weekend} of {total} dated items fall "
            f"on a weekend, across {len(weeks)} weeks."
        )
        supporting, contradicting = weekday, weekend
    else:
        return []

    # Support is the number of *weeks* the cycle was observed over, not the
    # number of items. A thousand files touched in one week is one observation
    # of a weekly habit.
    return [
        Candidate(
            dimension=Dimension.HABITS,
            detector="weekly_rhythm",
            subject_key="weekend_share",
            statement=statement,
            supporting=len(weeks),
            contradicting=0 if supporting > contradicting else len(weeks),
        )
    ]


async def _from_entity_cooccurrence(session: AsyncSession) -> list[Candidate]:
    """`workflows`, from entities that keep turning up together.

    The weakest deriver in the file and the one most likely to produce a
    horoscope, so it is bounded twice.

    **Coverage first.** If entity extraction has reached only a slice of the
    corpus, a pair co-occurring in three of the extracted memories is a fact
    about the slice rather than about a workflow. Below `_MIN_COVERAGE` this
    returns nothing and the assessment says the coverage rather than the count.

    **And it is still only a co-occurrence.** Two libraries named in the same
    files means they are used together, which for `alembic` and `sqlalchemy` is a
    property of the libraries and not of the person. Nothing here can tell those
    apart, which is why the statement says what was counted rather than what it
    means.
    """
    total = (
        await session.execute(
            select(func.count()).select_from(models.Memory).where(
                models.Memory.is_current.is_(True)
            )
        )
    ).scalar_one()
    covered = (
        await session.execute(
            select(func.count(func.distinct(models.EntityMention.memory_id)))
        )
    ).scalar_one()
    if not total or covered / total < _MIN_COVERAGE:
        return []

    left = models.EntityMention.__table__.alias("left_mention")
    right = models.EntityMention.__table__.alias("right_mention")
    pairs = list(
        await session.execute(
            select(
                left.c.entity_id,
                right.c.entity_id,
                func.count(func.distinct(left.c.memory_id)),
            )
            .select_from(left.join(right, left.c.memory_id == right.c.memory_id))
            .where(left.c.entity_id < right.c.entity_id)
            .group_by(left.c.entity_id, right.c.entity_id)
            .having(func.count(func.distinct(left.c.memory_id)) >= MIN_SUPPORT)
        )
    )
    if not pairs:
        return []

    names: dict[UUID, str] = {
        entity_id: name
        for entity_id, name in (
            await session.execute(
                select(models.Entity.id, models.Entity.canonical_name)
            )
        ).all()
    }
    return [
        Candidate(
            dimension=Dimension.WORKFLOWS,
            detector="entity_cooccurrence",
            subject_key=f"{first}:{second}",
            statement=(
                f"{names.get(first, str(first))} and {names.get(second, str(second))} appear "
                f"together in {count} memories."
            ),
            supporting=count,
        )
        for first, second, count in pairs
    ]


# Share of the corpus entity extraction must have reached before a co-occurrence
# says anything about the corpus. Measured against this project's own state: at
# 4.7% coverage the top pairs were `alembic + sqlalchemy` and `sqlalchemy +
# postgres`, which are facts about a Python project rather than about a person.
_MIN_COVERAGE = 0.5


_DERIVERS = (
    ("decision_patterns", _from_patterns),
    ("assumption_groups", _from_assumption_groups),
    ("activity", _from_activity),
    ("entity_cooccurrence", _from_entity_cooccurrence),
)


# --------------------------------------------------------------------------
# Deriving
# --------------------------------------------------------------------------


async def derive(session_factory: async_sessionmaker[AsyncSession]) -> DeriveReport:
    """Run every deriver, write what clears the bar, and report the rest."""
    written = superseded = unchanged = below_bar = skipped = 0
    considered: list[Candidate] = []
    rejected: list[tuple[str, int]] = []

    async with session_factory() as session, session.begin():
        for _, deriver in _DERIVERS:
            considered.extend(await deriver(session))

        dismissed_subjects = {
            (detector, subject)
            for detector, subject in (
                await session.execute(
                    select(
                        models.UserModelFacet.detector,
                        models.UserModelFacet.subject_key,
                    ).where(models.UserModelFacet.dismissed_at.is_not(None))
                )
            ).all()
        }

        for candidate in considered:
            if (candidate.detector, candidate.subject_key) in dismissed_subjects:
                skipped += 1
                continue
            if not candidate.clears:
                below_bar += 1
                rejected.append((candidate.statement, candidate.supporting))
                continue

            existing = (
                await session.execute(
                    select(models.UserModelFacet).where(
                        models.UserModelFacet.detector == candidate.detector,
                        models.UserModelFacet.subject_key == candidate.subject_key,
                        models.UserModelFacet.origin == FacetOrigin.DERIVED.value,
                        models.UserModelFacet.superseded_by.is_(None),
                        models.UserModelFacet.dismissed_at.is_(None),
                    )
                )
            ).scalar_one_or_none()

            if existing is not None and existing.statement == candidate.statement:
                # Same subject, same sentence. Running `derive` twice is not two
                # rows and not two timestamps.
                unchanged += 1
                continue

            facet = _insert(session, candidate)
            if existing is not None:
                # Supersede rather than update: how the model changed is part of
                # the model, and an UPDATE would leave nothing to compare.
                existing.superseded_by = facet.id
                superseded += 1
            written += 1

    assessments = await assess(session_factory)
    report = DeriveReport(
        written=written,
        superseded=superseded,
        unchanged=unchanged,
        considered=len(considered),
        below_bar=below_bar,
        skipped_dismissed=skipped,
        assessments=assessments,
        rejected=tuple(rejected),
    )
    logger.info(
        "user_model.derived",
        written=written,
        superseded=superseded,
        unchanged=unchanged,
        considered=len(considered),
        below_bar=below_bar,
        skipped_dismissed=skipped,
    )
    return report


def _insert(session: AsyncSession, candidate: Candidate) -> models.UserModelFacet:
    facet = models.UserModelFacet(
        id=new_id(),
        dimension=candidate.dimension.value,
        statement=candidate.statement,
        confidence=candidate.confidence,
        support_count=candidate.supporting,
        contradiction_count=candidate.contradicting,
        first_observed=candidate.first_observed,
        last_observed=candidate.last_observed,
        origin=FacetOrigin.DERIVED.value,
        detector=candidate.detector,
        subject_key=candidate.subject_key,
    )
    session.add(facet)
    seen: set[tuple[str, UUID, str]] = set()
    for kind, ref_id, relation in candidate.evidence:
        key = (kind.value, ref_id, relation.value)
        if key in seen:
            continue
        seen.add(key)
        session.add(
            models.FacetEvidence(
                id=new_id(),
                facet_id=facet.id,
                kind=kind.value,
                ref_id=ref_id,
                relation=relation.value,
            )
        )
    return facet


async def assess(
    session_factory: async_sessionmaker[AsyncSession],
) -> tuple[Assessment, ...]:
    """Every dimension, with its live facet count or the reason it has none.

    **Computed for all seven, including the two nothing derives.** A page that
    omitted them would read as though the model covered five things; a page that
    prints why two are absent tells the reader what would have to exist.
    """
    async with session_factory() as session:
        counts = dict(
            (dimension, count)
            for dimension, count in (
                await session.execute(
                    select(models.UserModelFacet.dimension, func.count())
                    .where(
                        models.UserModelFacet.superseded_by.is_(None),
                        models.UserModelFacet.dismissed_at.is_(None),
                    )
                    .group_by(models.UserModelFacet.dimension)
                )
            ).all()
        )
        candidates = await _best_support(session)
        structural = await structural_gaps(session)

    assessments: list[Assessment] = []
    for dimension in Dimension:
        found = counts.get(dimension.value, 0)
        if found:
            assessments.append(Assessment(dimension=dimension, facets=found))
            continue
        gap = UNDERIVABLE.get(dimension) or structural.get(
            dimension,
            f"nothing reached {MIN_SUPPORT} distinct observations",
        )
        assessments.append(
            Assessment(
                dimension=dimension,
                facets=0,
                gap=gap,
                best_support=candidates.get(dimension, 0),
            )
        )
    return tuple(assessments)


async def structural_gaps(session: AsyncSession) -> dict[Dimension, str]:
    """Why a deriver produced nothing, when the reason is not "too few".

    **"Nothing reached three" is the wrong answer when nothing was counted.**
    Two derivers here decline before they count anything — one because the dates
    are all file mtimes, the other because entity extraction has reached too
    little of the corpus — and reporting those as a support shortfall would send
    a reader off to record more decisions when what is missing is extraction
    coverage. Step 5 asks empty dimensions to say what would fill them; this is
    the part that makes that answer specific.
    """
    gaps: dict[Dimension, str] = {}

    total = (
        await session.execute(
            select(func.count()).select_from(models.Memory).where(
                models.Memory.is_current.is_(True)
            )
        )
    ).scalar_one()
    dated = (
        await session.execute(
            select(func.count())
            .select_from(models.Memory)
            .where(
                models.Memory.is_current.is_(True),
                models.Memory.occurred_at_source != "filesystem",
            )
        )
    ).scalar_one()
    if total and not dated:
        gaps[Dimension.HABITS] = (
            f"every one of {total} dated memories carries a filesystem mtime, which "
            "records when a file was last written rather than when work happened — "
            "a checkout or a reformat resets the whole tree at once. This needs a "
            "source that declares its own dates (commits, calendar, messages)"
        )

    covered = (
        await session.execute(
            select(func.count(func.distinct(models.EntityMention.memory_id)))
        )
    ).scalar_one()
    if total and covered / total < _MIN_COVERAGE:
        gaps[Dimension.WORKFLOWS] = (
            f"entity extraction has reached {covered} of {total} memories "
            f"({covered / total:.0%}); a co-occurrence over that slice is a fact "
            f"about the slice. This needs extraction over at least "
            f"{_MIN_COVERAGE:.0%} of the corpus"
        )
    return gaps


async def _best_support(session: AsyncSession) -> dict[Dimension, int]:
    """How close each empty dimension came, so the gap can say a number."""
    best: dict[Dimension, int] = {}
    for _, deriver in _DERIVERS:
        for candidate in await deriver(session):
            best[candidate.dimension] = max(
                best.get(candidate.dimension, 0), candidate.supporting
            )
    return best


# --------------------------------------------------------------------------
# Asserting, dismissing, reading
# --------------------------------------------------------------------------


async def assert_facet(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    dimension: Dimension,
    statement: str,
    supersedes: UUID | None = None,
) -> UUID:
    """State a facet directly. **The only way a goal enters the model.**

    No confidence, and that is not an omission: a goal somebody stated is not a
    claim with a probability attached, and writing 1.0 would sort every user
    statement above every derived facet in any ranking that reads the column.

    `supersedes` is how a stated facet changes. The old row stays and points at
    the new one, so "you used to be trying to ship by June" remains answerable.
    """
    text = statement.strip()
    if not text:
        raise ValueError("a facet needs a statement")

    async with session_factory() as session, session.begin():
        facet = models.UserModelFacet(
            id=new_id(),
            dimension=dimension.value,
            statement=text,
            confidence=None,
            support_count=0,
            contradiction_count=0,
            origin=FacetOrigin.ASSERTED.value,
            detector=None,
            subject_key=None,
        )
        session.add(facet)
        if supersedes is not None:
            previous = await session.get(models.UserModelFacet, supersedes)
            if previous is None:
                raise UnknownFacet(str(supersedes))
            previous.superseded_by = facet.id
    logger.info(
        "user_model.asserted", dimension=dimension.value, supersedes=str(supersedes or "")
    )
    return facet.id


async def dismiss(
    session_factory: async_sessionmaker[AsyncSession],
    facet_id: UUID,
    *,
    reason: str,
) -> None:
    """Reject a facet permanently. Derivation will not propose it again.

    Rejected **by subject** rather than by id, which is what `derive` reads: a
    person who rejected a claim rejected the claim, and re-proposing the same
    sentence under a new id on the next run would be a system arguing with its
    user.
    """
    text = reason.strip()
    if not text:
        raise ValueError("a dismissal needs a reason: it is the part that survives")

    async with session_factory() as session, session.begin():
        facet = await session.get(models.UserModelFacet, facet_id)
        if facet is None:
            raise UnknownFacet(str(facet_id))
        facet.dismissed_at = datetime.now(UTC)
        facet.dismissed_reason = text
    logger.info("user_model.dismissed", facet=str(facet_id))


async def view(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    dimension: Dimension | None = None,
) -> ModelView:
    """The live model, by dimension, with dismissed facets kept visible.

    Dismissed rows come back in their own list rather than being filtered away:
    a page that hid them would make a rejected claim look like one that was never
    made, and the rejection is the more interesting fact.
    """
    async with session_factory() as session:
        stmt = select(models.UserModelFacet).where(
            models.UserModelFacet.superseded_by.is_(None)
        )
        if dimension is not None:
            stmt = stmt.where(models.UserModelFacet.dimension == dimension.value)
        rows = list((await session.execute(stmt)).scalars())
        evidence = await _evidence_for(session, [row.id for row in rows])

    facets: dict[str, list[FacetRow]] = {}
    dismissed: list[FacetRow] = []
    for row in rows:
        built = _row(row, evidence.get(row.id, ()))
        if built.dismissed_at is not None:
            dismissed.append(built)
        else:
            facets.setdefault(built.dimension, []).append(built)
    for group in facets.values():
        group.sort(key=lambda item: (-(item.confidence or 0.0), item.statement))

    assessments = {item.dimension.value: item for item in await assess(session_factory)}
    if dimension is not None:
        assessments = {
            key: value for key, value in assessments.items() if key == dimension.value
        }
    return ModelView(facets=facets, assessments=assessments, dismissed=dismissed)


async def history(
    session_factory: async_sessionmaker[AsyncSession], facet_id: UUID
) -> list[FacetRow]:
    """The whole chain a facet belongs to, oldest first.

    **Walked in both directions from the id given**, because a caller holding the
    current row and a caller holding a superseded one are asking the same
    question and neither should have to know which end they are at.
    """
    async with session_factory() as session:
        start = await session.get(models.UserModelFacet, facet_id)
        if start is None:
            raise UnknownFacet(str(facet_id))

        chain = [start]
        # Backwards: whatever points at the row we currently hold first.
        while True:
            earlier = (
                await session.execute(
                    select(models.UserModelFacet).where(
                        models.UserModelFacet.superseded_by == chain[0].id
                    )
                )
            ).scalar_one_or_none()
            if earlier is None:
                break
            chain.insert(0, earlier)
        # Forwards.
        while chain[-1].superseded_by is not None:
            later = await session.get(models.UserModelFacet, chain[-1].superseded_by)
            if later is None:
                break
            chain.append(later)

        evidence = await _evidence_for(session, [row.id for row in chain])
    return [_row(row, evidence.get(row.id, ())) for row in chain]


async def _evidence_for(
    session: AsyncSession, facet_ids: Sequence[UUID]
) -> dict[UUID, tuple[tuple[str, UUID, str], ...]]:
    if not facet_ids:
        return {}
    rows = list(
        (
            await session.execute(
                select(models.FacetEvidence).where(
                    models.FacetEvidence.facet_id.in_(list(facet_ids))
                )
            )
        ).scalars()
    )
    grouped: dict[UUID, list[tuple[str, UUID, str]]] = {}
    for row in rows:
        grouped.setdefault(row.facet_id, []).append(
            (row.kind, row.ref_id, row.relation)
        )
    return {key: tuple(value) for key, value in grouped.items()}


def _row(
    facet: models.UserModelFacet, evidence: tuple[tuple[str, UUID, str], ...]
) -> FacetRow:
    return FacetRow(
        id=facet.id,
        dimension=facet.dimension,
        statement=facet.statement,
        confidence=facet.confidence,
        support_count=facet.support_count,
        contradiction_count=facet.contradiction_count,
        origin=facet.origin,
        detector=facet.detector,
        subject_key=facet.subject_key,
        first_observed=facet.first_observed,
        last_observed=facet.last_observed,
        superseded_by=facet.superseded_by,
        dismissed_at=facet.dismissed_at,
        dismissed_reason=facet.dismissed_reason,
        created_at=facet.created_at,
        evidence=evidence,
    )
