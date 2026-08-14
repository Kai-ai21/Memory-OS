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

### Withdraw, never delete (M8.2)

A re-derivation that finds **no** candidate for a live facet's subject retires it
with `superseded_at` and a written reason, and points `superseded_by` at nothing,
because there is nothing to point at. This is the other half of "supersede, never
update": under M8.0 the only way to stop being live was to be replaced, so a
facet whose evidence had gone stayed live indefinitely.

Deleting it was the alternative and it is the one thing M1.1 spent a phase
arguing against. An honest record of a claim the system used to make beats a
clean current state, and `model diff`, `model timeline` and `model stability`
exist to read that record.

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
    Stability,
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
    # M8.2. Facets retired because the evidence under them went away, with no
    # replacement statement. Counted apart from `superseded` because they are a
    # different event: a revision says the claim moved, a withdrawal says the
    # claim no longer has anything behind it, and pooling them would hide the
    # second inside the first.
    withdrawn: int = 0
    assessments: tuple[Assessment, ...] = ()
    rejected: tuple[tuple[str, int], ...] = ()
    # What was withdrawn and why, so a run that quietly retired five facets is
    # not a run that printed one number.
    withdrawals: tuple[tuple[str, str], ...] = ()


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
    superseded_at: datetime | None
    superseded_reason: str | None
    dismissed_at: datetime | None
    dismissed_reason: str | None
    created_at: datetime
    evidence: tuple[tuple[str, UUID, str], ...] = ()

    @property
    def live(self) -> bool:
        # **Read from `superseded_at`, not from `superseded_by`.** A facet whose
        # evidence went away is superseded by nothing, so the pointer is null and
        # the timestamp is not — and reading the pointer here is exactly how such
        # a facet would go on being displayed as current.
        return self.superseded_at is None and self.dismissed_at is None

    @property
    def withdrawn(self) -> bool:
        """Superseded with no replacement: the support disappeared."""
        return self.superseded_at is not None and self.superseded_by is None


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
    """Run every deriver, write what clears the bar, retire what lost its support.

    **The retirement pass is M8.2's half and it is the one that makes this a
    re-derivation rather than an append.** Every deriver runs over current data,
    so a facet whose subject nobody proposes any more is a facet whose evidence
    has gone — a group was regrouped, a pattern was dismissed, three decisions
    became two. Under M8.0 such a row stayed live indefinitely, asserting
    something with nothing behind it, because the only way to stop being live was
    to be replaced by a new statement and there was no new statement.

    It is retired, not deleted, and the reason is written down. M1.1's principle
    applied to a claim about a person: an honest record of a belief that changed
    beats a clean current state, and "you used to look like this and stopped in
    March" is only answerable while March's row exists.
    """
    written = superseded = unchanged = below_bar = skipped = withdrawn = 0
    considered: list[Candidate] = []
    rejected: list[tuple[str, int]] = []
    withdrawals: list[tuple[str, str]] = []
    now = datetime.now(UTC)

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
                        models.UserModelFacet.superseded_at.is_(None),
                        models.UserModelFacet.dismissed_at.is_(None),
                    )
                )
            ).scalar_one_or_none()

            if existing is not None and existing.statement == candidate.statement:
                # Same subject, same sentence. Running `derive` twice is not two
                # rows and not two timestamps.
                unchanged += 1
                continue

            facet_id = new_id()
            if existing is not None:
                # **The old row is retired before the new one is written**, and
                # the order is forced by the partial unique index: it allows one
                # live row per (detector, subject), so inserting first would put
                # two live rows in the table for the length of a statement and
                # Postgres rejects that. Supersede rather than update, because
                # how the model changed is part of the model and an UPDATE would
                # leave nothing to compare.
                existing.superseded_by = facet_id
                existing.superseded_at = now
                existing.superseded_reason = _restated_reason(existing, candidate)
                await session.flush()
                superseded += 1
            await _insert(session, candidate, facet_id=facet_id)
            written += 1

        # --------------------------------------------------------------
        # Retirement: facets whose support is no longer there
        # --------------------------------------------------------------
        #
        # Keyed by subject rather than by id, because the question is whether the
        # *claim* still has evidence. Candidates that fell below the bar are in
        # this map too, and a facet whose candidate is now below the bar is
        # retired with the number it fell to — which is a more useful sentence
        # than the absence of one, since it says how far the support moved rather
        # than only that it did.
        by_subject: dict[tuple[str | None, str | None], Candidate] = {
            (item.detector, item.subject_key): item for item in considered
        }
        stale = list(
            (
                await session.execute(
                    select(models.UserModelFacet).where(
                        models.UserModelFacet.origin == FacetOrigin.DERIVED.value,
                        models.UserModelFacet.superseded_at.is_(None),
                        models.UserModelFacet.dismissed_at.is_(None),
                    )
                )
            ).scalars()
        )
        for facet in stale:
            proposal = by_subject.get((facet.detector, facet.subject_key))
            if proposal is not None and proposal.clears:
                continue
            reason = _withdrawal_reason(facet, proposal)
            # **No `superseded_by`.** There is no replacement — that is the whole
            # difference between this and a revision, and inventing a successor
            # row to point at would put a statement in the table that no deriver
            # produced.
            facet.superseded_at = now
            facet.superseded_reason = reason
            withdrawn += 1
            withdrawals.append((facet.statement, reason))

    assessments = await assess(session_factory)
    report = DeriveReport(
        written=written,
        superseded=superseded,
        unchanged=unchanged,
        considered=len(considered),
        below_bar=below_bar,
        skipped_dismissed=skipped,
        withdrawn=withdrawn,
        assessments=assessments,
        rejected=tuple(rejected),
        withdrawals=tuple(withdrawals),
    )
    logger.info(
        "user_model.derived",
        written=written,
        superseded=superseded,
        unchanged=unchanged,
        considered=len(considered),
        below_bar=below_bar,
        skipped_dismissed=skipped,
        withdrawn=withdrawn,
    )
    return report


def _restated_reason(
    existing: models.UserModelFacet, candidate: Candidate
) -> str:
    """Why a revision happened, in numbers rather than in the word "changed".

    The statements here are templates with counts in them, so a statement that
    moved is a count that moved, and naming which one moved is free.
    """
    return (
        f"re-derived: support {existing.support_count} → {candidate.supporting}, "
        f"against {existing.contradiction_count} → {candidate.contradicting}"
    )


def _withdrawal_reason(
    facet: models.UserModelFacet, candidate: Candidate | None
) -> str:
    """Why a facet lost its support, distinguishing the two ways it can happen.

    A candidate that fell below the bar and a subject nobody proposes any more
    are different findings. The first says the evidence thinned and by how much;
    the second says the thing the facet was about is not in the data at all —
    a group was disbanded, a pattern was dismissed, a decision was edited.
    """
    if candidate is not None:
        return (
            f"support fell from {facet.support_count} to {candidate.supporting}, "
            f"below the bar of {MIN_SUPPORT} distinct observations"
        )
    return (
        f"the {facet.detector} deriver no longer proposes this subject: "
        "the evidence it rested on is not in the corpus any more"
    )


async def _insert(
    session: AsyncSession, candidate: Candidate, *, facet_id: UUID
) -> models.UserModelFacet:
    """Write the facet, then its evidence.

    The flush between them is not incidental. `user_model_facets` carries a
    self-referential foreign key, and with one in the table SQLAlchemy's unit of
    work stops ordering these two inserts by their dependency — the evidence goes
    first and Postgres rejects it for pointing at a row that does not exist yet.
    Flushing the parent explicitly says the order rather than hoping for it.
    """
    facet = models.UserModelFacet(
        id=facet_id,
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
    await session.flush()
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
                        models.UserModelFacet.superseded_at.is_(None),
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
            previous.superseded_at = datetime.now(UTC)
            previous.superseded_reason = "restated by hand"
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
            models.UserModelFacet.superseded_at.is_(None)
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
        superseded_at=facet.superseded_at,
        superseded_reason=facet.superseded_reason,
        dismissed_at=facet.dismissed_at,
        dismissed_reason=facet.dismissed_reason,
        created_at=facet.created_at,
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# M8.2: how the model changed
# --------------------------------------------------------------------------
#
# **The history of how the model changed is part of the model.** A facet that
# held for six months and then stopped is more informative than either state
# alone: the current state says what the system believes, and only the history
# says whether it has ever been wrong about it.
#
# Everything below reads the same three timestamps — `created_at`,
# `superseded_at`, `dismissed_at` — and nothing below writes. A view over a log,
# in the shape M4.0's timeline already established.


@dataclass(frozen=True, slots=True)
class FacetChange:
    """One thing that happened to one facet, inside a window.

    `replacement` is what separates a revision from a withdrawal, and it is null
    for the second because there is nothing to point at. A reader looking at a
    superseded facet needs to know which of the two it was: "the claim moved" and
    "the claim lost its evidence" are opposite findings about the same row.
    """

    at: datetime
    facet: FacetRow
    reason: str = ""
    replacement: FacetRow | None = None

    @property
    def withdrawn(self) -> bool:
        return self.replacement is None

    @property
    def confidence_move(self) -> tuple[float, float] | None:
        """Before and after, when both ends are derived facets with a number.

        None rather than a zero delta when either end has no confidence, because
        an asserted facet's null is not a low confidence — it is a claim
        confidence does not apply to, and subtracting from it would invent a
        movement.
        """
        if self.replacement is None:
            return None
        before, after = self.facet.confidence, self.replacement.confidence
        if before is None or after is None:
            return None
        return (before, after)


@dataclass(frozen=True, slots=True)
class ModelDiff:
    """What changed between two moments, in the four ways it can change.

    **A window with nothing in it is a result.** On a corpus where the model has
    no facets at all, every list here is empty and the command says so with the
    two dates — which is a different statement from "the model is unchanged", and
    the rendering keeps them apart.
    """

    since: datetime
    until: datetime
    added: tuple[FacetChange, ...] = ()
    superseded: tuple[FacetChange, ...] = ()
    dismissed: tuple[FacetChange, ...] = ()

    @property
    def confidence_changes(self) -> tuple[tuple[FacetChange, float, float], ...]:
        """Every revision where the number moved, with both ends.

        Derived from the revisions rather than stored beside them: a confidence
        change without the statement that produced it is a number nobody can
        check, and every confidence change here *is* a supersession, because a
        facet is never updated in place.
        """
        moves = []
        for change in self.superseded:
            pair = change.confidence_move
            if pair is not None and pair[0] != pair[1]:
                moves.append((change, pair[0], pair[1]))
        return tuple(moves)

    @property
    def empty(self) -> bool:
        return not (self.added or self.superseded or self.dismissed)


async def diff(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    since: datetime,
    until: datetime | None = None,
) -> ModelDiff:
    """Every change to the model inside a window, by the kind of change it was.

    **The replacement half of a revision is not also reported as an addition.**
    A statement that moved produces two rows — the old one retired and the new
    one inserted — and reporting both would make one event look like two, with
    the added list carrying a sentence that is really the second half of a
    supersession already listed above it. So a row that some superseded facet in
    this window points at is excluded from `added`.
    """
    window_end = until or datetime.now(UTC)
    async with session_factory() as session:
        rows = list(
            (
                await session.execute(
                    select(models.UserModelFacet).where(
                        # Anything that had an event in the window, plus the
                        # replacements those events point at, which may have been
                        # created at any time.
                        (
                            models.UserModelFacet.created_at.between(
                                since, window_end
                            )
                        )
                        | (
                            models.UserModelFacet.superseded_at.between(
                                since, window_end
                            )
                        )
                        | (
                            models.UserModelFacet.dismissed_at.between(
                                since, window_end
                            )
                        )
                    )
                )
            ).scalars()
        )
        # Replacements may fall outside the window if the clocks disagree, so
        # they are fetched by id rather than assumed present.
        wanted = {
            row.superseded_by
            for row in rows
            if row.superseded_by is not None
        } - {row.id for row in rows}
        if wanted:
            rows.extend(
                (
                    await session.execute(
                        select(models.UserModelFacet).where(
                            models.UserModelFacet.id.in_(list(wanted))
                        )
                    )
                )
                .scalars()
                .all()
            )
        evidence = await _evidence_for(session, [row.id for row in rows])

    built = {row.id: _row(row, evidence.get(row.id, ())) for row in rows}

    superseded: list[FacetChange] = []
    dismissed: list[FacetChange] = []
    replacements: set[UUID] = set()
    for facet in built.values():
        if facet.superseded_at is not None and since <= facet.superseded_at <= window_end:
            replacement = (
                built.get(facet.superseded_by)
                if facet.superseded_by is not None
                else None
            )
            if replacement is not None:
                replacements.add(replacement.id)
            superseded.append(
                FacetChange(
                    at=facet.superseded_at,
                    facet=facet,
                    reason=facet.superseded_reason or "",
                    replacement=replacement,
                )
            )
        if facet.dismissed_at is not None and since <= facet.dismissed_at <= window_end:
            dismissed.append(
                FacetChange(
                    at=facet.dismissed_at,
                    facet=facet,
                    reason=facet.dismissed_reason or "",
                )
            )

    added = [
        FacetChange(at=facet.created_at, facet=facet)
        for facet in built.values()
        if since <= facet.created_at <= window_end and facet.id not in replacements
    ]

    return ModelDiff(
        since=since,
        until=window_end,
        added=tuple(sorted(added, key=lambda item: item.at)),
        superseded=tuple(sorted(superseded, key=lambda item: item.at)),
        dismissed=tuple(sorted(dismissed, key=lambda item: item.at)),
    )


@dataclass(frozen=True, slots=True)
class ModelEvent:
    """One dated thing that happened to the model, for the timeline."""

    at: datetime
    # `written`, `revised`, `withdrawn` or `dismissed`.
    kind: str
    dimension: str
    facet_id: UUID
    statement: str
    detail: str = ""


async def timeline(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    dimension: Dimension | None = None,
) -> tuple[ModelEvent, ...]:
    """Every event in the model's life, oldest first.

    **One row can produce three events**, and they are separate entries rather
    than columns of one: a facet was written on one day, revised on another and
    dismissed on a third, and a timeline that collapsed those into the row's
    current state would be `model show` with dates on it.
    """
    async with session_factory() as session:
        stmt = select(models.UserModelFacet)
        if dimension is not None:
            stmt = stmt.where(models.UserModelFacet.dimension == dimension.value)
        rows = list((await session.execute(stmt)).scalars())

    events: list[ModelEvent] = []
    for row in rows:
        origin = "stated" if row.origin == FacetOrigin.ASSERTED.value else row.detector
        events.append(
            ModelEvent(
                at=row.created_at,
                kind="written",
                dimension=row.dimension,
                facet_id=row.id,
                statement=row.statement,
                detail=f"[{origin}] support {row.support_count}"
                f", against {row.contradiction_count}",
            )
        )
        if row.superseded_at is not None:
            events.append(
                ModelEvent(
                    at=row.superseded_at,
                    # The distinction the whole M8.2 schema change exists for.
                    kind="revised" if row.superseded_by is not None else "withdrawn",
                    dimension=row.dimension,
                    facet_id=row.id,
                    statement=row.statement,
                    detail=row.superseded_reason or "",
                )
            )
        if row.dismissed_at is not None:
            events.append(
                ModelEvent(
                    at=row.dismissed_at,
                    kind="dismissed",
                    dimension=row.dimension,
                    facet_id=row.id,
                    statement=row.statement,
                    detail=row.dismissed_reason or "",
                )
            )
    return tuple(sorted(events, key=lambda item: (item.at, item.kind)))


async def stability(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    now: datetime | None = None,
) -> tuple[Stability, ...]:
    """How often each dimension's facets change, and how long they last.

    **Every dimension is returned, including the ones with no facets.** M8.0's
    argument for printing empty dimensions applies here twice over: a stability
    table that listed only the dimensions with history would be a table of five
    rows implying two were stable, and the two with nothing are the ones whose
    absence is the finding.
    """
    moment = now or datetime.now(UTC)
    async with session_factory() as session:
        rows = list((await session.execute(select(models.UserModelFacet))).scalars())

    by_dimension: dict[str, list[models.UserModelFacet]] = {}
    for row in rows:
        by_dimension.setdefault(row.dimension, []).append(row)

    entries: list[Stability] = []
    for dimension in Dimension:
        group = by_dimension.get(dimension.value, [])
        lifetimes: list[float] = []
        ages: list[float] = []
        changes = 0
        live = 0
        for row in group:
            # A facet can be superseded and later dismissed, or the reverse.
            # Both are changes; the earlier of the two ends its life.
            ended_at = min(
                (when for when in (row.superseded_at, row.dismissed_at) if when),
                default=None,
            )
            changes += sum(
                1 for when in (row.superseded_at, row.dismissed_at) if when is not None
            )
            if ended_at is None:
                live += 1
                ages.append((moment - row.created_at).total_seconds() / 86400)
            else:
                lifetimes.append((ended_at - row.created_at).total_seconds() / 86400)
        observed = (
            (moment - min(row.created_at for row in group)).total_seconds() / 86400
            if group
            else 0.0
        )
        entries.append(
            Stability(
                dimension=dimension,
                total=len(group),
                live=live,
                closed=len(lifetimes),
                changes=changes,
                mean_lifetime_days=(
                    sum(lifetimes) / len(lifetimes) if lifetimes else None
                ),
                mean_live_age_days=sum(ages) / len(ages) if ages else None,
                observed_days=observed,
            )
        )
    return tuple(entries)
