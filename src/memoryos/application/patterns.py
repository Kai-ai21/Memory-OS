"""Four detectors, and the bar every one of them has to clear.

**The failure mode, stated before the code.** "You consistently underestimate
deployment effort" is either a finding backed by five specific decisions or a
horoscope — vague enough to feel true about anyone — and nothing in the sentence
tells you which. A system that produces confident behavioural claims from thin
evidence is worse than one that stays silent, because it sounds exactly like the
product working.

So every candidate passes three gates before it becomes a row, and each rejects
for a different reason:

1. **Support**, counted in *distinct decisions* rather than in rows. Four
   assumptions from two decisions is two observations of one person on two
   occasions. Three is the floor and `--min-support` exists to raise it.
2. **Counter-evidence**, searched for deliberately rather than noticed. Every
   detector runs a second query for the decisions that argue against its own
   candidate, and a candidate with more contradicting than supporting evidence
   is not emitted at all.
3. **Resolution**, for anything arithmetic. A calibration gap is only evidence
   if the observed rate falls outside what a sample that size can support —
   `domain/patterns.wilson_interval`. Fourteen assumptions that all held cannot
   distinguish "right 85% of the time" from "right always", so a stated 0.85
   against 14/14 is *not* a finding. This is M2.3a's resolution floor applied to
   behaviour, and it is the gate that keeps this module quiet on a small corpus.

Nothing here calls a language model. The detectors are rules over recorded data
and their statements are assembled from numbers, which means every claim can be
recomputed by hand from the same rows — the property M5.4's reflections will
need and the one a generated sentence could never have.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.ids import new_id
from memoryos.domain.patterns import (
    DEFAULT_MIN_SUPPORT,
    Interval,
    is_emittable,
    is_miscalibrated,
    pattern_confidence,
    wilson_interval,
)
from memoryos.domain.values import (
    AssumptionVerdict,
    DecisionStatus,
    OutcomeVerdict,
    PatternKind,
    PatternRelation,
)

logger = structlog.get_logger(__name__)

# Confidence bands for the calibration detectors. Quarters, because they are the
# coarsest split that still separates "I was unsure" from "I was certain", and
# because a finer grid over a corpus of this size would put one observation in
# each bucket and find a gap in every one of them.
_BANDS: tuple[tuple[float, float], ...] = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def _band_of(confidence: float) -> tuple[float, float]:
    for low, high in _BANDS:
        if low <= confidence < high:
            return (low, high)
    return _BANDS[-1]


class UnknownPattern(LookupError):
    """No pattern with that id."""


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    decision_id: UUID
    relation: PatternRelation
    note: str
    assumption_id: UUID | None = None
    outcome_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    """A proposed pattern, with everything needed to accept or reject it."""

    kind: PatternKind
    detector: str
    subject_key: str
    statement: str
    evidence: tuple[EvidenceItem, ...]
    # The numbers behind the statement, carried rather than embedded in it.
    # `calibration()` reads these; without them it would have to parse the
    # sentence it is reporting, and a value recovered from a display string is
    # a value that can disagree with the rows printed under it.
    #
    # Filled by the calibration detectors and only by them, because they are the
    # only ones with a reader. Every other detector's numbers are recoverable by
    # counting its evidence lists, and a dict populated for nobody is a second
    # copy of those counts that nothing keeps honest.
    metrics: dict[str, float] = field(default_factory=dict)
    # Why this candidate was not emitted, when it was not. Carried so the CLI
    # can report near-misses: on a small corpus the interesting output is
    # usually what *almost* cleared the bar, and a run that printed nothing
    # would not distinguish "no candidates" from "four candidates, all thin".
    rejected_because: str | None = None

    @property
    def supporting(self) -> frozenset[UUID]:
        return frozenset(
            item.decision_id
            for item in self.evidence
            if item.relation is PatternRelation.SUPPORTS
        )

    @property
    def contradicting(self) -> frozenset[UUID]:
        return frozenset(
            item.decision_id
            for item in self.evidence
            if item.relation is PatternRelation.CONTRADICTS
        )

    def confidence(self, *, min_support: int) -> float:
        return pattern_confidence(
            len(self.supporting), len(self.contradicting), min_support=min_support
        )


@dataclass(slots=True)
class DiscoveryReport:
    candidates: int = 0
    emitted: int = 0
    updated: int = 0
    below_support: int = 0
    outweighed_by_counter_evidence: int = 0
    within_noise: int = 0
    skipped_dismissed: int = 0
    # Detectors that produced no candidate at all, with the reason. Distinct
    # from a candidate that was rejected: a detector with nothing to say is
    # structurally inert on this corpus, and a report that showed only the
    # rejections would hide that half the machinery never ran.
    silent: list[tuple[str, str]] = field(default_factory=list)
    # Every candidate considered, kept so the CLI can show what nearly made it.
    considered: list[Candidate] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "candidates": self.candidates,
            "emitted": self.emitted,
            "updated": self.updated,
            "below_support": self.below_support,
            "outweighed_by_counter_evidence": self.outweighed_by_counter_evidence,
            "within_noise": self.within_noise,
            "skipped_dismissed": self.skipped_dismissed,
        }


# --------------------------------------------------------------------------
# Reading the corpus once
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Assumption:
    id: UUID
    decision_id: UUID
    statement: str
    confidence: float | None
    held: AssumptionVerdict | None
    group_id: UUID | None
    group_label: str | None


@dataclass(frozen=True, slots=True)
class _Decision:
    id: UUID
    question: str
    chosen: str
    confidence: float | None
    status: DecisionStatus
    decided_at: datetime


@dataclass(frozen=True, slots=True)
class _Outcome:
    id: UUID
    decision_id: UUID
    verdict: OutcomeVerdict
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class Corpus:
    """Everything the detectors read, fetched once.

    A snapshot rather than a set of queries per detector, because four detectors
    each running their own joins over the same four tables is four chances for
    two of them to disagree about what the corpus contains — and because at this
    size the whole of Phase 5's data is a few hundred rows.
    """

    decisions: tuple[_Decision, ...]
    assumptions: tuple[_Assumption, ...]
    outcomes: tuple[_Outcome, ...]

    @property
    def latest_outcome(self) -> dict[UUID, _Outcome]:
        """One verdict per decision, the most recent by `observed_at`.

        A decision that worked and then failed is one decision that failed, not
        one of each — the same collapse `outcomes.success_rate` makes, and for
        the same reason: a detector counting both would find a pattern in a
        single decision changing its mind.
        """
        latest: dict[UUID, _Outcome] = {}
        for outcome in sorted(self.outcomes, key=lambda row: row.observed_at):
            latest[outcome.decision_id] = outcome
        return latest


async def read_corpus(sessions: async_sessionmaker[AsyncSession]) -> Corpus:
    async with sessions() as session:
        decisions = [
            _Decision(
                id=row.id,
                question=row.question,
                chosen=row.chosen,
                confidence=row.confidence,
                status=DecisionStatus(row.status),
                decided_at=row.decided_at,
            )
            for row in (await session.execute(select(models.Decision))).scalars()
        ]
        assumptions = [
            _Assumption(
                id=row[0].id,
                decision_id=row[0].decision_id,
                statement=row[0].statement,
                confidence=row[0].confidence,
                held=AssumptionVerdict(row[0].held) if row[0].held else None,
                group_id=row[0].group_id,
                group_label=row[1],
            )
            for row in await session.execute(
                select(models.DecisionAssumption, models.AssumptionGroup.label).outerjoin(
                    models.AssumptionGroup,
                    models.AssumptionGroup.id == models.DecisionAssumption.group_id,
                )
            )
        ]
        outcomes = [
            _Outcome(
                id=row.id,
                decision_id=row.decision_id,
                verdict=OutcomeVerdict(row.verdict),
                observed_at=row.observed_at,
            )
            for row in (await session.execute(select(models.DecisionOutcome))).scalars()
        ]
    return Corpus(
        decisions=tuple(decisions),
        assumptions=tuple(assumptions),
        outcomes=tuple(outcomes),
    )


# --------------------------------------------------------------------------
# The detectors
# --------------------------------------------------------------------------


def detect_assumption_patterns(corpus: Corpus) -> list[Candidate]:
    """A belief that keeps breaking, across decisions that share nothing else.

    Reads M5.2's groups, because "the same assumption" is not a string
    comparison — "this will take two days", "the deploy is straightforward" and
    "integration should be quick" are one recurring belief in three sentences,
    and ungrouped each is a sample of size one.

    Supporting evidence is a member that failed or half-failed; contradicting
    is a member that held. **Both come from the same group**, which is what
    makes the counter-evidence search honest rather than a second query
    somebody could forget: the members that held are found by the same pass
    that finds the ones that broke.
    """
    by_group: dict[UUID, list[_Assumption]] = {}
    for assumption in corpus.assumptions:
        if assumption.group_id is not None:
            by_group.setdefault(assumption.group_id, []).append(assumption)

    candidates: list[Candidate] = []
    for group_id, members in by_group.items():
        evaluated = [member for member in members if member.held is not None]
        if not evaluated:
            continue
        label = next(
            (member.group_label for member in members if member.group_label), "?"
        )
        broke = [
            member
            for member in evaluated
            if member.held
            in (AssumptionVerdict.FAILED, AssumptionVerdict.PARTIALLY)
        ]
        held = [member for member in evaluated if member.held is AssumptionVerdict.HELD]

        evidence = [
            EvidenceItem(
                decision_id=member.decision_id,
                relation=PatternRelation.SUPPORTS,
                assumption_id=member.id,
                note=f"{member.held.value if member.held else '?'}: {member.statement}",
            )
            for member in broke
        ] + [
            EvidenceItem(
                decision_id=member.decision_id,
                relation=PatternRelation.CONTRADICTS,
                assumption_id=member.id,
                note=f"held: {member.statement}",
            )
            for member in held
        ]

        rate = len(held) / len(evaluated)
        # The statement describes what was observed and only *claims* something
        # when the observation supports the claim. A candidate that reads "this
        # breaks more often than it holds: held 100%" is a sentence arguing with
        # itself, and it would be printed verbatim in the near-miss listing.
        finding = (
            "A recurring assumption breaks more often than it holds: "
            if rate < 0.5
            else "A recurring assumption, held more often than not: "
        )
        candidates.append(
            Candidate(
                kind=PatternKind.ASSUMPTION,
                detector="assumption_group",
                subject_key=str(group_id),
                statement=(
                    f"{finding}{label!r} held {rate:.0%} of {len(evaluated)} times "
                    f"it was evaluated, across "
                    f"{len({m.decision_id for m in evaluated})} decisions."
                ),
                evidence=tuple(evidence),
            )
        )
    return candidates


def detect_timing_patterns(
    corpus: Corpus, *, window_days: float = 90.0
) -> list[Candidate]:
    """Outcomes arriving much later than the decision implied.

    **`expected_outcome` is prose and this does not parse it.** A decision's
    expectation is a sentence somebody wrote — "throughput is never the binding
    constraint" — with no date in it, and extracting one would mean asking a
    model to invent a deadline and then measuring against the invention. So the
    proxy is M5.1's window: the interval derived from the decision's own
    confidence, which is the system's stated guess at how long the answer should
    take to arrive.

    An outcome observed beyond that window supports the pattern; one inside it
    contradicts. Stated plainly because the proxy is doing real work and could
    be wrong: this measures *when somebody recorded an outcome*, which on a
    corpus where every outcome was written in one sitting is a fact about the
    sitting rather than about the work.
    """
    latest = corpus.latest_outcome
    supporting: list[EvidenceItem] = []
    contradicting: list[EvidenceItem] = []

    for decision in corpus.decisions:
        outcome = latest.get(decision.id)
        if outcome is None or outcome.verdict is OutcomeVerdict.TOO_EARLY:
            continue
        gap = (outcome.observed_at - decision.decided_at).total_seconds() / 86400.0
        if gap <= 0:
            continue
        item = EvidenceItem(
            decision_id=decision.id,
            relation=(
                PatternRelation.SUPPORTS
                if gap > window_days
                else PatternRelation.CONTRADICTS
            ),
            outcome_id=outcome.id,
            note=(
                f"resolved {gap:.0f} days after deciding, against a "
                f"{window_days:.0f}-day window"
            ),
        )
        (supporting if gap > window_days else contradicting).append(item)

    if not supporting:
        return []
    return [
        Candidate(
            kind=PatternKind.TIMING,
            detector="slow_resolution",
            subject_key=f"window={window_days:.0f}",
            statement=(
                f"Outcomes tend to arrive later than expected: "
                f"{len(supporting)} of {len(supporting) + len(contradicting)} "
                f"resolved decisions took longer than the {window_days:.0f}-day "
                f"window their confidence implied."
            ),
            evidence=tuple(supporting + contradicting),
        )
    ]


def detect_choice_patterns(corpus: Corpus) -> list[Candidate]:
    """Reaching for the same kind of option repeatedly, or reversing repeatedly.

    Two rules, and only the second is implemented as arithmetic here.

    **Reversals** are countable without interpretation: a decision whose status
    is `reversed` was undone, and a person who reverses often is a person whose
    first answer tends not to survive. Supporting evidence is a reversed
    decision; contradicting is a settled one, because a corpus full of settled
    decisions is the direct counter-example to "you keep changing your mind".

    **"The same kind of option"** is deliberately *not* implemented by
    clustering the `chosen` text. It would work — M5.2's embedder is right
    there — and it would produce exactly the failure this milestone is arranged
    against: a cluster of three choices that happen to be phrased alike, turned
    into a sentence about somebody's habits. Categorising choices needs a
    vocabulary the corpus does not have, and inventing one here would mean the
    detector deciding what the categories are and then finding them.
    """
    reversed_decisions = [
        decision
        for decision in corpus.decisions
        if decision.status is DecisionStatus.REVERSED
    ]
    if not reversed_decisions:
        return []
    settled = [
        decision
        for decision in corpus.decisions
        if decision.status is DecisionStatus.SETTLED
    ]
    evidence = [
        EvidenceItem(
            decision_id=decision.id,
            relation=PatternRelation.SUPPORTS,
            note=f"reversed: {decision.question}",
        )
        for decision in reversed_decisions
    ] + [
        EvidenceItem(
            decision_id=decision.id,
            relation=PatternRelation.CONTRADICTS,
            note=f"settled and still in force: {decision.question}",
        )
        for decision in settled
    ]
    return [
        Candidate(
            kind=PatternKind.CHOICE,
            detector="reversal_rate",
            subject_key="reversals",
            statement=(
                f"Decisions get reversed rather than settling: "
                f"{len(reversed_decisions)} reversed against {len(settled)} settled."
            ),
            evidence=tuple(evidence),
        )
    ]


def detect_calibration_patterns(corpus: Corpus) -> list[Candidate]:
    """Stated confidence against what actually happened.

    **The most interesting detector and the most honest, because it needs no
    interpretation — it is arithmetic over numbers recorded before the answer
    was known.** Two populations, run separately because they are different
    claims: decision confidence against outcome verdicts, and assumption
    confidence against hold verdicts.

    A band is only a finding when the observed rate falls *outside* the Wilson
    interval for that sample. Fourteen assumptions that all held cannot
    distinguish "right 85% of the time" from "right always", so a stated 0.85
    against 14/14 is consistent with perfect calibration and is not emitted. See
    `domain/patterns.is_miscalibrated`.

    **A caveat this detector cannot check and every reader needs.** Calibration
    is only meaningful when the confidence was written down before the outcome
    was known. Nothing in the schema records whether it was — `decisions.confidence`
    is never updated, which guarantees it did not move, and guarantees nothing
    about when it was first set. A corpus of confidences reconstructed after the
    fact would produce a calibration report measuring hindsight, and it would
    look exactly like this one.
    """
    candidates: list[Candidate] = []
    latest = corpus.latest_outcome

    # Decisions: stated confidence against whether the decision worked.
    by_band: dict[tuple[float, float], list[tuple[_Decision, _Outcome]]] = {}
    for decision in corpus.decisions:
        outcome = latest.get(decision.id)
        if decision.confidence is None or outcome is None:
            continue
        if outcome.verdict is OutcomeVerdict.TOO_EARLY:
            # Not a result. Counting it either way would make the calibration a
            # measurement of how much is still unresolved.
            continue
        by_band.setdefault(_band_of(decision.confidence), []).append((decision, outcome))

    for (low, high), rows in by_band.items():
        worked = [
            (decision, outcome)
            for decision, outcome in rows
            if outcome.verdict is OutcomeVerdict.WORKED
        ]
        did_not = [pair for pair in rows if pair not in worked]
        stated = sum(decision.confidence or 0.0 for decision, _ in rows) / len(rows)
        candidates.append(
            _calibration_candidate(
                detector="decision_calibration",
                subject_key=f"{low:.2f}-{high:.2f}",
                noun="decisions",
                band=(low, high),
                stated=stated,
                successes=len(worked),
                total=len(rows),
                supporting=[
                    EvidenceItem(
                        decision_id=decision.id,
                        relation=PatternRelation.SUPPORTS,
                        outcome_id=outcome.id,
                        note=(
                            f"stated {decision.confidence:.2f}, outcome "
                            f"{outcome.verdict.value}"
                        ),
                    )
                    for decision, outcome in did_not
                ],
                contradicting=[
                    EvidenceItem(
                        decision_id=decision.id,
                        relation=PatternRelation.CONTRADICTS,
                        outcome_id=outcome.id,
                        note=(
                            f"stated {decision.confidence:.2f}, outcome "
                            f"{outcome.verdict.value}"
                        ),
                    )
                    for decision, outcome in worked
                ],
            )
        )

    # Assumptions: stated confidence against whether the belief held. A larger
    # sample than the decisions one on any corpus, because a decision has
    # several assumptions and only one outcome.
    assumption_bands: dict[tuple[float, float], list[_Assumption]] = {}
    for assumption in corpus.assumptions:
        if assumption.confidence is None or assumption.held is None:
            continue
        assumption_bands.setdefault(_band_of(assumption.confidence), []).append(assumption)

    for (low, high), members in assumption_bands.items():
        held = [item for item in members if item.held is AssumptionVerdict.HELD]
        broke = [item for item in members if item not in held]
        stated = sum(item.confidence or 0.0 for item in members) / len(members)
        candidates.append(
            _calibration_candidate(
                detector="assumption_calibration",
                subject_key=f"{low:.2f}-{high:.2f}",
                noun="assumptions",
                band=(low, high),
                stated=stated,
                successes=len(held),
                total=len(members),
                supporting=[
                    EvidenceItem(
                        decision_id=item.decision_id,
                        relation=PatternRelation.SUPPORTS,
                        assumption_id=item.id,
                        note=(
                            f"stated {item.confidence:.2f}, "
                            f"{item.held.value if item.held else '?'}: {item.statement}"
                        ),
                    )
                    for item in broke
                ],
                contradicting=[
                    EvidenceItem(
                        decision_id=item.decision_id,
                        relation=PatternRelation.CONTRADICTS,
                        assumption_id=item.id,
                        note=f"stated {item.confidence:.2f}, held: {item.statement}",
                    )
                    for item in held
                ],
            )
        )
    return candidates


def _calibration_candidate(
    *,
    detector: str,
    subject_key: str,
    noun: str,
    band: tuple[float, float],
    stated: float,
    successes: int,
    total: int,
    supporting: Sequence[EvidenceItem],
    contradicting: Sequence[EvidenceItem],
) -> Candidate:
    """One band's finding, or the reason it is not one.

    Which side is "supporting" flips with the direction of the miscalibration
    and that is not a trick: the claim being tested is "your stated confidence
    in this band is wrong", so the cases that argue *for* it are the ones that
    went against the stated number, and the cases that argue against it are the
    ones that matched. Under-confidence therefore cites the successes as
    counter-evidence and over-confidence cites the failures, which is why the
    statement below says which direction it found.
    """
    interval = wilson_interval(successes, total)
    within_noise = not is_miscalibrated(stated, successes, total)
    direction = "over" if interval.observed < stated else "under"

    # Under-confidence is the mirror image: the cases arguing for it are the
    # ones that *worked* despite a low stated number.
    if direction == "under":
        supporting, contradicting = contradicting, supporting

    return Candidate(
        kind=PatternKind.OUTCOME,
        detector=detector,
        subject_key=subject_key,
        metrics={
            "low": band[0],
            "high": band[1],
            "stated": stated,
            "successes": float(successes),
            "total": float(total),
        },
        statement=(
            f"{direction.capitalize()}confident on {noun} stated between "
            f"{band[0]:.2f} and {band[1]:.2f}: mean stated {stated:.2f}, "
            f"actual {interval.describe()}."
        ),
        evidence=tuple([*supporting, *contradicting]),
        rejected_because=(
            (
                f"stated {stated:.2f} falls inside the {interval.describe()} the "
                f"sample supports, so the gap is not evidence"
            )
            if within_noise
            else None
        ),
    )


def silent_detectors(
    corpus: Corpus, candidates: Sequence[Candidate], *, window_days: float
) -> list[tuple[str, str]]:
    """Detectors that produced nothing, and why.

    A detector with no candidate is not the same as a candidate that was
    rejected, and reporting only the rejections would hide that half this
    module never ran. On a corpus whose outcomes were all recorded in one
    sitting, the timing detector cannot fire — and that is a fact about the
    corpus worth printing rather than an absence to leave unexplained.
    """
    produced = {candidate.detector for candidate in candidates}
    silent: list[tuple[str, str]] = []

    if not any(name == "assumption_group" for name in produced):
        grouped = sum(1 for row in corpus.assumptions if row.group_id is not None)
        silent.append(
            (
                "assumption_group",
                f"{grouped} assumptions are grouped and none of the groups has an "
                f"evaluated member",
            )
        )
    if "slow_resolution" not in produced:
        latest = corpus.latest_outcome
        resolved = [
            outcome
            for outcome in latest.values()
            if outcome.verdict is not OutcomeVerdict.TOO_EARLY
        ]
        silent.append(
            (
                "slow_resolution",
                f"no outcome among {len(resolved)} resolved arrived more than "
                f"{window_days:.0f} days after its decision",
            )
        )
    if "reversal_rate" not in produced:
        silent.append(
            (
                "reversal_rate",
                f"no decision among {len(corpus.decisions)} has been reversed",
            )
        )
    return silent


def all_detectors(corpus: Corpus, *, window_days: float = 90.0) -> list[Candidate]:
    return [
        *detect_assumption_patterns(corpus),
        *detect_timing_patterns(corpus, window_days=window_days),
        *detect_choice_patterns(corpus),
        *detect_calibration_patterns(corpus),
    ]


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


async def discover(
    sessions: async_sessionmaker[AsyncSession],
    *,
    min_support: int = DEFAULT_MIN_SUPPORT,
    window_days: float = 90.0,
) -> DiscoveryReport:
    """Run every detector, and write only what clears all three gates.

    Re-running is idempotent: a pattern is keyed on `(detector, subject_key)`,
    so a second pass updates the counts and the statement of what it already
    found rather than adding a near-duplicate with a different percentage in it.
    A pattern somebody dismissed is left alone entirely — a refusal that a
    weekly re-run undid would not be a refusal.
    """
    report = DiscoveryReport()
    corpus = await read_corpus(sessions)
    candidates = all_detectors(corpus, window_days=window_days)
    report.candidates = len(candidates)
    report.considered = candidates
    report.silent = silent_detectors(corpus, candidates, window_days=window_days)

    async with sessions() as session:
        dismissed = {
            (row[0], row[1])
            for row in await session.execute(
                select(models.Pattern.detector, models.Pattern.subject_key).where(
                    models.Pattern.dismissed_at.is_not(None)
                )
            )
        }

    decided_at = {decision.id: decision.decided_at for decision in corpus.decisions}

    for candidate in candidates:
        if (candidate.detector, candidate.subject_key) in dismissed:
            report.skipped_dismissed += 1
            continue
        if candidate.rejected_because is not None:
            report.within_noise += 1
            continue
        support = len(candidate.supporting)
        counter = len(candidate.contradicting)
        # The gate is `domain.is_emittable` and only that, so the rule lives in
        # one place. The two branches below decide which *sentence* to count the
        # refusal under; they do not decide the refusal. Written the other way
        # round — two inline comparisons that happen to agree with the domain
        # function — the two could drift, and the drift would be invisible: the
        # table would gain a pattern the rule forbids and every test of the rule
        # would still pass.
        if not is_emittable(support, counter, min_support=min_support):
            if support < min_support:
                report.below_support += 1
            else:
                report.outweighed_by_counter_evidence += 1
            continue

        dates = [
            decided_at[decision_id]
            for decision_id in candidate.supporting
            if decision_id in decided_at
        ]
        created = await _upsert(
            sessions,
            candidate,
            min_support=min_support,
            first_observed=min(dates) if dates else None,
            last_observed=max(dates) if dates else None,
        )
        if created:
            report.emitted += 1
        else:
            report.updated += 1

    logger.info("patterns.discovered", **report.as_dict())
    return report


async def _upsert(
    sessions: async_sessionmaker[AsyncSession],
    candidate: Candidate,
    *,
    min_support: int,
    first_observed: datetime | None,
    last_observed: datetime | None,
) -> bool:
    """Write or refresh one pattern with its evidence. True when newly created.

    The evidence is replaced wholesale rather than merged. A pattern's evidence
    is the state of the corpus as the detector last read it, and merging would
    leave rows behind from a run where a decision still supported a claim it now
    argues against — which is the one way this table could come to disagree with
    the decisions it cites.
    """
    async with sessions.begin() as session:
        existing = (
            await session.execute(
                select(models.Pattern).where(
                    models.Pattern.detector == candidate.detector,
                    models.Pattern.subject_key == candidate.subject_key,
                )
            )
        ).scalar_one_or_none()

        pattern_id = existing.id if existing is not None else new_id()
        created = existing is None
        confidence = candidate.confidence(min_support=min_support)

        if existing is None:
            session.add(
                models.Pattern(
                    id=pattern_id,
                    statement=candidate.statement,
                    kind=candidate.kind.value,
                    detector=candidate.detector,
                    subject_key=candidate.subject_key,
                    support_count=len(candidate.supporting),
                    contradiction_count=len(candidate.contradicting),
                    confidence=confidence,
                    first_observed=first_observed,
                    last_observed=last_observed,
                )
            )
        else:
            existing.statement = candidate.statement
            existing.support_count = len(candidate.supporting)
            existing.contradiction_count = len(candidate.contradicting)
            existing.confidence = confidence
            existing.first_observed = first_observed
            existing.last_observed = last_observed
            await session.execute(
                delete(models.PatternEvidence).where(
                    models.PatternEvidence.pattern_id == pattern_id
                )
            )

        seen: set[tuple[UUID, UUID | None, UUID | None]] = set()
        for item in candidate.evidence:
            key = (item.decision_id, item.assumption_id, item.outcome_id)
            if key in seen:
                continue
            seen.add(key)
            session.add(
                models.PatternEvidence(
                    id=new_id(),
                    pattern_id=pattern_id,
                    decision_id=item.decision_id,
                    assumption_id=item.assumption_id,
                    outcome_id=item.outcome_id,
                    relation=item.relation.value,
                    note=item.note,
                )
            )
    return created


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    decision_id: UUID
    decision_question: str
    decided_at: datetime
    relation: PatternRelation
    note: str | None


@dataclass(frozen=True, slots=True)
class PatternRow:
    id: UUID
    statement: str
    kind: PatternKind
    detector: str
    support_count: int
    contradiction_count: int
    confidence: float | None
    first_observed: datetime | None
    last_observed: datetime | None
    discovered_at: datetime
    dismissed_at: datetime | None
    dismissed_reason: str | None
    supporting: list[EvidenceRow]
    contradicting: list[EvidenceRow]

    @property
    def span_days(self) -> float | None:
        """How long the supporting decisions span.

        Reported beside the support count because the two together are the
        claim's real strength: three decisions made in one afternoon are three
        observations of one mood.
        """
        if self.first_observed is None or self.last_observed is None:
            return None
        return (self.last_observed - self.first_observed).total_seconds() / 86400.0


async def list_patterns(
    sessions: async_sessionmaker[AsyncSession],
    *,
    kind: PatternKind | None = None,
    include_dismissed: bool = False,
    limit: int = 100,
) -> list[PatternRow]:
    stmt = (
        select(models.Pattern)
        .order_by(
            models.Pattern.confidence.desc().nullslast(),
            models.Pattern.support_count.desc(),
        )
        .limit(limit)
    )
    if kind is not None:
        stmt = stmt.where(models.Pattern.kind == kind.value)
    if not include_dismissed:
        stmt = stmt.where(models.Pattern.dismissed_at.is_(None))

    async with sessions() as session:
        rows = list((await session.execute(stmt)).scalars())
        if not rows:
            return []
        evidence = list(
            await session.execute(
                select(models.PatternEvidence, models.Decision)
                .join(
                    models.Decision,
                    models.Decision.id == models.PatternEvidence.decision_id,
                )
                .where(
                    models.PatternEvidence.pattern_id.in_([row.id for row in rows])
                )
                .order_by(models.Decision.decided_at)
            )
        )

    by_pattern: dict[UUID, list[tuple[models.PatternEvidence, models.Decision]]] = {}
    for item, decision in evidence:
        by_pattern.setdefault(item.pattern_id, []).append((item, decision))

    result: list[PatternRow] = []
    for row in rows:
        items = by_pattern.get(row.id, [])
        supporting = [
            _evidence_row(item, decision)
            for item, decision in items
            if item.relation == PatternRelation.SUPPORTS.value
        ]
        contradicting = [
            _evidence_row(item, decision)
            for item, decision in items
            if item.relation == PatternRelation.CONTRADICTS.value
        ]
        result.append(
            PatternRow(
                id=row.id,
                statement=row.statement,
                kind=PatternKind(row.kind),
                detector=row.detector,
                support_count=row.support_count,
                contradiction_count=row.contradiction_count,
                confidence=row.confidence,
                first_observed=row.first_observed,
                last_observed=row.last_observed,
                discovered_at=row.discovered_at,
                dismissed_at=row.dismissed_at,
                dismissed_reason=row.dismissed_reason,
                supporting=supporting,
                contradicting=contradicting,
            )
        )
    return result


def _evidence_row(
    item: models.PatternEvidence, decision: models.Decision
) -> EvidenceRow:
    return EvidenceRow(
        decision_id=decision.id,
        decision_question=decision.question,
        decided_at=decision.decided_at,
        relation=PatternRelation(item.relation),
        note=item.note,
    )


async def show(
    sessions: async_sessionmaker[AsyncSession], pattern_id: UUID
) -> PatternRow:
    rows = await list_patterns(sessions, include_dismissed=True, limit=1000)
    for row in rows:
        if row.id == pattern_id:
            return row
    raise UnknownPattern(f"no pattern {pattern_id}")


async def dismiss(
    sessions: async_sessionmaker[AsyncSession], pattern_id: UUID, *, reason: str
) -> None:
    """Reject a pattern permanently.

    Permanently because detection over a corpus this small is never going to be
    right every time, and a claim about somebody's judgement that they have
    looked at and refused should stay refused — `discover` skips dismissed
    subjects entirely rather than re-emitting them with a fresh id. The reason
    is required by a CHECK constraint: a rejection nobody explained is one the
    next reader cannot tell from a stale row.
    """
    if not reason.strip():
        raise ValueError("a dismissal needs a reason")
    async with sessions.begin() as session:
        row = await session.get(models.Pattern, pattern_id)
        if row is None:
            raise UnknownPattern(f"no pattern {pattern_id}")
        row.dismissed_at = datetime.now(UTC)
        row.dismissed_reason = reason.strip()
    logger.info("pattern.dismissed", pattern_id=str(pattern_id), reason=reason.strip())


# --------------------------------------------------------------------------
# The calibration table, which is worth reading whether or not it emits
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CalibrationBand:
    low: float
    high: float
    stated: float
    interval: Interval
    miscalibrated: bool

    @property
    def gap(self) -> float:
        return self.interval.observed - self.stated


async def calibration(
    sessions: async_sessionmaker[AsyncSession],
) -> dict[str, list[CalibrationBand]]:
    """Every band with its interval, emitted or not.

    Separate from `discover` because the table is worth reading even when — and
    especially when — nothing in it clears the bar. "No patterns found" and
    "here are four bands, and every stated confidence falls inside what its
    sample supports" are the same result and only the second one is legible.
    """
    corpus = await read_corpus(sessions)
    bands: dict[str, list[CalibrationBand]] = {}
    for candidate in detect_calibration_patterns(corpus):
        metrics = candidate.metrics
        bands.setdefault(candidate.detector, []).append(
            CalibrationBand(
                low=metrics["low"],
                high=metrics["high"],
                stated=metrics["stated"],
                interval=wilson_interval(
                    int(metrics["successes"]), int(metrics["total"])
                ),
                # A band is a finding only when the detector did not reject it
                # for falling inside the interval its sample supports.
                miscalibrated=candidate.rejected_because is None,
            )
        )
    for values in bands.values():
        values.sort(key=lambda band: band.low)
    return bands


async def counts(sessions: async_sessionmaker[AsyncSession]) -> dict[str, int]:
    async with sessions() as session:
        total = int(
            (
                await session.execute(select(func.count()).select_from(models.Pattern))
            ).scalar_one()
        )
        dismissed = int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(models.Pattern)
                    .where(models.Pattern.dismissed_at.is_not(None))
                )
            ).scalar_one()
        )
    return {"total": total, "dismissed": dismissed, "active": total - dismissed}
