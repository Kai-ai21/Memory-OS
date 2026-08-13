"""Discovery over a real corpus, and the three ways a candidate is refused.

The arithmetic is checked without a database in `tests/unit/test_pattern_arithmetic.py`.
What this file checks is that the gates are actually wired to it: that a thin
candidate does not become a row, that counter-evidence is searched for rather
than assumed absent, and that a candidate the corpus argues with is dropped
rather than emitted quietly with a low number.

The fixtures build assumption groups by hand rather than through M5.2's
embedder. What is being tested here is the detector's rule — a group whose
members mostly broke is a pattern, one whose members mostly held is not — and
routing that through a similarity threshold would make the test about the
embedder instead.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.assumptions import evaluate, list_assumptions
from memoryos.application.decisions import (
    AssumptionInput,
    DecisionDraft,
    OptionInput,
)
from memoryos.application.decisions import record as record_decision
from memoryos.application.outcomes import OutcomeDraft
from memoryos.application.outcomes import record as record_outcome
from memoryos.application.patterns import (
    calibration as patterns_calibration,
)
from memoryos.application.patterns import (
    detect_assumption_patterns,
    discover,
    dismiss,
    list_patterns,
    read_corpus,
    show,
)
from memoryos.domain.patterns import DEFAULT_MIN_SUPPORT, pattern_confidence
from memoryos.domain.values import (
    AssumptionVerdict,
    ConfidenceHorizon,
    MergeStrategy,
    OutcomeVerdict,
    PatternKind,
    PatternRelation,
    TimeProvenance,
)
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration

DECIDED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

# The anchor the calibration fixtures use, and it has to be recent.
#
# A confidence only enters a calibration population when it was recorded at the
# time of deciding — `domain/patterns.classify_confidence` — so a fixture dating
# its decisions to a fixed point in the past builds a corpus of reconstructions
# and measures nothing. That is not a test artefact to work around; it is the
# rule doing its job, and the fixture has to record decisions the way somebody
# recording a decision would. Spread over hours rather than days for the same
# reason.
RECENTLY = datetime.now(UTC) - timedelta(hours=12)


async def a_decision(
    sessions: async_sessionmaker[AsyncSession],
    *,
    question: str,
    assumption: str,
    confidence: float | None = 0.8,
    # Left unset by default so the assumption-pattern tests exercise one
    # detector at a time. An assumption carrying a confidence is also a
    # calibration observation, and a fixture that supplied one would make every
    # test below assert about two detectors at once.
    assumption_confidence: float | None = None,
    days: int = 0,
    # Calibration tests pass `RECENTLY`; everything else keeps the fixed anchor,
    # because a pattern's span and ordering are easier to read against a
    # constant and no other detector cares when the row was written.
    anchor: datetime = DECIDED_AT,
    offset: timedelta | None = None,
) -> UUID:
    return await record_decision(
        sessions,
        DecisionDraft(
            question=question,
            chosen="A Postgres table",
            confidence=confidence,
            options=(
                OptionInput(description="Celery", rejected_because="No transaction."),
            ),
            assumptions=(
                AssumptionInput(statement=assumption, confidence=assumption_confidence),
            ),
        ),
        decided_at=anchor + (timedelta(days=days) if offset is None else offset),
        decided_at_source=TimeProvenance.DECLARED,
    )


async def group_all(
    sessions: async_sessionmaker[AsyncSession], label: str = "the recurring belief"
) -> UUID:
    """Put every assumption in the corpus into one group, by hand."""
    group_id = UUID("99999999-9999-7999-8999-999999999999")
    async with sessions.begin() as session:
        session.add(
            models.AssumptionGroup(
                id=group_id, label=label, strategy=MergeStrategy.MANUAL.value
            )
        )
        rows = list(
            (await session.execute(select(models.DecisionAssumption))).scalars()
        )
        for row in rows:
            row.group_id = group_id
    return group_id


async def judge(
    sessions: async_sessionmaker[AsyncSession], verdicts: list[AssumptionVerdict]
) -> None:
    rows = await list_assumptions(sessions, limit=1000)
    for row, verdict in zip(rows, verdicts, strict=True):
        await evaluate(sessions, row.id, verdict)


async def count_patterns(sessions: async_sessionmaker[AsyncSession]) -> int:
    async with sessions() as session:
        return int(
            (
                await session.execute(select(func.count()).select_from(models.Pattern))
            ).scalar_one()
        )


# --------------------------------------------------------------------------
# 1. Support below the minimum is not emitted
# --------------------------------------------------------------------------


async def test_a_candidate_below_the_minimum_support_is_not_emitted(
    harness: Harness,
) -> None:
    """Two decisions is not a pattern, however cleanly the belief broke.

    This is the gate the milestone leads with: a system that produces confident
    behavioural claims from thin evidence is worse than one that stays silent,
    because it sounds exactly like the product working.
    """
    for index in range(2):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(harness.sessions, [AssumptionVerdict.FAILED] * 2)

    report = await discover(harness.sessions)

    assert report.emitted == 0
    assert report.below_support == 1
    assert await count_patterns(harness.sessions) == 0
    # And the candidate is still reported, so a run that emits nothing is
    # distinguishable from a run that found nothing to consider.
    assert any(
        candidate.kind is PatternKind.ASSUMPTION for candidate in report.considered
    )


async def test_three_failing_decisions_do_clear_the_bar(harness: Harness) -> None:
    """The positive control. Without it, every other test here could pass on a
    detector that never emits anything at all."""
    for index in range(3):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(harness.sessions, [AssumptionVerdict.FAILED] * 3)

    report = await discover(harness.sessions)

    assert report.emitted == 1
    (pattern,) = await list_patterns(harness.sessions)
    assert pattern.kind is PatternKind.ASSUMPTION
    assert pattern.support_count == 3
    assert pattern.contradiction_count == 0
    # Three supporting with no counter-evidence sits at exactly the bar, so the
    # confidence is 0.5 rather than certainty — see `pattern_confidence`.
    assert pattern.confidence == pytest.approx(0.5)
    # A pattern that cannot cite is never written: every supporting decision is
    # named, with a note a reader can check against the decision itself.
    assert len(pattern.supporting) == 3
    assert all(item.note for item in pattern.supporting)
    assert {item.relation for item in pattern.supporting} == {PatternRelation.SUPPORTS}


async def test_raising_the_minimum_silences_a_pattern_that_would_have_emitted(
    harness: Harness,
) -> None:
    # `--min-support` is there to be raised. The test exists so that the flag is
    # known to be wired to the gate rather than only to the report.
    for index in range(3):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(harness.sessions, [AssumptionVerdict.FAILED] * 3)

    report = await discover(harness.sessions, min_support=4)

    assert report.emitted == 0
    assert report.below_support == 1


# --------------------------------------------------------------------------
# 2. Counter-evidence lowers confidence
# --------------------------------------------------------------------------


async def test_counter_evidence_is_found_and_lowers_the_confidence(
    harness: Harness,
) -> None:
    """The members that held are found by the same pass that finds the ones that
    broke, which is what makes the search honest rather than a second query
    somebody could forget to run."""
    for index in range(5):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(
        harness.sessions,
        [AssumptionVerdict.FAILED] * 4 + [AssumptionVerdict.HELD],
    )

    await discover(harness.sessions)

    (pattern,) = await list_patterns(harness.sessions)
    assert pattern.support_count == 4
    assert pattern.contradiction_count == 1
    # agreement 4/5 = 0.8, sufficiency 4/6 = 0.667 → 0.533
    assert pattern.confidence == pytest.approx(0.533, abs=1e-3)
    # Strictly lower than the same support with nothing against it.
    assert pattern.confidence is not None
    assert pattern.confidence < pattern_confidence(4, 0)
    assert len(pattern.contradicting) == 1
    assert pattern.contradicting[0].relation is PatternRelation.CONTRADICTS
    assert "held" in (pattern.contradicting[0].note or "")


async def test_partially_held_counts_towards_the_pattern_not_against_it(
    harness: Harness,
) -> None:
    # M5.2's third verdict has to land somewhere, and a belief that half broke
    # is evidence that the belief is unreliable — the same reading the groups
    # view takes when it counts `partially` towards a failure rate.
    for index in range(3):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(
        harness.sessions,
        [
            AssumptionVerdict.FAILED,
            AssumptionVerdict.PARTIALLY,
            AssumptionVerdict.FAILED,
        ],
    )

    corpus = await read_corpus(harness.sessions)
    (candidate,) = detect_assumption_patterns(corpus)

    assert len(candidate.supporting) == 3
    assert len(candidate.contradicting) == 0


# --------------------------------------------------------------------------
# 3. More contradicting than supporting is not a pattern
# --------------------------------------------------------------------------


async def test_a_candidate_the_corpus_argues_with_is_not_emitted(
    harness: Harness,
) -> None:
    """Not a weak pattern — not a pattern.

    Three decisions where the belief broke and four where it held is a corpus
    that mostly disagrees with the claim. Emitting it with a low confidence
    would still put the sentence in front of somebody, which is exactly the
    failure this milestone is arranged against.
    """
    for index in range(7):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(
        harness.sessions,
        [AssumptionVerdict.FAILED] * 3 + [AssumptionVerdict.HELD] * 4,
    )

    report = await discover(harness.sessions)

    assert report.emitted == 0
    assert report.outweighed_by_counter_evidence == 1
    assert await count_patterns(harness.sessions) == 0


async def test_an_equal_split_is_not_a_pattern_either(harness: Harness) -> None:
    # Three against three is a coin, and the rule is `supporting > contradicting`
    # rather than `>=` for that reason.
    for index in range(6):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(
        harness.sessions,
        [AssumptionVerdict.FAILED] * 3 + [AssumptionVerdict.HELD] * 3,
    )

    report = await discover(harness.sessions)

    assert report.emitted == 0
    assert report.outweighed_by_counter_evidence == 1


# --------------------------------------------------------------------------
# 4. Calibration over a real corpus, and the interval that silences it
# --------------------------------------------------------------------------


async def test_calibration_stays_silent_when_the_sample_cannot_resolve_the_gap(
    harness: Harness,
) -> None:
    """Four decisions stated at 0.8 that all worked.

    The observed rate is 100% against a stated 0.8, which looks like
    underconfidence and is not: the Wilson interval for 4/4 runs from about 51%
    to 100%, and 0.8 is inside it. A detector that reported this would report a
    gap on every small sample, because a gap is what small samples produce.
    """
    for index in range(4):
        decision_id = await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"Belief {index}",
            confidence=0.8,
            anchor=RECENTLY,
            offset=timedelta(minutes=index),
        )
        await record_outcome(
            harness.sessions,
            decision_id,
            OutcomeDraft(description="it worked", verdict=OutcomeVerdict.WORKED),
            observed_at=RECENTLY + timedelta(hours=1, minutes=index),
            observed_at_source=TimeProvenance.DECLARED,
        )

    report = await discover(harness.sessions)

    assert report.within_noise >= 1
    assert not [
        row for row in await list_patterns(harness.sessions) if row.kind is PatternKind.OUTCOME
    ]


async def test_calibration_emits_when_the_sample_is_large_enough(
    harness: Harness,
) -> None:
    """Twelve decisions stated at 0.9, of which four worked.

    The interval for 4/12 is roughly 14%-61%, and 0.9 falls outside it. That is
    a gap this sample can resolve, so it becomes a pattern — with the eight
    failures as support and the four successes as counter-evidence, because the
    claim being tested is "your stated confidence here is too high".
    """
    for index in range(12):
        decision_id = await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"Belief {index}",
            confidence=0.9,
            anchor=RECENTLY,
            offset=timedelta(minutes=index),
        )
        await record_outcome(
            harness.sessions,
            decision_id,
            OutcomeDraft(
                description="result",
                verdict=(
                    OutcomeVerdict.WORKED if index < 4 else OutcomeVerdict.FAILED
                ),
            ),
            observed_at=RECENTLY + timedelta(hours=1, minutes=index),
            observed_at_source=TimeProvenance.DECLARED,
        )

    await discover(harness.sessions)

    calibration = [
        row
        for row in await list_patterns(harness.sessions)
        if row.detector == "decision_calibration"
    ]
    assert len(calibration) == 1
    pattern = calibration[0]
    assert "Overconfident" in pattern.statement
    assert pattern.support_count == 8
    assert pattern.contradiction_count == 4
    # agreement 8/12 = 0.667, sufficiency 1.0 → 0.667
    assert pattern.confidence == pytest.approx(0.667, abs=1e-3)


async def test_under_confidence_cites_the_successes_as_its_support(
    harness: Harness,
) -> None:
    """The mirror image of the test above, and the one that was broken.

    Twelve decisions stated at 0.30 of which eleven worked. The interval for
    11/12 runs from about 65% to 99%, and 0.30 falls outside it, so this is a
    gap the sample can resolve — in the *other* direction. The claim being
    tested is "your stated confidence here is too low", so the cases arguing for
    it are the eleven that worked and the one arguing against is the failure.

    Before the fix in this milestone, `_calibration_candidate` swapped the two
    lists' positions without relabelling the evidence rows, so this candidate
    reported one supporting decision and eleven contradicting ones and was
    refused for having no support. Half the detector could not fire, nothing
    logged it, and a run proving it looked exactly like a run finding nothing.
    """
    for index in range(12):
        decision_id = await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"Belief {index}",
            confidence=0.3,
            anchor=RECENTLY,
            offset=timedelta(minutes=index),
        )
        await record_outcome(
            harness.sessions,
            decision_id,
            OutcomeDraft(
                description="result",
                verdict=(
                    OutcomeVerdict.FAILED if index < 1 else OutcomeVerdict.WORKED
                ),
            ),
            observed_at=RECENTLY + timedelta(hours=1, minutes=index),
            observed_at_source=TimeProvenance.DECLARED,
        )

    await discover(harness.sessions)

    calibration = [
        row
        for row in await list_patterns(harness.sessions)
        if row.detector == "decision_calibration"
    ]
    assert len(calibration) == 1
    pattern = calibration[0]
    assert "Underconfident" in pattern.statement
    assert pattern.support_count == 11
    assert pattern.contradiction_count == 1
    # And every cited row agrees with the count beside it, which is the property
    # the swap broke: the evidence said "contradicts" while the statement said
    # the opposite.
    assert {item.relation for item in pattern.supporting} == {PatternRelation.SUPPORTS}
    assert len(pattern.supporting) == 11
    assert len(pattern.contradicting) == 1


async def test_a_reconstructed_confidence_never_enters_a_calibration_band(
    harness: Harness,
) -> None:
    """The same twelve decisions that emit a pattern, dated `parsed` instead.

    Phase 5's retrospective called this its largest single defect: every
    calibration result the phase produced was calibration of hindsight, and
    nothing but a paragraph of prose said so. The population is now the
    foresight rows and only those, so a corpus of reconstructions produces an
    empty table rather than a confident one.
    """
    for index in range(12):
        decision_id = await record_decision(
            harness.sessions,
            DecisionDraft(
                question=f"Question {index}",
                chosen="A Postgres table",
                confidence=0.9,
                options=(
                    OptionInput(description="Celery", rejected_because="No transaction."),
                ),
                assumptions=(AssumptionInput(statement=f"Belief {index}"),),
            ),
            decided_at=DECIDED_AT + timedelta(days=index),
            # Read out of a document rather than asserted by anybody. The date
            # was reconstructed, so the confidence beside it was too.
            decided_at_source=TimeProvenance.PARSED,
        )
        await record_outcome(
            harness.sessions,
            decision_id,
            OutcomeDraft(
                description="result",
                verdict=(OutcomeVerdict.WORKED if index < 4 else OutcomeVerdict.FAILED),
            ),
            observed_at=DECIDED_AT + timedelta(days=index + 1),
            observed_at_source=TimeProvenance.DECLARED,
        )

    report = await discover(harness.sessions)
    calibration = await patterns_calibration(harness.sessions)

    # The identical corpus with `declared` dates emits an overconfidence pattern
    # — that is the test directly above this one. With reconstructed dates there
    # is no population at all, so there is no candidate to reject.
    assert not [
        row
        for row in await list_patterns(harness.sessions)
        if row.detector == "decision_calibration"
    ]
    assert report.candidates == 0
    assert calibration.bands == {}
    # And the exclusion is counted rather than silent. An empty table with no
    # number beside it reads as "nothing recorded yet".
    assert calibration.excluded_decisions == 12


async def test_a_caller_may_declare_hindsight_but_never_assert_foresight(
    harness: Harness,
) -> None:
    """Downgrades are believed; upgrades are not.

    The one error that matters is a reconstruction entering a calibration
    population, so a writer who knows better may always make the horizon worse
    and may never argue past the derivation to make it better.
    """
    honest = await record_decision(
        harness.sessions,
        DecisionDraft(
            question="Recorded at the time, but the number came later",
            chosen="A Postgres table",
            confidence=0.8,
            options=(OptionInput(description="Celery", rejected_because="No."),),
        ),
        decided_at=datetime.now(UTC),
        decided_at_source=TimeProvenance.DECLARED,
        confidence_horizon=ConfidenceHorizon.HINDSIGHT,
    )
    optimistic = await record_decision(
        harness.sessions,
        DecisionDraft(
            question="A date nobody asserted, claimed as foresight anyway",
            chosen="A Postgres table",
            confidence=0.8,
            options=(OptionInput(description="Celery", rejected_because="No."),),
        ),
        decided_at=datetime.now(UTC),
        decided_at_source=TimeProvenance.PARSED,
        confidence_horizon=ConfidenceHorizon.FORESIGHT,
    )

    async with harness.sessions() as session:
        horizons = {
            row.id: row.confidence_horizon
            for row in (await session.execute(select(models.Decision))).scalars()
        }
    assert horizons[honest] == ConfidenceHorizon.HINDSIGHT.value
    assert horizons[optimistic] == ConfidenceHorizon.HINDSIGHT.value


# --------------------------------------------------------------------------
# Re-running, and refusing
# --------------------------------------------------------------------------


async def test_rediscovery_updates_rather_than_duplicating(harness: Harness) -> None:
    """A pattern is keyed on what it is *about*, not on its sentence.

    The sentence carries the current numbers and changes every time the corpus
    grows, so keying on it would leave a row per run saying almost the same
    thing with a different percentage in it.
    """
    for index in range(3):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(harness.sessions, [AssumptionVerdict.FAILED] * 3)

    first = await discover(harness.sessions)
    second = await discover(harness.sessions)

    assert first.emitted == 1
    assert second.emitted == 0
    assert second.updated == 1
    assert await count_patterns(harness.sessions) == 1


async def test_a_dismissed_pattern_is_not_proposed_again(harness: Harness) -> None:
    """A refusal a weekly re-run undid would not be a refusal."""
    for index in range(3):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(harness.sessions, [AssumptionVerdict.FAILED] * 3)
    await discover(harness.sessions)
    (pattern,) = await list_patterns(harness.sessions)

    await dismiss(harness.sessions, pattern.id, reason="one belief, three phrasings")

    report = await discover(harness.sessions)
    assert report.skipped_dismissed == 1
    assert report.emitted == 0
    # Still there, still dismissed, with its reason — not deleted.
    assert await list_patterns(harness.sessions) == []
    kept = await show(harness.sessions, pattern.id)
    assert kept.dismissed_at is not None
    assert kept.dismissed_reason == "one belief, three phrasings"


async def test_a_dismissal_without_a_reason_is_refused(harness: Harness) -> None:
    for index in range(3):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(harness.sessions, [AssumptionVerdict.FAILED] * 3)
    await discover(harness.sessions)
    (pattern,) = await list_patterns(harness.sessions)

    with pytest.raises(ValueError, match="reason"):
        await dismiss(harness.sessions, pattern.id, reason="   ")


async def test_an_empty_corpus_produces_no_patterns_and_says_why(
    harness: Harness,
) -> None:
    # The default state of every new installation, and it must not look like a
    # failure. Three detectors report having nothing to propose.
    report = await discover(harness.sessions)

    assert report.candidates == 0
    assert report.emitted == 0
    assert {name for name, _ in report.silent} == {
        "assumption_group",
        "slow_resolution",
        "reversal_rate",
    }
    assert all(reason for _, reason in report.silent)


async def test_the_default_minimum_support_is_used_when_none_is_given(
    harness: Harness,
) -> None:
    for index in range(DEFAULT_MIN_SUPPORT):
        await a_decision(
            harness.sessions,
            question=f"Question {index}",
            assumption=f"The deploy is straightforward, take {index}",
            days=index,
        )
    await group_all(harness.sessions)
    await judge(harness.sessions, [AssumptionVerdict.FAILED] * DEFAULT_MIN_SUPPORT)

    report = await discover(harness.sessions)

    assert report.emitted == 1
