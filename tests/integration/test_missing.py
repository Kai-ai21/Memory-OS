"""The four properties gap analysis rests on, and every one is about silence.

This is the only capability in the system that names something absent, and
absence has infinite candidates — you are always missing something. So what these
check is that it refuses:

* a topic with no history produces no gaps **(the acceptance criterion)**,
* a gap below minimum support is not emitted,
* every emitted gap carries at least two citations,
* contradicting evidence suppresses a gap.

A real database, because the detectors are queries and a test against fakes would
assert that the Python agrees with itself.
"""

from datetime import UTC, datetime, timedelta

import pytest

from memoryos.adapters.db import models
from memoryos.application import missing
from memoryos.application.decisions import (
    AssumptionInput,
    DecisionDraft,
    OptionInput,
    record,
)
from memoryos.domain.missing import MIN_SUPPORT, GapKind
from memoryos.domain.values import AssumptionVerdict, TimeProvenance
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
LONG_AGO = NOW - timedelta(days=120)


async def a_decision(
    harness: Harness,
    question: str,
    *,
    assumptions: tuple[str, ...] = (),
    decided_at: datetime = NOW,
) -> models.Decision:
    decision_id = await record(
        harness.sessions,
        DecisionDraft(
            question=question,
            chosen="the first option",
            options=(OptionInput(description="the second option"),),
            assumptions=tuple(
                AssumptionInput(statement=statement) for statement in assumptions
            ),
        ),
        decided_at=decided_at,
        decided_at_source=TimeProvenance.DECLARED,
    )
    async with harness.sessions() as session:
        found = await session.get(models.Decision, decision_id)
        assert found is not None
        return found


async def evaluate_all(
    harness: Harness, decision_id: object, verdict: AssumptionVerdict
) -> None:
    async with harness.sessions() as session, session.begin():
        rows = list(
            (
                await session.execute(
                    models.DecisionAssumption.__table__.select().where(
                        models.DecisionAssumption.decision_id == decision_id
                    )
                )
            ).all()
        )
        for row in rows:
            assumption = await session.get(models.DecisionAssumption, row.id)
            assert assumption is not None
            assumption.held = verdict.value
            assumption.evaluated_at = NOW


# --------------------------------------------------------------------------
# 1. A topic with no history produces no gaps — the acceptance criterion
# --------------------------------------------------------------------------


async def test_a_topic_with_no_history_produces_no_gaps(harness: Harness) -> None:
    """**The acceptance criterion, and the one that would be easiest to lose.**

    Absence has infinite candidates: a system asked "what am I missing?" can
    produce output forever without ever being wrong in a way anybody can check.
    The only thing standing between this and a horoscope is that it declines when
    nothing in the corpus bears on the question — and the decline has to be a
    sentence, because a command that prints nothing is indistinguishable from one
    that is broken.
    """
    await a_decision(
        harness,
        "Which vector store backs retrieval?",
        assumptions=("The index fits in memory.",),
    )

    report = await missing.find_missing(
        harness.sessions, about="kubernetes ingress rate limiting", now=NOW
    )

    assert report.gaps == []
    assert report.silence.considered == 0
    # It says why, in words, rather than printing an empty list.
    assert "no history that bears on this" in report.silence.render()


async def test_an_empty_corpus_says_so_rather_than_guessing(harness: Harness) -> None:
    """With nothing recorded at all there is nothing to compare against, and the
    correct output is the same refusal rather than generic advice."""
    report = await missing.find_missing(harness.sessions, now=NOW)

    assert report.gaps == []
    assert "cannot say what is missing" in report.silence.render()


# --------------------------------------------------------------------------
# 2. Below minimum support, nothing is emitted
# --------------------------------------------------------------------------


async def test_one_past_instance_is_not_enough_to_name_an_absence(
    harness: Harness,
) -> None:
    """**One occasion is not a habit.**

    A single past decision that recorded a belief this one omits is a
    coincidence, and naming it would be the system inventing a norm out of one
    observation. Two is the floor, and below it the report says how far the best
    candidate got — because "insufficient history" with no number is an excuse
    and "the closest reached 1 of 2" is a measurement.
    """
    await a_decision(
        harness,
        "Which queue runs background work?",
        assumptions=("The queue survives a restart.",),
    )
    target = await a_decision(harness, "Which queue runs scheduled work?")

    report = await missing.find_missing(harness.sessions, about=target.question, now=NOW)

    assert report.gaps == []
    assert report.silence.considered >= 1
    assert report.silence.best_support == 1
    assert f"1 of {MIN_SUPPORT}" in report.silence.render()


async def test_two_past_instances_are_enough_and_the_gap_names_them(
    harness: Harness,
) -> None:
    """The other side of the same rule, so the test above is a bar rather than a
    detector that never fires."""
    for question in (
        "Which queue runs background work?",
        "Which queue runs scheduled work?",
    ):
        await a_decision(
            harness, question, assumptions=("The queue survives a restart.",)
        )
    target = await a_decision(harness, "Which queue runs deferred work?")

    report = await missing.find_missing(harness.sessions, about=target.question, now=NOW)

    gaps = [gap for gap in report.gaps if gap.kind is GapKind.UNSTATED_ASSUMPTION]
    assert len(gaps) == 1
    assert gaps[0].supporting == 2
    assert "survives" in gaps[0].statement
    # And it names the two occasions rather than reporting a statistic.
    assert {item.label for item in gaps[0].evidence} == {
        "Which queue runs background work?",
        "Which queue runs scheduled work?",
    }


# --------------------------------------------------------------------------
# 3. Every emitted gap carries at least two citations
# --------------------------------------------------------------------------


async def test_every_emitted_gap_cites_at_least_two_things(harness: Harness) -> None:
    """**A gap that cannot cite is not emitted.**

    M5.4's rule, and it binds harder here. A reflection is a claim about
    decisions a reader can go and read; a gap is a claim about something that is
    *not there*, which has no referent at all except the history that makes it
    sayable. Without citations it is unfalsifiable by construction — and the
    support count is exactly the part a reader cannot check for themselves.
    """
    for question in (
        "Which queue runs background work?",
        "Which queue runs scheduled work?",
        "Which queue runs retries?",
    ):
        await a_decision(
            harness, question, assumptions=("The queue survives a restart.",)
        )
    target = await a_decision(harness, "Which queue runs deferred work?")

    report = await missing.find_missing(harness.sessions, about=target.question, now=NOW)

    assert report.gaps
    for gap in report.gaps:
        assert len(gap.evidence) >= MIN_SUPPORT, gap.statement
        # The count and the evidence agree; a gap claiming three occasions and
        # showing one has counted something it cannot produce.
        assert len(gap.evidence) >= gap.supporting or gap.supporting <= len(gap.evidence)
        assert all(item.label for item in gap.evidence)


async def test_a_gap_whose_evidence_went_missing_is_not_emitted() -> None:
    """The rule enforced on the object rather than trusted from the detector.

    A `Gap` is `sayable` only if it both clears the bar *and* can show the
    instances it counted, so a detector that computed a support count without
    collecting the rows behind it produces nothing rather than an uncitable
    sentence.
    """
    uncitable = missing.Gap(
        kind=GapKind.UNSTATED_ASSUMPTION,
        statement="Two decisions recorded something this one did not.",
        subject="x",
        supporting=2,
        contradicting=0,
        evidence=(),
    )

    assert not uncitable.sayable


# --------------------------------------------------------------------------
# 4. Contradicting evidence suppresses a gap
# --------------------------------------------------------------------------


async def test_contradicting_evidence_suppresses_a_gap(harness: Harness) -> None:
    """**An assumption that broke both times it was made is not one to add.**

    Two decisions recorded the belief, so the support is there — and both were
    wrong about it. "You forgot to write this down" is the wrong sentence for a
    belief that has never held, and the confidence falls below the bar rather
    than the gap being emitted with a caveat nobody reads.
    """
    for question in (
        "Which queue runs background work?",
        "Which queue runs scheduled work?",
    ):
        decision = await a_decision(
            harness, question, assumptions=("The queue survives a restart.",)
        )
        await evaluate_all(harness, decision.id, AssumptionVerdict.FAILED)
    target = await a_decision(harness, "Which queue runs deferred work?")

    report = await missing.find_missing(harness.sessions, about=target.question, now=NOW)

    assert report.gaps == []
    assert report.silence.outweighed == 1
    assert "arguing against" in report.silence.render()


async def test_a_belief_that_held_is_not_suppressed(harness: Harness) -> None:
    """The control for the test above: the same shape with the verdicts the other
    way round does produce the gap."""
    for question in (
        "Which queue runs background work?",
        "Which queue runs scheduled work?",
    ):
        decision = await a_decision(
            harness, question, assumptions=("The queue survives a restart.",)
        )
        await evaluate_all(harness, decision.id, AssumptionVerdict.HELD)
    target = await a_decision(harness, "Which queue runs deferred work?")

    report = await missing.find_missing(harness.sessions, about=target.question, now=NOW)

    assert [gap.kind for gap in report.gaps] == [GapKind.UNSTATED_ASSUMPTION]


# --------------------------------------------------------------------------
# Unevaluated assumptions, the one whose evidence is a missing row
# --------------------------------------------------------------------------


async def test_old_unchecked_assumptions_are_a_gap_and_recent_ones_are_not(
    harness: Harness,
) -> None:
    """A belief recorded yesterday is not overdue. One from four months ago that
    nothing has tested is a decision still resting on an unexamined premise."""
    old = await a_decision(
        harness,
        "Which storage engine backs the index?",
        assumptions=("The index fits in memory.", "Rebuilds stay under an hour."),
        decided_at=LONG_AGO,
    )
    await a_decision(
        harness,
        "Which cache backs the reader?",
        assumptions=("The cache stays warm.", "Evictions are rare."),
        decided_at=NOW,
    )

    report = await missing.find_missing(harness.sessions, now=NOW)

    stale = [gap for gap in report.gaps if gap.kind is GapKind.UNEVALUATED_ASSUMPTION]
    assert len(stale) == 1
    assert stale[0].subject == str(old.id)
    assert stale[0].supporting == 2
    assert "120 days" in stale[0].statement
    assert len(stale[0].evidence) == 2


async def test_a_decision_whose_assumptions_were_checked_is_not_a_gap(
    harness: Harness,
) -> None:
    """The detector reports the absence of an evaluation, so an evaluation
    removes it — and a decision with one checked and one not has the checking
    counted against it."""
    old = await a_decision(
        harness,
        "Which storage engine backs the index?",
        assumptions=("The index fits in memory.", "Rebuilds stay under an hour."),
        decided_at=LONG_AGO,
    )
    await evaluate_all(harness, old.id, AssumptionVerdict.HELD)

    report = await missing.find_missing(harness.sessions, now=NOW)

    assert [gap for gap in report.gaps if gap.kind is GapKind.UNEVALUATED_ASSUMPTION] == []


async def test_the_kind_filter_narrows_without_changing_the_bar(
    harness: Harness,
) -> None:
    """`--kind` selects among what would have been emitted anyway. It is a filter
    on the output, never a lowering of the threshold."""
    await a_decision(
        harness,
        "Which storage engine backs the index?",
        assumptions=("The index fits in memory.", "Rebuilds stay under an hour."),
        decided_at=LONG_AGO,
    )

    everything = await missing.find_missing(harness.sessions, now=NOW)
    filtered = await missing.find_missing(
        harness.sessions, kind=GapKind.ORPHANED_WORK, now=NOW
    )

    assert any(gap.kind is GapKind.UNEVALUATED_ASSUMPTION for gap in everything.gaps)
    assert filtered.gaps == []
