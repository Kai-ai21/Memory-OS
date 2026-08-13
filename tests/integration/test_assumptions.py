"""The four claims M5.2 makes, each about a way the numbers could quietly lie.

**`partially` is counted separately.** A binary forces the interesting cases
into the wrong box, and a system that resolved them by rounding would report a
hold rate that flattered every vague belief somebody wrote down.

**Grouping puts the same belief together and leaves different ones apart.** This
is what makes M5.3 possible and what makes it dangerous: a false group invents a
recurrence — four members, one hold rate, a confident finding about how somebody
estimates — out of assumptions that have nothing to do with each other.

**An assumption on a `too_early` decision is still evaluable.** Some beliefs are
checkable long before anybody can say whether the decision they supported was
right, and a system that gated evaluation on the outcome would lose exactly the
ones that generalise.

**Unevaluated assumptions are in neither half of any rate.** The same rule
`too_early` follows for success rates: a percentage over whatever happened to
get attention is not a measurement.
"""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.assumption_groups import (
    AUTO_THRESHOLD,
    REVIEW_FLOOR,
    GroupAssumptions,
    accept,
    list_candidates,
    reject,
)
from memoryos.application.assumptions import (
    EvidenceInput,
    UnknownAssumption,
    evaluate,
    list_assumptions,
    stats,
)
from memoryos.application.decisions import (
    AssumptionInput,
    DecisionDraft,
    OptionInput,
)
from memoryos.application.decisions import record as record_decision
from memoryos.application.outcomes import OutcomeDraft
from memoryos.application.outcomes import record as record_outcome
from memoryos.domain.values import (
    EVALUATED_VERDICTS,
    AssumptionVerdict,
    MergeStatus,
    MergeStrategy,
    OutcomeVerdict,
    TimeProvenance,
)
from tests.integration.conftest import Harness
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

DECIDED_AT = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)


async def a_decision(
    sessions: async_sessionmaker[AsyncSession],
    *,
    question: str = "What runs background work?",
    assumptions: tuple[str, ...] = ("Throughput stays in the low thousands",),
) -> UUID:
    return await record_decision(
        sessions,
        DecisionDraft(
            question=question,
            chosen="A Postgres table",
            options=(
                OptionInput(
                    description="Celery with Redis",
                    rejected_because="Cannot share the transaction.",
                ),
            ),
            assumptions=tuple(
                AssumptionInput(statement=statement) for statement in assumptions
            ),
        ),
        decided_at=DECIDED_AT,
        decided_at_source=TimeProvenance.DECLARED,
    )


async def assumption_ids(
    sessions: async_sessionmaker[AsyncSession], decision_id: UUID
) -> list[UUID]:
    rows = await list_assumptions(sessions, decision_id=decision_id)
    return [row.id for row in rows]


# --------------------------------------------------------------------------
# 1. `partially` is a verdict of its own
# --------------------------------------------------------------------------


async def test_partially_is_accepted_and_counted_apart_from_held_and_failed(
    harness: Harness,
) -> None:
    """The verdict that exists because a binary produces noise.

    "The free tier's rate limits are workable" was true for months of ordinary
    use and false the first time a corpus-wide extraction ran. Recording that as
    `failed` loses the half that was right; as `held`, the milestone it blocked.
    """
    decision_id = await a_decision(
        harness.sessions,
        assumptions=("A held one", "A failed one", "A partial one"),
    )
    ids = await assumption_ids(harness.sessions, decision_id)
    for assumption_id, verdict in zip(
        ids,
        [AssumptionVerdict.HELD, AssumptionVerdict.FAILED, AssumptionVerdict.PARTIALLY],
        strict=True,
    ):
        await evaluate(harness.sessions, assumption_id, verdict)

    report = await stats(harness.sessions)

    assert report.held == 1
    assert report.failed == 1
    assert report.partially == 1
    assert report.evaluated == 3
    # In the denominator and not the numerator, which is a judgement rather than
    # an obvious truth — so it is reported on its own line as well.
    assert report.hold_rate == pytest.approx(1 / 3)
    assert AssumptionVerdict.PARTIALLY in EVALUATED_VERDICTS


async def test_a_partial_verdict_counts_as_a_failure_in_the_group_view(
    harness: Harness,
) -> None:
    """`hold_rate` and `failure_rate` are deliberately not complements.

    A belief that half held is a belief that half broke. The view whose job is
    to surface recurring trouble should show it; the number quoted as "how often
    this held" should not credit it.
    """
    decision_id = await a_decision(
        harness.sessions, assumptions=("A held one", "A partial one")
    )
    ids = await assumption_ids(harness.sessions, decision_id)
    async with harness.sessions.begin() as session:
        group_id = UUID("33333333-3333-7333-8333-333333333333")
        session.add(
            models.AssumptionGroup(
                id=group_id,
                label="A held one",
                strategy=MergeStrategy.MANUAL.value,
            )
        )
        for assumption_id in ids:
            row = await session.get(models.DecisionAssumption, assumption_id)
            assert row is not None
            row.group_id = group_id

    await evaluate(harness.sessions, ids[0], AssumptionVerdict.HELD)
    await evaluate(harness.sessions, ids[1], AssumptionVerdict.PARTIALLY)

    (group,) = (await stats(harness.sessions)).recurring
    assert group.hold_rate == pytest.approx(0.5)
    assert group.failure_rate == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 2. Grouping joins the same belief and separates different ones
# --------------------------------------------------------------------------


async def test_grouping_joins_identical_statements_and_leaves_distinct_ones_apart(
    harness: Harness,
) -> None:
    """The property M5.3 rests on, and the one it is endangered by.

    Uses the deterministic fake embedder rather than the real model, which is
    the same choice every other test in this suite makes: what is being checked
    is that identical text groups and unrelated text does not, and a fake whose
    vectors are a pure function of the string establishes that without
    downloading 90MB. Whether the *real* model separates these particular
    sentences is a question about a model, and `assumptions group` measures it
    against the corpus.
    """
    same = "The deployment will take about two days of work."
    first = await a_decision(
        harness.sessions, question="Which deploy path?", assumptions=(same,)
    )
    second = await a_decision(
        harness.sessions, question="Which migration path?", assumptions=(same,)
    )
    third = await a_decision(
        harness.sessions,
        question="Which database?",
        assumptions=("Postgres will hold the whole corpus without sharding.",),
    )

    await GroupAssumptions(harness.sessions, FakeEmbedder())()

    rows = await list_assumptions(harness.sessions)
    by_decision = {row.decision_id: row for row in rows}
    # The two identical beliefs share a group.
    assert by_decision[first].group_id is not None
    assert by_decision[first].group_id == by_decision[second].group_id
    # The unrelated one does not join them.
    assert by_decision[third].group_id != by_decision[first].group_id


async def test_a_pair_between_the_floor_and_the_threshold_waits_for_a_person(
    harness: Harness,
) -> None:
    """The asymmetry M3.2 states, applied to a worse failure.

    A missed group leaves two beliefs looking unrelated — visible, and fixed by
    accepting the pending candidate. A false group invents a recurrence, and
    M5.3 reads exactly this table. So the band between the review floor and the
    auto threshold goes to a queue rather than into a group.
    """
    assert REVIEW_FLOOR < AUTO_THRESHOLD
    first = await a_decision(
        harness.sessions,
        question="Which deploy path?",
        assumptions=("The deployment will take about two days of work.",),
    )
    second = await a_decision(
        harness.sessions,
        question="Which migration path?",
        assumptions=("The deployment will take about two days of effort.",),
    )

    # A threshold nothing can clear, with a floor everything clears: the pair is
    # forced into the review band whatever the fake embedder happens to score.
    # The floor is below zero because the fake's vectors are hashes — two
    # unrelated strings are as likely to point away from each other as towards,
    # and a floor of 0.0 would silently drop the pair this test is about.
    grouper = GroupAssumptions(
        harness.sessions, FakeEmbedder(), threshold=1.01, review_floor=-1.0
    )
    report = await grouper()

    assert report.auto_grouped == 0
    assert report.queued >= 1
    rows = await list_assumptions(harness.sessions)
    assert all(row.group_id is None for row in rows)

    pending = await list_candidates(harness.sessions)
    assert pending
    # Both statements and both decisions, because what the reviewer is judging
    # is whether two moments expressed the same belief.
    assert {pending[0].left_question, pending[0].right_question} == {
        "Which deploy path?",
        "Which migration path?",
    }

    group_id = await accept(harness.sessions, pending[0].id)

    regrouped = await list_assumptions(harness.sessions)
    assert {row.group_id for row in regrouped} == {group_id}
    assert not await list_candidates(harness.sessions, status=MergeStatus.PENDING)
    assert first != second


async def test_a_rejected_pair_is_kept_and_stays_ungrouped(harness: Harness) -> None:
    await a_decision(
        harness.sessions,
        question="Which deploy path?",
        assumptions=("The deployment will take about two days of work.",),
    )
    await a_decision(
        harness.sessions,
        question="Which migration path?",
        assumptions=("The deployment will take about two days of effort.",),
    )
    grouper = GroupAssumptions(
        harness.sessions, FakeEmbedder(), threshold=1.01, review_floor=-1.0
    )
    await grouper()
    pending = await list_candidates(harness.sessions)

    await reject(harness.sessions, pending[0].id)

    assert not await list_candidates(harness.sessions, status=MergeStatus.PENDING)
    # The row stays: it is what stops the pair being re-proposed on the next
    # run, and the count of rejections is the only measurement of how often the
    # embedder proposes two beliefs that are not the same one.
    kept = await list_candidates(harness.sessions, status=MergeStatus.REVERTED)
    assert [row.id for row in kept] == [pending[0].id]
    assert all(row.group_id is None for row in await list_assumptions(harness.sessions))


# --------------------------------------------------------------------------
# 3. An assumption outlives its decision's outcome being unknown
# --------------------------------------------------------------------------


async def test_an_assumption_on_a_too_early_decision_can_still_be_evaluated(
    harness: Harness,
) -> None:
    """Some beliefs are checkable long before the decision they served is.

    "The free tier's rate limits are workable" can be settled the first time a
    quota is hit, years before anybody can say whether choosing that provider
    was right. A system that gated evaluation on the outcome would lose exactly
    the assumptions that generalise — which is the whole reason this milestone
    exists.
    """
    decision_id = await a_decision(
        harness.sessions, assumptions=("The free tier's limits are workable",)
    )
    await record_outcome(
        harness.sessions,
        decision_id,
        OutcomeDraft(
            description="no result yet", verdict=OutcomeVerdict.TOO_EARLY
        ),
        observed_at=datetime.now(UTC),
        observed_at_source=TimeProvenance.DECLARED,
    )
    (assumption_id,) = await assumption_ids(harness.sessions, decision_id)

    await evaluate(
        harness.sessions,
        assumption_id,
        AssumptionVerdict.FAILED,
        note="the daily cap was exhausted mid-milestone",
    )

    (row,) = await list_assumptions(harness.sessions, decision_id=decision_id)
    assert row.held is AssumptionVerdict.FAILED
    assert row.evaluated_at is not None
    assert row.note == "the daily cap was exhausted mid-milestone"
    # The outcome is carried for context and is not a gate: the decision is
    # still `too_early` and the assumption is still evaluated.
    assert row.outcome_verdict is OutcomeVerdict.TOO_EARLY

    report = await stats(harness.sessions)
    assert report.evaluated == 1
    assert report.failed == 1


async def test_re_evaluating_replaces_the_verdict_and_its_reasoning(
    harness: Harness,
) -> None:
    """One verdict per assumption, with its note and evidence.

    The rule `query_judgements` applies to a search result: two contradictory
    opinions about whether the same belief held is not richer data. The note and
    the evidence go with it, because they are the argument *for* that verdict —
    keeping the old ones would leave a verdict explained by an argument for a
    different one.
    """
    decision_id = await a_decision(harness.sessions)
    (assumption_id,) = await assumption_ids(harness.sessions, decision_id)

    await evaluate(
        harness.sessions,
        assumption_id,
        AssumptionVerdict.HELD,
        note="nothing had gone wrong yet",
        evidence=[EvidenceInput(source_name="corpus", external_key="queue.md")],
    )
    await evaluate(
        harness.sessions,
        assumption_id,
        AssumptionVerdict.FAILED,
        note="then the queue backed up",
        evidence=[EvidenceInput(source_name="corpus", external_key="bread.txt")],
    )

    (row,) = await list_assumptions(harness.sessions, decision_id=decision_id)
    assert row.held is AssumptionVerdict.FAILED
    assert row.note == "then the queue backed up"
    assert [item.external_key for item in row.evidence] == ["bread.txt"]

    async with harness.sessions() as session:
        total = (
            await session.execute(
                select(func.count()).select_from(models.AssumptionEvidence)
            )
        ).scalar_one()
    assert total == 1


async def test_evaluating_an_assumption_that_does_not_exist_is_refused(
    harness: Harness,
) -> None:
    with pytest.raises(UnknownAssumption):
        await evaluate(
            harness.sessions,
            UUID("44444444-4444-7444-8444-444444444444"),
            AssumptionVerdict.HELD,
        )


# --------------------------------------------------------------------------
# 4. Unevaluated assumptions are in neither half of any rate
# --------------------------------------------------------------------------


async def test_stats_exclude_unevaluated_assumptions_from_the_hold_rate(
    harness: Harness,
) -> None:
    """The same rule `too_early` follows for a success rate.

    A hold rate over a corpus where most assumptions have never been looked at
    would be a percentage of whatever happened to get attention. Counting them
    as failures would punish writing assumptions down at all.
    """
    decision_id = await a_decision(
        harness.sessions, assumptions=("Checked one", "Never checked", "Also never")
    )
    ids = await assumption_ids(harness.sessions, decision_id)
    await evaluate(harness.sessions, ids[0], AssumptionVerdict.HELD)

    report = await stats(harness.sessions)

    assert report.total == 3
    assert report.evaluated == 1
    assert report.unevaluated == 2
    # One held over one evaluated. The two nobody has looked at are in neither
    # half — not 1/3, and certainly not two failures.
    assert report.hold_rate == pytest.approx(1.0)


async def test_a_corpus_with_nothing_evaluated_has_no_hold_rate(
    harness: Harness,
) -> None:
    """None, not 0.0 — zero would read as "none of them held"."""
    await a_decision(harness.sessions)

    report = await stats(harness.sessions)

    assert report.total == 1
    assert report.evaluated == 0
    assert report.hold_rate is None


async def test_a_group_nobody_has_evaluated_has_no_hold_rate(
    harness: Harness,
) -> None:
    # And it sorts last rather than as a zero, so an unexamined group cannot
    # top a view whose whole job is surfacing recurring trouble.
    first = await a_decision(
        harness.sessions,
        question="Which deploy path?",
        assumptions=("The deployment will take about two days of work.",),
    )
    await a_decision(
        harness.sessions,
        question="Which migration path?",
        assumptions=("The deployment will take about two days of work.",),
    )
    await GroupAssumptions(harness.sessions, FakeEmbedder())()

    report = await stats(harness.sessions)

    (group,) = report.recurring
    assert group.members == 2
    assert group.evaluated == 0
    assert group.hold_rate is None
    assert group.failure_rate is None
    assert first is not None


async def test_a_group_of_one_is_not_reported_as_recurring(
    harness: Harness,
) -> None:
    """Recurrence starts at two.

    A group of one is an assumption nothing else in the corpus resembles, which
    is a fact about the corpus rather than a finding about anybody's judgement.
    """
    await a_decision(harness.sessions, assumptions=("A belief held exactly once",))
    await GroupAssumptions(harness.sessions, FakeEmbedder())()

    report = await stats(harness.sessions)
    assert report.recurring == []
