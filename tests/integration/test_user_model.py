"""The four properties the user model rests on.

Every one of them is about the system refusing to say something, or refusing to
forget that it said something.

* below the bar produces a stated gap, not a quiet low-confidence claim,
* `superseded_by` keeps the old version retrievable,
* a dismissed facet is never re-derived,
* contradicting evidence lowers confidence.

The database is real because the properties are: `superseded_by` is a
self-referential foreign key, the partial unique index is what makes
re-derivation idempotent, and a test against fakes would assert that the Python
is consistent with itself.
"""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from memoryos.adapters.db import models
from memoryos.application import user_model
from memoryos.application.decisions import (
    AssumptionInput,
    DecisionDraft,
    OptionInput,
    record,
)
from memoryos.domain.ids import new_id
from memoryos.domain.user_model import MIN_SUPPORT, facet_confidence
from memoryos.domain.values import (
    AssumptionVerdict,
    Dimension,
    FacetOrigin,
    TimeProvenance,
)
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration

DECIDED_AT = datetime(2026, 5, 2, 9, 0, tzinfo=UTC)


async def a_decision(harness: Harness, question: str, assumption: str) -> UUID:
    """One decision carrying one assumption, which is the unit a group counts."""
    return await record(
        harness.sessions,
        DecisionDraft(
            question=question,
            chosen="something",
            options=(OptionInput(description="something else"),),
            assumptions=(AssumptionInput(statement=assumption),),
        ),
        decided_at=DECIDED_AT,
        decided_at_source=TimeProvenance.DECLARED,
    )


async def group_assumptions(
    harness: Harness, label: str, verdicts: list[AssumptionVerdict]
) -> UUID:
    """A user-authored group of assumptions, one per decision, each evaluated.

    One assumption per decision on purpose: the deriver counts *distinct
    decisions*, and a group built from four assumptions of one decision has to
    read as one observation rather than four.
    """
    group_id = new_id()
    async with harness.sessions() as session, session.begin():
        session.add(
            models.AssumptionGroup(
                id=group_id, label=label, strategy="manual", created_at=DECIDED_AT
            )
        )
    for index, verdict in enumerate(verdicts):
        decision_id = await a_decision(harness, f"Question {label} {index}?", label)
        async with harness.sessions() as session, session.begin():
            row = (
                await session.execute(
                    models.DecisionAssumption.__table__.select().where(
                        models.DecisionAssumption.decision_id == decision_id
                    )
                )
            ).first()
            assert row is not None
            assumption = await session.get(models.DecisionAssumption, row.id)
            assert assumption is not None
            assumption.group_id = group_id
            assumption.held = verdict.value
            assumption.evaluated_at = DECIDED_AT
    return group_id


# --------------------------------------------------------------------------
# 1. Below the bar is a stated gap, not a quiet claim
# --------------------------------------------------------------------------


async def test_below_minimum_support_yields_a_stated_gap_not_a_weak_facet(
    harness: Harness,
) -> None:
    """**A 0.2-confidence sentence and no sentence are different objects.**

    The first is something a person will read and remember, and the number beside
    it will not survive the reading. So a dimension whose best candidate falls
    short records the shortfall — with the number it reached, because
    "insufficient evidence" alone is an excuse and "the best reached two" is a
    measurement.
    """
    await group_assumptions(
        harness,
        "the index will stay small enough to rebuild",
        [AssumptionVerdict.HELD, AssumptionVerdict.HELD],
    )

    report = await user_model.derive(harness.sessions)

    assert report.written == 0
    assert report.below_bar == 1
    strengths = next(
        item for item in report.assessments if item.dimension is Dimension.STRENGTHS
    )
    assert not strengths.has_evidence
    assert strengths.best_support == 2
    assert "insufficient evidence" in strengths.render()
    assert str(MIN_SUPPORT) in strengths.render()

    # And nothing was written at a low confidence instead.
    model = await user_model.view(harness.sessions)
    assert model.facets == {}


async def test_the_bar_being_cleared_writes_the_facet_with_its_evidence(
    harness: Harness,
) -> None:
    """The other side of the same rule, so the first test is a bar rather than a
    deriver that never fires."""
    await group_assumptions(
        harness,
        "chunk offsets index into the memory text",
        [AssumptionVerdict.HELD] * 3,
    )

    report = await user_model.derive(harness.sessions)

    assert report.written == 1
    model = await user_model.view(harness.sessions)
    facets = model.facets[Dimension.STRENGTHS.value]
    assert len(facets) == 1
    assert facets[0].support_count == 3
    assert facets[0].origin == FacetOrigin.DERIVED.value
    # Three decisions, cited. A facet that cannot cite is never written.
    assert len(facets[0].evidence) == 3


# --------------------------------------------------------------------------
# 2. superseded_by preserves history
# --------------------------------------------------------------------------


async def test_superseding_keeps_the_old_facet_retrievable(harness: Harness) -> None:
    """**How the model changed is part of the model.**

    A person's model that could only show its current state cannot answer the
    question that makes it worth having, so a revision inserts a new row and
    points the old one at it rather than updating in place.
    """
    label = "the graph will be dense enough to be worth traversing"
    await group_assumptions(harness, label, [AssumptionVerdict.HELD] * 3)
    await user_model.derive(harness.sessions)
    first = (await user_model.view(harness.sessions)).facets[
        Dimension.STRENGTHS.value
    ][0]

    # A fourth decision evaluates the same belief, so the statement's counts move.
    decision_id = await a_decision(harness, "One more?", label)
    async with harness.sessions() as session, session.begin():
        row = (
            await session.execute(
                models.DecisionAssumption.__table__.select().where(
                    models.DecisionAssumption.decision_id == decision_id
                )
            )
        ).first()
        assert row is not None
        assumption = await session.get(models.DecisionAssumption, row.id)
        assert assumption is not None
        assumption.group_id = UUID(first.subject_key or "")
        assumption.held = AssumptionVerdict.HELD.value
        assumption.evaluated_at = DECIDED_AT

    report = await user_model.derive(harness.sessions)
    assert report.superseded == 1

    # The old row is still there, still readable, and points at its replacement.
    chain = await user_model.history(harness.sessions, first.id)
    assert chain[0].id == first.id
    assert len(chain) == 2
    assert chain[0].superseded_by == chain[1].id
    assert chain[0].support_count == 3
    assert chain[1].support_count == 4

    # And the same chain is reachable from the new end, because a caller holding
    # the current row is asking the same question as one holding the old one.
    assert [item.id for item in await user_model.history(harness.sessions, chain[1].id)] == [
        item.id for item in chain
    ]

    # The superseded row is not in the live view.
    live = (await user_model.view(harness.sessions)).facets[Dimension.STRENGTHS.value]
    assert [item.id for item in live] == [chain[1].id]


async def test_rederiving_an_unchanged_facet_writes_nothing(harness: Harness) -> None:
    """Running `derive` twice is not two rows and not two timestamps. Without
    this a nightly derivation would leave a row per night saying the same thing,
    and the history that `superseded_by` exists for would be noise."""
    await group_assumptions(
        harness, "a repeated belief", [AssumptionVerdict.HELD] * 3
    )
    await user_model.derive(harness.sessions)

    second = await user_model.derive(harness.sessions)

    assert second.written == 0
    assert second.unchanged == 1
    assert second.superseded == 0


# --------------------------------------------------------------------------
# 3. A dismissed facet is not re-derived
# --------------------------------------------------------------------------


async def test_a_dismissed_facet_is_not_proposed_again(harness: Harness) -> None:
    """**A system that re-proposed a rejected claim would be arguing with its
    user.**

    Rejected by *subject* rather than by id, which is the part that matters: a
    person who rejected "this belief keeps holding" rejected the claim, and
    re-proposing the same sentence under a fresh id on the next run would honour
    the letter of the dismissal and none of it.
    """
    await group_assumptions(
        harness, "a belief worth rejecting", [AssumptionVerdict.HELD] * 3
    )
    await user_model.derive(harness.sessions)
    facet = (await user_model.view(harness.sessions)).facets[
        Dimension.STRENGTHS.value
    ][0]

    await user_model.dismiss(harness.sessions, facet.id, reason="that is not a strength")

    report = await user_model.derive(harness.sessions)

    assert report.written == 0
    assert report.skipped_dismissed == 1
    model = await user_model.view(harness.sessions)
    assert Dimension.STRENGTHS.value not in model.facets
    # Visible rather than filtered away: a rejected claim that vanished would look
    # like one nobody ever made, and the rejection is the more interesting fact.
    assert [item.id for item in model.dismissed] == [facet.id]
    assert model.dismissed[0].dismissed_reason == "that is not a strength"


async def test_a_dismissal_without_a_reason_is_refused(harness: Harness) -> None:
    """The reason is the only part that survives being forgotten."""
    await group_assumptions(harness, "another belief", [AssumptionVerdict.HELD] * 3)
    await user_model.derive(harness.sessions)
    facet = (await user_model.view(harness.sessions)).facets[
        Dimension.STRENGTHS.value
    ][0]

    with pytest.raises(ValueError, match="reason"):
        await user_model.dismiss(harness.sessions, facet.id, reason="   ")


# --------------------------------------------------------------------------
# 4. Contradicting evidence lowers confidence
# --------------------------------------------------------------------------


def test_contradicting_evidence_lowers_confidence() -> None:
    """**A facet with nine for and eight against is not a 90% claim**, and a
    schema with nowhere to put the eight would report it as one.

    Checked on the formula directly as well as through the deriver, because this
    is the property the number means and the deriver is only one caller of it.
    """
    clean = facet_confidence(6, 0)
    contested = facet_confidence(6, 3)
    heavily = facet_confidence(6, 5)

    assert clean > contested > heavily

    # And volume does not rescue a near-tie: nine-for-eight is a coin flip with
    # a large sample, and the formula's agreement factor holds it near a half
    # however much evidence there is on both sides.
    assert facet_confidence(9, 8) < 0.6
    assert facet_confidence(90, 80) < 0.6
    # Which is the whole point of storing the contradicting side: without it,
    # nine supporting decisions would read as the same claim as nine-for-none.
    assert facet_confidence(9, 0) > 0.9


async def test_a_belief_that_mostly_failed_is_a_weakness_with_the_failures_counted(
    harness: Harness,
) -> None:
    """The deriver's half of the same property: the same group produces a
    different dimension and a confidence that reflects both sides."""
    await group_assumptions(
        harness,
        "extraction will cover enough of the corpus",
        [
            AssumptionVerdict.FAILED,
            AssumptionVerdict.FAILED,
            AssumptionVerdict.FAILED,
            AssumptionVerdict.HELD,
        ],
    )

    await user_model.derive(harness.sessions)

    facets = (await user_model.view(harness.sessions)).facets[
        Dimension.WEAKNESSES.value
    ]
    assert len(facets) == 1
    assert facets[0].support_count == 3
    assert facets[0].contradiction_count == 1
    assert facets[0].confidence is not None
    assert facets[0].confidence < facet_confidence(3, 0)
    # Both sides are cited, so a reader can check the one that argues against it.
    relations = {relation for _, _, relation in facets[0].evidence}
    assert relations == {"supports", "contradicts"}


# --------------------------------------------------------------------------
# Asserting
# --------------------------------------------------------------------------


async def test_a_goal_is_stated_and_carries_no_confidence(harness: Harness) -> None:
    """**Null rather than 1.0**, and not an omission: a goal somebody stated is
    not a claim with a probability, and 1.0 would sort every user statement above
    every derived facet in any ranking that reads the column."""
    facet_id = await user_model.assert_facet(
        harness.sessions,
        dimension=Dimension.GOALS,
        statement="Ship a corpus large enough that these dimensions can be derived.",
    )

    facets = (await user_model.view(harness.sessions)).facets[Dimension.GOALS.value]

    assert [item.id for item in facets] == [facet_id]
    assert facets[0].confidence is None
    assert facets[0].origin == FacetOrigin.ASSERTED.value
    assert facets[0].detector is None


async def test_derivation_never_touches_an_asserted_facet(harness: Harness) -> None:
    """A nightly re-derivation must not overwrite something somebody typed, which
    is what `origin` is enforced for rather than merely recorded."""
    facet_id = await user_model.assert_facet(
        harness.sessions,
        dimension=Dimension.WEAKNESSES,
        statement="I under-estimate how long extraction takes.",
    )

    await user_model.derive(harness.sessions)

    facets = (await user_model.view(harness.sessions)).facets[
        Dimension.WEAKNESSES.value
    ]
    assert [item.id for item in facets] == [facet_id]
    assert facets[0].superseded_by is None


# --------------------------------------------------------------------------
# M8.2: how the model changes
# --------------------------------------------------------------------------
#
# The three properties model evolution rests on, and all three are about the
# system refusing to forget. A facet that loses its evidence is retired rather
# than deleted; the retirement is visible in a diff between two dates; and a
# claim somebody rejected stays rejected across a re-derivation that would
# otherwise have nothing to stop it being proposed again.


async def _ungroup(harness: Harness, group_id: UUID) -> None:
    """Take every assumption out of a group, so its deriver stops proposing it.

    The realistic way support disappears here. Nothing is deleted — the
    decisions and the assumptions all remain — and the group simply stops being
    a group, which is exactly what happens when somebody regroups by hand.
    """
    async with harness.sessions() as session, session.begin():
        rows = list(
            (
                await session.execute(
                    models.DecisionAssumption.__table__.select().where(
                        models.DecisionAssumption.group_id == group_id
                    )
                )
            ).all()
        )
        for row in rows:
            assumption = await session.get(models.DecisionAssumption, row.id)
            assert assumption is not None
            assumption.group_id = None


async def test_a_facet_losing_its_support_is_superseded_not_deleted(
    harness: Harness,
) -> None:
    """**M1.1's principle, applied to a claim about a person.**

    An honest record of a belief that changed beats a clean current state. The
    facet stops being live, keeps its id, keeps its evidence, and gains a written
    reason — so "the system used to think this about me, and stopped on the 14th"
    stays answerable. Deleting the row would make it unanswerable and would look
    identical to a claim nobody ever made.
    """
    group_id = await group_assumptions(
        harness, "the projection can be rebuilt from Postgres", [AssumptionVerdict.HELD] * 3
    )
    await user_model.derive(harness.sessions)
    facet = (await user_model.view(harness.sessions)).facets[
        Dimension.STRENGTHS.value
    ][0]

    await _ungroup(harness, group_id)
    report = await user_model.derive(harness.sessions)

    assert report.withdrawn == 1
    assert report.written == 0
    # Not superseded-by-replacement: there is no replacement, and inventing one
    # would put a statement in the table that no deriver produced.
    assert report.superseded == 0

    # Still there, still readable, still citing what it rested on.
    chain = await user_model.history(harness.sessions, facet.id)
    assert [item.id for item in chain] == [facet.id]
    retired = chain[0]
    assert retired.superseded_at is not None
    assert retired.superseded_by is None
    assert retired.withdrawn
    assert not retired.live
    assert len(retired.evidence) == 3
    # And the reason is in words, not a status code: "superseded" alone tells a
    # reader that something happened and nothing about what would change it.
    assert retired.superseded_reason is not None
    assert "no longer proposes this subject" in retired.superseded_reason

    # Gone from the live model, and the dimension says what it lacks rather than
    # going quiet.
    model = await user_model.view(harness.sessions)
    assert Dimension.STRENGTHS.value not in model.facets
    assert not model.assessments[Dimension.STRENGTHS.value].has_evidence


async def test_support_falling_below_the_bar_retires_with_the_number_it_fell_to(
    harness: Harness,
) -> None:
    """The other way support disappears, and the reason distinguishes them.

    A subject nobody proposes any more and a subject whose evidence thinned are
    different findings — the first says the thing the facet was about is not in
    the data, the second says how far it moved — and a single reason string for
    both would lose the more actionable half.
    """
    group_id = await group_assumptions(
        harness, "the cache survives a replay", [AssumptionVerdict.HELD] * 3
    )
    await user_model.derive(harness.sessions)
    facet = (await user_model.view(harness.sessions)).facets[
        Dimension.STRENGTHS.value
    ][0]

    # One member loses its evaluation, so the group drops to two observations:
    # still proposed, no longer above the bar.
    async with harness.sessions() as session, session.begin():
        row = (
            await session.execute(
                models.DecisionAssumption.__table__.select().where(
                    models.DecisionAssumption.group_id == group_id
                )
            )
        ).first()
        assert row is not None
        assumption = await session.get(models.DecisionAssumption, row.id)
        assert assumption is not None
        assumption.held = None
        assumption.evaluated_at = None

    report = await user_model.derive(harness.sessions)

    assert report.withdrawn == 1
    retired = (await user_model.history(harness.sessions, facet.id))[0]
    assert retired.superseded_reason is not None
    assert "fell from 3 to 2" in retired.superseded_reason
    assert str(MIN_SUPPORT) in retired.superseded_reason


async def test_diff_reports_added_superseded_and_dismissed_between_two_dates(
    harness: Harness,
) -> None:
    """**All three kinds in one window, and each in the right bucket.**

    The trap the test is built around is double-counting a revision: it retires
    one row and inserts another, and a diff that reported both would show one
    event as an addition plus a supersession, with the added list carrying a
    sentence that is really the second half of the supersession above it.
    """
    before_everything = datetime.now(UTC)

    # One facet that will be revised, one that will be dismissed, one that stays.
    revised_group = await group_assumptions(
        harness, "chunk boundaries survive re-chunking", [AssumptionVerdict.HELD] * 3
    )
    await group_assumptions(
        harness, "entity resolution converges", [AssumptionVerdict.HELD] * 3
    )
    await user_model.derive(harness.sessions)
    facets = (await user_model.view(harness.sessions)).facets[
        Dimension.STRENGTHS.value
    ]
    assert len(facets) == 2
    revised = next(item for item in facets if item.subject_key == str(revised_group))
    doomed = next(item for item in facets if item.id != revised.id)

    # The window opens here: everything above is outside it.
    since = datetime.now(UTC)

    # A fourth decision moves the first facet's counts, so its statement moves.
    decision_id = await a_decision(
        harness, "And once more?", "chunk boundaries survive re-chunking"
    )
    async with harness.sessions() as session, session.begin():
        row = (
            await session.execute(
                models.DecisionAssumption.__table__.select().where(
                    models.DecisionAssumption.decision_id == decision_id
                )
            )
        ).first()
        assert row is not None
        assumption = await session.get(models.DecisionAssumption, row.id)
        assert assumption is not None
        assumption.group_id = revised_group
        assumption.held = AssumptionVerdict.HELD.value
        assumption.evaluated_at = DECIDED_AT
    await user_model.derive(harness.sessions)

    await user_model.dismiss(harness.sessions, doomed.id, reason="not a strength of mine")

    # And one facet asserted by hand inside the window, which is an addition with
    # no predecessor.
    stated = await user_model.assert_facet(
        harness.sessions,
        dimension=Dimension.GOALS,
        statement="Grow the corpus until these dimensions can be derived.",
    )

    report = await user_model.diff(harness.sessions, since=since)

    assert [item.facet.id for item in report.superseded] == [revised.id]
    assert report.superseded[0].replacement is not None
    assert report.superseded[0].replacement.support_count == 4
    assert not report.superseded[0].withdrawn

    # The replacement is *not* also an addition. One event, one entry.
    added = {item.facet.id for item in report.added}
    assert added == {stated}
    assert report.superseded[0].replacement.id not in added

    assert [item.facet.id for item in report.dismissed] == [doomed.id]

    # Confidence moved with the statement, and both ends are reported.
    moves = report.confidence_changes
    assert len(moves) == 1
    _, was, is_now = moves[0]
    assert was != is_now

    # A window that opened before any of it saw every write as well.
    everything = await user_model.diff(harness.sessions, since=before_everything)
    assert len(everything.added) == 3
    assert len(everything.superseded) == 1
    assert len(everything.dismissed) == 1

    # And a window that closes before the changes sees the writes and nothing
    # else, which is what makes the first assertion a boundary rather than a
    # filter that happens to work.
    early = await user_model.diff(
        harness.sessions, since=before_everything, until=since
    )
    assert len(early.added) == 2
    assert early.superseded == ()
    assert early.dismissed == ()


async def test_rederivation_does_not_resurrect_a_dismissed_facet(
    harness: Harness,
) -> None:
    """**A re-derivation that re-proposed a rejected claim would be arguing with
    its user**, and M8.2 gives it a new way to try.

    M8.0 checked this against a run whose evidence had not moved. Here the
    evidence goes away and comes back — the group is disbanded and rebuilt — so
    the second derivation meets the subject as though for the first time, with
    no live row to compare against. The dismissal is keyed by subject and read
    before the bar, which is what makes the claim stay dead.
    """
    label = "a belief that will be rejected and then re-evidenced"
    await group_assumptions(harness, label, [AssumptionVerdict.HELD] * 3)
    await user_model.derive(harness.sessions)
    facet = (await user_model.view(harness.sessions)).facets[
        Dimension.STRENGTHS.value
    ][0]
    await user_model.dismiss(harness.sessions, facet.id, reason="that is luck, not skill")

    # A fresh group with the same label and three fresh decisions: new subject
    # key, so this half is a control rather than the assertion.
    await group_assumptions(harness, label, [AssumptionVerdict.HELD] * 3)
    # And the original subject, re-evidenced.
    report = await user_model.derive(harness.sessions)

    assert report.skipped_dismissed == 1
    live = (await user_model.view(harness.sessions)).facets.get(
        Dimension.STRENGTHS.value, []
    )
    assert facet.id not in {item.id for item in live}
    # The dismissed row is still dismissed, still not live, and still carries the
    # reason — a resurrection would show up here as a row that is live again.
    model = await user_model.view(harness.sessions)
    dismissed = {item.id: item for item in model.dismissed}
    assert facet.id in dismissed
    assert dismissed[facet.id].dismissed_reason == "that is luck, not skill"
    assert not dismissed[facet.id].live
    # And it was never quietly retired instead: dismissal and withdrawal are
    # different endings and only one of them is a person's judgement.
    assert dismissed[facet.id].superseded_at is None


async def test_stability_withholds_a_verdict_on_thin_history(harness: Harness) -> None:
    """**"Cannot say" is the honest output at this data scale, and it is a
    number rather than a shrug.**

    A dimension with one closed facet has a mean lifetime equal to that facet's
    lifetime; printing it as a mean invites a reader to treat one event as a
    rate. The thresholds are what stop it, and the test asserts the refusal
    rather than the arithmetic — the arithmetic is trivially right and the
    refusal is the part that would rot.
    """
    entries = {item.dimension: item for item in await user_model.stability(harness.sessions)}

    # Every dimension is present, including the two nothing derives.
    assert set(entries) == set(Dimension)
    empty = entries[Dimension.HABITS]
    assert empty.total == 0
    assert empty.mean_lifetime_days is None
    assert empty.verdict() == "no facets: nothing to measure"

    # One facet, written just now, closed by nothing.
    await group_assumptions(
        harness, "a single strength", [AssumptionVerdict.HELD] * 3
    )
    await user_model.derive(harness.sessions)

    entries = {item.dimension: item for item in await user_model.stability(harness.sessions)}
    strengths = entries[Dimension.STRENGTHS]
    assert strengths.total == 1
    assert strengths.live == 1
    assert strengths.closed == 0
    assert strengths.changes == 0
    # A lower bound, reported apart from the mean lifetime rather than pooled
    # into it: a live facet has not finished its lifetime.
    assert strengths.mean_lifetime_days is None
    assert strengths.mean_live_age_days is not None
    assert not strengths.has_verdict
    assert "cannot say" in strengths.verdict()
    # Which failure mode is not decidable either way, so neither is claimed.
    assert "fitting noise" not in strengths.verdict()
    assert "stable" not in strengths.verdict()


async def test_timeline_records_every_event_including_the_withdrawal(
    harness: Harness,
) -> None:
    """One row can produce three events, and a timeline that collapsed them into
    the row's current state would be `model show` with dates on it."""
    group_id = await group_assumptions(
        harness, "a belief with a whole life", [AssumptionVerdict.HELD] * 3
    )
    await user_model.derive(harness.sessions)
    await _ungroup(harness, group_id)
    await user_model.derive(harness.sessions)

    events = await user_model.timeline(harness.sessions)

    kinds = [item.kind for item in events]
    assert kinds == ["written", "withdrawn"]
    # Oldest first, and the withdrawal carries the reason rather than a status.
    assert events[0].at <= events[1].at
    assert "no longer proposes" in events[1].detail

    # Filtering by dimension does not change what an event is.
    assert (
        await user_model.timeline(harness.sessions, dimension=Dimension.STRENGTHS)
        == events
    )
    assert await user_model.timeline(harness.sessions, dimension=Dimension.GOALS) == ()
