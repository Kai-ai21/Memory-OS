"""The four claims M5.1 makes, and the one property the whole milestone rests on.

**A candidate before the decision is never suggested.** Everything here is a
causal claim, and a causal claim running backwards is not a weak one — it is
incoherent. The rule is enforced in the query, in the module, and by a CHECK
constraint, because it is the premise the other three tests assume.

**Suggestions land in review.** M5.0's rule, for a proposal that is easier to get
wrong: post hoc ergo propter hoc is the oldest error there is, and two documents
from one repository are related by default.

**`too_early` is a verdict and is outside the success rate.** A system that
counted it as a failure would punish caution; one that counted it as a success
would be lying. It is in neither half of the fraction.

**Deleting a memory takes its outcome evidence and leaves the outcome.** An
outcome is not made false by losing a piece of the evidence for it.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.decisions import (
    AssumptionInput,
    DecisionDraft,
    EvidenceInput,
    OptionInput,
)
from memoryos.application.decisions import (
    link_evidence as link_decision_evidence,
)
from memoryos.application.decisions import (
    record as record_decision,
)
from memoryos.application.outcome_suggest import (
    DEFAULT_WINDOW_DAYS,
    MAX_WINDOW_DAYS,
    MIN_WINDOW_DAYS,
    SuggestOutcomes,
    accept,
    find_candidates,
    list_suggestions,
    open_decisions,
    reject,
    window_for,
)
from memoryos.application.outcomes import (
    InvalidOutcome,
    OutcomeDraft,
    OutcomeEvidenceInput,
    for_decision,
    link_evidence,
    record,
    success_rate,
)
from memoryos.domain.values import (
    RESOLVED_VERDICTS,
    EvidenceKind,
    OutcomeVerdict,
    SuggestionStatus,
    TimeProvenance,
)
from tests.integration.conftest import Harness
from tests.support.fakes import FakeLanguageModel

pytestmark = pytest.mark.integration

# The harness corpus is dated by the fixture's own mtimes, so the decision is
# placed a long way before them and the window is left wide. What is being
# tested is the ordering rule, not the calendar.
DECIDED_AT = datetime(2020, 1, 1, 12, 0, tzinfo=UTC)


def decision_draft(**overrides: object) -> DecisionDraft:
    fields: dict[str, object] = {
        "question": "What runs background work?",
        "chosen": "A Postgres table",
        "reasoning": "The enqueue and the row it refers to are one transaction.",
        "confidence": 0.9,
        "expected_outcome": "Throughput is never the binding constraint.",
        "options": (
            OptionInput(
                description="Celery with Redis",
                rejected_because="Cannot enlist in the Postgres transaction.",
            ),
        ),
        "assumptions": (AssumptionInput(statement="Throughput stays low", confidence=0.9),),
    }
    fields.update(overrides)
    return DecisionDraft(**fields)  # type: ignore[arg-type]


async def a_decision(
    sessions: async_sessionmaker[AsyncSession],
    *,
    decided_at: datetime = DECIDED_AT,
    **overrides: object,
) -> UUID:
    return await record_decision(
        sessions,
        decision_draft(**overrides),
        decided_at=decided_at,
        decided_at_source=TimeProvenance.DECLARED,
    )


async def corpus_bounds(
    sessions: async_sessionmaker[AsyncSession],
) -> tuple[datetime, datetime]:
    """The corpus's own `occurred_at` range, refused if it has none.

    Asserted rather than assumed: every test below places a decision relative to
    these, and a fixture that produced undated memories would otherwise fail
    somewhere much further along with a `None` in the arithmetic.
    """
    async with sessions() as session:
        row = (
            await session.execute(
                select(func.min(models.Memory.occurred_at), func.max(models.Memory.occurred_at))
            )
        ).one()
    earliest, latest = row
    assert earliest is not None and latest is not None, "the corpus has no dated memories"
    return earliest, latest


async def count(
    sessions: async_sessionmaker[AsyncSession], model: type[models.Base]
) -> int:
    async with sessions() as session:
        return int(
            (await session.execute(select(func.count()).select_from(model))).scalar_one()
        )


# --------------------------------------------------------------------------
# 1. Nothing before the decision is ever a candidate
# --------------------------------------------------------------------------


async def test_a_memory_occurring_before_the_decision_is_never_a_candidate(
    harness: Harness,
) -> None:
    """The premise the whole milestone rests on.

    The corpus is dated by filesystem mtimes in the recent past. A decision
    placed *after* all of it therefore has nothing that followed it, and the
    correct answer is an empty candidate list rather than the nearest documents
    — which is what a search would have returned and what makes this a temporal
    question rather than a retrieval one.
    """
    _, latest = await corpus_bounds(harness.sessions)

    decision_id = await a_decision(harness.sessions, decided_at=latest + timedelta(days=1))
    (context,) = await open_decisions(harness.sessions, decision_id=decision_id)

    candidates, _ = await find_candidates(
        harness.sessions, context, window_days=MAX_WINDOW_DAYS
    )

    assert candidates == []


async def test_a_memory_at_the_same_instant_is_not_after_the_decision(
    harness: Harness,
) -> None:
    """The boundary `memories_in_range` admits and this milestone must not.

    M4.0's range is half-open — closed at the start — so a memory occurring at
    exactly `decided_at` is inside it. Simultaneous is not afterwards, and a
    zero gap would be a causal claim about nothing. The CHECK constraint says
    `gap_days > 0` for the same reason.
    """
    earliest, _ = await corpus_bounds(harness.sessions)

    decision_id = await a_decision(harness.sessions, decided_at=earliest)
    (context,) = await open_decisions(harness.sessions, decision_id=decision_id)

    candidates, _ = await find_candidates(
        harness.sessions, context, window_days=MAX_WINDOW_DAYS
    )

    assert all(candidate.gap_days > 0 for candidate in candidates)
    assert all(candidate.occurred_at > earliest for candidate in candidates)


async def test_a_decisions_own_evidence_is_not_its_outcome(harness: Harness) -> None:
    """A memory that informed a decision cannot be evidence of its result.

    It falls inside the window whenever its mtime happens to, and admitting it
    would make every decision look as though its own reasoning had proved it
    right.
    """
    decision_id = await a_decision(harness.sessions)
    await link_decision_evidence(
        harness.sessions,
        decision_id,
        EvidenceInput(source_name="corpus", external_key="queue.md"),
    )
    (context,) = await open_decisions(harness.sessions, decision_id=decision_id)

    candidates, _ = await find_candidates(
        harness.sessions, context, window_days=MAX_WINDOW_DAYS
    )

    assert "queue.md" not in {candidate.external_key for candidate in candidates}


async def test_the_window_widens_with_the_decisions_confidence(harness: Harness) -> None:
    """The heuristic, asserted where it is stated rather than where it is used.

    A low-confidence decision is one you expected to learn about sooner, so the
    window grows with confidence. There is no evidence for this — it is a stated
    guess, and the test exists so that changing it is deliberate.
    """
    assert window_for(0.0) == MIN_WINDOW_DAYS
    assert window_for(1.0) == MAX_WINDOW_DAYS
    assert window_for(0.45) < window_for(0.95)
    # No confidence is not 0.5. A missing number gets the default rather than a
    # midpoint, so nothing invents a confidence in order to derive a window.
    assert window_for(None) == DEFAULT_WINDOW_DAYS
    # And an explicit override wins outright, which is the point of stating a
    # heuristic: somebody can disagree with it per run.
    assert window_for(0.9, override=7.0) == 7.0


async def test_a_narrow_window_excludes_what_a_wide_one_admits(
    harness: Harness,
) -> None:
    # The window is doing real work rather than being decoration: the same
    # decision with a one-day window sees strictly less than with a wide one.
    earliest, _ = await corpus_bounds(harness.sessions)

    decision_id = await a_decision(
        harness.sessions, decided_at=earliest - timedelta(days=1)
    )
    (context,) = await open_decisions(harness.sessions, decision_id=decision_id)

    wide, _ = await find_candidates(harness.sessions, context, window_days=365, limit=99)
    narrow, _ = await find_candidates(
        harness.sessions, context, window_days=0.5, limit=99
    )

    assert len(wide) > 0
    assert len(narrow) < len(wide)


# --------------------------------------------------------------------------
# 2. Suggestions land in review, never in the table
# --------------------------------------------------------------------------

JUDGED_YES = """
{"answer": "yes", "verdict": "worked",
 "description": "The queue drained without a broker.",
 "rationale": "the worker claims a task and holds a lease",
 "confidence": 0.8}
"""

JUDGED_UNSURE = '{"answer": "unsure", "verdict": null, "description": null, ' \
    '"rationale": null, "confidence": 0.0}'


async def a_decision_with_candidates(harness: Harness) -> UUID:
    earliest, _ = await corpus_bounds(harness.sessions)
    return await a_decision(harness.sessions, decided_at=earliest - timedelta(hours=1))


async def test_a_suggestion_lands_in_review_and_is_never_auto_committed(
    harness: Harness,
) -> None:
    """The safety property, and it matters more here than in M5.0.

    A wrong decision suggestion proposes a record of a choice nobody made. A
    wrong outcome suggestion asserts that one thing *caused* another, and M5.4
    would state it as a fact about how somebody works.
    """
    decision_id = await a_decision_with_candidates(harness)
    suggest = SuggestOutcomes(harness.sessions, FakeLanguageModel(JUDGED_YES))

    report = await suggest(decision_id=decision_id, window_days=365, limit=3)

    assert report.proposed >= 1
    assert await count(harness.sessions, models.OutcomeSuggestion) == report.proposed
    # The table M5.3 will read is still empty.
    assert await count(harness.sessions, models.DecisionOutcome) == 0
    assert await for_decision(harness.sessions, decision_id) == []

    queued = await list_suggestions(harness.sessions)
    assert all(row.status is SuggestionStatus.PENDING for row in queued)
    # Every row carries the basis of its claim, not just the conclusion.
    assert all(row.gap_days > 0 for row in queued)
    assert all(row.window_days == 365 for row in queued)
    assert all(row.entity_filter in ("applied", "unavailable") for row in queued)
    assert all(row.source_text for row in queued)


async def test_an_unsure_judgement_never_reaches_the_queue(harness: Harness) -> None:
    """The model is allowed to say unsure, and an unsure is a drop.

    Not a low-confidence row. A reviewer shown weak candidates among strong ones
    learns to skim, and skimming is what the queue exists to prevent.
    """
    decision_id = await a_decision_with_candidates(harness)
    suggest = SuggestOutcomes(harness.sessions, FakeLanguageModel(JUDGED_UNSURE))

    report = await suggest(decision_id=decision_id, window_days=365, limit=3)

    assert report.judged_unsure > 0
    assert report.proposed == 0
    assert await count(harness.sessions, models.OutcomeSuggestion) == 0


async def test_a_yes_below_the_confidence_floor_is_dropped(harness: Harness) -> None:
    decision_id = await a_decision_with_candidates(harness)
    timid = (
        '{"answer": "yes", "verdict": "worked", "description": "maybe", '
        '"rationale": null, "confidence": 0.2}'
    )
    suggest = SuggestOutcomes(harness.sessions, FakeLanguageModel(timid))

    report = await suggest(decision_id=decision_id, window_days=365, limit=3)

    assert report.judged_yes > 0
    assert report.below_confidence > 0
    assert report.proposed == 0


async def test_accepting_writes_an_inferred_outcome_never_a_declared_one(
    harness: Harness,
) -> None:
    """Accepting is not the same as having watched it happen.

    An accepted suggestion promoted to `declared` would be indistinguishable
    from testimony to M5.3, which is the one thing `evidence_kind` exists to
    prevent. The schema forbids the confidence from reaching 1.0 as well.
    """
    decision_id = await a_decision_with_candidates(harness)
    await SuggestOutcomes(harness.sessions, FakeLanguageModel(JUDGED_YES))(
        decision_id=decision_id, window_days=365, limit=1
    )
    (queued,) = (await list_suggestions(harness.sessions))[:1]

    outcome_id = await accept(harness.sessions, queued.id)

    (outcome,) = await for_decision(harness.sessions, decision_id)
    assert outcome.id == outcome_id
    assert outcome.evidence_kind is EvidenceKind.INFERRED
    assert outcome.confidence is not None and outcome.confidence < 1.0
    # The date is the candidate memory's own, carrying its provenance — an mtime
    # stays an mtime — rather than the moment somebody cleared the queue.
    assert outcome.observed_at == queued.candidate_occurred_at
    assert outcome.observed_at_source is TimeProvenance.FILESYSTEM
    # And the candidate is kept as the evidence.
    assert [item.external_key for item in outcome.evidence] == [queued.external_key]

    async with harness.sessions() as session:
        row = await session.get(models.OutcomeSuggestion, queued.id)
        assert row is not None
        assert row.status == SuggestionStatus.ACCEPTED.value
        assert row.outcome_id == outcome_id


async def test_a_rejected_candidate_is_kept_and_not_proposed_again(
    harness: Harness,
) -> None:
    decision_id = await a_decision_with_candidates(harness)
    await SuggestOutcomes(harness.sessions, FakeLanguageModel(JUDGED_YES))(
        decision_id=decision_id, window_days=365, limit=1
    )
    (queued,) = (await list_suggestions(harness.sessions))[:1]

    await reject(harness.sessions, queued.id)

    assert await list_suggestions(harness.sessions, status=SuggestionStatus.PENDING) == []
    kept = await list_suggestions(harness.sessions, status=SuggestionStatus.REJECTED)
    assert [row.id for row in kept] == [queued.id]


async def test_an_inferred_outcome_cannot_claim_certainty(harness: Harness) -> None:
    # Enforced in the use case as well as by the CHECK, so the message names the
    # reason rather than the constraint.
    decision_id = await a_decision(harness.sessions)
    with pytest.raises(InvalidOutcome, match="cannot claim certainty"):
        await record(
            harness.sessions,
            decision_id,
            OutcomeDraft(description="it worked", verdict=OutcomeVerdict.WORKED),
            observed_at=datetime.now(UTC),
            observed_at_source=TimeProvenance.INFERRED,
            evidence_kind=EvidenceKind.INFERRED,
            confidence=1.0,
        )


# --------------------------------------------------------------------------
# 3. `too_early` is a verdict, and it is outside the rate
# --------------------------------------------------------------------------


async def test_too_early_is_accepted_and_excluded_from_the_success_rate(
    harness: Harness,
) -> None:
    """The verdict that has to be recordable and must not count.

    Counting it as a failure would punish caution; counting it as a success
    would be lying. It is in neither half of the fraction, and `undecided` — a
    decision nobody has looked at — is a third number again.
    """
    worked = await a_decision(harness.sessions, question="Did the queue hold?")
    failed = await a_decision(harness.sessions, question="Did the cache help?")
    early = await a_decision(harness.sessions, question="Was the graph worth it?")
    await a_decision(harness.sessions, question="Nobody has looked at this one")

    for decision_id, verdict in (
        (worked, OutcomeVerdict.WORKED),
        (failed, OutcomeVerdict.FAILED),
        (early, OutcomeVerdict.TOO_EARLY),
    ):
        await record(
            harness.sessions,
            decision_id,
            OutcomeDraft(description=f"{verdict.value} outcome", verdict=verdict),
            observed_at=datetime.now(UTC),
            observed_at_source=TimeProvenance.DECLARED,
        )

    rate = await success_rate(harness.sessions)

    assert rate.worked == 1
    assert rate.failed == 1
    assert rate.too_early == 1
    assert rate.undecided == 1
    # One worked over one worked plus one failed. The `too_early` is in neither.
    assert rate.resolved == 2
    assert rate.rate == pytest.approx(0.5)
    assert OutcomeVerdict.TOO_EARLY not in RESOLVED_VERDICTS


async def test_a_corpus_of_only_too_early_has_no_success_rate(
    harness: Harness,
) -> None:
    """None, not 0.0.

    Zero would read as "everything failed", which on a young project is the
    opposite of what the data says — and it is the number a dashboard would
    print with a decimal point.
    """
    decision_id = await a_decision(harness.sessions)
    await record(
        harness.sessions,
        decision_id,
        OutcomeDraft(description="no result yet", verdict=OutcomeVerdict.TOO_EARLY),
        observed_at=datetime.now(UTC),
        observed_at_source=TimeProvenance.DECLARED,
    )

    rate = await success_rate(harness.sessions)

    assert rate.too_early == 1
    assert rate.resolved == 0
    assert rate.rate is None


async def test_a_later_verdict_supersedes_an_earlier_one_in_the_rate(
    harness: Harness,
) -> None:
    """Both rows are kept; the summary counts the decision once.

    A decision that worked and then failed is one decision that failed, not one
    of each — and the sequence stays intact for M5.3, which is the milestone
    that cares about it.
    """
    decision_id = await a_decision(harness.sessions)
    for verdict, when in (
        (OutcomeVerdict.TOO_EARLY, datetime(2026, 1, 1, tzinfo=UTC)),
        (OutcomeVerdict.WORKED, datetime(2026, 6, 1, tzinfo=UTC)),
    ):
        await record(
            harness.sessions,
            decision_id,
            OutcomeDraft(description=verdict.value, verdict=verdict),
            observed_at=when,
            observed_at_source=TimeProvenance.DECLARED,
        )

    recorded = await for_decision(harness.sessions, decision_id)
    rate = await success_rate(harness.sessions)

    # Oldest first: the story reads in order.
    assert [row.verdict for row in recorded] == [
        OutcomeVerdict.TOO_EARLY,
        OutcomeVerdict.WORKED,
    ]
    assert rate.worked == 1
    assert rate.too_early == 0


# --------------------------------------------------------------------------
# 4. An outcome survives losing a piece of its evidence
# --------------------------------------------------------------------------


async def test_deleting_a_memory_cascades_to_outcome_evidence_and_leaves_the_outcome(
    harness: Harness,
) -> None:
    """The same property `decision_evidence` has, on the outcome side.

    A link to a document that no longer exists is a citation to nothing; an
    outcome that lost one of two citations is still an outcome somebody
    recorded.
    """
    decision_id = await a_decision(harness.sessions)
    outcome_id = await record(
        harness.sessions,
        decision_id,
        OutcomeDraft(
            description="the queue drained",
            verdict=OutcomeVerdict.WORKED,
            evidence=(
                OutcomeEvidenceInput(source_name="corpus", external_key="queue.md"),
            ),
        ),
        observed_at=datetime.now(UTC),
        observed_at_source=TimeProvenance.DECLARED,
    )
    await link_evidence(
        harness.sessions,
        outcome_id,
        OutcomeEvidenceInput(source_name="corpus", external_key="bread.txt"),
    )
    assert await count(harness.sessions, models.OutcomeEvidence) == 2

    async with harness.sessions.begin() as session:
        await session.execute(
            delete(models.Memory).where(models.Memory.external_key == "queue.md")
        )

    (outcome,) = await for_decision(harness.sessions, decision_id)
    assert outcome.description == "the queue drained"
    assert outcome.verdict is OutcomeVerdict.WORKED
    # One link gone, one left. Not a dangling row, and not an outcome deleted
    # along with a document.
    assert [item.external_key for item in outcome.evidence] == ["bread.txt"]


async def test_deleting_a_decision_takes_its_outcomes(harness: Harness) -> None:
    # The other direction, and it cascades: an outcome of a decision that no
    # longer exists is an answer to a question nobody asked.
    decision_id = await a_decision(harness.sessions)
    await record(
        harness.sessions,
        decision_id,
        OutcomeDraft(description="it worked", verdict=OutcomeVerdict.WORKED),
        observed_at=datetime.now(UTC),
        observed_at_source=TimeProvenance.DECLARED,
    )

    async with harness.sessions.begin() as session:
        await session.execute(
            delete(models.Decision).where(models.Decision.id == decision_id)
        )

    assert await count(harness.sessions, models.DecisionOutcome) == 0


async def test_a_replay_relinks_outcome_evidence_by_natural_key(
    harness: Harness,
) -> None:
    """`outcome_evidence` is `decision_evidence`'s problem a second time.

    Both hold cascading foreign keys into `memories`, so `TRUNCATE ... CASCADE`
    takes them however they are classified. Replay walks a declared list of
    evidence tables rather than naming one, which is what stops the second — and
    the third — from being forgotten silently.
    """
    decision_id = await a_decision(harness.sessions)
    outcome_id = await record(
        harness.sessions,
        decision_id,
        OutcomeDraft(
            description="the queue drained",
            verdict=OutcomeVerdict.WORKED,
            evidence=(
                OutcomeEvidenceInput(source_name="corpus", external_key="queue.md"),
            ),
        ),
        observed_at=datetime.now(UTC),
        observed_at_source=TimeProvenance.DECLARED,
    )
    (before,) = (await for_decision(harness.sessions, decision_id))[0].evidence

    report = await harness.replay(clear_cache=False)
    assert report.evidence_preserved >= 1
    assert report.evidence_relinked >= 1

    (outcome,) = await for_decision(harness.sessions, decision_id)
    assert outcome.id == outcome_id
    (after,) = outcome.evidence
    assert after.external_key == "queue.md"
    # A new row pointing at a new memory: the link survived, the id did not.
    assert after.memory_id != before.memory_id
    # And the snapshot came across rather than being re-derived.
    assert after.occurred_at == before.occurred_at
