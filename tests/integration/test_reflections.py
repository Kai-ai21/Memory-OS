"""Reflections over a real corpus: what is written, what is refused, what stops.

The refusal itself and the grounding check are unit-tested without a database in
`tests/unit/test_reflection_grounding.py`, including the milestone's golden case.
What this file checks is that the guards are wired to something: that a pattern
below the bar produces no *row*, that a fabricated citation leaves nothing
behind, that the numbering a reflection cites is frozen against the decisions it
was generated from, and that a dismissal survives the next run.

The model is a fake throughout. `LanguageModel` is a port this project owns, and
what a real model does with the prompt is the subject of the milestone's own
report rather than of a test — a test that called Groq would be asserting about
somebody else's weights. What can be established here is that the pipeline
assembles the right evidence, verifies what comes back, and stores only what
verified, however badly the model behaves.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

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
from memoryos.application.patterns import discover, list_patterns
from memoryos.application.reflections import (
    acknowledge,
    dismiss,
    list_reflections,
    reflect,
)
from memoryos.domain.values import (
    AssumptionVerdict,
    MergeStrategy,
    PatternRelation,
    TimeProvenance,
)
from tests.integration.conftest import Harness
from tests.support.fakes import FakeLanguageModel

pytestmark = pytest.mark.integration

DECIDED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

# Cites [1] in both sentences, so it verifies whatever the pattern's size.
GROUNDED = "You underestimated this work [1]. The same belief broke again [2]."


async def a_corpus(
    sessions: async_sessionmaker[AsyncSession], *, broke: int, held: int = 0
) -> None:
    """`broke + held` decisions sharing one grouped assumption, then evaluated.

    Built by hand rather than through M5.2's embedder, for the reason
    `test_patterns.py` gives: routing it through a similarity threshold would
    make these tests about the embedder.
    """
    for index in range(broke + held):
        await record_decision(
            sessions,
            DecisionDraft(
                question=f"Should we build thing {index} ourselves?",
                chosen="A Postgres table",
                confidence=0.8,
                options=(
                    OptionInput(description="Celery", rejected_because="No transaction."),
                ),
                assumptions=(
                    AssumptionInput(statement=f"The deploy is straightforward, take {index}"),
                ),
            ),
            decided_at=DECIDED_AT + timedelta(days=index),
            decided_at_source=TimeProvenance.DECLARED,
        )
    async with sessions.begin() as session:
        group_id = uuid4()
        session.add(
            models.AssumptionGroup(
                id=group_id, label="the deploy is straightforward",
                strategy=MergeStrategy.MANUAL.value,
            )
        )
        for row in (await session.execute(select(models.DecisionAssumption))).scalars():
            row.group_id = group_id

    rows = await list_assumptions(sessions, limit=1000)
    verdicts = [AssumptionVerdict.FAILED] * broke + [AssumptionVerdict.HELD] * held
    for assumption, verdict in zip(rows, verdicts, strict=True):
        await evaluate(sessions, assumption.id, verdict)

    await discover(sessions)


async def count_reflections(sessions: async_sessionmaker[AsyncSession]) -> int:
    async with sessions() as session:
        return int(
            (
                await session.execute(select(func.count()).select_from(models.Reflection))
            ).scalar_one()
        )


# --------------------------------------------------------------------------
# 1. Below the bar, nothing is written and the model is not called
# --------------------------------------------------------------------------


async def test_a_pattern_below_the_threshold_produces_no_row(harness: Harness) -> None:
    """Three decisions agreeing is a pattern and is not a reflection.

    The pattern bar and the reflection bar are different numbers on purpose: a
    pattern is a row read beside its evidence, and a reflection is prose read as
    a claim about the reader.
    """
    await a_corpus(harness.sessions, broke=3)
    model = FakeLanguageModel(GROUNDED)

    report = await reflect(harness.sessions, model)

    assert report.considered == 1
    assert report.written == 0
    assert len(report.refused) == 1
    assert model.calls == []
    assert await count_reflections(harness.sessions) == 0
    # The refusal carries what would change it, which is what makes this output
    # a result rather than a shrug.
    assert report.refused[0].needed is not None
    assert "1 more decision(s)" in report.refused[0].needed


async def test_four_agreeing_decisions_do_clear_the_bar(harness: Harness) -> None:
    """The positive control, and the shape everything below is built on."""
    await a_corpus(harness.sessions, broke=4)
    model = FakeLanguageModel(GROUNDED)

    report = await reflect(harness.sessions, model)

    assert report.written == 1
    assert len(model.calls) == 1
    (row,) = await list_reflections(harness.sessions)
    assert row.text == GROUNDED
    assert row.citation_rate == pytest.approx(1.0)
    assert row.model_id == "fake/llm@1"
    assert row.uncited == []
    assert row.support_count == 4
    assert row.contradiction_count == 0


async def test_the_prompt_carries_the_counter_evidence_in_the_same_list(
    harness: Harness,
) -> None:
    """Not "the evidence, and separately some caveats".

    A model shown counter-evidence in a second block writes it last if it writes
    it at all, and the milestone's requirement is that it appear in the same
    paragraph as the claim it weakens.
    """
    await a_corpus(harness.sessions, broke=5, held=1)
    model = FakeLanguageModel(GROUNDED)

    await reflect(harness.sessions, model)

    prompt = model.last_user_prompt
    assert "ARGUES FOR" in prompt
    assert "ARGUES AGAINST" in prompt
    # The counter-evidence is numbered inside the one list the model must cite
    # from, so citing it is the same act as citing anything else.
    assert prompt.index("ARGUES AGAINST") > prompt.index("ARGUES FOR")
    assert "[6]" in prompt


# --------------------------------------------------------------------------
# 2. What comes back, and what is stored
# --------------------------------------------------------------------------


async def test_an_uncited_sentence_is_flagged_and_kept(harness: Harness) -> None:
    await a_corpus(harness.sessions, broke=4)
    model = FakeLanguageModel(
        "You underestimated this work [1]. You are simply an optimist."
    )

    report = await reflect(harness.sessions, model)

    assert report.written == 1
    (row,) = await list_reflections(harness.sessions)
    assert row.citation_rate == pytest.approx(0.5)
    assert row.uncited == ["You are simply an optimist."]
    # Kept in the text rather than trimmed out of it.
    assert "optimist" in row.text


async def test_a_citation_outside_the_evidence_stores_nothing(
    harness: Harness,
) -> None:
    """Rejected rather than flagged, and the row never exists.

    A reflection citing a decision it was never shown is describing somebody
    using evidence nobody put in front of it. There is no charitable reading
    under which the rest of the paragraph is still worth keeping.
    """
    await a_corpus(harness.sessions, broke=4)
    model = FakeLanguageModel("You underestimated this work [1] and also this [11].")

    report = await reflect(harness.sessions, model)

    assert report.written == 0
    assert len(report.rejected) == 1
    assert "[11]" in (report.rejected[0].rejected_because or "")
    assert await count_reflections(harness.sessions) == 0
    assert await list_reflections(harness.sessions) == []


async def test_the_citations_are_frozen_against_the_decisions_cited(
    harness: Harness,
) -> None:
    """`[1]` is stored as a foreign key, not recomputed on read.

    `patterns discover` replaces a pattern's evidence wholesale on every run, so
    a numbering re-derived at read time would silently renumber every citation
    the first time the corpus grew — and the claim would still read correctly
    while linking to a different decision. That is M1.4a's failure with a person
    on the other end of it.
    """
    await a_corpus(harness.sessions, broke=4)
    model = FakeLanguageModel(GROUNDED)

    await reflect(harness.sessions, model)

    (row,) = await list_reflections(harness.sessions)
    assert [item.marker for item in row.citations] == [1, 2]
    assert all(
        item.relation is PatternRelation.SUPPORTS for item in row.citations
    )
    # Every marker resolves to a decision that is genuinely in the pattern's
    # supporting evidence.
    (pattern,) = await list_patterns(harness.sessions)
    supporting = {item.decision_id for item in pattern.supporting}
    assert {item.decision_id for item in row.citations} <= supporting
    # And only the markers the prose actually used were written. A row for a
    # decision the paragraph never mentions is a link a reader cannot find.
    assert len(row.citations) == 2


# --------------------------------------------------------------------------
# 3. Refusing, and making the refusal stick
# --------------------------------------------------------------------------


async def test_a_dismissed_reflection_is_not_regenerated(harness: Harness) -> None:
    """"This is wrong about me", and the system believes you.

    Hiding the row would not be enough: the next `reflect --all` would write the
    same claim again under a new id. `--regenerate` does not override it either,
    because a rejection a re-run undid would not be a rejection.
    """
    await a_corpus(harness.sessions, broke=4)
    model = FakeLanguageModel(GROUNDED)
    await reflect(harness.sessions, model)
    (row,) = await list_reflections(harness.sessions)

    await dismiss(harness.sessions, row.id, reason="that is not why I did any of that")

    again = await reflect(harness.sessions, model)
    assert again.written == 0
    assert again.skipped_dismissed == 1
    assert len(model.calls) == 1

    forced = await reflect(harness.sessions, model, regenerate=True)
    assert forced.written == 0
    assert forced.skipped_dismissed == 1
    assert len(model.calls) == 1

    # Hidden by default, kept with its reason, and only one row ever existed.
    assert await list_reflections(harness.sessions) == []
    (kept,) = await list_reflections(harness.sessions, include_dismissed=True)
    assert kept.dismissed_reason == "that is not why I did any of that"
    assert await count_reflections(harness.sessions) == 1


async def test_a_dismissal_without_a_reason_is_refused(harness: Harness) -> None:
    await a_corpus(harness.sessions, broke=4)
    await reflect(harness.sessions, FakeLanguageModel(GROUNDED))
    (row,) = await list_reflections(harness.sessions)

    with pytest.raises(ValueError, match="reason"):
        await dismiss(harness.sessions, row.id, reason="  ")


async def test_acknowledging_is_not_dismissing(harness: Harness) -> None:
    # Read is not agreed with, and nothing downstream weights a reflection by
    # it. It exists so a view can stop putting an unread claim first.
    await a_corpus(harness.sessions, broke=4)
    await reflect(harness.sessions, FakeLanguageModel(GROUNDED))
    (row,) = await list_reflections(harness.sessions)

    await acknowledge(harness.sessions, row.id)

    (after,) = await list_reflections(harness.sessions)
    assert after.acknowledged_at is not None
    assert after.dismissed_at is None


async def test_an_existing_reflection_is_not_rewritten_without_asking(
    harness: Harness,
) -> None:
    # Generation costs a model call and produces prose about a person. Doing it
    # again on every run would also stack near-duplicate claims about one
    # pattern, which is how a tool starts nagging.
    await a_corpus(harness.sessions, broke=4)
    model = FakeLanguageModel(GROUNDED)
    await reflect(harness.sessions, model)

    second = await reflect(harness.sessions, model)
    assert second.written == 0
    assert second.skipped_existing == 1
    assert len(model.calls) == 1

    third = await reflect(harness.sessions, model, regenerate=True)
    assert third.written == 1
    assert await count_reflections(harness.sessions) == 2


async def test_deleting_the_pattern_takes_the_reflection_with_it(
    harness: Harness,
) -> None:
    # A claim about somebody outliving the evidence it was drawn from is the one
    # row this schema must not be able to hold.
    await a_corpus(harness.sessions, broke=4)
    await reflect(harness.sessions, FakeLanguageModel(GROUNDED))
    (pattern,) = await list_patterns(harness.sessions)

    async with harness.sessions.begin() as session:
        await session.delete(await session.get_one(models.Pattern, pattern.id))

    assert await count_reflections(harness.sessions) == 0


async def test_one_pattern_by_id_leaves_the_others_alone(harness: Harness) -> None:
    await a_corpus(harness.sessions, broke=4)
    (pattern,) = await list_patterns(harness.sessions)
    model = FakeLanguageModel(GROUNDED)

    report = await reflect(harness.sessions, model, pattern_id=pattern.id)
    assert report.considered == 1
    assert report.written == 1

    nothing = await reflect(harness.sessions, model, pattern_id=UUID(int=0))
    assert nothing.considered == 0
    assert nothing.written == 0


async def test_an_empty_corpus_produces_nothing_and_does_not_call_the_model(
    harness: Harness,
) -> None:
    model = FakeLanguageModel(GROUNDED)

    report = await reflect(harness.sessions, model)

    assert report.considered == 0
    assert report.written == 0
    assert model.calls == []
