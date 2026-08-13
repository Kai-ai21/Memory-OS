"""The refusal, the arithmetic behind it, and the check on what comes back.

No database and no model. Everything here is a pure function, which is the point:
the guard that decides whether a behavioural claim may be written about somebody
is arithmetic, not a prompt instruction, and it should be checkable against a
hand-computed number.

**The golden case lives here** — a deliberately weak pattern, two supporting and
two contradicting, must produce no reflection — because it can be stated without
a database. It cannot even be stored as one: `ck_patterns_support_exceeds_
contradiction` forbids a row with equal counts, so the shape this test uses is
one the schema itself refuses. That is worth having anyway, since
`generate_reflection` takes a pattern rather than an id and must refuse on its
own rather than because a table did.
"""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from memoryos.application.patterns import EvidenceRow, PatternRow
from memoryos.application.reflections import generate_reflection, number_evidence
from memoryos.domain.grounding import check_reflection
from memoryos.domain.patterns import (
    DEFAULT_MIN_SUPPORT,
    REFLECTION_MIN_CONFIDENCE,
    clears_reflection_bar,
    pattern_confidence,
    support_needed_for_reflection,
)
from memoryos.domain.values import PatternKind, PatternRelation
from tests.support.fakes import FakeLanguageModel

DECIDED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def evidence(index: int, relation: PatternRelation) -> EvidenceRow:
    return EvidenceRow(
        decision_id=UUID(int=index),
        decision_question=f"Question {index}?",
        decided_at=DECIDED_AT,
        relation=relation,
        note=f"note {index}",
    )


def a_pattern(*, supporting: int, contradicting: int) -> PatternRow:
    return PatternRow(
        id=UUID(int=999),
        statement="You underestimate how long deployment takes.",
        kind=PatternKind.ASSUMPTION,
        detector="assumption_group",
        support_count=supporting,
        contradiction_count=contradicting,
        confidence=pattern_confidence(supporting, contradicting),
        first_observed=DECIDED_AT,
        last_observed=DECIDED_AT,
        discovered_at=DECIDED_AT,
        dismissed_at=None,
        dismissed_reason=None,
        supporting=[
            evidence(index, PatternRelation.SUPPORTS) for index in range(supporting)
        ],
        contradicting=[
            evidence(100 + index, PatternRelation.CONTRADICTS)
            for index in range(contradicting)
        ],
    )


# --------------------------------------------------------------------------
# The bar
# --------------------------------------------------------------------------


def test_the_reflection_bar_is_strictly_above_the_pattern_bar() -> None:
    """The whole reason there are two numbers.

    A pattern is a row read beside its evidence; a reflection is prose read as a
    claim. If these two were equal, everything that became a pattern would
    become a paragraph about somebody, and the riskier output would be governed
    by the weaker rule.
    """
    assert pattern_confidence(DEFAULT_MIN_SUPPORT, 0) < REFLECTION_MIN_CONFIDENCE
    # Derived rather than picked: one more agreeing decision than the floor.
    assert pytest.approx(0.667, abs=1e-3) == REFLECTION_MIN_CONFIDENCE


def test_a_pattern_with_no_confidence_recorded_is_not_a_near_miss() -> None:
    assert not clears_reflection_bar(None)


def test_what_a_weak_pattern_would_need_is_countable() -> None:
    # Two supporting against two contradicting: four more agreeing decisions,
    # with no further counter-evidence, reaches 6/2 = 0.75.
    assert support_needed_for_reflection(2, 2) == 4
    assert pattern_confidence(6, 2) >= REFLECTION_MIN_CONFIDENCE
    assert pattern_confidence(5, 2) < REFLECTION_MIN_CONFIDENCE
    # A pattern sitting exactly on the pattern bar needs one more.
    assert support_needed_for_reflection(3, 0) == 1


def test_no_threshold_above_the_ceiling_is_accepted() -> None:
    # `pattern_confidence` caps at 0.95, so a threshold above it would loop
    # forever looking for evidence that cannot exist.
    with pytest.raises(ValueError, match="ceiling"):
        support_needed_for_reflection(3, 0, threshold=0.99)


# --------------------------------------------------------------------------
# The golden case: a weak pattern produces nothing at all
# --------------------------------------------------------------------------


async def test_a_pattern_below_the_threshold_produces_no_reflection() -> None:
    """Two supporting, two contradicting. The milestone's acceptance criterion.

    Not a hedged reflection, not a short one, not one with a caveat — none. And
    the model is never called, which is the part that makes this a guarantee
    rather than a hope: there is no fluent paragraph anywhere in the process for
    anybody to be tempted by, and no prompt instruction a model under pressure to
    be helpful can smooth over.
    """
    model = FakeLanguageModel("You are overconfident about everything [1].")

    reflection = await generate_reflection(a_pattern(supporting=2, contradicting=2), model)

    assert reflection.text is None
    assert not reflection.written
    assert model.calls == []
    assert reflection.refused_because is not None
    assert "below" in reflection.refused_because
    # And it says what would change the answer, which is the output that makes
    # a refusal usable rather than a shrug.
    assert reflection.needed is not None
    assert "4 more decision(s)" in reflection.needed


async def test_a_pattern_at_the_pattern_bar_is_still_refused() -> None:
    # Three agreeing decisions with nothing against them is a pattern — it is
    # exactly the floor `is_emittable` allows — and it is still not enough to be
    # written about in prose.
    model = FakeLanguageModel("anything at all [1].")

    reflection = await generate_reflection(a_pattern(supporting=3, contradicting=0), model)

    assert reflection.confidence == pytest.approx(0.5)
    assert not reflection.written
    assert model.calls == []


async def test_a_pattern_above_the_bar_is_generated_and_verified() -> None:
    """The positive control. Without it every test here passes on a dead path."""
    model = FakeLanguageModel(
        "You underestimated the work in Question 0? and in Question 1? [1, 2]. "
        "The same belief held in Question 100? [6]."
    )

    # Five agreeing against one arguing back scores 0.69, just over the bar.
    reflection = await generate_reflection(a_pattern(supporting=5, contradicting=1), model)

    assert reflection.written
    assert reflection.check is not None
    assert reflection.check.citation_rate == 1.0
    assert reflection.check.grounded
    assert reflection.model_id == "fake/llm@1"
    # Counter-evidence was in the same numbered list, not a second one after it.
    assert [item.relation for item in reflection.citations] == [
        PatternRelation.SUPPORTS
    ] * 5 + [PatternRelation.CONTRADICTS]


# --------------------------------------------------------------------------
# What comes back
# --------------------------------------------------------------------------


async def test_an_uncited_sentence_is_flagged_and_the_reflection_is_kept() -> None:
    """Flagged, never removed.

    A sentence deleted from the middle of a paragraph leaves prose that reads as
    complete and is not, which is worse than prose admitting which part has
    nothing behind it.
    """
    model = FakeLanguageModel(
        "You underestimated Question 0? [1]. You are impatient by nature."
    )

    reflection = await generate_reflection(a_pattern(supporting=4, contradicting=0), model)

    assert reflection.written
    assert reflection.check is not None
    assert reflection.check.uncited == ["You are impatient by nature."]
    assert reflection.check.citation_rate == pytest.approx(0.5)
    assert not reflection.check.grounded
    # Still present in the text, and marked rather than cut.
    assert "impatient" in (reflection.text or "")
    assert "[uncited]" in reflection.check.marked()


async def test_a_citation_outside_the_evidence_is_rejected_outright() -> None:
    """Not flagged — rejected. Nothing is stored.

    An index the prompt never contained is unambiguous: the paragraph is
    describing somebody using evidence nobody showed it, and there is no
    charitable reading under which the rest of it is still trustworthy.
    """
    model = FakeLanguageModel("You underestimated the work [1]. And again [9].")

    reflection = await generate_reflection(a_pattern(supporting=4, contradicting=0), model)

    assert not reflection.written
    assert reflection.text is None
    assert reflection.check is not None
    assert reflection.check.out_of_evidence == [9]
    assert reflection.rejected_because is not None
    assert "[9]" in reflection.rejected_because


async def test_prose_that_cites_nothing_at_all_is_rejected() -> None:
    # The unfalsifiable behavioural claim in its purest form. Marking every
    # sentence would still leave every sentence on the screen.
    model = FakeLanguageModel("You are the sort of person who moves too fast.")

    reflection = await generate_reflection(a_pattern(supporting=4, contradicting=0), model)

    assert not reflection.written
    assert reflection.rejected_because is not None
    assert "nothing to check" in reflection.rejected_because


# --------------------------------------------------------------------------
# The numbering the citations are frozen against
# --------------------------------------------------------------------------


def test_evidence_is_numbered_per_decision_and_counter_evidence_is_in_the_list() -> None:
    numbered = number_evidence(a_pattern(supporting=3, contradicting=2))

    assert [item.marker for item in numbered] == [1, 2, 3, 4, 5]
    # Supporting first, then contradicting — but in one list, because a model
    # given "the evidence, and separately some caveats" writes the caveats last
    # if it writes them at all.
    assert [item.relation.value for item in numbered] == [
        "supports",
        "supports",
        "supports",
        "contradicts",
        "contradicts",
    ]


def test_two_evidence_rows_from_one_decision_share_one_number() -> None:
    """Otherwise a reflection citing both reads as two observations of a person.

    That is the mistake `DEFAULT_MIN_SUPPORT` exists to prevent — four
    assumptions from two decisions is two observations — reintroduced at the
    point where somebody actually reads the claim.
    """
    pattern = a_pattern(supporting=1, contradicting=0)
    pattern.supporting.append(
        EvidenceRow(
            decision_id=UUID(int=0),
            decision_question="Question 0?",
            decided_at=DECIDED_AT,
            relation=PatternRelation.SUPPORTS,
            note="a second belief from the same decision",
        )
    )

    numbered = number_evidence(pattern)

    assert len(numbered) == 1
    assert numbered[0].notes == ("note 0", "a second belief from the same decision")


# --------------------------------------------------------------------------
# The check itself
# --------------------------------------------------------------------------


def test_every_sentence_needs_a_citation_including_the_hedges() -> None:
    """The tightening over M2.6, and the one line that states it.

    `verify_citations` lets "the passages do not cover this" go uncited, because
    that sentence is a correct refusal rather than a claim. A reflection has no
    such sentence — the refusal happens before the model is called — so a hedge
    here is a claim about the strength of the evidence and cites the decisions
    that make it thin.
    """
    check = check_reflection(
        "There is no strong signal here. This is tentative [2].", {1, 2}
    )

    assert check.claims == 2
    assert check.uncited == ["There is no strong signal here."]
    assert check.citation_rate == pytest.approx(0.5)


def test_an_empty_reflection_scores_zero_rather_than_one() -> None:
    # `VerificationResult` scores an empty answer 1.0, because a refusal has
    # nothing to cite. Here there is no such case, and a generation that produced
    # nothing must not report the number a perfectly cited one would.
    check = check_reflection("   ", {1})

    assert check.citation_rate == 0.0
    assert not check.writable


def test_a_quoted_decision_question_does_not_split_a_sentence() -> None:
    """The reason this check has its own sentence splitter.

    The prompt asks for decisions to be named by their questions, and a question
    ends in a question mark. Under M2.6's splitter this fully-cited sentence
    becomes three fragments, two uncited, and the citation rate reports 33% for a
    reflection that did exactly what it was asked. A safety metric that reads
    worst when the system behaves best is worse than no metric.
    """
    check = check_reflection(
        "You underestimated the work in Should we use Celery or a table? [1]. "
        "The same belief held in Do we need a worker? [2].",
        {1, 2},
    )

    assert check.claims == 2
    assert check.uncited == []
    assert check.citation_rate == 1.0
