"""The four properties the guardrail rests on, against a stub embedder.

Every threshold in `verify.py` is a cosine similarity measured on a real model,
and a test that loaded that model would take ten seconds to assert something the
model is not what is under test for. So the embedder here returns vectors chosen
to sit on the near or far side of the thresholds, and what is asserted is the
*decision* the module makes from them.

* a citation to a hop the trajectory does not have is rejected,
* an unsupported factual sentence is flagged and still present,
* a connective sentence is not asked to cite,
* below the support threshold, the answer is replaced by a refusal.

`scripts/calibrate_verification.py` is where the thresholds themselves are
measured, on the real embedder and the real corpus. Two different questions, two
different instruments; conflating them would produce a test that fails when the
corpus grows.
"""

import math
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from memoryos.application.agent.planner import Step, StopReason, Trajectory
from memoryos.application.agent.tools import ToolResult
from memoryos.application.agent.verify import (
    DIRECT,
    REFUSAL,
    Support,
    VerificationResult,
    presented,
    verify,
)
from memoryos.domain.citation import Citation


class AngleEmbedder:
    """Vectors on a circle, so similarity is an angle a test can state.

    A claim and a passage are given the same "topic" by being handed the same
    angle; `_cosine` then returns exactly cos(difference). That makes a test say
    "this claim is 0.9 similar to that passage" rather than "these two float
    lists happen to score above a constant", which is the difference between a
    test that documents the rule and one that pins an accident.
    """

    def __init__(self, angles: dict[str, float]) -> None:
        self._angles = angles

    def _vectors(self, texts: Sequence[str]) -> list[list[float]]:
        return [
            [math.cos(self._angles.get(text, 3.0)), math.sin(self._angles.get(text, 3.0))]
            for text in texts
        ]

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        return self._vectors(texts)

    def embed_passage(self, texts: Sequence[str]) -> list[list[float]]:
        return self._vectors(texts)


def citation(number: int) -> Citation:
    return Citation(
        memory_id=UUID(int=number),
        source_name="self",
        external_key=f"src/hop_{number}.py",
        chunk_ordinal=0,
        char_start=0,
        char_end=10,
        prefix_chars=0,
        excerpt=f"excerpt {number}",
        definition=None,
        occurred_at=None,
        version=1,
    )


def step(content: str, *, truncated: bool = False, cite: int = 1) -> Step:
    return Step(
        thought="",
        tool="search_memories",
        args={"query": "x"},
        result=ToolResult(
            content=content, citations=[citation(cite)], truncated=truncated
        ),
    )


def trajectory(answer: str, *steps: Step) -> Trajectory:
    return Trajectory(
        question="q",
        steps=list(steps),
        answer=answer,
        stopped_because=StopReason.CONFIDENCE,
    )


def aligned(*texts: str) -> AngleEmbedder:
    """Everything named here is identical; everything else is far away."""
    return AngleEmbedder({text: 0.0 for text in texts})


# A passage long enough to survive `MIN_UNIT_CHARS`. Every unit in these tests
# has to clear it, which is itself worth having encoded once.
PASSAGE = (
    "The worker takes a lease on a job and the lease expires after thirty "
    "seconds without a heartbeat from the holder."
)


# --------------------------------------------------------------------------
# 1. A citation to a hop that does not exist
# --------------------------------------------------------------------------


def test_a_citation_to_a_nonexistent_step_is_rejected() -> None:
    """**The one failure detectable with certainty**, and M2.6's rule carried
    over: the model referenced a retrieval that never happened, and no reading of
    the surrounding sentence makes that acceptable.

    It caps the verdict at `partial` however well the rest scores, because an
    answer containing one has shown it will invent provenance.
    """
    claim = "The lease expires after thirty seconds [4]."
    result = verify(
        trajectory(claim, step(PASSAGE)),
        aligned(claim, PASSAGE),
    )

    assert result.invalid_citations == [4]
    assert result.verdict == "partial"
    # The claim itself is supported — the content was retrieved, the attribution
    # was invented. Reporting those as one failure would lose both.
    assert result.support_rate == 1.0
    assert result.claims[0].supported


def test_every_hop_in_one_bracket_is_checked_not_only_the_first() -> None:
    """**Found by running it.** A real answer over a one-hop trajectory wrote
    `[2, 3]`; with only the singular `cited_step`, hop 2 was reported invalid and
    hop 3 went unmentioned.

    The unambiguous failure is the one number that must never be under-counted —
    it is the whole reason M2.6 has it — so integrity reads every index and
    `cited_step` keeps M7.2's shape by naming the first.
    """
    claim = "The system stops regenerating dismissed patterns [2, 3]."
    result = verify(trajectory(claim, step(PASSAGE)), aligned(claim, PASSAGE))

    assert result.invalid_citations == [2, 3]
    assert result.claims[0].cited_steps == (2, 3)
    assert result.claims[0].cited_step == 2


def test_a_citation_to_a_hop_that_exists_is_not_rejected() -> None:
    claim = "The lease expires after thirty seconds [1]."
    result = verify(trajectory(claim, step(PASSAGE)), aligned(claim, PASSAGE))

    assert result.invalid_citations == []
    assert result.verdict == "grounded"
    assert result.claims[0].cited_step == 1


def test_a_cited_step_that_was_truncated_is_reported_as_lower_confidence() -> None:
    """Not a failure and not folded into the rate: the model may be citing
    something it only partly saw, and that is a reason to trust the sentence
    less rather than to reject it."""
    claim = "The lease expires after thirty seconds [1]."
    result = verify(
        trajectory(claim, step(PASSAGE, truncated=True)),
        aligned(claim, PASSAGE),
    )

    assert result.truncated_citations == [1]
    assert result.claims[0].from_truncated
    assert result.verdict == "grounded"


# --------------------------------------------------------------------------
# 2. Unsupported claims are flagged, not removed
# --------------------------------------------------------------------------


def test_an_unsupported_factual_sentence_is_flagged_and_still_present() -> None:
    """**Silent removal produces prose that reads as complete and is not.**

    M2.6's rule, and the reason is unchanged by there being six results instead
    of one: a reader shown four sentences cannot tell that a fifth was deleted,
    and a reader shown five with one marked can.
    """
    supported = "The lease expires after thirty seconds."
    invented = "The team agreed to change this at the March retrospective."
    answer = f"{supported} {invented}"

    result = verify(
        trajectory(answer, step(PASSAGE)),
        AngleEmbedder({supported: 0.0, PASSAGE: 0.0, invented: 2.0}),
    )

    assert [claim.text for claim in result.unsupported] == [invented]
    assert result.claims[1].support is Support.UNSUPPORTED
    # Present in the returned text, and marked.
    marked = result.marked(answer)
    assert invented in marked
    assert marked.endswith("[unsupported]")
    assert supported in marked
    # Half supported is exactly the threshold, so this is `partial` rather than
    # withheld — the boundary is worth pinning because it is where the milestone's
    # acceptance criterion actually sits.
    assert result.support_rate == 0.5
    assert result.verdict == "partial"


def test_two_results_bearing_on_a_claim_neither_states_is_inferred() -> None:
    """**Legitimate for multi-hop, and weaker — so it is reported separately.**

    Nothing here checks that the model's combination of the two is valid. That is
    the honest limit of this instrument, and `direct_rate` is what makes it
    visible: an answer built mostly of inference is one reasoning past its
    evidence.
    """
    claim = "The lease and the reclaim interval were chosen together."
    near = math.acos(DIRECT) * 1.05  # inside INFERRED, outside DIRECT
    first = "A worker takes a lease on the job it claims, and holds it by heartbeat."
    second = "Reclaiming an expired job is what returns it to the queue for others."

    result = verify(
        trajectory(claim, step(first, cite=1), step(second, cite=2)),
        AngleEmbedder({claim: 0.0, first: near, second: near}),
    )

    assert result.claims[0].support is Support.INFERRED
    assert result.claims[0].steps == (1, 2)
    assert result.support_rate == 1.0
    assert result.direct_rate == 0.0


def test_one_weak_resemblance_to_one_result_is_not_support() -> None:
    """The other side of the same rule. A single below-threshold likeness to one
    passage is what a plausible fabrication looks like from here, and calling it
    `inferred` would let the level launder exactly the sentences it exists to
    catch."""
    claim = "The lease and the reclaim interval were chosen together."
    near = math.acos(DIRECT) * 1.05

    result = verify(
        trajectory(claim, step(PASSAGE)),
        AngleEmbedder({claim: 0.0, PASSAGE: near}),
    )

    assert result.claims[0].support is Support.UNSUPPORTED


# --------------------------------------------------------------------------
# 3. Connective sentences need no citation
# --------------------------------------------------------------------------


def test_a_connective_sentence_is_not_required_to_cite() -> None:
    """Demanding evidence for "In summary:" would put scaffolding in the flagged
    list and drag the support rate down for a well-behaved answer."""
    answer = (
        "Here are the decisions I found. "
        "The lease expires after thirty seconds. "
        "In summary, the queue is a table."
    )
    supported = "The lease expires after thirty seconds."

    result = verify(
        trajectory(answer, step(PASSAGE)),
        AngleEmbedder({supported: 0.0, PASSAGE: 0.0}),
    )

    factual = [claim.text for claim in result.claims if claim.factual]
    assert factual == [supported]
    assert result.connective_claims == 2
    assert result.unsupported == []
    assert result.support_rate == 1.0
    # And they are not marked in the output, which is the visible half.
    assert "[unsupported]" not in result.marked(answer)


def test_a_statement_of_absence_is_connective_rather_than_an_uncited_claim() -> None:
    """**The behaviour this whole phase most wants must not score worst.**

    "The corpus does not contain this" is the correct answer to most of the
    adversarial questions. Requiring it to cite something would rate the honest
    reply below the confident one — the exact inversion every grounding check in
    this project is arranged against.
    """
    answer = (
        "I could not find any decisions marked as reversed. "
        "The corpus does not contain a record of production incidents."
    )

    result = verify(trajectory(answer, step(PASSAGE)), aligned(PASSAGE))

    assert result.factual_claims == 0
    assert result.support_rate == 1.0
    assert result.verdict == "grounded"


# --------------------------------------------------------------------------
# 4. Below the threshold, the answer is not returned
# --------------------------------------------------------------------------


def test_a_negation_buried_in_a_claim_does_not_excuse_it_from_citing() -> None:
    """**The connective escape hatch is the one to keep narrow**, because a
    sentence that takes it is never checked against anything.

    Found by running it. This sentence — a real one, with a quotation in it —
    matched `not … finding` ninety characters in and was classified as a
    statement of absence, so a claim about Jaccard similarity was excused from
    needing any support. A sentence that is *about* an absence says so first.
    """
    claim = (
        "There's also a mechanism using Jaccard similarity to prevent re-showing "
        '"the same list with an edit, not a new finding".'
    )

    result = verify(
        trajectory(claim, step(PASSAGE)),
        AngleEmbedder({claim: 2.0, PASSAGE: 0.0}),
    )

    assert result.factual_claims == 1
    assert result.unsupported and result.unsupported[0].text == claim
    # And the genuine article still passes, matched inside the window.
    honest = "The corpus does not contain a record of production incidents."
    assert verify(trajectory(honest, step(PASSAGE)), aligned(PASSAGE)).factual_claims == 0


def test_below_threshold_support_returns_a_refusal_rather_than_the_answer() -> None:
    """**The milestone's acceptance criterion**, and the same one M2.6 set.

    Three invented sentences and one real: the draft is fluent, cites a real
    trajectory, and rests on almost nothing. What the caller gets is the refusal;
    the draft stays on the trajectory so a trace can still show what was
    withheld, which is the difference between withholding and hiding.
    """
    real = "The lease expires after thirty seconds."
    answer = " ".join(
        [
            real,
            "Your architectural choices caused three production incidents.",
            "Your writing became more concise over three years.",
            "The team changed its deployment process in March.",
        ]
    )
    angles: dict[str, Any] = {real: 0.0, PASSAGE: 0.0}

    result = verify(trajectory(answer, step(PASSAGE)), AngleEmbedder(angles))
    withheld = trajectory(answer, step(PASSAGE))

    assert result.support_rate == 0.25
    assert result.verdict == "ungrounded"
    assert not result.returnable
    assert presented(withheld, result) == REFUSAL
    # The draft is still there to be inspected, and is not what a caller shows.
    assert withheld.answer == answer


def test_a_trajectory_that_retrieved_nothing_supports_nothing() -> None:
    """Zero hops is the `ANSWERED` stop M7.1 measured, and an answer written over
    no results is the case this check exists for."""
    answer = "Your decision-making has become steadily more risk-averse."

    result = verify(trajectory(answer), AngleEmbedder({}))

    assert result.support_rate == 0.0
    assert result.verdict == "ungrounded"


def test_an_error_trajectory_is_not_given_a_verdict_it_cannot_have() -> None:
    """No answer means nothing to be ungrounded about, and `stopped_because`
    already says why. Inventing a verdict here would overwrite it."""
    empty = Trajectory(
        question="q", steps=[], answer=None, stopped_because=StopReason.ERROR
    )

    result = verify(empty, AngleEmbedder({}))

    assert result == VerificationResult()
    assert presented(empty, result) == REFUSAL


def test_a_refusal_the_model_wrote_itself_is_returned_rather_than_replaced() -> None:
    """An answer that is entirely connective has no factual claims, rates 1.0 by
    the empty convention, and passes. The system should not refuse on behalf of a
    model that already did."""
    answer = "I could not find anything in the corpus about production incidents."
    made = trajectory(answer, step(PASSAGE))

    result = verify(made, aligned(PASSAGE))

    assert result.returnable
    assert presented(made, result) == answer
