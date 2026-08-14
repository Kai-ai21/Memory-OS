"""The three properties the trajectory metrics rest on.

Every metric here is a claim about reasoning, and a metric that flatters the
thing it audits is worse than no metric at all — so what these check is that the
three most gameable ones say the *low* number when the low number is true.

* information gain is zero for a hop that returned what an earlier hop returned,
* dependency is zero for a query that ignored the previous result,
* efficiency compares against the golden minimum and does not reward beating it.

No model, no network, no database. `Trajectory` was built to carry everything
these need, which is the property being relied on rather than tested.
"""

from uuid import UUID

from memoryos.application.agent.evaluate import (
    Failure,
    GoldenAgentQuestion,
    Report,
    TrajectoryScore,
    score,
    variance,
)
from memoryos.application.agent.planner import Step, StopReason, Trajectory
from memoryos.application.agent.tools import ToolResult
from memoryos.domain.citation import Citation


def citation(number: int) -> Citation:
    return Citation(
        memory_id=UUID(int=number),
        source_name="self",
        external_key=f"src/file_{number}.py",
        chunk_ordinal=0,
        char_start=0,
        char_end=10,
        prefix_chars=0,
        excerpt=f"excerpt {number}",
        definition=None,
        occurred_at=None,
        version=1,
    )


def step(
    tool: str,
    *,
    args: dict[str, object] | None = None,
    content: str = "a result",
    memories: tuple[int, ...] = (),
    thought: str = "",
    novel: bool = True,
) -> Step:
    return Step(
        thought=thought,
        tool=tool,
        args=args or {"query": "x"},
        result=ToolResult(
            content=content, citations=[citation(number) for number in memories]
        ),
        novel=novel,
    )


def trajectory(*steps: Step, stopped: StopReason = StopReason.CONFIDENCE) -> Trajectory:
    return Trajectory(
        question="q",
        steps=list(steps),
        answer="An answer.",
        stopped_because=stopped,
    )


GOLDEN = GoldenAgentQuestion(
    id="test", question="q", required_tools=("search_memories",), min_hops=2
)


# --------------------------------------------------------------------------
# 1. Information gain
# --------------------------------------------------------------------------


def test_a_hop_returning_only_seen_results_gains_nothing() -> None:
    """**A hop that cost a model call and moved nothing.**

    This is the wasted-hop signal, and it has to be zero rather than small: an
    agent rewording a search until something sticks produces three hops of
    identical memories, and a metric that scored that 0.3 would let the waste
    average away against the hops that worked.
    """
    result = score(
        trajectory(
            step("search_memories", memories=(1, 2, 3)),
            step("search_memories", memories=(1, 2, 3)),
        ),
        GOLDEN,
    )

    assert [hop.gain for hop in result.per_hop] == [1.0, 0.0]
    assert result.information_gain == 0.5


def test_partly_new_results_gain_the_part_that_is_new() -> None:
    """Four seen and one unseen is a fact the agent did not have, and a rule that
    scored it zero would stop a working loop for being repetitive."""
    result = score(
        trajectory(
            step("search_memories", memories=(1, 2, 3, 4)),
            step("search_memories", memories=(1, 2, 3, 4, 5)),
        ),
        GOLDEN,
    )

    assert result.per_hop[1].gain == 0.2


def test_a_result_with_no_citations_falls_back_to_the_loops_own_verdict() -> None:
    """"No silences of 30 days or more" cites nothing and is a real result. The
    loop already decided whether it was new, against the rendered text; deciding
    it a second way here would let the two disagree."""
    fresh = score(trajectory(step("find_gaps", novel=True)), GOLDEN)
    stale = score(trajectory(step("find_gaps", novel=False)), GOLDEN)

    assert fresh.per_hop[0].gain == 1.0
    assert stale.per_hop[0].gain == 0.0


# --------------------------------------------------------------------------
# 2. Dependency
# --------------------------------------------------------------------------


def test_a_query_that_ignored_the_previous_result_scores_zero() -> None:
    """**The metric that tells multi-hop reasoning from repeated retrieval.**

    Both produce several hops, several calls and a fluent answer. The difference
    is whether hop 2 was written from hop 1's output — and rewording the question
    is not writing from anything, which is why the question's own terms are
    excluded from what counts as borrowed.
    """
    result = score(
        trajectory(
            step("search_memories", args={"query": "repeated mistakes"},
                 content="worker.py holds a lease on a claimed job"),
            step("search_memories", args={"query": "mistakes I have repeated"}),
        ),
        GoldenAgentQuestion(
            id="test", question="what mistakes have I repeated", min_hops=2
        ),
    )

    assert result.per_hop[1].dependency == 0.0
    assert result.dependency == 0.0


def test_an_id_taken_from_the_previous_result_is_the_strong_signal() -> None:
    """A memory id appears nowhere but in a result, and no model invents a
    matching UUID. It is the one piece of evidence the agent could only have got
    by reading what came back."""
    found = str(UUID(int=7))
    result = score(
        trajectory(
            step("get_decisions", content=f"Evidence: 1 linked memories, ids: {found}"),
            step("get_memory", args={"memory_id": found}),
        ),
        GOLDEN,
    )

    assert result.per_hop[1].dependency == 1.0
    assert "id " in result.per_hop[1].dependency_evidence


def test_a_term_borrowed_from_the_result_is_the_weak_signal() -> None:
    """Half marks, and the exclusion is what makes it mean anything: a word
    shared with the previous result *and absent from the question* could only
    have come from the result."""
    result = score(
        trajectory(
            step("search_memories", args={"query": "background work"},
                 content="the queue is claimed with SKIP LOCKED under Postgres"),
            step("search_memories", args={"query": "skip locked semantics"}),
        ),
        GoldenAgentQuestion(id="test", question="how is background work run", min_hops=2),
    )

    assert result.per_hop[1].dependency == 0.5
    assert "locked" in result.per_hop[1].dependency_evidence


def test_the_first_hop_is_not_counted_against_the_dependency_mean() -> None:
    """It has nothing to depend on. Counting it zero would cap a perfect two-hop
    trajectory at 0.5 and make the metric unreadable at exactly the length most
    of these questions run to."""
    found = str(UUID(int=3))
    result = score(
        trajectory(
            step("get_decisions", content=f"ids: {found}"),
            step("get_memory", args={"memory_id": found}),
        ),
        GOLDEN,
    )

    assert result.dependency == 1.0


# --------------------------------------------------------------------------
# 3. Efficiency
# --------------------------------------------------------------------------


def test_efficiency_is_the_golden_minimum_over_hops_taken() -> None:
    two = GoldenAgentQuestion(id="test", question="q", min_hops=2)

    assert score(trajectory(step("a"), step("b")), two).efficiency == 1.0
    assert score(trajectory(step("a"), step("b"), step("c")), two).efficiency == 2 / 3
    assert score(
        trajectory(step("a"), step("b"), step("c"), step("d")), two
    ).efficiency == 0.5


def test_beating_the_minimum_does_not_score_above_one() -> None:
    """**Finishing in fewer hops than the minimum is not efficiency.**

    It means the minimum was wrong or the answer skipped a step it needed, and
    letting it score 2.0 would let one badly-set question carry the mean — which
    is the exact failure mode that makes a benchmark stop being evidence.
    """
    three = GoldenAgentQuestion(id="test", question="q", min_hops=3)

    result = score(trajectory(step("a")), three)

    assert result.efficiency == 1.0
    # And the run is still marked as having stopped short, which is where the
    # cost of the short trajectory is actually reported.
    assert result.failure is Failure.STOPPED_EARLY


def test_a_trajectory_with_no_hops_is_zero_rather_than_undefined() -> None:
    result = score(trajectory(), GOLDEN)

    assert result.efficiency == 0.0
    assert result.hops == 0


# --------------------------------------------------------------------------
# The taxonomy, and the category that is not the agent's fault
# --------------------------------------------------------------------------


def test_a_question_the_corpus_cannot_answer_is_not_an_agent_failure() -> None:
    """**The reason the taxonomy exists.**

    With sixteen decisions and a corpus of source code, several questions cannot
    be answered by any agent. Counting those as agent failures would send the
    next milestone off to fix the loop when what is missing is the data.
    """
    unanswerable = GoldenAgentQuestion(
        id="test", question="q", min_hops=1, answerable=False
    )

    refused = score(
        trajectory(step("search_memories")), unanswerable, refused=True, verdict="ungrounded"
    )
    asserted = score(
        trajectory(step("search_memories")), unanswerable, verdict="grounded"
    )

    assert refused.failure is Failure.INSUFFICIENT_DATA
    assert asserted.failure is Failure.WRONG_CONCLUSION

    report = Report(scores=(refused,))
    assert report.agent_failures == 0


def test_a_forbidden_tool_is_reported_before_anything_downstream() -> None:
    """A run that used the wrong tool *and* stopped early is a tool problem: the
    description that routed it there is the thing to fix, and the early stop is
    plausibly a consequence."""
    golden = GoldenAgentQuestion(
        id="test", question="q", forbidden_tools=("find_gaps",), min_hops=3
    )

    result = score(trajectory(step("find_gaps")), golden)

    assert result.failure is Failure.WRONG_TOOL
    assert result.forbidden_used == ("find_gaps",)


def test_a_provider_failure_is_not_scored_as_reasoning() -> None:
    """A rate limit is not the agent stopping too early. Putting quota in the
    taxonomy would make the counts unreadable on a free tier, which is every run
    this project has made."""
    dead = Trajectory(
        question="q", steps=[], answer=None, stopped_because=StopReason.ERROR
    )

    result = score(dead, GOLDEN)

    assert result.failure is Failure.INSUFFICIENT_DATA
    assert result.termination == 0.0


def test_tool_appropriateness_is_undefined_rather_than_zero_without_narration() -> None:
    """Most steps carry no narration. Scoring silence as failure would make this
    a measurement of how chatty a provider is."""
    quiet = score(trajectory(step("search_memories")), GOLDEN)
    spoken = score(
        trajectory(step("get_decisions", thought="Let me check why that was chosen.")),
        GOLDEN,
    )
    wrong = score(
        trajectory(step("find_gaps", thought="Let me check why that was chosen.")),
        GOLDEN,
    )

    assert quiet.per_hop[0].appropriate is None
    assert quiet.judgeable == 0
    assert spoken.per_hop[0].appropriate is True
    assert wrong.per_hop[0].appropriate is False


# --------------------------------------------------------------------------
# Variance
# --------------------------------------------------------------------------


def test_variance_reports_the_spread_per_metric_and_per_question() -> None:
    """**A stable mean can hide two questions swinging in opposite directions**,
    which is the shape non-determinism actually takes here: one different tool
    choice at hop one changes every hop after it."""

    def report(overall: float) -> Report:
        return Report(
            scores=(
                TrajectoryScore(
                    question_id="a",
                    hops=1,
                    min_hops=1,
                    tool_appropriateness=0.0,
                    judgeable=0,
                    information_gain=overall,
                    dependency=overall,
                    efficiency=overall,
                    termination=overall,
                    failure=Failure.NONE,
                    stopped_because="confidence",
                ),
            )
        )

    spread = variance([report(0.4), report(0.9), report(0.7)])

    assert spread["overall"]["range"] == 0.5
    assert spread["per_question_overall"]["a"] == 0.5
    # One run is not a floor, and reporting one as though it were would be the
    # worst possible version of this number.
    assert variance([report(0.4)]) == {}
