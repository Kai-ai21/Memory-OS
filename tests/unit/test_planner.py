"""The four properties the loop rests on, against a model that cannot surprise it.

Termination is the part of M7.1 that is hard, and it is hard in a way live calls
cannot test: a loop that runs forever is one you find out about by paying for it.
So the model here is scripted. It returns exactly the turns each test needs —
including the pathological ones a real model produces rarely and expensively — and
every assertion is about what the *loop* did with them.

* the hop limit stops a loop that would otherwise run forever,
* two consecutive no-new-information steps stop it earlier than the limit,
* compaction keeps the citations of findings the model can no longer see,
* a tool that raises comes back as something the model can act on.

No network, no database, no embedder. The tools are fakes because what is under
test is orchestration; `tests/integration/test_agent_tools.py` is where the real
six are exercised.
"""

from typing import Any

from pydantic import BaseModel

from memoryos.application.agent.compaction import compact
from memoryos.application.agent.planner import (
    STALE_LIMIT,
    MultiHopPlanner,
    Step,
    StopReason,
    summarise,
)
from memoryos.application.agent.tools import ToolRegistry, ToolResult, ToolSpec, spec_for
from memoryos.application.ports import ModelTurn, ToolCall
from memoryos.domain.citation import Citation


class Args(BaseModel):
    query: str = ""


class ScriptedModel:
    """A `ToolCallingModel` that reads from a list instead of from a network.

    The last entry repeats forever, which is what makes the hop-limit test a test
    of the limit: the model never volunteers a stop, so anything that ends the
    run came from the loop.

    A call with no tools offered is the loop's forced final answer, and this
    answers it rather than reading the script. A model handed no tools cannot ask
    for one, so a script that returned a tool call there would be testing a turn
    no provider can produce.
    """

    model_id = "fake/scripted@1"

    def __init__(self, turns: list[ModelTurn]) -> None:
        self._turns = turns
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.exchanges: list[int] = []
        self.offered: list[int] = []

    async def converse(
        self,
        system: str,
        user: str,
        *,
        tools: Any = (),
        exchanges: Any = (),
        max_tokens: int = 1024,
    ) -> ModelTurn:
        self.systems.append(system)
        self.prompts.append(user)
        self.exchanges.append(len(exchanges))
        self.offered.append(len(tools))
        if not tools:
            return answers()
        index = min(len(self.prompts) - 1, len(self._turns) - 1)
        return self._turns[index]


class CountingTool:
    """Returns something different every call, so nothing is ever stale."""

    arguments: type[BaseModel] = Args

    def __init__(self, name: str = "search_memories") -> None:
        self._name = name
        self.calls = 0

    @property
    def spec(self) -> ToolSpec:
        return spec_for(Args, name=self._name, description="Find passages.")

    async def call(self, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(
            content=f"result {self.calls}", citations=[citation(self.calls)]
        )


class RepeatingTool:
    """Returns the same thing every call, whatever it is asked."""

    arguments: type[BaseModel] = Args

    @property
    def spec(self) -> ToolSpec:
        return spec_for(Args, name="search_memories", description="Find passages.")

    async def call(self, **kwargs: Any) -> ToolResult:
        # Whitespace differs between calls on purpose: the rule compares
        # *information*, and two renderings of one result that differ by a line
        # break are the same information. A raw hash would call them different
        # and the loop would never notice it was going in circles.
        return ToolResult(content=f"the  same\nthing {' ' * len(kwargs)}")


class BrokenTool:
    arguments: type[BaseModel] = Args

    @property
    def spec(self) -> ToolSpec:
        return spec_for(Args, name="search_memories", description="Find passages.")

    async def call(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("the vector store is not there")


class FakeCounter:
    """Four characters to a token, which is close enough for a budget test.

    The real counter is the embedder's, and loading it here would spend seconds
    and a HuggingFace round trip to make a division more accurate.
    """

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def citation(number: int) -> Citation:
    from uuid import UUID

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


def registry_of(tool: Any) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def wants(name: str = "search_memories", **arguments: Any) -> ModelTurn:
    return ModelTurn(
        text="Looking further.",
        tool_calls=(ToolCall(id="call_1", name=name, arguments=arguments),),
        prompt_tokens=100,
        completion_tokens=10,
    )


def answers(text: str = "Here is the answer.") -> ModelTurn:
    return ModelTurn(text=text, prompt_tokens=100, completion_tokens=20)


def planner(model: ScriptedModel, tool: Any, **kwargs: Any) -> MultiHopPlanner:
    return MultiHopPlanner(model, registry_of(tool), FakeCounter(), **kwargs)


# --------------------------------------------------------------------------
# 1. The hop limit
# --------------------------------------------------------------------------


async def test_the_hop_limit_stops_a_loop_that_would_run_forever() -> None:
    """**The only condition that catches a model which never stops asking.**

    This one is not a fabrication: the scripted model wants a tool on every
    single turn, and each call returns something genuinely new, so neither of the
    other two conditions can fire. Without the limit this test does not fail, it
    hangs — which is exactly the failure mode in production, where it would hang
    while spending quota.
    """
    tool = CountingTool()
    model = ScriptedModel([wants(query="again")])

    trajectory = await planner(model, tool).run("what did I miss", max_hops=3)

    assert trajectory.stopped_because is StopReason.HOP_LIMIT
    assert trajectory.hops == 3
    assert tool.calls == 3
    # Three hops and the forced final answer: the loop does not simply stop, it
    # asks for what it has. A trajectory that ended with no answer would be a
    # bound reported as a failure.
    assert trajectory.model_calls == 4
    assert trajectory.answer == "Here is the answer."
    # Tools are withdrawn for that last call. Offering them to a model that has
    # just run out of hops invites the one response the loop cannot use.
    assert model.offered[-1] == 0
    assert trajectory.tokens == 3 * 110 + 120


async def test_the_limit_is_the_callers_when_it_gives_one() -> None:
    """`--max-hops` has to reach the loop, and one is a legal answer."""
    tool = CountingTool()
    trajectory = await planner(
        ScriptedModel([wants(query="again")]), tool, max_hops=6
    ).run("anything", max_hops=1)

    assert trajectory.hops == 1
    assert trajectory.stopped_because is StopReason.HOP_LIMIT


# --------------------------------------------------------------------------
# 2. No new information
# --------------------------------------------------------------------------


async def test_two_stale_hops_stop_the_loop_before_the_limit() -> None:
    """**The condition that catches a loop which is technically progressing.**

    New queries, new arguments, same results. The hop limit would catch it too,
    six hops later and six model calls poorer, which is the whole reason this
    exists as a separate condition.

    Two consecutive, not one: a model re-reading a result before it pivots is
    normal, and stopping on the first repeat would cut off the pivot.
    """
    model = ScriptedModel([wants(query="a"), wants(query="b"), wants(query="c")])

    trajectory = await planner(model, RepeatingTool()).run("go", max_hops=6)

    assert trajectory.stopped_because is StopReason.NO_NEW_INFORMATION
    # Hop 1 is novel by definition; hops 2 and 3 are not, and the second of those
    # is the one that stops it.
    assert trajectory.hops == 1 + STALE_LIMIT
    assert [step.novel for step in trajectory.steps] == [True, False, False]
    assert trajectory.answer is not None


async def test_one_repeat_is_not_enough_to_stop() -> None:
    """The other half of the same rule, and the one that would be easy to lose.

    A loop that stopped on a single repeat would end the moment a model checked
    something twice, which is a normal thing to do before changing direction.
    """

    class OnceRepeating:
        arguments: type[BaseModel] = Args

        def __init__(self) -> None:
            self.calls = 0

        @property
        def spec(self) -> ToolSpec:
            return spec_for(Args, name="search_memories", description="Find.")

        async def call(self, **kwargs: Any) -> ToolResult:
            self.calls += 1
            # first, first again, then something new
            return ToolResult(content="first" if self.calls <= 2 else "second")

    trajectory = await planner(
        ScriptedModel([wants(query="x")]), OnceRepeating()
    ).run("go", max_hops=4)

    assert trajectory.stopped_because is StopReason.HOP_LIMIT
    assert [step.novel for step in trajectory.steps] == [True, False, True, False]


async def test_a_model_that_stops_on_its_own_is_recorded_as_having_done_so() -> None:
    """The third condition, and the only one that can stop at the right time.

    `CONFIDENCE` rather than `ANSWERED`, because it retrieved first. The
    distinction is not cosmetic: M7.3 must not score an answer standing on two
    retrievals beside one standing on none.
    """
    model = ScriptedModel([wants(query="x"), answers("Because of the lease.")])

    trajectory = await planner(model, CountingTool()).run("why", max_hops=6)

    assert trajectory.stopped_because is StopReason.CONFIDENCE
    assert trajectory.hops == 1
    assert trajectory.answer == "Because of the lease."
    # No forced final call: it already answered.
    assert trajectory.model_calls == 2


async def test_an_answer_that_retrieved_nothing_is_not_the_same_event() -> None:
    """Zero hops, so nothing in the corpus backed this paragraph."""
    trajectory = await planner(
        ScriptedModel([answers("I am an agent over your corpus.")]), CountingTool()
    ).run("what can you do", max_hops=6)

    assert trajectory.stopped_because is StopReason.ANSWERED
    assert trajectory.hops == 0
    assert trajectory.citations == []


# --------------------------------------------------------------------------
# 3. Compaction keeps provenance
# --------------------------------------------------------------------------


async def test_compaction_keeps_the_citations_of_findings_it_compacted() -> None:
    """**A finding that loses its provenance is a fabrication waiting to happen.**

    By hop five the model is writing about material it can no longer see. If the
    compacted form dropped the locators, every sentence resting on hop one would
    be uncited by construction — fluent, corpus-derived and impossible to check.

    Asserted on the prompt the model actually received, because that is the only
    place the property has any effect. A test against `Compacted.findings` would
    pass whether or not the locators were rendered.
    """
    tool = CountingTool()
    model = ScriptedModel([wants(query="hop")])

    trajectory = await planner(model, tool).run("what happened", max_hops=5)

    # By the last hop, only the two most recent results are still replayed in
    # full; hops 1 to 3 have been compacted into findings. `[-2]` is that hop —
    # `[-1]` is the forced final call, which replays nothing.
    assert model.exchanges[-2] == 2
    for prompt in (model.prompts[-2], model.prompts[-1]):
        assert "hop 1 · search_memories" in prompt
        # The locator, in the prompt, for a result the model can no longer see.
        assert citation(1).locator in prompt, prompt
        assert "sources:" in prompt

    # The separate guarantee: the trajectory carries every citation, including
    # from hops the model could no longer read when it answered.
    assert [c.locator for c in trajectory.citations] == [
        citation(number).locator for number in range(1, 6)
    ]


def test_a_finding_that_does_not_fit_is_dropped_whole_and_counted() -> None:
    """M2.6's rule: whole items, never a sentence cut in half.

    And the model is told, in words, that the history it is reading is not
    everything it found — a model reasoning over a silently shortened history
    reports a partial search as a complete one.
    """
    steps = [
        Step(
            thought="looking",
            tool="search_memories",
            args={"query": f"q{number}"},
            result=ToolResult(content="x" * 400, citations=[citation(number)]),
        )
        for number in range(1, 6)
    ]

    compacted = compact(steps, counter=FakeCounter(), budget=140)

    assert compacted.dropped
    assert compacted.verbatim == steps[-2:]
    # Nothing was cut mid-finding: every kept finding still ends with its
    # sources line.
    for finding in compacted.findings:
        assert finding.locators
    assert "dropped for space" in compacted.render()
    assert compacted.tokens <= 140


def test_hop_numbers_survive_a_dropped_finding() -> None:
    """Renumbering to close the gap would tell the model it made fewer calls
    than it did, which is the one thing a history must never do."""
    steps = [
        Step(
            thought="",
            tool="search_memories",
            args={"query": f"q{number}"},
            result=ToolResult(content="y" * 600),
        )
        for number in range(1, 7)
    ]

    compacted = compact(steps, counter=FakeCounter(), budget=200)

    assert compacted.dropped
    assert compacted.verbatim_from == 5
    # The kept findings keep their original hop numbers rather than 1..n.
    assert [finding.hop for finding in compacted.findings] != [
        index + 1 for index in range(len(compacted.findings))
    ]


# --------------------------------------------------------------------------
# 4. A tool that fails is a message, not an exception
# --------------------------------------------------------------------------


async def test_a_tool_error_reaches_the_model_as_something_it_can_act_on() -> None:
    """**Not raised.** M7.0 let this propagate and was right to: with one call
    there was no turn left in which the model could do anything about it.

    With hops there is. The failure becomes a sentence naming what broke and
    saying it is not the arguments' fault, the run continues, and the trajectory
    records the whole thing — so a person reading it afterwards sees the fault
    rather than an answer that mysteriously ignored one tool.
    """
    model = ScriptedModel([wants(query="x"), answers("I could not look that up.")])

    trajectory = await planner(model, BrokenTool()).run("anything", max_hops=4)

    assert trajectory.stopped_because is StopReason.CONFIDENCE
    step = trajectory.steps[0]
    assert step.result is not None
    assert "the vector store is not there" in step.result.content
    assert "RuntimeError" in step.result.content
    # The model saw it, which is the half that matters. It is in the replayed
    # exchange rather than only in the trajectory.
    assert trajectory.answer == "I could not look that up."
    assert model.exchanges[1] == 1


async def test_an_unknown_tool_name_is_also_correctable_now() -> None:
    """The same change of stance, for the other failure M7.0 could not recover
    from: with five hops left, naming a tool that does not exist is a mistake
    the next hop fixes, and the message carries the names that do."""
    model = ScriptedModel([wants(name="do_the_thing"), answers()])

    trajectory = await planner(model, CountingTool()).run("anything", max_hops=4)

    step = trajectory.steps[0]
    assert step.result is not None
    assert "no tool named 'do_the_thing'" in step.result.content
    assert "search_memories" in step.result.content


async def test_only_the_first_of_several_calls_in_one_turn_runs() -> None:
    """A hop is one decision. A model asking for three at once has guessed three
    independent retrievals, which is the thing this milestone replaces — and it
    is told so, rather than having two calls silently dropped."""
    tool = CountingTool()
    turn = ModelTurn(
        text="",
        tool_calls=(
            ToolCall(id="a", name="search_memories", arguments={"query": "one"}),
            ToolCall(id="b", name="search_memories", arguments={"query": "two"}),
        ),
    )
    model = ScriptedModel([turn, answers()])

    trajectory = await planner(model, tool).run("anything", max_hops=4)

    assert tool.calls == 1
    step = trajectory.steps[0]
    assert step.result is not None
    assert "one at a time" in step.result.content


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


async def test_a_provider_failure_keeps_the_hops_that_worked() -> None:
    """**A rate limit at hop three did not undo hops one and two.**

    On these free tiers a trajectory ending in a 429 is the expected case rather
    than the exceptional one, and raising would throw away the retrievals that
    had already been paid for.
    """
    from memoryos.domain.jobs import TransientError

    class FailsThird(ScriptedModel):
        async def converse(self, *args: Any, **kwargs: Any) -> ModelTurn:
            if len(self.prompts) >= 2:
                raise TransientError("language model rate limited: try again in 23s")
            return await super().converse(*args, **kwargs)

    model = FailsThird([wants(query="x")])

    trajectory = await planner(model, CountingTool()).run("anything", max_hops=6)

    assert trajectory.stopped_because is StopReason.ERROR
    assert trajectory.answer is None
    assert trajectory.error is not None and "rate limited" in trajectory.error
    assert trajectory.hops == 2
    assert len(trajectory.citations) == 2


def test_stop_reasons_are_counted_for_the_report() -> None:
    """If `HOP_LIMIT` dominates, the loop is not converging — which is a finding
    about the design and not a detail of one run."""
    from memoryos.application.agent.planner import Trajectory

    def ended(reason: StopReason) -> Trajectory:
        return Trajectory(question="q", steps=[], answer="a", stopped_because=reason)

    counted = summarise(
        [
            ended(StopReason.HOP_LIMIT),
            ended(StopReason.HOP_LIMIT),
            ended(StopReason.CONFIDENCE),
        ]
    )

    assert counted["hop_limit"] == 2
    assert counted["confidence"] == 1
    # Every reason present, including the zeroes: a distribution that omitted
    # the reasons that never fired would read as if they could not.
    assert set(counted) == {reason.value for reason in StopReason}
