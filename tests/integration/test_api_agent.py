"""`POST /agent/ask`: the answer, and the trajectory that produced it.

The planner is real and the *model* is scripted, which is the only split that
lets this run in a suite. A test that called a provider would be slow, would need
a key, and would fail for reasons unrelated to the endpoint — and on a free tier
would fail for the most boring reason of all.

What is asserted is the contract a client depends on: the steps are in the body
rather than behind a flag, and a trajectory that ended badly still comes back
with the hops that worked.
"""

from typing import Any

import pytest
from httpx import AsyncClient

from memoryos.application.agent.planner import MultiHopPlanner
from memoryos.application.agent.tools import ToolRegistry, ToolResult, ToolSpec, spec_for
from memoryos.application.agent.verify import VerifiedAgent
from memoryos.application.ports import ModelTurn, ToolCall
from memoryos.container import Container
from tests.unit.test_planner import Args, FakeCounter, citation
from tests.unit.test_verification import AngleEmbedder

pytestmark = pytest.mark.integration


class TwoHopModel:
    """Calls a tool once, then answers. The shortest real trajectory."""

    model_id = "fake/scripted@1"

    async def converse(
        self,
        system: str,
        user: str,
        *,
        tools: Any = (),
        exchanges: Any = (),
        max_tokens: int = 1024,
    ) -> ModelTurn:
        if exchanges:
            return ModelTurn(text=ANSWER, prompt_tokens=200, completion_tokens=12)
        return ModelTurn(
            text="I should look this up.",
            tool_calls=(
                ToolCall(id="c1", name="search_memories", arguments={"query": "lease"}),
            ),
            prompt_tokens=180,
            completion_tokens=9,
        )


class OneTool:
    arguments = Args

    @property
    def spec(self) -> ToolSpec:
        return spec_for(Args, name="search_memories", description="Find passages.")

    async def call(self, **kwargs: Any) -> ToolResult:
        return ToolResult(content=PASSAGE, citations=[citation(1)])


# The one sentence the scripted model answers with, and the one passage the
# scripted tool returns. Given the same angle they are identical, so the answer
# verifies as directly supported and the endpoint's shape — not the threshold —
# is what these tests are about.
ANSWER = "The lease is 30 seconds."
PASSAGE = "worker.py: a worker holds a lease on the job it claimed, 30 seconds long"


def scripted(model: Any, harness_sessions: Any) -> VerifiedAgent:
    registry = ToolRegistry()
    registry.register(OneTool())
    planner = MultiHopPlanner(model, registry, FakeCounter())
    return VerifiedAgent(
        planner, harness_sessions, AngleEmbedder({ANSWER: 0.0, PASSAGE: 0.0})
    )


@pytest.fixture
def scripted_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        Container,
        "agent",
        lambda self: scripted(TwoHopModel(), self.database.session_factory),
    )


async def test_the_steps_come_back_with_the_answer(
    client: AsyncClient, scripted_agent: None
) -> None:
    """**The trajectory is the artifact**, so it is in the body rather than
    behind a debug flag. A client given only the paragraph cannot tell four
    rewordings of one search from two dependent hops."""
    response = await client.post(
        "/agent/ask", json={"question": "how long is a lease", "max_hops": 4}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ANSWER
    assert body["raw_answer"] == ANSWER
    # **The verification block travels with the answer**, not behind a flag.
    checked = body["verification"]
    assert checked["verdict"] == "grounded"
    assert checked["support_rate"] == 1.0
    assert checked["refused"] is False
    assert [claim["support"] for claim in checked["claims"]] == ["direct"]
    assert body["stopped_because"] == "confidence"
    assert body["hops"] == 1
    assert [step["tool"] for step in body["steps"]] == ["search_memories", None]
    assert body["steps"][0]["args"] == {"query": "lease"}
    assert body["steps"][0]["citations"] == 1
    assert body["citations"][0]["external_key"] == "src/hop_1.py"
    # Both halves of the cost, because a client deciding whether to offer this
    # button needs the number the provider billed rather than a hop count.
    assert body["cost"]["model_calls"] == 2
    assert body["cost"]["prompt_tokens"] == 380


async def test_a_failed_trajectory_is_200_with_the_hops_that_worked(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A rate limit at hop two did not undo hop one.**

    500 would hide the retrievals that were already paid for, which on these free
    tiers is the difference between a partial result and nothing at all. The
    error is on the body, where a client can read it beside the steps.
    """
    from memoryos.domain.jobs import TransientError

    class FailsAfterOne(TwoHopModel):
        async def converse(self, *args: Any, **kwargs: Any) -> ModelTurn:
            if kwargs.get("exchanges"):
                raise TransientError("language model rate limited")
            return await super().converse(*args, **kwargs)

    monkeypatch.setattr(
        Container,
        "agent",
        lambda self: scripted(FailsAfterOne(), self.database.session_factory),
    )

    response = await client.post("/agent/ask", json={"question": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["stopped_because"] == "error"
    assert "rate limited" in body["error"]
    assert body["hops"] == 1
    assert len(body["citations"]) == 1


async def test_an_empty_question_is_refused_before_a_model_is_built(
    client: AsyncClient,
) -> None:
    """422 rather than a trajectory over whitespace. No container is touched, so
    this also holds on a deployment with no key."""
    response = await client.post("/agent/ask", json={"question": "   "})

    assert response.status_code == 422


async def test_an_ungrounded_answer_is_withheld_over_http_too(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The guardrail is on the agent, not on the CLI.**

    A refusal that only the terminal applied would be no refusal at all: this
    endpoint is what the web UI and the editor extension read, and they are the
    surfaces where a fluent unsupported paragraph would actually be believed.

    200 rather than 4xx. The system answered — with a refusal, which is a
    legitimate answer — and `verification.refused` is how a client tells the two
    apart without reading the prose.
    """

    class Fabricating(TwoHopModel):
        async def converse(self, *args: Any, **kwargs: Any) -> ModelTurn:
            if kwargs.get("exchanges"):
                return ModelTurn(
                    text=(
                        "Your architectural choices caused three production "
                        "incidents. The team changed its deployment process in "
                        "March. Your writing became more concise over three years."
                    ),
                    prompt_tokens=200,
                    completion_tokens=30,
                )
            return await super().converse(*args, **kwargs)

    monkeypatch.setattr(
        Container,
        "agent",
        lambda self: scripted(Fabricating(), self.database.session_factory),
    )

    response = await client.post("/agent/ask", json={"question": "anything"})

    assert response.status_code == 200
    body = response.json()
    assert body["verification"]["verdict"] == "ungrounded"
    assert body["verification"]["refused"] is True
    assert "could not answer" in body["answer"]
    # The draft is withheld from the wire, not merely marked on it.
    assert body["raw_answer"] is None
    assert "production incidents" not in body["answer"]
    # And the trajectory is still there: the hops happened and are inspectable.
    assert body["hops"] == 1


async def test_a_citation_that_does_not_resolve_is_reported(
    client: AsyncClient, scripted_agent: None
) -> None:
    """**M2.5's identity, checked against the database at answer time.**

    The scripted tool cites a memory id that was never ingested, which is the
    same shape as the real failure this guards: a citation whose offsets no
    longer match the stored text because the corpus moved under the answer.

    Reported rather than fatal. A drifted citation does not make the prose wrong,
    it makes the prose uncheckable — and the person reading it is the one who
    should decide what that is worth.
    """
    response = await client.post("/agent/ask", json={"question": "anything"})

    body = response.json()
    assert body["verification"]["unresolved_citations"] == [
        "self::src/hop_1.py#0 @0-10 (v1)"
    ]
