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
from memoryos.application.ports import ModelTurn, ToolCall
from memoryos.container import Container
from tests.unit.test_planner import Args, FakeCounter, citation

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
            return ModelTurn(
                text="The lease is 30 seconds.",
                prompt_tokens=200,
                completion_tokens=12,
            )
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
        return ToolResult(content="worker.py: lease = 30s", citations=[citation(1)])


def scripted(model: Any) -> MultiHopPlanner:
    registry = ToolRegistry()
    registry.register(OneTool())
    return MultiHopPlanner(model, registry, FakeCounter())


@pytest.fixture
def scripted_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Container, "agent", lambda self: scripted(TwoHopModel()))


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
    assert body["answer"] == "The lease is 30 seconds."
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

    monkeypatch.setattr(Container, "agent", lambda self: scripted(FailsAfterOne()))

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
