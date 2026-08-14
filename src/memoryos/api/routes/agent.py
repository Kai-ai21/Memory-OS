"""Multi-hop answers, with the trajectory that produced them.

**The steps are in the response body, not behind a debug flag.** They are the
artifact — M7.1 says so and M7.3 will score them — and a client that received
only the paragraph would have no way to tell four rewordings of one search from
five dependent hops. Both produce the same fluent prose; only the steps differ.

A trajectory that ended in `ERROR` still comes back 200 with its steps and a null
answer, for the same reason `/answer` returns an ungrounded answer flagged rather
than a 500: a rate limit at hop five did not undo hops one to four, and hiding
them behind a status code loses the only evidence of what the run had found. The
error statuses below are for the failures that produce no trajectory at all —
there is no model, or the configured one cannot call tools.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.api.routes.search import CitationOut, _citation_out
from memoryos.application.agent.planner import Trajectory
from memoryos.container import Container, ToolsUnsupported

router = APIRouter(tags=["agent"])

# The same clamp `/search` puts on `k`, for the same reason: a bound the caller
# cannot exceed belongs on the server, because the cost of twenty hops is paid
# here in quota and latency rather than there.
MAX_HOPS = 12


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class AgentAskIn(BaseModel):
    question: str
    # None means the deployment's configured default rather than "no limit":
    # there is no such thing as no limit here, and a body that could ask for one
    # would be a way to spend somebody else's quota.
    max_hops: int | None = None


class StepOut(BaseModel):
    thought: str
    tool: str | None
    args: dict[str, object] = Field(default_factory=dict)
    # A summary rather than the whole result. The full text of six tool results
    # is most of a context window, and a client that wants a passage has the
    # citations to fetch it with.
    result: str | None = None
    citations: int = 0
    truncated: bool = False
    # False when this step returned only content an earlier step had already
    # returned — the signal behind a `no_new_information` stop.
    novel: bool = True
    tokens: int = 0
    duration_ms: int = 0


class CostOut(BaseModel):
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    duration_ms: int


class AgentAskOut(BaseModel):
    question: str
    # Null only when the trajectory ended in `error`, in which case `error` says
    # why and the steps say how far it got.
    answer: str | None
    stopped_because: str
    hops: int
    steps: list[StepOut] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    cost: CostOut
    truncated: bool = False
    error: str | None = None
    retry_after: float | None = None


@router.post("/agent/ask", response_model=AgentAskOut)
async def ask(body: AgentAskIn, container: ContainerDep) -> AgentAskOut:
    if not body.question.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "question is empty")
    try:
        agent = container.agent()
    except MissingApiKey as exc:
        # 501 rather than 500: no language model is a configuration state, not a
        # fault, and everything else in this API still works.
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from exc
    except ToolsUnsupported as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from exc

    hops = None if body.max_hops is None else max(1, min(body.max_hops, MAX_HOPS))
    return _to_response(await agent.run(body.question, max_hops=hops))


def _to_response(trajectory: Trajectory) -> AgentAskOut:
    return AgentAskOut(
        question=trajectory.question,
        answer=trajectory.answer,
        stopped_because=trajectory.stopped_because.value,
        hops=trajectory.hops,
        steps=[
            StepOut(
                thought=step.thought,
                tool=step.tool,
                args=dict(step.args),
                result=None if step.result is None else _summary(step.result.content),
                citations=0 if step.result is None else len(step.result.citations),
                truncated=step.result.truncated if step.result else False,
                novel=step.novel,
                tokens=step.tokens,
                duration_ms=step.duration_ms,
            )
            for step in trajectory.steps
        ],
        citations=[_citation_out(citation) for citation in trajectory.citations],
        cost=CostOut(
            model_calls=trajectory.model_calls,
            prompt_tokens=trajectory.prompt_tokens,
            completion_tokens=trajectory.completion_tokens,
            duration_ms=trajectory.duration_ms,
        ),
        truncated=trajectory.truncated,
        error=trajectory.error,
        retry_after=trajectory.retry_after,
    )


def _summary(content: str, limit: int = 600) -> str:
    return content if len(content) <= limit else content[: limit - 1] + "…"
