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

**An ungrounded answer is withheld here too, and 200 is the right code for it.**
The system answered — with a refusal, which is a legitimate answer and the one
this phase most wants — and `verification.refused` says so. A 4xx would tell a
client its request was wrong when what happened is that the corpus could not
support a reply.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.api.routes.search import CitationOut, _citation_out
from memoryos.application.agent.verify import Claim, VerifiedAnswer
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


class ClaimOut(BaseModel):
    text: str
    sentence_index: int
    cited_step: int | None = None
    # Every hop the sentence named. `cited_step` is the first of these, kept for
    # the shape M7.2 specifies; integrity is checked against all of them.
    cited_steps: list[int] = Field(default_factory=list)
    supported: bool
    support_excerpt: str | None = None
    # direct | inferred | unsupported. A client rendering the answer needs the
    # level and not only the boolean: an inferred claim is the model's own
    # combination of two results and is worth showing differently.
    support: str
    similarity: float = 0.0
    steps: list[int] = Field(default_factory=list)
    # False for connective sentences — "In summary", "I could not find any" —
    # which need no citation and must not be rendered as unsupported.
    factual: bool = True
    from_truncated: bool = False


class VerificationOut(BaseModel):
    support_rate: float
    direct_rate: float
    verdict: str
    factual_claims: int
    connective_claims: int
    claims: list[ClaimOut] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    truncated_citations: list[int] = Field(default_factory=list)
    unresolved_citations: list[str] = Field(default_factory=list)
    # True when an answer was drafted and withheld. The client shows the refusal
    # in `answer`; `raw_answer` is null in that case, deliberately.
    refused: bool = False


class AgentAskOut(BaseModel):
    question: str
    # **What the caller may show.** Already marked where sentences are
    # unsupported, or already replaced by the refusal — never the raw draft.
    answer: str | None
    # The unmarked text, and null when the answer was withheld. A client that
    # wanted to render its own marking reads `verification.claims` instead.
    raw_answer: str | None = None
    verification: VerificationOut
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
    return _to_response(await agent.ask(body.question, max_hops=hops))


def _to_response(verified: VerifiedAnswer) -> AgentAskOut:
    trajectory = verified.trajectory
    checked = verified.verification
    return AgentAskOut(
        question=trajectory.question,
        answer=None if trajectory.answer is None else verified.answer,
        raw_answer=None if verified.refused else trajectory.answer,
        verification=VerificationOut(
            support_rate=round(checked.support_rate, 4),
            direct_rate=round(checked.direct_rate, 4),
            verdict=checked.verdict,
            factual_claims=checked.factual_claims,
            connective_claims=checked.connective_claims,
            claims=[_claim_out(claim) for claim in checked.claims],
            invalid_citations=list(checked.invalid_citations),
            truncated_citations=list(checked.truncated_citations),
            unresolved_citations=list(checked.unresolved_citations),
            refused=verified.refused,
        ),
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


def _claim_out(claim: Claim) -> ClaimOut:
    return ClaimOut(
        text=claim.text,
        sentence_index=claim.sentence_index,
        cited_step=claim.cited_step,
        cited_steps=list(claim.cited_steps),
        supported=claim.supported,
        support_excerpt=claim.support_excerpt,
        support=claim.support.value,
        similarity=claim.similarity,
        steps=list(claim.steps),
        factual=claim.factual,
        from_truncated=claim.from_truncated,
    )


def _summary(content: str, limit: int = 600) -> str:
    return content if len(content) <= limit else content[: limit - 1] + "…"
