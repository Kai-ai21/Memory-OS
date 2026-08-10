"""Grounded answers over retrieved memories.

The response carries the verification result rather than a claim of
correctness. An answer that cited nothing still comes back — flagged — because
the caller needs to see what the system produced and how far it strayed, and a
500 would hide both.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from memoryos.adapters.llm.gemini import MissingApiKey
from memoryos.api.routes.search import CitationOut, SearchIn, _citation_out, build_filters
from memoryos.application.answering import GroundedAnswer
from memoryos.container import Container
from memoryos.domain.jobs import PermanentError, TransientError

router = APIRouter(tags=["answer"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class AnswerIn(SearchIn):
    """Everything a search takes, plus the generation budget.

    Inherited rather than redeclared: an answer is a search with a model on the
    end, and two definitions of "how do I filter" is how they drift apart.
    """

    max_tokens: int = 1024


class SentenceOut(BaseModel):
    text: str
    citations: list[int]
    factual: bool
    # True for a factual sentence with no citation. Flagged, never removed —
    # silently deleting a sentence mid-answer produces prose that reads
    # complete while missing a step.
    unsupported: bool


class VerificationOut(BaseModel):
    citation_rate: float
    factual_sentences: int
    supported_sentences: int
    # Indices the answer cited that were never supplied. Must be empty; a
    # non-empty list is the unambiguous hallucination signal.
    hallucinated_indices: list[int]
    cited_indices: list[int]
    is_refusal: bool
    grounded: bool
    sentences: list[SentenceOut] = Field(default_factory=list)


class ContextOut(BaseModel):
    passages: int
    dropped: int
    token_budget: int
    tokens_used: int


class AnswerTimingOut(BaseModel):
    retrieve_ms: int
    assemble_ms: int
    generate_ms: int
    verify_ms: int
    total_ms: int


class AnswerOut(BaseModel):
    question: str
    answer: str
    # The same answer with unsupported sentences marked, for a client that
    # would rather render the warning inline than compute it.
    marked_answer: str
    model_id: str
    context: ContextOut
    verification: VerificationOut
    # Resolved to full M2.5 citations, and only the passages the answer cited.
    citations: list[CitationOut] = Field(default_factory=list)
    timing: AnswerTimingOut


@router.post("/answer", response_model=AnswerOut)
async def answer(body: AnswerIn, container: ContainerDep) -> AnswerOut:
    filters = await build_filters(container, body)
    try:
        result = await container.answer()(
            body.q,
            k=max(1, min(body.k, 50)),
            filters=filters,
            max_tokens=body.max_tokens,
        )
    except MissingApiKey as exc:
        # 501 rather than 500: the deployment has no language model, which is a
        # configuration state rather than a fault, and search still works.
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from exc
    except TransientError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except PermanentError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    return _to_response(result)


def _to_response(result: GroundedAnswer) -> AnswerOut:
    verification = result.verification
    payload: dict[str, Any] = verification.as_dict()
    return AnswerOut(
        question=result.question,
        answer=result.answer,
        marked_answer=verification.marked(),
        model_id=result.model_id,
        context=ContextOut(**result.context.as_dict()),  # type: ignore[arg-type]
        verification=VerificationOut(
            citation_rate=float(payload["citation_rate"]),
            factual_sentences=int(payload["factual_sentences"]),
            supported_sentences=int(payload["supported_sentences"]),
            hallucinated_indices=list(verification.hallucinated_indices),
            cited_indices=sorted(set(verification.cited_indices)),
            is_refusal=verification.is_refusal,
            grounded=verification.grounded,
            sentences=[
                SentenceOut(
                    text=sentence.text,
                    citations=sentence.citations,
                    factual=sentence.factual,
                    unsupported=sentence.unsupported,
                )
                for sentence in verification.sentences
            ],
        ),
        citations=[
            _citation_out(citation)
            for explained in result.citations
            for citation in explained.citations
        ],
        timing=AnswerTimingOut(**result.timing.as_dict()),
    )
