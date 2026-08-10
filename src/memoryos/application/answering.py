"""Retrieve, assemble, generate, verify.

The prompt below is the only place in this system where behaviour is requested
rather than enforced, and every line of it defends against one specific way a
grounded answer goes wrong. It is also, on its own, insufficient — which is why
`domain/grounding.py` checks the output afterwards and the response carries the
result of that check rather than a promise.

**Passages before the question.** Long-context models attend better to material
that precedes the instruction, and the instruction here is the part that must
not be forgotten: the request to refuse competes with every passage that looks
vaguely relevant.
"""

import time
from dataclasses import dataclass, field

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.application.citations import ExplainedHit, explain_hits
from memoryos.application.context import (
    DEFAULT_TOKEN_BUDGET,
    AssembledContext,
    assemble_context,
)
from memoryos.application.ports import LanguageModel, SearchFilters, TokenCounter
from memoryos.application.search import FusionWeights, SearchMemories
from memoryos.domain.fusion import DEFAULT_RRF_K
from memoryos.domain.grounding import VerificationResult, verify_citations

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
You answer questions using only the numbered passages you are given.

Rules, in order of importance:

1. Use ONLY the numbered passages. Do not use anything you know from training. \
If the passages are about a different system than the question assumes, say so.
2. Cite every factual claim with the passage number it came from, like [1] or \
[2]. Put the marker at the end of the sentence it supports. A sentence with a \
fact and no marker is a failure.
3. If the passages do not contain the answer, say plainly that they do not, and \
stop. Do not assemble a plausible answer from general knowledge. Do not guess. \
A short "the passages do not cover this" is a correct and valuable answer.
4. Do not speculate, extrapolate, or describe what a system like this one \
usually does. Only what these passages say.
5. Prefer the passages' own wording to a paraphrase. Where a passage names \
something exactly — a function, a column, a SQL clause — use that name.

Write plain prose. No preamble, no restating the question, no summary of what \
you are about to do."""

USER_TEMPLATE = """\
{passages}

---

Answer this question using only the passages above, citing each claim with its \
passage number: {question}"""

REFUSAL_WITHOUT_CONTEXT = (
    "The retrieved passages do not contain anything about this, so there is "
    "nothing here to answer from."
)


@dataclass(frozen=True, slots=True)
class AnswerTiming:
    retrieve_ms: int = 0
    assemble_ms: int = 0
    generate_ms: int = 0
    verify_ms: int = 0
    total_ms: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "retrieve_ms": self.retrieve_ms,
            "assemble_ms": self.assemble_ms,
            "generate_ms": self.generate_ms,
            "verify_ms": self.verify_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    question: str
    answer: str
    model_id: str
    context: AssembledContext
    verification: VerificationResult
    # The passages the answer actually cited, resolved to full M2.5 citations.
    # Only cited ones: listing everything retrieved would present passages the
    # answer never used as though they supported it.
    citations: list[ExplainedHit] = field(default_factory=list)
    timing: AnswerTiming = field(default_factory=AnswerTiming)

    @property
    def refused(self) -> bool:
        return self.verification.is_refusal

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "marked_answer": self.verification.marked(),
            "model_id": self.model_id,
            "context": self.context.as_dict(),
            "verification": self.verification.as_dict(),
            "timing": self.timing.as_dict(),
        }


class AnswerQuestion:
    """The whole grounded-answer path, as one use case."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        search: SearchMemories,
        model: LanguageModel,
        counter: TokenCounter,
        *,
        weights: FusionWeights | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._sessions = session_factory
        self._search = search
        self._model = model
        self._counter = counter
        self._weights = weights or FusionWeights()
        self._token_budget = token_budget
        self._rrf_k = rrf_k

    async def __call__(
        self,
        question: str,
        *,
        k: int = 10,
        filters: SearchFilters | None = None,
        max_tokens: int = 1024,
    ) -> GroundedAnswer:
        started = time.monotonic()

        retrieve_started = time.monotonic()
        result = await self._search(question, k=k, filters=filters)
        retrieve_ms = _ms(retrieve_started)

        assemble_started = time.monotonic()
        context = assemble_context(
            result.hits, counter=self._counter, token_budget=self._token_budget
        )
        assemble_ms = _ms(assemble_started)

        if context.is_empty:
            # Nothing retrieved, so there is nothing to answer from and no
            # reason to spend a generation finding that out. Refusing here is
            # the same answer the model would give, arrived at for free and
            # without the opportunity to invent one.
            logger.info("answer.no_context", question_length=len(question))
            return GroundedAnswer(
                question=question,
                answer=REFUSAL_WITHOUT_CONTEXT,
                model_id=self._model.model_id,
                context=context,
                verification=verify_citations(REFUSAL_WITHOUT_CONTEXT, set()),
                timing=AnswerTiming(
                    retrieve_ms=retrieve_ms,
                    assemble_ms=assemble_ms,
                    total_ms=_ms(started),
                ),
            )

        generate_started = time.monotonic()
        answer = await self._model.complete(
            SYSTEM_PROMPT,
            USER_TEMPLATE.format(passages=context.render(), question=question),
            max_tokens=max_tokens,
        )
        generate_ms = _ms(generate_started)

        verify_started = time.monotonic()
        verification = verify_citations(answer, context.valid_indices)
        verify_ms = _ms(verify_started)

        citations = await self._resolve(context, verification)

        logger.info(
            "answer.finished",
            question_length=len(question),
            passages=len(context.passages),
            citation_rate=round(verification.citation_rate, 3),
            hallucinated=len(verification.hallucinated_indices),
            refused=verification.is_refusal,
        )
        return GroundedAnswer(
            question=question,
            answer=answer,
            model_id=self._model.model_id,
            context=context,
            verification=verification,
            citations=citations,
            timing=AnswerTiming(
                retrieve_ms=retrieve_ms,
                assemble_ms=assemble_ms,
                generate_ms=generate_ms,
                verify_ms=verify_ms,
                total_ms=_ms(started),
            ),
        )

    async def _resolve(
        self, context: AssembledContext, verification: VerificationResult
    ) -> list[ExplainedHit]:
        """The cited passages, as full M2.5 citations.

        Hallucinated indices resolve to nothing by construction: they are not in
        the context, so there is no hit to look up. The verification result
        names them; this returns only what genuinely exists.
        """
        cited = sorted(set(verification.cited_indices))
        hits = [
            passage.hit
            for number in cited
            if (passage := context.passage(number)) is not None
        ]
        if not hits:
            return []
        return await explain_hits(
            self._sessions, hits, weights=self._weights, rrf_k=self._rrf_k
        )


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
