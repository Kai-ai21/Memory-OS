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

**M10.0 added conversational context and changed nothing else.** `history` lets a
follow-up resolve what it refers to — "what about the other one?" is
unanswerable without the turn that named the first one — and it is deliberately
the weakest possible addition: the system prompt is untouched, the verification
is untouched, and with an empty history the prompt this builds is byte-identical
to the one it built before. The turns go into the question slot, labelled as not
being evidence, which keeps rule 1 governing them: only the numbered passages may
be cited, and a conversation is not a passage.

The turns reach the *retrieval query* only for a question that cannot stand
alone. That distinction was measured rather than designed: folding them in
unconditionally turned a question whose answer sat at ranks one and two into a
refusal. See `_retrieval_query`.
"""

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto

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
from memoryos.domain.message_intent import refers_back

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

# Turns of conversation carried into a follow-up.
#
# Three, and the number is a ceiling rather than a preference. Two turns back is
# where a pronoun's referent usually lives; five turns back is a different
# subject, and the cost of including it is not a longer prompt but a *diluted
# retrieval query* — the extra terms pull results towards a topic the question
# has moved on from, and the results still look like results. Drift in retrieval
# is the failure nobody can see.
DEFAULT_HISTORY_TURNS = 3

# How much of one turn is carried. A pasted page of notes is a legitimate turn
# and quoting all of it into the next question would spend the context budget on
# something that is already in the corpus and retrievable on its merits.
_TURN_CHARS = 400

CONVERSATION_TEMPLATE = """\
Earlier in this conversation. This is here so you can work out what the question \
refers to. It is not evidence, it is not a passage, and nothing in it may be \
cited:

{turns}

The question: {question}"""


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One earlier turn: what was typed, and what came back if anything.

    `answer` is null for a statement, which is most turns. Both halves are
    carried because both resolve references — a follow-up may point at something
    the person said or at something the corpus answered — but only `text` reaches
    the *retrieval* query. See `_retrieval_query`.
    """

    text: str
    answer: str | None = None

    def render(self) -> str:
        lines = [f"- typed: {_clip(self.text)}"]
        if self.answer:
            lines.append(f"  answered: {_clip(self.answer)}")
        return "\n".join(lines)


class EventKind(StrEnum):
    """What one event in a streamed answer is.

    The two retrieval events are the ones that would be easy to leave out and are
    the reason this streams at all. M10.0 measured the shape of the wait: embedding
    the query, searching, and reranking fifty candidates is *seven to eleven
    seconds* on this machine, and generation — the only part token streaming makes
    visible — is well under two. Streaming just the tokens would replace a
    ten-second blank screen with an eight-second blank screen.
    """

    # Sent before anything is embedded, so the first thing on screen arrives in
    # milliseconds rather than seconds.
    RETRIEVAL_STARTED = auto()
    # How much was searched and how much came back. A number makes the wait
    # legible: "searching 3,833 chunks" is a system working, and a spinner is a
    # system that might be broken.
    RETRIEVAL_DONE = auto()
    TOKEN = auto()
    CITATION = auto()
    # Carries the verification verdict and, when verification rejected the draft,
    # the refusal that replaces it. Always sent last on a successful stream.
    DONE = auto()
    ERROR = auto()


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One event, and whatever it carries.

    A kind and a dict rather than a class per event, because every one of these
    is serialised straight to SSE as JSON and a hierarchy would exist only to be
    flattened again at the transport. The kinds are closed; the payloads are what
    each kind documents.
    """

    kind: EventKind
    data: dict[str, object] = field(default_factory=dict)


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
        history: Sequence[ConversationTurn] = (),
    ) -> GroundedAnswer:
        started = time.monotonic()
        recent = tuple(history)[-DEFAULT_HISTORY_TURNS:]

        retrieve_started = time.monotonic()
        result = await self._search(_retrieval_query(question, recent), k=k, filters=filters)
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
            USER_TEMPLATE.format(
                passages=context.render(), question=_asked(question, recent)
            ),
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

    async def stream(
        self,
        question: str,
        *,
        k: int = 10,
        filters: SearchFilters | None = None,
        max_tokens: int = 1024,
        history: Sequence[ConversationTurn] = (),
    ) -> AsyncIterator[StreamEvent]:
        """The same answer, as events, in the order they become true.

        **Every guardrail runs, and runs where it ran before.** Retrieval,
        assembly, generation, verification, citation resolution — the same calls
        in the same order as `__call__`, with the only difference being that the
        generation step yields as it goes. In particular:

        * An empty context still refuses *without a model call*, before a single
          token is streamed. There is nothing to invent from and nothing is asked
          to.
        * **Verification runs on the joined text after the stream ends**, because
          it has to: a citation marker can arrive split across two chunks, and a
          per-chunk check would see `[` and `1]` and find neither. So tokens are
          streamed as a *draft*, and `done` carries the verdict.
        * When verification rejects the draft outright, `done` carries a
          `replacement` — the refusal — and the interface swaps the text it has
          been drawing. Visibly. An interface that quietly kept the draft would be
          showing text this system has decided not to stand behind.

        Errors are yielded as an `error` event rather than raised, because by the
        time one happens the caller has already sent an HTTP 200 and streamed some
        of a body. Raising would truncate the response and leave the client unable
        to tell a finished answer from a broken pipe — which is the whole reason
        step 4 exists.
        """
        started = time.monotonic()
        recent = tuple(history)[-DEFAULT_HISTORY_TURNS:]

        yield StreamEvent(EventKind.RETRIEVAL_STARTED, {"question": question, "k": k})

        retrieve_started = time.monotonic()
        try:
            result = await self._search(
                _retrieval_query(question, recent), k=k, filters=filters
            )
        except Exception as exc:
            yield StreamEvent(EventKind.ERROR, {"message": str(exc), "stage": "retrieval"})
            return
        retrieve_ms = _ms(retrieve_started)

        context = assemble_context(
            result.hits, counter=self._counter, token_budget=self._token_budget
        )
        yield StreamEvent(
            EventKind.RETRIEVAL_DONE,
            {
                # What was actually found, not what was searched. The corpus size
                # belongs to `/stats` and the interface already has it; reporting
                # it from here would be this layer guessing at a number another
                # one owns.
                "hits": len(result.hits),
                "chunks": sum(len(hit.matched_chunks) for hit in result.hits),
                "passages": len(context.passages),
                "dropped": len(context.dropped),
                "retrieve_ms": retrieve_ms,
                "rerank_ms": result.timing.rerank_ms,
            },
        )

        if context.is_empty:
            # The strongest form of the guardrail, and it survives streaming
            # unchanged: nothing was retrieved, so nothing is asked, so there is
            # nothing that could have been invented.
            verification = verify_citations(REFUSAL_WITHOUT_CONTEXT, set())
            yield StreamEvent(EventKind.TOKEN, {"text": REFUSAL_WITHOUT_CONTEXT})
            yield StreamEvent(
                EventKind.DONE,
                _done_payload(
                    REFUSAL_WITHOUT_CONTEXT, verification, [], self._model.model_id,
                    total_ms=_ms(started),
                ),
            )
            return

        pieces: list[str] = []
        generate_started = time.monotonic()
        try:
            async for piece in self._model.stream(
                SYSTEM_PROMPT,
                USER_TEMPLATE.format(
                    passages=context.render(), question=_asked(question, recent)
                ),
                max_tokens=max_tokens,
            ):
                pieces.append(piece)
                yield StreamEvent(EventKind.TOKEN, {"text": piece})
        except Exception as exc:
            # Partial text has already been drawn. The event says so, and the
            # interface marks what it has rather than leaving it looking finished.
            yield StreamEvent(
                EventKind.ERROR,
                {"message": str(exc), "stage": "generation", "partial": "".join(pieces)},
            )
            return

        answer = "".join(pieces)
        verification = verify_citations(answer, context.valid_indices)
        citations = await self._resolve(context, verification)

        for explained in citations:
            for citation in explained.citations[:1]:
                yield StreamEvent(
                    EventKind.CITATION,
                    {
                        "memory_id": str(citation.memory_id),
                        "locator": citation.locator,
                        "excerpt": " ".join(citation.excerpt.split()),
                    },
                )

        logger.info(
            "answer.streamed",
            question_length=len(question),
            passages=len(context.passages),
            citation_rate=round(verification.citation_rate, 3),
            hallucinated=len(verification.hallucinated_indices),
            refused=verification.is_refusal,
            generate_ms=_ms(generate_started),
        )
        yield StreamEvent(
            EventKind.DONE,
            _done_payload(
                answer, verification, citations, self._model.model_id,
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


def _done_payload(
    answer: str,
    verification: VerificationResult,
    citations: Sequence[ExplainedHit],
    model_id: str,
    *,
    total_ms: int,
) -> dict[str, object]:
    """What `done` carries, including the replacement when there is one.

    `replacement` is the mechanism M10.3 asks for by name: a draft that
    verification rejects is not quietly kept. It is set when the answer cited an
    index it was never given, which is the unambiguous fabrication signal — every
    other verification result is a matter of degree and is reported as a mark on
    the sentence rather than a reason to discard the whole answer.
    """
    replacement = None
    if verification.hallucinated_indices:
        replacement = (
            "This answer cited passages that were never retrieved "
            f"({', '.join(str(index) for index in verification.hallucinated_indices)}), "
            "so it has been withdrawn. Nothing here supports it."
        )
    return {
        "answer": answer,
        "replacement": replacement,
        "marked_answer": verification.marked(),
        "model_id": model_id,
        "refused": verification.is_refusal,
        "grounded": verification.grounded,
        "citation_rate": round(verification.citation_rate, 4),
        "hallucinated_indices": list(verification.hallucinated_indices),
        "citations": len(citations),
        "total_ms": total_ms,
    }


def _retrieval_query(question: str, history: Sequence[ConversationTurn]) -> str:
    """What is actually searched for.

    **The conversation is folded in only when the question cannot stand without
    it**, and that condition was added after measuring the alternative. Folding
    three turns into every query cost more than it bought: "why did I use
    external_key instead of a memory id on the transcript?" asked with unrelated
    turns attached returned passages the model declined to answer from, and the
    identical question asked alone put the two thoughts that answer it at ranks
    one and two. See `domain.message_intent.refers_back`.

    When it does fold in, it folds in what was *typed* and never what was
    answered. That asymmetry is the same rule that keeps answers out of the
    corpus, applied one step earlier: an answer is generated prose, and letting
    generated prose steer the next retrieval is how a conversation converges on
    whatever the model said first rather than on what the corpus holds. The
    typed turns are a person's own words and are the only thing that can resolve
    "the other one".
    """
    if not history or not refers_back(question):
        return question
    return " ".join([*(_clip(turn.text) for turn in history), question])


def _asked(question: str, history: Sequence[ConversationTurn]) -> str:
    """The question as the model sees it.

    Byte-identical to `question` with no history, which is what keeps every
    existing test and every existing measurement comparable across this change.
    """
    if not history:
        return question
    return CONVERSATION_TEMPLATE.format(
        turns="\n".join(turn.render() for turn in history), question=question
    )


def _clip(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _TURN_CHARS:
        return collapsed
    return collapsed[:_TURN_CHARS].rstrip() + "…"


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
