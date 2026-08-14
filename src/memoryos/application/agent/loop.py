"""One question, at most one tool call, one answer.

**The simplest loop that proves the tools work, and deliberately no more.** No
planning, no chaining, no second call on the strength of the first result: if the
model asks for two tools, the first is run and the rest are refused with a
sentence saying so. Multi-hop is M7.1, and building half of it here would mean
M7.1 inheriting a design nobody chose.

The shape is fixed and short enough to state completely:

1. the model is given the question and the six specs,
2. it either answers or asks for one tool,
3. the tool runs, and its result and the original call go back,
4. the model answers.

Two model calls at most, one tool call at most, and no state between questions.

### What comes back is the answer *and* its provenance

`AgentAnswer` carries the citations the tool produced, not because the loop uses
them but because the caller must be able to check the answer against them.
Phase 2 made every search result attributable and M2.6 made every generated
answer cite; an agent that read cited passages and returned an uncited paragraph
would undo both, quietly, in the layer above them.

The citations are the *tool's*, unedited. This loop does not try to work out
which of them the model actually used — that is a grounding question, M2.6 has
machinery for it, and guessing here would produce a citation list that looks
verified and is not.
"""

import time
from dataclasses import dataclass, field
from datetime import date

import structlog

from memoryos.application.agent.tools import ToolRegistry, ToolResult, UnknownTool
from memoryos.application.ports import ModelTurn, ToolCallingModel, ToolExchange
from memoryos.domain.citation import Citation

logger = structlog.get_logger(__name__)

# What the model is told about its job.
#
# **Short, and it says what not to do twice.** The two failure modes this
# milestone can produce are answering from training data instead of the corpus,
# and inventing a source for something a tool did not return — and both are
# instruction-shaped rather than architecture-shaped at this size. The guardrail
# that does not depend on instructions is that every tool result carries
# citations the caller can check.
_SYSTEM_TEMPLATE = """You answer questions about one person's private corpus: their files, \
notes, code and the decisions they recorded.

You have tools that read that corpus. Use one when the question is about what is \
IN the corpus. Answer directly, without a tool, only when the question is about \
you or about how to use this system.

Rules:
- Answer from what the tool returned. If it returned nothing useful, say so \
plainly rather than answering from general knowledge.
- Never invent a file name, a date or a quotation. If you name a source, it must \
be one the tool result showed you.
- If a result says it was truncated, say that your answer covers only part of \
what exists.
- Be brief. Two or three sentences unless the question needs more.

Today is {today}."""


def system_prompt(*, today: date | None = None) -> str:
    """The instructions, with today's date in them.

    **A temporal tool with no clock is broken for the questions it exists for.**
    Asked "what was I working on in August", the model routed to `query_timeline`
    correctly and then asked for August *2023* — its training prior, since
    nothing in the prompt said otherwise — and the tool answered honestly that
    the corpus holds nothing from then. Correct behaviour all the way down,
    producing a useless answer.

    One line fixes it, and it belongs here rather than in a tool description: the
    date is a fact about the conversation, not about any one tool, and putting it
    in six descriptions would be six places for it to go stale.
    """
    return _SYSTEM_TEMPLATE.format(today=(today or date.today()).isoformat())


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """What one question produced, and everything needed to check it."""

    question: str
    answer: str
    # The tool the model chose, or None when it answered without one. This is
    # the number M7.0 is judged on — routing — so it is on the result rather
    # than only in a log line.
    tool: str | None = None
    tool_arguments: dict[str, object] = field(default_factory=dict)
    citations: list[Citation] = field(default_factory=list)
    truncated: bool = False
    model_calls: int = 0
    duration_ms: int = 0

    @property
    def used_a_tool(self) -> bool:
        return self.tool is not None


class SingleCallAgent:
    """One turn, one tool, one answer."""

    def __init__(
        self,
        model: ToolCallingModel,
        registry: ToolRegistry,
        *,
        max_tokens: int = 700,
    ) -> None:
        self._model = model
        self._registry = registry
        self._max_tokens = max_tokens

    async def ask(self, question: str) -> AgentAnswer:
        """Answer one question, using at most one tool.

        `UnknownTool` is deliberately **not** caught. A model naming a tool that
        does not exist is not a question this loop can recover from by asking
        again — the specs it was given are the specs that exist — and it is a
        `PermanentError` so that a caller running this behind the worker gets one
        dead letter rather than five attempts at the same impossible name.
        """
        started = time.monotonic()
        instructions = system_prompt()
        first = await self._model.converse(
            instructions,
            question,
            tools=self._registry.specs(),
            max_tokens=self._max_tokens,
        )

        if not first.wants_tools:
            # Answered without a tool. Legitimate for "what can you do", and a
            # routing failure for anything about the corpus — which is why the
            # tool is reported as None rather than hidden.
            logger.info("agent.answered_without_tool", question=question)
            return AgentAnswer(
                question=question,
                answer=first.text,
                model_calls=1,
                duration_ms=_ms(started),
            )

        call = first.tool_calls[0]
        if len(first.tool_calls) > 1:
            # One call per turn is this milestone's whole scope. Logged rather
            # than silently dropped, because "the model wanted three" is the
            # measurement M7.1 starts from.
            logger.info(
                "agent.extra_calls_ignored",
                chosen=call.name,
                ignored=[other.name for other in first.tool_calls[1:]],
            )

        result = await self._call(call.name, call.arguments)
        second = await self._model.converse(
            instructions,
            question,
            tools=self._registry.specs(),
            exchanges=[ToolExchange(call=call, result=result.content)],
            max_tokens=self._max_tokens,
        )

        answer = second.text or _fallback(second, result)
        logger.info(
            "agent.answered",
            question=question,
            tool=call.name,
            citations=len(result.citations),
            truncated=result.truncated,
            answer_chars=len(answer),
        )
        return AgentAnswer(
            question=question,
            answer=answer,
            tool=call.name,
            tool_arguments=dict(call.arguments),
            citations=result.citations,
            truncated=result.truncated,
            model_calls=2,
            duration_ms=_ms(started),
        )

    async def _call(self, name: str, arguments: dict[str, object]) -> ToolResult:
        try:
            return await self._registry.call(name, dict(arguments))
        except UnknownTool:
            # Re-raised rather than turned into a result. See `ask`.
            logger.warning("agent.unknown_tool", tool=name)
            raise


def _fallback(turn: ModelTurn, result: ToolResult) -> str:
    """What to say when the second turn produced no prose.

    It happens: a model given tools again on the follow-up sometimes calls
    another one instead of answering, and this milestone has no second hop to
    give it. Returning the empty string would present silence as an answer, so
    the tool's own content is returned with a line saying it is unsummarised —
    which is honest, and still fully cited.
    """
    if turn.wants_tools:
        return (
            "The model asked for another tool instead of answering, which this "
            "milestone does not do — one call per question. What the first tool "
            f"returned:\n\n{result.content}"
        )
    return f"The model returned no answer. What the tool returned:\n\n{result.content}"


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
