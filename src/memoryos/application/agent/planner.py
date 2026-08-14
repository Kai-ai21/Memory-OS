"""Several dependent retrievals, each shaped by what the last one returned.

M7.0 proved the tools work. This is the part that makes them worth having:
questions that cannot be embedded into one useful query, because the query you
need second depends on what came back first.

    What mistakes have I repeated?

There is no vector that finds that. Answering it means finding decisions whose
outcome went badly, reading what each of them assumed, noticing that two
assumptions are the same mistake in different words, checking that they belong to
independent decisions rather than one decision recorded twice, and gathering the
evidence. Five retrievals, each one written from the previous one's output. That
is orchestration over Phase 2, not a replacement for it — every hop is an
existing use case behind an existing schema.

### The trajectory is the primary artifact

Not the answer. `Trajectory` carries every step — what the model said it was
doing, which tool it called, with what arguments, and what came back — and the
answer is one field on it. Two reasons, and the second is the load-bearing one:

* M7.3 scores trajectories, and a scorer handed only the final paragraph can
  measure fluency and nothing else.
* **You cannot debug a bad answer without seeing how it was reached.** A
  multi-hop agent fails in ways that are invisible at the output: it searches the
  same thing four times with reworded queries, it takes one weak hit and spends
  the rest of its budget elaborating on it, it answers from the question rather
  than from the corpus. All three produce a confident paragraph. Only the
  trajectory tells them apart.

### Termination, which is the genuinely hard part

Too early gives thin answers; too late burns quota and drifts into synthesis.
Three conditions, all of them, because **each one fails on its own**:

* A **hard hop limit** catches loops, and nothing else does. A model that has
  decided to keep searching will keep searching politely and forever.
* **No new information** catches the loop that is technically progressing — new
  queries, new arguments, same results — which the hop limit only stops after it
  has spent the whole budget. Two consecutive stale hops, not one: a single
  repeat is often a model re-reading something before pivoting. What counts as
  "already seen" is *which memories came back*, not the bytes of the rendering;
  see `_signature`.
* **The model saying it has enough** is the only one of the three that can stop
  at the *right* time rather than at a bound, because it is the only one that
  knows what the question needed. It is trusted, and only within the other two,
  because it is also the one that can be wrong in both directions.

`stopped_because` records which fired, and it is reported per question rather
than aggregated away. If `HOP_LIMIT` dominates, the loop is not converging, and
that is a finding about this design rather than a detail of one run.

### One tool per hop

`Step` holds one tool because a hop is one decision. A model that asks for three
at once has not planned three dependent retrievals — it has guessed three
independent ones, which is the thing this milestone exists to replace. The first
is run and the rest come back as a sentence saying to ask again, which costs a
hop and is the correct price.
"""

import asyncio
import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

import structlog

from memoryos.application.agent.compaction import Compacted, Counter, compact
from memoryos.application.agent.tools import ToolRegistry, ToolResult, ToolSpec, UnknownTool
from memoryos.application.ports import (
    ModelTurn,
    ToolCall,
    ToolCallingModel,
    ToolExchange,
)
from memoryos.domain.backoff import wait_for
from memoryos.domain.citation import Citation
from memoryos.domain.jobs import PermanentError, TransientError

logger = structlog.get_logger(__name__)

DEFAULT_MAX_HOPS = 6

# Tokens the compacted findings block may spend. See `compaction.py`; the number
# is a judgement about how much history is worth carrying against how much of
# the window the last two verbatim results already take (~1,400 for two
# searches).
DEFAULT_FINDING_BUDGET = 1200

# Consecutive stale hops before the loop stops. One is not enough — a model
# re-reading a result before changing direction is normal — and three is a hop
# spent proving something two already showed.
STALE_LIMIT = 2

# Attempts at one model call before the trajectory ends.
#
# **M7.1 said "rate limits are TransientError, so existing backoff applies" and
# nothing here applied it.** That went unnoticed because the only limit either
# provider hit during M7.1 was a *daily* one, where retrying is pointless and
# ending the trajectory is right. A per-minute limit is the opposite case and it
# is the common one: `openai/gpt-oss-20b` refused hop four of a four-hop run and
# asked to be tried again in **285 milliseconds**, and the trajectory ended.
#
# Five rather than three, and that too was measured. A tokens-per-minute limit of
# 8,000 against a ~4,000-token prompt is two calls a minute, so a six-hop
# trajectory spends more of its life waiting than talking; at three attempts,
# `openai/gpt-oss-120b` lost hop six after five good ones. Bounded by the ceiling
# below, so the worst case is minutes of waiting rather than unbounded.
RETRY_ATTEMPTS = 5

# The longest advised wait worth sitting through inside one question.
#
# The distinction this draws is the whole point of retrying here at all. A
# sliding token-per-minute window advises tenths of a second and a retry costs
# nothing; a daily quota advises half an hour, `wait_for` caps that at two
# minutes, and sleeping two minutes three times in a foreground command is
# indistinguishable from a hang — while the four hops already completed sit
# unread in a trajectory nobody can see yet. Above this, fail now and let the
# caller see the partial work.
RETRY_CEILING_SECONDS = 30.0


class StopReason(StrEnum):
    """Why the loop stopped, which is half of what a trajectory is for.

    **`ANSWERED` and `CONFIDENCE` are both voluntary stops and they are not the
    same event**, which is why they are two members rather than one:

    * `CONFIDENCE` — the model called at least one tool, then chose to answer
      while hops and novelty budget both remained. It said it had enough. This is
      the third termination condition, and the only one that can stop at the
      right time rather than at a bound.
    * `ANSWERED` — the model answered without calling anything. Nothing in the
      corpus backed it. That is legitimate for "what can you do" and it is a
      routing failure for everything else, and either way an answer standing on
      zero retrievals must not be scored beside one standing on four.

    The other three are bounds. `ERROR` means the loop could not continue at all
    — the provider refused, or a hop failed in a way no correction reaches — and
    is the only reason that can leave `answer` as None.
    """

    ANSWERED = "answered"
    HOP_LIMIT = "hop_limit"
    NO_NEW_INFORMATION = "no_new_information"
    CONFIDENCE = "confidence"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Step:
    """One hop: what the model was thinking, what it called, what came back.

    `thought` is whatever prose the turn carried alongside its tool call, and it
    is often empty — providers narrate inconsistently, and a model that calls a
    tool with no comment has still taken a step. It is recorded rather than
    required.

    The fields after `result` are not part of the shape M7.1 specifies; they are
    what makes the trajectory readable afterwards. `call_id` is the provider's
    own handle, needed to replay this step to Groq; `novel` is what the
    no-new-information rule decided about it; the token counts are the
    provider's, per hop, because a total that cannot be attributed to a hop
    cannot tell you which hop was expensive.
    """

    thought: str
    tool: str | None
    args: dict[str, Any]
    result: ToolResult | None
    call_id: str = ""
    novel: bool = True
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Everything one question did, with the answer as one field of it."""

    question: str
    steps: list[Step]
    answer: str | None
    stopped_because: StopReason
    # Every citation every hop produced, including hops the model could no
    # longer see by the time it wrote the answer. **A distinct guarantee from
    # the locators compaction keeps**: these are for a reader checking the
    # answer, those are for the model attributing a claim as it writes.
    citations: list[Citation] = field(default_factory=list)
    # Cost. Model calls rather than hops, because the forced final answer is a
    # call that is not a hop and the difference is a third of a cheap run.
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    # Set only when `stopped_because` is ERROR: the provider's own sentence.
    error: str | None = None
    # Seconds the provider asked to be left alone for, when it said. Read off the
    # exception by attribute rather than by importing `RateLimited`, which lives
    # in `adapters/` and has no business being named here — and a rate limit is
    # the *expected* end of a trajectory on a free tier, so "come back in 23s" is
    # operational information rather than a detail of a failure.
    retry_after: float | None = None

    @property
    def hops(self) -> int:
        """Steps that actually called something. The number `--max-hops` bounds."""
        return sum(1 for step in self.steps if step.tool is not None)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        return any(step.result.truncated for step in self.steps if step.result)


_SYSTEM_TEMPLATE = """You answer questions about one person's private corpus: their files, \
notes, code and the decisions they recorded.

You work in HOPS. On each turn you either call ONE tool, or give your final \
answer. A question that needs several retrievals is answered by calling a tool, \
reading what it returned, and calling the next one based on what you learned — \
not by guessing every query up front.

Rules:
- ONE tool per turn. If you ask for several, only the first runs.
- Each call should be shaped by what the previous one returned. Repeating a \
search with reworded terms is not a new hop; if a result was empty or unhelpful, \
try a DIFFERENT tool or a different kind of query.
- Answer from what the tools returned. If they returned nothing useful, say so \
plainly rather than answering from general knowledge. A corpus this size often \
does not contain what a question assumes it does, and saying that is a better \
answer than a fluent one.
- Never invent a file name, a date or a quotation. If you name a source, it must \
be one a tool result showed you.
- In your final answer, mark every factual sentence with the hop it came from, \
in brackets: "The lease expires after 30 seconds [2]." Use the hop numbers you \
were given. A sentence you cannot attribute to a hop is one you should not write.
- Before your first call, decide what the question DECOMPOSES into. "What have I \
repeated" is not one lookup: it is find the cases, read what each assumed, then \
check whether they are really the same thing. Take those steps.
- Stop when you can answer, and not before. One search that returned things only \
loosely related to the question is not an answer — it is the first hop. Extra \
hops cost real money, and so does a confident paragraph about the wrong subject.
- When a result says it was truncated, or when findings were dropped for space, \
say that your answer covers only part of what exists.

Today is {today}."""

_FINAL_TEMPLATE = """{question}

{history}

You have no more hops: {why}. Answer the question now from what you gathered \
above, and nothing else, marking each factual sentence with the hop it came from \
in brackets. If what you gathered does not answer it, say exactly that and say \
what is missing — an honest "the corpus does not contain this" is correct and a \
confident synthesis of four weak results is not."""

_WHY = {
    StopReason.HOP_LIMIT: "you have used every hop you were given",
    StopReason.NO_NEW_INFORMATION: (
        "your last two calls returned nothing you had not already seen"
    ),
}


def system_prompt(*, today: date | None = None) -> str:
    """The instructions, with today's date in them.

    The date is M7.0's fix and it survives for M7.0's reason: a temporal tool
    with no clock resolves "August" against the model's training prior and asks
    the corpus about a year it has nothing from. It belongs here rather than in
    a tool description because it is a fact about the conversation.
    """
    return _SYSTEM_TEMPLATE.format(today=(today or date.today()).isoformat())


class MultiHopPlanner:
    """The loop. Ask, act, read, decide, repeat — under three bounds."""

    def __init__(
        self,
        model: ToolCallingModel,
        registry: ToolRegistry,
        counter: Counter,
        *,
        max_hops: int = DEFAULT_MAX_HOPS,
        finding_budget: int = DEFAULT_FINDING_BUDGET,
        max_tokens: int = 700,
    ) -> None:
        self._model = model
        self._registry = registry
        self._counter = counter
        self._max_hops = max_hops
        self._finding_budget = finding_budget
        self._max_tokens = max_tokens

    async def run(self, question: str, *, max_hops: int | None = None) -> Trajectory:
        """One question, up to `max_hops` retrievals, one trajectory.

        Never raises for anything the model or a tool did. A provider that
        refuses at hop five ends the trajectory with `ERROR` and the four hops
        that worked still in it, because a rate limit arriving late should not
        delete the evidence of what the run had already found — and on these free
        tiers a rate limit arriving late is the normal case rather than the
        exceptional one.
        """
        limit = self._max_hops if max_hops is None else max(1, max_hops)
        started = time.monotonic()
        instructions = system_prompt()
        specs = self._registry.specs()

        steps: list[Step] = []
        seen_memories: set[str] = set()
        seen_digests: set[str] = set()
        stale = 0
        calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        stop = StopReason.HOP_LIMIT

        for hop in range(1, limit + 1):
            history = compact(
                steps,
                counter=self._counter,
                budget=self._finding_budget,
            )
            turn_started = time.monotonic()
            try:
                turn = await self._converse(
                    instructions,
                    _user_message(question, history, hop=hop, limit=limit),
                    tools=specs,
                    exchanges=_exchanges(history),
                )
            except (TransientError, PermanentError) as exc:
                logger.warning("agent.model_failed", hop=hop, error=str(exc))
                return self._finish(
                    question,
                    steps,
                    answer=None,
                    stop=StopReason.ERROR,
                    calls=calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    started=started,
                    error=str(exc),
                    retry_after=getattr(exc, "retry_after", None),
                )
            calls += 1
            prompt_tokens += turn.prompt_tokens
            completion_tokens += turn.completion_tokens

            if not turn.wants_tools:
                # The third condition: the model stopped on its own, inside both
                # bounds. `CONFIDENCE` only if it actually retrieved something —
                # an answer that called nothing is a different object and gets a
                # different reason.
                stop = (
                    StopReason.CONFIDENCE
                    if any(step.tool for step in steps)
                    else StopReason.ANSWERED
                )
                steps.append(
                    Step(
                        thought=turn.text,
                        tool=None,
                        args={},
                        result=None,
                        prompt_tokens=turn.prompt_tokens,
                        completion_tokens=turn.completion_tokens,
                        duration_ms=_ms(turn_started),
                    )
                )
                logger.info(
                    "agent.stopped", question=question, reason=stop.value, hops=hop - 1
                )
                return self._finish(
                    question,
                    steps,
                    answer=turn.text,
                    stop=stop,
                    calls=calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    started=started,
                )

            call = turn.tool_calls[0]
            result = await self._call(call, extra=len(turn.tool_calls) - 1)
            memories, digest = _signature(result)
            # A result that cited memories is new when any of them is new. One
            # unseen memory among five seen ones is still a fact the model did
            # not have, and calling that stale would stop a loop that is working.
            novel = bool(memories - seen_memories) if memories else digest not in seen_digests
            seen_memories |= memories
            seen_digests.add(digest)
            steps.append(
                Step(
                    thought=turn.text,
                    tool=call.name,
                    args=dict(call.arguments),
                    result=result,
                    call_id=call.id,
                    novel=novel,
                    prompt_tokens=turn.prompt_tokens,
                    completion_tokens=turn.completion_tokens,
                    duration_ms=_ms(turn_started),
                )
            )
            logger.info(
                "agent.hop",
                hop=hop,
                tool=call.name,
                novel=novel,
                citations=len(result.citations),
                tokens=turn.prompt_tokens + turn.completion_tokens,
            )

            stale = stale + 1 if not novel else 0
            if stale >= STALE_LIMIT:
                stop = StopReason.NO_NEW_INFORMATION
                break
        else:
            stop = StopReason.HOP_LIMIT

        # A bound fired, so the answer has to be asked for rather than waited
        # for. Tools are withdrawn for this call: offering them again to a model
        # that has just run out of hops is inviting the one response the loop
        # cannot use.
        answer, final = await self._final(question, steps, stop, instructions)
        if final is not None:
            calls += 1
            prompt_tokens += final.prompt_tokens
            completion_tokens += final.completion_tokens
        logger.info(
            "agent.stopped", question=question, reason=stop.value, hops=len(steps)
        )
        return self._finish(
            question,
            steps,
            answer=answer,
            stop=stop if answer is not None else StopReason.ERROR,
            calls=calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            started=started,
            error=None if answer is not None else "the final answer call failed",
        )

    async def _converse(
        self,
        instructions: str,
        user: str,
        *,
        tools: Sequence[ToolSpec] = (),
        exchanges: Sequence[ToolExchange] = (),
    ) -> ModelTurn:
        """One model call, retried while the provider says to come back soon.

        `wait_for` prefers the provider's own number over an estimate, which is
        the whole reason this can distinguish the two kinds of 429 — see
        `RETRY_CEILING_SECONDS`. `PermanentError` is not retried: the adapter has
        already made that judgement and a second identical request will get the
        same answer.
        """
        for attempt in range(RETRY_ATTEMPTS):
            try:
                return await self._model.converse(
                    instructions,
                    user,
                    tools=tools,
                    exchanges=exchanges,
                    max_tokens=self._max_tokens,
                )
            except TransientError as exc:
                delay = wait_for(exc, attempt)
                if attempt == RETRY_ATTEMPTS - 1 or delay > RETRY_CEILING_SECONDS:
                    raise
                logger.info(
                    "agent.retrying", attempt=attempt + 1, delay=round(delay, 2)
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _final(
        self,
        question: str,
        steps: list[Step],
        stop: StopReason,
        instructions: str,
    ) -> tuple[str | None, ModelTurn | None]:
        history = compact(steps, counter=self._counter, budget=self._finding_budget)
        # Everything, not just the compacted older half: this is the last call,
        # there is no next hop to save budget for, and the two verbatim results
        # are the ones most likely to carry the answer.
        rendered = "\n\n".join(
            part
            for part in (history.render(), _verbatim(history))
            if part
        ) or "You gathered nothing."
        try:
            turn = await self._converse(
                instructions,
                _FINAL_TEMPLATE.format(
                    question=question,
                    history=rendered,
                    why=_WHY.get(stop, "the loop has ended"),
                ),
            )
        except (TransientError, PermanentError) as exc:
            logger.warning("agent.final_failed", error=str(exc))
            return None, None
        return turn.text, turn

    async def _call(self, call: ToolCall, *, extra: int) -> ToolResult:
        """Run one tool. **Nothing a tool does raises out of here.**

        M7.0 let `UnknownTool` propagate, and that was right for a loop with one
        call: there was no turn left in which the model could pick a different
        name. With hops there is, so every failure becomes a sentence the model
        reads and can act on — which is the same argument the registry already
        makes for invalid arguments, extended to the two cases M7.0 could not
        extend it to.

        The broad `except` is deliberate and is the uncomfortable part. A tool
        raising `AttributeError` is a bug in this system, not a mistake the model
        made, and turning it into prose hides it from the person who should see
        it. So it is logged with its traceback and named as a failure in the
        result: the run continues, and the bug is still in the log.
        """
        note = (
            ""
            if not extra
            else (
                f"\n\n(You asked for {extra} more tool call(s) in the same turn. "
                "Only the first runs. Ask for the others one at a time.)"
            )
        )
        try:
            result = await self._registry.call(call.name, dict(call.arguments))
        except UnknownTool as exc:
            logger.warning("agent.unknown_tool", tool=call.name)
            return ToolResult(content=f"{exc}. Call one of those instead.{note}")
        except Exception as exc:  # see docstring
            logger.exception("agent.tool_failed", tool=call.name)
            return ToolResult(
                content=(
                    f"{call.name} failed: {type(exc).__name__}: {exc}. That is a "
                    "fault in the tool rather than in your arguments — use a "
                    f"different tool, or answer without it.{note}"
                )
            )
        if not note:
            return result
        return ToolResult(
            content=result.content + note,
            citations=result.citations,
            truncated=result.truncated,
        )

    def _finish(
        self,
        question: str,
        steps: list[Step],
        *,
        answer: str | None,
        stop: StopReason,
        calls: int,
        prompt_tokens: int,
        completion_tokens: int,
        started: float,
        error: str | None = None,
        retry_after: float | None = None,
    ) -> Trajectory:
        citations: list[Citation] = []
        seen: set[str] = set()
        for step in steps:
            for citation in step.result.citations if step.result else ():
                # Deduplicated by locator: five hops over one corpus revisit the
                # same chunks, and an answer whose source list repeats a file
                # nine times reads as nine pieces of evidence.
                if citation.locator not in seen:
                    seen.add(citation.locator)
                    citations.append(citation)
        return Trajectory(
            question=question,
            steps=steps,
            answer=answer,
            stopped_because=stop,
            citations=citations,
            model_calls=calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration_ms=_ms(started),
            error=error,
            retry_after=retry_after,
        )


def _user_message(
    question: str, history: Compacted, *, hop: int, limit: int
) -> str:
    """The question, what is already known, and how much budget is left.

    **The hop counter is in the prompt on purpose.** A model that does not know
    it has one hop left spends it on another search; told, it answers. This is
    the cheapest of the three termination conditions to influence, because it is
    the only one that can end the run at the right moment rather than at a bound.
    """
    remaining = limit - hop + 1
    parts = [question]
    block = history.render()
    if block:
        parts.append("What you have found so far:\n\n" + block)
    if history.verbatim:
        parts.append(
            "The most recent result(s) follow in full below, as tool messages."
        )
    parts.append(
        f"Hop {hop} of {limit}. "
        + (
            "This is your last hop: answer now, from what you have."
            if remaining == 1
            else f"{remaining} hops remain. Call one tool, or answer if you have enough."
        )
    )
    return "\n\n".join(parts)


def _exchanges(history: Compacted) -> list[ToolExchange]:
    """The verbatim tail, in the provider's own replay shape."""
    return [
        ToolExchange(
            call=ToolCall(
                id=step.call_id or step.tool or "",
                name=step.tool or "",
                arguments=dict(step.args),
            ),
            result=step.result.content if step.result else "",
        )
        for step in history.verbatim
    ]


def _verbatim(history: Compacted) -> str:
    """The kept-in-full results as text, for the call that has no tool messages."""
    return "\n\n".join(
        f"hop {hop} · {step.tool}:\n{step.result.content}"
        for hop, step in enumerate(history.verbatim, start=history.verbatim_from)
        if step.result
    )


def _signature(result: ToolResult) -> tuple[frozenset[str], str]:
    """What a step returned, in the terms the novelty rule should compare.

    **The memories, not the bytes.** M7.1 hashed the whitespace-normalised
    rendering, and on this system that hash can essentially never repeat for the
    tool it most needed to catch: `search_memories` prints `score 5.237`, and two
    calls returning the identical five memories at fractionally different scores
    produce different hashes. Measured on a real run — "repeated mistakes" then
    "mistakes I have repeated" — both hops were recorded as new, and the rule
    that exists to catch a reworded search watched one go past.

    So the primary signature is the set of memory ids the result cited, which is
    what "results already seen" means in any reading that is about information.

    The digest survives as the fallback for results that cite nothing, and those
    are not an edge case: "No silences of 30 days or more", "no recorded decision
    matches", an argument correction. Two of those in a row is a loop, and with
    no ids to compare it is the only thing left to compare.
    """
    memories = frozenset(str(citation.memory_id) for citation in result.citations)
    return memories, hashlib.sha256(" ".join(result.content.split()).encode()).hexdigest()


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def summarise(trajectories: Sequence[Trajectory]) -> dict[str, int]:
    """Stop reasons counted. **If `HOP_LIMIT` dominates, the loop is not
    converging**, and that is a finding about the design rather than about a run.
    """
    counted: dict[str, int] = {reason.value: 0 for reason in StopReason}
    for trajectory in trajectories:
        counted[trajectory.stopped_because.value] += 1
    return counted
