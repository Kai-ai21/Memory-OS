"""Scoring how the agent reasoned, not what it concluded.

**An agent can reach a correct answer through terrible reasoning** — the wrong
tool, four rewordings of one search, and a lucky hit in the last of them. Scoring
only the answer optimises for that: it rewards the luck and cannot see the waste,
and every improvement it motivates is an improvement to the wrong thing. Same
argument M2.0 makes about measuring retrieval instead of a proxy that correlates
with it sometimes.

So these are functions over a `Trajectory`. No model, no network, no database —
the trajectory carries everything, which is exactly what M7.1 built it to do.

### The five, and what each is really asking

* **Tool appropriateness** — did the call match what the step said it wanted?
  Judged from the model's own narration, which is the only statement of intent
  available. **It is unmeasurable for most steps**, because providers narrate
  inconsistently and a turn that calls a tool with no comment is normal; the
  score is over the steps that spoke, and `judgeable` says how many those were.
* **Information gain** — did the result contain anything not already seen? A hop
  returning memories three earlier hops returned is a hop that cost a model call
  and moved nothing.
* **Dependency** — did the query use the previous result, or ignore it? See
  below; this is the one that matters.
* **Efficiency** — hops taken against the golden minimum. Above 1.0 is capped:
  finishing in fewer hops than the minimum means the minimum was wrong, not that
  the agent was brilliant.
* **Termination** — did it stop in the right place, given what the question
  needed?

### Dependency is the metric that can embarrass the design

Multi-hop reasoning and repeated retrieval produce identical-looking
trajectories: several hops, several tool calls, a fluent answer. The difference
is whether hop N's query was *written from* hop N-1's result. An agent that
searches "repeated mistakes", reads five files, and then searches "mistakes I
have repeated" has taken two hops and reasoned across none of them.

So dependency looks for the previous result inside the next query, in three
descending strengths — an id it returned, a phrase it contained that the question
did not, or nothing. It is deliberately mechanical and deliberately generous at
the edges: a false *high* score here would be this module flattering the thing it
exists to audit, so the strong signal requires something the agent could only
have got from the result.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import mean, pstdev

import structlog

from memoryos.application.agent.planner import Step, StopReason, Trajectory

logger = structlog.get_logger(__name__)

# What each tool is for, in the words a model uses when it narrates wanting it.
#
# Rules-based for the reason M7.2's claim classifier is: the alternative is
# asking a language model whether a language model's tool choice matched its own
# stated intent, which is the thing under test marking its own paper.
_INTENT: dict[str, tuple[str, ...]] = {
    "search_memories": (
        "search", "find", "look for", "look up", "passage", "mention",
        "about", "what does", "where is", "locate",
    ),
    "get_decisions": (
        "decision", "decided", "chose", "choice", "why", "rationale",
        "alternative", "rejected", "assumption", "trade-off", "tradeoff",
    ),
    "query_timeline": (
        "timeline", "when", "period", "month", "week", "activity", "how busy",
        "over time", "date range", "chronolog",
    ),
    "find_gaps": (
        "gap", "silence", "silent", "stopped", "abandon", "break", "quiet",
        "went dark", "hiatus",
    ),
    "traverse_graph": (
        "connect", "related", "linked", "neighbour", "neighbor", "graph",
        "entity", "shares", "structural",
    ),
    "get_memory": (
        "read", "full", "whole", "in full", "entire", "open", "contents",
        "the file itself",
    ),
}

# A memory id as the tools print it.
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# Words too common to count as evidence that a query came from a result. Not a
# general stopword list — a general one would drop the technical vocabulary that
# is the whole signal here.
_COMMON = frozenset(
    (
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
        "from", "at", "is", "are", "was", "were", "be", "been", "this", "that",
        "these", "those", "it", "its", "as", "not", "no", "what", "which", "who",
        "when", "where", "how", "why", "do", "does", "did", "can", "could", "will",
        "would", "should", "i", "my", "me", "you", "your", "we", "our", "they",
        "their", "have", "has", "had", "more", "most", "some", "any", "all", "one",
        "two", "into", "out", "up", "down", "about",
    )
)

# Shortest word worth treating as a content term. Three characters admits `api`
# and `rrf`; two admits `of`.
_MIN_TERM = 3


class Failure(StrEnum):
    """Why a question did not come out right.

    **The last one is the reason this is an enum rather than a boolean.** With
    eight to twelve decisions and a corpus of source code, several of these
    questions cannot be answered by any agent, and counting those as agent
    failures would send the next milestone off to fix the loop when the problem
    is that nobody wrote the data down. Separating the two is the point.
    """

    NONE = "none"
    WRONG_TOOL = "wrong_tool"
    WRONG_ARGUMENTS = "wrong_arguments"
    STOPPED_EARLY = "stopped_early"
    LOOPED = "looped"
    WRONG_CONCLUSION = "wrong_conclusion"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True, slots=True)
class HopScore:
    """One hop, judged on its own terms."""

    hop: int
    tool: str
    # None when the step carried no narration — not zero. A hop the model did not
    # explain is not a hop it explained badly.
    appropriate: bool | None
    # Share of the memories this hop returned that no earlier hop had.
    gain: float
    # 1.0 an id from the previous result, 0.5 a phrase from it, 0.0 neither.
    dependency: float
    dependency_evidence: str = ""


@dataclass(frozen=True, slots=True)
class TrajectoryScore:
    """The five metrics, and enough of the working to argue with them."""

    question_id: str
    hops: int
    min_hops: int
    tool_appropriateness: float
    # How many hops the appropriateness score is actually over. A 1.0 across one
    # narrated hop out of five is not a 1.0 about the trajectory.
    judgeable: int
    information_gain: float
    dependency: float
    efficiency: float
    termination: float
    failure: Failure
    stopped_because: str
    # Golden facts found in the answer, and the ones that were not.
    facts_found: tuple[str, ...] = ()
    facts_missing: tuple[str, ...] = ()
    required_missing: tuple[str, ...] = ()
    forbidden_used: tuple[str, ...] = ()
    per_hop: tuple[HopScore, ...] = ()
    # Cost, carried here so one row of a report has both halves of the trade.
    tokens: int = 0
    duration_ms: int = 0
    support_rate: float = 0.0
    verdict: str = ""
    refused: bool = False

    @property
    def overall(self) -> float:
        """The four metrics that are always defined, averaged.

        **Appropriateness is excluded on purpose.** It is undefined for a
        trajectory nobody narrated, and folding a default into a mean would let
        a silent provider raise or lower the headline number without the agent
        doing anything differently.
        """
        return mean(
            [self.information_gain, self.dependency, self.efficiency, self.termination]
        )


@dataclass(frozen=True, slots=True)
class GoldenAgentQuestion:
    """What a question is expected to need, and what would be wrong.

    **No exact tool sequence**, as M7.3 requires: several paths through six tools
    are legitimately correct, and pinning one would score the agent on matching a
    guess rather than on reaching the answer. What is pinned is which tools must
    appear, which must not, and how few hops could do it.

    `answerable` is the field that makes the taxonomy honest. A question the
    corpus cannot support is not a question the agent can pass, and marking that
    in the answer key rather than discovering it in the results is what keeps
    "the data is not there" from being counted as "the loop is broken".
    """

    id: str
    question: str
    key_facts: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    min_hops: int = 1
    answerable: bool = True
    notes: str = ""


def score(
    trajectory: Trajectory,
    golden: GoldenAgentQuestion,
    *,
    support_rate: float = 0.0,
    verdict: str = "",
    refused: bool = False,
) -> TrajectoryScore:
    """Every metric for one run of one question."""
    acted = [step for step in trajectory.steps if step.tool and step.result]
    per_hop = _hops(acted, golden.question)

    judged = [hop.appropriate for hop in per_hop if hop.appropriate is not None]
    used = {step.tool for step in acted}
    required_missing = tuple(
        name for name in golden.required_tools if name not in used
    )
    forbidden_used = tuple(name for name in golden.forbidden_tools if name in used)

    answer = (trajectory.answer or "").lower()
    found = tuple(fact for fact in golden.key_facts if _mentions(answer, fact))
    missing = tuple(fact for fact in golden.key_facts if fact not in found)

    failure = _classify(
        trajectory,
        golden,
        acted,
        per_hop=per_hop,
        required_missing=required_missing,
        forbidden_used=forbidden_used,
        facts_missing=missing,
        verdict=verdict,
        refused=refused,
    )
    return TrajectoryScore(
        question_id=golden.id,
        hops=len(acted),
        min_hops=golden.min_hops,
        tool_appropriateness=mean(judged) if judged else 0.0,
        judgeable=len(judged),
        information_gain=mean([hop.gain for hop in per_hop]) if per_hop else 0.0,
        # Dependency is over hops 2..n: the first query has nothing to depend on,
        # and scoring it zero would cap a perfect two-hop trajectory at 0.5.
        dependency=(
            mean([hop.dependency for hop in per_hop[1:]]) if len(per_hop) > 1 else 0.0
        ),
        efficiency=_efficiency(len(acted), golden.min_hops),
        termination=_termination(trajectory, per_hop, golden),
        failure=failure,
        stopped_because=trajectory.stopped_because.value,
        facts_found=found,
        facts_missing=missing,
        required_missing=required_missing,
        forbidden_used=forbidden_used,
        per_hop=tuple(per_hop),
        tokens=trajectory.tokens,
        duration_ms=trajectory.duration_ms,
        support_rate=support_rate,
        verdict=verdict,
        refused=refused,
    )


def _hops(acted: Sequence[Step], question: str) -> list[HopScore]:
    scored: list[HopScore] = []
    seen: set[str] = set()
    question_terms = _terms(question)

    for index, step in enumerate(acted):
        assert step.tool is not None and step.result is not None
        returned = {
            str(citation.memory_id) for citation in step.result.citations
        }
        # A result with no citations has no memories to be new; its novelty was
        # already decided by the loop against the rendered text, and that verdict
        # is reused rather than recomputed differently here.
        gain = (
            len(returned - seen) / len(returned)
            if returned
            else (1.0 if step.novel else 0.0)
        )
        seen |= returned

        dependency, evidence = (
            (0.0, "")
            if index == 0
            else _dependency(step, acted[index - 1], question_terms)
        )
        scored.append(
            HopScore(
                hop=index + 1,
                tool=step.tool,
                appropriate=_appropriate(step),
                gain=gain,
                dependency=dependency,
                dependency_evidence=evidence,
            )
        )
    return scored


def _appropriate(step: Step) -> bool | None:
    """Whether the tool called is the one the narration asked for.

    None when there is no narration, which is most of the time. A hop the model
    did not explain is not a hop it explained badly, and scoring silence as
    failure would make this metric a measurement of how chatty a provider is.
    """
    thought = step.thought.strip().lower()
    if not thought or step.tool is None:
        return None
    wanted = {
        name
        for name, markers in _INTENT.items()
        if any(marker in thought for marker in markers)
    }
    if not wanted:
        # The model said something that names no tool's job. Unjudgeable rather
        # than wrong: "Let me look further" is not evidence either way.
        return None
    return step.tool in wanted


def _dependency(
    step: Step, previous: Step, question_terms: frozenset[str]
) -> tuple[float, str]:
    """Whether this query was written from the previous result.

    Three strengths, and the strong one is deliberately hard to reach by
    accident: a memory id in the arguments is something the agent could only have
    obtained from a result, since ids appear nowhere else and no model invents a
    matching UUID.

    The middle strength is a content word shared with the previous result and
    *absent from the question*. The exclusion is what makes it mean anything —
    two searches for terms lifted from the question are two independent searches,
    however much text they have in common with each other.
    """
    assert previous.result is not None
    arguments = " ".join(str(value) for value in step.args.values())
    previous_text = previous.result.content

    for found in _UUID.findall(arguments.lower()):
        if found in previous_text.lower():
            return 1.0, f"id {found[:8]}… from hop {previous.tool}"

    borrowed = (_terms(arguments) & _terms(previous_text)) - question_terms
    if borrowed:
        return 0.5, "borrowed " + ", ".join(sorted(borrowed)[:4])
    return 0.0, ""


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        word
        for word in re.findall(r"[a-z_][a-z0-9_]+", text.lower())
        if len(word) >= _MIN_TERM and word not in _COMMON
    )


def _efficiency(hops: int, minimum: int) -> float:
    """Golden minimum over hops taken, capped at 1.0.

    Capped because finishing in fewer hops than the minimum does not mean the
    agent was brilliant — it means the minimum was wrong, or the answer skipped a
    step it needed. Either way it is not efficiency, and letting it score above
    1.0 would let a question with a badly-set minimum carry the mean.
    """
    if hops <= 0:
        return 0.0
    return min(1.0, minimum / hops)


def _termination(
    trajectory: Trajectory, per_hop: Sequence[HopScore], golden: GoldenAgentQuestion
) -> float:
    """Did it stop in the right place?

    Full marks for stopping voluntarily with an answer, having taken at least the
    minimum hops and learned something in the last one. Half for stopping at a
    bound, which is a bound doing its job rather than the agent judging well.
    Zero for a run that produced nothing.
    """
    if trajectory.stopped_because is StopReason.ERROR or trajectory.answer is None:
        return 0.0
    if trajectory.stopped_because is StopReason.ANSWERED:
        # Answered with no retrieval at all. Correct only for a question nothing
        # in the corpus could support.
        return 1.0 if not golden.answerable else 0.0
    if trajectory.stopped_because in (
        StopReason.HOP_LIMIT,
        StopReason.NO_NEW_INFORMATION,
    ):
        return 0.5
    # CONFIDENCE: the model's own judgement, which is right only if it had done
    # the work. Stopping early is the failure M7.1 measured and this is where it
    # shows up as a number.
    if len(per_hop) < golden.min_hops:
        return 0.0
    return 1.0 if not per_hop or per_hop[-1].gain > 0 else 0.5


def _classify(
    trajectory: Trajectory,
    golden: GoldenAgentQuestion,
    acted: Sequence[Step],
    *,
    per_hop: Sequence[HopScore],
    required_missing: Sequence[str],
    forbidden_used: Sequence[str],
    facts_missing: Sequence[str],
    verdict: str,
    refused: bool,
) -> Failure:
    """Classify, most diagnostic first.

    Order matters and is an argument. A run that used a forbidden tool *and*
    stopped early is reported as the tool problem, because a description that
    routes wrong is the thing to fix and the early stop is plausibly a
    consequence of it.
    """
    if trajectory.stopped_because is StopReason.ERROR:
        # Not the agent's reasoning: the provider stopped answering. Counted as
        # insufficient rather than as a reasoning failure, because scoring a rate
        # limit as "stopped too early" would put quota in the taxonomy.
        return Failure.INSUFFICIENT_DATA

    if forbidden_used or required_missing:
        return Failure.WRONG_TOOL

    # A tool that answered with a correction rather than a result. The registry's
    # own words, which is the only thing that distinguishes bad arguments from an
    # honestly empty corpus.
    for step in acted:
        assert step.result is not None
        content = step.result.content
        if "were not valid" in content or "is not a memory id" in content:
            return Failure.WRONG_ARGUMENTS
        if "is not a period" in content or "has to be positive" in content:
            return Failure.WRONG_ARGUMENTS

    if not golden.answerable:
        # The corpus cannot support this. Refusing or saying so is the correct
        # outcome and is not a failure; asserting an answer anyway is.
        answered_anyway = not refused and verdict != "ungrounded"
        return Failure.WRONG_CONCLUSION if answered_anyway else Failure.INSUFFICIENT_DATA

    if not per_hop:
        return Failure.STOPPED_EARLY

    if trajectory.stopped_because is StopReason.NO_NEW_INFORMATION or (
        sum(1 for hop in per_hop if hop.gain == 0.0) >= 2
    ):
        return Failure.LOOPED

    if len(per_hop) < golden.min_hops:
        return Failure.STOPPED_EARLY

    if facts_missing:
        # It reached the data — required tools ran, hops happened — and did not
        # draw the conclusion. The one category here that is squarely about
        # reasoning rather than about plumbing.
        return Failure.WRONG_CONCLUSION

    return Failure.NONE


def _mentions(answer: str, fact: str) -> bool:
    """Whether the answer contains a key fact.

    Every content word of the fact has to appear somewhere in the answer, in any
    order. Substring matching on the whole phrase would fail on a correct answer
    that used a synonym for one connective; requiring all content words is loose
    enough to survive rewording and tight enough that an unrelated paragraph does
    not pass.
    """
    if not answer:
        return False
    return _terms(fact) <= _terms(answer)


@dataclass(frozen=True, slots=True)
class Report:
    """Every question, once, plus the means a person actually reads."""

    scores: tuple[TrajectoryScore, ...] = ()

    def mean_of(self, metric: str) -> float:
        values = [getattr(row, metric) for row in self.scores]
        return mean(values) if values else 0.0

    @property
    def failures(self) -> dict[str, int]:
        counted = {member.value: 0 for member in Failure}
        for row in self.scores:
            counted[row.failure.value] += 1
        return counted

    @property
    def agent_failures(self) -> int:
        """Everything except the two categories that are not the agent's fault."""
        return sum(
            1
            for row in self.scores
            if row.failure not in (Failure.NONE, Failure.INSUFFICIENT_DATA)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "questions": len(self.scores),
            "means": {
                name: round(self.mean_of(name), 4)
                for name in (
                    "tool_appropriateness",
                    "information_gain",
                    "dependency",
                    "efficiency",
                    "termination",
                    "overall",
                    "support_rate",
                )
            },
            "cost": {
                "mean_tokens": round(self.mean_of("tokens")),
                "mean_duration_ms": round(self.mean_of("duration_ms")),
                "total_tokens": sum(row.tokens for row in self.scores),
            },
            "failures": self.failures,
            "agent_failures": self.agent_failures,
            "scores": [_row_dict(row) for row in self.scores],
        }


def variance(reports: Sequence[Report]) -> dict[str, dict[str, float]]:
    """How far the same question moves between identical runs.

    **The floor every future claim about agent improvement has to clear.** M2.3a
    established the discipline for retrieval, where the answer was ~0.012 and the
    only source of movement was floating-point ordering. Here the model is
    sampled, the tools are ranked by a model, and a single different tool choice
    at hop one changes every hop after it — so the number will be far larger, and
    a milestone reporting a gain smaller than it has reported nothing.

    Population standard deviation rather than sample: these are all the runs
    there are, not a sample of a larger set of runs.
    """
    if len(reports) < 2:
        return {}
    metrics = (
        "tool_appropriateness",
        "information_gain",
        "dependency",
        "efficiency",
        "termination",
        "overall",
        "support_rate",
    )
    spread: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [report.mean_of(metric) for report in reports]
        spread[metric] = {
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "range": round(max(values) - min(values), 4),
            "stdev": round(pstdev(values), 4),
        }

    # Per question as well as per metric, because a stable mean can hide two
    # questions swinging in opposite directions — which is the shape of
    # non-determinism this loop actually produces.
    by_question: dict[str, list[float]] = {}
    for report in reports:
        for row in report.scores:
            by_question.setdefault(row.question_id, []).append(row.overall)
    spread["per_question_overall"] = {
        question: round(max(values) - min(values), 4)
        for question, values in by_question.items()
        if len(values) > 1
    }
    return spread


def _row_dict(row: TrajectoryScore) -> dict[str, object]:
    return {
        "id": row.question_id,
        "hops": row.hops,
        "min_hops": row.min_hops,
        "tool_appropriateness": round(row.tool_appropriateness, 4),
        "judgeable": row.judgeable,
        "information_gain": round(row.information_gain, 4),
        "dependency": round(row.dependency, 4),
        "efficiency": round(row.efficiency, 4),
        "termination": round(row.termination, 4),
        "overall": round(row.overall, 4),
        "failure": row.failure.value,
        "stopped_because": row.stopped_because,
        "facts_found": list(row.facts_found),
        "facts_missing": list(row.facts_missing),
        "required_missing": list(row.required_missing),
        "forbidden_used": list(row.forbidden_used),
        "tokens": row.tokens,
        "duration_ms": row.duration_ms,
        "support_rate": round(row.support_rate, 4),
        "verdict": row.verdict,
        "refused": row.refused,
        "hops_detail": [
            {
                "hop": hop.hop,
                "tool": hop.tool,
                "appropriate": hop.appropriate,
                "gain": round(hop.gain, 4),
                "dependency": hop.dependency,
                "evidence": hop.dependency_evidence,
            }
            for hop in row.per_hop
        ],
    }


def compare(current: Report, baseline: dict[str, object]) -> list[tuple[str, float, float]]:
    """Metric, baseline, now — for the metrics both runs carry.

    Returns rows rather than printing them, and reports every metric rather than
    only the ones that moved: a table that hides the unchanged rows makes a
    regression in one of them look like an absence of information.
    """
    means = baseline.get("means")
    if not isinstance(means, dict):
        return []
    rows: list[tuple[str, float, float]] = []
    for name, was in means.items():
        if isinstance(was, int | float):
            rows.append((name, float(was), current.mean_of(name)))
    return rows


def load_golden(path: Path) -> list[GoldenAgentQuestion]:
    """The agent answer key, from JSON.

    Loud on a malformed entry rather than skipping it: an answer key that
    silently shrinks reports a rising score for a benchmark that is losing
    questions, which is the failure `golden.py` describes at length for
    retrieval and which is no less available here.
    """
    payload = json.loads(path.read_text())
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{path} contains no questions")

    loaded: list[GoldenAgentQuestion] = []
    for entry in questions:
        loaded.append(
            GoldenAgentQuestion(
                id=str(entry["id"]),
                question=str(entry["question"]),
                key_facts=tuple(entry.get("key_facts", ())),
                required_tools=tuple(entry.get("required_tools", ())),
                forbidden_tools=tuple(entry.get("forbidden_tools", ())),
                min_hops=int(entry.get("min_hops", 1)),
                answerable=bool(entry.get("answerable", True)),
                notes=str(entry.get("notes", "")),
            )
        )
    if len({question.id for question in loaded}) != len(loaded):
        raise ValueError(f"{path} has duplicate question ids")
    return loaded
