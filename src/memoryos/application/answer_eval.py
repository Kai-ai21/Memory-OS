"""Measuring answers the way `evaluate` measures rankings.

Three numbers, and the second two matter more than the first.

**Citation rate** — factual sentences carrying a marker. Watched across runs
rather than judged in absolute terms: a single uncited sentence is noise, a rate
that drops after a prompt change is a regression.

**Refusal rate on out-of-corpus questions** — should be at or near 100%. This is
the guardrail. A system that answers "who is on the engineering team" from a
repository of Python has invented a team, and no retrieval metric would notice.

**Hallucinated citation rate** — indices outside the supplied range. Must be
exactly zero. Unlike the other two this admits no interpretation: the model
referenced a passage that was never in the prompt.

What this cannot measure is whether a cited passage actually *supports* its
sentence. The only available judge is another language model, and a model
grading its own grounding is not evidence. That check is a person reading five
answers, which the milestone asks for by hand and this deliberately does not
fake.
"""

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from memoryos.application.answering import AnswerQuestion, GroundedAnswer
from memoryos.domain.jobs import PermanentError, TransientError

logger = structlog.get_logger(__name__)

DEFAULT_REFUSAL_QUERIES = Path("var/refusal-queries.json")


@dataclass(frozen=True, slots=True)
class AnswerRun:
    question: str
    # True when the question is one the corpus cannot answer, so refusing is the
    # correct behaviour rather than a failure to find something.
    out_of_corpus: bool
    answer: GroundedAnswer | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerReport:
    runs: list[AnswerRun] = field(default_factory=list)

    @property
    def answered(self) -> list[AnswerRun]:
        return [run for run in self.runs if run.answer is not None]

    @property
    def in_corpus(self) -> list[AnswerRun]:
        return [run for run in self.answered if not run.out_of_corpus]

    @property
    def out_of_corpus(self) -> list[AnswerRun]:
        return [run for run in self.answered if run.out_of_corpus]

    @property
    def citation_rate(self) -> float:
        """Over in-corpus questions only.

        A refusal scores 1.0 by construction, so including the out-of-corpus set
        would inflate this with the questions the system is supposed to decline.
        """
        rates = [
            run.answer.verification.citation_rate
            for run in self.in_corpus
            if run.answer is not None
        ]
        return statistics.fmean(rates) if rates else 0.0

    @property
    def refusal_rate(self) -> float:
        runs = self.out_of_corpus
        if not runs:
            return 0.0
        refused = sum(1 for run in runs if run.answer is not None and run.answer.refused)
        return refused / len(runs)

    @property
    def false_refusal_rate(self) -> float:
        """In-corpus questions the system declined. The cost of the guardrail."""
        runs = self.in_corpus
        if not runs:
            return 0.0
        return sum(
            1 for run in runs if run.answer is not None and run.answer.refused
        ) / len(runs)

    @property
    def hallucinated_rate(self) -> float:
        runs = self.answered
        if not runs:
            return 0.0
        return sum(
            1
            for run in runs
            if run.answer is not None and run.answer.verification.hallucinated_indices
        ) / len(runs)

    def latency(self, stage: str) -> tuple[int, int]:
        values = [
            int(run.answer.timing.as_dict()[stage])
            for run in self.answered
            if run.answer is not None
        ]
        if not values:
            return (0, 0)
        ordered = sorted(values)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        return (int(statistics.median(ordered)), p95)

    def render(self) -> str:
        failures = [run for run in self.runs if run.answer is None]
        lines = [
            f"questions        {len(self.runs)} "
            f"({len(self.in_corpus)} in corpus, {len(self.out_of_corpus)} out)",
            "",
            f"citation rate    {self.citation_rate:.1%}  "
            "(factual sentences carrying a marker, in-corpus only)",
            f"refusal rate     {self.refusal_rate:.1%}  "
            "(out-of-corpus questions declined; should be ~100%)",
            f"false refusals   {self.false_refusal_rate:.1%}  "
            "(in-corpus questions declined; the guardrail's cost)",
            f"hallucinated     {self.hallucinated_rate:.1%}  "
            "(answers citing an index never supplied; must be 0%)",
        ]
        if failures:
            lines.append(f"errors           {len(failures)}")

        lines += ["", f"{'stage':<12}{'p50':>8}{'p95':>8}"]
        for stage in ("retrieve_ms", "assemble_ms", "generate_ms", "verify_ms", "total_ms"):
            p50, p95 = self.latency(stage)
            lines.append(f"{stage.removesuffix('_ms'):<12}{p50:>8}{p95:>8}")

        ungrounded = [
            run
            for run in self.in_corpus
            if run.answer is not None and not run.answer.verification.grounded
        ]
        if ungrounded:
            lines += ["", f"{len(ungrounded)} in-corpus answers not fully grounded:"]
            for run in ungrounded[:10]:
                assert run.answer is not None
                unsupported = [
                    sentence.text
                    for sentence in run.answer.verification.sentences
                    if sentence.unsupported
                ]
                lines.append(
                    f"  {run.answer.verification.citation_rate:.0%}  {run.question}"
                )
                for sentence in unsupported[:2]:
                    lines.append(f"        uncited: {sentence[:100]}")

        answered_anyway = [
            run for run in self.out_of_corpus if run.answer is not None and not run.answer.refused
        ]
        if answered_anyway:
            lines += [
                "",
                f"{len(answered_anyway)} out-of-corpus questions were ANSWERED rather "
                "than declined — each one is a fabrication:",
            ]
            for run in answered_anyway:
                assert run.answer is not None
                lines.append(f"  {run.question}")
                lines.append(f"      {run.answer.answer[:160]}")

        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "questions": len(self.runs),
            "citation_rate": round(self.citation_rate, 4),
            "refusal_rate": round(self.refusal_rate, 4),
            "false_refusal_rate": round(self.false_refusal_rate, 4),
            "hallucinated_rate": round(self.hallucinated_rate, 4),
            "latency": {
                stage: dict(zip(("p50", "p95"), self.latency(stage), strict=True))
                for stage in (
                    "retrieve_ms",
                    "assemble_ms",
                    "generate_ms",
                    "verify_ms",
                    "total_ms",
                )
            },
            "answers": [
                {
                    "question": run.question,
                    "out_of_corpus": run.out_of_corpus,
                    "error": run.error,
                    **({} if run.answer is None else run.answer.as_dict()),
                }
                for run in self.runs
            ],
        }


def load_refusal_queries(path: Path = DEFAULT_REFUSAL_QUERIES) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [str(query) for query in payload.get("queries", [])]


async def evaluate_answers(
    ask: AnswerQuestion,
    *,
    questions: Sequence[str],
    refusals: Sequence[str],
    k: int = 10,
) -> AnswerReport:
    """Ask everything, and record what came back — including the failures.

    A provider error is recorded rather than raised, so one rate limit part way
    through does not discard the measurement of everything before it.
    """
    runs: list[AnswerRun] = []
    for question, out_of_corpus in [
        *((query, False) for query in questions),
        *((query, True) for query in refusals),
    ]:
        try:
            answer = await ask(question, k=k)
            runs.append(AnswerRun(question, out_of_corpus, answer))
        except (TransientError, PermanentError) as exc:
            runs.append(AnswerRun(question, out_of_corpus, None, error=str(exc)))

    report = AnswerReport(runs=runs)
    logger.info(
        "answers.evaluated",
        questions=len(runs),
        citation_rate=round(report.citation_rate, 3),
        refusal_rate=round(report.refusal_rate, 3),
        hallucinated_rate=round(report.hallucinated_rate, 3),
    )
    return report
