"""Running the golden set through search, and reporting what came back.

Measurement only. Nothing here changes how anything ranks — if a number moves in
this milestone, something is wrong. The search path used is the one the CLI and
the UI use, because a harness that scores its own private retrieval path scores
nothing.

The output is arranged around one belief: **the mean tells you whether you
improved, and the worst list tells you what to fix.** Aggregates go at the top
because they are what gets pasted into a milestone report, but the section
anybody actually works from is the worst queries by MRR, and `--verbose` exists
so that a bad score can be diagnosed instead of merely observed.
"""

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.golden import GoldenQuery, GoldenSet
from memoryos.application.metrics import EvalResult, score
from memoryos.application.ports import SearchFilters
from memoryos.application.search import MemoryHit, SearchMemories
from memoryos.domain.values import DEFAULT_SEARCH_MODE, MemoryKind, SearchMode

logger = structlog.get_logger(__name__)

# The four, in the order they are reported. Label, then the attribute on
# EvalResult. Kept in one place so the table, the JSON, and the comparison
# cannot drift into disagreeing about what was measured.
METRICS: tuple[tuple[str, str], ...] = (
    ("recall@{k}", "recall"),
    ("precision@{k}", "precision"),
    ("MRR", "mrr"),
    ("nDCG@{k}", "ndcg"),
)


@dataclass(frozen=True, slots=True)
class RetrievedRow:
    """One line of a ranking, with everything `--verbose` needs to explain it."""

    rank: int
    key: str
    external_key: str
    score: float
    matched_ordinals: list[int]
    relevant: bool


@dataclass(frozen=True, slots=True)
class QueryRun:
    result: EvalResult
    rows: list[RetrievedRow]
    filters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Summary:
    metric: str
    mean: float
    p50: float
    minimum: float


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    k: int
    ran_at: str
    runs: list[QueryRun]
    golden: GoldenSet
    # Which retriever produced these numbers. Recorded because M2.1 and M2.2 are
    # several runs over one golden set, and a comparison that silently crossed
    # modes would read as a catastrophic regression.
    mode: SearchMode = DEFAULT_SEARCH_MODE
    # Present when the corpus could not be reached in the shape the golden set
    # expects; carried rather than raised so a partial run still reports.
    warnings: list[str] = field(default_factory=list)

    @property
    def results(self) -> list[EvalResult]:
        return [run.result for run in self.runs]

    def summaries(self) -> list[Summary]:
        summaries: list[Summary] = []
        for label, attribute in METRICS:
            values = [float(getattr(result, attribute)) for result in self.results]
            summaries.append(
                Summary(
                    metric=label.format(k=self.k),
                    mean=_mean(values),
                    p50=_median(values),
                    minimum=min(values, default=0.0),
                )
            )
        return summaries

    def worst(self, count: int) -> list[QueryRun]:
        """Lowest MRR first, ties broken by nDCG then by text, so a rerun agrees."""
        return sorted(
            self.runs,
            key=lambda run: (run.result.mrr, run.result.ndcg, run.result.query_text),
        )[:count]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ran_at": self.ran_at,
            "k": self.k,
            "mode": self.mode.value,
            "golden_generated_at": self.golden.generated_at,
            "queries": len(self.runs),
            "summary": {
                summary.metric: {
                    "mean": round(summary.mean, 6),
                    "p50": round(summary.p50, 6),
                    "min": round(summary.minimum, 6),
                }
                for summary in self.summaries()
            },
            "results": [run.result.as_dict() for run in self.runs],
            "excluded": [
                {"query_text": item.query_text, "reason": item.reason}
                for item in self.golden.excluded
            ],
            "unresolved": [
                {
                    "query_text": item.query_text,
                    "item": item.item.key,
                    "reason": item.reason,
                }
                for item in self.golden.unresolved
            ],
            "warnings": list(self.warnings),
        }


async def evaluate(
    golden: GoldenSet,
    search: SearchMemories,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    k: int,
    now: datetime,
    mode: SearchMode = DEFAULT_SEARCH_MODE,
) -> EvaluationRun:
    """Score every golden query. One search per query, through the real path."""
    runs: list[QueryRun] = []
    warnings: list[str] = []

    for query in golden.queries:
        filters, filter_warnings = await _resolve_filters(query.filters, session_factory)
        warnings.extend(f"{query.query_text}: {note}" for note in filter_warnings)

        result = await search(query.query_text, k=k, filters=filters, mode=mode)
        rows = _rows(query, result.hits)
        runs.append(
            QueryRun(
                result=score(
                    query.query_text,
                    [row.key for row in rows],
                    query.relevant_keys,
                    k,
                ),
                rows=rows,
                filters=dict(query.filters),
            )
        )

    logger.info("evaluate.finished", queries=len(runs), k=k, mode=mode.value)
    return EvaluationRun(
        k=k,
        ran_at=now.isoformat(),
        runs=runs,
        golden=golden,
        mode=mode,
        warnings=warnings,
    )


def _rows(query: GoldenQuery, hits: Sequence[MemoryHit]) -> list[RetrievedRow]:
    relevant = query.relevant_keys
    rows: list[RetrievedRow] = []
    for rank, hit in enumerate(hits, start=1):
        ordinals = sorted(chunk.ordinal for chunk in hit.matched_chunks)
        key = query.project(hit.source_name, hit.external_key, ordinals)
        rows.append(
            RetrievedRow(
                rank=rank,
                key=key,
                external_key=hit.external_key,
                score=hit.score,
                matched_ordinals=ordinals,
                relevant=key in relevant,
            )
        )
    return rows


async def _resolve_filters(
    filters: Mapping[str, Any], session_factory: async_sessionmaker[AsyncSession]
) -> tuple[SearchFilters, list[str]]:
    """Turn the filters a judgement was recorded under into search filters.

    A verdict is only meaningful under the filters it was given, so the search
    that scores it has to run under them too. `k` and `exact` from the recorded
    blob are deliberately ignored: `k` is the harness's parameter and comparing
    runs at different cutoffs is meaningless, and `exact` would evaluate a code
    path no user's search takes.
    """
    names = [str(name) for name in (filters.get("sources") or [])]
    kind_name = filters.get("kind")
    notes: list[str] = []

    source_ids = None
    if names:
        async with session_factory() as session:
            found = {
                row[0]: row[1]
                for row in await session.execute(
                    select(models.Source.name, models.Source.id).where(
                        models.Source.name.in_(names)
                    )
                )
            }
        missing = sorted(set(names) - set(found))
        if missing:
            notes.append(f"filtered on unknown source(s) {', '.join(missing)}")
        source_ids = list(found.values())

    kinds = None
    if kind_name:
        try:
            kinds = [MemoryKind(str(kind_name))]
        except ValueError:
            notes.append(f"ignored unknown kind filter {kind_name!r}")

    return SearchFilters(source_ids=source_ids, kinds=kinds), notes


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def format_report(run: EvaluationRun, *, worst: int = 3) -> str:
    lines = [f"queries: {len(run.runs)}   |  k={run.k}   |  mode={run.mode.value}", ""]
    lines.append(f"{'':<14}{'mean':>8}{'p50':>8}{'min':>8}")
    for summary in run.summaries():
        lines.append(
            f"{summary.metric:<14}{summary.mean:>8.3f}{summary.p50:>8.3f}"
            f"{summary.minimum:>8.3f}"
        )

    if run.runs:
        lines += ["", f"worst {min(worst, len(run.runs))} queries by MRR:"]
        for query_run in run.worst(worst):
            lines.append(
                f"  {query_run.result.mrr:.2f}  {query_run.result.query_text}"
            )

    lines += _notes(run)
    return "\n".join(lines)


def format_verbose(run: EvaluationRun) -> str:
    """Every ranking, marked. What turns a bad score into a diagnosis."""
    blocks: list[str] = []
    for query_run in run.runs:
        result = query_run.result
        header = (
            f"{result.query_text}\n"
            f"  recall {result.recall:.3f}  precision {result.precision:.3f}  "
            f"MRR {result.mrr:.3f}  nDCG {result.ndcg:.3f}"
        )
        lines = [header]
        if not query_run.rows:
            lines.append("    (nothing retrieved)")
        for row in query_run.rows:
            mark = "*" if row.relevant else " "
            chunks = ",".join(str(ordinal) for ordinal in row.matched_ordinals)
            lines.append(
                f"  {mark} {row.rank:>2}. {row.score:.4f}  {row.external_key}  "
                f"[chunks {chunks}]"
            )
        for missed in sorted(result.missed):
            lines.append(f"    missed: {missed}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _notes(run: EvaluationRun) -> list[str]:
    lines: list[str] = []
    if run.golden.excluded:
        lines += ["", f"excluded {len(run.golden.excluded)} unscoreable queries:"]
        lines += [
            f"  {item.query_text}  ({item.reason})" for item in run.golden.excluded
        ]
    if run.golden.unresolved:
        lines += ["", f"unresolved {len(run.golden.unresolved)} judged items:"]
        lines += [f"  {item.describe()}" for item in run.golden.unresolved]
    if run.warnings:
        lines += ["", "warnings:"] + [f"  {note}" for note in run.warnings]
    return lines


# --------------------------------------------------------------------------
# Comparison
#
# Every later Phase 2 milestone reports this diff against the baseline. That is
# what the baseline is for, and why it is committed rather than regenerated.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Comparison:
    deltas: list[tuple[str, float, float]]
    regressions: list[tuple[str, float, float]]
    added: list[str]
    removed: list[str]
    # Set when the two runs used different retrievers. Not an error — comparing
    # keyword against the vector baseline is exactly what M2.1 does — but the
    # numbers below then describe two systems rather than one change, and a
    # reader has to be told which they are looking at.
    modes: tuple[str, str] | None = None

    def render(self) -> str:
        lines: list[str] = []
        if self.modes is not None:
            baseline_mode, current_mode = self.modes
            lines += [
                f"NOTE: different retrievers — baseline is {baseline_mode}, this run "
                f"is {current_mode}. These are two systems, not one change.",
                "",
            ]
        lines.append(f"{'':<14}{'baseline':>10}{'now':>10}{'delta':>10}")
        for metric, before, after in self.deltas:
            # Rounded before formatting, and negative zero flattened. The
            # baseline is stored to six places and recomputed in full precision,
            # so an unchanged run otherwise prints "-0.000" — which reads as a
            # regression to anybody skimming a milestone report.
            delta = round(after - before, 6) or 0.0
            lines.append(f"{metric:<14}{before:>10.3f}{after:>10.3f}{delta:>+10.3f}")
        if self.regressions:
            lines += ["", f"{len(self.regressions)} queries lost MRR:"]
            lines += [
                f"  {before:.3f} -> {after:.3f}  {query}"
                for query, before, after in self.regressions
            ]
        else:
            lines += ["", "no query lost MRR"]
        if self.added:
            lines += ["", "new since the baseline:"] + [f"  {q}" for q in self.added]
        if self.removed:
            lines += ["", "absent from this run:"] + [f"  {q}" for q in self.removed]
        return "\n".join(lines)


def compare(baseline: Mapping[str, Any], run: EvaluationRun) -> Comparison:
    """Per-metric deltas, and every query whose MRR dropped.

    Metrics are matched by their rendered label, so a comparison across two
    different `k` values shows up as "new metric" rather than as a delta between
    `recall@10` and `recall@20` — numbers that have no business being subtracted.
    """
    before_summary = baseline.get("summary", {})
    now_summary = {summary.metric: summary for summary in run.summaries()}

    deltas = [
        (metric, float(before_summary[metric]["mean"]), now_summary[metric].mean)
        for metric in now_summary
        if metric in before_summary
    ]

    before_mrr = {
        str(entry["query_text"]): float(entry["mrr"])
        for entry in baseline.get("results", [])
    }
    now_mrr = {run_.result.query_text: run_.result.mrr for run_ in run.runs}

    # Compared at the precision the baseline was written with. Without the
    # tolerance an unchanged run reports regressions of 1e-16, and a regression
    # list that cries wolf is one nobody reads.
    regressions = sorted(
        (
            (query, before_mrr[query], after)
            for query, after in now_mrr.items()
            if query in before_mrr and round(after - before_mrr[query], 6) < 0
        ),
        key=lambda row: row[2] - row[1],
    )

    # Older runs predate the field; absent means vector, which is what they were.
    before_mode = str(baseline.get("mode", SearchMode.VECTOR.value))

    return Comparison(
        deltas=deltas,
        regressions=regressions,
        added=sorted(set(now_mrr) - set(before_mrr)),
        removed=sorted(set(before_mrr) - set(now_mrr)),
        modes=(
            None
            if before_mode == run.mode.value
            else (before_mode, run.mode.value)
        ),
    )


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0
