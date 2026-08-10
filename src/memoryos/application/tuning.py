"""Grid search over fusion weights, scored by the harness.

**Searched rather than hand-picked.** Adjusting weights until the results look
right is fitting a model to a sample of one, with the person doing the adjusting
as the loss function. A grid at least states its hypothesis space up front and
reports the whole surface, so a flat one is visible as flat.

**And it has no train/test split.** The grid is scored against the same 41
queries the baseline is scored against, so the winning combination is chosen
partly for fitting this particular corpus and these particular judgements.
There is no honest way around that at this size — holding out 8 queries would
make every score move by more than the effect being measured. So the number to
compare a winner against is not the runner-up, it is M2.3a's **resolution
floor**: a gain under 0.012 is the harness's own noise wearing a result's
clothing, and `format_grid` says so rather than leaving it to the reader.

Retrieval runs once per query, not once per combination. The candidates and
their signal scores do not depend on the weights — only the fusion does — so
each query is embedded and searched a single time and every combination is
re-fused over the cached candidates. That is what makes a fine grid a minute
rather than an afternoon.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.application.evaluate import _resolve_filters, _rows
from memoryos.application.golden import GoldenSet, excluded_by
from memoryos.application.metrics import EvalResult, score
from memoryos.application.ports import ScoredChunk
from memoryos.application.search import (
    CHUNK_FANOUT,
    FUSION_FANOUT,
    FusionWeights,
    MemoryHit,
    SearchMemories,
    SignalTable,
    fuse,
)
from memoryos.domain.fusion import DEFAULT_RRF_K
from memoryos.domain.values import MemoryKind

logger = structlog.get_logger(__name__)

# The smallest difference the harness can honestly detect, measured in M2.3a by
# perturbing the corpus four ways and watching what moved. Repetition and
# neutral edits moved nothing; adding one file that genuinely answers a query
# moved MRR by this much. A tuning gain under it is noise.
RESOLUTION_FLOOR = 0.0122

# Two grids. Coarse answers "do these signals do anything at all" in a minute;
# fine is for when the answer was yes. The retriever weights stay at 1.0 in both
# — M2.2 established that the two retrievers are complementary and roughly
# equally trustworthy, and this milestone is about the signals.
COARSE: dict[str, tuple[float, ...]] = {
    "recency": (0.0, 0.15, 0.3, 0.6),
    "importance": (0.0, 0.1, 0.2, 0.4),
}
FINE: dict[str, tuple[float, ...]] = {
    "recency": (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9),
    "importance": (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9),
}


@dataclass(frozen=True, slots=True)
class Candidates:
    """One query's retrieval, cached so every weight combination can reuse it."""

    query_text: str
    vector: list[ScoredChunk]
    keyword: list[ScoredChunk]
    signals: SignalTable
    metadata: dict[str, tuple[str, str | None, str, object, str]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class GridRow:
    weights: FusionWeights
    recall: float
    precision: float
    mrr: float
    ndcg: float

    def as_dict(self) -> dict[str, float]:
        return {
            **self.weights.as_dict(),
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "mrr": round(self.mrr, 6),
            "ndcg": round(self.ndcg, 6),
        }


async def collect_candidates(
    golden: GoldenSet,
    search: SearchMemories,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    k: int,
) -> list[Candidates]:
    """Run retrieval once per golden query and keep everything fusion needs."""
    wanted = max(k * FUSION_FANOUT, k * CHUNK_FANOUT, k)
    collected: list[Candidates] = []

    for query in golden.queries:
        filters, _ = await _resolve_filters(query.filters, session_factory)
        vector = await search.vector_candidates(query.query_text, k=wanted, filters=filters)
        keyword = await search.keyword_candidates(
            query.query_text, k=wanted, filters=filters
        )
        signals = await search.signals_for(vector, keyword)
        memory_ids = {chunk.memory_id for chunk in [*vector, *keyword]}
        metadata = {
            str(memory_id): row
            for memory_id, row in (await search.memory_metadata(list(memory_ids))).items()
        }
        collected.append(
            Candidates(
                query_text=query.query_text,
                vector=vector,
                keyword=keyword,
                signals=signals,
                metadata=metadata,
            )
        )

    logger.info("tuning.candidates_collected", queries=len(collected), k=k)
    return collected


def score_grid(
    golden: GoldenSet,
    candidates: Sequence[Candidates],
    *,
    k: int,
    grid: dict[str, tuple[float, ...]],
    base: FusionWeights,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[GridRow]:
    """Every weight combination, scored over the cached candidates."""
    by_text = {query.query_text: query for query in golden.queries}
    rows: list[GridRow] = []

    for recency, importance in itertools.product(grid["recency"], grid["importance"]):
        weights = FusionWeights(
            vector=base.vector,
            keyword=base.keyword,
            recency=recency,
            importance=importance,
        )
        results = [
            _score_one(by_text[found.query_text], found, golden, k=k, weights=weights, rrf_k=rrf_k)
            for found in candidates
            if found.query_text in by_text
        ]
        if not results:
            continue
        rows.append(
            GridRow(
                weights=weights,
                recall=_mean(r.recall for r in results),
                precision=_mean(r.precision for r in results),
                mrr=_mean(r.mrr for r in results),
                ndcg=_mean(r.ndcg for r in results),
            )
        )

    # nDCG first because it is the metric that accounts for the whole ordering,
    # which is what a weight change actually moves. Ties broken by MRR then by
    # the weights themselves, so a rerun produces the same table.
    rows.sort(
        key=lambda row: (-row.ndcg, -row.mrr, row.weights.recency, row.weights.importance)
    )
    return rows


def _score_one(
    query: object,
    found: Candidates,
    golden: GoldenSet,
    *,
    k: int,
    weights: FusionWeights,
    rrf_k: int,
) -> EvalResult:
    chunks = fuse(
        found.vector,
        found.keyword,
        signals=found.signals,
        weights=weights,
        rrf_k=rrf_k,
    )
    hits = _group(chunks, found.metadata)
    kept = [
        hit for hit in hits if excluded_by(golden.exclude, hit.external_key) is None
    ][:k]
    rows = _rows(query, kept)  # type: ignore[arg-type]
    return score(
        found.query_text,
        [row.key for row in rows],
        query.relevant_keys,  # type: ignore[attr-defined]
        k,
    )


def _group(
    chunks: Sequence[ScoredChunk],
    metadata: dict[str, tuple[str, str | None, str, object, str]],
) -> list[MemoryHit]:
    """The same chunks-to-memories collapse the search use case performs.

    Duplicated deliberately rather than reached for through `SearchMemories`,
    which would re-query the database once per weight combination. The shape it
    has to match is asserted by the "signal weight 0 reproduces M2.2 hybrid"
    test, which compares this path against the real one.
    """
    grouped: dict[str, list[ScoredChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(str(chunk.memory_id), []).append(chunk)

    hits: list[MemoryHit] = []
    for memory_id, matched in grouped.items():
        row = metadata.get(memory_id)
        if row is None:
            continue
        external_key, title, kind, occurred_at, source_name = row
        hits.append(
            MemoryHit(
                memory_id=matched[0].memory_id,
                external_key=external_key,
                source_name=source_name,
                title=title,
                kind=MemoryKind(kind),
                occurred_at=occurred_at,
                score=max(chunk.score for chunk in matched),
                matched_chunks=sorted(matched, key=lambda chunk: chunk.ordinal),
            )
        )

    hits.sort(
        key=lambda hit: (
            hit.score,
            sum(chunk.score for chunk in hit.matched_chunks) / len(hit.matched_chunks),
        ),
        reverse=True,
    )
    return hits


def format_grid(
    rows: Sequence[GridRow],
    *,
    baseline: GridRow | None,
    floor: float,
    top: int = 5,
) -> str:
    """The top combinations, with every gain judged against the floor."""
    lines = [
        f"{'recency':>8}{'import':>8}{'nDCG@10':>10}{'MRR':>9}{'recall':>9}{'prec':>9}"
    ]
    for row in rows[:top]:
        lines.append(
            f"{row.weights.recency:>8.2f}{row.weights.importance:>8.2f}"
            f"{row.ndcg:>10.4f}{row.mrr:>9.4f}{row.recall:>9.4f}{row.precision:>9.4f}"
        )

    if baseline is None or not rows:
        return "\n".join(lines)

    best = rows[0]
    gain = best.ndcg - baseline.ndcg
    lines += [
        "",
        f"signals off (recency 0, importance 0): nDCG {baseline.ndcg:.4f}",
        f"best combination:                      nDCG {best.ndcg:.4f}  ({gain:+.4f})",
        "",
    ]
    if abs(gain) < floor:
        lines.append(
            f"That difference is smaller than the measured resolution floor "
            f"({floor:.4f}). It is not evidence. Reporting it as an improvement "
            f"would be reporting noise."
        )
    else:
        lines.append(
            f"That exceeds the resolution floor ({floor:.4f}), so it is a real "
            f"difference on this corpus — though tuned and evaluated on the same "
            f"41 queries, with no held-out split."
        )
    return "\n".join(lines)


def baseline_row(rows: Sequence[GridRow]) -> GridRow | None:
    """The combination with both signals off: M2.2 hybrid, inside this grid."""
    for row in rows:
        if row.weights.signals_are_off:
            return row
    return None


def _mean(values: object) -> float:
    collected = list(values)  # type: ignore[call-overload]
    return sum(collected) / len(collected) if collected else 0.0


__all__ = [
    "COARSE",
    "FINE",
    "Candidates",
    "GridRow",
    "baseline_row",
    "collect_candidates",
    "format_grid",
    "score_grid",
]
