"""Why a result is where it is, reconstructed from the numbers that put it there.

Pure Python, and deliberately not a language model. The explanation has to be
available on every result of every query, cost nothing, and say the same thing
twice for the same inputs — three properties a generated sentence does not have.
It is also the only kind of explanation that can be *wrong in a detectable way*:
the shares below are recomputed from the same `1/(k+rank)` terms that produced
the fused score, so if they stop summing to 1.0 the reconstruction has drifted
from the fusion and a test says so.

**`share` is the number that answers the question.** Raw ranks and scores say
what happened; the share says which retriever is responsible for this result
being here, as a percentage, which is what somebody looking at a surprising
third place actually wants to know.
"""

from dataclasses import dataclass

# Rank bands, for turning a position into a word. Chosen to match how a reader
# reads a result list rather than anything statistical: the first few are what
# gets looked at, the top ten are what gets returned, past that is noise.
_STRONG = 3
_MODERATE = 10

# The graph ranking's name in `ranks`, matched when assembling the sentence so the
# route can be appended to its clause. A constant because two places compare it.
_GRAPH = "graph"

# The retrievers. A result found by neither of them was introduced by something
# else, which for now can only be the graph.
_RETRIEVERS = frozenset({"semantic", "keyword"})


@dataclass(frozen=True, slots=True)
class SignalContribution:
    """One ranking's part in a fused score."""

    name: str
    rank: int
    # The retriever's own score, on its own scale, or None for a signal that has
    # no score of its own beyond the ordering.
    score: float | None
    weight: float
    # `weight / (rrf_k + rank)`: this ranking's term in the sum.
    contribution: float
    # That term as a fraction of the fused score. Sums to 1.0 across
    # contributions, which is asserted rather than assumed.
    share: float

    def describe(self) -> str:
        return f"{_strength(self.rank)} {self.name} match (rank {self.rank})"


@dataclass(frozen=True, slots=True)
class RankExplanation:
    final_rank: int
    fused_score: float
    contributions: list[SignalContribution]
    rerank_score: float | None
    why: str
    # The entity route that reached this result, when the graph is what put it
    # here. M3.5's explainability guardrail: expansion is the one ranking that
    # introduces a result rather than reordering one, so it is the one whose
    # contribution a reader cannot reconstruct from the text in front of them.
    graph_path: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "final_rank": self.final_rank,
            "fused_score": round(self.fused_score, 6),
            "rerank_score": (
                None if self.rerank_score is None else round(self.rerank_score, 4)
            ),
            "why": self.why,
            "graph_path": (
                None if self.graph_path is None else " -> ".join(self.graph_path)
            ),
            "contributions": [
                {
                    "name": item.name,
                    "rank": item.rank,
                    "score": None if item.score is None else round(item.score, 4),
                    "weight": item.weight,
                    "contribution": round(item.contribution, 6),
                    "share": round(item.share, 4),
                }
                for item in self.contributions
            ],
        }


def build_explanation(
    *,
    final_rank: int,
    fused_score: float,
    ranks: dict[str, tuple[int, float | None, float]],
    rrf_k: int,
    rerank_score: float | None = None,
    previous_rank: int | None = None,
    graph_path: tuple[str, ...] | None = None,
) -> RankExplanation:
    """Reconstruct the fusion arithmetic for one result.

    `ranks` maps a ranking's name to `(rank, its own score, its weight)`. Only
    rankings that actually found the chunk appear — a null rank means the
    retriever never returned it, which is different from returning it last, and
    an explanation that listed it at 0% would suggest otherwise.

    `previous_rank` is this *result's* position before reranking, not the
    cross-encoder's position for its best chunk. The two are different units —
    one counts memories, the other counts chunks in a shortlist — and comparing
    them would produce a confident sentence about a movement that never
    happened.
    """
    contributions: list[tuple[str, int, float | None, float, float]] = []
    for name, (rank, score, weight) in ranks.items():
        if weight == 0.0:
            # A ranking switched off contributed nothing and does not belong in
            # a list of reasons.
            continue
        contributions.append((name, rank, score, weight, weight / (rrf_k + rank)))

    total = sum(item[4] for item in contributions)
    resolved = [
        SignalContribution(
            name=name,
            rank=rank,
            score=score,
            weight=weight,
            contribution=contribution,
            # Guarded rather than assumed: a fused score of zero means no
            # ranking found it, and this is an explanation, not a place to
            # raise.
            share=(contribution / total) if total else 0.0,
        )
        for name, rank, score, weight, contribution in sorted(
            contributions, key=lambda item: -item[4]
        )
    ]

    return RankExplanation(
        final_rank=final_rank,
        fused_score=fused_score,
        contributions=resolved,
        rerank_score=rerank_score,
        graph_path=graph_path,
        why=_why(final_rank, resolved, previous_rank, graph_path),
    )


def _why(
    final_rank: int,
    contributions: list[SignalContribution],
    previous_rank: int | None,
    graph_path: tuple[str, ...] | None = None,
) -> str:
    """One plain sentence, assembled from the numbers.

    Ordered by share, so the first clause is always the reason that mattered
    most. The reranking clause names the *movement* rather than the score,
    because a cross-encoder logit means nothing to a reader while "up from 5th"
    means exactly what it says.

    The graph clause names the *route*, for a reason the others do not need: a
    result no retriever found shares no word with the query, so "weak graph match"
    tells a reader that something connected it and not what. `via queue -> SKIP
    LOCKED` is checkable; a rank is not.
    """
    if not contributions:
        return f"Ranked {_ordinal(final_rank)}: no ranking signal found this result."

    parts = [
        item.describe()
        + (
            f" via {' -> '.join(graph_path)}"
            if item.name == _GRAPH and graph_path
            else ""
        )
        for item in contributions
    ]
    sentence = f"Ranked {_ordinal(final_rank)}: {', '.join(parts)}"

    if _introduced_by_the_graph(contributions):
        # Stated outright rather than left to be inferred from the absence of the
        # other clauses. "Neither retriever found this" is the single most
        # important fact about such a result, and a reader scanning shares would
        # have to notice that two names are missing to learn it.
        sentence += ", found by the graph alone"

    if previous_rank is not None and previous_rank != final_rank:
        direction = "up" if previous_rank > final_rank else "down"
        sentence += f", reranked {direction} from {_ordinal(previous_rank)}"
    elif previous_rank is not None:
        sentence += ", unchanged by reranking"

    return sentence + "."


def _introduced_by_the_graph(contributions: list[SignalContribution]) -> bool:
    names = {item.name for item in contributions}
    return _GRAPH in names and not (names & _RETRIEVERS)


def _strength(rank: int) -> str:
    if rank <= _STRONG:
        return "strong"
    if rank <= _MODERATE:
        return "moderate"
    return "weak"


def _ordinal(value: int) -> str:
    # 11th, 12th and 13th are the exceptions the last-digit rule gets wrong.
    teens = 10 <= value % 100 <= 20
    suffix = "th" if teens else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"
