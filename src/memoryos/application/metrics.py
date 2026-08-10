"""Retrieval metrics. Pure functions, no I/O, no knowledge of this corpus.

Four metrics rather than one, because each is blind to something the next one
sees, and a single number would hide exactly the failure it was chosen not to
notice:

- **recall@k** — of the things that should have been found, how many were? Says
  nothing about order: a relevant result at rank 10 counts as much as one at
  rank 1.
- **precision@k** — of what came back, how much was relevant? Penalises noise,
  which recall rewards you for adding.
- **MRR** — how high did the *first* correct result rank? The one that matters
  when the reader reads one answer and stops.
- **nDCG@k** — the whole ordering, discounted by position. The most complete and
  the hardest to reason about, which is why it is not the only one here.

Relevance is binary: an item is in `relevant` or it is not. Graded relevance is
a later refinement, and nDCG is defined for binary gains without any special
casing — the gain is simply 1 or 0.

**Empty inputs return 0.0 rather than raising.** A query whose relevant set is
empty cannot be scored, and `application/golden.py` drops those before they get
here; if one arrives anyway it must not take the whole run down with a
ZeroDivisionError.

Every function treats `retrieved` as a ranking of *distinct* items and counts
each relevant item at most once. A duplicate in the list is a bug upstream, but
one that would otherwise inflate nDCG while leaving recall unmoved — a
disagreement between two metrics computed from the same list is the worst way to
find out.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant items that appear in the top k."""
    if not relevant or k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that is relevant.

    Divided by `k`, not by how many results actually came back. A denominator
    that shrinks when retrieval returns fewer than k would make a query that
    returned three results look more precise than one that returned ten, and the
    only reason this milestone exists is to compare numbers across runs.
    """
    if not relevant or k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / the rank of the first relevant item; 0.0 if there is none.

    Uncapped by k on purpose: the signature has no k because the question is
    "how far down is the first right answer", and truncating the list to answer
    it would report 0.0 for a query whose answer sat at rank 11 — which is a
    meaningfully different failure from one where the answer is not there at
    all. Callers that want it capped can slice.
    """
    if not relevant:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Discounted cumulative gain over the top k, normalised by the best possible.

    The ideal ranking puts every relevant item first, so the denominator is the
    DCG of `min(len(relevant), k)` hits in a row. That is what makes 1.0 mean
    "as good as this query could have scored" rather than "perfect in the
    abstract": a query with 20 relevant items cannot be faulted at k=10 for
    returning only 10 of them.
    """
    if not relevant or k <= 0:
        return 0.0

    gained: set[str] = set()
    dcg = 0.0
    for position, item in enumerate(retrieved[:k], start=1):
        if item in relevant and item not in gained:
            gained.add(item)
            # log2(position + 1): rank 1 is undiscounted, and the penalty for
            # each step down shrinks — the gap between rank 1 and 2 matters more
            # than the gap between 9 and 10.
            dcg += 1.0 / math.log2(position + 1)

    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


@dataclass(frozen=True, slots=True)
class EvalResult:
    """One query's score, with enough context to diagnose it.

    The retrieved list is carried alongside the numbers because a score on its
    own says a query is bad without saying why, and `--verbose` printing what
    actually came back is what turns a run from a report into a starting point.
    """

    query_text: str
    k: int
    # In rank order, one entry per result. The key space is the harness's —
    # `source::external_key` for a memory, `source::external_key#ordinal` when
    # the golden set pins a chunk. See `application/golden.py`.
    retrieved: list[str]
    relevant: set[str]
    recall: float
    precision: float
    mrr: float
    ndcg: float
    # Retained so an unmeasurable query can be explained rather than only
    # excluded — see `golden.py`.
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> set[str]:
        return set(self.retrieved[: self.k]) & self.relevant

    @property
    def missed(self) -> set[str]:
        return self.relevant - self.found

    def as_dict(self) -> dict[str, object]:
        return {
            "query_text": self.query_text,
            "k": self.k,
            "recall": round(self.recall, 6),
            "precision": round(self.precision, 6),
            "mrr": round(self.mrr, 6),
            "ndcg": round(self.ndcg, 6),
            "retrieved": list(self.retrieved),
            "relevant": sorted(self.relevant),
            "missed": sorted(self.missed),
            "notes": list(self.notes),
        }


def score(query_text: str, retrieved: Sequence[str], relevant: set[str], k: int) -> EvalResult:
    """All four metrics over one ranking. The only place they are computed together."""
    return EvalResult(
        query_text=query_text,
        k=k,
        retrieved=list(retrieved),
        relevant=set(relevant),
        recall=recall_at_k(retrieved, relevant, k),
        precision=precision_at_k(retrieved, relevant, k),
        mrr=reciprocal_rank(retrieved, relevant),
        ndcg=ndcg_at_k(retrieved, relevant, k),
    )
