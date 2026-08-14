"""Reciprocal rank fusion. Pure Python, no I/O, no knowledge of retrieval.

## Why ranks and not scores

The obvious way to combine two retrievers is a weighted sum of their scores:
`0.7 * cosine + 0.3 * ts_rank_cd`. It does not work, and the reason is not
tuning. The two numbers are not on comparable scales and cannot be made
comparable by any fixed transformation:

- Cosine similarity from bge-small occupies a narrow band — M1.6.1 measured this
  corpus at roughly 0.45 to 0.80, with the third on-topic document beating the
  first off-topic one by about 0.006. Almost all of the range carries no signal.
- `ts_rank_cd` is unbounded above and depends on term frequencies across the
  whole corpus, so the same document scores differently after ingesting
  unrelated files.

Any weighting has to normalise first, and every normalisation — min-max over the
result set, z-scores, dividing by the top score — encodes an assumption about
the score distribution that stops holding as the corpus grows. The failure is
silent: the ranking degrades and nothing errors.

RRF throws the scores away and keeps only the ordering, which is the part both
retrievers agree about the meaning of. Rank 1 means the same thing to both.

## The formula

    score(d) = Σ  weight_i / (k + rank_i(d))

over every ranking that contains `d`, with ranks starting at 1.

`k = 60` comes from Cormack, Clarke and Buettcher (2009), and its job is to
flatten the curve. At k=60 rank 1 contributes 1/61 and rank 2 contributes 1/62 —
a 1.6% difference — so being first in one ranking is worth far less than
appearing high in both. That is the behaviour the whole method exists for: a
document ranked third by both retrievers (1/63 + 1/63 = 0.0317) beats one ranked
first by only one (1/61 = 0.0164). Agreement outweighs enthusiasm.

Left configurable, but **do not tune it before M2.3.** With two retrievers and
21 golden queries there is not enough signal to distinguish a real improvement
from a fit to this particular corpus.

`weights` exists so M2.3 can add recency and importance as additional rankings
rather than as a second, differently-shaped mechanism bolted on beside this one.
"""

from collections.abc import Sequence

# Cormack et al. 2009. Flattens the contribution curve so that agreement between
# retrievers matters more than the exact position within any one of them.
DEFAULT_RRF_K = 60


def contribution(rank: int, *, k: int = DEFAULT_RRF_K) -> float:
    """What one ranking adds to a fused score by placing an item at `rank`.

    The single term of the sum below, exposed because two callers outside this
    module need to reason about a *part* of a fused score rather than the whole
    of it. M6.3's gate is the reason it exists: it asks whether the routes that
    are actually about the focus agree, which means adding up the terms for some
    rankings and not others.

    Extracted rather than repeated. A second copy of `1 / (k + rank)` somewhere
    else is a copy that stops matching the day `k` moves, and it would fail
    silently — the gate would go on comparing against a threshold expressed in
    units of a formula the fusion no longer uses.
    """
    if rank < 1:
        raise ValueError(f"rank is 1-based, got {rank}")
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists. Returns (id, score) sorted descending.

    An empty ranking contributes nothing rather than penalising the documents
    the other retrievers found — which is what makes a stopword query, where the
    keyword half legitimately returns nothing, degrade to pure vector instead of
    scoring everything to zero.

    Ties break on the ID, ascending. Deterministic on purpose: two documents with
    identical fused scores are common — every document found at the same rank by
    the same set of retrievers ties exactly — and without a total order the same
    query returns the same set in a different sequence run to run, which the
    evaluation harness would report as change.
    """
    if k < 0:
        # k = 0 is legal and means "no flattening": rank 1 contributes 1.0.
        # Negative k puts a pole at rank -k and inverts the ordering around it.
        raise ValueError(f"rrf k must be >= 0, got {k}")

    resolved = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(resolved) != len(rankings):
        raise ValueError(
            f"got {len(resolved)} weights for {len(rankings)} rankings; "
            "they have to correspond one to one"
        )

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, resolved, strict=True):
        seen: set[str] = set()
        for position, identifier in enumerate(ranking, start=1):
            # A repeated id in one ranking keeps its best position. Counting it
            # twice would let one retriever outvote the other by returning
            # duplicates.
            if identifier in seen:
                continue
            seen.add(identifier)
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + position)

    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
