"""Choosing which items fit, when everything relevant does not.

**This milestone is a budgeting problem wearing a retrieval problem's clothes.**
Five phases went into finding relevant material; retrieval returning fifty
relevant memories is useless when eight fit, and getting *which eight* wrong
wastes all of it. Everything here is the choosing, with no I/O, so the rules can
be checked against a hand-worked example rather than against a corpus.

Three rules, and each defends against a different way a top-k list goes wrong.

**Relevance alone returns k variations of one idea.** The eight highest-scoring
chunks of a codebase discussion are frequently eight paragraphs about the same
function, and they crowd out the three distinct perspectives that would have fit
in the same budget. `maximal_marginal_relevance` trades a little relevance for
coverage: each pick is scored on how relevant it is *minus* how much it repeats
what has already been picked.

**Similarity is not the only kind of sameness.** Twelve code files can be
mutually dissimilar by cosine and still be a worse answer than nine files, a
decision and a discussion. MMR cannot see that, because it only knows what the
embedder knows. So there is a cap per category as well, applied as a hard limit
during selection rather than as a filter afterwards — filtering afterwards
throws away the budget the rejected items were already counted against.

**An item that does not fit is dropped whole.** M2.6's rule, and it matters more
here: a truncated memory may lose the very thing that made it relevant, and
nothing downstream can tell that it was cut. The loop continues past a
too-large item rather than stopping, because a short item further down may still
fit where a long one did not.

The order those three are applied in is not arbitrary. Caps and the budget are
constraints — an item either violates them or does not — while MMR is a
*preference* over what is still admissible. So MMR proposes and the constraints
dispose, on every pick, rather than MMR producing a list that constraints then
edit. Editing afterwards would leave the k+1th item unconsidered when the kth was
rejected for a cap it never came close to.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto

from memoryos.domain.values import MemoryKind


class ContextCategory(StrEnum):
    """What kind of thing an item is, for the purpose of not returning twelve.

    Coarser than `MemoryKind` on purpose. The cap exists to stop one *sort* of
    material crowding out another, and a reader does not experience a note and a
    document as different sorts — they experience "prose about the system" versus
    "the code" versus "a decision somebody recorded". Splitting those finely
    would give each a cap so generous that nothing ever hits one.

    `DECISION` is the category with no `MemoryKind` behind it, and it is the
    reason this enum exists rather than reusing that one. A decision is not a
    memory: it has no chunks, no embedding and no place in the corpus, and it is
    frequently the single most useful item in a context.
    """

    CODE = auto()
    PROSE = auto()
    DECISION = auto()
    OTHER = auto()


# How a memory's kind maps onto the coarse category. Absent kinds fall to
# `OTHER`, which is deliberate: a new `MemoryKind` should not silently join
# whichever category happens to be listed first.
_CATEGORY_OF: Mapping[MemoryKind, ContextCategory] = {
    MemoryKind.CODE: ContextCategory.CODE,
    MemoryKind.NOTE: ContextCategory.PROSE,
    MemoryKind.DOCUMENT: ContextCategory.PROSE,
    MemoryKind.EMAIL: ContextCategory.PROSE,
    MemoryKind.MEETING: ContextCategory.PROSE,
    MemoryKind.COMMIT: ContextCategory.CODE,
}


def category_of(kind: MemoryKind) -> ContextCategory:
    return _CATEGORY_OF.get(kind, ContextCategory.OTHER)


# How many of any one category may appear, as a fraction of `max_items`.
#
# **A fraction rather than a count, because the cap has to mean the same thing at
# every budget.** A literal cap of five is most of a request for six items and
# almost none of a request for thirty; the failure it exists to prevent — one
# category crowding out the rest — is proportional, so the cap is too.
#
# 0.5 rather than 1/len(categories). An even split would be a different rule:
# it would force variety rather than prevent monoculture, and on a focus that
# genuinely is about code it would hold back the code to make room for prose
# that is not as good. Half is the point at which one category stops being able
# to *dominate* while still being allowed to lead.
DEFAULT_CATEGORY_SHARE = 0.5

# How much of the score is relevance and how much is novelty.
#
# **0.5, and 0.7 was tried first and does not work.** The obvious instinct is to
# lean towards relevance — the cost of an irrelevant item is that a reader stops
# trusting the list, while the cost of a redundant one is only a wasted slot —
# and the arithmetic says that instinct produces a diversity term that never
# changes an outcome.
#
# Work the case this exists for. The top of the ranking is near-duplicates of one
# file, normalised relevance around 0.96; a genuinely different perspective sits
# further down at 0.50; the duplicate's similarity to what is already chosen is
# 0.90:
#
#     lambda 0.7 -> duplicate 0.402  vs  distinct 0.350   duplicate still wins
#     lambda 0.6 -> duplicate 0.216  vs  distinct 0.300   distinct wins
#     lambda 0.5 -> duplicate 0.030  vs  distinct 0.250   distinct wins
#
# At 0.7 the penalty is capped at 0.3 while the relevance gap it has to overcome
# is 0.7 times whatever min-max normalisation produced — so MMR reorders only
# items that were already near-tied, which is not the failure it was added to
# fix. "Trade a little relevance for coverage" is an even trade, and 0.5 is
# where the trade is actually made.
#
# There is no measurement behind the exact value on this corpus — this is
# reasoning about the formula, not an experiment — which is why it is a
# parameter on `ContextRequest` rather than a constant callers cannot reach.
DEFAULT_LAMBDA = 0.5


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, computed rather than assumed.

    The embedder normalizes, so a dot product would do — and relying on that
    means a future model that does not normalize degrades silently into
    nonsense similarities, which is the exact shape of M1.6.1's defect. Dividing
    by the norms costs nothing at these sizes and cannot be wrong.

    Zero-length vectors are maximally *dissimilar* (0.0) rather than identical.
    A missing embedding must not read as "the same as everything", which would
    let one unembedded item suppress the whole rest of the list.
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One thing that could go in the context, with everything selection needs.

    `vector` is optional and its absence is handled rather than defaulted. A
    decision has no embedding — it is not in the corpus — and treating a missing
    vector as a zero vector would make every decision maximally novel and let
    them win every MMR round.
    """

    key: str
    category: ContextCategory
    tokens: int
    # The fused RRF score. Only its *ordering* is used; RRF scores are not on a
    # scale that means anything on its own, which is the whole reason M2.2 chose
    # rank fusion over score blending.
    relevance: float
    vector: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class Chosen:
    """One selected item, with the arithmetic that selected it.

    Kept so `--explain` can answer "why is this here" with numbers rather than
    with a shrug. `redundancy` is the highest similarity this item had to
    anything already chosen *at the moment it was picked* — not to the final
    set, which would be a different and less honest number, because it would
    include items chosen after this one was already committed to.
    """

    key: str
    position: int
    relevance: float
    redundancy: float
    score: float
    tokens: int


class Rejection(StrEnum):
    """Why an item that was a candidate is not in the result."""

    OVER_BUDGET = auto()
    CATEGORY_FULL = auto()
    MAX_ITEMS = auto()
    EMPTY = auto()


@dataclass(frozen=True, slots=True)
class Selection:
    chosen: list[Chosen] = field(default_factory=list)
    # Every candidate that did not make it, with the reason. Reported rather
    # than discarded: on a small budget the interesting output is usually what
    # *almost* fit, and "eight items" alone does not distinguish a rich corpus
    # from an empty one.
    rejected: list[tuple[str, Rejection]] = field(default_factory=list)
    tokens_used: int = 0

    @property
    def keys(self) -> list[str]:
        return [item.key for item in self.chosen]


def maximal_marginal_relevance(
    candidates: Sequence[Candidate],
    *,
    lambda_: float = DEFAULT_LAMBDA,
    limit: int | None = None,
) -> list[Chosen]:
    """Rank by relevance minus similarity to what is already ranked.

        score(d) = lambda * relevance(d) - (1 - lambda) * max similarity(d, s)
                                                          over s already chosen

    Relevance here is the *rank-derived* RRF score, which is bounded and small
    (a handful of hundredths), while similarity is a cosine in [-1, 1]. Those
    are not on the same scale, so relevance is normalised against the candidate
    set's own range before being combined — otherwise the similarity term would
    dominate completely and this would be a pure novelty ranking with a
    relevance-shaped ornament on it.

    **That normalisation makes the trade adaptive, which is a consequence worth
    naming rather than a side effect.** Min-max means what counts is an item's
    position within *this* candidate set, not the absolute size of its score. So
    when relevance genuinely separates the candidates the relevance term spans
    the full range and dominates; when every candidate scores alike — which is
    the common case on a small corpus — the range collapses, relevance stops
    discriminating, and novelty decides. That is the right way round, and it is
    also why `lambda` has to be low enough to matter at all: see the constant.

    The first pick is the most relevant candidate: there is nothing chosen yet,
    so the penalty is zero for everybody and the ordering is unchanged.
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda must be in [0, 1], got {lambda_}")

    remaining = list(candidates)
    normalise = _relevance_normaliser(remaining)
    chosen: list[Chosen] = []
    vectors: list[tuple[float, ...]] = []

    while remaining and (limit is None or len(chosen) < limit):
        best: tuple[float, float, Candidate] | None = None
        for candidate in remaining:
            redundancy = _redundancy(candidate, vectors)
            score = lambda_ * normalise(candidate.relevance) - (1 - lambda_) * redundancy
            if best is None or score > best[0]:
                best = (score, redundancy, candidate)

        assert best is not None
        score, redundancy, winner = best
        chosen.append(
            Chosen(
                key=winner.key,
                position=len(chosen) + 1,
                relevance=winner.relevance,
                redundancy=redundancy,
                score=score,
                tokens=winner.tokens,
            )
        )
        if winner.vector is not None:
            vectors.append(winner.vector)
        remaining.remove(winner)

    return chosen


def select(
    candidates: Sequence[Candidate],
    *,
    token_budget: int,
    max_items: int,
    lambda_: float = DEFAULT_LAMBDA,
    category_share: float = DEFAULT_CATEGORY_SHARE,
) -> Selection:
    """The whole choice: diversity, category caps and the budget, in one pass.

    One pass rather than three stages, because the stages interact. An item
    rejected for a category cap frees budget that the next item should be able
    to use, and an item that does not fit the budget should not consume a
    category slot. Running them separately gets both of those wrong in the same
    direction — a context smaller than the budget allows.

    Rejections are recorded with their reason rather than dropped, because on a
    tight budget what nearly fit is the useful diagnostic.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if max_items <= 0:
        raise ValueError(f"max_items must be positive, got {max_items}")

    # At least one of any category, however small the request. A cap that
    # rounded to zero would silently make a category unreachable, and the
    # category most likely to round away — decisions, of which there are few —
    # is the one most worth having.
    cap = max(1, int(max_items * category_share))

    ordered = maximal_marginal_relevance(candidates, lambda_=lambda_)
    by_key = {candidate.key: candidate for candidate in candidates}

    chosen: list[Chosen] = []
    rejected: list[tuple[str, Rejection]] = []
    used = 0
    per_category: dict[ContextCategory, int] = {}

    for item in ordered:
        candidate = by_key[item.key]
        if item.tokens <= 0:
            rejected.append((item.key, Rejection.EMPTY))
            continue
        if len(chosen) >= max_items:
            rejected.append((item.key, Rejection.MAX_ITEMS))
            continue
        if per_category.get(candidate.category, 0) >= cap:
            rejected.append((item.key, Rejection.CATEGORY_FULL))
            continue
        if used + item.tokens > token_budget:
            # Dropped whole, and the loop continues: a shorter item further down
            # may still fit, and stopping here would leave budget unspent for no
            # reason. See the module docstring on why nothing is truncated.
            rejected.append((item.key, Rejection.OVER_BUDGET))
            continue

        used += item.tokens
        per_category[candidate.category] = per_category.get(candidate.category, 0) + 1
        # Renumbered against what was actually kept. The position MMR gave it
        # counts items that were later rejected, and a context whose items are
        # numbered 1, 2, 5, 9 invites the reader to wonder what 3 and 4 were.
        chosen.append(
            Chosen(
                key=item.key,
                position=len(chosen) + 1,
                relevance=item.relevance,
                redundancy=item.redundancy,
                score=item.score,
                tokens=item.tokens,
            )
        )

    return Selection(chosen=chosen, rejected=rejected, tokens_used=used)


def _relevance_normaliser(
    candidates: Sequence[Candidate],
) -> Callable[[float], float]:
    """Map this set's relevance range onto [0, 1].

    Min-max over the candidate set, which `domain/fusion.py` warns against for
    *combining retrievers* and which is right here for a different reason: this
    is not being compared against anything outside the set. Every candidate in
    one selection was fused by the same RRF call, so the range is internally
    consistent, and the only thing the normalisation has to do is put relevance
    on the same footing as a cosine so `lambda` means what it says.

    A set where every candidate has the same score normalises to 1.0 for all of
    them, which leaves MMR ranking purely on novelty — the correct behaviour
    when relevance cannot distinguish anything.
    """
    scores = [candidate.relevance for candidate in candidates]
    if not scores:
        return lambda _: 0.0
    low, high = min(scores), max(scores)
    if high <= low:
        return lambda _: 1.0
    span = high - low
    return lambda value: (value - low) / span


def _redundancy(candidate: Candidate, chosen: Sequence[Sequence[float]]) -> float:
    """How much this repeats the most similar thing already chosen.

    Zero when either side has no vector. A decision has no embedding, so it is
    never penalised for similarity and never suppresses anything — which is the
    right default given how few of them there are, and is stated here because
    the alternative (treating a missing vector as the zero vector) would make
    every unembedded item maximally novel and let them take every slot.
    """
    if candidate.vector is None or not chosen:
        return 0.0
    return max(cosine(candidate.vector, other) for other in chosen)
