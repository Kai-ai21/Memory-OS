"""What a facet may claim, and how sure it is allowed to be.

Pure arithmetic and one bar. The interesting decisions here are all about
refusing to say things.

### The bar is the same three M5.3 set, and for a harder reason

A pattern needs three distinct decisions before it may be written down. A facet
is a claim about the *person* rather than about a run of decisions — goals,
strengths, weaknesses — which is the register a horoscope is written in, and
"you prefer simple solutions" is true of every engineer who has ever lived.
Nothing in such a sentence reveals whether three observations or zero produced
it, so the bar has to live in the code that emits it rather than in the reader's
judgement.

**Below the bar a facet is not written at a low confidence.** It is not written
at all, and the dimension records that it has nothing — see `Assessment`. A
0.2-confidence facet and no facet are very different objects on a page: the
first is a sentence somebody will read and remember, and the confidence beside
it will not survive the reading.

### Confidence reuses M5.3's shape rather than inventing one

`facet_confidence` is `pattern_confidence` with the same two factors — how
consistent the evidence is, and how much of it there is — because a second
formula for the same idea is a second thing to calibrate and the two would
disagree in the third decimal for no reason anybody could explain. It is
imported rather than copied.
"""

from dataclasses import dataclass

from memoryos.domain.patterns import DEFAULT_MIN_SUPPORT, pattern_confidence
from memoryos.domain.values import Dimension

# Distinct observations a facet needs. Three, the same as a pattern's, and
# counted the same way: in distinct *sources of evidence* rather than in rows,
# because four assumptions from two decisions is two observations.
MIN_SUPPORT = DEFAULT_MIN_SUPPORT

# Dimensions this milestone will derive at all.
#
# `GOALS` is absent because a goal is stated, never inferred: the difference
# between "you are trying to ship this by June" and "you have edited this file a
# lot" is the whole distinction between a model and a guess, and no amount of
# activity data crosses it.
#
# `LEARNING_STYLE` is absent because the evidence for it does not exist in any
# system of this shape. It needs the outcomes of learning attempts — a course
# against a project, and what was still true a year later — and a heuristic over
# file types would produce a sentence that sounds like insight and is astrology.
DERIVABLE: frozenset[Dimension] = frozenset(
    {
        Dimension.DECISION_PATTERNS,
        Dimension.STRENGTHS,
        Dimension.WEAKNESSES,
        Dimension.HABITS,
        Dimension.WORKFLOWS,
    }
)

# Why each underivable dimension is underivable, in the words `model show` uses.
UNDERIVABLE: dict[Dimension, str] = {
    Dimension.GOALS: (
        "goals are stated, never inferred — use `model assert --dimension goals`"
    ),
    Dimension.LEARNING_STYLE: (
        "no deriver exists: this needs the outcomes of learning attempts "
        "(a course against a project, and what held a year later), which nothing "
        "in this system records"
    ),
}


def facet_confidence(supporting: int, contradicting: int) -> float:
    """How sure a facet is allowed to be, from its evidence alone.

    M5.3's formula, imported rather than restated. Two factors: how one-sided
    the evidence is, and how much of it there is — so three-for-none and
    thirty-for-none are not the same number, and nine-for-eight is not a strong
    claim however large nine is.
    """
    return pattern_confidence(supporting, contradicting, min_support=MIN_SUPPORT)


def clears_bar(supporting: int, contradicting: int) -> bool:
    """Whether this may be written down at all.

    Two conditions, and the second is what stops a facet being asserted out of
    evidence that mostly argues against it: enough distinct observations, and
    more for than against. A tie is not a finding.
    """
    return supporting >= MIN_SUPPORT and supporting > contradicting


@dataclass(frozen=True, slots=True)
class Assessment:
    """What a dimension has, or the reason it has nothing.

    **The reason is the payload.** A dimension rendered as absent looks like an
    oversight in the page; a dimension rendered as "nothing clears three distinct
    decisions — the most any candidate reached was two" tells a reader what would
    have to change, which is the only useful thing an empty section can do.
    """

    dimension: Dimension
    facets: int
    # Non-empty exactly when `facets` is zero.
    gap: str = ""
    # What was considered and fell short, for a dimension that has candidates
    # but none above the bar. Empty when nothing was even proposed.
    best_support: int = 0

    @property
    def has_evidence(self) -> bool:
        return self.facets > 0

    def render(self) -> str:
        if self.has_evidence:
            return f"{self.facets} facet(s)"
        detail = f" (best candidate reached {self.best_support})" if self.best_support else ""
        return f"insufficient evidence: {self.gap}{detail}"


# --------------------------------------------------------------------------
# M8.2: how much the model moves
# --------------------------------------------------------------------------

# Closed facets a dimension needs before a mean lifetime is a mean rather than
# one facet's lifetime with a bar chart around it. Three, the same number and
# the same argument as MIN_SUPPORT: two observations describe themselves.
MIN_CLOSED_FOR_VERDICT = 3

# How long a dimension has to have been watched before "nothing changed" is
# distinguishable from "nothing has had time to change". A month, chosen because
# it is the interval M8.1's staleness threshold already uses for the same kind of
# judgement, and because a fortnight's corpus — which is what this one is —
# should fall below it rather than squeak past.
MIN_OBSERVATION_DAYS = 30.0

# Below this mean lifetime a dimension is rewriting itself faster than the
# evidence under it can plausibly change. A week: a facet needs three distinct
# observations to exist at all, and three observations that arrive and are
# overturned inside seven days are a detector responding to arrival order rather
# than to a regularity.
NOISE_LIFETIME_DAYS = 7.0


@dataclass(frozen=True, slots=True)
class Stability:
    """How often one dimension's facets change, and how long they last.

    **The verdict is allowed to be "cannot say", and usually is.** A dimension
    with one closed facet has a mean lifetime equal to that facet's lifetime, and
    printing it as a mean invites a reader to treat one event as a rate. The two
    thresholds above are what stop that: a verdict needs three closed facets and
    a month of watching, and below either the number is still printed and the
    judgement is withheld.
    """

    dimension: Dimension
    # Every facet ever written in this dimension, live and not.
    total: int
    live: int
    # Facets that stopped being live: superseded, withdrawn or dismissed.
    closed: int
    # Events, not rows: supersessions plus dismissals. A facet superseded twice
    # is two changes and one is not the other.
    changes: int
    # Mean days from `created_at` to whenever it stopped, over closed facets.
    # None when nothing has closed — an average of nothing is not zero.
    mean_lifetime_days: float | None
    # Mean age of the facets that are still live. Censored: a live facet has not
    # finished its lifetime, so this is a lower bound and is reported apart from
    # the mean above rather than pooled into it.
    mean_live_age_days: float | None
    # Oldest facet in this dimension to now. Zero when the dimension is empty.
    observed_days: float

    @property
    def changes_per_facet(self) -> float | None:
        """Churn per facet. None when there are no facets to divide by."""
        return self.changes / self.total if self.total else None

    @property
    def has_verdict(self) -> bool:
        return (
            self.closed >= MIN_CLOSED_FOR_VERDICT
            and self.observed_days >= MIN_OBSERVATION_DAYS
        )

    def verdict(self) -> str:
        """Which of the two failure modes this dimension resembles, or neither.

        The question M8.2 asks is whether the model is fitting noise or has
        stopped learning. Both are real failures and they are opposite, so the
        space between them is where a working model lives — but the space is only
        visible with enough closed facets to average over, and this corpus has
        none. Saying so is the answer, not a placeholder for one.
        """
        if not self.total:
            return "no facets: nothing to measure"
        if self.observed_days < MIN_OBSERVATION_DAYS:
            return (
                f"cannot say: {self.observed_days:.0f} days of history, "
                f"under the {MIN_OBSERVATION_DAYS:.0f} a rate needs"
            )
        if self.closed < MIN_CLOSED_FOR_VERDICT:
            return (
                f"cannot say: {self.closed} facet(s) have ever changed, "
                f"under the {MIN_CLOSED_FOR_VERDICT} a mean needs"
            )
        assert self.mean_lifetime_days is not None
        if self.mean_lifetime_days < NOISE_LIFETIME_DAYS:
            return (
                f"fitting noise: facets last {self.mean_lifetime_days:.1f} days "
                f"on average, under the {NOISE_LIFETIME_DAYS:.0f} a regularity needs"
            )
        return f"stable: mean lifetime {self.mean_lifetime_days:.0f} days"


def stopped_learning(entries: "tuple[Stability, ...]") -> str | None:
    """Whether the model as a whole has gone quiet, or None if that is unanswerable.

    Separate from `Stability.verdict` because it is a claim about the *model*
    rather than about a dimension, and the evidence differs: a dimension with no
    facets says nothing about whether the system has stopped learning, while a
    model with facets, a month of history and no changes at all says exactly
    that.
    """
    total = sum(item.total for item in entries)
    if not total:
        return None
    observed = max(item.observed_days for item in entries)
    if observed < MIN_OBSERVATION_DAYS:
        return None
    if sum(item.changes for item in entries):
        return None
    return (
        f"stopped learning: {total} facet(s) over {observed:.0f} days "
        "and not one of them has ever changed"
    )
