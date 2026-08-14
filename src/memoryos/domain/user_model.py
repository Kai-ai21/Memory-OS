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
