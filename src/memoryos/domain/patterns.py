"""The arithmetic behind a pattern's confidence, and the floor it has to clear.

Pure functions, no I/O, deliberately in `domain/`. These are the numbers every
behavioural claim in Phase 5 rests on, and they should be checkable against a
hand-computed fixture without a database, a corpus or a model — the same reason
`domain/fusion.py` holds RRF rather than `application/search.py`.

**The single most important idea here is the interval.** "You are overconfident
about deployment" is either a finding or a horoscope, and what separates them is
whether the observed rate is far enough from the stated one that a sample this
size could not have produced the difference by chance. M2.3a established the
principle for retrieval — it measured a 0.0122 resolution floor and M2.3b
refused to ship a 0.0109 gain because it fell below it. This is the same rule
applied to behaviour, and it is stricter than it looks: fourteen assumptions that
all held cannot distinguish "I am right 85% of the time" from "I am right always".
"""

import math
from dataclasses import dataclass

# 95%, two-sided. Not a knob anybody should turn to produce output: lowering it
# widens nothing and narrows the interval, which is exactly how a system starts
# reporting noise as insight.
_Z = 1.959963984540054


@dataclass(frozen=True, slots=True)
class Interval:
    """A proportion with the range a sample of this size can actually support."""

    observed: float
    low: float
    high: float
    n: int

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def describe(self) -> str:
        return f"{self.observed:.0%} (95% CI {self.low:.0%}-{self.high:.0%}, n={self.n})"


def wilson_interval(successes: int, n: int) -> Interval:
    """The Wilson score interval for a proportion.

    Wilson rather than the textbook normal approximation, and the reason is
    exactly the case this corpus produces: at p̂ = 0 or p̂ = 1 the normal
    interval has zero width, so fourteen assumptions that all held would report
    "100%, ± nothing" and every calibration comparison would find a gap. Wilson
    stays finite at the boundaries — fourteen out of fourteen gives roughly
    78%-100%, which is the honest statement that a run of fourteen cannot tell a
    perfect record from a good one.

    Worked by hand for 14/14 at 95%:

        p̂ = 1.0, z = 1.96, z² = 3.8415
        centre = (1.0 + 3.8415/28) / (1 + 3.8415/14) = 1.1372 / 1.2744 = 0.8923
        margin = (1.96/1.2744) · √(1.0·0.0/14 + 3.8415/784) = 1.5379 · 0.0700 = 0.1077
        → 0.785 to 1.000  (clamped at 1)
    """
    if n <= 0:
        return Interval(observed=0.0, low=0.0, high=1.0, n=0)
    observed = successes / n
    z_squared = _Z * _Z
    denominator = 1.0 + z_squared / n
    centre = (observed + z_squared / (2 * n)) / denominator
    margin = (_Z / denominator) * math.sqrt(
        observed * (1 - observed) / n + z_squared / (4 * n * n)
    )
    return Interval(
        observed=observed,
        low=max(0.0, centre - margin),
        high=min(1.0, centre + margin),
        n=n,
    )


def is_miscalibrated(stated: float, successes: int, n: int) -> bool:
    """Whether a stated confidence is outside what the outcomes support.

    The whole calibration test, and it is one line because the interval is doing
    the work. A stated 0.85 against fourteen successes out of fourteen is *not*
    miscalibration: 0.85 sits inside 0.785-1.0, so the data is consistent with
    the person being exactly as reliable as they said.

    This is the check that keeps `patterns discover` quiet on a small corpus,
    and it is meant to. A detector that reported every gap would report a gap
    every time, because a gap is what small samples produce.
    """
    return not wilson_interval(successes, n).contains(stated)


# The bar a pattern's supporting evidence has to clear, counted in **distinct
# decisions** rather than in rows. Four assumptions from two decisions is two
# observations of one person on two occasions, not four — and a threshold
# counting rows would let a single decision with five assumptions manufacture a
# pattern about a career.
DEFAULT_MIN_SUPPORT = 3

# The ceiling on any pattern's confidence.
#
# A rules-based detector over one person's own records, which that person also
# wrote, cannot reach certainty however much evidence agrees. Capping it is the
# difference between a number that means "as sure as this method gets" and one
# that means "true".
MAX_CONFIDENCE = 0.95


def pattern_confidence(
    supporting: int, contradicting: int, *, min_support: int = DEFAULT_MIN_SUPPORT
) -> float:
    """How much to believe a pattern, from its evidence alone.

    Two factors, multiplied, both in 0..1, and each answers a question the other
    cannot:

        agreement   = supporting / (supporting + contradicting)
        sufficiency = min(1, supporting / (2 * min_support))
        confidence  = min(MAX_CONFIDENCE, agreement * sufficiency)

    **Agreement** is what proportion of the relevant evidence points the same
    way. Four supporting against three contradicting is 0.57 — barely better
    than a coin, which is the correct reading of a "pattern" that half the
    corpus disagrees with.

    **Sufficiency** is how far past the minimum the support goes, reaching full
    credit at twice the bar. A candidate sitting exactly on the threshold with
    no counter-evidence scores 0.5 rather than 1.0, because three agreeing
    observations is the least that counts as anything at all and should not read
    as certainty.

    Worked examples:

        3 supporting,  0 contradicting → 1.00 * 0.50 = 0.50
        6 supporting,  0 contradicting → 1.00 * 1.00 → capped at 0.95
        6 supporting,  2 contradicting → 0.75 * 1.00 → 0.75
        4 supporting,  3 contradicting → 0.57 * 0.67 = 0.38

    Returns 0.0 when there is no supporting evidence at all, which callers treat
    as "do not emit" rather than as a weak pattern.
    """
    if supporting <= 0:
        return 0.0
    agreement = supporting / (supporting + contradicting)
    sufficiency = min(1.0, supporting / (2 * min_support))
    return min(MAX_CONFIDENCE, agreement * sufficiency)


def is_emittable(
    supporting: int, contradicting: int, *, min_support: int = DEFAULT_MIN_SUPPORT
) -> bool:
    """Whether a candidate may become a pattern at all.

    Two rules, and the second is the one that stops this becoming confirmation
    bias in SQL:

    1. **At least `min_support` distinct decisions agree.** Three is the floor
       and it is not lowered to produce output — a system that produces
       confident behavioural claims from thin evidence is worse than one that
       stays silent, because it sounds exactly like the product working.
    2. **More agree than disagree.** A candidate with four supporting and three
       contradicting is not a weak pattern, it is not a pattern: the corpus
       contains almost as many counter-examples as examples, and reporting it
       with a low confidence would still put the claim in front of somebody.
    """
    return supporting >= min_support and supporting > contradicting
