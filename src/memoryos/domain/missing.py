"""What may be called an absence, and what may not.

**Every other capability in this system finds what exists.** Retrieval finds
passages, the graph finds neighbours, the timeline finds buckets, decisions find
what somebody wrote down. This one names what is *not* there, and nothing in the
corpus states what is not there.

That makes it the easiest thing in the project to fake and the hardest to do
honestly, for one reason: **absence has infinite candidates.** You are always
missing something. A system asked "what am I missing?" can produce output forever
without ever being wrong in a way anybody can check, which is the precise shape
of a horoscope — and eight phases of this project have been arranged against
exactly that.

So the entire difficulty is deciding which absences are worth naming, and the
answer here is a rule rather than a judgement: **an absence may be named only
when the thing that is missing exists somewhere else in your own history.** Not
in a model's training, not in a best-practices list. In your corpus.

### What that rule excludes

"You should consider rate limiting" is advice. It may be good advice. It is not
grounded in anything about the person it is offered to, it would be produced with
equal confidence for an empty corpus, and it cannot cite. `GapKind` has no member
for it and no deriver produces one.

### The four that survive

Each is computable from data another milestone already built, and each cites the
history that makes it sayable:

* **`UNSTATED_ASSUMPTION`** — decisions like this one recorded an assumption this
  one does not. Grounded because the assumption is in the corpus, attached to
  named decisions.
* **`REPEATED_PATTERN`** — this resembles a pattern whose outcomes went badly.
  Grounded in M5.3, which already refuses to emit a pattern below three distinct
  decisions.
* **`ORPHANED_WORK`** — an entity was active and then was not, and no decision
  records why. Grounded in M4.0's gap detection, which is the one capability in
  this system that already measures an absence.
* **`UNEVALUATED_ASSUMPTION`** — a belief old enough to be checkable that nobody
  has checked. Grounded in M5.2 plus a clock, and the only one of the four whose
  evidence is the *absence of a row* rather than the presence of one.

### Two supporting instances, and silence below that

The bar is two rather than three, and the difference from M5.3 is deliberate. A
pattern asserts a regularity in somebody's behaviour and needs three occasions
before that word applies. A gap asserts something narrower — "these two decisions
wrote down a thing this one did not" — which two instances genuinely establish,
and which names the two so a reader can disagree with a specific claim rather
than with a statistic.

**Below the bar the correct output is nothing.** Not a low-confidence gap: an
absence stated tentatively is still an absence stated, and the reader remembers
the sentence rather than the hedge. On a corpus of sixteen decisions "I do not
have enough history to say what you are missing here" is the common and correct
answer, and a threshold lowered until output appears is a threshold that has
stopped measuring anything.
"""

from dataclasses import dataclass
from enum import StrEnum

from memoryos.domain.patterns import pattern_confidence

# Historical instances a gap needs before it may be named.
#
# Two, against M5.3's three, and the gap between the numbers is the difference
# between the claims. "You repeatedly underestimate X" needs three occasions
# before *repeatedly* is honest; "these two decisions recorded an assumption this
# one did not" is fully established by the two, and cites both.
MIN_SUPPORT = 2

# Confidence below which nothing is said at all.
#
# Set to what two clean supporting instances and no contradiction produce, so the
# threshold and the minimum support agree by construction rather than by two
# numbers somebody has to keep in step. A literal here would drift the first time
# `MIN_SUPPORT` moved.
MIN_CONFIDENCE = pattern_confidence(MIN_SUPPORT, 0, min_support=MIN_SUPPORT)


class GapKind(StrEnum):
    """The four absences this system can ground.

    Closed, and the closure is the design. An open vocabulary here is how "you
    should consider X" arrives: every new detector invents a label, one of them
    is eventually a language model's opinion, and by then the type says nothing
    about whether the output can be checked.
    """

    UNSTATED_ASSUMPTION = "unstated_assumption"
    REPEATED_PATTERN = "repeated_pattern"
    ORPHANED_WORK = "orphaned_work"
    UNEVALUATED_ASSUMPTION = "unevaluated_assumption"


def gap_confidence(supporting: int, contradicting: int) -> float:
    """How much to believe an absence, from the history that supports it.

    M5.3's formula, with this milestone's minimum. Reused rather than reinvented
    for the reason `domain/user_model.py` reuses it: a second formula for the
    same idea is a second thing to calibrate, and the two would disagree in the
    third decimal for no reason anybody could explain.
    """
    return pattern_confidence(supporting, contradicting, min_support=MIN_SUPPORT)


def worth_saying(supporting: int, contradicting: int) -> bool:
    """Whether this absence may be named at all.

    Three conditions, and the middle one is the one that does the work on a small
    corpus. Two supporting instances is the floor; **contradicting evidence must
    not outweigh the support**, because an assumption that two decisions recorded
    and three deliberately did not is a choice rather than an oversight; and the
    confidence has to clear the bar, which by construction it does exactly when
    the first two hold.
    """
    return (
        supporting >= MIN_SUPPORT
        and supporting > contradicting
        and gap_confidence(supporting, contradicting) >= MIN_CONFIDENCE
    )


@dataclass(frozen=True, slots=True)
class Silence:
    """Why nothing was said, which is the usual output and the useful one.

    **A command that prints nothing is indistinguishable from a broken one.**
    `patterns discover` learned this and says what it considered; the same
    argument holds harder here, because the honest answer on a corpus this size
    is silence almost every time and a reader has to be able to tell "no gaps
    found" from "no gaps looked for".
    """

    considered: int = 0
    below_support: int = 0
    outweighed: int = 0
    # The most any single candidate reached, so the gap between the corpus and
    # the bar is a number rather than an adjective.
    best_support: int = 0

    def render(self) -> str:
        if not self.considered:
            return (
                "I have no history that bears on this, so I cannot say what is "
                "missing from it."
            )
        detail = (
            f"; the closest reached {self.best_support} of {MIN_SUPPORT}"
            if self.best_support
            else ""
        )
        outweighed = (
            f", and {self.outweighed} had evidence arguing against"
            if self.outweighed
            else ""
        )
        return (
            f"I do not have enough history to say what you are missing here: "
            f"{self.considered} candidate(s) considered, none of them supported by "
            f"{MIN_SUPPORT} past instances{detail}{outweighed}."
        )
