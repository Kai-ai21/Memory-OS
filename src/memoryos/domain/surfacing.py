"""When to volunteer context, and — mostly — when not to.

**The bar for volunteering something is much higher than the bar for answering a
question, and this module is where that difference is written down as
arithmetic.** Everything through M6.2 was still a pull: a panel asked, and a
mediocre answer cost the reader a glance. From here the system speaks first, and
a mediocre answer costs something that does not come back. A tool that
interrupts with mediocre suggestions gets muted, and then its good suggestions
are muted too.

So the asymmetry is the design. A false positive costs trust; a false negative
costs nothing, because the pull path still exists and is one keystroke away.
Every branch below that cannot decide resolves to silence.

### The threshold is a structural property, not a tuned number

Relevance here is M2.2's fused RRF score, which is not on a scale that means
anything on its own — `domain/fusion` says so, and `domain/context` uses only its
*ordering* for that reason. One number in it does mean something, though: the
most a single ranking can contribute is `1 / (k + 1)`, when it puts an item
first. Everything above that line requires a second route to have found the same
item.

    SINGLE_ROUTE_BEST = 1 / (60 + 1) = 0.01639

So the threshold is expressed in multiples of that, and the base of 1.8 is chosen
for a property rather than for a score: **no item found by only one of the four
sources can ever be surfaced, however highly that one source ranked it.** Search
is happy to return such an item — it is a legitimate answer to a question
somebody asked. Volunteering it is a guess, and this is the milestone that has to
stop guessing.

For calibration, 1.8 is roughly "second by one route and eighth by another"
(1/62 + 1/68 = 0.0308 = 1.88x), or "first and twelfth" (1/61 + 1/72 = 0.0303).
Two independent routes agreeing near the top. Anything less stays quiet.

### Similar, not identical

"Nothing similar was surfaced recently" cannot be answered by comparing hashes:
one item different is a different hash and the same interruption. So contexts are
compared by the overlap of their item keys, and `SIMILARITY` is the line. The
hash is kept anyway, as the row's identity for the log — it answers "is this
exactly what you saw" while the overlap answers "is this the same thing again".

### Adaptation is per-focus, and asymmetric

A focus whose context is dismissed repeatedly raises its own threshold; one whose
context gets acted on lowers it. Per-focus rather than global, because one noisy
file must not be able to silence the whole system — which is the failure mode a
global counter has by construction.

The two steps are deliberately not equal. Three dismissals reach the ceiling and
that focus goes quiet; it takes nine acted-on items to walk the same distance
back. That is the same asymmetry as everything above: being wrong loudly costs
more than being wrong quietly, so the system moves fast towards silence and
slowly away from it.

The floor is `SINGLE_ROUTE_BEST` exactly, so no amount of positive feedback can
ever buy an item its way in on one route. That invariant holds at every point of
the adaptation, not just at the default.

Pure, with the clock passed in. Nothing here reads a database or a wall clock,
so every rule can be checked against a hand-worked example.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum, auto

from memoryos.domain.fusion import DEFAULT_RRF_K

# The most one ranking can contribute to a fused score: first place in it.
#
# Every threshold in this module is a multiple of this, which is what keeps the
# numbers meaningful when M2.2's `k` changes. A bare 0.03 in the source would
# silently become a different rule the day somebody tuned `k`.
SINGLE_ROUTE_BEST = 1.0 / (DEFAULT_RRF_K + 1)

# The default bar, as a multiple of the line above. See the module docstring:
# 1.8 is not a tuned value, it is the smallest multiple that no single-source
# item can reach.
BASE_MULTIPLE = 1.8

# Adaptation limits, in the same units.
#
# The floor is exactly one route's best, so the "two routes or silence" property
# survives any amount of positive feedback. The ceiling is four, which on this
# corpus is unreachable — the graph source contributes nothing until entities are
# extracted, so three routes is the practical maximum and a focus at the ceiling
# is a focus that has been switched off. That is the intended end state for
# somewhere a person has dismissed context three times.
FLOOR_MULTIPLE = 1.0
CEILING_MULTIPLE = 4.0

# What one piece of feedback moves the threshold by, in multiples.
DISMISSAL_STEP = 0.6
USEFUL_STEP = 0.2

# How much two contexts must overlap to count as the same interruption.
#
# Jaccard over item keys. Two-thirds of one context's items appearing in the
# other is the same list with an edit, not a new finding, and re-showing it is
# the thing this milestone exists to prevent.
SIMILARITY = 0.6

# How long the same context stays suppressed for the same focus.
#
# Long enough to cover the working session the context was built for. Shorter
# than a day on purpose: the same file tomorrow, with the corpus moved on, is a
# genuinely new question — which is the same argument M6.0's partial index makes
# about re-focusing a file the next morning.
REPEAT_WINDOW = timedelta(hours=4)

# How long a *dismissed* context stays suppressed. Two orders of magnitude
# longer, and that ratio is the whole point rather than a tuning choice.
# Repeating something somebody explicitly refused is the fastest way to be
# ignored permanently, and there is no evidence at all that thirty days later
# they want it back — so this is set by what a mistake costs, not by a half-life.
DISMISSAL_WINDOW = timedelta(days=30)

# How many of a context's items decide its identity.
#
# The whole list would make the hash change every time MMR reordered a tail item
# nobody read. The top five are what a reader actually sees before deciding
# whether this is worth their attention.
IDENTITY_ITEMS = 5


class SurfaceReason(StrEnum):
    """Why context was surfaced, or why it was not.

    **Recorded for both outcomes, which is the point.** "Why didn't it show me
    anything?" is the question a proactive system has to be able to answer, and
    a system that only logs what it did cannot: silence looks identical whether
    the gate refused, the corpus was empty, or nothing ever ran.
    """

    # Surfaced.
    CLEARED = auto()

    # Not surfaced, in the order the gate checks them.
    NO_CONTEXT = auto()
    NOTHING_NEW = auto()
    BELOW_THRESHOLD = auto()
    DISMISSED = auto()
    ALREADY_SURFACED = auto()


# One line per reason, in the second person, for the CLI and the panel.
#
# Written here rather than at each call site because the same explanation has to
# read identically in three places, and three copies of a sentence is three
# chances for one of them to describe behaviour that has changed.
EXPLANATIONS: dict[SurfaceReason, str] = {
    SurfaceReason.CLEARED: (
        "two independent routes agreed on something you do not already have open"
    ),
    SurfaceReason.NO_CONTEXT: "nothing was assembled for this focus",
    SurfaceReason.NOTHING_NEW: (
        "everything found was the file you are already looking at"
    ),
    SurfaceReason.BELOW_THRESHOLD: (
        "the best item did not clear this focus's bar — one route found it, or two "
        "found it well down their rankings"
    ),
    SurfaceReason.DISMISSED: "you dismissed this context, and it stays quiet for a month",
    SurfaceReason.ALREADY_SURFACED: "you have already been shown this, recently",
}


@dataclass(frozen=True, slots=True)
class TopItem:
    """The best thing in a context that the reader does not already have open."""

    key: str
    title: str
    score: float
    # How many of the four sources proposed it. Not used by the gate — the score
    # already encodes it — but reported, because "found by two routes" is the
    # sentence a reader can check and "0.0308" is not.
    routes: int


@dataclass(frozen=True, slots=True)
class PriorSurfacing:
    """Something already shown for this focus, and what became of it."""

    context_hash: str
    keys: tuple[str, ...]
    surfaced_at: datetime
    dismissed_at: datetime | None = None
    acted_on_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SurfaceDecision:
    """Whether to speak, and why — with the arithmetic that decided it.

    `score` and `threshold` are both carried even when the decision was made on
    something else entirely, because the first question anybody asks about a
    refusal is how close it came.
    """

    surface: bool
    reason: SurfaceReason
    score: float
    threshold: float
    context_hash: str
    top: TopItem | None = None

    @property
    def explanation(self) -> str:
        return EXPLANATIONS[self.reason]

    @property
    def margin(self) -> float:
        """How far over the bar it was. Negative means it was refused by this."""
        return self.score - self.threshold


def threshold_for(*, dismissed: int, acted_on: int) -> float:
    """This focus's bar, after whatever feedback it has collected.

    Linear in both counts rather than a ratio, and that is deliberate: a ratio
    would treat one dismissal out of one as identical to fifty out of fifty, so a
    single irritated click would silence a focus outright and a long good history
    could be erased by two bad days. Counts move the bar by what actually
    happened, and the clamp stops either direction running away.
    """
    multiple = BASE_MULTIPLE + DISMISSAL_STEP * dismissed - USEFUL_STEP * acted_on
    clamped = min(CEILING_MULTIPLE, max(FLOOR_MULTIPLE, multiple))
    return clamped * SINGLE_ROUTE_BEST


def context_hash(keys: Sequence[str]) -> str:
    """The identity of one context, for the log.

    The top few keys, sorted, hashed. Sorted because two assemblies that chose
    the same items in a different order are the same interruption; truncated
    because the tail is where reordering lives. An empty context still gets a
    hash rather than a null, so the column can be `NOT NULL` and a refusal row
    is queryable like any other.
    """
    material = "\x00".join(sorted(keys[:IDENTITY_ITEMS]))
    return hashlib.sha256(material.encode()).hexdigest()[:32]


def overlap(left: Sequence[str], right: Sequence[str]) -> float:
    """Jaccard similarity of two contexts' item keys.

    Zero when either is empty, rather than the 1.0 that "no difference between
    two empty sets" would give. An empty context is not the same interruption as
    another empty one; it is not an interruption at all, and letting the two
    match would make the first empty assembly suppress every later one.
    """
    first, second = set(left), set(right)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def names_the_focus(external_key: str | None, focus: str) -> bool:
    """Whether this item *is* the thing the reader is already looking at.

    The focus from an editor is a repository-relative path and so is the corpus
    key, but they need not be rooted the same way — a watcher started inside a
    package sends `search.py` for what the corpus calls
    `src/memoryos/application/search.py`. So either may be the tail of the other,
    and the comparison is anchored on a separator so `search.py` does not match
    `research.py`.

    A focus that is not a path — a meeting title — matches nothing here, which is
    correct: none of the context is "already open" in that case.
    """
    if not external_key or not focus:
        return False
    if external_key == focus:
        return True
    return external_key.endswith(f"/{focus}") or focus.endswith(f"/{external_key}")


def decide(
    *,
    top: TopItem | None,
    keys: Sequence[str],
    threshold: float,
    recent: Sequence[PriorSurfacing],
    now: datetime,
    repeat_window: timedelta = REPEAT_WINDOW,
    dismissal_window: timedelta = DISMISSAL_WINDOW,
) -> SurfaceDecision:
    """Whether this context is worth interrupting for.

    **The order of the checks is the order the reasons are worth hearing in**,
    not the order they are cheapest in. The reason returned is the first thing
    that would have stopped it, and "there was nothing" is a more useful answer
    than "you dismissed something like it a week ago" when both are true.

    Suppression is checked after the threshold for the same reason: a context
    that would not have been shown anyway was not *suppressed*, and counting it
    as suppressed would inflate the one number in `surfacing stats` that says
    whether suppression is doing any work.
    """
    digest = context_hash(keys)

    if top is None:
        return SurfaceDecision(
            surface=False,
            reason=(
                SurfaceReason.NOTHING_NEW if keys else SurfaceReason.NO_CONTEXT
            ),
            score=0.0,
            threshold=threshold,
            context_hash=digest,
        )

    if top.score <= threshold:
        # `<=` rather than `<`: at exactly the floor, an item found by one route
        # in first place scores precisely `SINGLE_ROUTE_BEST`, and the property
        # this module is built on is that such an item is never surfaced.
        return SurfaceDecision(
            surface=False,
            reason=SurfaceReason.BELOW_THRESHOLD,
            score=top.score,
            threshold=threshold,
            context_hash=digest,
            top=top,
        )

    # Both windows are checked against *every* similar prior before either
    # verdict is returned, rather than returning on the first match. A context
    # shown twice and dismissed once is dismissed, whichever row the query
    # happened to return first, and a suppression that depended on row order
    # would be one that stopped holding when somebody changed an ORDER BY.
    similar = [prior for prior in recent if overlap(keys, prior.keys) >= SIMILARITY]
    refusal: SurfaceReason | None = None
    for prior in similar:
        if prior.dismissed_at is not None and now - prior.dismissed_at < dismissal_window:
            refusal = SurfaceReason.DISMISSED
            break
        if now - prior.surfaced_at < repeat_window:
            refusal = SurfaceReason.ALREADY_SURFACED

    if refusal is not None:
        return SurfaceDecision(
            surface=False,
            reason=refusal,
            score=top.score,
            threshold=threshold,
            context_hash=digest,
            top=top,
        )

    return SurfaceDecision(
        surface=True,
        reason=SurfaceReason.CLEARED,
        score=top.score,
        threshold=threshold,
        context_hash=digest,
        top=top,
    )
