"""Whether every claim in a multi-hop answer traces to something retrieved.

M2.6 checked an answer against one context block: the passages were numbered, the
model was asked to cite the numbers, and an index outside the range was an
unambiguous fabrication. **That check does not survive the move to a trajectory.**
The evidence is now spread across six tool results the model saw at different
times, half of them compacted out of its window by the time it wrote a sentence,
and the drift is correspondingly harder to see — the answer is fluent, the
citations resolve, and one clause in the middle came from the model rather than
the corpus.

So this checks the harder property M2.6 explicitly declined: not "did it cite
something" but **"does anything it retrieved actually bear on this sentence"**.

### Why that check is possible here and was not there

M2.6's stated reason for not attempting it was that judging support needs a
judge, the only available judge is another language model, and a model grading
its own grounding is not evidence. That reasoning still holds and this module
does not break it — **there is no model here**. Support is measured with the
embedder the corpus was indexed with, which is a fixed function that knows
nothing about the answer, cannot be persuaded, and produces the same number
twice. It is a weaker instrument than a judge and it is an honest one.

What that buys is a real limit, stated up front: this measures *semantic
proximity to retrieved text*, not entailment. A sentence that says the opposite
of a passage it closely paraphrases scores as supported. Catching that needs
entailment, entailment needs a judge, and the judge is the thing we do not have.
M7.3 is where a scorer with a different instrument belongs.

### Three levels, because "supported" is not one thing

* **Direct** — one retrieved passage says this. The claim is a paraphrase of
  something a tool returned.
* **Inferred** — no single result says it, but passages from two different hops
  each bear on it. This is what a multi-hop answer legitimately produces, and it
  is weaker: the *combination* is the model's, and nothing here checks that the
  combination is valid.
* **Unsupported** — nothing in the trajectory is close to it.

The direct/inferred split is the number worth watching over time. **Heavy
inference means the agent is reasoning beyond its evidence**, which is precisely
the failure that a support rate alone would report as success.

### Unsupported claims are flagged, never removed

The rule M2.6 set, for the reason M2.6 gave: prose with a sentence quietly
deleted from the middle reads as complete and is not. What *is* withheld is the
whole answer, when too little of it survives — see `verdict`.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.application.agent.planner import MultiHopPlanner, Step, Trajectory
from memoryos.application.citations import unresolved_locators
from memoryos.domain.grounding import split_sentences

logger = structlog.get_logger(__name__)

# Cosine similarity at which one retrieved passage is taken to *say* a claim.
#
# **Measured, not chosen** — `scripts/calibrate_verification.py` re-runs it, over
# real tool results and three sets of claims:
#
#   sentences real runs wrote from those results   0.631 - 0.835  (median 0.670)
#   fluent claims about things this corpus lacks   0.525 - 0.596  (median 0.559)
#
# The bands do not overlap, and 0.62 sits in the gap: 0.024 above the highest
# invention and 0.011 below the lowest real claim. That margin is small because
# bge-small-en-v1.5 compresses everything worth comparing into roughly 0.5-0.85,
# which is a property of the model rather than of the corpus — and it is the
# reason this number belongs beside the script that produced it rather than in a
# config file where somebody would round it.
DIRECT = 0.62

# Where a passage stops bearing on a claim at all. Two hops that each clear this
# without either clearing DIRECT is what "the claim combines two results" looks
# like from here.
#
# **The band between the two is thin, and the calibration says so**: with the
# thresholds that separate the measured bands cleanly, `INFERRED` fired for none
# of the twenty-two calibration claims. Widening it to make the level interesting
# would mean moving DIRECT up into the supported band and flagging true claims,
# which is a worse trade than a level that rarely fires. See the README.
INFERRED = 0.58

# Below this share of factual claims supported, the answer is not returned.
MIN_SUPPORT = 0.5

# Support rate at or above which nothing is qualified.
GROUNDED = 0.8

# Bounds on the work. A trajectory of six hops with five citations and a long
# rendering each is a few hundred short embeddings, which is tens of
# milliseconds warm — but a `get_memory` on a large file can produce a hundred
# blocks on its own, and an unbounded loop here would put a visible pause
# between the answer and the screen.
MAX_UNITS_PER_STEP = 48
UNIT_CHARS = 700
# Blocks shorter than this are headers, counts and list bullets. They embed to
# nothing useful and they crowd out the real passages under the cap.
MIN_UNIT_CHARS = 40

# `[2]` or `[1, 3]`, the same shape M2.6 uses — here the number is a HOP rather
# than a passage, which is the only thing about the notation that changed.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Sentences that are *about* the answer rather than claims drawn from the corpus.
#
# **Rules-based on purpose, and the reason is the one M2.6 gives for not using a
# judge at all**: asking the model whether its own sentence was factual is asking
# the thing under test to mark its own paper, and it fails in the direction that
# hides the defect — a model that invented a claim is not well placed to notice.
#
# Three shapes, and each is here because it appeared in a real run:
#
# * discourse and framing — "In summary", "Here are the decisions where …",
# * process narration — "Let me search for", "I should look this up",
# * statements of absence — "I could not find any decisions marked as reversed",
#   which is the single behaviour this whole phase most wants, and demanding a
#   citation for it would score the honest answer below the fluent one.
_CONNECTIVE_PREFIXES = (
    "in summary",
    "in conclusion",
    "to summarise",
    "to summarize",
    "overall",
    "in short",
    "here are",
    "here is",
    "here's",
    "these are",
    "this means",
    "looking at",
    "let me",
    "i will",
    "i'll",
    "i should",
    "i need to",
    "first,",
    "next,",
    "finally,",
    "based on the above",
    "based on what",
    "to answer",
    "note that",
)

# Absence, refusal and hedging about the corpus itself. Shape rather than
# literal phrases, for the reason `_REFUSAL_PATTERN` in `domain/grounding.py`
# gives: the model paraphrases freely and every miss silently reclassifies a
# correct refusal as an uncited claim.
_ABSENCE = re.compile(
    r"\b(?:do(?:es)?\s+not|did\s+not|do(?:es)?n't|didn't|cannot|can\s?not|can't"
    r"|could\s+not|couldn't|unable\s+to|no|none|nothing|not)\b"
    r"(?:\W+\w+){0,4}\W+"
    r"(?:contain|cover|mention|describe|discuss|address|specify|answer|include"
    r"|provide|state|say|reference|detail|record|find|found|available|evidence"
    r"|information|data|corpus|memories|decisions)",
    re.IGNORECASE,
)

_HAS_LETTERS = re.compile(r"[A-Za-z]")

# Markdown scaffolding a model puts around a list. Stripped before classifying,
# so "* **DECISION 019ff…**: …" is judged on its sentence and not on its bullet.
_ORNAMENT = re.compile(r"^[\s\-*#>•]+|\*+")


class Support(StrEnum):
    DIRECT = "direct"
    INFERRED = "inferred"
    UNSUPPORTED = "unsupported"


class Embedder(Protocol):
    """The two halves of embedding, and nothing else.

    Narrower than the `Embedder` port so this module can be tested against a
    stub that returns fixed vectors — which is what makes the threshold logic
    testable without a model, and the thresholds themselves measurable with one.
    """

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_passage(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class Claim:
    """One sentence of the answer, and what in the trajectory bears on it."""

    text: str
    sentence_index: int
    # The hop the sentence cited, when it cited one. None is not a failure —
    # support is measured against the whole trajectory, and a model that got the
    # citation right and the content wrong is exactly what this is for.
    cited_step: int | None
    supported: bool
    support_excerpt: str | None
    # The fields below are not in M7.2's shape; they are what makes a flagged
    # answer debuggable rather than merely marked.
    support: Support = Support.UNSUPPORTED
    similarity: float = 0.0
    # Hops with a passage above `INFERRED`. Two or more is what "the claim
    # combines two results" means, and it is the evidence for the level above.
    steps: tuple[int, ...] = ()
    factual: bool = True
    # True when the best-supporting result said it had been cut. The model may
    # be citing something it only partly saw.
    from_truncated: bool = False


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """The verdict, and everything it was computed from."""

    support_rate: float = 1.0
    direct_rate: float = 1.0
    unsupported: list[Claim] = field(default_factory=list)
    # Hop numbers the answer cited that the trajectory does not have. The
    # unambiguous failure, and the one that must always be empty.
    invalid_citations: list[int] = field(default_factory=list)
    verdict: str = "grounded"
    claims: list[Claim] = field(default_factory=list)
    # Cited hops whose result was truncated. Not a failure: a reason to trust the
    # sentence less, reported rather than folded into the rate.
    truncated_citations: list[int] = field(default_factory=list)
    # Citations whose offsets no longer resolve against the stored memory. Empty
    # is the only acceptable value; a non-empty one means the corpus moved under
    # the answer.
    unresolved_citations: list[str] = field(default_factory=list)

    @property
    def factual_claims(self) -> int:
        return sum(1 for claim in self.claims if claim.factual)

    @property
    def connective_claims(self) -> int:
        return sum(1 for claim in self.claims if not claim.factual)

    @property
    def returnable(self) -> bool:
        """**Whether the answer may be shown at all.**

        This is the milestone's acceptance criterion and it is deliberately not a
        flag: an answer that fails here is replaced by a refusal on every surface,
        because a guardrail that a caller can decline to apply is a preference.
        """
        return self.verdict != "ungrounded"

    def marked(self, answer: str, marker: str = " [unsupported]") -> str:
        """The answer with unsupported sentences marked, not removed.

        Rebuilt from the claims rather than by editing the string, so the marker
        lands at a sentence boundary the splitter agreed with — and a sentence
        that was never classified cannot silently lose its mark.
        """
        if not self.claims:
            return answer
        return " ".join(
            claim.text + (marker if claim.factual and not claim.supported else "")
            for claim in self.claims
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "support_rate": round(self.support_rate, 4),
            "direct_rate": round(self.direct_rate, 4),
            "verdict": self.verdict,
            "factual_claims": self.factual_claims,
            "connective_claims": self.connective_claims,
            "unsupported": [claim.text for claim in self.unsupported],
            "invalid_citations": list(self.invalid_citations),
            "truncated_citations": list(self.truncated_citations),
            "unresolved_citations": list(self.unresolved_citations),
        }


REFUSAL = (
    "I could not answer that from what I retrieved. The searches I ran did not "
    "return material that supports an answer to this question, and the answer I "
    "drafted rested mostly on claims nothing in the corpus backs — so it is not "
    "worth showing. What was retrieved is in the trajectory; run with --trace to "
    "see it."
)


def verify(
    trajectory: Trajectory,
    embedder: Embedder,
    *,
    direct: float = DIRECT,
    inferred: float = INFERRED,
    min_support: float = MIN_SUPPORT,
    unresolved: Sequence[str] = (),
) -> VerificationResult:
    """Check an answer against the trajectory that produced it.

    `unresolved` is the locators whose offsets no longer resolve, checked by the
    caller because it needs a database and this does not. Empty is the normal
    case and the only good one.
    """
    answer = (trajectory.answer or "").strip()
    if not answer:
        # Nothing was returned, so there is nothing to be ungrounded about. The
        # trajectory already says why in `stopped_because`, and inventing a
        # verdict here would overwrite it.
        return VerificationResult(unresolved_citations=list(unresolved))

    acted = [step for step in trajectory.steps if step.tool is not None and step.result]
    valid_hops = set(range(1, len(acted) + 1))

    claims = _classify(answer)
    factual = [claim for claim in claims if claim.factual]
    checked = _support(factual, acted, embedder, direct=direct, inferred=inferred)

    by_index = {claim.sentence_index: claim for claim in checked}
    claims = [by_index.get(claim.sentence_index, claim) for claim in claims]

    invalid = sorted(
        {
            hop
            for claim in claims
            if claim.cited_step is not None and claim.cited_step not in valid_hops
            for hop in (claim.cited_step,)
        }
    )
    truncated = sorted(
        {
            claim.cited_step
            for claim in claims
            if claim.cited_step in valid_hops
            and acted[claim.cited_step - 1].result is not None
            and acted[claim.cited_step - 1].result.truncated  # type: ignore[union-attr]
        }
    )

    total = len(factual)
    supported = [claim for claim in claims if claim.factual and claim.supported]
    direct_count = sum(1 for claim in supported if claim.support is Support.DIRECT)
    # **1.0 for an answer with no factual claims**, which is M2.6's convention
    # and is here for M2.6's reason: a refusal has nothing to support, and
    # scoring it zero would make the safest possible answer the worst-rated one.
    support_rate = 1.0 if total == 0 else len(supported) / total
    direct_rate = 1.0 if not supported else direct_count / len(supported)

    result = VerificationResult(
        support_rate=support_rate,
        direct_rate=direct_rate,
        unsupported=[claim for claim in claims if claim.factual and not claim.supported],
        invalid_citations=invalid,
        verdict=_verdict(support_rate, invalid, min_support=min_support),
        claims=claims,
        truncated_citations=truncated,
        unresolved_citations=list(unresolved),
    )
    logger.info(
        "agent.verified",
        verdict=result.verdict,
        support_rate=round(support_rate, 3),
        direct_rate=round(direct_rate, 3),
        factual=total,
        connective=result.connective_claims,
        invalid_citations=len(invalid),
    )
    return result


def presented(trajectory: Trajectory, result: VerificationResult) -> str:
    """What the caller shows: the marked answer, or the refusal.

    One function, so the refusal cannot be applied on one surface and forgotten
    on another. The CLI, the API and anything after them all read this.
    """
    if trajectory.answer is None:
        return REFUSAL
    if not result.returnable:
        return REFUSAL
    return result.marked(trajectory.answer)


@dataclass(frozen=True, slots=True)
class VerifiedAnswer:
    """A trajectory, its verdict, and the text a caller may actually show."""

    trajectory: Trajectory
    verification: VerificationResult
    # Already marked, or already replaced by the refusal. Callers print this
    # rather than `trajectory.answer`, which is kept beside it precisely so the
    # withheld text is still inspectable in a trace.
    answer: str

    @property
    def refused(self) -> bool:
        return not self.verification.returnable and self.trajectory.answer is not None


class VerifiedAgent:
    """The planner with the guardrail attached, and the only way in.

    **Verification is not a flag.** `--verify` decides whether the per-claim
    breakdown is *printed*; it does not decide whether the check runs, because a
    guardrail a caller can decline to apply is a preference. Every surface goes
    through here, so an ungrounded answer is withheld on all of them or on none.

    The cost is one embedding pass over the answer's sentences and the
    trajectory's passages — tens of milliseconds against a loop that just spent
    twenty seconds and several thousand tokens.
    """

    def __init__(
        self,
        planner: MultiHopPlanner,
        sessions: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        *,
        min_support: float = MIN_SUPPORT,
    ) -> None:
        self._planner = planner
        self._sessions = sessions
        self._embedder = embedder
        self._min_support = min_support

    async def ask(self, question: str, *, max_hops: int | None = None) -> VerifiedAnswer:
        trajectory = await self._planner.run(question, max_hops=max_hops)
        unresolved = await unresolved_locators(self._sessions, trajectory.citations)
        result = verify(
            trajectory,
            self._embedder,
            min_support=self._min_support,
            unresolved=unresolved,
        )
        return VerifiedAnswer(
            trajectory=trajectory,
            verification=result,
            answer=presented(trajectory, result),
        )


def _verdict(rate: float, invalid: Sequence[int], *, min_support: float) -> str:
    """grounded | partial | ungrounded.

    A citation to a hop that does not exist caps the verdict at `partial` however
    well the rest scores. It is the one failure detectable with certainty — the
    model referenced a retrieval that never happened — and an answer containing
    one has demonstrated it is willing to invent provenance.
    """
    if rate < min_support:
        return "ungrounded"
    if invalid or rate < GROUNDED:
        return "partial"
    return "grounded"


def _classify(answer: str) -> list[Claim]:
    """Sentences, split and marked factual or connective."""
    claims: list[Claim] = []
    for index, sentence in enumerate(split_sentences(answer)):
        cited = _cited_step(sentence)
        factual = _is_factual(sentence)
        claims.append(
            Claim(
                text=sentence,
                sentence_index=index,
                cited_step=cited,
                # A connective sentence is `supported` by construction: it needs
                # no citation, so it cannot fail to have one. Reporting it as
                # unsupported would put "In summary:" in the flagged list.
                supported=not factual,
                support_excerpt=None,
                support=Support.UNSUPPORTED,
                factual=factual,
            )
        )
    return claims


def _support(
    claims: Sequence[Claim],
    steps: Sequence[Step],
    embedder: Embedder,
    *,
    direct: float,
    inferred: float,
) -> list[Claim]:
    """Each factual claim against every passage the trajectory retrieved.

    Against **every** step rather than only the cited one. A model that wrote a
    true sentence and attributed it to the wrong hop has made a citation mistake,
    not a fabrication, and conflating the two would report the honest error and
    the invented one identically.
    """
    if not claims:
        return []

    units: list[str] = []
    owners: list[int] = []
    truncated: dict[int, bool] = {}
    for hop, step in enumerate(steps, start=1):
        assert step.result is not None  # `verify` filtered
        truncated[hop] = step.result.truncated
        for text in _units(step):
            units.append(text)
            owners.append(hop)

    if not units:
        # A trajectory that retrieved nothing supports nothing. Every factual
        # claim is unsupported, which is the correct reading of an answer written
        # over zero results.
        return [
            Claim(
                text=claim.text,
                sentence_index=claim.sentence_index,
                cited_step=claim.cited_step,
                supported=False,
                support_excerpt=None,
                support=Support.UNSUPPORTED,
                factual=True,
            )
            for claim in claims
        ]

    claim_vectors = embedder.embed_query([claim.text for claim in claims])
    unit_vectors = embedder.embed_passage(units)

    checked: list[Claim] = []
    for claim, vector in zip(claims, claim_vectors, strict=True):
        scores = [_cosine(vector, unit) for unit in unit_vectors]
        best_at = max(range(len(scores)), key=scores.__getitem__)
        best = scores[best_at]
        bearing = tuple(
            sorted({owners[at] for at, score in enumerate(scores) if score >= inferred})
        )

        if best >= direct:
            level = Support.DIRECT
        elif len(bearing) >= 2:
            # Two hops each bear on it and neither says it. That is the
            # multi-hop combination — legitimate, and the model's own.
            level = Support.INFERRED
        else:
            level = Support.UNSUPPORTED

        checked.append(
            Claim(
                text=claim.text,
                sentence_index=claim.sentence_index,
                cited_step=claim.cited_step,
                supported=level is not Support.UNSUPPORTED,
                support_excerpt=(
                    None if level is Support.UNSUPPORTED else _clip(units[best_at], 240)
                ),
                support=level,
                similarity=round(best, 4),
                steps=bearing,
                factual=True,
                from_truncated=truncated.get(owners[best_at], False),
            )
        )
    return checked


def _units(step: Step) -> list[str]:
    """The passages one step offers as evidence.

    **Citations and rendered content both**, and the second is not optional. Four
    of the six tools return facts that are not in their citations at all: a
    decision's reasoning and rejected options are rows, and the memories cited
    beside them are its *evidence*, whose text is about something else entirely.
    Checking a claim about a decision against only that decision's citations
    would mark every correct sentence unsupported.
    """
    result = step.result
    assert result is not None
    units: list[str] = []
    for citation in result.citations:
        text = citation.context.text if citation.context else citation.excerpt
        if len(text.strip()) >= MIN_UNIT_CHARS:
            units.append(_clip(text, UNIT_CHARS))
        if len(units) >= MAX_UNITS_PER_STEP // 2:
            break
    for block in _blocks(result.content):
        units.append(block)
        if len(units) >= MAX_UNITS_PER_STEP:
            break
    return units


def _blocks(content: str) -> list[str]:
    """A tool result cut where it already has seams.

    Blank lines, because every tool's rendering separates its entries with one.
    Cutting at a fixed width instead would split an entry across two units and
    embed each half as something neither of them says.
    """
    blocks: list[str] = []
    for raw in re.split(r"\n\s*\n", content):
        block = " ".join(raw.split())
        if len(block) >= MIN_UNIT_CHARS:
            blocks.append(_clip(block, UNIT_CHARS))
    return blocks


def _cited_step(sentence: str) -> int | None:
    """The hop a sentence cited, or None.

    The first when several are named. `Claim.cited_step` is singular by M7.2's
    shape, and the rest are not lost — support is computed against every step
    regardless, so a sentence citing [2][4] is checked against both whatever this
    returns.
    """
    match = _CITATION.search(sentence)
    if match is None:
        return None
    return int(match.group(1).split(",")[0])


def _is_factual(sentence: str) -> bool:
    """Whether this sentence asserts something the trajectory should support."""
    if _HAS_LETTERS.search(sentence) is None:
        return False
    bare = _ORNAMENT.sub("", sentence).strip()
    lowered = bare.lower()
    if lowered.startswith(_CONNECTIVE_PREFIXES):
        return False
    # A statement of absence is the behaviour this phase exists to encourage.
    # Requiring evidence for "the corpus does not contain this" would score the
    # honest answer below the confident one, which is the exact inversion every
    # grounding check in this project is arranged against.
    return _ABSENCE.search(bare) is None


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine, computed rather than assumed.

    The configured embedder returns unit vectors and a dot product would do —
    but this module takes an `Embedder` protocol, and a stub in a test that
    returns whatever it likes must not silently produce similarities above one.
    """
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return float(dot / (left_norm * right_norm))


def _clip(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
