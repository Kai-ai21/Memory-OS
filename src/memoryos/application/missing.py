"""Finding what is absent, from four things that are present.

Every detector here answers the same shaped question — *what did your history do
that this does not?* — and each is grounded in a milestone that already built the
measurement. Nothing is inferred from a model's opinion of good practice, and
`domain/missing.py` says at length why.

**A gap that cannot cite is not emitted.** M5.4's rule for reflections, and it
matters more here: a reflection is a claim about a run of decisions that a reader
can check against the decisions; a gap is a claim about something that *is not
there*, which has no referent at all except the history that makes it sayable.
Without citations it is unfalsifiable by construction.

### The four

* `UNSTATED_ASSUMPTION` — decisions resembling this one recorded a belief this one
  does not. Cites the decisions that recorded it.
* `REPEATED_PATTERN` — this resembles a pattern whose outcomes went badly. Cites
  the pattern and its evidence; M5.3 has already refused it below three decisions.
* `ORPHANED_WORK` — an entity was active, then was not, and no decision says why.
  Cites the memories either side of the silence.
* `UNEVALUATED_ASSUMPTION` — beliefs old enough to check that nobody has checked.
  Cites the assumptions, and is the only one whose evidence is a *missing row*.

### Matching is term overlap, not embedding

`--about` finds context by counting shared content words, the same mechanism
`get_decisions` uses and with the fix M7.3 forced on it: ranked by how many terms
match rather than filtered by whether any does, because an `any()` lets the most
generic word in a query decide the result.

Deliberately not embedded. Sixteen decisions is not a retrieval problem, a second
ranking would be a second thing to explain, and the failure mode of term overlap —
finding nothing — is the failure mode this milestone wants: silence is the correct
answer here far more often than a match is.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.missing import (
    MIN_SUPPORT,
    GapKind,
    Silence,
    gap_confidence,
    worth_saying,
)

logger = structlog.get_logger(__name__)

# How old an assumption must be before nobody having checked it is a gap rather
# than a normal state of affairs.
#
# Thirty days, and it is the one number here a corpus can move under. A belief
# recorded yesterday is not overdue; one from a quarter ago that nothing has
# tested is a decision still resting on an unexamined premise. On this project's
# own corpus — every decision inside a fortnight — this fires for nothing, which
# is the correct answer rather than a reason to lower it.
STALE_AFTER = timedelta(days=30)

# Silence long enough to call an entity orphaned. M4.3's own default, reused so
# that `missing` and `gaps` do not disagree about what a gap is.
ORPHAN_GAP = timedelta(days=30)

# Content terms too common to establish that two questions are about the same
# thing. Shorter than a general stopword list on purpose: the technical
# vocabulary is the signal, and dropping "index" or "store" would make every
# comparison in this corpus vacuous.
_COMMON = frozenset(
    (
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
        "from", "at", "is", "are", "was", "were", "be", "been", "this", "that",
        "it", "its", "as", "not", "no", "what", "which", "who", "when", "where",
        "how", "why", "do", "does", "did", "can", "could", "will", "would",
        "should", "we", "our", "they", "their", "have", "has", "had", "into",
        "one", "two", "three", "more", "most", "some", "any", "all", "out", "up",
        "rather", "than", "then", "there", "here", "over", "under", "about",
    )
)

_MIN_TERM = 3


@dataclass(frozen=True, slots=True)
class Evidence:
    """One thing a gap points at, so a reader can disagree with something."""

    kind: str
    ref_id: UUID | None
    label: str


@dataclass(frozen=True, slots=True)
class Gap:
    """One absence, with the history that makes it sayable."""

    kind: GapKind
    statement: str
    subject: str
    supporting: int
    contradicting: int = 0
    evidence: tuple[Evidence, ...] = ()

    @property
    def confidence(self) -> float:
        return gap_confidence(self.supporting, self.contradicting)

    @property
    def sayable(self) -> bool:
        """The bar, and the citation rule.

        **Two citations minimum, checked here rather than trusted.** A gap whose
        support count says two and whose evidence list says none has counted
        something it cannot show, and the count is the part a reader cannot
        verify.
        """
        return worth_saying(self.supporting, self.contradicting) and len(
            self.evidence
        ) >= MIN_SUPPORT


@dataclass(frozen=True, slots=True)
class MissingReport:
    """What was found, and — usually — why nothing was."""

    gaps: list[Gap] = field(default_factory=list)
    silence: Silence = field(default_factory=Silence)
    # What `--about` resolved to, so a reader can see the analysis ran against
    # what they meant. An empty string means the whole corpus.
    context: str = ""


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        word
        for word in re.findall(r"[a-z_][a-z0-9_]*", text.lower())
        if len(word) >= _MIN_TERM and word not in _COMMON
    )


def _overlap(left: str, right: str) -> int:
    return len(_terms(left) & _terms(right))


def _normalise(statement: str) -> str:
    """An assumption reduced to its content words, for comparing two of them.

    Two people writing down the same belief will not write the same sentence, and
    comparing the sentences would make every assumption unique — which would make
    `UNSTATED_ASSUMPTION` fire never rather than fire wrongly. Comparing the
    content-word sets is coarse and is the coarseness this corpus can support.
    """
    return " ".join(sorted(_terms(statement)))


# --------------------------------------------------------------------------
# 1. Unstated assumptions
# --------------------------------------------------------------------------


async def _unstated_assumptions(
    session: AsyncSession, target: models.Decision | None, about: str
) -> list[Gap]:
    """Beliefs that decisions like this one wrote down and this one did not.

    Peers are decisions sharing content words with the subject. For each belief
    two or more peers recorded, and the subject did not, that is a gap — and the
    evidence is the peer decisions themselves, named, so the reader can look at
    two specific occasions rather than at a statistic.

    **Contradicting evidence is a peer whose version of the belief broke.** If
    the last two people to write this assumption down were wrong about it, "you
    forgot to write it down" is not the useful sentence, and the confidence
    should fall rather than the gap being silently dropped.
    """
    subject = target.question if target is not None else about
    if not subject.strip():
        return []

    rows = list(
        await session.execute(
            select(
                models.Decision.id,
                models.Decision.question,
                models.DecisionAssumption.id,
                models.DecisionAssumption.statement,
                models.DecisionAssumption.held,
            ).join(
                models.DecisionAssumption,
                models.DecisionAssumption.decision_id == models.Decision.id,
            )
        )
    )

    target_id = target.id if target is not None else None
    already = {
        _normalise(statement)
        for decision_id, _, _, statement, _ in rows
        if decision_id == target_id
    }

    # Grouped by belief, then by decision, because the unit of support is a
    # decision. Four assumptions from two decisions is two occasions.
    by_belief: dict[str, dict[UUID, tuple[str, str, str | None]]] = {}
    for decision_id, question, _assumption_id, statement, held in rows:
        if decision_id == target_id:
            continue
        if _overlap(subject, question) == 0:
            continue
        key = _normalise(statement)
        if not key or key in already:
            continue
        by_belief.setdefault(key, {})[decision_id] = (question, statement, held)

    gaps: list[Gap] = []
    for belief, peers in by_belief.items():
        broke = sum(1 for _, _, held in peers.values() if held in ("failed", "partially"))
        example = next(iter(peers.values()))[1]
        gaps.append(
            Gap(
                kind=GapKind.UNSTATED_ASSUMPTION,
                subject=belief,
                statement=(
                    f"{len(peers)} decision(s) like this one recorded an assumption "
                    f"this one does not — e.g. {example!r}."
                ),
                supporting=len(peers),
                contradicting=broke,
                evidence=tuple(
                    Evidence(kind="decision", ref_id=decision_id, label=question)
                    for decision_id, (question, _, _) in peers.items()
                ),
            )
        )
    return gaps


# --------------------------------------------------------------------------
# 2. Repeated patterns
# --------------------------------------------------------------------------


async def _repeated_patterns(
    session: AsyncSession, target: models.Decision | None, about: str
) -> list[Gap]:
    """Patterns whose history went badly, that this resembles.

    The thinnest detector and the most defensible, because M5.3 did the work: a
    pattern has already been required to rest on three distinct decisions, has
    already had its supporting and contradicting counts computed, and has already
    been offered for dismissal. Dismissed ones are excluded at the source — a
    behavioural claim somebody rejected must not return wearing the word "gap".
    """
    subject = target.question if target is not None else about
    patterns = list(
        (
            await session.execute(
                select(models.Pattern).where(models.Pattern.dismissed_at.is_(None))
            )
        ).scalars()
    )

    gaps: list[Gap] = []
    for pattern in patterns:
        if subject.strip() and _overlap(subject, pattern.statement) == 0:
            continue
        evidence = list(
            (
                await session.execute(
                    select(models.PatternEvidence).where(
                        models.PatternEvidence.pattern_id == pattern.id
                    )
                )
            ).scalars()
        )
        gaps.append(
            Gap(
                kind=GapKind.REPEATED_PATTERN,
                subject=pattern.subject_key,
                statement=(
                    f"This resembles a recorded pattern: {pattern.statement} "
                    "Nothing here says you have accounted for it."
                ),
                supporting=pattern.support_count,
                contradicting=pattern.contradiction_count,
                evidence=(
                    Evidence(kind="pattern", ref_id=pattern.id, label=pattern.statement),
                    *(
                        Evidence(
                            kind="decision",
                            ref_id=row.decision_id,
                            label=f"{row.relation} the pattern",
                        )
                        for row in evidence
                        if row.decision_id is not None
                    ),
                ),
            )
        )
    return gaps


# --------------------------------------------------------------------------
# 3. Orphaned work
# --------------------------------------------------------------------------


async def _orphaned_work(session: AsyncSession, about: str) -> list[Gap]:
    """Entities that were active, then were not, with no decision recording why.

    M4.0 measures the silence; this asks whether anything explains it. **The
    "with no decision" half is what makes it a gap rather than a timeline
    entry** — work that stopped because a decision says it stopped is not
    missing anything.

    Support is the memories that mentioned the entity before it went quiet, which
    is what establishes the work was real rather than a passing mention. Two is
    the floor: one mention and a silence is a word somebody used once.
    """
    mentions = list(
        await session.execute(
            select(
                models.EntityMention.entity_id,
                models.Entity.canonical_name,
                func.count(func.distinct(models.EntityMention.memory_id)),
                func.max(models.Memory.occurred_at),
            )
            .join(models.Entity, models.Entity.id == models.EntityMention.entity_id)
            .join(models.Memory, models.Memory.id == models.EntityMention.memory_id)
            .where(models.Memory.is_current.is_(True))
            .group_by(models.EntityMention.entity_id, models.Entity.canonical_name)
            .having(func.count(func.distinct(models.EntityMention.memory_id)) >= MIN_SUPPORT)
        )
    )
    if not mentions:
        return []

    newest = (
        await session.execute(
            select(func.max(models.Memory.occurred_at)).where(
                models.Memory.is_current.is_(True)
            )
        )
    ).scalar_one()
    if newest is None:
        return []

    decisions = list(
        (await session.execute(select(models.Decision.id, models.Decision.question))).all()
    )

    gaps: list[Gap] = []
    for entity_id, name, seen, last in mentions:
        if last is None or newest - last < ORPHAN_GAP:
            continue
        if about.strip() and _overlap(about, name) == 0:
            continue
        # A decision that names the entity is an explanation, so it counts
        # against rather than being ignored.
        explaining = [
            (decision_id, question)
            for decision_id, question in decisions
            if _overlap(name, question) > 0
        ]
        memories = list(
            await session.execute(
                select(models.Memory.id, models.Memory.external_key)
                .join(
                    models.EntityMention,
                    models.EntityMention.memory_id == models.Memory.id,
                )
                .where(models.EntityMention.entity_id == entity_id)
                .limit(5)
            )
        )
        gaps.append(
            Gap(
                kind=GapKind.ORPHANED_WORK,
                subject=name,
                statement=(
                    f"{name!r} was active across {seen} memories and nothing has "
                    f"mentioned it since {last:%Y-%m-%d}. No decision records why "
                    "it stopped."
                ),
                supporting=seen,
                contradicting=len(explaining),
                evidence=tuple(
                    Evidence(kind="memory", ref_id=memory_id, label=key)
                    for memory_id, key in memories
                ),
            )
        )
    return gaps


# --------------------------------------------------------------------------
# 4. Unevaluated assumptions
# --------------------------------------------------------------------------


async def _unevaluated_assumptions(
    session: AsyncSession, target: models.Decision | None, about: str, *, now: datetime
) -> list[Gap]:
    """Beliefs old enough to check that nobody has checked.

    **The only one of the four whose evidence is the absence of a row.** The
    others cite something that exists and observe it is not here; this cites the
    assumptions themselves and observes that none of them has an `evaluated_at`.

    Per decision rather than per assumption, and two is the floor: one unchecked
    belief on a decision is ordinary, and a decision resting entirely on
    unexamined premises after a month is the thing worth a sentence.
    """
    cutoff = now - STALE_AFTER
    rows = list(
        await session.execute(
            select(
                models.Decision.id,
                models.Decision.question,
                models.Decision.decided_at,
                models.DecisionAssumption.id,
                models.DecisionAssumption.statement,
                models.DecisionAssumption.held,
            )
            .join(
                models.DecisionAssumption,
                models.DecisionAssumption.decision_id == models.Decision.id,
            )
            .where(models.Decision.decided_at < cutoff)
        )
    )

    by_decision: dict[UUID, list[tuple[str, datetime, UUID, str, str | None]]] = {}
    for decision_id, question, decided_at, assumption_id, statement, held in rows:
        by_decision.setdefault(decision_id, []).append(
            (question, decided_at, assumption_id, statement, held)
        )

    gaps: list[Gap] = []
    for decision_id, members in by_decision.items():
        question, decided_at, *_ = members[0]
        if target is not None and decision_id != target.id:
            continue
        if target is None and about.strip() and _overlap(about, question) == 0:
            continue
        unevaluated = [row for row in members if row[4] is None]
        evaluated = len(members) - len(unevaluated)
        if not unevaluated:
            continue
        age = (now - decided_at).days
        gaps.append(
            Gap(
                kind=GapKind.UNEVALUATED_ASSUMPTION,
                subject=str(decision_id),
                statement=(
                    f"{len(unevaluated)} assumption(s) behind {question!r} have gone "
                    f"{age} days without being checked. The decision still rests on "
                    "them."
                ),
                supporting=len(unevaluated),
                # An assumption already evaluated on the same decision argues that
                # the checking happens, so the remaining ones are less notable.
                contradicting=evaluated,
                evidence=tuple(
                    Evidence(kind="assumption", ref_id=assumption_id, label=statement)
                    for _, _, assumption_id, statement, _ in unevaluated
                ),
            )
        )
    return gaps


# --------------------------------------------------------------------------
# The use case
# --------------------------------------------------------------------------


async def find_missing(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    about: str = "",
    kind: GapKind | None = None,
    now: datetime | None = None,
) -> MissingReport:
    """What is absent, given a topic or a decision — or nothing, for the corpus.

    `about` is matched against decision questions first, so a caller who names a
    decision gets that decision's context rather than a text search over it. The
    fallback is the free text itself, which is what a topic like "choosing a
    vector store" resolves to.
    """
    moment = now or datetime.now(UTC)

    async with session_factory() as session:
        target = await _resolve(session, about)
        candidates: list[Gap] = []
        candidates.extend(await _unstated_assumptions(session, target, about))
        candidates.extend(await _repeated_patterns(session, target, about))
        candidates.extend(await _orphaned_work(session, about))
        candidates.extend(
            await _unevaluated_assumptions(session, target, about, now=moment)
        )

    if kind is not None:
        candidates = [gap for gap in candidates if gap.kind is kind]

    sayable = [gap for gap in candidates if gap.sayable]
    outweighed = sum(
        1
        for gap in candidates
        if gap.supporting >= MIN_SUPPORT and gap.contradicting >= gap.supporting
    )
    silence = Silence(
        considered=len(candidates),
        below_support=sum(1 for gap in candidates if gap.supporting < MIN_SUPPORT),
        outweighed=outweighed,
        best_support=max((gap.supporting for gap in candidates), default=0),
    )
    sayable.sort(key=lambda gap: (-gap.confidence, gap.statement))

    logger.info(
        "missing.analysed",
        about=about or "(corpus)",
        considered=len(candidates),
        emitted=len(sayable),
    )
    return MissingReport(
        gaps=sayable,
        silence=silence,
        context=(target.question if target is not None else about),
    )


async def _resolve(session: AsyncSession, about: str) -> models.Decision | None:
    """The decision `about` names, if it names one.

    Ranked by how many terms match, which is M7.3's fix to `get_decisions`
    applied here at the same time rather than a milestone later: an `any()` over
    substrings lets the most generic word in a query pick the answer, and this
    module would have inherited that.
    """
    if not about.strip():
        return None
    try:
        return await session.get(models.Decision, UUID(about))
    except ValueError:
        pass

    decisions = list(
        (await session.execute(select(models.Decision))).scalars()
    )
    scored = [
        (_overlap(about, decision.question), decision)
        for decision in decisions
    ]
    best = max(scored, key=lambda row: row[0], default=(0, None))
    # Two shared content words before a free-text topic is treated as naming a
    # specific decision. One is a coincidence, and resolving to the wrong
    # decision would analyse the wrong context silently.
    return best[1] if best[0] >= 2 else None


def by_kind(gaps: Sequence[Gap]) -> dict[str, int]:
    counted = {member.value: 0 for member in GapKind}
    for gap in gaps:
        counted[gap.kind.value] += 1
    return counted
