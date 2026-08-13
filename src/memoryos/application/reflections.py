"""Turning a pattern into prose, and refusing when the evidence is not there.

**Every prior milestone returns retrieved text or computed numbers. This one
produces claims about a person, in fluent English.** That difference is the
whole content of the module, and everything below is arranged around it.

A `patterns` row is already a behavioural claim, but it is a claim shaped like a
table: a sentence assembled from counts, with the decisions it was counted from
listed beside it, and a reader looks at both. A reflection is read the way prose
is read. "You tend to underestimate how long integration takes" is trusted
because it is personal, it sounds exactly like the product working, and there is
nothing in it to argue with — which is why an unfalsifiable behavioural claim is
the most damaging thing this system can emit.

So there are two guards, and the first one is much stronger than the second.

**The refusal happens before the model is called.** A pattern below
`REFLECTION_MIN_CONFIDENCE` produces no reflection — not a hedged one, not a
short one, none. This is not a prompt instruction that a model under pressure to
be helpful can smooth over; it is arithmetic, and on a weak pattern the
`LanguageModel` is never invoked at all. `reflect` prints what would be needed
instead, which is the correct output rather than a failure: "two more decisions
where this belief broke, with none where it held" is something a person can go
and look for, and "no reflections" is not.

**What comes back is verified against the evidence, exactly as M2.6 does, with
the tolerance removed.** M2.6 requires every *factual* sentence to carry a
citation and lets a refusal go uncited. There is no equivalent sentence here, so
every sentence must cite: `domain/grounding.check_reflection`. Uncited sentences
are flagged and kept, because prose with a sentence quietly deleted from the
middle reads as complete and is not. A citation to a decision outside the
pattern's evidence rejects the whole reflection, because at that point the
paragraph is describing somebody using evidence nobody showed it.

What neither guard can check is whether the cited decision actually supports the
sentence. That needs a judge, the only available judge is another language model,
and a model grading its own grounding is not evidence. The answer to it is the
one M2.6 gave: a person reads the output and says whether it is true about them.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.patterns import EvidenceRow, PatternRow, list_patterns
from memoryos.application.ports import LanguageModel
from memoryos.domain.grounding import ReflectionCheck, check_reflection
from memoryos.domain.ids import new_id
from memoryos.domain.patterns import (
    REFLECTION_MIN_CONFIDENCE,
    clears_reflection_bar,
    support_needed_for_reflection,
)
from memoryos.domain.values import PatternRelation

logger = structlog.get_logger(__name__)

# Enough for a paragraph and not enough for an essay. A reflection that runs
# long is one that has started describing the person rather than the decisions,
# and the prompt asks for four sentences for the same reason.
MAX_TOKENS = 400

SYSTEM_PROMPT = """\
You describe one pattern in somebody's past decisions, using only the numbered \
decisions you are given.

Rules, in order of importance:

1. EVERY sentence must cite at least one decision, like [1] or [2, 3], with the \
marker at the end of the sentence. Unlike other writing tasks, there is no \
sentence here that does not need one: a sentence about somebody with no decision \
behind it is the exact failure this task exists to avoid.
2. Cite ONLY numbers from the list you were given. A number outside it discards \
the whole reflection.
3. Decisions marked ARGUES AGAINST are counter-evidence. State them in the same \
paragraph as the claim they weaken — in the same sentence where it reads \
naturally — never in a closing caveat and never left out.
4. Describe only what these specific decisions show. Do not generalise past them, \
do not explain why the person is like this, and do not say what people who do \
this usually do.
5. If the support is thin, or the counter-evidence is close to the support, say \
plainly that the observation is tentative and say what would settle it.
6. No advice, and no "you should". The only exception is advice a cited decision \
records for itself, repeated with that citation.
7. Address the reader as "you", and name decisions by their questions rather than \
by their numbers alone.

One paragraph, at most four sentences. No preamble, no heading, no restating the \
pattern."""

USER_TEMPLATE = """\
Pattern: {statement}

Evidence: {support} decision(s) argue for this, {counter} argue against it.

{decisions}

---

Describe what these decisions show, in one paragraph, citing every sentence."""


class UnknownReflection(LookupError):
    """No reflection with that id."""


# --------------------------------------------------------------------------
# What the model is shown, and what the citations are frozen against
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CitedDecision:
    """One numbered decision in the prompt, and the target of one `[n]`.

    **Numbered per distinct decision and per relation, which is exactly how the
    pattern counts its own support.** Numbering per evidence *row* would let one
    decision with two grouped assumptions appear as `[1]` and `[4]`, and a
    reflection citing both would read as two observations of a person when it is
    one — the mistake `DEFAULT_MIN_SUPPORT` exists to prevent, reintroduced at
    the point where somebody actually reads the claim.

    A decision that both supports and contradicts gets two markers rather than a
    merged one. That happens when two beliefs from the same decision went
    different ways, and it genuinely is evidence on both sides; collapsing it
    would mean choosing a relation to show and hiding the other.
    """

    marker: int
    decision_id: UUID
    question: str
    decided_at: datetime
    relation: PatternRelation
    notes: tuple[str, ...]

    def render(self) -> str:
        side = "ARGUES FOR" if self.relation is PatternRelation.SUPPORTS else ("ARGUES AGAINST")
        lines = [f"[{self.marker}] {side} — {self.question} (decided {self.decided_at:%Y-%m-%d})"]
        lines += [f"      {note}" for note in self.notes]
        return "\n".join(lines)


def number_evidence(pattern: PatternRow) -> tuple[CitedDecision, ...]:
    """The pattern's evidence as a numbered list, deterministically ordered.

    Supporting first, then contradicting, each by date. The order is fixed
    rather than incidental because these numbers are stored: two runs over the
    same pattern must produce the same `[3]`, or a regenerated reflection and an
    old one would disagree about what they cite.

    Counter-evidence is in the *same* list rather than a second one shown after
    it. A model given "here is the evidence, and separately, some caveats" will
    write the caveats last if it writes them at all.
    """
    numbered: list[CitedDecision] = []
    for relation, rows in (
        (PatternRelation.SUPPORTS, pattern.supporting),
        (PatternRelation.CONTRADICTS, pattern.contradicting),
    ):
        for decision_id, group in _by_decision(rows):
            numbered.append(
                CitedDecision(
                    marker=len(numbered) + 1,
                    decision_id=decision_id,
                    question=group[0].decision_question,
                    decided_at=group[0].decided_at,
                    relation=relation,
                    notes=tuple(row.note for row in group if row.note),
                )
            )
    return tuple(numbered)


def _by_decision(
    rows: Sequence[EvidenceRow],
) -> list[tuple[UUID, list[EvidenceRow]]]:
    grouped: dict[UUID, list[EvidenceRow]] = {}
    for row in sorted(rows, key=lambda item: (item.decided_at, str(item.decision_id))):
        grouped.setdefault(row.decision_id, []).append(row)
    return list(grouped.items())


# --------------------------------------------------------------------------
# The result, whether or not anything was written
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reflection:
    """What asking for a reflection produced: prose, or the reason there is none.

    One type for both outcomes on purpose. A separate "refusal" type would let a
    caller handle the happy path and forget the other, and on this corpus the
    other is the common case — `reflect` prints refusals as its main output
    rather than as an error tail.
    """

    pattern_id: UUID
    statement: str
    confidence: float | None
    support: int
    contradictions: int
    citations: tuple[CitedDecision, ...] = ()
    # None whenever nothing was written, for either reason below.
    text: str | None = None
    check: ReflectionCheck | None = None
    model_id: str | None = None
    # Set when the bar was not cleared. The model was not called.
    refused_because: str | None = None
    # What clearing the bar would take. Only meaningful alongside a refusal.
    needed: str | None = None
    # Set when the model *was* called and what came back could not be stored.
    rejected_because: str | None = None
    # Set once stored.
    id: UUID | None = None

    @property
    def written(self) -> bool:
        return self.text is not None

    @property
    def citation_rate(self) -> float | None:
        return None if self.check is None else self.check.citation_rate

    def as_dict(self) -> dict[str, object]:
        return {
            "pattern_id": str(self.pattern_id),
            "statement": self.statement,
            "confidence": self.confidence,
            "written": self.written,
            "text": self.text,
            "model_id": self.model_id,
            "refused_because": self.refused_because,
            "needed": self.needed,
            "rejected_because": self.rejected_because,
            "check": None if self.check is None else self.check.as_dict(),
        }


@dataclass(slots=True)
class ReflectionReport:
    considered: int = 0
    written: int = 0
    # Below the bar: no model call, no text. The expected outcome on a young
    # corpus and the one the CLI leads with.
    refused: list[Reflection] = field(default_factory=list)
    # Generated and thrown away: a fabricated citation, or nothing cited at all.
    rejected: list[Reflection] = field(default_factory=list)
    reflections: list[Reflection] = field(default_factory=list)
    skipped_dismissed: int = 0
    skipped_existing: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "considered": self.considered,
            "written": self.written,
            "refused": len(self.refused),
            "rejected": len(self.rejected),
            "skipped_dismissed": self.skipped_dismissed,
            "skipped_existing": self.skipped_existing,
        }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


async def generate_reflection(
    pattern: PatternRow,
    model: LanguageModel,
    *,
    threshold: float = REFLECTION_MIN_CONFIDENCE,
) -> Reflection:
    """One pattern, described in prose — or refused before a token is spent.

    Three things can come back, and only the first involves the model saying
    anything useful:

    * **A reflection.** Verified, with its citation rate and any uncited
      sentence recorded rather than removed.
    * **A refusal.** The pattern is below the bar. The model is not called, so
      there is no text to be tempted by, and `needed` says what would change it.
    * **A rejection.** The model was called and what came back cited a decision
      it was never given, or cited nothing at all. Nothing is stored.
    """
    citations = number_evidence(pattern)
    base = Reflection(
        pattern_id=pattern.id,
        statement=pattern.statement,
        confidence=pattern.confidence,
        support=pattern.support_count,
        contradictions=pattern.contradiction_count,
        citations=citations,
    )

    if not clears_reflection_bar(pattern.confidence, threshold=threshold):
        # The milestone's acceptance criterion, and the reason it is checked
        # here rather than in the prompt: a threshold a model is *asked* to
        # respect is a threshold that fails exactly when the output is most
        # fluent.
        more = support_needed_for_reflection(
            pattern.support_count, pattern.contradiction_count, threshold=threshold
        )
        shown = "no confidence" if pattern.confidence is None else f"{pattern.confidence:.2f}"
        logger.info(
            "reflection.refused",
            pattern_id=str(pattern.id),
            confidence=pattern.confidence,
            threshold=threshold,
        )
        return replace(
            base,
            refused_because=(f"confidence {shown} is below the {threshold:.2f} a reflection needs"),
            needed=(
                f"{more} more decision(s) agreeing, with no further "
                f"counter-evidence, against the {pattern.support_count} supporting "
                f"and {pattern.contradiction_count} contradicting it has now"
            ),
        )

    if not citations:
        # Unreachable while `ck_patterns_support_positive` holds. Kept because
        # the thing it guards against — prose about somebody with no decision
        # under it — is the one failure worth a redundant check.
        return replace(base, rejected_because="the pattern cites no decisions")

    text = (
        await model.complete(
            SYSTEM_PROMPT,
            USER_TEMPLATE.format(
                statement=pattern.statement,
                support=pattern.support_count,
                counter=pattern.contradiction_count,
                decisions="\n".join(item.render() for item in citations),
            ),
            max_tokens=MAX_TOKENS,
        )
    ).strip()

    check = check_reflection(text, {item.marker for item in citations})
    logger.info(
        "reflection.generated",
        pattern_id=str(pattern.id),
        model_id=model.model_id,
        citation_rate=round(check.citation_rate, 3),
        uncited=len(check.uncited),
        out_of_evidence=len(check.out_of_evidence),
    )

    if not check.writable:
        reason = (
            f"cited {check.out_of_evidence} — not among the {len(citations)} decisions it was given"
            if check.out_of_evidence
            else "nothing in it cites a decision, so there is nothing to check"
        )
        return replace(base, check=check, model_id=model.model_id, rejected_because=reason)

    return replace(base, text=text, check=check, model_id=model.model_id)


# --------------------------------------------------------------------------
# Orchestration and storage
# --------------------------------------------------------------------------


async def reflect(
    sessions: async_sessionmaker[AsyncSession],
    model: LanguageModel,
    *,
    pattern_id: UUID | None = None,
    threshold: float = REFLECTION_MIN_CONFIDENCE,
    regenerate: bool = False,
) -> ReflectionReport:
    """Generate for every pattern that clears the bar, and store what verifies.

    A pattern whose reflection was dismissed is skipped outright and `regenerate`
    does not override it. Everything else in this file is a judgement about
    evidence; that one is a judgement about the person's own judgement, and it
    outranks.
    """
    patterns = await list_patterns(sessions, limit=1000)
    if pattern_id is not None:
        patterns = [row for row in patterns if row.id == pattern_id]

    existing = await _existing_by_pattern(sessions, [row.id for row in patterns])
    report = ReflectionReport(considered=len(patterns))

    for pattern in patterns:
        states = existing.get(pattern.id, [])
        if any(dismissed for dismissed, _ in states):
            report.skipped_dismissed += 1
            continue
        if states and not regenerate:
            report.skipped_existing += 1
            continue

        reflection = await generate_reflection(pattern, model, threshold=threshold)
        if reflection.refused_because is not None:
            report.refused.append(reflection)
            continue
        if reflection.rejected_because is not None:
            report.rejected.append(reflection)
            continue

        stored = await store(sessions, reflection)
        report.reflections.append(stored)
        report.written += 1

    logger.info("reflections.run", **report.as_dict())
    return report


async def store(sessions: async_sessionmaker[AsyncSession], reflection: Reflection) -> Reflection:
    """Write the text and freeze its numbering, in one transaction.

    The citations written are the markers the text actually uses, not every
    decision the model was shown. A row for a decision the prose never mentions
    would be a link the reader cannot find in the paragraph.
    """
    if reflection.text is None or reflection.check is None:
        raise ValueError("a refused or rejected reflection is not storable")

    by_marker = {item.marker: item for item in reflection.citations}
    used = sorted(set(reflection.check.cited_indices))
    reflection_id = new_id()

    async with sessions.begin() as session:
        session.add(
            models.Reflection(
                id=reflection_id,
                pattern_id=reflection.pattern_id,
                text=reflection.text,
                citation_rate=reflection.check.citation_rate,
                model_id=reflection.model_id or "unknown",
            )
        )
        for marker in used:
            item = by_marker[marker]
            session.add(
                models.ReflectionCitation(
                    id=new_id(),
                    reflection_id=reflection_id,
                    marker=marker,
                    decision_id=item.decision_id,
                    relation=item.relation.value,
                )
            )
    logger.info(
        "reflection.stored",
        reflection_id=str(reflection_id),
        pattern_id=str(reflection.pattern_id),
        citations=len(used),
    )
    return replace(reflection, id=reflection_id)


async def _existing_by_pattern(
    sessions: async_sessionmaker[AsyncSession], pattern_ids: Sequence[UUID]
) -> dict[UUID, list[tuple[bool, UUID]]]:
    if not pattern_ids:
        return {}
    async with sessions() as session:
        rows = await session.execute(
            select(
                models.Reflection.pattern_id,
                models.Reflection.dismissed_at,
                models.Reflection.id,
            ).where(models.Reflection.pattern_id.in_(list(pattern_ids)))
        )
    found: dict[UUID, list[tuple[bool, UUID]]] = {}
    for pattern_id, dismissed_at, reflection_id in rows:
        found.setdefault(pattern_id, []).append((dismissed_at is not None, reflection_id))
    return found


# --------------------------------------------------------------------------
# Reading, acknowledging, refusing
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CitationRow:
    marker: int
    decision_id: UUID
    decision_question: str
    relation: PatternRelation


@dataclass(frozen=True, slots=True)
class ReflectionRow:
    id: UUID
    pattern_id: UUID
    pattern_statement: str
    text: str
    citation_rate: float | None
    model_id: str
    generated_at: datetime
    acknowledged_at: datetime | None
    dismissed_at: datetime | None
    dismissed_reason: str | None
    support_count: int
    contradiction_count: int
    citations: list[CitationRow]

    @property
    def uncited(self) -> list[str]:
        """Sentences carrying no citation, recomputed from the stored text.

        Derivable from the text alone — a sentence either contains a `[n]` or it
        does not — so there is no column for it to disagree with. `citation_rate`
        *is* stored, because it was measured against the evidence as it stood at
        generation and that evidence can move underneath it.
        """
        return check_reflection(self.text, {item.marker for item in self.citations}).uncited


async def list_reflections(
    sessions: async_sessionmaker[AsyncSession],
    *,
    include_dismissed: bool = False,
    limit: int = 100,
) -> list[ReflectionRow]:
    stmt = (
        select(models.Reflection, models.Pattern)
        .join(models.Pattern, models.Pattern.id == models.Reflection.pattern_id)
        .order_by(models.Reflection.generated_at.desc())
        .limit(limit)
    )
    if not include_dismissed:
        stmt = stmt.where(models.Reflection.dismissed_at.is_(None))

    async with sessions() as session:
        rows = list(await session.execute(stmt))
        if not rows:
            return []
        citations = list(
            await session.execute(
                select(models.ReflectionCitation, models.Decision.question)
                .join(
                    models.Decision,
                    models.Decision.id == models.ReflectionCitation.decision_id,
                )
                .where(models.ReflectionCitation.reflection_id.in_([row[0].id for row in rows]))
                .order_by(models.ReflectionCitation.marker)
            )
        )

    by_reflection: dict[UUID, list[CitationRow]] = {}
    for citation, question in citations:
        by_reflection.setdefault(citation.reflection_id, []).append(
            CitationRow(
                marker=citation.marker,
                decision_id=citation.decision_id,
                decision_question=question,
                relation=PatternRelation(citation.relation),
            )
        )

    return [
        ReflectionRow(
            id=reflection.id,
            pattern_id=pattern.id,
            pattern_statement=pattern.statement,
            text=reflection.text,
            citation_rate=reflection.citation_rate,
            model_id=reflection.model_id,
            generated_at=reflection.generated_at,
            acknowledged_at=reflection.acknowledged_at,
            dismissed_at=reflection.dismissed_at,
            dismissed_reason=reflection.dismissed_reason,
            support_count=pattern.support_count,
            contradiction_count=pattern.contradiction_count,
            citations=by_reflection.get(reflection.id, []),
        )
        for reflection, pattern in rows
    ]


async def acknowledge(sessions: async_sessionmaker[AsyncSession], reflection_id: UUID) -> None:
    """Record that somebody read it.

    Not agreement, and nothing weights a reflection by it. It exists so a view
    can stop putting an unread claim first, which is the whole of its job.
    """
    async with sessions.begin() as session:
        row = await session.get(models.Reflection, reflection_id)
        if row is None:
            raise UnknownReflection(f"no reflection {reflection_id}")
        row.acknowledged_at = datetime.now(UTC)
    logger.info("reflection.acknowledged", reflection_id=str(reflection_id))


async def dismiss(
    sessions: async_sessionmaker[AsyncSession], reflection_id: UUID, *, reason: str
) -> None:
    """ "This is wrong about me." Permanent, and it stops regeneration.

    Hiding the row would not be enough: the next `reflect --all` would write the
    same claim again with a new id, and a system that argues back is a system you
    stop reading. `reflect` skips a pattern with a dismissed reflection whatever
    else has changed about it.
    """
    if not reason.strip():
        raise ValueError("a dismissal needs a reason")
    async with sessions.begin() as session:
        row = await session.get(models.Reflection, reflection_id)
        if row is None:
            raise UnknownReflection(f"no reflection {reflection_id}")
        row.dismissed_at = datetime.now(UTC)
        row.dismissed_reason = reason.strip()
    logger.info("reflection.dismissed", reflection_id=str(reflection_id), reason=reason)
