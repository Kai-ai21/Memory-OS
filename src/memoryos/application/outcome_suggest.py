"""Finding candidate outcomes with the temporal layer, and refusing to trust them.

**This is where Phase 4 pays for itself.** An outcome is a temporal claim: this
happened *after* that, close enough in time to be connected, and about the same
things. M4.0 stored the machinery for the first two — `memories_in_range` over
`occurred_at`, with nulls excluded rather than defaulted — and M3.2's resolved
entities supply the third. Without any of it, "what happened after this
decision" is a question nothing in the system could ask.

**And this is the milestone where being wrong is most expensive.** A decision
suggestion that is wrong proposes a record of a choice nobody made; an outcome
suggestion that is wrong asserts that one thing *caused* another. Post hoc ergo
propter hoc is the oldest error there is, and a language model shown two
related-looking documents from the same repository will make it every time,
fluently and with a plausible rationale. M5.4 would then produce a behavioural
claim — "you consistently underestimate X" — resting on a coincidence of file
modification times.

So the pipeline is conservative at four separate points, and each one is a place
a candidate is dropped rather than downgraded:

1. **Strictly after.** A memory occurring at or before `decided_at` is not a
   weak outcome, it is not an outcome. The database says so too: `gap_days > 0`.
2. **Inside a window**, derived per decision — see `window_for`.
3. **Sharing a resolved entity** with the decision's evidence. When the corpus
   has no extraction coverage this test cannot be *run*, which is different from
   failing it, and the difference is recorded on every row rather than silently
   resolved either way.
4. **Judged by a model that is allowed to say "unsure"**, and required to. An
   unsure verdict never reaches the queue, and neither does a `yes` below the
   confidence floor.

Whatever survives lands in a review queue and is never auto-committed, the same
rule M5.0 established and for a sharper reason.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application import temporal
from memoryos.application.decisions import UnknownDecision
from memoryos.application.outcomes import (
    InvalidOutcome,
    OutcomeDraft,
    OutcomeEvidenceInput,
    record,
)
from memoryos.application.ports import LanguageModel
from memoryos.domain.backoff import wait_for
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import TransientError
from memoryos.domain.values import (
    DecisionStatus,
    EvidenceKind,
    OutcomeVerdict,
    SuggestionStatus,
    TimeProvenance,
)

logger = structlog.get_logger(__name__)

PROMPT_VERSION = "v1"
_MAX_ATTEMPTS = 6
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

# --------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------

# There is no correct window and this is a heuristic. A deployment decision
# shows its outcome in days; an architectural one in months; some never resolve
# at all. Nothing in this corpus, or in any corpus this system has seen,
# measures which — so what follows is a *stated guess* rather than a finding,
# and it is written here in one place so that replacing it later is a diff
# rather than an excavation.
#
# The rule the milestone asks for: derive it from the decision's own confidence,
# because **a low-confidence decision is one you expected to learn about
# sooner**. That is a real intuition — uncertainty is usually uncertainty about
# something that will show itself — and it has the useful property of being
# falsifiable once M5.2 has evaluated enough assumptions to say whether
# low-confidence decisions did in fact resolve faster.
#
# Linear between two bounds that are themselves judgement calls:
#
#   30 days   the shortest window worth asking about. Below a month, a corpus
#             dated by filesystem mtimes cannot distinguish "after" from "at the
#             same time as".
#   180 days  six months, past which "after" stops implying "because of" for
#             anything but the largest decisions.
#
# A decision with no recorded confidence gets `DEFAULT_WINDOW_DAYS` rather than
# a midpoint, because the absence of a number is not the same as 0.5 and
# pretending otherwise would put a made-up confidence into a derived window.
MIN_WINDOW_DAYS = 30.0
MAX_WINDOW_DAYS = 180.0
DEFAULT_WINDOW_DAYS = 90.0


def window_for(confidence: float | None, *, override: float | None = None) -> float:
    """How long after a decision to look, in days.

    A heuristic. See the block above for what it assumes and what would falsify
    it. `override` is the `--window-days` flag, which wins outright — the point
    of a stated heuristic is that somebody can disagree with it per run.
    """
    if override is not None:
        if override <= 0:
            raise ValueError(f"a window is a positive number of days, got {override}")
        return float(override)
    if confidence is None:
        return DEFAULT_WINDOW_DAYS
    return MIN_WINDOW_DAYS + (MAX_WINDOW_DAYS - MIN_WINDOW_DAYS) * confidence


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------

# Below this, the model is guessing and the candidate is dropped. Higher than
# M3.1's 0.5 on purpose: a wrongly extracted entity is noise in a graph, and a
# wrongly linked outcome is a false causal claim that M5.4 will state as a fact
# about how somebody works.
MIN_JUDGE_CONFIDENCE = 0.6

_SYSTEM = """\
You judge whether a document describes the OUTCOME of a decision. You return \
JSON and nothing else.

You are given a decision — the question, what was chosen, and what the decider \
expected — and one document that was written or modified AFTER it. Occurring \
afterwards is not evidence of anything on its own. Two documents in the same \
project are related by default; that is not an outcome.

An outcome is a document that says something about WHAT HAPPENED as a result of \
the choice: that it worked, that it broke, that it was reversed, that it cost \
something unforeseen, that a measurement came in.

Rules, in order of importance:

1. If the document merely mentions the same topic, technology, or files, answer \
"no". Same subject matter is NOT an outcome.
2. If you cannot tell, answer "unsure". This is a common and correct answer and \
it is much better than a confident guess. Do not reason your way to "yes".
3. Use only what the document says. Never supply a consequence you can imagine \
or that seems likely given what you know.
4. Answer "yes" only if you could quote the part of the document that reports \
the result.

When the answer is "yes", give a verdict:
  worked     — the choice achieved what it was for
  failed     — it did not, or it was reversed
  mixed      — it achieved its aim and cost something unforeseen
  too_early  — the document is about the decision but reports no result yet

Return this JSON shape and nothing else — no prose, no markdown fences:

{"answer": "yes|no|unsure", "verdict": "worked|failed|mixed|too_early", \
"description": "one sentence on what happened", "rationale": "the part of the \
document that reports it", "confidence": 0.0}

For "no" and "unsure", the other fields may be null."""

_RETRY_REMINDER = """\

Your previous response was not valid JSON. Return ONLY a JSON object. Start \
your response with { and end it with }. No explanation, no markdown fences, no \
text before or after the JSON."""


class _Judgement(BaseModel):
    answer: str
    verdict: str | None = None
    description: str | None = None
    rationale: str | None = None
    confidence: float = Field(default=0.0)


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    """One memory that occurred after a decision, with the basis for saying so."""

    memory_id: UUID
    source_name: str
    external_key: str
    occurred_at: datetime
    occurred_at_source: TimeProvenance
    text: str
    gap_days: float
    shared_entities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """The decision as the candidate search and the prompt both need it."""

    id: UUID
    question: str
    chosen: str
    expected_outcome: str | None
    confidence: float | None
    decided_at: datetime
    evidence_memory_ids: tuple[UUID, ...]


async def open_decisions(
    sessions: async_sessionmaker[AsyncSession], *, decision_id: UUID | None = None
) -> list[DecisionContext]:
    """Decisions worth looking for outcomes of.

    Open ones only, by default. A settled or reversed decision has had its
    verdict recorded by a person, and proposing further outcomes for it would be
    asking a model to second-guess testimony.
    """
    stmt = select(models.Decision).order_by(models.Decision.decided_at)
    if decision_id is not None:
        stmt = stmt.where(models.Decision.id == decision_id)
    else:
        stmt = stmt.where(models.Decision.status == DecisionStatus.OPEN.value)

    async with sessions() as session:
        rows = list((await session.execute(stmt)).scalars())
        if decision_id is not None and not rows:
            raise UnknownDecision(f"no decision {decision_id}")
        evidence = list(
            await session.execute(
                select(
                    models.DecisionEvidence.decision_id,
                    models.DecisionEvidence.memory_id,
                ).where(
                    models.DecisionEvidence.decision_id.in_([row.id for row in rows])
                    if rows
                    else text("false")
                )
            )
        )

    by_decision: dict[UUID, list[UUID]] = {}
    for decision, memory_id in evidence:
        by_decision.setdefault(decision, []).append(memory_id)

    return [
        DecisionContext(
            id=row.id,
            question=row.question,
            chosen=row.chosen,
            expected_outcome=row.expected_outcome,
            confidence=row.confidence,
            decided_at=row.decided_at,
            evidence_memory_ids=tuple(by_decision.get(row.id, [])),
        )
        for row in rows
    ]


async def resolved_entities(
    sessions: async_sessionmaker[AsyncSession], memory_ids: tuple[UUID, ...]
) -> dict[UUID, str]:
    """The entities those memories mention, followed through M3.2's merges.

    Keyed by the *winner's* id, because a merged-away entity is a name for the
    same thing with none of the rows: a decision whose evidence was extracted
    before a merge and a candidate extracted after it would share nothing at all
    if the raw ids were compared, and the overlap test would fail for a reason
    that has nothing to do with the corpus.
    """
    if not memory_ids:
        return {}
    winner = func.coalesce(models.Entity.merged_into_id, models.Entity.id)
    stmt = (
        select(winner, models.Entity.canonical_name)
        .join(models.EntityMention, models.EntityMention.entity_id == models.Entity.id)
        .where(models.EntityMention.memory_id.in_(memory_ids))
        .distinct()
    )
    async with sessions() as session:
        rows = await session.execute(stmt)
    return {row[0]: row[1] for row in rows}


async def _source_names(
    sessions: async_sessionmaker[AsyncSession],
) -> dict[UUID, str]:
    async with sessions() as session:
        rows = await session.execute(select(models.Source.id, models.Source.name))
    return {row[0]: row[1] for row in rows}


async def find_candidates(
    sessions: async_sessionmaker[AsyncSession],
    decision: DecisionContext,
    *,
    window_days: float,
    limit: int = 10,
) -> tuple[list[Candidate], str]:
    """Memories that occurred after this decision, inside the window.

    Returns the candidates and whether the entity filter was `applied` or
    `unavailable`. **Those are different outcomes and the caller must not
    collapse them.** A corpus where nothing has been extracted cannot fail the
    entity test; it cannot take it. Treating that as "no overlap" would return
    nothing and look like a corpus in which nothing is connected; treating it as
    "overlap" would silently drop the constraint and admit every memory in the
    window. Recording which happened is the only honest option, and the queue
    shows it beside every candidate.

    The range query is M4.0's, unchanged, which is the point: `occurred_at`,
    nulls excluded rather than defaulted, half-open so a memory on a boundary
    belongs to one window rather than two. The one thing added here is a strict
    `>`, because `memories_in_range` is closed at the start and a memory
    occurring at the same instant as the decision is not after it.
    """
    end = decision.decided_at + timedelta(days=window_days)
    in_window = await temporal.memories_in_range(sessions, decision.decided_at, end)

    decision_entities = await resolved_entities(sessions, decision.evidence_memory_ids)
    entity_filter = "applied" if decision_entities else "unavailable"
    # `Memory` carries `source_id`; the durable key needs the name. Looked up
    # once for the whole window rather than per candidate — there are a handful
    # of sources and one query says so.
    names = await _source_names(sessions)

    kept: list[Candidate] = []
    for memory in in_window:
        if memory.occurred_at is None or memory.occurred_at <= decision.decided_at:
            # The `==` case `memories_in_range` admits. Simultaneous is not
            # afterwards, and a zero gap would be a causal claim about nothing.
            continue
        if memory.id in decision.evidence_memory_ids:
            # A memory that informed the decision cannot be its outcome. It is
            # in the window by construction whenever its mtime happens to fall
            # there, and admitting it would make every decision look as though
            # its own evidence had proved it right.
            continue
        if not memory.content or not memory.content.strip():
            continue

        shared: tuple[str, ...] = ()
        if decision_entities:
            candidate_entities = await resolved_entities(sessions, (memory.id,))
            overlap = set(decision_entities) & set(candidate_entities)
            if not overlap:
                continue
            shared = tuple(sorted(decision_entities[key] for key in overlap))

        gap = (memory.occurred_at - decision.decided_at).total_seconds() / 86400.0
        kept.append(
            Candidate(
                memory_id=memory.id,
                source_name=names.get(memory.source_id, ""),
                external_key=memory.external_key,
                occurred_at=memory.occurred_at,
                occurred_at_source=memory.occurred_at_source,
                text=memory.content,
                gap_days=gap,
                shared_entities=shared,
            )
        )

    # Closest first. A candidate three days after a decision is a better
    # causal claim than one three months after it, and a `--limit` run should
    # spend its calls on the strongest ones rather than on an arbitrary slice.
    kept.sort(key=lambda candidate: candidate.gap_days)
    return kept[:limit], entity_filter


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SuggestReport:
    decisions: int = 0
    # Reported separately because they are three different states of the same
    # corpus and only one of them is about the mechanism.
    decisions_without_candidates: int = 0
    decisions_without_entity_coverage: int = 0
    candidates: int = 0
    calls: int = 0
    judged_yes: int = 0
    judged_no: int = 0
    judged_unsure: int = 0
    below_confidence: int = 0
    unparseable: int = 0
    proposed: int = 0
    duplicates: int = 0
    windows: list[tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "decisions": self.decisions,
            "decisions_without_candidates": self.decisions_without_candidates,
            "decisions_without_entity_coverage": self.decisions_without_entity_coverage,
            "candidates": self.candidates,
            "calls": self.calls,
            "judged_yes": self.judged_yes,
            "judged_no": self.judged_no,
            "judged_unsure": self.judged_unsure,
            "below_confidence": self.below_confidence,
            "unparseable": self.unparseable,
            "proposed": self.proposed,
            "duplicates": self.duplicates,
        }


class SuggestOutcomes:
    """Propose outcomes into a queue. Never writes a `decision_outcomes` row.

    Stated in the class rather than only in the module for the reason M5.0's
    `SuggestDecisions` states it: a future caller looking for "the thing that
    finds outcomes in the corpus" must find something that produces
    *suggestions*, or the safety property is one refactor from gone.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        model: LanguageModel,
        *,
        max_tokens: int = 1024,
        min_confidence: float = MIN_JUDGE_CONFIDENCE,
    ) -> None:
        self._sessions = session_factory
        self._model = model
        self._max_tokens = max_tokens
        self._min_confidence = min_confidence

    @property
    def version(self) -> str:
        return f"outcome-{PROMPT_VERSION}:{self._model.model_id}"

    async def __call__(
        self,
        *,
        decision_id: UUID | None = None,
        window_days: float | None = None,
        limit: int = 10,
    ) -> SuggestReport:
        report = SuggestReport()
        decisions = await open_decisions(self._sessions, decision_id=decision_id)
        report.decisions = len(decisions)

        for decision in decisions:
            window = window_for(decision.confidence, override=window_days)
            report.windows.append((decision.question, window))
            candidates, entity_filter = await find_candidates(
                self._sessions, decision, window_days=window, limit=limit
            )
            if entity_filter == "unavailable":
                report.decisions_without_entity_coverage += 1
            if not candidates:
                report.decisions_without_candidates += 1
                continue
            report.candidates += len(candidates)

            for candidate in candidates:
                draft = await self._judge(decision, candidate, report)
                if draft is None:
                    continue
                if await self._queue(
                    decision, candidate, draft, entity_filter, window, report
                ):
                    report.proposed += 1

        logger.info("outcomes.suggested", **report.as_dict())
        return report

    async def _judge(
        self, decision: DecisionContext, candidate: Candidate, report: SuggestReport
    ) -> OutcomeDraft | None:
        """Ask whether this really is an outcome, and believe a "no".

        The three answers are not collapsed into a score. A model that must
        choose between yes, no and unsure will use unsure; one asked for a
        probability will produce 0.6 for everything and let the threshold decide,
        which moves the judgement from the model to a constant.
        """
        user = _render(decision, candidate)
        parsed: _Judgement | None = None
        for attempt in range(2):
            system = _SYSTEM if attempt == 0 else _SYSTEM + _RETRY_REMINDER
            report.calls += 1
            raw = await self._with_backoff(system, user)
            parsed = _parse(raw)
            if parsed is not None:
                break
        if parsed is None:
            report.unparseable += 1
            return None

        answer = parsed.answer.strip().lower()
        if answer == "unsure":
            report.judged_unsure += 1
            return None
        if answer != "yes":
            report.judged_no += 1
            return None
        report.judged_yes += 1

        if parsed.confidence < self._min_confidence:
            # A "yes" the model does not stand behind. Dropped rather than
            # queued with a low score: a reviewer shown a weak candidate among
            # strong ones learns to skim, and skimming is what the queue exists
            # to prevent.
            report.below_confidence += 1
            return None

        description = (parsed.description or "").strip()
        if not description:
            report.below_confidence += 1
            return None
        try:
            verdict = OutcomeVerdict((parsed.verdict or "").strip().lower())
        except ValueError:
            # A verdict outside the vocabulary is a model that has stopped
            # following the schema, which is not something to coerce.
            report.judged_no += 1
            return None

        return OutcomeDraft(
            description=description,
            verdict=verdict,
            rationale=(parsed.rationale or "").strip() or None,
            judged_confidence=parsed.confidence,
        )

    async def _with_backoff(self, system: str, user: str) -> str:
        """One candidate, waiting out a rate limit rather than losing the run."""
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return await self._model.complete(
                    system, user, max_tokens=self._max_tokens
                )
            except TransientError as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise
                delay = wait_for(exc, attempt)
                logger.info("outcomes.rate_limited", waiting_seconds=round(delay))
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def _queue(
        self,
        decision: DecisionContext,
        candidate: Candidate,
        draft: OutcomeDraft,
        entity_filter: str,
        window: float,
        report: SuggestReport,
    ) -> bool:
        stmt = (
            insert(models.OutcomeSuggestion)
            .values(
                id=new_id(),
                decision_id=decision.id,
                draft=draft.as_dict(),
                # The whole document rather than a chunk. The claim is that this
                # *item* is an outcome, and a reviewer needs to see enough of it
                # to disagree; a span would show the sentence the model liked.
                source_text=candidate.text[:4000],
                source_name=candidate.source_name,
                external_key=candidate.external_key,
                chunk_ordinal=None,
                memory_id=candidate.memory_id,
                chunk_id=None,
                candidate_occurred_at=candidate.occurred_at,
                gap_days=candidate.gap_days,
                window_days=window,
                shared_entities=list(candidate.shared_entities),
                entity_filter=entity_filter,
                status=SuggestionStatus.PENDING.value,
                model_id=self._model.model_id,
                suggester_version=self.version,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    "decision_id",
                    "source_name",
                    "external_key",
                    "chunk_ordinal",
                ],
                index_where=text("status = 'pending'"),
            )
            .returning(models.OutcomeSuggestion.id)
        )
        async with self._sessions.begin() as session:
            inserted = (await session.execute(stmt)).scalar_one_or_none()
        if inserted is None:
            report.duplicates += 1
            return False
        return True


def _render(decision: DecisionContext, candidate: Candidate) -> str:
    """The decision, the gap, and the document. In that order, deliberately.

    The gap is stated in the prompt as well as shown in the UI, because a model
    asked to judge a consequence over an unstated interval will assume a
    convenient one — and "four days later" and "four months later" support very
    different readings of the same two documents.
    """
    expected = decision.expected_outcome or "(not recorded)"
    return (
        f"DECISION\n"
        f"  question: {decision.question}\n"
        f"  chosen:   {decision.chosen}\n"
        f"  expected: {expected}\n"
        f"  decided:  {decision.decided_at.date().isoformat()}\n\n"
        f"DOCUMENT, {candidate.gap_days:.1f} days later "
        f"({candidate.external_key})\n"
        f"<<<DOCUMENT>>>\n{candidate.text[:6000]}\n<<<END DOCUMENT>>>"
    )


def _parse(raw: str) -> _Judgement | None:
    text_body = _FENCE.sub("", raw).strip()
    if not text_body:
        return None
    start, end = text_body.find("{"), text_body.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload: Any = json.loads(text_body[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        return _Judgement.model_validate(payload)
    except ValidationError:
        return None


# --------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuggestionRow:
    id: UUID
    decision_id: UUID
    decision_question: str
    decision_decided_at: datetime
    draft: OutcomeDraft
    source_text: str
    source_name: str
    external_key: str
    candidate_occurred_at: datetime
    gap_days: float
    window_days: float
    shared_entities: list[str]
    entity_filter: str
    status: SuggestionStatus
    model_id: str
    suggested_at: datetime
    reviewed_at: datetime | None
    outcome_id: UUID | None


async def list_suggestions(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: SuggestionStatus | None = SuggestionStatus.PENDING,
    limit: int = 50,
) -> list[SuggestionRow]:
    """The review queue, joined to the decision each candidate is about.

    Joined rather than fetched per row, because the queue is unreadable without
    it: "is this an outcome" is not a question anybody can answer without the
    decision on screen beside the candidate.
    """
    stmt = (
        select(models.OutcomeSuggestion, models.Decision)
        .join(models.Decision, models.Decision.id == models.OutcomeSuggestion.decision_id)
        # Closest gap first: the strongest causal claims at the top, so a
        # reviewer working down the list is spending attention in the right
        # order.
        .order_by(models.OutcomeSuggestion.gap_days)
        .limit(limit)
    )
    if status is not None:
        stmt = stmt.where(models.OutcomeSuggestion.status == status.value)

    async with session_factory() as session:
        rows = list(await session.execute(stmt))

    return [
        SuggestionRow(
            id=row[0].id,
            decision_id=row[0].decision_id,
            decision_question=row[1].question,
            decision_decided_at=row[1].decided_at,
            draft=OutcomeDraft.from_dict(row[0].draft),
            source_text=row[0].source_text,
            source_name=row[0].source_name,
            external_key=row[0].external_key,
            candidate_occurred_at=row[0].candidate_occurred_at,
            gap_days=row[0].gap_days,
            window_days=row[0].window_days,
            shared_entities=list(row[0].shared_entities),
            entity_filter=row[0].entity_filter,
            status=SuggestionStatus(row[0].status),
            model_id=row[0].model_id,
            suggested_at=row[0].suggested_at,
            reviewed_at=row[0].reviewed_at,
            outcome_id=row[0].outcome_id,
        )
        for row in rows
    ]


class AlreadyReviewed(ValueError):
    """A suggestion that has already been accepted or rejected."""


async def accept(
    session_factory: async_sessionmaker[AsyncSession],
    suggestion_id: UUID,
    *,
    edited: OutcomeDraft | None = None,
) -> UUID:
    """Write the outcome, keeping the candidate as evidence.

    **The result is `inferred`, whoever accepted it, and its confidence is the
    model's rather than 1.0.** Accepting means "this reading is worth keeping",
    not "I watched this happen" — and an accepted suggestion promoted to
    `declared` would be indistinguishable from testimony to M5.3, which is the
    one thing `evidence_kind` exists to prevent. A reviewer who *did* observe the
    outcome should record it with `memoryos outcome`, which is the declared path.

    `observed_at` is the candidate memory's own `occurred_at`, carrying that
    memory's provenance — which for this corpus means a filesystem mtime, marked
    `~` everywhere it is shown. Not the review time: the outcome happened when
    the document says it did, and stamping it now would date every inferred
    outcome to the afternoon somebody cleared the queue.
    """
    async with session_factory() as session:
        row = await session.get(models.OutcomeSuggestion, suggestion_id)
        if row is None:
            raise UnknownDecision(f"no outcome suggestion {suggestion_id}")
        if row.status != SuggestionStatus.PENDING.value:
            raise AlreadyReviewed(f"suggestion {suggestion_id} was already {row.status}")
        draft = edited or OutcomeDraft.from_dict(row.draft)
        decision_id = row.decision_id
        source_name = row.source_name
        external_key = row.external_key
        observed_at = row.candidate_occurred_at
        stated = draft.judged_confidence or 0.0
        provenance = await _provenance_of(session, source_name, external_key)

    outcome_id = await record(
        session_factory,
        decision_id,
        OutcomeDraft(
            description=draft.description,
            verdict=draft.verdict,
            rationale=draft.rationale,
            evidence=(
                OutcomeEvidenceInput(
                    source_name=source_name, external_key=external_key
                ),
            ),
        ),
        observed_at=observed_at,
        observed_at_source=provenance,
        evidence_kind=EvidenceKind.INFERRED,
        # Never 1.0: the schema forbids it and so does the meaning. A stated
        # confidence of zero — a model that filled nothing in — becomes None
        # rather than a claim of impossibility.
        confidence=min(stated, 0.99) if stated > 0 else None,
    )

    async with session_factory.begin() as session:
        queued = await session.get(models.OutcomeSuggestion, suggestion_id)
        if queued is not None:
            queued.status = SuggestionStatus.ACCEPTED.value
            queued.reviewed_at = datetime.now(UTC)
            queued.outcome_id = outcome_id

    logger.info(
        "outcome.suggestion_accepted",
        suggestion_id=str(suggestion_id),
        outcome_id=str(outcome_id),
    )
    return outcome_id


async def _provenance_of(
    session: AsyncSession, source_name: str, external_key: str
) -> TimeProvenance:
    """The candidate memory's own date provenance, carried onto the outcome.

    An mtime stays an mtime all the way through, which is what makes the `~` on
    the outcome's date honest rather than decorative. Falls back to `inferred`
    when the memory has gone — the outcome's date is then a number this system
    wrote down, which is exactly what `inferred` means.
    """
    found = (
        await session.execute(
            select(models.Memory.occurred_at_source)
            .join(models.Source, models.Source.id == models.Memory.source_id)
            .where(
                models.Source.name == source_name,
                models.Memory.external_key == external_key,
                models.Memory.is_current.is_(True),
            )
        )
    ).scalar_one_or_none()
    if found is None or found == TimeProvenance.UNKNOWN.value:
        return TimeProvenance.INFERRED
    return TimeProvenance(found)


async def reject(
    session_factory: async_sessionmaker[AsyncSession], suggestion_id: UUID
) -> None:
    """Mark a candidate as not an outcome. The row stays.

    Kept for the two reasons M5.0's rejections are: the pair is then excluded
    from the next run, and the count of rejections is the only measurement of
    how often the temporal-plus-entity filter admits a coincidence.
    """
    async with session_factory.begin() as session:
        row = await session.get(models.OutcomeSuggestion, suggestion_id)
        if row is None:
            raise UnknownDecision(f"no outcome suggestion {suggestion_id}")
        if row.status != SuggestionStatus.PENDING.value:
            raise AlreadyReviewed(f"suggestion {suggestion_id} was already {row.status}")
        row.status = SuggestionStatus.REJECTED.value
        row.reviewed_at = datetime.now(UTC)
    logger.info("outcome.suggestion_rejected", suggestion_id=str(suggestion_id))


__all__ = [
    "DEFAULT_WINDOW_DAYS",
    "MAX_WINDOW_DAYS",
    "MIN_JUDGE_CONFIDENCE",
    "MIN_WINDOW_DAYS",
    "PROMPT_VERSION",
    "AlreadyReviewed",
    "Candidate",
    "DecisionContext",
    "InvalidOutcome",
    "SuggestOutcomes",
    "SuggestReport",
    "SuggestionRow",
    "accept",
    "find_candidates",
    "list_suggestions",
    "open_decisions",
    "reject",
    "resolved_entities",
    "window_for",
]
