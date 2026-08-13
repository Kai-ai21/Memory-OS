"""Proposing decision records from the corpus, into a queue nobody skips.

The assistive half of M5.0, and the half that has to be kept on a short leash.

**Why a queue and not a write.** A language model asked to find decisions in a
corpus of explanatory prose will find them, because prose that explains a choice
is shaped exactly like a record of one. What it cannot find is the half that
makes a decision record worth having: the confidence somebody held at the time,
what they expected to happen, and what they were assuming. Asked for those
anyway, a model produces them — fluent, specific, and invented. That row then
becomes a pattern in M5.3 and a reflection in M5.4, and the resulting claim
about how somebody makes decisions is both plausible and unfalsifiable, because
the evidence for it is a sentence a model wrote.

So this module writes to `decision_suggestions` and never to `decisions`.
Accepting is a separate act, performed by a person looking at the passage beside
the draft. The prompt is written to make that judgement easy rather than to make
the drafts look good: it is told to leave `confidence`, `expected_outcome` and
`assumptions` empty unless the passage states them, and every draft carries the
memory and the chunk it came from.

**What it is allowed to fill in.** The question, the choice, the alternatives,
and the reasoning — all four of which a passage can genuinely contain. Anything
else is left for the reviewer. A draft with three empty fields is the correct
output for a passage that says "we used RRF rather than a weighted sum, because
the two scores are not on comparable scales", and a draft with all six filled in
from that same passage would be the failure this design exists to prevent.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.decisions import (
    AssumptionInput,
    DecisionDraft,
    EvidenceInput,
    OptionInput,
    UnknownDecision,
    record,
)
from memoryos.application.ports import LanguageModel
from memoryos.domain.ids import new_id
from memoryos.domain.values import (
    DecisionStatus,
    EvidenceRelation,
    SuggestionStatus,
    TimeProvenance,
)

logger = structlog.get_logger(__name__)

# Bump when the prompt changes in a way that could change what comes back.
# Recorded on every suggestion, the same as M1.4's chunker version and M3.1's
# extractor version, so improving the prompt is a query over the queue rather
# than a truncation of it.
PROMPT_VERSION = "v1"

# One chunk per request. Unlike entity extraction this is not run over the whole
# corpus — it is a targeted pass a person invokes with a limit — so the batching
# that milestone needed would trade the one thing that matters here, which is
# that every draft is attributable to exactly one passage.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

# Passages shorter than this cannot hold a decision and its alternatives. Mostly
# it filters import blocks and one-line helpers, which cost a request each and
# return nothing.
MIN_PASSAGE_CHARS = 320

_SYSTEM = """\
You find records of DECISIONS in text. You return JSON and nothing else.

A decision has a question, a choice, and at least one alternative that was \
considered and not taken. Text that merely describes how something works is NOT \
a decision, however detailed. "The worker holds a lease" is a description. "A \
lease rather than a visibility timeout, because a crashed worker must not hold \
the job forever" is a decision.

Rules, in order of importance:

1. If the passage does not name at least one REJECTED alternative, return no \
decision for it. A choice with no alternative is a description.
2. Use only what the passage says. Never supply a fact you know from elsewhere, \
and never smooth over a gap with something plausible.
3. Leave "confidence", "expected_outcome" and "assumptions" EMPTY unless the \
passage states them outright. These are almost never written down. An empty \
field is the correct and expected answer; a filled-in guess is a serious error.
4. Quote or closely paraphrase the passage for "reasoning" and \
"rejected_because". Do not improve the argument.

Return this JSON shape and nothing else — no prose, no markdown fences:

{"decisions": [{"question": "...", "chosen": "...", "reasoning": "...", \
"confidence": null, "expected_outcome": null, \
"options": [{"description": "...", "rejected_because": "..."}], \
"assumptions": [{"statement": "...", "confidence": null}]}]}

Return {"decisions": []} when the passage records no decision. That is a common \
and correct answer."""

_RETRY_REMINDER = """\

Your previous response was not valid JSON. Return ONLY a JSON object. Start \
your response with { and end it with }. No explanation, no markdown fences, no \
text before or after the JSON."""


class _Option(BaseModel):
    description: str
    rejected_because: str | None = None


class _Assumption(BaseModel):
    statement: str
    confidence: float | None = None


class _Decision(BaseModel):
    question: str
    chosen: str
    reasoning: str | None = None
    confidence: float | None = None
    expected_outcome: str | None = None
    options: list[_Option] = Field(default_factory=list)
    assumptions: list[_Assumption] = Field(default_factory=list)


class _Response(BaseModel):
    decisions: list[_Decision] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Passage:
    """One chunk, with the provenance a suggestion has to carry."""

    memory_id: UUID
    chunk_id: UUID
    source_name: str
    external_key: str
    ordinal: int
    text: str


async def find_passages(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    source: str | None = None,
    limit: int = 20,
) -> list[Passage]:
    """Chunks worth spending a model call on.

    A cheap lexical pre-filter rather than a pass over the corpus, and the
    saving is not the point — the honesty is. Every chunk in this corpus is
    prose about software, so a model asked "does this record a decision" over
    1,308 of them returns a great many maybes. Narrowing to passages that
    actually contain a comparative construction means a draft's *absence* says
    something: this passage looked like a decision and was not one.

    Already-queued and already-reviewed passages are excluded, so a second run
    proposes new things instead of the same things again.
    """
    reviewed = select(
        models.DecisionSuggestion.source_name,
        models.DecisionSuggestion.external_key,
        models.DecisionSuggestion.chunk_ordinal,
    ).subquery()

    stmt = (
        select(
            models.Memory.id,
            models.MemoryChunk.id,
            models.Source.name,
            models.Memory.external_key,
            models.MemoryChunk.ordinal,
            models.MemoryChunk.content,
        )
        .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
        .join(models.Source, models.Source.id == models.Memory.source_id)
        .where(
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
            func.length(models.MemoryChunk.content) >= MIN_PASSAGE_CHARS,
            # The comparative constructions a decision is written in. Not a
            # classifier — it is a way of not paying for chunks that cannot
            # possibly match.
            models.MemoryChunk.content.op("~*")(
                r"(rather than|instead of|in favou?r of|opted for|we chose|"
                r"we decided|the decision to|trade-?off|over [A-Z])"
            ),
            ~select(1)
            .select_from(reviewed)
            .where(
                reviewed.c.source_name == models.Source.name,
                reviewed.c.external_key == models.Memory.external_key,
                reviewed.c.chunk_ordinal == models.MemoryChunk.ordinal,
            )
            .exists(),
        )
        # Longest first. A longer passage is likelier to hold both halves of a
        # decision, and a first run with a small limit should see the best
        # candidates rather than an arbitrary slice.
        .order_by(func.length(models.MemoryChunk.content).desc())
        .limit(limit)
    )
    if source is not None:
        stmt = stmt.where(models.Source.name == source)

    async with session_factory() as session:
        rows = list(await session.execute(stmt))

    return [
        Passage(
            memory_id=row[0],
            chunk_id=row[1],
            source_name=row[2],
            external_key=row[3],
            ordinal=row[4],
            text=row[5],
        )
        for row in rows
    ]


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SuggestReport:
    passages: int = 0
    calls: int = 0
    proposed: int = 0
    # Drafts the model returned that this module refused to queue, and why.
    # Counted rather than logged away: the number is the extractor's own
    # false-positive rate as measured before a human ever sees the queue.
    rejected_no_alternatives: int = 0
    unparseable: int = 0
    duplicates: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "passages": self.passages,
            "calls": self.calls,
            "proposed": self.proposed,
            "rejected_no_alternatives": self.rejected_no_alternatives,
            "unparseable": self.unparseable,
            "duplicates": self.duplicates,
        }


class SuggestDecisions:
    """Read passages, propose drafts, queue them for review.

    Never writes a `decisions` row. That is the whole contract, and it is worth
    stating in the class rather than only in the module: a future caller looking
    for "the thing that creates decisions from the corpus" must find something
    that creates *suggestions*, or the safety property is one refactor from
    gone.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        model: LanguageModel,
        *,
        max_tokens: int = 1536,
    ) -> None:
        self._sessions = session_factory
        self._model = model
        self._max_tokens = max_tokens

    @property
    def version(self) -> str:
        """Prompt and model, for the reason M3.1's extractor version carries both."""
        return f"suggest-{PROMPT_VERSION}:{self._model.model_id}"

    async def __call__(
        self, *, source: str | None = None, limit: int = 20
    ) -> SuggestReport:
        report = SuggestReport()
        passages = await find_passages(self._sessions, source=source, limit=limit)
        report.passages = len(passages)

        for passage in passages:
            drafts = await self._propose(passage, report)
            for draft in drafts:
                if await self._queue(passage, draft, report):
                    report.proposed += 1

        logger.info("decisions.suggested", **report.as_dict())
        return report

    async def _propose(
        self, passage: Passage, report: SuggestReport
    ) -> list[DecisionDraft]:
        """One call, one retry on malformed JSON, then give up on this passage.

        A passage that produces unparseable output twice will produce it a third
        time, and this is a foreground command a person is watching — burning
        further calls on it delays the ones that would have worked.
        """
        user = (
            "Find any decisions recorded in the following passage. It comes from "
            f"{passage.external_key!r}.\n\n<<<PASSAGE>>>\n{passage.text}\n"
            "<<<END PASSAGE>>>"
        )
        for attempt in range(2):
            system = _SYSTEM if attempt == 0 else _SYSTEM + _RETRY_REMINDER
            report.calls += 1
            raw = await self._model.complete(system, user, max_tokens=self._max_tokens)
            parsed = _parse(raw)
            if parsed is not None:
                return self._validated(parsed, report)
        report.unparseable += 1
        logger.warning(
            "decisions.suggest_unparseable",
            external_key=passage.external_key,
            ordinal=passage.ordinal,
        )
        return []

    def _validated(
        self, response: _Response, report: SuggestReport
    ) -> list[DecisionDraft]:
        """Drafts that would survive `decisions.record`, and nothing else.

        The alternatives rule is applied here rather than only at accept time,
        and that is not belt-and-braces. A queue holding drafts that cannot be
        accepted teaches a reviewer to click through them, and the whole value of
        the queue is that clicking accept is a considered act.
        """
        kept: list[DecisionDraft] = []
        for candidate in response.decisions:
            alternatives = tuple(
                OptionInput(
                    description=option.description.strip(),
                    rejected_because=(option.rejected_because or "").strip() or None,
                )
                for option in candidate.options
                if option.description.strip()
                and option.description.strip().casefold()
                != candidate.chosen.strip().casefold()
            )
            if not alternatives or not candidate.question.strip() or not candidate.chosen.strip():
                report.rejected_no_alternatives += 1
                continue
            kept.append(
                DecisionDraft(
                    question=candidate.question.strip(),
                    chosen=candidate.chosen.strip(),
                    reasoning=(candidate.reasoning or "").strip() or None,
                    confidence=candidate.confidence,
                    expected_outcome=(candidate.expected_outcome or "").strip() or None,
                    options=alternatives,
                    assumptions=tuple(
                        AssumptionInput(
                            statement=item.statement.strip(), confidence=item.confidence
                        )
                        for item in candidate.assumptions
                        if item.statement.strip()
                    ),
                )
            )
        return kept

    async def _queue(
        self, passage: Passage, draft: DecisionDraft, report: SuggestReport
    ) -> bool:
        """Insert the draft, or do nothing if that passage is already queued.

        `ON CONFLICT DO NOTHING` against the partial unique index rather than a
        read-then-write: a second run started before the first finished would
        otherwise fail on a constraint the caller can do nothing about.
        """
        stmt = (
            insert(models.DecisionSuggestion)
            .values(
                id=new_id(),
                draft=draft.as_dict(),
                source_text=passage.text,
                source_name=passage.source_name,
                external_key=passage.external_key,
                chunk_ordinal=passage.ordinal,
                memory_id=passage.memory_id,
                chunk_id=passage.chunk_id,
                status=SuggestionStatus.PENDING.value,
                model_id=self._model.model_id,
                suggester_version=self.version,
            )
            # The predicate as well as the columns: the index is partial, and
            # without `index_where` Postgres cannot tell which constraint this
            # `ON CONFLICT` means and refuses the statement outright.
            .on_conflict_do_nothing(
                index_elements=["source_name", "external_key", "chunk_ordinal"],
                index_where=text("status = 'pending'"),
            )
            .returning(models.DecisionSuggestion.id)
        )
        async with self._sessions.begin() as session:
            inserted = (await session.execute(stmt)).scalar_one_or_none()
        if inserted is None:
            report.duplicates += 1
            return False
        return True


def _parse(raw: str) -> _Response | None:
    """The model's text as a validated response, or None if it is not one.

    Lifted from `adapters/extraction/llm.py` in shape rather than imported: that
    one validates an entity response, this one a decision response, and the only
    shared part is the fence-and-braces salvage. Sharing it would mean a module
    in `application/` importing from `adapters/`, which is the dependency rule
    this project has held since M1.0.
    """
    text = _FENCE.sub("", raw).strip()
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload: Any = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        return _Response.model_validate(payload)
    except ValidationError:
        return None


# --------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SuggestionRow:
    id: UUID
    draft: DecisionDraft
    source_text: str
    source_name: str
    external_key: str
    chunk_ordinal: int | None
    status: SuggestionStatus
    model_id: str
    suggested_at: datetime
    reviewed_at: datetime | None
    decision_id: UUID | None


async def list_suggestions(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: SuggestionStatus | None = SuggestionStatus.PENDING,
    limit: int = 50,
) -> list[SuggestionRow]:
    """The review queue. Pending by default, because that is what needs a person."""
    stmt = (
        select(models.DecisionSuggestion)
        .order_by(models.DecisionSuggestion.suggested_at)
        .limit(limit)
    )
    if status is not None:
        stmt = stmt.where(models.DecisionSuggestion.status == status.value)

    async with session_factory() as session:
        rows = list((await session.execute(stmt)).scalars())

    return [
        SuggestionRow(
            id=row.id,
            draft=DecisionDraft.from_dict(row.draft),
            source_text=row.source_text,
            source_name=row.source_name,
            external_key=row.external_key,
            chunk_ordinal=row.chunk_ordinal,
            status=SuggestionStatus(row.status),
            model_id=row.model_id,
            suggested_at=row.suggested_at,
            reviewed_at=row.reviewed_at,
            decision_id=row.decision_id,
        )
        for row in rows
    ]


class AlreadyReviewed(ValueError):
    """A suggestion that has already been accepted or rejected."""


async def accept(
    session_factory: async_sessionmaker[AsyncSession],
    suggestion_id: UUID,
    *,
    edited: DecisionDraft | None = None,
    decided_at: datetime | None = None,
) -> UUID:
    """Turn one suggestion into a decision, keeping the passage as evidence.

    `edited` is the accept-with-changes path, and it is the common one. A
    reviewer who has read the passage usually knows the confidence and at least
    one assumption the model could not have known, and forcing them to accept
    first and edit afterwards would leave a decision in the table that nobody
    stands behind, however briefly.

    The evidence relation is `RECORDS`, not `INFORMED`, and the distinction is
    load-bearing. The passage this came from is a *description of the decision*
    written afterwards; it did not inform it. M5.1 needs that ordering to say
    anything about prediction, and a suggestion pass that marked its own source
    as an input would make every extracted decision look as though it had been
    argued for in advance.

    `decided_at` defaults to the memory's `occurred_at` when the corpus knows
    one, and carries that memory's provenance with it — which for this corpus
    means `filesystem`, an mtime, and Phase 4's weighting applies. It is not
    defaulted to now: the decision was made when the passage says it was, and
    stamping the review time on it would date every extracted decision to the
    afternoon somebody cleared the queue.
    """
    async with session_factory() as session:
        row = await session.get(models.DecisionSuggestion, suggestion_id)
        if row is None:
            raise UnknownDecision(f"no suggestion {suggestion_id}")
        if row.status != SuggestionStatus.PENDING.value:
            raise AlreadyReviewed(f"suggestion {suggestion_id} was already {row.status}")
        draft = edited or DecisionDraft.from_dict(row.draft)
        source_name = row.source_name
        external_key = row.external_key
        chunk_ordinal = row.chunk_ordinal
        stamped, provenance = await _stamp(session, row, decided_at)

    decision_id = await record(
        session_factory,
        DecisionDraft(
            question=draft.question,
            chosen=draft.chosen,
            reasoning=draft.reasoning,
            confidence=draft.confidence,
            expected_outcome=draft.expected_outcome,
            options=draft.options,
            assumptions=draft.assumptions,
            evidence=(
                EvidenceInput(
                    source_name=source_name,
                    external_key=external_key,
                    relation=EvidenceRelation.RECORDS,
                    chunk_ordinal=chunk_ordinal,
                ),
            ),
        ),
        decided_at=stamped,
        decided_at_source=provenance,
        status=DecisionStatus.OPEN,
    )

    async with session_factory.begin() as session:
        queued = await session.get(models.DecisionSuggestion, suggestion_id)
        if queued is not None:
            queued.status = SuggestionStatus.ACCEPTED.value
            queued.reviewed_at = datetime.now(UTC)
            queued.decision_id = decision_id

    logger.info(
        "decision.suggestion_accepted",
        suggestion_id=str(suggestion_id),
        decision_id=str(decision_id),
    )
    return decision_id


async def _stamp(
    session: AsyncSession,
    row: models.DecisionSuggestion,
    override: datetime | None,
) -> tuple[datetime, TimeProvenance]:
    """When the decision was made, and how well that is known.

    An explicit override is `declared` — somebody typed it. Otherwise the
    memory's own `occurred_at` and its provenance, unchanged, so an mtime stays
    an mtime all the way through. Only if the corpus has neither does this fall
    back to the suggestion's own timestamp, marked `inferred`, which is the
    honest label for "the date this was noticed".
    """
    if override is not None:
        return override, TimeProvenance.DECLARED

    found = (
        await session.execute(
            select(models.Memory.occurred_at, models.Memory.occurred_at_source)
            .join(models.Source, models.Source.id == models.Memory.source_id)
            .where(
                models.Source.name == row.source_name,
                models.Memory.external_key == row.external_key,
                models.Memory.is_current.is_(True),
            )
        )
    ).first()
    if found is not None and found[0] is not None:
        return found[0], TimeProvenance(found[1])
    return row.suggested_at, TimeProvenance.INFERRED


async def reject(
    session_factory: async_sessionmaker[AsyncSession], suggestion_id: UUID
) -> None:
    """Mark a draft as not a decision. The row stays.

    Kept rather than deleted for two reasons that both matter: the passage is
    then excluded from the next `suggest` run, and the count of rejections is
    the only measurement of what the extractor gets wrong.
    """
    async with session_factory.begin() as session:
        row = await session.get(models.DecisionSuggestion, suggestion_id)
        if row is None:
            raise UnknownDecision(f"no suggestion {suggestion_id}")
        if row.status != SuggestionStatus.PENDING.value:
            raise AlreadyReviewed(f"suggestion {suggestion_id} was already {row.status}")
        row.status = SuggestionStatus.REJECTED.value
        row.reviewed_at = datetime.now(UTC)
    logger.info("decision.suggestion_rejected", suggestion_id=str(suggestion_id))


def summarise_drafts(rows: Sequence[SuggestionRow]) -> dict[str, int]:
    """What the queue holds, and how much of it the model actually filled in.

    The three "with" counts are the measurement this milestone predicted would
    be low, reported rather than argued about: a suggestion pass that filled in
    confidence and assumptions from a codebase would be fabricating them.
    """
    return {
        "pending": sum(1 for row in rows if row.status is SuggestionStatus.PENDING),
        "with_reasoning": sum(1 for row in rows if row.draft.reasoning),
        "with_confidence": sum(1 for row in rows if row.draft.confidence is not None),
        "with_assumptions": sum(1 for row in rows if row.draft.assumptions),
    }


__all__ = [
    "PROMPT_VERSION",
    "AlreadyReviewed",
    "Passage",
    "SuggestDecisions",
    "SuggestReport",
    "SuggestionRow",
    "accept",
    "find_passages",
    "list_suggestions",
    "reject",
    "summarise_drafts",
]
