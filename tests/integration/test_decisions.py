"""The four claims M5.0 makes, each about a way it could be quietly wrong.

Every one of them is about something that would produce a database nobody
noticed was corrupt:

**A decision with no alternatives is rejected.** Without the rule, the table
fills with statements of what happened. Nothing errors, `decisions list` looks
healthy, and M5.1 measures outcomes against records that contain no
counterfactual — so every decision "worked", because there was never another
answer to compare it against.

**Suggestions land in review, never in the table.** A fabricated decision record
is the worst thing that can happen to Phase 5, because it produces a behavioural
claim in M5.3 that sounds insightful and is unfalsifiable.

**A replay leaves decisions untouched.** The same guarantee `query_judgements`
has, on the table the whole phase reads.

**Deleting a memory takes its evidence and leaves the decision.** A decision
survives losing a piece of its evidence, and the link it loses is not left
dangling.
"""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.decision_suggest import (
    SuggestDecisions,
    accept,
    list_suggestions,
    reject,
)
from memoryos.application.decisions import (
    AssumptionInput,
    DecisionDraft,
    DecisionEdit,
    EvidenceInput,
    InvalidDecision,
    OptionInput,
    edit,
    link_evidence,
    list_decisions,
    record,
    show,
)
from memoryos.domain.values import (
    DecisionStatus,
    EvidenceRelation,
    SuggestionStatus,
    TimeProvenance,
)
from tests.integration.conftest import Harness
from tests.support.fakes import FakeLanguageModel

pytestmark = pytest.mark.integration

DECIDED_AT = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)


def draft(**overrides: object) -> DecisionDraft:
    fields: dict[str, object] = {
        "question": "What runs background work?",
        "chosen": "A Postgres table",
        "reasoning": "The enqueue and the row it refers to are one transaction.",
        "confidence": 0.9,
        "expected_outcome": "Throughput is never the binding constraint.",
        "options": (
            OptionInput(
                description="Celery with Redis",
                rejected_because="Cannot enlist in the Postgres transaction.",
            ),
        ),
        "assumptions": (
            AssumptionInput(statement="Throughput stays in the low thousands", confidence=0.9),
        ),
    }
    fields.update(overrides)
    return DecisionDraft(**fields)  # type: ignore[arg-type]


async def count(
    sessions: async_sessionmaker[AsyncSession], model: type[models.Base]
) -> int:
    async with sessions() as session:
        return int(
            (await session.execute(select(func.count()).select_from(model))).scalar_one()
        )


async def write(
    sessions: async_sessionmaker[AsyncSession], **overrides: object
) -> UUID:
    return await record(
        sessions,
        draft(**overrides),
        decided_at=DECIDED_AT,
        decided_at_source=TimeProvenance.DECLARED,
    )


# --------------------------------------------------------------------------
# 1. A decision without alternatives is a description
# --------------------------------------------------------------------------


async def test_a_decision_with_no_options_is_rejected(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The rule this milestone has an opinion about.

    "We used Postgres" is a statement in the present tense: there is no
    counterfactual in it, so no outcome can ever say whether it was right.
    Refusing it at capture time is the only place the rule can be enforced —
    afterwards there is nothing to distinguish a decision whose alternatives
    were never recorded from one that genuinely had none.
    """
    with pytest.raises(InvalidDecision, match="no alternatives"):
        await write(sessions, options=())

    assert await count(sessions, models.Decision) == 0
    # And nothing half-written: the options and the decision are one transaction.
    assert await count(sessions, models.DecisionOption) == 0
    assert await count(sessions, models.DecisionAssumption) == 0


async def test_an_option_equal_to_the_choice_is_not_an_alternative(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The obvious way around the rule, closed.

    Passing the chosen thing back as its own option would satisfy a naive count
    and record nothing. It is the winner, so it does not count towards the
    alternatives — and it is written once, from `chosen`, rather than twice.
    """
    with pytest.raises(InvalidDecision, match="no alternatives"):
        await write(
            sessions, options=(OptionInput(description="a postgres table"),)
        )


async def test_the_chosen_option_is_written_from_the_decision(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    # One winner, derived rather than supplied, so the text in `chosen` and the
    # row flagged `was_chosen` cannot disagree.
    decision_id = await write(sessions)
    detail = await show(sessions, decision_id)

    chosen = [option for option in detail.options if option.was_chosen]
    assert len(chosen) == 1
    assert chosen[0].description == "A Postgres table"
    assert chosen[0].rejected_because is None
    assert [option.description for option in detail.rejected] == ["Celery with Redis"]


async def test_an_edit_cannot_remove_the_last_alternative(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    # The rule has to survive editing, or it is a rule about the first write.
    decision_id = await write(sessions)
    with pytest.raises(InvalidDecision, match="last alternative"):
        await edit(sessions, decision_id, DecisionEdit(options=()))


async def test_an_unevaluated_assumption_is_not_a_broken_one(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """`held` is null until M5.2, and null is a third state.

    A system that could not tell an unevaluated assumption from a broken one
    would report every new decision as built on sand.
    """
    decision_id = await write(sessions)
    (assumption,) = (await show(sessions, decision_id)).assumptions
    assert assumption.held is None
    assert assumption.evaluated_at is None


# --------------------------------------------------------------------------
# 2. Suggestions land in review, never in the table
# --------------------------------------------------------------------------

SUGGESTED = """
{"decisions": [{"question": "How are two retrievers combined?",
  "chosen": "Reciprocal rank fusion",
  "reasoning": "The two score scales are not comparable.",
  "confidence": null, "expected_outcome": null,
  "options": [{"description": "A weighted sum", "rejected_because": "Not comparable"}],
  "assumptions": []}]}
"""

# A passage `find_passages` will actually offer to the model: long enough to hold
# both halves of a decision, and carrying a comparative construction. The
# pre-filter is lexical rather than a classifier, so a fixture that did not match
# it would be testing the model's behaviour on an empty candidate list.
DECISION_TEXT = (
    "Both retrievers run concurrently and their two rankings are fused by "
    "reciprocal rank rather than by a weighted sum of the raw scores. The two "
    "numbers are not on comparable scales: cosine from this model occupies a "
    "narrow band where almost all the range carries no signal, and the lexical "
    "score is unbounded and depends on term frequencies across the whole "
    "corpus, so the same document scores differently after ingesting unrelated "
    "files. Every normalisation that would make them comparable encodes an "
    "assumption about the score distribution that stops holding as the corpus "
    "grows, and it fails silently. Fusion keeps the ordering, which is the part "
    "both retrievers mean the same thing by. "
)


@pytest.fixture
async def corpus_with_a_decision(harness: Harness) -> Harness:
    """The harness corpus plus one file that records a decision.

    Written into the same source and ingested through the real pipeline rather
    than inserted, so the memory is in the log and a replay recreates it. Local
    to this file: adding it to the shared harness would change the row counts
    every replay test compares against.
    """
    (harness.root / "fusion.md").write_text("# Fusion\n\n" + DECISION_TEXT * 2 + "\n")
    await harness.ingest()
    return harness


async def test_a_suggestion_lands_in_review_and_is_never_auto_committed(
    corpus_with_a_decision: Harness,
) -> None:
    """The safety property the whole extraction path rests on.

    A model asked to find decisions in explanatory prose will find them, because
    prose that explains a choice is shaped exactly like a record of one. What it
    cannot know is the confidence somebody held or what they were assuming, and
    asked for those anyway it produces them — fluently. So the pass writes to the
    queue and a person writes to the table.
    """
    harness = corpus_with_a_decision
    model = FakeLanguageModel(SUGGESTED)
    suggest = SuggestDecisions(harness.sessions, model)

    report = await suggest(limit=5)

    assert report.proposed >= 1
    assert await count(harness.sessions, models.DecisionSuggestion) == report.proposed
    # The table Phase 5 reads is still empty. Nothing was committed.
    assert await count(harness.sessions, models.Decision) == 0
    assert await list_decisions(harness.sessions) == []

    queued = await list_suggestions(harness.sessions)
    assert all(row.status is SuggestionStatus.PENDING for row in queued)
    # Every draft carries the passage it came from, which is what makes the
    # review a judgement about evidence rather than about plausibility.
    assert all(row.source_text for row in queued)
    assert all(row.external_key for row in queued)


async def test_a_draft_with_no_alternative_never_reaches_the_queue(
    corpus_with_a_decision: Harness,
) -> None:
    """The rule applies one step earlier than accept.

    A queue holding drafts that cannot be accepted teaches a reviewer to click
    through them, and the whole value of the queue is that accepting is a
    considered act.
    """
    harness = corpus_with_a_decision
    model = FakeLanguageModel(
        '{"decisions": [{"question": "Which database?", "chosen": "Postgres", '
        '"options": [], "assumptions": []}]}'
    )
    report = await SuggestDecisions(harness.sessions, model)(limit=3)

    assert report.proposed == 0
    assert report.rejected_no_alternatives >= 1
    assert await count(harness.sessions, models.DecisionSuggestion) == 0


async def test_accepting_writes_a_decision_and_keeps_the_passage_as_evidence(
    corpus_with_a_decision: Harness,
) -> None:
    """Accept is the only path from the queue into the table.

    The relation is `records`, not `informed`: the passage describes the
    decision rather than having fed it. M5.1 needs that ordering, and a
    suggestion pass that marked its own source as an input would make every
    extracted decision look as though it had been argued for in advance.
    """
    harness = corpus_with_a_decision
    await SuggestDecisions(harness.sessions, FakeLanguageModel(SUGGESTED))(limit=5)
    (queued,) = (await list_suggestions(harness.sessions))[:1]

    decision_id = await accept(harness.sessions, queued.id)

    detail = await show(harness.sessions, decision_id)
    assert detail.question == "How are two retrievers combined?"
    assert [item.relation for item in detail.evidence] == [EvidenceRelation.RECORDS]
    # The fields the model was told to leave alone are still empty, rather than
    # filled with something plausible.
    assert detail.confidence is None
    assert detail.assumptions == []

    async with harness.sessions() as session:
        row = await session.get(models.DecisionSuggestion, queued.id)
        assert row is not None
        assert row.status == SuggestionStatus.ACCEPTED.value
        assert row.decision_id == decision_id


async def test_a_rejected_suggestion_is_kept_and_not_proposed_again(
    corpus_with_a_decision: Harness,
) -> None:
    """Rejections are the only measurement of what the extractor gets wrong.

    Deleted rows would also mean the same passage came back on the next run
    looking like a new finding.
    """
    harness = corpus_with_a_decision
    await SuggestDecisions(harness.sessions, FakeLanguageModel(SUGGESTED))(limit=5)
    (queued,) = (await list_suggestions(harness.sessions))[:1]

    await reject(harness.sessions, queued.id)

    assert await list_suggestions(harness.sessions, status=SuggestionStatus.PENDING) == []
    kept = await list_suggestions(harness.sessions, status=SuggestionStatus.REJECTED)
    assert [row.id for row in kept] == [queued.id]

    report = await SuggestDecisions(harness.sessions, FakeLanguageModel(SUGGESTED))(limit=5)
    assert queued.external_key not in {
        row.external_key
        for row in await list_suggestions(harness.sessions, status=SuggestionStatus.PENDING)
    } or report.proposed == 0


# --------------------------------------------------------------------------
# 3. A replay leaves decisions untouched
# --------------------------------------------------------------------------


async def test_a_full_replay_leaves_decisions_untouched(harness: Harness) -> None:
    """The `USER_AUTHORED` guarantee, on the table the whole phase reads.

    Every memory id changes during a rebuild. The decision must not — nothing in
    the log produces somebody's account of a choice they made, so a replay that
    truncated this table would destroy the corpus Phase 5 operates on and no
    count anywhere would report it.
    """
    decision_id = await write(harness.sessions)
    before = await show(harness.sessions, decision_id)

    report = await harness.replay(clear_cache=False)
    assert report.memories > 0, "the fixture did not actually rebuild"

    after = await show(harness.sessions, decision_id)
    assert after.question == before.question
    assert after.chosen == before.chosen
    # Confidence in particular: it is the number M5.2 measures calibration
    # against, and a rebuild that quietly moved it would make that measurement a
    # measurement of the rebuild.
    assert after.confidence == before.confidence
    assert after.decided_at == before.decided_at
    assert [option.description for option in after.options] == [
        option.description for option in before.options
    ]
    assert [item.statement for item in after.assumptions] == [
        item.statement for item in before.assumptions
    ]


async def test_a_replay_relinks_evidence_by_natural_key(harness: Harness) -> None:
    """The half a classification could not deliver.

    `decision_evidence` holds ON DELETE CASCADE foreign keys into `memories`, so
    `TRUNCATE memories CASCADE` takes it whatever set it is classified in. The
    row does not survive; the link does — replay reads it out by
    `(source_name, external_key, chunk_ordinal)` and writes it back against the
    rebuilt corpus, which is why the ids on the other side are new.
    """
    decision_id = await write(harness.sessions)
    await link_evidence(
        harness.sessions,
        decision_id,
        EvidenceInput(source_name="corpus", external_key="queue.md"),
    )
    before = (await show(harness.sessions, decision_id)).evidence
    assert len(before) == 1

    report = await harness.replay(clear_cache=False)
    assert report.evidence_preserved == 1
    assert report.evidence_relinked == 1

    after = (await show(harness.sessions, decision_id)).evidence
    assert len(after) == 1
    assert after[0].external_key == "queue.md"
    assert after[0].relation is EvidenceRelation.INFORMED
    # A new row pointing at a new memory. The link is what survived, not the id.
    assert after[0].memory_id != before[0].memory_id


async def test_a_shadow_replay_leaves_decisions_and_their_evidence(
    harness: Harness,
) -> None:
    """The swap drops the live `memories` table, which the evidence FK blocks.

    `PostgresShadowSchema.swap_in` lifts the inbound constraints off and puts
    them back rather than dropping them with CASCADE, which would make the error
    go away and leave the schema no longer matching the models. This is the test
    that fails if somebody reaches for CASCADE.
    """
    decision_id = await write(harness.sessions)
    await link_evidence(
        harness.sessions,
        decision_id,
        EvidenceInput(
            source_name="corpus",
            external_key="queue.md",
            relation=EvidenceRelation.RECORDS,
        ),
    )

    await harness.replay(into_shadow=True)

    detail = await show(harness.sessions, decision_id)
    assert detail.question == "What runs background work?"
    assert [item.relation for item in detail.evidence] == [EvidenceRelation.RECORDS]

    # And the constraint is back, not quietly dropped: an evidence row naming a
    # memory that does not exist must still be refused.
    async with harness.sessions.begin() as session:
        with pytest.raises(IntegrityError, match=r"foreign key"):
            session.add(
                models.DecisionEvidence(
                    id=UUID("11111111-1111-7111-8111-111111111111"),
                    decision_id=decision_id,
                    memory_id=UUID("22222222-2222-7222-8222-222222222222"),
                    chunk_id=None,
                    source_name="corpus",
                    external_key="gone.md",
                    chunk_ordinal=None,
                    relation=EvidenceRelation.INFORMED.value,
                )
            )
            await session.flush()


async def test_a_suggestion_survives_a_replay_and_keeps_its_passage(
    corpus_with_a_decision: Harness,
) -> None:
    """The queue is user-authored too: it carries somebody's accept or reject.

    Its provenance is a natural key plus id snapshots and there is no foreign
    key, which is exactly what lets it be classified by argument rather than
    forced into the derived set by a constraint.
    """
    harness = corpus_with_a_decision
    await SuggestDecisions(harness.sessions, FakeLanguageModel(SUGGESTED))(limit=5)
    before = await list_suggestions(harness.sessions)
    assert before

    await harness.replay(clear_cache=False)

    after = await list_suggestions(harness.sessions)
    assert [row.id for row in after] == [row.id for row in before]
    assert [row.source_text for row in after] == [row.source_text for row in before]


# --------------------------------------------------------------------------
# 4. A decision survives losing a piece of its evidence
# --------------------------------------------------------------------------


async def test_deleting_a_memory_cascades_to_evidence_and_leaves_the_decision(
    harness: Harness,
) -> None:
    """The property the schema chose a cascade for.

    A link to a document that no longer exists is a citation to nothing, and
    M2.5 spent a milestone making sure a citation always resolves. So the
    evidence row goes with the memory — and the decision, its options and its
    assumptions do not, because a decision is not made false by losing a piece
    of the evidence for it.
    """
    decision_id = await write(harness.sessions)
    await link_evidence(
        harness.sessions,
        decision_id,
        EvidenceInput(source_name="corpus", external_key="queue.md"),
    )
    await link_evidence(
        harness.sessions,
        decision_id,
        EvidenceInput(source_name="corpus", external_key="bread.txt"),
    )
    assert await count(harness.sessions, models.DecisionEvidence) == 2

    async with harness.sessions.begin() as session:
        await session.execute(
            delete(models.Memory).where(models.Memory.external_key == "queue.md")
        )

    detail = await show(harness.sessions, decision_id)
    assert detail.question == "What runs background work?"
    assert len(detail.options) == 2
    assert len(detail.assumptions) == 1
    # One link gone, one left. Not a dangling row, and not a decision deleted
    # along with its source.
    assert [item.external_key for item in detail.evidence] == ["bread.txt"]


async def test_a_chunk_level_link_carries_the_ordinal_that_survives_a_rebuild(
    harness: Harness,
) -> None:
    # `chunk_id` is minted per write; the ordinal is deterministic. A link
    # missing the ordinal would silently widen to the whole memory after a
    # replay, which is a citation moving without saying so.
    decision_id = await write(harness.sessions)
    await link_evidence(
        harness.sessions,
        decision_id,
        EvidenceInput(source_name="corpus", external_key="queue.md", chunk_ordinal=0),
    )

    (item,) = (await show(harness.sessions, decision_id)).evidence
    assert item.chunk_ordinal == 0
    assert item.chunk_id is not None

    await harness.replay(clear_cache=False)

    (rebuilt,) = (await show(harness.sessions, decision_id)).evidence
    assert rebuilt.chunk_ordinal == 0
    assert rebuilt.chunk_id is not None
    assert rebuilt.chunk_id != item.chunk_id


async def test_a_decision_lists_with_the_counts_that_say_how_complete_it_is(
    harness: Harness,
) -> None:
    # Three correlated subqueries rather than three joins: joining three
    # one-to-many tables multiplies their rows, and every count would come back
    # as the product of the other two.
    decision_id = await write(harness.sessions)
    await link_evidence(
        harness.sessions,
        decision_id,
        EvidenceInput(source_name="corpus", external_key="queue.md"),
    )

    (summary,) = await list_decisions(harness.sessions)
    assert summary.options == 2
    assert summary.assumptions == 1
    assert summary.evidence == 1
    assert summary.status is DecisionStatus.OPEN
