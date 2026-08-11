"""Entity extraction against a real database, with a fake extractor.

The fake is right for all of this. Whether a real model finds *useful* entities
is the milestone's corpus measurement and cannot be settled by a test; what
these establish is that the pipeline stores what it is given, refuses what it
cannot find in the text, does nothing on a second run, and lets the database
clean up after a deletion.

The one place a real client is used is the malformed-JSON test, which drives the
adapter through a `FakeLanguageModel` — because the retry-then-give-up decision
lives in the adapter, above the provider, and is exactly the part a fake
extractor would skip over.
"""

import re
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.extraction.llm import LlmEntityExtractor
from memoryos.application.extraction import ExtractEntities, ExtractOutcome
from memoryos.domain.jobs import PermanentError
from memoryos.domain.values import EntityType, MemoryKind
from tests.integration.conftest import Pipeline
from tests.support.fakes import FakeEntityExtractor, FakeLanguageModel

pytestmark = pytest.mark.integration


async def one_memory(pipeline: Pipeline) -> tuple[UUID, list[tuple[UUID, str]]]:
    """Ingest the fixture corpus and return one memory with its chunks."""
    await pipeline.ingest()
    async with pipeline.sessions() as session:
        memory_id = (
            await session.execute(
                select(models.Memory.id).order_by(models.Memory.external_key).limit(1)
            )
        ).scalar_one()
        chunks = [
            (row[0], row[1])
            for row in await session.execute(
                select(models.MemoryChunk.id, models.MemoryChunk.content)
                .where(models.MemoryChunk.memory_id == memory_id)
                .order_by(models.MemoryChunk.ordinal)
            )
        ]
    return memory_id, chunks


def a_word_in(text: str) -> str:
    """A real word from the text, for building a response that should verify.

    Robust rather than `text.split()[0]`, which on this fixture is the markdown
    heading marker and strips to the empty string — a name the adapter discards
    before it counts anything, which made the assertion below fail for a reason
    that had nothing to do with the behaviour under test.
    """
    for token in re.findall(r"[A-Za-z]{4,}", text):
        return str(token)
    raise AssertionError(f"no usable word in {text[:80]!r}")


async def mention_count(sessions: async_sessionmaker[AsyncSession]) -> int:
    async with sessions() as session:
        return int(
            (
                await session.execute(select(func.count()).select_from(models.EntityMention))
            ).scalar_one()
        )


# --------------------------------------------------------------------------
# The offset guarantee
# --------------------------------------------------------------------------


async def test_a_name_that_is_not_in_the_text_is_dropped(pipeline: Pipeline) -> None:
    """The rule that makes a mention worth storing.

    A language model asked for entities will occasionally return one that is not
    in the text — a paraphrase, an expansion, or an outright invention. Stored,
    it would carry offsets pointing at real text that says something else, and
    the provenance chain M2.5 built would be citing a span nobody wrote. That is
    worse than losing the mention, because it is wrong rather than absent.

    Driven through the real adapter rather than the fake, because dropping is
    the adapter's job: the fake deliberately returns phantoms so the layer under
    test has something to refuse.
    """
    _memory_id, chunks = await one_memory(pipeline)
    _, first_chunk_text = chunks[0]

    # One real name from the text and one that appears nowhere in the corpus.
    present = a_word_in(first_chunk_text)
    payload = (
        '{"results": [{"index": 0, "entities": ['
        f'{{"name": "{present}", "type": "concept", "confidence": 0.9}}, '
        '{"name": "Kubernetes", "type": "technology", "confidence": 0.95}'
        "]}]}"
    )
    extractor = LlmEntityExtractor(FakeLanguageModel(payload), batch_size=1)

    found = await extractor.extract(first_chunk_text, kind=MemoryKind.NOTE)

    names = [entity.name for entity in found]
    assert "Kubernetes" not in names
    assert present in names
    assert extractor.stats.dropped_not_found == 1

    # And the offsets that survived are real, which is the actual invariant.
    for entity in found:
        assert first_chunk_text[entity.char_start : entity.char_end] == entity.name


async def test_a_dropped_mention_is_never_written(pipeline: Pipeline) -> None:
    """The same rule, one layer down: nothing unverifiable reaches the table."""
    memory_id, _chunks = await one_memory(pipeline)
    extract = ExtractEntities(
        pipeline.sessions,
        FakeEntityExtractor(phantom_names=["Kubernetes", "Kafka"]),
    )

    await extract(memory_id)

    async with pipeline.sessions() as session:
        rows = await session.execute(
            select(models.Entity.name, models.EntityMention.char_start,
                   models.EntityMention.char_end, models.MemoryChunk.content)
            .join(models.EntityMention, models.EntityMention.entity_id == models.Entity.id)
            .join(models.MemoryChunk, models.MemoryChunk.id == models.EntityMention.chunk_id)
        )
        stored = list(rows)

    assert stored, "the fake should have found real entities too"
    # The phantoms carry offsets of (0, len(name)), which is why storing them
    # unchecked would be actively misleading rather than merely wrong.
    for name, char_start, char_end, content in stored:
        assert content[char_start:char_end] == name


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------


async def test_re_running_at_the_same_version_does_nothing(pipeline: Pipeline) -> None:
    """The M1.4/M1.5 skip, and the reason a re-run is free.

    Without it, every run of `extract-entities` is a second full spend against a
    rate-limited free tier for a result already in the table.
    """
    memory_id, _ = await one_memory(pipeline)
    extractor = FakeEntityExtractor()
    extract = ExtractEntities(pipeline.sessions, extractor)

    first = await extract(memory_id)
    assert first.outcome is ExtractOutcome.EXTRACTED
    assert first.mentions > 0
    calls_after_first = len(extractor.calls)
    mentions_after_first = await mention_count(pipeline.sessions)

    second = await extract(memory_id)

    assert second.outcome is ExtractOutcome.SKIPPED
    assert second.mentions == 0
    # The model was not consulted a second time, which is the point.
    assert len(extractor.calls) == calls_after_first
    assert await mention_count(pipeline.sessions) == mentions_after_first


async def test_a_new_extractor_version_redoes_the_work(pipeline: Pipeline) -> None:
    """The other half of the same mechanism.

    A skip keyed on "any mentions exist" would make a prompt improvement
    unshippable — the corpus would keep the old extraction forever and nothing
    would say so.
    """
    memory_id, _ = await one_memory(pipeline)
    await ExtractEntities(pipeline.sessions, FakeEntityExtractor())(memory_id)

    report = await ExtractEntities(
        pipeline.sessions, FakeEntityExtractor(version="fake-extractor@2")
    )(memory_id)

    assert report.outcome is ExtractOutcome.EXTRACTED
    async with pipeline.sessions() as session:
        versions = {
            row[0]
            for row in await session.execute(
                select(models.EntityMention.extractor_version).distinct()
            )
        }
    # Exactly one version present: the old mentions were replaced, not added to.
    # Two versions at once would double-count every query over this table.
    assert versions == {"fake-extractor@2"}


# --------------------------------------------------------------------------
# Malformed output
# --------------------------------------------------------------------------


async def test_unparseable_json_raises_permanent_after_one_retry() -> None:
    """One retry, then stop. No database needed — this is the adapter's decision.

    A model that cannot produce JSON twice will not produce it on the fifth
    attempt, and the worker's remaining attempts are better spent on other jobs.
    Retrying to exhaustion also multiplies the cost of a broken prompt by the
    attempt budget, against a quota measured per day.
    """
    model = FakeLanguageModel("not json at all", "still not json")
    extractor = LlmEntityExtractor(model, batch_size=1)

    with pytest.raises(PermanentError, match="unparseable JSON twice"):
        await extractor.extract("some text", kind=MemoryKind.NOTE)

    # Exactly two calls: the attempt and the one retry.
    assert len(model.calls) == 2
    assert extractor.stats.retries == 1
    # The retry carried the stricter reminder; the first did not.
    assert "was not valid JSON" not in model.calls[0][0]
    assert "was not valid JSON" in model.calls[1][0]


async def test_a_retry_that_parses_is_not_an_error() -> None:
    """The retry has to be able to succeed, or it is just a slower failure."""
    model = FakeLanguageModel(
        "here you go:", '{"results": [{"index": 0, "entities": ['
        '{"name": "queue", "type": "concept", "confidence": 0.9}]}]}'
    )
    extractor = LlmEntityExtractor(model, batch_size=1)

    found = await extractor.extract("the queue holds jobs", kind=MemoryKind.NOTE)

    assert [entity.name for entity in found] == ["queue"]
    assert extractor.stats.retries == 1


# --------------------------------------------------------------------------
# Cascade
# --------------------------------------------------------------------------


async def test_deleting_a_memory_cascades_to_its_mentions(pipeline: Pipeline) -> None:
    """A mention whose chunk is gone has no text to point at.

    Enforced by the database rather than by application code, because the
    application is not the only writer — a replay's `DELETE FROM memories` and a
    manual fix both have to leave the same invariant standing.
    """
    memory_id, _ = await one_memory(pipeline)
    await ExtractEntities(pipeline.sessions, FakeEntityExtractor())(memory_id)
    assert await mention_count(pipeline.sessions) > 0

    async with pipeline.sessions.begin() as session:
        await session.execute(delete(models.Memory).where(models.Memory.id == memory_id))

    assert await mention_count(pipeline.sessions) == 0

    # The entities themselves survive, deliberately: an entity outliving its
    # last mention is an orphan for a later sweep, not a row to delete inside
    # somebody else's transaction.
    async with pipeline.sessions() as session:
        entities = (
            await session.execute(select(func.count()).select_from(models.Entity))
        ).scalar_one()
    assert entities > 0


async def test_entity_rows_are_shared_across_mentions(pipeline: Pipeline) -> None:
    """Identity, without which "most-mentioned" is a list of coincidences.

    The same name in twenty chunks is one entity with twenty mentions, not
    twenty entities. That is exact-match deduplication on a casefolded name and
    emphatically not resolution — see M3.2.
    """
    memory_id, _ = await one_memory(pipeline)
    await ExtractEntities(pipeline.sessions, FakeEntityExtractor())(memory_id)

    async with pipeline.sessions() as session:
        duplicated = (
            await session.execute(
                select(models.Entity.canonical_name, models.Entity.type, func.count())
                .group_by(models.Entity.canonical_name, models.Entity.type)
                .having(func.count() > 1)
            )
        ).all()

    assert duplicated == [], "canonical_name and type must identify one row"


async def test_the_entity_type_vocabulary_is_closed(pipeline: Pipeline) -> None:
    """A type outside the enum is dropped, not coerced.

    A model that invented a type has probably invented the entity, and coercing
    it to `concept` would put a guess into a column that filters are written
    against.
    """
    _, chunks = await one_memory(pipeline)
    text = chunks[0][1]
    present = a_word_in(text)
    payload = (
        '{"results": [{"index": 0, "entities": ['
        f'{{"name": "{present}", "type": "spacecraft", "confidence": 0.99}}'
        "]}]}"
    )
    extractor = LlmEntityExtractor(FakeLanguageModel(payload), batch_size=1)

    assert await extractor.extract(text, kind=MemoryKind.NOTE) == []
    assert extractor.stats.dropped_bad_type == 1
    assert set(EntityType) and "spacecraft" not in {member.value for member in EntityType}
