"""The grounded-answer path end to end, with a scripted model.

The fake makes the model's behaviour a test input rather than a variable, which
is the only way to assert what happens when it misbehaves — and misbehaving is
the case that matters. A real model that happens to answer well proves nothing
about what this does when it does not.
"""

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.answering import AnswerQuestion
from memoryos.application.embed import EmbedMemory
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.ports import SearchFilters
from memoryos.application.search import SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType, TransientError
from memoryos.domain.values import SourceKind
from tests.support.fakes import FakeEmbedder, FakeLanguageModel

pytestmark = pytest.mark.integration

QUEUE = (
    "The claim query selects the oldest pending job and marks it running. "
    "FOR UPDATE SKIP LOCKED on the inner select stops two workers taking the "
    "same row. "
)
BREAD = (
    "A wild yeast starter is fed flour and water until it doubles reliably, "
    "then folded gently and given a long cold rest. "
)


async def build(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession], model: FakeLanguageModel
) -> AnswerQuestion:
    root = tmp_path / "corpus"
    if not root.exists():
        root.mkdir()
        (root / "queue.md").write_text("# Queue\n\n" + QUEUE * 4 + "\n")
        (root / "bread.md").write_text("# Bread\n\n" + BREAD * 4 + "\n")

        source = Source(
            id=new_id(),
            kind=SourceKind.FILESYSTEM,
            name="fixture",
            config={"root": str(root)},
        )
        async with sessions.begin() as session:
            await SqlAlchemySourceRepository(session).add(source)

        blobs = FilesystemBlobStore(tmp_path / "blobs")
        embedder = FakeEmbedder()
        await SyncSource(sessions, FilesystemConnector(blobs), blobs)(source.id, full=True)
        await _drain(
            sessions,
            JobType.NORMALIZE_MEMORY,
            NormalizeMemory(sessions, blobs, build_parsers(), StructuralChunker(embedder)),
        )
        await _drain(
            sessions,
            JobType.EMBED_MEMORY,
            EmbedMemory(sessions, embedder, PostgresEmbeddingCache(sessions)),
        )

    embedder = FakeEmbedder()
    search = SearchMemories(
        sessions,
        embedder,
        PgVectorStore(sessions, embedder, default_ef_search=100),
        PostgresKeywordStore(sessions),
    )
    return AnswerQuestion(sessions, search, model, embedder)


async def test_a_well_behaved_answer_is_grounded_and_resolves_its_citations(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    model = FakeLanguageModel("The claim query uses FOR UPDATE SKIP LOCKED [1].")
    ask = await build(tmp_path, sessions, model)

    result = await ask("what stops two workers claiming the same job", k=5)

    assert result.verification.grounded
    assert result.verification.citation_rate == 1.0
    assert result.verification.hallucinated_indices == []
    assert result.citations, "a cited passage resolves to a full M2.5 citation"
    for explained in result.citations:
        for citation in explained.citations:
            assert citation.version >= 1
            assert citation.char_end > citation.char_start

    # The passages precede the question in the prompt: long-context models
    # attend better to material before the instruction, and the instruction is
    # the part that must not be forgotten.
    prompt = model.last_user_prompt
    assert prompt.index("[1]") < prompt.index("what stops two workers")
    assert "citing each claim" in prompt

    assert result.timing.generate_ms >= 0
    assert result.timing.total_ms >= result.timing.retrieve_ms


async def test_an_invented_citation_index_is_reported_not_resolved(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The model cites a passage it was never given.

    The answer still comes back — the caller needs to see what was produced —
    but the index appears in `hallucinated_indices` and resolves to no citation,
    because there is no passage behind it to resolve.
    """
    model = FakeLanguageModel("Workers coordinate through a distributed lock [99].")
    ask = await build(tmp_path, sessions, model)

    result = await ask("what stops two workers claiming the same job", k=3)

    assert result.verification.hallucinated_indices == [99]
    assert not result.verification.grounded
    assert result.citations == []
    assert result.answer  # returned, not swallowed


async def test_an_uncited_sentence_is_marked_in_place(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    model = FakeLanguageModel(
        "The claim query uses SKIP LOCKED [1]. "
        "It also uses an advisory lock as a second layer of protection."
    )
    ask = await build(tmp_path, sessions, model)

    result = await ask("how does claiming work", k=3)

    assert result.verification.citation_rate == pytest.approx(0.5)
    marked = result.verification.marked()
    assert "second layer of protection. [unsupported]" in marked
    # And the grounded sentence is untouched.
    assert "SKIP LOCKED [1]." in marked


async def test_a_transient_model_failure_propagates_for_the_worker_to_retry(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Not swallowed into a fabricated apology.

    `TransientError` is what the worker's backoff is keyed on, so the use case
    lets it through unchanged rather than turning a rate limit into an answer
    saying the corpus is empty.
    """
    model = FakeLanguageModel(raises=TransientError("429 quota exceeded"))
    ask = await build(tmp_path, sessions, model)

    with pytest.raises(TransientError, match="429"):
        await ask("what stops two workers claiming the same job", k=3)


async def test_nothing_retrieved_refuses_without_asking_the_model(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """No context, no generation, no opportunity to invent one."""
    model = FakeLanguageModel("Here is a confident answer about nothing [1].")
    ask = await build(tmp_path, sessions, model)

    result = await ask(
        "quarterly revenue by region",
        k=3,
        filters=_impossible_filter(),
    )

    assert result.refused
    assert model.calls == [], "the model was never asked"
    assert result.verification.hallucinated_indices == []


def _impossible_filter() -> SearchFilters:
    # A source id nothing belongs to: retrieval returns nothing at all.
    return SearchFilters(source_ids=[new_id()])


async def _drain(
    sessions: async_sessionmaker[AsyncSession], job_type: JobType, handler: object
) -> None:
    async with sessions() as session:
        targets = [
            UUID(row[0]["memory_id"])
            for row in await session.execute(
                select(models.Job.payload).where(models.Job.job_type == job_type.value)
            )
        ]
    for memory_id in targets:
        await handler(memory_id)  # type: ignore[operator]
    async with sessions.begin() as session:
        await session.execute(delete(models.Job).where(models.Job.job_type == job_type.value))
