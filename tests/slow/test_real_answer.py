"""The assertion a fake language model cannot make.

Every other test in this milestone proves the machinery around the model works:
the context is assembled correctly, the citations are checked, an uncited
sentence is flagged. All of them would pass against a model that answered every
question with training-data recall and a plausible `[1]`.

This one asks a real model a question whose answer is in the corpus, and checks
the thing that actually matters: that a cited passage genuinely contains the
claim. Not by asking another model — a model grading its own grounding is not
evidence — but by requiring a distinctive term from the answer to appear in the
passage it cited.

Skipped rather than failed without a key. A test that fails on every machine
without a credential trains people to ignore the suite, and the key is optional
for everything except answering.
"""

import os
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
from memoryos.adapters.embedding.sentence_transformers import SentenceTransformerEmbedder
from memoryos.adapters.llm.gemini import GeminiLanguageModel
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.answering import AnswerQuestion
from memoryos.application.embed import EmbedMemory
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.search import SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import SourceKind

pytestmark = pytest.mark.slow

API_KEY = os.environ.get("MEMOS_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

requires_key = pytest.mark.skipif(
    not API_KEY,
    reason="needs MEMOS_GEMINI_API_KEY; answering is the only feature that does",
)

# A distinctive, checkable fact. `FOR UPDATE SKIP LOCKED` is the phrase a
# grounded answer has to reach for, and it appears nowhere else in the fixture.
QUEUE = (
    "# Claiming work\n\n"
    "The claim query selects the oldest pending job and marks it running. "
    "FOR UPDATE SKIP LOCKED on the inner select is the clause that stops two "
    "workers taking the same row: without it every worker selects the same top "
    "row and all but one block on its lock, so the queue drains serially "
    "however many workers are running.\n"
)
BREAD = (
    "# Sourdough\n\n"
    "A wild yeast starter is fed flour and water until it doubles reliably, "
    "then folded gently and given a long cold rest in the refrigerator before "
    "baking.\n"
)


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@pytest.fixture
async def ask(
    tmp_path: Path,
    sessions: async_sessionmaker[AsyncSession],
    embedder: SentenceTransformerEmbedder,
) -> AnswerQuestion:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "queue.md").write_text(QUEUE)
    (root / "bread.md").write_text(BREAD)

    source = Source(
        id=new_id(), kind=SourceKind.FILESYSTEM, name="fixture", config={"root": str(root)}
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
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

    search = SearchMemories(
        sessions,
        embedder,
        PgVectorStore(sessions, embedder, default_ef_search=100),
        PostgresKeywordStore(sessions),
    )
    return AnswerQuestion(
        sessions, search, GeminiLanguageModel(API_KEY), embedder
    )


@requires_key
async def test_the_answer_cites_a_passage_that_contains_the_claim(
    ask: AnswerQuestion,
) -> None:
    result = await ask("what stops two workers claiming the same job", k=5)

    assert result.answer.strip()
    assert not result.refused, result.answer
    assert result.verification.hallucinated_indices == []
    assert result.verification.cited_indices, "the answer cited nothing"

    # The claim is in the answer...
    assert "skip locked" in result.answer.lower(), result.answer

    # ...and in a passage the answer actually cited. This is the assertion the
    # whole milestone rests on: a citation that does not support its sentence is
    # indistinguishable from a fabrication.
    cited = [
        result.context.passage(index)
        for index in set(result.verification.cited_indices)
    ]
    supporting = [
        passage
        for passage in cited
        if passage is not None and "SKIP LOCKED" in passage.text
    ]
    assert supporting, [passage.label for passage in cited if passage]


@requires_key
async def test_it_refuses_a_question_the_corpus_cannot_answer(
    ask: AnswerQuestion,
) -> None:
    """The guardrail, against a real model that would rather be helpful."""
    result = await ask("what is our AWS billing setup", k=5)

    assert result.refused, result.answer
    assert result.verification.hallucinated_indices == []
    # Nothing about AWS is in the fixture, so any specific claim is invented.
    assert "aws" not in result.answer.lower() or "not" in result.answer.lower()


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
