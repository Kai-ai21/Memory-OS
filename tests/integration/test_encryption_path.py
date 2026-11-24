"""Encryption on the path content actually takes (M12.1).

M11.2's tests all started from `encrypt_memory`, the function behind
`encrypt-existing`, and that is how both halves of the gap stayed invisible:
nothing tested that *ingestion* encrypts, and nothing tested that anything
*reading* content gets text back. On the real corpus that meant new memories
were stored in plaintext, and the encrypted ones reached search, the reranker
and the model as base64.

So every test here goes through the pipeline — sync, normalize, embed — or
through the session the application reads with, and checks the stored bytes
through the owner connection, past every policy, where the claim lives.
"""

import os

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.plaintext import (
    KEEP_CIPHERTEXT,
    CiphertextWithoutCipher,
    PlaintextContentWrite,
    session_info,
)
from memoryos.application.ports import SearchFilters
from memoryos.crypto import Cipher, is_encrypted, load_master_key
from tests.integration.conftest import PARAGRAPH, Pipeline

pytestmark = pytest.mark.integration


@pytest.fixture
def strict(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Sessions exactly as the application opens them: a cipher, no allowances."""
    return async_sessionmaker(
        engine,
        expire_on_commit=False,
        info=session_info(Cipher(master_key=load_master_key())),
    )


async def stored_chunks(admin_engine: AsyncEngine) -> list[tuple[str, str | None]]:
    """What is on disk, read as the owner: content and the search vector."""
    async with admin_engine.connect() as connection:
        rows = await connection.execute(
            text("SELECT content, search_vector::text FROM memory_chunks")
        )
        return [(row[0], row[1]) for row in rows]


async def test_ingestion_encrypts_what_it_writes(
    pipeline: Pipeline, admin_engine: AsyncEngine
) -> None:
    """The half M11.2 left out: nothing on this path encrypted until M12.1."""
    await pipeline.ingest()

    chunks = await stored_chunks(admin_engine)
    assert chunks
    assert all(is_encrypted(content) for content, _ in chunks)
    assert not any("quick brown fox" in content for content, _ in chunks)

    async with admin_engine.connect() as connection:
        memories = (
            await connection.execute(text("SELECT content FROM memories"))
        ).scalars().all()
    assert memories and all(is_encrypted(content) for content in memories)


async def test_the_search_vector_is_built_from_the_plaintext(
    pipeline: Pipeline, admin_engine: AsyncEngine
) -> None:
    """A vector derived from the envelope would hold base64 and match nothing."""
    await pipeline.ingest()

    vectors = [vector for _, vector in await stored_chunks(admin_engine)]
    assert all(vector for vector in vectors)
    assert any("'fox'" in vector for vector in vectors if vector)


async def test_keyword_search_returns_text_not_ciphertext(pipeline: Pipeline) -> None:
    await pipeline.ingest()

    hits = await PostgresKeywordStore(pipeline.sessions).search(
        "fudge", k=5, filters=SearchFilters()
    )
    assert hits
    assert all("fudge" in hit.text for hit in hits)
    assert not any(is_encrypted(hit.text) for hit in hits)


async def test_embedding_reads_the_plaintext(pipeline: Pipeline) -> None:
    """The embedder has to see words. An embedding of base64 is a vector of noise."""
    await pipeline.ingest()
    await pipeline.embed_all()

    embedded = [text for batch in pipeline.embedder.calls for text in batch]
    assert embedded
    assert all(not is_encrypted(passage) for passage in embedded)
    assert any("quick brown fox" in passage for passage in embedded)


async def test_content_selected_without_an_id_is_still_decrypted(
    pipeline: Pipeline,
) -> None:
    """The read path's last resort: identify the row by its ciphertext."""
    await pipeline.ingest()

    async with pipeline.sessions() as session:
        texts = (await session.execute(select(models.MemoryChunk.content))).scalars().all()
    assert texts and all("quick brown fox" in value for value in texts)


async def test_the_stored_form_is_available_when_asked_for(pipeline: Pipeline) -> None:
    await pipeline.ingest()

    async with pipeline.sessions() as session:
        stored = (
            await session.execute(
                select(models.MemoryChunk.content).execution_options(
                    **{KEEP_CIPHERTEXT: True}
                )
            )
        ).scalars().all()
    assert stored and all(is_encrypted(value) for value in stored)


async def test_a_decrypted_row_is_never_written_back_as_plaintext(
    pipeline: Pipeline,
    strict: async_sessionmaker[AsyncSession],
    admin_engine: AsyncEngine,
) -> None:
    """Decrypted text is set as the *committed* value, so it is not a change.

    If it were, loading a chunk and touching any other column would flush the
    plaintext back over the ciphertext — encryption undone by a read.
    """
    await pipeline.ingest()

    async with strict.begin() as session:
        chunk = (await session.execute(select(models.MemoryChunk).limit(1))).scalar_one()
        assert "quick brown fox" in chunk.content
        chunk.meta = {**chunk.meta, "touched": True}

    assert all(is_encrypted(content) for content, _ in await stored_chunks(admin_engine))


async def test_an_application_session_refuses_a_plaintext_write(
    pipeline: Pipeline, strict: async_sessionmaker[AsyncSession]
) -> None:
    """The guard against the next write path that forgets to encrypt."""
    await pipeline.ingest()

    with pytest.raises(PlaintextContentWrite):
        async with strict.begin() as session:
            chunk = (
                await session.execute(select(models.MemoryChunk).limit(1))
            ).scalar_one()
            chunk.content = "written in the clear"


async def test_a_session_without_a_cipher_refuses_ciphertext(
    pipeline: Pipeline, engine: AsyncEngine
) -> None:
    """Better an error than base64 presented as somebody's notes."""
    await pipeline.ingest()

    bare = async_sessionmaker(engine, expire_on_commit=False)
    with pytest.raises(CiphertextWithoutCipher):
        async with bare() as session:
            await session.execute(select(models.MemoryChunk.content))


async def test_adopted_chunks_are_resealed_under_the_new_version(
    pipeline: Pipeline, admin_engine: AsyncEngine
) -> None:
    """A cosmetic edit moves chunks to a new memory id, and the old seal binds the old id.

    Line endings change the bytes and therefore the version, not the text, so
    the earlier version's chunks are adopted rather than rebuilt. Moved as they
    were, they would fail to authenticate under the new memory.
    """
    await pipeline.ingest()
    async with admin_engine.connect() as connection:
        before = set(
            (await connection.execute(text("SELECT content FROM memory_chunks")))
            .scalars()
            .all()
        )

    notes = pipeline.root / "notes.txt"
    notes.write_bytes(notes.read_bytes().replace(b"\n", b"\r\n"))
    # The adoption is decided by the normalized text, which CRLF does not change.
    await pipeline.ingest()

    async with pipeline.sessions() as session:
        rows = (
            await session.execute(
                select(models.Memory.external_key, models.MemoryChunk.content)
                .join(models.MemoryChunk, models.MemoryChunk.memory_id == models.Memory.id)
                .where(models.Memory.is_current.is_(True))
            )
        ).all()
    notes_chunks = [content for key, content in rows if key == "notes.txt"]
    assert notes_chunks and all(PARAGRAPH.split()[0] in c for c in notes_chunks)

    async with admin_engine.connect() as connection:
        after = set(
            (await connection.execute(text("SELECT content FROM memory_chunks")))
            .scalars()
            .all()
        )
    # Resealed, not merely moved: the envelopes changed.
    assert after - before


async def test_a_shredded_key_reads_as_nothing_rather_than_failing(
    pipeline: Pipeline, admin_engine: AsyncEngine
) -> None:
    """Between destroying a key and purging its rows, a page of results still renders."""
    await pipeline.ingest()
    async with admin_engine.begin() as connection:
        await connection.execute(
            update(models.MemoryKey).values(wrapped_key=None, destroyed_at=func.now())
        )

    async with pipeline.sessions() as session:
        texts = (
            await session.execute(
                select(models.MemoryChunk.memory_id, models.MemoryChunk.content)
            )
        ).all()
    assert texts and all(content == "" for _, content in texts)


def test_the_suite_runs_with_a_master_key() -> None:
    """Every test above assumes the conftest's key; say so if it is missing."""
    assert os.environ.get("MEMOS_MASTER_KEY")

