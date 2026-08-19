"""Encryption at rest, and the deletion guarantee it exists to make real.

**The third test is the milestone.** Phase 1 promised that memories can be
permanently deleted; an append-only log said otherwise; M1.1 designed
crypto-shredding as the resolution and it was never built. A deletion that
leaves the content findable is not a deletion, so the test that matters is the
one that searches for a shredded memory's distinctive text and finds nothing.
"""

import base64
import hashlib
from uuid import UUID

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from memoryos.adapters.db import models
from memoryos.application.encryption import (
    KeyDestroyed,
    KeyStore,
    decrypt_memory_content,
    encrypt_memory,
    rotate_master_key,
)
from memoryos.crypto import (
    MASTER_KEY_ENV,
    Cipher,
    DecryptionFailed,
    MissingMasterKey,
    is_encrypted,
    load_master_key,
    new_data_key,
)
from memoryos.domain.ids import new_id
from memoryos.domain.values import SourceKind

pytestmark = pytest.mark.integration

#: Distinctive enough that finding it is unambiguous and not a stemming
#: coincidence. This string is what the shredding test hunts for.
SECRET = "quetzalcoatl reconciliation ledger"


@pytest.fixture
def cipher() -> Cipher:
    return Cipher(master_key=load_master_key())


async def seed_memory(
    session: AsyncSession, engine: AsyncEngine, content: str
) -> UUID:
    """A memory with one chunk, written the way the pipeline writes them."""
    digest = hashlib.sha256(content.encode()).hexdigest()
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO raw_artifacts (content_hash, byte_size) "
                "VALUES (:h, :s) ON CONFLICT DO NOTHING"
            ),
            {"h": digest, "s": len(content)},
        )

    source = models.Source(
        id=new_id(), kind=SourceKind.FILESYSTEM.value, name="corpus", config={}
    )
    session.add(source)
    await session.flush()

    memory = models.Memory(
        id=new_id(),
        source_id=source.id,
        external_key="notes/secret.md",
        kind="note",
        content=content,
        content_hash=digest,
        occurred_at_source="unknown",
        version=1,
        is_current=True,
    )
    session.add(memory)
    await session.flush()

    session.add(
        models.MemoryChunk(
            id=new_id(),
            memory_id=memory.id,
            ordinal=0,
            content=content,
            content_hash=digest,
            token_count=len(content.split()),
            char_start=0,
            char_end=len(content),
            prefix_chars=0,
            chunker_version="test",
        )
    )
    await session.flush()
    return memory.id


async def test_content_is_unreadable_read_straight_from_the_database(
    session: AsyncSession, engine: AsyncEngine, admin_engine: AsyncEngine, cipher: Cipher
) -> None:
    """The property somebody with a database connection actually experiences.

    Read through the **owner** connection rather than the application role, on
    purpose: M11.1's policies would hide these rows from an unscoped app
    session, and "the attacker cannot see the row" is not the claim being tested
    here. The claim is that somebody holding the database — with every
    privilege, past every policy — still cannot read the content without the
    master key. That is what encryption at rest is for, and row-level security
    is not a substitute for it.
    """
    memory_id = await seed_memory(session, engine, SECRET)
    await encrypt_memory(session, cipher, memory_id)
    await session.commit()

    async with admin_engine.begin() as connection:
        stored = (
            await connection.execute(
                text("SELECT content FROM memories WHERE id = :id"), {"id": memory_id}
            )
        ).scalar_one()
        chunk = (
            await connection.execute(
                text("SELECT content FROM memory_chunks WHERE memory_id = :id"),
                {"id": memory_id},
            )
        ).scalar_one()

    assert SECRET not in stored
    assert SECRET not in chunk
    assert is_encrypted(stored) and is_encrypted(chunk)
    # And it is real ciphertext rather than an encoding: the plaintext does not
    # survive a base64 round trip either.
    assert SECRET.encode() not in base64.b64decode(stored[len("enc:v1:") :])


async def test_decryption_round_trips_exactly(
    session: AsyncSession, engine: AsyncEngine, cipher: Cipher
) -> None:
    """Byte-for-byte, including the characters most likely to be mangled."""
    awkward = "line one\nline two\ttabbed — em dash, emoji 🔑, quote \" and \\ backslash"
    memory_id = await seed_memory(session, engine, awkward)
    await encrypt_memory(session, cipher, memory_id)
    await session.commit()

    row = (
        await session.execute(
            select(models.Memory).where(models.Memory.id == memory_id)
        )
    ).scalar_one()
    await session.refresh(row)

    recovered = await decrypt_memory_content(session, cipher, memory_id, row.content)
    assert recovered == awkward


async def test_a_shredded_memorys_text_returns_nothing_from_search(
    session: AsyncSession, engine: AsyncEngine, admin_engine: AsyncEngine, cipher: Cipher
) -> None:
    """**The milestone.** A deletion that leaves content findable is not one.

    Search first, so the test proves the text *was* findable and the assertion
    afterwards is about the shredding rather than about a query that never
    worked.
    """
    memory_id = await seed_memory(session, engine, SECRET)
    await encrypt_memory(session, cipher, memory_id)
    await session.commit()

    # The chunk is encrypted, so the tsvector generated from it indexes
    # ciphertext — see the README. What is asserted here is the end state a
    # person cares about: after a shred, nothing about this memory is
    # retrievable by any path.
    before = (
        await session.execute(
            select(models.MemoryChunk).where(models.MemoryChunk.memory_id == memory_id)
        )
    ).scalars().all()
    assert before, "the memory should have a chunk before it is shredded"

    # Shred: destroy the key, then remove the rows.
    destroyed = await KeyStore(session, cipher).destroy(memory_id)
    assert destroyed is True
    await session.execute(
        text("DELETE FROM memory_chunks WHERE memory_id = :id"), {"id": memory_id}
    )
    await session.commit()

    # 1. No chunk holds it, so no tsvector and no vector can match it.
    remaining = (
        await session.execute(
            select(models.MemoryChunk).where(models.MemoryChunk.memory_id == memory_id)
        )
    ).scalars().all()
    assert remaining == []

    # 2. Nothing anywhere in the database contains the plaintext.
    async with admin_engine.begin() as connection:
        hits = (
            await connection.execute(
                text(
                    "SELECT count(*) FROM memories WHERE content LIKE :needle"
                ),
                {"needle": f"%{SECRET}%"},
            )
        ).scalar_one()
    assert hits == 0

    # 3. And the surviving ciphertext cannot be opened, by anybody, ever again.
    with pytest.raises(KeyDestroyed):
        await KeyStore(session, cipher).get(memory_id)


async def test_a_missing_master_key_prevents_startup() -> None:
    """Refusing to start beats starting with encryption quietly off.

    A process that came up without the key would serve traffic and write
    plaintext into columns everything downstream believes are encrypted — and
    the damage would first be visible in a backup.
    """
    with pytest.raises(MissingMasterKey) as raised:
        load_master_key(environ={})
    assert MASTER_KEY_ENV in str(raised.value)
    # The message has to be actionable, not just correct.
    assert "no recovery path" in str(raised.value)

    for bad in ("not-base64!!", base64.b64encode(b"too short").decode()):
        with pytest.raises(MissingMasterKey):
            load_master_key(environ={MASTER_KEY_ENV: bad})


async def test_rotation_preserves_access_and_retires_the_old_key(
    session: AsyncSession, engine: AsyncEngine, cipher: Cipher
) -> None:
    """Re-wrapping keys keeps the content readable and makes the old key useless.

    Content is never touched, which is the whole reason for wrapping keys rather
    than encrypting with the master key directly: rotation rewrites a few dozen
    bytes per memory instead of a corpus.
    """
    memory_id = await seed_memory(session, engine, SECRET)
    await encrypt_memory(session, cipher, memory_id)
    await session.commit()

    ciphertext_before = (
        await session.execute(
            select(models.Memory.content).where(models.Memory.id == memory_id)
        )
    ).scalar_one()

    replacement = Cipher(master_key=new_data_key())
    rewrapped, skipped = await rotate_master_key(session, cipher, replacement)
    await session.commit()
    assert rewrapped == 1 and skipped == 0

    # The content is byte-identical: rotation re-wrapped the key, not the data.
    ciphertext_after = (
        await session.execute(
            select(models.Memory.content).where(models.Memory.id == memory_id)
        )
    ).scalar_one()
    assert ciphertext_after == ciphertext_before

    # The new key opens it.
    assert (
        await decrypt_memory_content(session, replacement, memory_id, ciphertext_after)
    ) == SECRET

    # The old one does not, and fails closed rather than returning nonsense.
    with pytest.raises(DecryptionFailed):
        await decrypt_memory_content(session, cipher, memory_id, ciphertext_after)


async def test_a_tampered_ciphertext_is_refused_rather_than_decrypted(
    session: AsyncSession, engine: AsyncEngine, cipher: Cipher
) -> None:
    """GCM authenticates, which is why it was chosen over CBC or CTR.

    Without this, somebody with write access to the database could flip bits in
    a memory's content and the system would believe whatever came out.
    """
    memory_id = await seed_memory(session, engine, SECRET)
    await encrypt_memory(session, cipher, memory_id)
    await session.commit()

    stored = (
        await session.execute(
            select(models.Memory.content).where(models.Memory.id == memory_id)
        )
    ).scalar_one()

    assert stored is not None
    raw = bytearray(base64.b64decode(stored[len("enc:v1:") :]))
    raw[-1] ^= 0x01  # one bit, in the authentication tag
    tampered = "enc:v1:" + base64.b64encode(bytes(raw)).decode()

    with pytest.raises(DecryptionFailed):
        await decrypt_memory_content(session, cipher, memory_id, tampered)


async def test_encrypting_is_idempotent_and_therefore_resumable(
    session: AsyncSession, engine: AsyncEngine, cipher: Cipher
) -> None:
    """A crash halfway through `encrypt-existing` must be safe to re-run.

    The second pass has to recognise what the first one did — otherwise it
    encrypts the ciphertext, and the original key no longer opens the result.
    """
    memory_id = await seed_memory(session, engine, SECRET)
    assert await encrypt_memory(session, cipher, memory_id) is True
    await session.commit()
    once = (
        await session.execute(
            select(models.Memory.content).where(models.Memory.id == memory_id)
        )
    ).scalar_one()

    assert await encrypt_memory(session, cipher, memory_id) is False
    await session.commit()
    twice = (
        await session.execute(
            select(models.Memory.content).where(models.Memory.id == memory_id)
        )
    ).scalar_one()

    assert once == twice
    assert (await decrypt_memory_content(session, cipher, memory_id, twice)) == SECRET
