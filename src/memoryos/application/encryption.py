"""Using the keys: encrypting a memory, shredding one, rotating the master key.

**Encryption here is explicit, not transparent, and that is a finding rather
than a shortcut.** A SQLAlchemy `TypeDecorator` would make encrypted columns
invisible to the fifty-nine modules that read them — the trick M11.1 used for
scoping — but it cannot work here. Decrypting needs this memory's data key, the
key lives in another table, and a type decorator runs inside result processing
where there is no session and no way to await a lookup.

The alternative is to put the wrapped key in the ciphertext itself, and that
destroys the entire point: a backup would then carry the key next to the data it
unlocks, and shredding one row in `memory_keys` would protect nothing. So the
lookup is unavoidable, and encryption happens at named boundaries.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memoryos.adapters.db import models
from memoryos.crypto import (
    ALGORITHM,
    Cipher,
    DecryptionFailed,
    is_encrypted,
    new_data_key,
)

logger = structlog.get_logger(__name__)


class KeyDestroyed(RuntimeError):
    """The memory was permanently deleted. Its content is gone, not withheld.

    A distinct type from `DecryptionFailed` because the two need different
    sentences on screen: one is "this was deleted on purpose" and the other is
    "something is wrong with your key or your data".
    """


@dataclass(slots=True)
class KeyStore:
    """Reads and writes `memory_keys`, and nothing else touches that table."""

    session: AsyncSession
    cipher: Cipher

    async def ensure(self, memory_id: UUID) -> bytes:
        """This memory's data key, creating one the first time.

        Idempotent, which is what makes `encrypt-existing` resumable: a second
        pass over a memory that already has a key gets the same key back rather
        than a new one that would orphan the content encrypted under the first.
        """
        row = await self._row(memory_id)
        if row is not None:
            if row.wrapped_key is None:
                raise KeyDestroyed(f"Memory {memory_id} was permanently deleted.")
            return self.cipher.unwrap(row.wrapped_key, memory_id)

        data_key = new_data_key()
        self.session.add(
            models.MemoryKey(
                memory_id=memory_id,
                wrapped_key=self.cipher.wrap(data_key, memory_id),
                algorithm=ALGORITHM,
            )
        )
        await self.session.flush()
        return data_key

    async def get(self, memory_id: UUID) -> bytes | None:
        """The data key, or None when this memory has never been encrypted."""
        row = await self._row(memory_id)
        if row is None:
            return None
        if row.wrapped_key is None:
            raise KeyDestroyed(f"Memory {memory_id} was permanently deleted.")
        return self.cipher.unwrap(row.wrapped_key, memory_id)

    async def destroy(self, memory_id: UUID) -> bool:
        """**Crypto-shred.** Blank the key and stamp the time. Irreversible.

        The row survives without a key in it, which is the difference between
        "this was shredded" and "this never existed" — and the first is what an
        audit has to be able to say.

        Returns False when there was no key to destroy, so a caller can tell a
        second shred from a first without treating either as an error.
        """
        result = await self.session.execute(
            update(models.MemoryKey)
            .where(
                models.MemoryKey.memory_id == memory_id,
                models.MemoryKey.wrapped_key.isnot(None),
            )
            .values(wrapped_key=None, destroyed_at=datetime.now(UTC))
            .returning(models.MemoryKey.memory_id)
        )
        destroyed = result.first() is not None
        if destroyed:
            logger.info("encryption.key_destroyed", memory_id=str(memory_id))
        return destroyed

    async def _row(self, memory_id: UUID) -> models.MemoryKey | None:
        return (
            await self.session.execute(
                select(models.MemoryKey).where(
                    models.MemoryKey.memory_id == memory_id
                )
            )
        ).scalar_one_or_none()


async def encrypt_memory(
    session: AsyncSession, cipher: Cipher, memory_id: UUID
) -> bool:
    """Encrypt one memory's content and its chunks, in place.

    **Resumable, and the resumability is a property of the data rather than of a
    checkpoint file.** Every ciphertext carries a prefix, so a row that has
    already been done is recognisable on sight; the key is created idempotently;
    and the whole memory moves in one transaction. A crash halfway through a
    corpus leaves some memories encrypted and some not, which is a state the
    reader already handles because `Cipher.decrypt` passes plaintext through.

    Returns False when there was nothing to do.
    """
    store = KeyStore(session, cipher)
    memory = (
        await session.execute(
            select(models.Memory).where(models.Memory.id == memory_id)
        )
    ).scalar_one_or_none()
    if memory is None or memory.content is None:
        return False

    data_key = await store.ensure(memory_id)
    changed = False
    if not is_encrypted(memory.content):
        memory.content = cipher.encrypt(memory.content, data_key, memory_id)
        changed = True

    chunks = (
        (
            await session.execute(
                select(models.MemoryChunk).where(
                    models.MemoryChunk.memory_id == memory_id
                )
            )
        )
        .scalars()
        .all()
    )
    for chunk in chunks:
        if chunk.content is None:
            continue
        if is_encrypted(chunk.content):
            # Already encrypted, but possibly by a build that still had a
            # generated tsvector — in which case the index holds base64 and this
            # is the pass that repairs it. Decrypt, re-derive, leave the
            # ciphertext alone.
            if chunk.search_vector is None or _looks_like_ciphertext_index(chunk):
                plain = cipher.decrypt(chunk.content, data_key, memory_id)
                chunk.search_vector = func.to_tsvector("english", plain)
                changed = True
            continue

        # **The index is derived before the content is encrypted**, from the
        # plaintext that is still in hand. This is the one place that ordering
        # matters, and getting it backwards indexes base64.
        chunk.search_vector = func.to_tsvector("english", chunk.content)
        # The same data key as its parent. A chunk is part of a memory, so a
        # shred that spared the chunks would not be a shred.
        chunk.content = cipher.encrypt(chunk.content, data_key, memory_id)
        changed = True

    await session.flush()
    return changed


async def decrypt_memory_content(
    session: AsyncSession, cipher: Cipher, memory_id: UUID, stored: str | None
) -> str | None:
    """Read one memory's content back, whatever state it is stored in."""
    if stored is None or not is_encrypted(stored):
        return stored
    key = await KeyStore(session, cipher).get(memory_id)
    if key is None:
        raise DecryptionFailed(
            f"Memory {memory_id} holds ciphertext but has no key row."
        )
    return cipher.decrypt(stored, key, memory_id)


async def rotate_master_key(
    session: AsyncSession, old: Cipher, new: Cipher
) -> tuple[int, int]:
    """Re-wrap every data key under a new master key.

    **Content is not touched.** That is the whole reason for envelope
    encryption: rotating the master key rewrites a few dozen bytes per memory
    rather than re-encrypting a corpus, so it is a second-long operation that
    can be done often instead of an hour-long one that never is.

    Destroyed keys are skipped rather than failed on. A shredded memory has no
    key to re-wrap, and a rotation that stopped at the first one would be
    impossible to complete on any system where a deletion had ever happened.

    Returns `(rewrapped, skipped)`.
    """
    rows = (
        (await session.execute(select(models.MemoryKey))).scalars().all()
    )
    rewrapped = skipped = 0
    for row in rows:
        if row.wrapped_key is None:
            skipped += 1
            continue
        data_key = old.unwrap(row.wrapped_key, row.memory_id)
        row.wrapped_key = new.wrap(data_key, row.memory_id)
        rewrapped += 1
    await session.flush()
    logger.info("encryption.rotated", rewrapped=rewrapped, skipped=skipped)
    return rewrapped, skipped


def _looks_like_ciphertext_index(chunk: models.MemoryChunk) -> bool:
    """Whether this chunk's tsvector was built from ciphertext.

    The marker is the envelope prefix: `to_tsvector` on an `enc:v1:…` string
    produces a lexeme for `enc` and one for `v1`, and no real chunk of English
    contains both next to a wall of base64. Cheap, and wrong only in the
    direction of re-deriving an index that was already fine.
    """
    vector = chunk.search_vector
    return bool(vector) and "'enc'" in str(vector)
