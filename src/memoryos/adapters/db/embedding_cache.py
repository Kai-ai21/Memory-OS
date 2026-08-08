"""Postgres-backed embedding cache.

Owns its transactions, like the job queue and for the same reason: a cache
write must not be entangled with whatever the caller happens to be doing.
"""

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import CacheEntry, EmbeddingCache
from memoryos.domain.values import ContentHash


def cache_key_for(model_id: str, text: str) -> str:
    """The key for one (model, text) pair.

    The model identity is in the key because a vector is only meaningful in the
    coordinate system that produced it. Keying on text alone would let a model
    upgrade quietly reuse the old model's vectors — no error, just two
    incompatible geometries in one index, with similarity scores between them
    that are arithmetically fine and semantically nonsense.

    The null byte separates the two fields so that a text ending in the model
    id cannot collide with a different pairing that happens to concatenate the
    same way.
    """
    return ContentHash.of(f"{model_id}\0{text}".encode()).value


class PostgresEmbeddingCache(EmbeddingCache):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        stmt = select(
            models.EmbeddingCacheEntry.cache_key, models.EmbeddingCacheEntry.embedding
        ).where(models.EmbeddingCacheEntry.cache_key.in_(list(keys)))

        async with self._sessions() as session:
            rows = await session.execute(stmt)
            return {key: list(embedding) for key, embedding in rows}

    async def put_many(self, entries: Sequence[CacheEntry]) -> None:
        if not entries:
            return

        # DO NOTHING rather than DO UPDATE: the key is a hash of the model and
        # the text, so a conflicting row already holds exactly these bytes.
        # Two workers racing on the same chunk is a no-op, not a conflict.
        stmt = insert(models.EmbeddingCacheEntry).values(
            [
                {
                    "cache_key": entry.cache_key,
                    "model_id": entry.model_id,
                    "dimension": entry.dimension,
                    "embedding": entry.embedding,
                }
                for entry in entries
            ]
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=["cache_key"])

        async with self._sessions.begin() as session:
            await session.execute(stmt)
