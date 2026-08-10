"""Filling the `importance` column that has been nullable and empty since M1.1.

It was left empty deliberately. The comment on the entity has said since then
that "a placeholder heuristic becomes load-bearing the moment anything
downstream trusts it" — so nothing wrote to it until something needed it and
could say what the number meant. That is now: `domain/signals.py` defines it as
a proxy over observable evidence, and this walks the corpus applying it.

**Not on the ingest path.** Two of the three inputs are properties of an item's
*history* rather than of the bytes just observed: version count only grows on
later syncs, and the freshness term decays with the clock. Computing it during
normalization would freeze both at the moment of first ingest and then never
correct them, which is worse than leaving the column null — a stale number is
indistinguishable from a fresh one downstream.

So it is a command somebody runs, and re-runs. Idempotent for a fixed corpus and
a fixed clock, and cheap enough to run after every sync.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import Select, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.signals import DEFAULT_HALF_LIFE_DAYS, importance_score

logger = structlog.get_logger(__name__)

# Rows updated per statement. The corpus fits in memory at this size, but the
# write is chunked anyway so that a corpus that does not still works.
BATCH = 500


@dataclass(frozen=True, slots=True)
class ImportanceReport:
    scanned: int
    updated: int
    unchanged: int
    minimum: float
    maximum: float
    mean: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "scanned": self.scanned,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "mean": round(self.mean, 4),
        }


def _evidence() -> Select[Any]:
    """Per current memory: how many chunks, how many versions, last edited when.

    `version_count` counts every row sharing the item's identity, including
    superseded ones — that history is exactly the evidence, and it is why this
    cannot be computed from the current row alone.
    """
    versions = (
        select(
            models.Memory.source_id.label("source_id"),
            models.Memory.external_key.label("external_key"),
            func.count().label("versions"),
            func.max(models.Memory.ingested_at).label("last_ingested_at"),
        )
        .group_by(models.Memory.source_id, models.Memory.external_key)
        .subquery()
    )
    chunks = (
        select(
            models.MemoryChunk.memory_id.label("memory_id"),
            func.count().label("chunks"),
        )
        .group_by(models.MemoryChunk.memory_id)
        .subquery()
    )

    return (
        select(
            models.Memory.id,
            func.coalesce(chunks.c.chunks, 0),
            func.coalesce(versions.c.versions, 1),
            # `occurred_at` is when the file was last written, which is the edit
            # this signal is about. `ingested_at` is when we noticed, and using
            # it would score a freshly cloned repository as uniformly urgent.
            func.coalesce(models.Memory.occurred_at, versions.c.last_ingested_at),
            models.Memory.importance,
        )
        .outerjoin(chunks, chunks.c.memory_id == models.Memory.id)
        .outerjoin(
            versions,
            (versions.c.source_id == models.Memory.source_id)
            & (versions.c.external_key == models.Memory.external_key),
        )
        .where(
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
        )
    )


async def recompute_importance(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    now: datetime,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> ImportanceReport:
    """Score every current memory and write the ones that changed.

    `now` is a parameter rather than read here, so a test can pin the clock and
    two runs a second apart can be compared.
    """
    async with session_factory() as session:
        rows = list(await session.execute(_evidence()))

    scores: list[tuple[object, float]] = []
    unchanged = 0
    for memory_id, chunks, versions, last_edited_at, existing in rows:
        score = importance_score(
            chunk_count=int(chunks),
            version_count=int(versions),
            last_edited_at=last_edited_at,
            now=now,
            half_life_days=half_life_days,
        )
        # Rounded before comparing and before writing. The column is REAL, so a
        # full-precision float round-trips to something slightly different and
        # every run would report every row as changed.
        rounded = round(score, 4)
        if existing is not None and abs(float(existing) - rounded) < 1e-4:
            unchanged += 1
            continue
        scores.append((memory_id, rounded))

    for start in range(0, len(scores), BATCH):
        async with session_factory.begin() as session:
            for memory_id, score in scores[start : start + BATCH]:
                await session.execute(
                    update(models.Memory)
                    .where(models.Memory.id == memory_id)
                    .values(importance=score)
                )

    written = [score for _, score in scores]
    all_scores = written or [float(row[4]) for row in rows if row[4] is not None]
    report = ImportanceReport(
        scanned=len(rows),
        updated=len(scores),
        unchanged=unchanged,
        minimum=min(all_scores, default=0.0),
        maximum=max(all_scores, default=0.0),
        mean=sum(all_scores) / len(all_scores) if all_scores else 0.0,
    )
    logger.info("importance.recomputed", **report.as_dict())
    return report
