"""Recall measurement for the approximate index.

The milestone's real deliverable. An ANN index trades recall for latency, and
the only way to know what it traded is to compare it against something
exhaustive. `search_exact` exists for this and nothing else.

The lesson this rehearses for Phase 2's evaluation harness is the ordering:
measure first, tune second. Tuning `ef_search` without this table is guessing.
"""

import asyncio
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import Embedder, SearchFilters, VectorStore

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RecallRow:
    ef_search: int
    queries: int
    mean_recall: float
    p50_ms: float
    p95_ms: float
    self_retrieval: float

    def as_row(self) -> str:
        return (
            f"{self.ef_search:>9} | {self.mean_recall:>13.3f} | "
            f"{self.p50_ms:>7.1f} | {self.p95_ms:>7.1f} | {self.self_retrieval:>9.3f}"
        )


async def sample_query_texts(
    session_factory: async_sessionmaker[AsyncSession], count: int
) -> list[tuple[str, str]]:
    """Random embedded chunks, used as their own queries.

    Crude, and it needs no hand-labelling. It also doubles as a correctness
    check on the whole pipeline: a chunk used as its own query should retrieve
    itself first, and if it does not, something upstream is wrong in a way no
    structural test would catch.
    """
    stmt = (
        select(models.MemoryChunk.id, models.MemoryChunk.content)
        .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
        .where(
            models.MemoryChunk.embedding.is_not(None),
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
        )
        .order_by(func.random())
        .limit(count)
    )
    async with session_factory() as session:
        return [(str(row[0]), row[1]) for row in await session.execute(stmt)]


async def measure_recall(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    store: VectorStore,
    *,
    queries: int = 50,
    k: int = 10,
    ef_search_values: Sequence[int] = (40, 100, 200, 400),
    filters: SearchFilters | None = None,
) -> list[RecallRow]:
    resolved = filters or SearchFilters()
    samples = await sample_query_texts(session_factory, queries)
    if not samples:
        return []

    # Embedded once and reused across every ef_search value, so the table
    # compares index settings rather than model warm-up.
    #
    # `embed_query`, because these texts are standing in for queries and the
    # measurement is worth nothing if it exercises a path real searches do not.
    # It does change what `self_retrieval` means on an asymmetric model: the
    # query vector is no longer bit-identical to the stored passage vector, so
    # self@1 stops being a tautology and starts being a real assertion that the
    # two halves of the model's geometry line up. Recall is unaffected either
    # way — exact and approximate are compared using the same vector.
    vectors = await asyncio.to_thread(embedder.embed_query, [text for _, text in samples])

    truth: list[set[str]] = []
    for vector in vectors:
        exact = await store.search_exact(vector, k=k, filters=resolved)
        truth.append({str(chunk.chunk_id) for chunk in exact})

    rows: list[RecallRow] = []
    for ef_search in ef_search_values:
        recalls: list[float] = []
        latencies: list[float] = []
        self_hits = 0

        for (chunk_id, _), vector, expected in zip(samples, vectors, truth, strict=True):
            started = time.perf_counter()
            found = await store.search(vector, k=k, filters=resolved, ef_search=ef_search)
            latencies.append((time.perf_counter() - started) * 1000)

            got = {str(chunk.chunk_id) for chunk in found}
            recalls.append(len(got & expected) / len(expected) if expected else 1.0)
            if found and str(found[0].chunk_id) == chunk_id:
                self_hits += 1

        rows.append(
            RecallRow(
                ef_search=ef_search,
                queries=len(samples),
                mean_recall=statistics.fmean(recalls),
                p50_ms=_percentile(latencies, 50),
                p95_ms=_percentile(latencies, 95),
                self_retrieval=self_hits / len(samples),
            )
        )

    return rows


def format_table(rows: list[RecallRow], k: int) -> str:
    header = (
        f"{'ef_search':>9} | {f'mean recall@{k}':>13} | "
        f"{'p50 ms':>7} | {'p95 ms':>7} | {'self@1':>9}"
    )
    rule = "-" * len(header)
    return "\n".join([header, rule, *(row.as_row() for row in rows)])


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = round((percentile / 100) * (len(ordered) - 1))
    return ordered[min(len(ordered) - 1, max(0, position))]
