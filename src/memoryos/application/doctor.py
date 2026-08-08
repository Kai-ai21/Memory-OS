"""Standing checks for the class of defect that caused M1.6.1.

The window misalignment was invisible: no error, no failing test, just
retrieval that was quietly worse than it looked. The startup assertion stops
that specific pair from drifting again; this reports the state of data already
written, which an assertion cannot see.
"""

from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import Embedder


@dataclass(slots=True)
class Finding:
    check: str
    count: int
    detail: str
    examples: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return self.count == 0


@dataclass(slots=True)
class DoctorReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return all(finding.healthy for finding in self.findings)


async def run_doctor(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    *,
    sample_limit: int = 5,
) -> DoctorReport:
    findings: list[Finding] = []
    window = embedder.max_sequence_tokens

    async with session_factory() as session:
        # Counting every chunk with the real tokenizer would mean tokenizing the
        # whole corpus, so the stored token_count is used as the filter and the
        # tokenizer only confirms the suspects.
        candidates = (
            await session.execute(
                select(models.MemoryChunk.id, models.MemoryChunk.content)
                .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
                .where(
                    models.Memory.is_current.is_(True),
                    models.MemoryChunk.token_count >= window // 2,
                )
            )
        ).all()

        oversized = [
            str(chunk_id)
            for chunk_id, content in candidates
            if embedder.count_tokens(content) > window
        ]
        findings.append(
            Finding(
                check="chunks_over_model_window",
                count=len(oversized),
                detail=(
                    f"chunks whose text exceeds {window} model tokens; everything "
                    f"past the window is discarded before embedding"
                ),
                examples=oversized[:sample_limit],
            )
        )

        stale = (
            await session.execute(
                select(models.MemoryChunk.embedding_model, func.count())
                .where(
                    models.MemoryChunk.embedding.is_not(None),
                    models.MemoryChunk.embedding_model.is_distinct_from(
                        embedder.model_id
                    ),
                )
                .group_by(models.MemoryChunk.embedding_model)
            )
        ).all()
        findings.append(
            Finding(
                check="chunks_from_another_model",
                count=sum(count for _, count in stale),
                detail=(
                    f"chunks embedded by a model other than {embedder.model_id}; "
                    f"vectors from different models are not comparable"
                ),
                examples=[f"{model or '(none)'}: {count}" for model, count in stale],
            )
        )

        unembedded = (
            await session.execute(
                select(func.count())
                .select_from(models.MemoryChunk)
                .where(models.MemoryChunk.embedding.is_(None))
            )
        ).scalar_one()
        findings.append(
            Finding(
                check="chunks_without_embeddings",
                count=unembedded,
                detail="chunks with no vector; invisible to search",
            )
        )

        empty = (
            await session.execute(
                select(models.Memory.external_key)
                .outerjoin(
                    models.MemoryChunk,
                    models.MemoryChunk.memory_id == models.Memory.id,
                )
                .where(
                    models.Memory.is_current.is_(True),
                    models.Memory.deleted_at.is_(None),
                    models.Memory.content.is_not(None),
                    func.length(func.trim(models.Memory.content)) > 0,
                )
                .group_by(models.Memory.id, models.Memory.external_key)
                .having(func.count(models.MemoryChunk.id) == 0)
            )
        ).all()
        findings.append(
            Finding(
                check="memories_with_content_but_no_chunks",
                count=len(empty),
                detail="normalized text exists but produced no chunks",
                examples=[row[0] for row in empty[:sample_limit]],
            )
        )

    return DoctorReport(findings=findings)
