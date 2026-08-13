"""The standing check for silently-degrading corpus conditions."""

from pathlib import Path

import pytest
from sqlalchemy import delete, select, update

from memoryos.adapters.db import models
from memoryos.application.doctor import DoctorReport, Finding, run_doctor
from tests.integration.conftest import Pipeline
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration


def finding(report: DoctorReport, check: str) -> Finding:
    return next(f for f in report.findings if f.check == check)


async def test_a_healthy_corpus_reports_healthy(pipeline: Pipeline) -> None:
    await pipeline.ingest()
    await pipeline.embed_all()

    report = await run_doctor(pipeline.sessions, pipeline.embedder)

    assert report.healthy, [(f.check, f.count) for f in report.findings]


async def test_unembedded_chunks_are_reported(pipeline: Pipeline) -> None:
    await pipeline.ingest()

    report = await run_doctor(pipeline.sessions, pipeline.embedder)

    assert not report.healthy
    assert finding(report, "chunks_without_embeddings").count > 0


async def test_chunks_from_another_model_are_reported(pipeline: Pipeline) -> None:
    await pipeline.ingest()
    await pipeline.embed_all()

    report = await run_doctor(pipeline.sessions, FakeEmbedder(model_id="fake/other@1"))

    stale = finding(report, "chunks_from_another_model")
    assert stale.count > 0
    assert any("deterministic" in example for example in stale.examples)


async def test_oversized_chunks_are_reported(pipeline: Pipeline) -> None:
    """The condition that caused this hotfix, as a standing check.

    A chunk longer than the window is not an error anywhere — the model simply
    reads the first N tokens. Nothing else in the system would ever mention it.
    """
    await pipeline.ingest()
    await pipeline.embed_all()

    # A model with a much smaller window than the chunks were sized for.
    narrow = FakeEmbedder(max_sequence_tokens=16)
    report = await run_doctor(pipeline.sessions, narrow)

    oversized = finding(report, "chunks_over_model_window")
    assert oversized.count > 0
    assert "discarded" in oversized.detail


async def test_a_corpus_nobody_has_extracted_is_reported_but_not_unhealthy(
    pipeline: Pipeline,
) -> None:
    """The condition two full replays created during M5.0 and nothing reported.

    A replay truncates the entity tables and does not rebuild them, which
    `application/replay.py` says and says to fix by re-running extraction —
    and until this check nothing anywhere reported whether anybody had. It
    matters because M5.1 narrows outcome candidates to memories sharing entities
    with a decision's evidence, and a corpus with no mentions makes that filter
    return nothing while looking exactly like a corpus where nothing shares one.

    Advisory rather than failing, and that is the same judgement `GraphStatus`
    makes about an unreachable Neo4j: extraction needs an API key, a keyless
    deployment is legitimate and fully working for everything Phase 1 and 2 do,
    and exiting non-zero here would report an optional capability as damage.
    """
    await pipeline.ingest()
    await pipeline.embed_all()

    report = await run_doctor(pipeline.sessions, pipeline.embedder)

    unextracted = finding(report, "memories_without_entity_extraction")
    assert unextracted.count > 0
    assert unextracted.advisory
    # Reported, and the report still passes.
    assert report.healthy


async def test_a_memory_with_content_but_no_chunks_is_reported(
    pipeline: Pipeline, tmp_path: Path
) -> None:
    await pipeline.ingest()
    await pipeline.embed_all()

    async with pipeline.sessions.begin() as session:
        memory_id = (
            await session.execute(select(models.Memory.id).limit(1))
        ).scalar_one()
        await session.execute(
            delete(models.MemoryChunk).where(models.MemoryChunk.memory_id == memory_id)
        )
        await session.execute(
            update(models.Memory)
            .where(models.Memory.id == memory_id)
            .values(content="text that produced no chunks")
        )

    report = await run_doctor(pipeline.sessions, pipeline.embedder)

    assert finding(report, "memories_with_content_but_no_chunks").count == 1
