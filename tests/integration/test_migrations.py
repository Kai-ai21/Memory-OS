import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from tests.conftest import run_alembic

pytestmark = pytest.mark.integration

TABLES = frozenset(
    {"sources", "raw_artifacts", "ingestion_events", "memories", "memory_chunks"}
)


async def public_tables(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as connection:
        result = await connection.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        )
        return {row[0] for row in result}


async def test_migration_applies_to_an_empty_database_and_round_trips(
    engine: AsyncEngine,
    clean_database: None,
) -> None:
    """A first-ever install, and a downgrade that is not a stub.

    `clean_database` is required rather than incidental, and M10.4 is where that
    became visible. Three migrations — 0027, 0029 and 0030 — deliberately *refuse*
    to downgrade while rows exist that the older schema cannot hold: a chat source
    has no file to be re-read from, an upload has no path, and a purge event is the
    only record that a permanent deletion happened. Each raises rather than choosing
    between destroying data and leaving it invalid, which is right.

    That makes this test's outcome depend on what the previous test file left
    behind, which is exactly the order-dependent failure `clean_database` exists to
    prevent — and it failed for real: a new test file whose name sorts just above
    this one left a chat source, and the downgrade correctly refused. The
    refusals are the feature; the missing fixture was the bug.
    """
    # Start from empty, so this exercises a first-ever install and not just a
    # no-op against a database that already happens to be at head.
    run_alembic(command.downgrade, "base")
    assert TABLES.isdisjoint(await public_tables(engine))

    run_alembic(command.upgrade, "head")
    assert TABLES.issubset(await public_tables(engine))

    # And the downgrade actually works, rather than being a stub that raises.
    run_alembic(command.downgrade, "base")
    assert TABLES.isdisjoint(await public_tables(engine))

    run_alembic(command.upgrade, "head")
    assert TABLES.issubset(await public_tables(engine))


async def test_embedding_column_is_a_384_dimension_vector(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        result = await connection.execute(
            text(
                "SELECT format_type(a.atttypid, a.atttypmod) "
                "FROM pg_attribute a "
                "WHERE a.attrelid = 'memory_chunks'::regclass AND a.attname = 'embedding'"
            )
        )
        assert result.scalar_one() == "vector(384)"


async def test_the_embedding_column_has_an_hnsw_inner_product_index(
    engine: AsyncEngine,
) -> None:
    """Arrives in M1.6, after the pipeline that fills the column.

    `vector_ip_ops` is load-bearing: the embedder normalizes to unit length, so
    inner product and cosine similarity are the same number and inner product
    is the cheaper one. A different operator class here would not error — it
    would quietly rank by the wrong measure.
    """
    async with engine.connect() as connection:
        result = await connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE tablename = 'memory_chunks'")
        )
        definitions = [row[0] for row in result]

    (hnsw,) = [d for d in definitions if "embedding" in d]
    assert "USING hnsw" in hnsw
    assert "vector_ip_ops" in hnsw
    assert "m='16'" in hnsw
    assert "ef_construction='64'" in hnsw
