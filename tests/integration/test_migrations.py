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
) -> None:
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


async def test_no_index_exists_on_the_embedding_column(engine: AsyncEngine) -> None:
    # The HNSW index is M1.6, built once after bulk loading rather than
    # maintained through every insert.
    async with engine.connect() as connection:
        result = await connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE tablename = 'memory_chunks'")
        )
        definitions = [row[0] for row in result]
    assert not [d for d in definitions if "embedding" in d]
