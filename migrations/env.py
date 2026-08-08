import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from memoryos.adapters.db.models import Base
from memoryos.config import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# The database URL comes from Settings, not from alembic.ini. One place reads
# the environment, and migrations cannot end up pointed at a different database
# than the application.
_settings = get_settings()


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it against a database."""
    context.configure(
        url=_settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Autogenerate is used here as a check that the hand-written migrations
        # and the models agree, so column type drift has to be visible.
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    # The URL is injected into the config mapping rather than via
    # `set_main_option`, which would run it through ConfigParser interpolation
    # and choke on a '%' in a password.
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _settings.database_url

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
