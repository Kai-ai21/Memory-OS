"""A separate Postgres schema to rebuild the derived tables in.

## Why not the swap this was supposed to be

The obvious design is to build everything in `memoryos_shadow` and then rename
the schemas:

    BEGIN;
    ALTER SCHEMA public RENAME TO memoryos_old;
    ALTER SCHEMA memoryos_shadow RENAME TO public;
    COMMIT;

That does not work here, and the reason is worth writing down because it looks
like it works. Postgres stores every object reference by OID, not by name, so
renaming a schema moves no references. Measured on this database:

    -- shadow table with source_id REFERENCES public.sources(id)
    -- before the swap: references public.sources
    -- after  the swap: references memoryos_old.sources

and the new `public` contains **no** `sources` table at all, because the source
of truth is deliberately never copied into the workspace. So the swap yields a
live schema that is missing half its tables while its foreign keys reach
backwards into the schema you were about to drop. Dropping that schema then
either fails or, with CASCADE, takes the constraints and the referenced data
with it. Exactly the "appears to work but leaves dangling references" outcome —
and count-based checks would all pass.

## What this does instead

Build in a real separate schema, then move the tables into `public` one at a
time inside a single transaction:

    BEGIN;
    DROP TABLE public.memory_chunks;  -- child first
    DROP TABLE public.memories;
    ALTER TABLE memoryos_shadow.memories SET SCHEMA public;
    ALTER TABLE memoryos_shadow.memory_chunks SET SCHEMA public;
    COMMIT;

`SET SCHEMA` moves the table itself, so its foreign keys keep pointing at
`public.sources` — verified, not assumed. Constraint and index names come across
unchanged, and they can be the canonical ones because names are unique per
schema rather than per database: the shadow `memories` can carry `pk_memories`
while the live one still exists. That is what keeps the schema identical to the
models afterwards, so `alembic check` stays clean and the next replay works.

Postgres has transactional DDL, so a failure anywhere in that block leaves the
live tables exactly as they were.

## How the rebuild lands in the workspace

`search_path`, set on every connection of a dedicated engine. The ORM models
carry no schema, so unqualified `memories` resolves to `memoryos_shadow.memories`
while `sources` — which does not exist there — falls through to `public.sources`.
The pipeline needs no changes at all, which is the whole reason for choosing a
schema over suffixed table names in one schema: with suffixes, every model and
every query would need a parallel definition.
"""

from collections.abc import Sequence

import structlog
from sqlalchemy import MetaData, Table, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.schema import (
    AddConstraint,
    CreateSchema,
    DropSchema,
    ForeignKeyConstraint,
)

from memoryos.adapters.db.models import Base
from memoryos.application.ports import ShadowWorkspace
from memoryos.application.replay import (
    CACHE_TABLE,
    SCOPED_TABLES,
    SHADOW_TABLES,
    SOURCE_OF_TRUTH_TABLES,
    USER_AUTHORED_TABLES,
)

logger = structlog.get_logger(__name__)

SHADOW_SCHEMA = "memoryos_shadow"
LIVE_SCHEMA = "public"

# Indexes that are built after the data is loaded rather than maintained through
# every insert. Only the vector index qualifies: it is the one whose build
# strategy changes its quality, not just its cost.
_DEFERRED_INDEXES = frozenset({"ix_memory_chunks_embedding_hnsw"})


class UnclassifiedTable(RuntimeError):
    """A table the replay module has not said is derived or source of truth."""


class PostgresShadowSchema(ShadowWorkspace):
    def __init__(self, url: str, *, echo: bool = False) -> None:
        self._url = url
        self._echo = echo
        # Two engines: one on the default search path for the DDL that moves
        # tables between schemas, one pinned to the workspace for the rebuild.
        self._engine: AsyncEngine | None = None
        self._shadow_engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[AsyncSession] | None = None
        self._metadata = _shadow_metadata()

    async def create(self) -> None:
        """A fresh schema with the derived tables in it, and no data."""
        engine = await self._live_engine()
        async with engine.begin() as connection:
            # Dropped first: a workspace left behind by an interrupted run holds
            # a partial rebuild, and silently continuing into it would produce a
            # corpus that is half of one replay and half of another.
            await connection.execute(DropSchema(SHADOW_SCHEMA, cascade=True, if_exists=True))
            await connection.execute(CreateSchema(SHADOW_SCHEMA))
            await connection.run_sync(self._create_tables)
        logger.info("shadow.created", schema=SHADOW_SCHEMA)

    async def sessions(self) -> async_sessionmaker[AsyncSession]:
        """Sessions whose unqualified table names resolve to the workspace.

        A dedicated engine rather than a `SET search_path` on borrowed
        connections: a session-level SET rides a pooled connection into every
        later query on it, which is how a replay would end up writing the live
        tables long after it finished.
        """
        if self._sessions is None:
            engine = create_async_engine(
                self._url,
                echo=self._echo,
                pool_pre_ping=True,
                connect_args={
                    "server_settings": {
                        "search_path": f"{SHADOW_SCHEMA},{LIVE_SCHEMA}"
                    }
                },
            )
            self._shadow_engine = engine
            self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        return self._sessions

    async def build_indexes(self) -> None:
        engine = await self._live_engine()
        async with engine.begin() as connection:
            for table in _shadow_only(self._metadata):
                for index in table.indexes:
                    if index.name in _DEFERRED_INDEXES:
                        await connection.run_sync(index.create)
        logger.info("shadow.indexes_built", indexes=sorted(_DEFERRED_INDEXES))

    async def swap_in(self) -> None:
        """Replace the live derived tables with the workspace's, in one transaction.

        The inbound foreign keys are lifted off and put back around the swap.
        Until M5.0 there were none, and the drops below worked because nothing
        outside the derived set referenced anything inside it. `decision_evidence`
        is the first table to break that: it is user-authored, it is not copied
        into the workspace, and it holds an ON DELETE CASCADE reference into
        `memories` — so `DROP TABLE public.memories` fails on a dependency that
        has nothing to do with this rebuild.

        `DROP TABLE ... CASCADE` would make the error go away and take the
        constraints with it, silently, leaving a live schema that no longer
        matches the models and an `alembic check` that fails on the next run.
        Dropping them by name and re-adding them by definition keeps the schema
        exactly where it was, and fails loudly if a constraint cannot be
        restored.

        Re-adding validates, so the referencing rows must already be gone.
        `ReplayCorpus._preserve_evidence` empties that table before the rebuild
        starts and re-links it from natural keys afterwards, which is the only
        ordering in which both halves are true: the constraint comes back, and
        the decisions keep their provenance.
        """
        inbound = _inbound_foreign_keys()
        engine = await self._live_engine()
        async with engine.begin() as connection:
            for table, constraint in inbound:
                await connection.execute(
                    text(
                        f'ALTER TABLE {LIVE_SCHEMA}."{table}" '
                        f'DROP CONSTRAINT IF EXISTS "{constraint.name}"'
                    )
                )
            # Child before parent, so no foreign key is left pointing at a table
            # that is going away.
            for name in SHADOW_TABLES:
                await connection.execute(
                    text(f'DROP TABLE IF EXISTS {LIVE_SCHEMA}."{name}"')
                )
            # Parent before child on the way in, for the same reason in reverse.
            for name in reversed(SHADOW_TABLES):
                await connection.execute(
                    text(f'ALTER TABLE {SHADOW_SCHEMA}."{name}" SET SCHEMA {LIVE_SCHEMA}')
                )
            # **M11.1: the policies do not travel with the table.** `SET SCHEMA`
            # moves a table built in the shadow schema, and that table was never
            # given row-level security — so a swap-in would replace fourteen
            # protected tables with fourteen unprotected ones, and every user
            # would silently be able to read every other user's rebuilt corpus.
            # Nothing would error; the isolation would just be gone.
            for name in SHADOW_TABLES:
                if name not in SCOPED_TABLES:
                    continue
                await connection.execute(
                    text(f'ALTER TABLE {LIVE_SCHEMA}."{name}" ENABLE ROW LEVEL SECURITY')
                )
                await connection.execute(
                    text(f'ALTER TABLE {LIVE_SCHEMA}."{name}" FORCE ROW LEVEL SECURITY')
                )
                await connection.execute(
                    text(
                        f'CREATE POLICY {name}_user_isolation ON {LIVE_SCHEMA}."{name}" '
                        "USING (user_id = memos_current_user_id()) "
                        "WITH CHECK (user_id = memos_current_user_id())"
                    )
                )
                await connection.execute(
                    text(
                        f'ALTER TABLE {LIVE_SCHEMA}."{name}" '
                        "ALTER COLUMN user_id SET DEFAULT memos_current_user_id()"
                    )
                )

            for _table, constraint in inbound:
                await connection.execute(AddConstraint(constraint))
            await connection.execute(DropSchema(SHADOW_SCHEMA, if_exists=True))
        logger.info(
            "shadow.swapped_in",
            tables=list(SHADOW_TABLES),
            constraints_restored=[constraint.name for _, constraint in inbound],
        )
        # Terminal: both pools go, rather than being left open for the lifetime
        # of whatever built this. A replay per hour otherwise accumulates a
        # connection pool per hour.
        await self.dispose()

    async def discard(self) -> None:
        """Drop the workspace. The live tables were never touched."""
        engine = await self._live_engine()
        async with engine.begin() as connection:
            await connection.execute(
                DropSchema(SHADOW_SCHEMA, cascade=True, if_exists=True)
            )
        logger.info("shadow.discarded", schema=SHADOW_SCHEMA)
        await self.dispose()

    async def dispose(self) -> None:
        """Release both pools. Idempotent, so the terminal paths can both call it."""
        await self._dispose_shadow()
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    def _create_tables(self, connection: Connection) -> None:
        """Create the shadow tables, minus the indexes that are deferred.

        The HNSW index is excluded here and built by `build_indexes` after the
        rows are in. Migration 0005 makes the same point: a graph built on an
        empty table and grown one insert at a time is both slower to produce and
        worse connected than one build over finished data.
        """
        tables = _shadow_only(self._metadata)
        deferred = [
            (table, index)
            for table in tables
            for index in list(table.indexes)
            if index.name in _DEFERRED_INDEXES
        ]
        for table, index in deferred:
            table.indexes.discard(index)
        try:
            # `tables=` explicitly: without it `create_all` would consider the
            # `public` definitions above and, on a database where one was
            # somehow absent, create it.
            self._metadata.create_all(connection, tables=tables)
        finally:
            for table, index in deferred:
                table.indexes.add(index)

    async def _live_engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(
                self._url, echo=self._echo, pool_pre_ping=True
            )
        return self._engine

    async def _dispose_shadow(self) -> None:
        if self._shadow_engine is not None:
            await self._shadow_engine.dispose()
            self._shadow_engine = None
        self._sessions = None


def _shadow_metadata() -> MetaData:
    """The derived tables, redefined in the shadow schema.

    Copied from `Base.metadata` rather than written out again, so the workspace
    cannot drift from the real schema — a shadow table missing a column would
    rebuild a corpus that silently lost it.

    Foreign keys need the care here. A derived table's reference to another
    derived table has to follow it into the shadow schema, or the rebuild would
    write chunks pointing at the *live* memories. A reference to a
    source-of-truth table has to stay in `public`, because that table is not
    copied and must not be.
    """
    shadow = MetaData(schema=SHADOW_SCHEMA)
    shadowed = set(SHADOW_TABLES)

    def referred_schema(
        table: Table,
        to_schema: str | None,
        constraint: ForeignKeyConstraint,
        referred_schema_name: str | None,
    ) -> str | None:
        target = constraint.referred_table.name
        if target in shadowed:
            return SHADOW_SCHEMA
        if target == CACHE_TABLE:
            # Shared with the live schema on purpose; see SHADOW_TABLES.
            return LIVE_SCHEMA
        if target in SOURCE_OF_TRUTH_TABLES:
            return LIVE_SCHEMA
        # M11.1. `user_id REFERENCES users` now appears on every derived table,
        # and `users` is user-authored: a replay never rebuilds it, so the
        # foreign key resolves to the live schema for exactly the same reason a
        # source-of-truth table does. Checked as a set rather than by naming
        # `users` here, because the next user-authored table something derived
        # references should not have to rediscover this.
        if target in USER_AUTHORED_TABLES:
            return LIVE_SCHEMA
        raise UnclassifiedTable(
            f"{table.name} references {target}, which is in neither the derived "
            f"nor the source-of-truth set; classify it in application/replay.py"
        )

    # The source-of-truth tables are copied in too, under `public`, purely so the
    # foreign keys above have something to resolve against. They are definitions,
    # not instructions: `_create_tables` only issues DDL for tables in the shadow
    # schema, so nothing here can create or alter a live table.
    for name in SOURCE_OF_TRUTH_TABLES:
        Base.metadata.tables[name].to_metadata(shadow, schema=LIVE_SCHEMA)

    # M11.1: `users` for the same reason, one set along. Every shadowed table
    # carries `user_id REFERENCES users`, so the definition has to be present
    # under `public` or the copy has nothing to resolve against — and the copy
    # is a definition rather than an instruction, exactly as above. Only the
    # tables something derived actually references are pulled in, which today
    # is `users` and not `sessions`.
    referenced_user_authored = {
        constraint.referred_table.name
        for name in SHADOW_TABLES
        for constraint in Base.metadata.tables[name].foreign_key_constraints
    } & USER_AUTHORED_TABLES
    for name in sorted(referenced_user_authored):
        Base.metadata.tables[name].to_metadata(shadow, schema=LIVE_SCHEMA)

    for name in SHADOW_TABLES:
        Base.metadata.tables[name].to_metadata(
            shadow, schema=SHADOW_SCHEMA, referred_schema_fn=referred_schema
        )
    return shadow


def _inbound_foreign_keys() -> list[tuple[str, ForeignKeyConstraint]]:
    """Constraints from tables the workspace does not own into tables it replaces.

    Read off the real metadata rather than listed by hand, for the reason
    `_shadow_metadata` is copied rather than written out: a hand-kept list goes
    stale the first time somebody adds a table, and the failure would be a
    `DROP TABLE` refusing in the middle of a swap.

    Returned in metadata order and applied in that order both ways, which is
    enough here because every one of these is a leaf: the referencing tables are
    user-authored and nothing references *them*.
    """
    shadowed = set(SHADOW_TABLES)
    found: list[tuple[str, ForeignKeyConstraint]] = []
    for table in Base.metadata.sorted_tables:
        if table.name in shadowed:
            continue
        for constraint in table.constraints:
            if not isinstance(constraint, ForeignKeyConstraint):
                continue
            if constraint.referred_table.name in shadowed:
                found.append((table.name, constraint))
    return found


def inbound_foreign_key_names() -> list[str]:
    """Exposed for the test that asserts the swap knows about them."""
    return [str(constraint.name) for _, constraint in _inbound_foreign_keys()]


def _shadow_only(metadata: MetaData) -> list[Table]:
    """The tables this workspace owns, in dependency order."""
    return [table for table in metadata.sorted_tables if table.schema == SHADOW_SCHEMA]


def shadow_table_names() -> Sequence[str]:
    """Exposed for tests that assert the workspace covers what it should."""
    return SHADOW_TABLES
