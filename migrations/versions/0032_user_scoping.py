"""M11.1: every row belongs to a user, and Postgres enforces it.

M11.0 put a login on the front door and left everything behind it owned by
nobody. This is the milestone that makes a second account safe.

**Enforcement is row-level security, not application code.** The argument is in
the README; the short version is that this codebase builds `select()` in
fifty-nine modules and has three repository classes, so "add the filter to the
base class" would mean rewriting the fifty-nine — and would still not bind the
CLI, the worker, or anybody at a `psql` prompt. A policy binds all of them.

**The column defaults to the session's user**, which is what makes this a
migration rather than a rewrite. Every `INSERT` in the application omits
`user_id` and gets the right one from `app.current_user_id`; nothing in
`application/` had to learn that scoping exists.

**Unset means nothing, in both directions.** `memos_current_user_id()` returns
NULL when the GUC is absent, so a `SELECT` matches no rows and an `INSERT` fails
the NOT NULL. A connection that forgets to identify itself sees an empty system
rather than everybody's.

**Order matters here and it is load-bearing.** The column is added nullable, the
existing rows are backfilled to the one existing account, and only then is the
NOT NULL applied and RLS enabled. Enabling the policy first would hide the rows
from the backfill that is meant to be updating them.

Revision ID: 0032
Revises: 0031
"""

import os
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0032"
down_revision: str | None = "0031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Every table that holds somebody's data. `users` and `sessions` are absent
# deliberately: they are the identity tables, and a login has to read a user
# row *before* it knows which user it is.
SCOPED_TABLES: tuple[str, ...] = (
    "sources",
    "memories",
    "memory_chunks",
    "ingestion_events",
    "entities",
    "entity_mentions",
    "entity_relationships",
    "decisions",
    "query_judgements",
    "chat_sessions",
    "chat_messages",
    "jobs",
    "patterns",
    "user_model_facets",
)


# Indexes that had to be rebuilt with `user_id` on the leading edge.
#
# **Leading, not trailing, and the difference is whether the index is usable at
# all.** Every query under RLS carries an implicit `user_id = $me`, so an index
# on `(external_key, user_id)` still has to scan every user's entries for that
# key and filter; `(user_id, external_key)` seeks straight to this user's slice.
# On a single-user database the two are indistinguishable, which is exactly why
# this is the kind of thing that ships wrong.
#
# `(name, definition)` — dropped and recreated, because Postgres cannot reorder
# an index's columns in place.
REBUILT_INDEXES: tuple[tuple[str, str], ...] = (
    ("ix_memories_occurred_at", "ON memories (user_id, occurred_at)"),
    ("ix_memories_ingested_at", "ON memories (user_id, ingested_at)"),
    (
        "ix_memories_entity_extractor_version",
        "ON memories (user_id, entity_extractor_version)",
    ),
    ("ix_memory_chunks_chunker_version", "ON memory_chunks (user_id, chunker_version)"),
    ("ix_memory_chunks_content_hash", "ON memory_chunks (user_id, content_hash)"),
    (
        "ix_ingestion_events_source_id_external_key",
        "ON ingestion_events (user_id, source_id, external_key)",
    ),
    ("ix_ingestion_events_recorded_at", "ON ingestion_events (user_id, recorded_at)"),
    ("ix_entities_canonical_name", "ON entities (user_id, canonical_name)"),
    ("ix_entities_merged_into", "ON entities (user_id, merged_into_id)"),
    ("ix_entity_mentions_entity", "ON entity_mentions (user_id, entity_id)"),
    (
        "ix_entity_mentions_memory_version",
        "ON entity_mentions (user_id, memory_id, extractor_version)",
    ),
    (
        "ix_entity_relationships_subject",
        "ON entity_relationships (user_id, subject_id, predicate)",
    ),
    (
        "ix_entity_relationships_object",
        "ON entity_relationships (user_id, object_id, predicate)",
    ),
    (
        "ix_entity_relationships_memory_version",
        "ON entity_relationships (user_id, memory_id, extractor_version)",
    ),
    ("ix_decisions_status_decided_at", "ON decisions (user_id, status, decided_at)"),
    ("ix_query_judgements_query_text", "ON query_judgements (user_id, query_text)"),
    (
        "ix_chat_sessions_live",
        "ON chat_sessions (user_id, last_activity) WHERE archived_at IS NULL",
    ),
    ("ix_chat_messages_created_at", "ON chat_messages (user_id, created_at)"),
    (
        "ix_chat_messages_external_key",
        "ON chat_messages (user_id, external_key) WHERE external_key IS NOT NULL",
    ),
    ("ix_chat_messages_session", "ON chat_messages (user_id, session_id, ordinal)"),
    (
        "ix_jobs_claim",
        "ON jobs (user_id, priority DESC, run_after) WHERE status = 'pending'",
    ),
    (
        "ix_jobs_lease",
        "ON jobs (user_id, lease_expires_at) WHERE status = 'running'",
    ),
    ("ix_jobs_status_type", "ON jobs (user_id, status, job_type)"),
    ("ix_patterns_kind", "ON patterns (user_id, kind)"),
    (
        "ix_user_model_facets_live",
        "ON user_model_facets (user_id, dimension) "
        "WHERE superseded_at IS NULL AND dismissed_at IS NULL",
    ),
    (
        "ix_user_model_facets_superseded_at",
        "ON user_model_facets (user_id, superseded_at) WHERE superseded_at IS NOT NULL",
    ),
)


# Uniques that could collide between two people, rebuilt with `user_id` first.
#
# **Not every unique needed this, and the ones left alone are left alone on
# purpose.** `uq_memory_chunks_memory_ordinal (memory_id, ordinal)` cannot
# collide across users because a memory already belongs to one; the same is true
# of `uq_entity_mentions_span`, `uq_entity_relationships_assertion` and
# `uq_chat_messages_session_ordinal`, each of which is keyed through a row that
# is itself scoped. Prefixing those would add a column to an index for the
# appearance of consistency and buy nothing.
#
# `uq_ingestion_events_seq` is deliberately still global: `seq` is the log's
# total order, and it is a `BIGSERIAL` shared by everybody. Two users' events
# interleave in one sequence, which is what makes "replay in order" mean
# anything at all.
# Rebuilt as table *constraints*. These are `UniqueConstraint` in the ORM, and
# a unique index is not the same object to Alembic even when it is the same
# thing to Postgres — declaring one and creating the other is a permanent
# `alembic check` failure.
REBUILT_UNIQUE_CONSTRAINTS: tuple[tuple[str, str, str], ...] = (
    # (name, table, the column list as SQL)
    ("uq_sources_kind_name", "sources", "(user_id, kind, name)"),
    (
        "uq_memories_source_key_version",
        "memories",
        "(user_id, source_id, external_key, version)",
    ),
    ("uq_entities_canonical_type", "entities", "(user_id, canonical_name, type)"),
    (
        "uq_query_judgements_query_item",
        "query_judgements",
        "NULLS NOT DISTINCT (user_id, query_text, source_name, external_key, chunk_ordinal)",
    ),
    ("uq_patterns_detector_subject", "patterns", "(user_id, detector, subject_key)"),
)

# Rebuilt as unique *indexes*, because every one of them is partial and a
# partial unique cannot be expressed as a table constraint at all.
REBUILT_UNIQUE_INDEXES: tuple[tuple[str, str], ...] = (
    (
        "uq_memories_current_version",
        "ON memories (user_id, source_id, external_key) WHERE is_current",
    ),
    (
        "uq_jobs_dedupe",
        "ON jobs (user_id, job_type, dedupe_key) WHERE dedupe_key IS NOT NULL "
        "AND status IN ('pending', 'running')",
    ),
    (
        "uq_user_model_facets_live_subject",
        "ON user_model_facets (user_id, detector, subject_key) "
        "WHERE origin = 'derived' AND superseded_at IS NULL AND dismissed_at IS NULL",
    ),
)


def upgrade() -> None:
    connection = op.get_bind()

    # **The application role, and the reason this migration creates one.**
    #
    # Row-level security is skipped entirely for superusers and for any role
    # holding BYPASSRLS, and `FORCE ROW LEVEL SECURITY` does not change that —
    # it only removes the *owner's* exemption. Until this migration the
    # application connected as `memos`, which `docker-compose` creates as a
    # superuser, so every policy below would have been enabled, listed in
    # `pg_policies`, and enforcing precisely nothing. That is the failure mode
    # this milestone is most likely to ship with, because everything looks
    # right from the outside.
    #
    # So: a role with LOGIN and DML and nothing else. Migrations keep running
    # as the owner (`database_admin_url`); the application, the CLI and the
    # worker all connect as this one (`database_url`).
    password = os.environ.get("MEMOS_APP_DB_PASSWORD", "memos_app")
    exists = connection.execute(
        sa.text("SELECT 1 FROM pg_roles WHERE rolname = 'memos_app'")
    ).scalar()
    if not exists:
        # Interpolated rather than bound, because `CREATE ROLE` takes no
        # parameters — the password is part of the statement text. Quoted with
        # the doubling Postgres expects so a password containing a quote is a
        # password rather than a syntax error.
        quoted = password.replace("'", "''")
        op.execute(f"CREATE ROLE memos_app LOGIN PASSWORD '{quoted}'")
    op.execute("ALTER ROLE memos_app NOSUPERUSER NOBYPASSRLS")
    # CREATE as well as USAGE: swapping a rebuilt table in is
    # `ALTER TABLE ... SET SCHEMA public`, which the schema must permit.
    op.execute("GRANT USAGE, CREATE ON SCHEMA public TO memos_app")
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES "
        "ON ALL TABLES IN SCHEMA public TO memos_app"
    )
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO memos_app")
    # So tables added by later migrations are reachable without another GRANT.
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES ON TABLES TO memos_app"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT USAGE, SELECT ON SEQUENCES TO memos_app"
    )
    # Replay builds its corpus in a `memoryos_shadow` schema and swaps it in, so
    # the application role needs to be able to create one. `CREATE ON DATABASE`
    # is schema creation and nothing else — it does not touch the policies, and
    # a role that can make a schema still cannot see another user's rows.
    database = connection.execute(sa.text("SELECT current_database()")).scalar_one()
    op.execute(f'GRANT CREATE ON DATABASE "{database}" TO memos_app')

    # **Ownership moves to `memos_app`, and `FORCE` is what makes that safe.**
    #
    # `replay` is an application command that does DDL: it drops inbound foreign
    # keys, builds a corpus in a shadow schema and swaps the tables in. All of
    # that requires ownership, and routing it through the admin connection would
    # mean the one command that rewrites every row runs with the policies turned
    # off — the exact opposite of what this migration is for.
    #
    # An owner is normally exempt from its own policies, which would make this
    # pointless. `FORCE ROW LEVEL SECURITY`, applied below, removes that
    # exemption: `memos_app` owns these tables and is still filtered by them.
    #
    # What this does concede, stated rather than buried: a process holding the
    # application's credentials could `ALTER TABLE ... NO FORCE ROW LEVEL
    # SECURITY` and turn its own scoping off. That is a smaller boundary than
    # "nobody can bypass it" and a real one — it is the boundary between
    # *forgotten code* and *deliberate action*, and forgotten code is what this
    # milestone is defending against.
    tables = connection.execute(
        sa.text(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
            "ORDER BY tablename"
        )
    ).scalars().all()
    for table in tables:
        op.execute(f'ALTER TABLE public."{table}" OWNER TO memos_app')
    sequences = connection.execute(
        sa.text(
            "SELECT sequencename FROM pg_sequences WHERE schemaname = 'public' "
            "ORDER BY sequencename"
        )
    ).scalars().all()
    for sequence in sequences:
        op.execute(f'ALTER SEQUENCE public."{sequence}" OWNER TO memos_app')

    # `NULLIF` because an unset GUC reads as the empty string, and `''::uuid`
    # raises rather than returning NULL — which would turn "this connection did
    # not say who it is" into an error at a random query instead of an empty
    # result. STABLE so the planner may call it once per statement rather than
    # once per row.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION memos_current_user_id() RETURNS uuid
        LANGUAGE sql STABLE AS $$
          SELECT NULLIF(current_setting('app.current_user_id', true), '')::uuid
        $$
        """
    )

    owner = connection.execute(
        sa.text("SELECT id FROM users ORDER BY created_at LIMIT 1")
    ).scalar()

    if owner is None:
        populated = [
            table
            for table in SCOPED_TABLES
            if connection.execute(
                sa.text(f"SELECT EXISTS (SELECT 1 FROM {table})")
            ).scalar()
        ]
        if populated:
            raise RuntimeError(
                "M11.1 gives every row an owner, and this database has rows in "
                f"{', '.join(populated)} but no account to attribute them to. "
                "Run `memoryos auth create-user --email you@example.com` first, "
                "then re-run this migration."
            )

    for table in SCOPED_TABLES:
        op.add_column(table, sa.Column("user_id", sa.Uuid(as_uuid=True), nullable=True))
        if owner is not None:
            op.execute(sa.text(f"UPDATE {table} SET user_id = :owner").bindparams(owner=owner))
        op.alter_column(table, "user_id", nullable=False)
        op.create_foreign_key(
            f"fk_{table}_user_id", table, "users", ["user_id"], ["id"], ondelete="CASCADE"
        )
        # The default is what lets fifty-nine modules of `INSERT` stay unchanged.
        op.execute(f"ALTER TABLE {table} ALTER COLUMN user_id SET DEFAULT memos_current_user_id()")

    for name, definition in REBUILT_INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {name}")
        op.execute(f"CREATE INDEX {name} {definition}")

    for name, table, columns in REBUILT_UNIQUE_CONSTRAINTS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}")
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} UNIQUE {columns}")

    for name, definition in REBUILT_UNIQUE_INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {name}")
        op.execute(f"CREATE UNIQUE INDEX {name} {definition}")

    for table in SCOPED_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # **`FORCE` is the line that makes this real.** Without it the policy is
        # skipped for the table's owner, and the application connects as the
        # owner — so RLS would be enabled, visibly, and enforce nothing.
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table}_user_isolation ON {table}
              USING (user_id = memos_current_user_id())
              WITH CHECK (user_id = memos_current_user_id())
            """
        )


def downgrade() -> None:
    for table in SCOPED_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {table}_user_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")

    for name, table, _columns in REBUILT_UNIQUE_CONSTRAINTS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}")
    for name, _definition in REBUILT_UNIQUE_INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {name}")
    for name, _definition in REBUILT_INDEXES:
        op.execute(f"DROP INDEX IF EXISTS {name}")

    for table in SCOPED_TABLES:
        op.drop_constraint(f"fk_{table}_user_id", table, type_="foreignkey")
        op.drop_column(table, "user_id")

    op.execute("DROP FUNCTION IF EXISTS memos_current_user_id()")
    # The role is deliberately left in place. Dropping it would fail while
    # anything still holds privileges granted to it, and a downgrade that
    # removes a login somebody's connection string names is a worse outcome
    # than an unused role.

    # The pre-M11.1 shapes, so a downgrade leaves a working database rather than
    # one missing half its indexes.
    op.execute("ALTER TABLE sources ADD CONSTRAINT uq_sources_kind_name UNIQUE (kind, name)")
    op.execute(
        "ALTER TABLE memories ADD CONSTRAINT uq_memories_source_key_version "
        "UNIQUE (source_id, external_key, version)"
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_memories_current_version "
        "ON memories (source_id, external_key) WHERE is_current"
    )
    op.execute(
        "ALTER TABLE entities ADD CONSTRAINT uq_entities_canonical_type "
        "UNIQUE (canonical_name, type)"
    )
    op.execute(
        "ALTER TABLE query_judgements ADD CONSTRAINT uq_query_judgements_query_item "
        "UNIQUE NULLS NOT DISTINCT (query_text, source_name, external_key, chunk_ordinal)"
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_jobs_dedupe ON jobs (job_type, dedupe_key) "
        "WHERE dedupe_key IS NOT NULL AND status IN ('pending', 'running')"
    )
    op.execute(
        "ALTER TABLE patterns ADD CONSTRAINT uq_patterns_detector_subject "
        "UNIQUE (detector, subject_key)"
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_user_model_facets_live_subject ON user_model_facets "
        "(detector, subject_key) WHERE origin = 'derived' AND superseded_at IS NULL "
        "AND dismissed_at IS NULL"
    )
    op.execute("CREATE INDEX ix_memories_occurred_at ON memories (occurred_at)")
    op.execute("CREATE INDEX ix_memories_ingested_at ON memories (ingested_at)")
    op.execute(
        "CREATE INDEX ix_memories_entity_extractor_version ON memories (entity_extractor_version)"
    )
    op.execute("CREATE INDEX ix_memory_chunks_chunker_version ON memory_chunks (chunker_version)")
    op.execute("CREATE INDEX ix_memory_chunks_content_hash ON memory_chunks (content_hash)")
    op.execute(
        "CREATE INDEX ix_ingestion_events_source_id_external_key "
        "ON ingestion_events (source_id, external_key)"
    )
    op.execute("CREATE INDEX ix_ingestion_events_recorded_at ON ingestion_events (recorded_at)")
    op.execute("CREATE INDEX ix_entities_canonical_name ON entities (canonical_name)")
    op.execute("CREATE INDEX ix_entities_merged_into ON entities (merged_into_id)")
    op.execute("CREATE INDEX ix_entity_mentions_entity ON entity_mentions (entity_id)")
    op.execute(
        "CREATE INDEX ix_entity_mentions_memory_version "
        "ON entity_mentions (memory_id, extractor_version)"
    )
    op.execute(
        "CREATE INDEX ix_entity_relationships_subject "
        "ON entity_relationships (subject_id, predicate)"
    )
    op.execute(
        "CREATE INDEX ix_entity_relationships_object "
        "ON entity_relationships (object_id, predicate)"
    )
    op.execute(
        "CREATE INDEX ix_entity_relationships_memory_version "
        "ON entity_relationships (memory_id, extractor_version)"
    )
    op.execute("CREATE INDEX ix_decisions_status_decided_at ON decisions (status, decided_at)")
    op.execute("CREATE INDEX ix_query_judgements_query_text ON query_judgements (query_text)")
    op.execute(
        "CREATE INDEX ix_chat_sessions_live ON chat_sessions (last_activity) "
        "WHERE archived_at IS NULL"
    )
    op.execute("CREATE INDEX ix_chat_messages_created_at ON chat_messages (created_at)")
    op.execute(
        "CREATE INDEX ix_chat_messages_external_key ON chat_messages (external_key) "
        "WHERE external_key IS NOT NULL"
    )
    op.execute("CREATE INDEX ix_chat_messages_session ON chat_messages (session_id, ordinal)")
    op.execute(
        "CREATE INDEX ix_jobs_claim ON jobs (priority DESC, run_after) WHERE status = 'pending'"
    )
    op.execute("CREATE INDEX ix_jobs_lease ON jobs (lease_expires_at) WHERE status = 'running'")
    op.execute("CREATE INDEX ix_jobs_status_type ON jobs (status, job_type)")
    op.execute("CREATE INDEX ix_patterns_kind ON patterns (kind)")
    op.execute(
        "CREATE INDEX ix_user_model_facets_live ON user_model_facets (dimension) "
        "WHERE superseded_at IS NULL AND dismissed_at IS NULL"
    )
    op.execute(
        "CREATE INDEX ix_user_model_facets_superseded_at ON user_model_facets (superseded_at) "
        "WHERE superseded_at IS NOT NULL"
    )
