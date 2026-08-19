"""M11.2: per-memory data keys, so a deletion can be permanent.

One table. Content stays in the `TEXT` columns it is already in — encrypted
content is stored as a prefixed base64 envelope rather than as `BYTEA`, because
changing five columns' types would touch every mapper, query and test in the
system for a representation nobody reads by hand, and the prefix makes the
column self-describing in `psql` anyway.

**`memory_keys` is the deletion guarantee.** Phase 1 promised that memories can
be permanently deleted and an append-only log said otherwise; crypto-shredding
was the resolution designed in M1.1 and never built. This is it: destroy a few
dozen bytes and every copy of that memory's ciphertext — here, in a backup, on
a replica — becomes undecryptable at the same instant.

Deliberately *not* scoped by M11.1's policies. The parent row is, and this
reaches nothing else; a `user_id` here would be a second copy of a fact
`memories` already holds and free to disagree with it. It inherits isolation
through the foreign key: you cannot name a `memory_id` you cannot see.

Revision ID: 0033
Revises: 0032
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0033"
down_revision: str | None = "0032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_keys",
        sa.Column("memory_id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("wrapped_key", sa.LargeBinary(), nullable=True),
        sa.Column("algorithm", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("destroyed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.id"],
            name="fk_memory_keys_memory_id",
            ondelete="CASCADE",
        ),
        # A key is either present or destroyed, never both and never neither.
        sa.CheckConstraint(
            "(wrapped_key IS NULL) = (destroyed_at IS NOT NULL)",
            name="ck_memory_keys_destroyed_has_no_key",
        ),
        sa.CheckConstraint(
            "length(btrim(algorithm)) > 0", name="ck_memory_keys_algorithm_non_empty"
        ),
    )
    # M11.1 moved ownership of every table to the application role so that
    # `replay` can do its DDL; a table created later has to join it, or the
    # first swap-in after this migration fails on a table it does not own.
    op.execute("ALTER TABLE memory_keys OWNER TO memos_app")


def downgrade() -> None:
    op.drop_table("memory_keys")
