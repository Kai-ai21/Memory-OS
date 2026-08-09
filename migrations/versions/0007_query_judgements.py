"""query judgements

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-09

The first table in this schema that a machine cannot regenerate. Everything else
is either bytes observed at a source or something computed from them; this is a
person's opinion about a search result, and losing it means asking them again.

**No foreign key to `memories`, deliberately.** A replay deletes and recreates
every memory row with a fresh UUID, so a reference to `memories.id` either blocks
the rebuild or dies with it — measured: `TRUNCATE ... CASCADE` reports "truncate
cascades to" and empties the referencing table, and `DELETE FROM memories` takes
it via `ON DELETE CASCADE`. Either way a routine replay would destroy the golden
set. Identity is therefore `(source_name, external_key)`, the same natural key
`verify-replay` compares on; `memory_id` and `chunk_id` are snapshots of what the
system pointed at when the verdict was given.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | Sequence[str] | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "query_judgements",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        # The durable identity, stable across rebuilds.
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=False),
        # Snapshots of what the system said, not references to it.
        sa.Column("memory_id", sa.Uuid(), nullable=True),
        sa.Column("chunk_id", sa.Uuid(), nullable=True),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("rank_at_judgement", sa.Integer(), nullable=True),
        sa.Column("score_at_judgement", sa.REAL(), nullable=True),
        sa.Column(
            "filters",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "judged_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_query_judgements"),
        sa.UniqueConstraint(
            "query_text",
            "source_name",
            "external_key",
            name="uq_query_judgements_query_item",
        ),
        sa.CheckConstraint(
            "verdict IN ('relevant', 'not_relevant', 'missing')",
            name="ck_query_judgements_verdict",
        ),
        sa.CheckConstraint(
            "rank_at_judgement IS NULL OR rank_at_judgement >= 1",
            name="ck_query_judgements_rank_positive",
        ),
        sa.CheckConstraint(
            "length(btrim(query_text)) > 0", name="ck_query_judgements_query_text"
        ),
        # A 'missing' verdict names something that was not in the ranking, so a
        # rank on it is a contradiction — and would silently corrupt any recall
        # computed from this table.
        sa.CheckConstraint(
            "verdict <> 'missing' OR rank_at_judgement IS NULL",
            name="ck_query_judgements_missing_has_no_rank",
        ),
    )
    op.create_index("ix_query_judgements_query_text", "query_judgements", ["query_text"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_query_judgements_query_text", table_name="query_judgements")
    op.drop_table("query_judgements")
