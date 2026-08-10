"""chunk-level judgements

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-10

A memory-level verdict cannot say "right file, wrong chunk", and that is the
shape of the most interesting failure the corpus has: `why do we store two
timestamps` ranks `domain/entities.py` first, but on the chunk about
`last_sync_at`/`last_full_sync_at` rather than the one that explains `occurred_at`
against `ingested_at`. Scored per memory that query looks solved. It is not.

So the judged item gains an optional ordinal, and the identity of a judgement
becomes `(query_text, source_name, external_key, chunk_ordinal)`. NULL keeps its
old meaning: the verdict is about the memory, whichever chunk matched.

**`NULLS NOT DISTINCT` is load-bearing, not decoration.** Postgres treats NULLs
as distinct in a unique index by default, so under the default the memory-level
rows `(q, self, entities.py, NULL)` would no longer collide with each other and
`ON CONFLICT DO UPDATE` — the thing that makes re-judging replace rather than
append — would silently stop firing for every memory-level verdict in the table.
The golden set would then hold two contradictory opinions about the same pair,
which is exactly what the constraint was added in 0007 to prevent.

Widening a unique constraint can only ever be permissive, so no existing row can
conflict: every row currently in the table gets `chunk_ordinal = NULL` and keeps
the identity it already had.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | Sequence[str] | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "query_judgements",
        sa.Column("chunk_ordinal", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        "ck_query_judgements_chunk_ordinal_non_negative",
        "query_judgements",
        "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
    )
    op.drop_constraint(
        "uq_query_judgements_query_item", "query_judgements", type_="unique"
    )
    op.create_unique_constraint(
        "uq_query_judgements_query_item",
        "query_judgements",
        ["query_text", "source_name", "external_key", "chunk_ordinal"],
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Narrowing the key can conflict where widening could not: two rows differing
    # only in their ordinal become one identity. Chunk-level rows are the newer
    # data and the ones the old schema has no way to represent, so they go.
    op.execute(sa.text("DELETE FROM query_judgements WHERE chunk_ordinal IS NOT NULL"))
    op.drop_constraint(
        "uq_query_judgements_query_item", "query_judgements", type_="unique"
    )
    op.create_unique_constraint(
        "uq_query_judgements_query_item",
        "query_judgements",
        ["query_text", "source_name", "external_key"],
    )
    op.drop_constraint(
        "ck_query_judgements_chunk_ordinal_non_negative",
        "query_judgements",
        type_="check",
    )
    op.drop_column("query_judgements", "chunk_ordinal")
