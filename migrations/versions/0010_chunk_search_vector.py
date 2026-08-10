"""chunk search vector

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-10

The lexical half of retrieval. M2.0 measured where the semantic half fails and
the answer was unambiguous: `SKIP LOCKED` scores 0.000 — nothing relevant in ten
results — because an opaque SQL fragment has almost no meaning for a sentence
embedder to encode, while the clause itself sits in exactly one file. That is the
shape of query a term-frequency index answers trivially.

**A generated column, not a trigger.** Postgres recomputes it on every insert and
update, so it cannot drift from `content`. A trigger is a second definition of
the same derivation that somebody has to keep in agreement, and the failure mode
when it drifts is silent: search keeps working, on stale text.

`STORED` because the ratio is lopsided — this is read on every keyword query and
written once per chunk, and a `VIRTUAL` column would recompute `to_tsvector` for
every candidate row at query time. Postgres 17 does not offer `VIRTUAL` anyway;
the word is here because the choice is worth stating.

GIN rather than GiST. GIN is slower to build and materially faster to query, and
this index is built once per chunk and queried constantly. GiST's advantage —
cheap updates — is worth nothing here.

The `ALTER TABLE` rewrites the table to fill the column, so the corpus needs no
re-ingest: every existing chunk is vectorised as the migration runs.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | Sequence[str] | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "memory_chunks",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_memory_chunks_search",
        "memory_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_memory_chunks_search", table_name="memory_chunks")
    op.drop_column("memory_chunks", "search_vector")
