"""chunk metadata

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-09

The chunker has always recorded which definition a span falls inside, so that a
citation can say *which function* the matched text belongs to. That metadata
lived on the in-memory `TextChunk` and was thrown away when the row was written,
which meant it was unavailable at exactly the moment something needs it — query
time.

`NOT NULL DEFAULT '{}'` rather than nullable: "the chunker recorded nothing about
this span" and "nobody has looked" are the same state here, and an empty object
says so without every reader having to handle a null. Existing rows adopt the
default, so this backfills without a rewrite pass. They keep an empty object
until they are re-chunked, which is honest — the metadata was never captured for
them and inventing it now would be a fabrication.

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "memory_chunks",
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("memory_chunks", "metadata")
