"""chunk prefix_chars

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-09

`char_start`/`char_end` tile a memory's text contiguously, but `content` is
longer than that span for every chunk after the first: the chunker prepends an
overlap head borrowed from the previous chunk, and until now nothing recorded
how long it was. M1.4's docstring said the offsets index exactly into the stored
text, which holds only at ordinal 0. Anything computing a span from the
documented meaning highlights text the chunk does not claim, and does so
plausibly — the text it highlights is real text from the same document.

The column closes that: `content[prefix_chars:] == memory.content[char_start:char_end]`.

`NOT NULL DEFAULT 0` rather than nullable, and no backfill pass. Zero is wrong
for existing rows — 28% of stored chunk text is borrowed — but the length is not
recoverable from this migration, and the chunker version bumped in the same
change, so every existing row is stale and `memoryos rechunk` rewrites it with
the real value. A default that is honest about being provisional beats a
computed guess that looks authoritative.

The CHECK is the same shape as the other offset constraints: a negative prefix
is not a small error, it is an offset that indexes outside the text it claims.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | Sequence[str] | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "memory_chunks",
        sa.Column(
            "prefix_chars",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_memory_chunks_prefix_chars_non_negative",
        "memory_chunks",
        "prefix_chars >= 0",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "ck_memory_chunks_prefix_chars_non_negative", "memory_chunks", type_="check"
    )
    op.drop_column("memory_chunks", "prefix_chars")
