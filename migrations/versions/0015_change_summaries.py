"""cache a generated description of what changed between two versions

Revision ID: 0015
Revises: 0014
Create Date: 2026-08-12

A version pair is immutable. Both memories are rows that are never updated after
they are written — M1.1 made that true of the whole projection — so the diff
between them is fixed forever, and so is any description of it. That makes this
the one cache in the system that can never go stale for the reason caches usually
do, and it is the reason the summary is worth storing at all: the alternative is
paying a model every time somebody opens a memory's history.

Keyed on the *pair* rather than on the newer version, because the useful diff is
not always against the immediate predecessor. "What changed between v1 and v4"
is a different question from three consecutive diffs, and both are cacheable.

`summarizer_version` follows the M1.4 chunker-version and M3.1 extractor-version
pattern, and is part of the key rather than a column beside it. A better prompt
should be a query — find the summaries carrying the old version, redo those —
rather than a corpus-wide DELETE that also throws away the ones nobody has
complained about.

The grounding result is stored beside the text and not recomputed on read. It
was produced against the diff *as the model saw it*, and a later reader
recomputing it against a diff rebuilt with different context settings would get a
different answer for the same stored summary.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0015"
down_revision: str | Sequence[str] | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "change_summaries",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("from_memory_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("to_memory_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("summarizer_version", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        # What `domain.grounding.check_summary` found, as it was found.
        sa.Column("grounded", sa.Boolean(), nullable=False),
        sa.Column(
            "unsupported_terms",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_change_summaries"),
        # Both cascade. A summary of a version that no longer exists describes a
        # diff nobody can reproduce, and keeping it would leave the cache
        # answering for content that is gone.
        sa.ForeignKeyConstraint(
            ["from_memory_id"],
            ["memories.id"],
            name="fk_change_summaries_from_memory_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["to_memory_id"],
            ["memories.id"],
            name="fk_change_summaries_to_memory_id",
            ondelete="CASCADE",
        ),
        # The cache key, and the natural key. Without it a concurrent pair of
        # requests for the same history writes the summary twice and every later
        # read has to choose one.
        sa.UniqueConstraint(
            "from_memory_id",
            "to_memory_id",
            "summarizer_version",
            name="uq_change_summaries_pair_version",
        ),
        # A pair is two different versions. A self-diff is empty by definition
        # and there is nothing for a model to describe.
        sa.CheckConstraint(
            "from_memory_id <> to_memory_id", name="ck_change_summaries_distinct_pair"
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("change_summaries")
