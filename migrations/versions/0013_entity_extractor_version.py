"""record which extractor ran over each memory

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-12

M3.1 keyed extraction's skip check on "does this memory have mentions at the
current extractor version". That is a different question from "has this memory
been extracted", and it answers it wrongly for every memory that legitimately
contains no entities: those write no mention rows, so they never satisfy the
check, so every run extracts them again — for real money, forever, and the
pending count never reaches zero.

Measured on this corpus before the fix: 56 memories processed, 34 with mentions,
and the queue barely moved between runs.

Recording the attempt is the only thing that distinguishes "not yet done" from
"done, found nothing". Backfilled from the mentions that already exist, so the
memories extracted before this migration are not extracted a second time.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0013"
down_revision: str | Sequence[str] | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "memories", sa.Column("entity_extractor_version", sa.Text(), nullable=True)
    )
    # Backfill from what is already known. A memory with mentions was extracted
    # by whatever version wrote them, and re-running it would be a second spend
    # for an identical result.
    op.execute(
        """
        UPDATE memories AS m
        SET entity_extractor_version = latest.extractor_version
        FROM (
            SELECT DISTINCT ON (memory_id) memory_id, extractor_version
            FROM entity_mentions
            ORDER BY memory_id, extracted_at DESC
        ) AS latest
        WHERE latest.memory_id = m.id
        """
    )
    # The extraction queue's predicate.
    op.create_index(
        "ix_memories_entity_extractor_version",
        "memories",
        ["entity_extractor_version"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_memories_entity_extractor_version", table_name="memories")
    op.drop_column("memories", "entity_extractor_version")
