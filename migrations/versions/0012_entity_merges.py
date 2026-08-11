"""entity merges

Revision ID: 0012
Revises: 0011
Create Date: 2026-08-11

Resolution, and the schema is shaped around one fact: **it is never perfect.**
M3.2's own report names the merges it got wrong. So nothing here deletes
anything, and every merge carries what it needs to be undone.

**`entities.merged_into_id`** marks a loser rather than removing it. The row
survives, its mentions move to the winner, and the pointer is what tells every
reader to look elsewhere. Deleting the loser would make a wrong merge
permanent — the name, the surface form, and the evidence for the decision all
disappear with the row, and no amount of re-running the resolver brings back an
entity whose only record was that row.

**`entity_merges.moved_mention_ids`** is what makes `unmerge` exact rather than
approximate. Repointing is destructive: once `entity_mentions.entity_id` says
"winner", nothing distinguishes the mentions that arrived from the loser from
the ones the winner always had. Recording the ids at merge time is the
difference between restoring the previous state and restoring something that
looks like it.

**One table for proposals and merges**, separated by `status`. A pending
candidate and an applied merge carry identical information and differ only in
whether somebody said yes; two tables would mean moving rows between them to
answer that question. The partial unique indexes carry the real invariants: an
entity can be merged away once at a time, and a pair is not proposed twice while
a proposal is outstanding — both predicated on status, so reverting a merge
frees the pair to be merged again rather than forbidding it forever.

`status` and `evidence` are additions to the milestone's column list. Status
because deriving "is this in force" from a nullable timestamp is unreadable in
every query that needs it; evidence because a reviewer being asked to judge a
pending merge cannot judge "0.87", only "cosine 0.87 between 'postgres' and
'postgresql'".
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: str | Sequence[str] | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("entities", sa.Column("merged_into_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_entities_merged_into_id",
        "entities",
        "entities",
        ["merged_into_id"],
        ["id"],
        # Not CASCADE: deleting a winner must not delete everything merged into
        # it. Those entities become active again, which is wrong and
        # recoverable, where cascading would be silent data loss.
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "ck_entities_not_merged_into_self",
        "entities",
        "merged_into_id IS NULL OR merged_into_id <> id",
    )
    # Every read of active entities filters on this column.
    op.create_index("ix_entities_merged_into", "entities", ["merged_into_id"])

    op.create_table(
        "entity_merges",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("winner_id", sa.Uuid(), nullable=False),
        sa.Column("loser_id", sa.Uuid(), nullable=False),
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.Text(), server_default=sa.text("'pending'"), nullable=False
        ),
        sa.Column("confidence", sa.REAL(), nullable=False),
        # What a reviewer reads to make the judgement.
        sa.Column("evidence", sa.Text(), server_default=sa.text("''"), nullable=False),
        # Exactly the mentions this merge moved, so an unmerge puts back those
        # and only those.
        sa.Column(
            "moved_mention_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "proposed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("merged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reverted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_entity_merges"),
        sa.ForeignKeyConstraint(
            ["winner_id"],
            ["entities.id"],
            name="fk_entity_merges_winner_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["loser_id"],
            ["entities.id"],
            name="fk_entity_merges_loser_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "strategy IN ('exact', 'embedding', 'alias', 'llm', 'manual')",
            name="ck_entity_merges_strategy",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'applied', 'reverted')",
            name="ck_entity_merges_status",
        ),
        sa.CheckConstraint(
            "confidence BETWEEN 0.0 AND 1.0", name="ck_entity_merges_confidence_range"
        ),
        # Reachable through `entity merge <a> <a>`, and unrecoverable without a
        # manual UPDATE if it were allowed through.
        sa.CheckConstraint("winner_id <> loser_id", name="ck_entity_merges_distinct"),
        # Keeps `status` and the timestamps from disagreeing about whether a
        # merge is in force.
        sa.CheckConstraint(
            "(status = 'applied' AND merged_at IS NOT NULL) "
            "OR (status = 'pending' AND merged_at IS NULL) "
            "OR (status = 'reverted' AND merged_at IS NOT NULL "
            "    AND reverted_at IS NOT NULL)",
            name="ck_entity_merges_status_timestamps",
        ),
    )

    # An entity can be merged away exactly once at a time. Partial, so a
    # reverted merge leaves the pair free to be merged again.
    op.create_index(
        "uq_entity_merges_active_loser",
        "entity_merges",
        ["loser_id"],
        unique=True,
        postgresql_where=sa.text("status = 'applied'"),
    )
    # A re-run of the resolver must not grow the review queue by a copy of
    # itself.
    op.create_index(
        "uq_entity_merges_pending_pair",
        "entity_merges",
        ["winner_id", "loser_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "ix_entity_merges_status", "entity_merges", ["status", "confidence"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_entity_merges_status", table_name="entity_merges")
    op.drop_index("uq_entity_merges_pending_pair", table_name="entity_merges")
    op.drop_index("uq_entity_merges_active_loser", table_name="entity_merges")
    op.drop_table("entity_merges")
    op.drop_index("ix_entities_merged_into", table_name="entities")
    op.drop_constraint("ck_entities_not_merged_into_self", "entities", type_="check")
    op.drop_constraint("fk_entities_merged_into_id", "entities", type_="foreignkey")
    op.drop_column("entities", "merged_into_id")
