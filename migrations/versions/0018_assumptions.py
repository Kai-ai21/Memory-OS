"""evaluate assumptions, and group the ones that say the same thing

Revision ID: 0018
Revises: 0017
Create Date: 2026-08-13

M5.0 declared `held` and `evaluated_at` and left them null, for the reason M1.1
declared `occurred_at` six milestones before anything read it. This fills them,
and widens one of them on the way.

**`held` stops being a boolean.** Forcing a binary produces noise rather than
data: almost nothing anybody assumes is cleanly right or wrong. "The free tier's
rate limits are workable for a corpus of this size" was true for months of
ordinary use and false the first time a corpus-wide extraction ran — recording
that as `false` loses the half that was right, and as `true` loses the milestone
it blocked. So the column becomes text over `held | failed | partially`, NULL
still meaning nobody has judged it.

The column keeps its name. `held = 'failed'` reads oddly and renaming it to fix
one sentence is how a schema and its documentation drift apart; everything M5.0
wrote about the column is still true of it.

**`assumption_groups` is what makes M5.3 possible.** A pattern is the same
assumption failing repeatedly, and "the same assumption" is not a string
comparison — "this will take two days", "the deploy is straightforward" and
"integration should be quick" are one recurring belief wearing three sentences.
Grouping is M3.2's machinery over a different column, with M3.2's asymmetry
intact: a false grouping invents a recurrence out of unrelated beliefs, so the
auto bar is high and everything under it becomes a pending candidate for a
person.

`assumption_evidence` is the third table with the two-identity shape, after
`decision_evidence` and `outcome_evidence`, and inherits their handling: the
foreign keys cascade, the natural key beside them is what a replay re-links
against, and it is listed in `EVIDENCE_TABLES` where a test checks the list
against the metadata rather than trusting it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0018"
down_revision: str | Sequence[str] | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "assumption_groups",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        # The statement of whichever member the group was built around. No
        # attempt to synthesise a better label: a generated summary of three
        # sentences is a fourth sentence nobody wrote, and M5.3 would then find
        # patterns in text this migration invented.
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_assumption_groups"),
        sa.CheckConstraint(
            "strategy IN ('exact', 'embedding', 'alias', 'llm', 'manual')",
            name="ck_assumption_groups_strategy",
        ),
        sa.CheckConstraint("length(btrim(label)) > 0", name="ck_assumption_groups_label"),
    )

    # `held` from boolean to a three-state text column.
    #
    # Every existing value is NULL — M5.0 declared the column and nothing has
    # written it — so the `USING` clause below is exercised by nothing today and
    # is written correctly anyway, because a migration that only works on an
    # empty column is a migration that fails on the one database that matters.
    op.alter_column(
        "decision_assumptions",
        "held",
        existing_type=sa.Boolean(),
        type_=sa.Text(),
        existing_nullable=True,
        postgresql_using="CASE WHEN held THEN 'held' WHEN NOT held THEN 'failed' END",
    )
    op.create_check_constraint(
        "ck_decision_assumptions_held",
        "decision_assumptions",
        "held IN ('held', 'failed', 'partially')",
    )
    op.add_column(
        "decision_assumptions",
        # Why the evaluator reached that verdict. Separate from `statement`,
        # which is what was believed at the time and must never be edited to
        # match what happened.
        sa.Column("note", sa.Text(), nullable=True),
    )
    op.add_column(
        "decision_assumptions",
        sa.Column("group_id", sa.Uuid(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_decision_assumptions_group_id",
        "decision_assumptions",
        "assumption_groups",
        ["group_id"],
        ["id"],
        # SET NULL, not CASCADE: deleting a group ungroups its members, it does
        # not delete the assumptions.
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_decision_assumptions_held", "decision_assumptions", ["held"]
    )
    op.create_index(
        "ix_decision_assumptions_group", "decision_assumptions", ["group_id"]
    )

    op.create_table(
        "assumption_group_candidates",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("left_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("right_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("similarity", sa.REAL(), nullable=False),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'pending'")
        ),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column(
            "proposed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_assumption_group_candidates"),
        sa.ForeignKeyConstraint(
            ["left_id"],
            ["decision_assumptions.id"],
            name="fk_assumption_group_candidates_left_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["right_id"],
            ["decision_assumptions.id"],
            name="fk_assumption_group_candidates_right_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'applied', 'reverted')",
            name="ck_assumption_group_candidates_status",
        ),
        sa.CheckConstraint(
            "similarity BETWEEN 0.0 AND 1.0",
            name="ck_assumption_group_candidates_similarity_range",
        ),
        sa.CheckConstraint(
            "left_id <> right_id", name="ck_assumption_group_candidates_distinct"
        ),
        sa.CheckConstraint(
            "(status = 'pending') = (reviewed_at IS NULL)",
            name="ck_assumption_group_candidates_review_pairing",
        ),
    )
    op.create_index(
        "uq_assumption_group_candidates_pending_pair",
        "assumption_group_candidates",
        ["left_id", "right_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "ix_assumption_group_candidates_status",
        "assumption_group_candidates",
        ["status", "similarity"],
    )

    op.create_table(
        "assumption_evidence",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("assumption_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("memory_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("chunk_id", sa.Uuid(as_uuid=True), nullable=True),
        # The durable identity, for re-linking after a rebuild.
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=False),
        sa.Column("chunk_ordinal", sa.Integer(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "linked_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_assumption_evidence"),
        sa.ForeignKeyConstraint(
            ["assumption_id"],
            ["decision_assumptions.id"],
            name="fk_assumption_evidence_assumption_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.id"],
            name="fk_assumption_evidence_memory_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["memory_chunks.id"],
            name="fk_assumption_evidence_chunk_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "assumption_id",
            "memory_id",
            "chunk_id",
            name="uq_assumption_evidence_link",
            postgresql_nulls_not_distinct=True,
        ),
        sa.CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_assumption_evidence_chunk_ordinal_non_negative",
        ),
        sa.CheckConstraint(
            "(chunk_id IS NULL) = (chunk_ordinal IS NULL)",
            name="ck_assumption_evidence_chunk_pairing",
        ),
    )
    op.create_index(
        "ix_assumption_evidence_assumption", "assumption_evidence", ["assumption_id"]
    )
    op.create_index(
        "ix_assumption_evidence_memory", "assumption_evidence", ["memory_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("assumption_evidence")
    op.drop_table("assumption_group_candidates")
    op.drop_index("ix_decision_assumptions_group", table_name="decision_assumptions")
    op.drop_index("ix_decision_assumptions_held", table_name="decision_assumptions")
    op.drop_constraint(
        "fk_decision_assumptions_group_id", "decision_assumptions", type_="foreignkey"
    )
    op.drop_column("decision_assumptions", "group_id")
    op.drop_column("decision_assumptions", "note")
    op.drop_constraint(
        "ck_decision_assumptions_held", "decision_assumptions", type_="check"
    )
    # `partially` has no boolean image, so it becomes NULL — unevaluated. Losing
    # the verdict is better than mapping it onto either of the other two, which
    # would be this migration inventing an evaluation on the way down.
    op.alter_column(
        "decision_assumptions",
        "held",
        existing_type=sa.Text(),
        type_=sa.Boolean(),
        existing_nullable=True,
        postgresql_using=(
            "CASE WHEN held = 'held' THEN true WHEN held = 'failed' THEN false END"
        ),
    )
    op.drop_table("assumption_groups")
