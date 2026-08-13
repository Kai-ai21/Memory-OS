"""what actually happened after a decision

Revision ID: 0017
Revises: 0016
Create Date: 2026-08-13

M5.0 recorded what was decided and what had to be true for it to be right. This
is the other half: what happened afterwards, and how much anybody actually knows
about it.

**Two columns carry the weight.** `verdict` includes `too_early`, which is a
verdict rather than a null — most decisions in a young project have no outcome
yet, and "we looked and it is too soon to say" is a different fact from "nobody
has looked". It is excluded from every success rate, so a corpus with two wins
and thirty unresolved decisions reports two out of two rather than a number that
reads like a track record.

`evidence_kind` separates an outcome somebody observed from one the system
inferred. They are not equally trustworthy: a declared outcome is testimony, an
inferred one is a correlation in time plus a language model's opinion that the
correlation means something. M5.3 has to weight them differently or its patterns
will rest mostly on the cheaper kind, because the cheaper kind is the one that
scales. A CHECK forbids an inferred outcome from claiming confidence 1.0, so a
future writer cannot quietly promote a guess to testimony.

**Several outcomes per decision, and no unique constraint saying otherwise.** A
decision can work in the first month and fail in the sixth. Collapsing that into
one mutable row would destroy exactly the sequence M5.3 exists to find, so
`observed_at` orders them and a `too_early` recorded early is not contradicted by
a later verdict — it is the honest first half of the story.

`outcome_evidence` repeats `decision_evidence`'s two-identity design and inherits
its consequence: the foreign keys cascade, so `TRUNCATE memories CASCADE` reaches
it, and the natural key beside the ids is what `application/replay.py` re-links
against afterwards. The shadow swap picks the new constraints up automatically —
it reads them off `Base.metadata` rather than from a list somebody has to
remember to extend.

`outcome_suggestions` is the review queue, and it stores the whole basis of a
proposal rather than only its conclusion: the candidate memory, the gap in days,
the window that admitted it, and the entities the two share. Post hoc ergo
propter hoc is the oldest error there is, and a model shown two related-looking
documents will make it fluently — so the reviewer gets the evidence, not a score.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0017"
down_revision: str | Sequence[str] | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "decision_outcomes",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("decision_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observed_at_source", sa.Text(), nullable=False),
        # declared = somebody watched it happen. inferred = a model read a
        # memory that occurred afterwards and judged it a consequence.
        sa.Column("evidence_kind", sa.Text(), nullable=False),
        sa.Column("confidence", sa.REAL(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_decision_outcomes"),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_decision_outcomes_decision_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "verdict IN ('worked', 'failed', 'mixed', 'too_early')",
            name="ck_decision_outcomes_verdict",
        ),
        sa.CheckConstraint(
            "evidence_kind IN ('declared', 'inferred')",
            name="ck_decision_outcomes_evidence_kind",
        ),
        sa.CheckConstraint(
            "observed_at_source IN "
            "('declared', 'parsed', 'filesystem', 'inferred', 'unknown')",
            name="ck_decision_outcomes_observed_at_source",
        ),
        sa.CheckConstraint(
            "observed_at_source <> 'unknown'",
            name="ck_decision_outcomes_observed_at_known",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_decision_outcomes_confidence_range",
        ),
        sa.CheckConstraint(
            "length(btrim(description)) > 0", name="ck_decision_outcomes_description"
        ),
        # A guess may not present itself as testimony.
        sa.CheckConstraint(
            "evidence_kind <> 'inferred' OR confidence IS NULL OR confidence < 1.0",
            name="ck_decision_outcomes_inferred_is_not_certain",
        ),
    )
    op.create_index(
        "ix_decision_outcomes_decision",
        "decision_outcomes",
        ["decision_id", "observed_at"],
    )
    op.create_index("ix_decision_outcomes_verdict", "decision_outcomes", ["verdict"])

    op.create_table(
        "outcome_evidence",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("outcome_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("memory_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("chunk_id", sa.Uuid(as_uuid=True), nullable=True),
        # The durable identity, for re-linking after a rebuild.
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=False),
        sa.Column("chunk_ordinal", sa.Integer(), nullable=True),
        # A snapshot of the evidence memory's own clock, taken at link time. The
        # gap between this and the decision's date is the claim being made, and
        # re-deriving it later would let a re-sync silently change how strong a
        # link somebody already reviewed.
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "linked_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_outcome_evidence"),
        sa.ForeignKeyConstraint(
            ["outcome_id"],
            ["decision_outcomes.id"],
            name="fk_outcome_evidence_outcome_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.id"],
            name="fk_outcome_evidence_memory_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["memory_chunks.id"],
            name="fk_outcome_evidence_chunk_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "outcome_id",
            "memory_id",
            "chunk_id",
            name="uq_outcome_evidence_link",
            postgresql_nulls_not_distinct=True,
        ),
        sa.CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_outcome_evidence_chunk_ordinal_non_negative",
        ),
        sa.CheckConstraint(
            "(chunk_id IS NULL) = (chunk_ordinal IS NULL)",
            name="ck_outcome_evidence_chunk_pairing",
        ),
    )
    op.create_index("ix_outcome_evidence_outcome", "outcome_evidence", ["outcome_id"])
    op.create_index("ix_outcome_evidence_memory", "outcome_evidence", ["memory_id"])

    op.create_table(
        "outcome_suggestions",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("decision_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("draft", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=False),
        sa.Column("chunk_ordinal", sa.Integer(), nullable=True),
        # Snapshots, no foreign key: this table is user-authored and must
        # outlive the replay its provenance points into.
        sa.Column("memory_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("chunk_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("candidate_occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gap_days", sa.REAL(), nullable=False),
        sa.Column("window_days", sa.REAL(), nullable=False),
        sa.Column(
            "shared_entities",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        # 'applied' or 'unavailable'. Not a boolean: the third state a boolean
        # invites — false meaning "no overlap" — is exactly the conflation this
        # column exists to prevent. No overlap is a rejection; no coverage is a
        # test that could not be run.
        sa.Column("entity_filter", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'pending'")
        ),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("suggester_version", sa.Text(), nullable=False),
        sa.Column("outcome_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column(
            "suggested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_outcome_suggestions"),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_outcome_suggestions_decision_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["outcome_id"],
            ["decision_outcomes.id"],
            name="fk_outcome_suggestions_outcome_id",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'accepted', 'rejected')",
            name="ck_outcome_suggestions_status",
        ),
        sa.CheckConstraint(
            "entity_filter IN ('applied', 'unavailable')",
            name="ck_outcome_suggestions_entity_filter",
        ),
        sa.CheckConstraint(
            "length(btrim(source_text)) > 0", name="ck_outcome_suggestions_source_text"
        ),
        # The premise of the whole suggestion. A candidate that occurred before
        # the decision is not a weak outcome — it is a causal claim running
        # backwards, and the database refuses it as well as the query.
        sa.CheckConstraint("gap_days > 0", name="ck_outcome_suggestions_gap_positive"),
        sa.CheckConstraint(
            "window_days > 0", name="ck_outcome_suggestions_window_positive"
        ),
        sa.CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_outcome_suggestions_chunk_ordinal_non_negative",
        ),
        sa.CheckConstraint(
            "(status = 'pending') = (reviewed_at IS NULL)",
            name="ck_outcome_suggestions_review_pairing",
        ),
        sa.CheckConstraint(
            "outcome_id IS NULL OR status = 'accepted'",
            name="ck_outcome_suggestions_outcome_requires_accept",
        ),
    )
    op.create_index(
        "ix_outcome_suggestions_status",
        "outcome_suggestions",
        ["status", "suggested_at"],
    )
    op.create_index(
        "uq_outcome_suggestions_pending_pair",
        "outcome_suggestions",
        ["decision_id", "source_name", "external_key", "chunk_ordinal"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Children first: both reference `decision_outcomes`.
    op.drop_table("outcome_suggestions")
    op.drop_table("outcome_evidence")
    op.drop_table("decision_outcomes")
