"""decisions, their options, assumptions, evidence, and the review queue

Revision ID: 0016
Revises: 0015
Create Date: 2026-08-13

Phase 5 needs data the first four phases never had to collect. Every table
before this one holds either bytes observed at a source or something computed
from them; `query_judgements` is the single exception and it records an opinion
about a search result. A decision record is the first thing here that is a claim
about the world — made at a moment, by a person, under uncertainty — and none of
it is recoverable after the fact. Somebody who chose Postgres over Celery in
March can tell you in November what they chose and can no longer tell you what
they thought the odds were.

**Five tables where the milestone named four.** The fifth,
`decision_suggestions`, is the review queue, and it is a table rather than a
status column on `decisions` because a pending draft is not a decision in an
early state. It is a language model's reading of a passage. Giving it a row in
`decisions` would mean every query in M5.1 through M5.4 had to remember to
exclude it, and one forgotten predicate is then a pattern built on drafts.

`decision_assumptions` is the one that matters most and the one that will look
least useful today, because `held` and `evaluated_at` stay null until M5.2. It is
declared now for the reason M1.1 declared `occurred_at`: the column is cheap
today and what it would have recorded is unrecoverable later.

**`decision_evidence` carries two identities on purpose.** The foreign keys
cascade, so a memory leaving the corpus takes its evidence with it — a link to a
document that no longer exists is a citation to nothing. That also means a full
replay, which truncates `memories`, takes this whole table with it. The natural
key beside the ids is what survives that: `application/replay.py` snapshots these
rows before truncating and re-links them by `(source_name, external_key,
chunk_ordinal)` afterwards. Same lesson as M1.7's, applied at schema-design time
rather than after a rebuild had eaten something.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "0016"
down_revision: str | Sequence[str] | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "decisions",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("chosen", sa.Text(), nullable=False),
        sa.Column("reasoning", sa.Text(), nullable=True),
        # At the time of deciding, and never refreshed. A number updated in
        # hindsight measures nothing.
        sa.Column("confidence", sa.REAL(), nullable=True),
        sa.Column("expected_outcome", sa.Text(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        # M1.1's provenance, unchanged: a date parsed out of a document is not a
        # date somebody declared, and Phase 4's weighting rules apply to both.
        sa.Column("decided_at_source", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'open'")
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_decisions"),
        sa.CheckConstraint(
            "status IN ('open', 'settled', 'reversed')", name="ck_decisions_status"
        ),
        sa.CheckConstraint(
            "decided_at_source IN "
            "('declared', 'parsed', 'filesystem', 'inferred', 'unknown')",
            name="ck_decisions_decided_at_source",
        ),
        # `decided_at` is NOT NULL, so M1.1's null-pairing rule becomes a
        # prohibition instead: there is no missing date for 'unknown' to
        # describe.
        sa.CheckConstraint(
            "decided_at_source <> 'unknown'", name="ck_decisions_decided_at_known"
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_decisions_confidence_range",
        ),
        sa.CheckConstraint("length(btrim(question)) > 0", name="ck_decisions_question"),
        sa.CheckConstraint("length(btrim(chosen)) > 0", name="ck_decisions_chosen"),
    )
    op.create_index(
        "ix_decisions_status_decided_at", "decisions", ["status", "decided_at"]
    )

    op.create_table(
        "decision_options",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("decision_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "was_chosen", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        # Why *this* alternative lost, which is a different statement from why
        # the winner won and diverges from it constantly.
        sa.Column("rejected_because", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_decision_options"),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_decision_options_decision_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "length(btrim(description)) > 0", name="ck_decision_options_description"
        ),
        sa.CheckConstraint(
            "NOT was_chosen OR rejected_because IS NULL",
            name="ck_decision_options_chosen_has_no_rejection",
        ),
    )
    # One winner per decision. Partial, because nothing limits how many
    # alternatives were weighed.
    op.create_index(
        "uq_decision_options_one_chosen",
        "decision_options",
        ["decision_id"],
        unique=True,
        postgresql_where=sa.text("was_chosen"),
    )
    op.create_index(
        "ix_decision_options_decision", "decision_options", ["decision_id"]
    )

    op.create_table(
        "decision_assumptions",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("decision_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("confidence", sa.REAL(), nullable=True),
        # Both written by M5.2, both null until then. NULL means "not yet
        # judged" and is deliberately not `false`.
        sa.Column("held", sa.Boolean(), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_decision_assumptions"),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_decision_assumptions_decision_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "length(btrim(statement)) > 0", name="ck_decision_assumptions_statement"
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_decision_assumptions_confidence_range",
        ),
        sa.CheckConstraint(
            "(held IS NULL) = (evaluated_at IS NULL)",
            name="ck_decision_assumptions_evaluation_pairing",
        ),
    )
    op.create_index(
        "ix_decision_assumptions_decision", "decision_assumptions", ["decision_id"]
    )

    op.create_table(
        "decision_evidence",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("decision_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("memory_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("chunk_id", sa.Uuid(as_uuid=True), nullable=True),
        # The durable identity of the same item, so a rebuild can re-link.
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=False),
        sa.Column("chunk_ordinal", sa.Integer(), nullable=True),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column(
            "linked_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_decision_evidence"),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_decision_evidence_decision_id",
            ondelete="CASCADE",
        ),
        # Cascading, so a memory that leaves the corpus takes its links with it.
        # The decision survives; the citation does not, because a citation that
        # resolves to nothing is worse than an absent one.
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.id"],
            name="fk_decision_evidence_memory_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["memory_chunks.id"],
            name="fk_decision_evidence_chunk_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "decision_id",
            "memory_id",
            "chunk_id",
            "relation",
            name="uq_decision_evidence_link",
            postgresql_nulls_not_distinct=True,
        ),
        sa.CheckConstraint(
            "relation IN ('informed', 'records', 'contradicts')",
            name="ck_decision_evidence_relation",
        ),
        sa.CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_decision_evidence_chunk_ordinal_non_negative",
        ),
        # A chunk-level link must carry the ordinal that survives a rebuild, or
        # a re-link would silently widen it to the whole memory.
        sa.CheckConstraint(
            "(chunk_id IS NULL) = (chunk_ordinal IS NULL)",
            name="ck_decision_evidence_chunk_pairing",
        ),
    )
    op.create_index("ix_decision_evidence_decision", "decision_evidence", ["decision_id"])
    op.create_index("ix_decision_evidence_memory", "decision_evidence", ["memory_id"])

    op.create_table(
        "decision_suggestions",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("draft", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        # The passage the model read, verbatim. The queue shows it beside the
        # draft, so accepting is a judgement about evidence rather than about
        # plausibility.
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=False),
        sa.Column("chunk_ordinal", sa.Integer(), nullable=True),
        # Snapshots, with no foreign key. This table is USER_AUTHORED and must
        # outlive the replay its provenance points into.
        sa.Column("memory_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("chunk_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column(
            "status", sa.Text(), nullable=False, server_default=sa.text("'pending'")
        ),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("suggester_version", sa.Text(), nullable=False),
        sa.Column("decision_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column(
            "suggested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_decision_suggestions"),
        # SET NULL, not CASCADE: deleting a decision must not erase the record
        # that a suggestion was once accepted, which is part of how the
        # extractor is scored.
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_decision_suggestions_decision_id",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'accepted', 'rejected')",
            name="ck_decision_suggestions_status",
        ),
        sa.CheckConstraint(
            "length(btrim(source_text)) > 0", name="ck_decision_suggestions_source_text"
        ),
        sa.CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_decision_suggestions_chunk_ordinal_non_negative",
        ),
        sa.CheckConstraint(
            "(status = 'pending') = (reviewed_at IS NULL)",
            name="ck_decision_suggestions_review_pairing",
        ),
        sa.CheckConstraint(
            "decision_id IS NULL OR status = 'accepted'",
            name="ck_decision_suggestions_decision_requires_accept",
        ),
    )
    op.create_index(
        "ix_decision_suggestions_status",
        "decision_suggestions",
        ["status", "suggested_at"],
    )
    # The same passage is not queued twice while a proposal is outstanding, so a
    # re-run does not grow the queue by a copy of itself. Keyed on the durable
    # identity rather than on `chunk_id`, which a replay replaces.
    op.create_index(
        "uq_decision_suggestions_pending_passage",
        "decision_suggestions",
        ["source_name", "external_key", "chunk_ordinal"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Children first: every one of these references `decisions`.
    op.drop_table("decision_suggestions")
    op.drop_table("decision_evidence")
    op.drop_table("decision_assumptions")
    op.drop_table("decision_options")
    op.drop_table("decisions")
