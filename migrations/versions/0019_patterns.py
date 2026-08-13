"""behavioural patterns, with the evidence that has to exist for one to be written

Revision ID: 0019
Revises: 0018
Create Date: 2026-08-13

The last schema in Phase 5 before reflections, and the one that most needs its
constraints to be constraints rather than intentions.

Everything else in this database records something that happened. This records a
*generalisation about a person*, and a generalisation is the thing that sounds
most like the product working when it is wrong. "You consistently underestimate
deployment effort" is either a finding backed by five specific decisions or a
horoscope, and nothing in the sentence distinguishes the two — only the evidence
does.

So three of the CHECK constraints below are the milestone's actual content:

* `support_count > 0` — a pattern that cannot cite is never written. The
  application enforces a minimum of three *distinct decisions*; the database
  enforces that the number is never zero, so no future writer can create a
  behavioural claim with nothing behind it.
* `support_count > contradiction_count` — more evidence must agree than
  disagree. A candidate with four supporting and three contradicting is not a
  weak pattern, it is not a pattern, and admitting it with a low confidence
  would still put the claim in front of somebody.
* `(dismissed_at IS NULL) = (dismissed_reason IS NULL)` — a rejection carries
  its reason, or the next run cannot tell a considered refusal from a stale row.

`UNIQUE (detector, subject_key)` is what makes re-running discovery idempotent. A
pattern is identified by what it is *about* — this assumption group, this
confidence band — rather than by its sentence, because the sentence carries the
current numbers and changes every time the corpus grows. Keyed on the statement
instead, a weekly `patterns discover` would leave a row per week saying almost
the same thing with a different percentage in it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0019"
down_revision: str | Sequence[str] | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "patterns",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        # Which rule produced it, and what it is about.
        sa.Column("detector", sa.Text(), nullable=False),
        sa.Column("subject_key", sa.Text(), nullable=False),
        # Distinct decisions agreeing and disagreeing. Denormalised from
        # `pattern_evidence` because every read sorts on them, and written only
        # alongside the evidence rows in one transaction.
        sa.Column("support_count", sa.Integer(), nullable=False),
        sa.Column(
            "contradiction_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("confidence", sa.REAL(), nullable=True),
        # The span the supporting decisions cover. Three decisions made in one
        # afternoon support a different claim from three across a year.
        sa.Column("first_observed", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_observed", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "discovered_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_patterns"),
        sa.CheckConstraint(
            "kind IN ('assumption', 'timing', 'choice', 'outcome')",
            name="ck_patterns_kind",
        ),
        sa.CheckConstraint(
            "length(btrim(statement)) > 0", name="ck_patterns_statement"
        ),
        sa.CheckConstraint("support_count > 0", name="ck_patterns_support_positive"),
        sa.CheckConstraint(
            "contradiction_count >= 0", name="ck_patterns_contradictions_non_negative"
        ),
        sa.CheckConstraint(
            "support_count > contradiction_count",
            name="ck_patterns_support_exceeds_contradiction",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_patterns_confidence_range",
        ),
        sa.CheckConstraint(
            "(dismissed_at IS NULL) = (dismissed_reason IS NULL)",
            name="ck_patterns_dismissal_pairing",
        ),
        sa.CheckConstraint(
            "first_observed IS NULL OR last_observed IS NULL "
            "OR first_observed <= last_observed",
            name="ck_patterns_observation_order",
        ),
        sa.UniqueConstraint(
            "detector", "subject_key", name="uq_patterns_detector_subject"
        ),
    )
    op.create_index("ix_patterns_kind", "patterns", ["kind"])

    op.create_table(
        "pattern_evidence",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("pattern_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("decision_id", sa.Uuid(as_uuid=True), nullable=False),
        # Nullable because the four detectors cite different things: an
        # assumption pattern points at the assumption that broke, a calibration
        # pattern at the outcome that resolved, a choice pattern at neither.
        sa.Column("assumption_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("outcome_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("relation", sa.Text(), nullable=False),
        # Why this decision counts for or against, written by the detector and
        # shown verbatim. "supports" beside a decision title is not something a
        # reader can check.
        sa.Column("note", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_pattern_evidence"),
        sa.ForeignKeyConstraint(
            ["pattern_id"],
            ["patterns.id"],
            name="fk_pattern_evidence_pattern_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_pattern_evidence_decision_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["assumption_id"],
            ["decision_assumptions.id"],
            name="fk_pattern_evidence_assumption_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["outcome_id"],
            ["decision_outcomes.id"],
            name="fk_pattern_evidence_outcome_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "pattern_id",
            "decision_id",
            "assumption_id",
            "outcome_id",
            name="uq_pattern_evidence_link",
            postgresql_nulls_not_distinct=True,
        ),
        sa.CheckConstraint(
            "relation IN ('supports', 'contradicts')",
            name="ck_pattern_evidence_relation",
        ),
    )
    op.create_index(
        "ix_pattern_evidence_pattern", "pattern_evidence", ["pattern_id", "relation"]
    )
    op.create_index("ix_pattern_evidence_decision", "pattern_evidence", ["decision_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("pattern_evidence")
    op.drop_table("patterns")
