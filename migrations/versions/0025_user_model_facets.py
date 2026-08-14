"""M8.0: the user model — facets, their evidence, and their history.

Two tables and one self-reference that carries most of the design.

**`superseded_by` is why a facet is never updated in place.** A model of a person
has to be able to change, and the way it changes is itself the interesting part:
"you used to defer schema decisions and stopped in March" is a claim you can only
make if the March version of the facet still exists. An `UPDATE` would replace
the old statement with the new one and leave nothing to compare, so a revision
inserts a new row and points the old one at it. The chain is walkable in both
directions and `model history` walks it.

`dismissed_at` is the second half of the same idea and the one M5.3 already paid
for. A person reading a claim about themselves and rejecting it is not in any
log; nothing regenerates it; and a derivation that re-proposed a dismissed facet
every time it ran would be a system arguing with its user. Both tables are
therefore `USER_AUTHORED` in `replay.py` — see the note there, which is the same
note `patterns` carries for the same two columns.

Revision ID: 0025
Revises: 0024
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The seven M8.0 names. A check constraint rather than a Postgres enum, for the
# reason every other status column in this schema gives: adding a dimension to an
# enum type is a migration that locks the table, while adding one to a check
# constraint is a migration that rewrites a rule.
_DIMENSIONS = (
    "goals",
    "habits",
    "strengths",
    "weaknesses",
    "learning_style",
    "decision_patterns",
    "workflows",
)

_EVIDENCE_KINDS = ("decision", "pattern", "memory", "outcome")
_RELATIONS = ("supports", "contradicts")


def upgrade() -> None:
    op.create_table(
        "user_model_facets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("dimension", sa.Text(), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        # Null for an asserted facet. **Not 1.0**: a goal somebody stated is not
        # a claim with 100% confidence, it is a claim that is not the kind of
        # thing confidence applies to, and writing 1.0 would put user statements
        # at the top of every ranking sorted by it.
        sa.Column("confidence", sa.REAL(), nullable=True),
        sa.Column("support_count", sa.Integer(), nullable=False),
        sa.Column("contradiction_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("first_observed", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("last_observed", postgresql.TIMESTAMP(timezone=True), nullable=True),
        # How this row came to exist. `derive` may only ever write or supersede
        # rows marked `derived`; an asserted facet is untouchable by it, which is
        # what makes `assert` safe to use on a corpus that is still being
        # re-derived nightly.
        sa.Column("origin", sa.Text(), nullable=False, server_default="derived"),
        # The detector that produced it, for a derived facet. Null for asserted.
        # Carried so that a change in one deriver can be re-run without touching
        # facets another deriver owns.
        sa.Column("detector", sa.Text(), nullable=True),
        # Stable identity of what the facet is *about*, within its dimension.
        # Two derivations of the same underlying regularity must produce the same
        # key so the second supersedes the first rather than duplicating it.
        sa.Column("subject_key", sa.Text(), nullable=True),
        # **Deferrable, and it has to be.** Retiring a facet and writing its
        # replacement is circular under two constraints that are both wanted:
        # the partial unique index below allows one live row per subject, so the
        # old row must stop being live before the new one is inserted; this
        # foreign key requires the new row to exist before the old one may point
        # at it. Deferring the check to commit lets both hold at the only moment
        # anybody can observe the table.
        sa.Column(
            "superseded_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "user_model_facets.id",
                ondelete="SET NULL",
                deferrable=True,
                initially="DEFERRED",
            ),
            nullable=True,
        ),
        sa.Column("dismissed_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("dismissed_reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "dimension IN ("
            + ", ".join(f"'{name}'" for name in _DIMENSIONS)
            + ")",
            name="ck_user_model_facets_dimension",
        ),
        sa.CheckConstraint(
            "origin IN ('derived', 'asserted')",
            name="ck_user_model_facets_origin",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_user_model_facets_confidence_range",
        ),
        sa.CheckConstraint(
            "support_count >= 0 AND contradiction_count >= 0",
            name="ck_user_model_facets_counts_non_negative",
        ),
        # A dismissal without a reason is a row nobody can interpret later, and
        # the reason is the only part that survives being forgotten.
        sa.CheckConstraint(
            "(dismissed_at IS NULL) = (dismissed_reason IS NULL)",
            name="ck_user_model_facets_dismissal_paired",
        ),
        # An asserted facet carries no detector and no derived confidence; a
        # derived one must name the detector that made it. Enforced here because
        # the distinction is what `derive` reads to decide what it may overwrite,
        # and a row that lies about its origin would let a derivation silently
        # replace something a person wrote.
        sa.CheckConstraint(
            "(origin = 'asserted' AND detector IS NULL)"
            " OR (origin = 'derived' AND detector IS NOT NULL)",
            name="ck_user_model_facets_origin_detector",
        ),
    )
    # The live model: everything not superseded and not dismissed, by dimension.
    # Partial, because that is the only query `model show` makes and the
    # superseded rows are a history nobody scans.
    op.create_index(
        "ix_user_model_facets_live",
        "user_model_facets",
        ["dimension"],
        postgresql_where=sa.text("superseded_by IS NULL AND dismissed_at IS NULL"),
    )
    # One live derived facet per (detector, subject). The partial unique index is
    # what makes re-derivation idempotent: the second run finds the first row
    # instead of inserting a twin, and supersedes it only if the statement moved.
    op.create_index(
        "uq_user_model_facets_live_subject",
        "user_model_facets",
        ["detector", "subject_key"],
        unique=True,
        postgresql_where=sa.text(
            "origin = 'derived' AND superseded_by IS NULL AND dismissed_at IS NULL"
        ),
    )

    op.create_table(
        "facet_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "facet_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("user_model_facets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        # **No foreign key, deliberately.** `ref_id` points into one of four
        # tables depending on `kind`, which no single constraint can express —
        # the alternative is four nullable columns and four constraints to keep
        # exactly one populated. `entity_mentions` made the same call for the
        # same reason. What it costs is a dangling ref when a memory is deleted,
        # and `model show` reports an unresolvable reference rather than hiding
        # it.
        sa.Column("ref_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "kind IN (" + ", ".join(f"'{name}'" for name in _EVIDENCE_KINDS) + ")",
            name="ck_facet_evidence_kind",
        ),
        sa.CheckConstraint(
            "relation IN (" + ", ".join(f"'{name}'" for name in _RELATIONS) + ")",
            name="ck_facet_evidence_relation",
        ),
        # The same item cannot both support and contradict one facet twice over.
        sa.UniqueConstraint(
            "facet_id", "kind", "ref_id", "relation", name="uq_facet_evidence_item"
        ),
    )
    op.create_index("ix_facet_evidence_facet", "facet_evidence", ["facet_id"])


def downgrade() -> None:
    op.drop_index("ix_facet_evidence_facet", table_name="facet_evidence")
    op.drop_table("facet_evidence")
    op.drop_index("uq_user_model_facets_live_subject", table_name="user_model_facets")
    op.drop_index("ix_user_model_facets_live", table_name="user_model_facets")
    op.drop_table("user_model_facets")
