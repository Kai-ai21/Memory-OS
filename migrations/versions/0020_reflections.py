"""reflections: a pattern in prose, with the citations that make it checkable

Revision ID: 0020
Revises: 0019
Create Date: 2026-08-13

The last schema in Phase 5, and the only table in this database whose contents a
language model wrote.

Everything before it stores retrieved text or computed numbers. A `patterns` row
is a statement assembled from counts with the decisions it was counted from
sitting beside it, and a reader looks at both together. A reflection is fluent
English about the reader's own judgement — "you tend to underestimate how long
integration takes" — and prose is read as a claim rather than as a summary of a
table. An unfalsifiable sentence of that kind is the single most damaging thing
this system can emit: it sounds like the product working, it is trusted because
it is personal, and there is nothing in it to argue with.

Three parts of this migration are the milestone's actual content.

**`reflection_citations` is a table rather than a column, and M1.4a is why.**
That milestone stored citations as offsets, the offsets drifted under the text
they pointed into, and nothing failed — row counts were right, every test passed,
and the highlights pointed a few hundred characters from the answer. The same
drift is available here and it is worse: a reflection's `[3]` means "the third
decision in the list this reflection was generated from", and `patterns discover`
replaces a pattern's evidence *wholesale* on every re-run. Re-deriving the
numbering at read time would silently renumber every citation the first time the
corpus grew, and the claim would still read correctly while linking to a
different decision. Frozen here as a foreign key, that cannot happen.

**`dismissed_at` with its paired reason.** You have to be able to say "this is
wrong about me" and have the system stop repeating it *and* stop regenerating it.
The application refuses to generate for a pattern whose reflection was dismissed,
rather than only hiding the row — a rejection a weekly re-run undid would not be
a rejection. The pairing CHECK is the one `patterns` carries, for the same
reason: a refusal nobody explained cannot be told from a stale row.

**`citation_rate` NOT NULL is deliberately absent, and `model_id` NOT NULL is
deliberately present.** The rate is what the grounding check measured at
generation and it is stored rather than recomputed, because recomputing it later
against a pattern whose evidence has moved would be a different measurement
wearing the same name. `model_id` is required because the wording of this row is
the one thing in the database that is not reproducible from the data, so what
produced the wording is part of the record.

There is no `pattern_id` UNIQUE constraint. A pattern may accumulate several
reflections over time — a dismissed one and, if the evidence later changes
substantially, a newer one — and collapsing them to a single row would mean
overwriting text somebody had already read and judged.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0020"
down_revision: str | Sequence[str] | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "reflections",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("pattern_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        # Cited sentences over sentences, as measured at generation. Every
        # sentence must cite, not only the ones a heuristic calls factual —
        # see `domain/grounding.check_reflection`.
        sa.Column("citation_rate", sa.REAL(), nullable=True),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        # Which model wrote it. Not optional: this is the only row in the
        # database whose wording is not reproducible from the data.
        sa.Column("model_id", sa.Text(), nullable=False),
        # Read, which is not agreement and is weighted by nothing.
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_reason", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_reflections"),
        sa.ForeignKeyConstraint(
            ["pattern_id"],
            ["patterns.id"],
            name="fk_reflections_pattern_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("length(btrim(text)) > 0", name="ck_reflections_text"),
        sa.CheckConstraint(
            "citation_rate IS NULL OR citation_rate BETWEEN 0.0 AND 1.0",
            name="ck_reflections_citation_rate_range",
        ),
        sa.CheckConstraint(
            "(dismissed_at IS NULL) = (dismissed_reason IS NULL)",
            name="ck_reflections_dismissal_pairing",
        ),
    )
    op.create_index("ix_reflections_pattern", "reflections", ["pattern_id"])

    op.create_table(
        "reflection_citations",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("reflection_id", sa.Uuid(as_uuid=True), nullable=False),
        # The number as it appears in the text, 1-based, exactly as the model
        # was shown it.
        sa.Column("marker", sa.Integer(), nullable=False),
        sa.Column("decision_id", sa.Uuid(as_uuid=True), nullable=False),
        # What the reflection was *told* about this decision. Carried rather
        # than joined back to `pattern_evidence`, which reports what the
        # detector thinks today.
        sa.Column("relation", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_reflection_citations"),
        sa.ForeignKeyConstraint(
            ["reflection_id"],
            ["reflections.id"],
            name="fk_reflection_citations_reflection_id",
            ondelete="CASCADE",
        ),
        # A cited decision that is later deleted takes its citation with it,
        # rather than leaving a marker in the prose pointing at nothing.
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decisions.id"],
            name="fk_reflection_citations_decision_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("reflection_id", "marker", name="uq_reflection_citations_marker"),
        sa.CheckConstraint("marker > 0", name="ck_reflection_citations_marker_positive"),
        sa.CheckConstraint(
            "relation IN ('supports', 'contradicts')",
            name="ck_reflection_citations_relation",
        ),
    )
    op.create_index("ix_reflection_citations_decision", "reflection_citations", ["decision_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("reflection_citations")
    op.drop_table("reflections")
