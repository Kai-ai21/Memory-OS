"""entity relationships

Revision ID: 0014
Revises: 0013
Create Date: 2026-08-12

Typed, directed edges between entities, each carrying the chunk that asserted
it.

**Provenance is the column that makes the rest worth storing.** Without
`chunk_id` a relationship is an unfalsifiable claim — something believes React
depends on Postgres and nothing can say why — which is exactly what M2.5
eliminated for answers. A Phase 3 answer built on these edges is only as
citable as its least-supported edge.

**`UNIQUE (subject_id, predicate, object_id, chunk_id)` is scoped per chunk on
purpose.** The same relationship asserted in five chunks is five rows, and that
is the evidence rather than duplication: M3.5 weights edges by assertion count,
so one claim and a five-times-repeated claim have to remain distinguishable.
Collapsing them here would discard the only signal that separates them.

Direction lives in the column names. `subject_id` does `predicate` to
`object_id`, and "A supersedes B" is a different row from "B supersedes A"
rather than the same edge read backwards.

The span is nullable, unlike a mention's offsets. The model quotes the sentence
that asserts a relationship rather than pointing at a name, and a quote that
cannot be located in the chunk still leaves a usable edge — where a *mention*
without a verified span is worthless, because the span is the whole of what a
mention is.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0014"
down_revision: str | Sequence[str] | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # The same marker M3.1 added for entities, and separate from it: the two
    # prompts are independently re-runnable, so improving one must not
    # invalidate the other's output and force both to be re-extracted.
    op.add_column(
        "memories",
        sa.Column("relationship_extractor_version", sa.Text(), nullable=True),
    )
    op.create_table(
        "entity_relationships",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("subject_id", sa.Uuid(), nullable=False),
        sa.Column("object_id", sa.Uuid(), nullable=False),
        sa.Column("predicate", sa.Text(), nullable=False),
        sa.Column("confidence", sa.REAL(), nullable=True),
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=True),
        sa.Column("char_end", sa.Integer(), nullable=True),
        sa.Column("extractor_version", sa.Text(), nullable=False),
        sa.Column(
            "extracted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "predicate IN ('uses', 'depends_on', 'part_of', 'authored_by', "
            "'mentions', 'supersedes', 'relates_to')",
            name="ck_entity_relationships_predicate",
        ),
        sa.CheckConstraint(
            "(char_start IS NULL) = (char_end IS NULL)",
            name="ck_entity_relationships_span_pairing",
        ),
        sa.CheckConstraint(
            "char_end IS NULL OR char_end > char_start",
            name="ck_entity_relationships_char_range",
        ),
        sa.CheckConstraint(
            "char_start IS NULL OR char_start >= 0",
            name="ck_entity_relationships_char_start_non_negative",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_entity_relationships_confidence_range",
        ),
        # A self-relationship asserts nothing, and is what a model returns when
        # it has nothing to say about a chunk that names one entity twice.
        sa.CheckConstraint(
            "subject_id <> object_id", name="ck_entity_relationships_distinct"
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["memory_chunks.id"],
            name="fk_entity_relationships_chunk_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.id"],
            name="fk_entity_relationships_memory_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["object_id"],
            ["entities.id"],
            name="fk_entity_relationships_object_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["subject_id"],
            ["entities.id"],
            name="fk_entity_relationships_subject_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_entity_relationships")),
        sa.UniqueConstraint(
            "subject_id",
            "predicate",
            "object_id",
            "chunk_id",
            name="uq_entity_relationships_assertion",
        ),
    )
    op.create_index(
        "ix_entity_relationships_memory_version",
        "entity_relationships",
        ["memory_id", "extractor_version"],
        unique=False,
    )
    op.create_index(
        "ix_entity_relationships_object",
        "entity_relationships",
        ["object_id", "predicate"],
        unique=False,
    )
    op.create_index(
        "ix_entity_relationships_subject",
        "entity_relationships",
        ["subject_id", "predicate"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("memories", "relationship_extractor_version")
    op.drop_index("ix_entity_relationships_subject", table_name="entity_relationships")
    op.drop_index("ix_entity_relationships_object", table_name="entity_relationships")
    op.drop_index(
        "ix_entity_relationships_memory_version", table_name="entity_relationships"
    )
    op.drop_table("entity_relationships")
