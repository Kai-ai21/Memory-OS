"""entities and mentions

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-11

The first tables written by a language model rather than computed from bytes,
and the schema is shaped around not trusting it.

**Offsets are the reason `entity_mentions` exists at all.** An entity without a
mention is a claim; an entity with a mention is a claim plus the span of text
that produced it, which is the provenance chain M2.5 built for citations
extended to a second kind of derived fact. The columns are only ever written
after the extractor has confirmed the name really occurs at that offset — a
model asked where a name appears returns a plausible number, and a mention
stored at a guessed offset points at whatever text happens to occupy it.

**`UNIQUE (entity_id, chunk_id, char_start)`** makes re-extraction idempotent at
the row level, and those three columns are exactly the natural key: the same
entity named twice in one chunk is two real mentions at two offsets.

**`UNIQUE (canonical_name, type)` on `entities` is an addition to the specified
schema**, and without it the table has no identity. Every re-extraction of a
chunk would insert another row for the same name, and "the twenty most-mentioned
entities" would be a list of twenty coincidences. It is exact-match
deduplication on a minimally normalised name — casefold and collapsed whitespace
— and deliberately not resolution: it knows "Neo4j" and "neo4j" are one entity
and has no opinion about "Dr. Chen" versus "Chen". Normalising harder here would
shrink the duplicate count M3.2 is scoped against, which is improving a number
by moving the ruler.

Both foreign keys cascade. A deleted memory takes its chunks and its chunks take
their mentions, because a mention whose chunk is gone has no text to point at.
`entities` is deliberately *not* cascaded from anything: an entity outliving its
last mention is an orphan the next extraction re-attaches or a later milestone
sweeps, not a row to delete inside somebody else's transaction.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: str | Sequence[str] | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "entities",
        sa.Column("id", sa.Uuid(), nullable=False),
        # The surface form as first seen, kept alongside the canonical form
        # because losing what the text said loses the evidence for any later
        # resolution decision.
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("confidence", sa.REAL(), nullable=True),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_entities"),
        sa.UniqueConstraint("canonical_name", "type", name="uq_entities_canonical_type"),
        # Generated from `EntityType`, so the database and the Python enum
        # cannot drift without the migration diff showing it. A closed
        # vocabulary on purpose: an open one is what a language model produces
        # if allowed, and `person` alongside `people` is a filter that silently
        # returns half its rows.
        sa.CheckConstraint(
            "type IN ('person', 'technology', 'project', 'organization', "
            "'concept', 'file', 'decision')",
            name="ck_entities_type",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_entities_confidence_range",
        ),
        sa.CheckConstraint("length(btrim(name)) > 0", name="ck_entities_name_non_empty"),
        sa.CheckConstraint(
            "length(btrim(canonical_name)) > 0", name="ck_entities_canonical_non_empty"
        ),
    )
    # Resolution's read pattern in M3.2 — find the candidates a name might
    # collapse into — and what makes the duplicate measurement cheap.
    op.create_index("ix_entities_canonical_name", "entities", ["canonical_name"])

    op.create_table(
        "entity_mentions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("entity_id", sa.Uuid(), nullable=False),
        sa.Column("memory_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        # Into `memory_chunks.content`, exactly, and verified before they are
        # written: content[char_start:char_end] is the entity's surface form.
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.REAL(), nullable=True),
        # The M1.4 chunker-version pattern: encodes the model and the prompt, so
        # improving extraction is a query over this column rather than a rebuild.
        sa.Column("extractor_version", sa.Text(), nullable=False),
        sa.Column(
            "extracted_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_entity_mentions"),
        sa.UniqueConstraint(
            "entity_id", "chunk_id", "char_start", name="uq_entity_mentions_span"
        ),
        sa.ForeignKeyConstraint(
            ["entity_id"],
            ["entities.id"],
            name="fk_entity_mentions_entity_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memories.id"],
            name="fk_entity_mentions_memory_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["memory_chunks.id"],
            name="fk_entity_mentions_chunk_id",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "char_start >= 0", name="ck_entity_mentions_char_start_non_negative"
        ),
        sa.CheckConstraint("char_end > char_start", name="ck_entity_mentions_char_range"),
        sa.CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_entity_mentions_confidence_range",
        ),
    )
    # The skip check runs per memory on every extraction job: are there already
    # mentions for this memory at the current version? Without this it is a
    # sequential scan on every job.
    op.create_index(
        "ix_entity_mentions_memory_version",
        "entity_mentions",
        ["memory_id", "extractor_version"],
    )
    # "The twenty most-mentioned entities", and every traversal that starts from
    # an entity and asks where it was seen.
    op.create_index("ix_entity_mentions_entity", "entity_mentions", ["entity_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_entity_mentions_entity", table_name="entity_mentions")
    op.drop_index("ix_entity_mentions_memory_version", table_name="entity_mentions")
    op.drop_table("entity_mentions")
    op.drop_index("ix_entities_canonical_name", table_name="entities")
    op.drop_table("entities")
