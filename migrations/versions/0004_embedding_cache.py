"""embedding cache

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-08

"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "embedding_cache",
        # BLAKE2b-256 of "<model_id>\\0<text>". The model is inside the key
        # because reusing a vector across models is not a stale cache, it is a
        # different coordinate system silently mixed into the same index.
        sa.Column("cache_key", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(384), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("cache_key", name="pk_embedding_cache"),
        sa.CheckConstraint(
            "cache_key ~ '^[0-9a-f]{64}$'", name="ck_embedding_cache_key_hex"
        ),
        sa.CheckConstraint("dimension > 0", name="ck_embedding_cache_dimension_positive"),
    )
    # A re-embed sweeps by model, not by key.
    op.create_index("ix_embedding_cache_model", "embedding_cache", ["model_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_embedding_cache_model", table_name="embedding_cache")
    op.drop_table("embedding_cache")
