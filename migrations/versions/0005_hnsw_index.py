"""hnsw index on chunk embeddings

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-08

`vector_ip_ops` — inner product — because M1.5's embedder normalizes to unit
length. For unit vectors, cosine similarity and inner product are the same
number, and inner product skips a division that would buy nothing. The adapter
asserts `embedder.normalizes` at startup, because with non-normalized vectors
this operator does not error; it silently returns a wrong ranking.

**On a large corpus, create this index after bulk loading, not before.**
Building on an empty table and maintaining it through every insert is slower
overall and yields a worse-connected graph than one build over the finished
data. M1.1 deliberately left it out for this reason; it arrives now, after the
pipeline that fills the column.

`m` is edges per node and `ef_construction` is the build-time search width;
both are fixed at build time. The knob actually worth tuning is
`hnsw.ef_search`, which is a session setting the adapter sets per query —
higher for recall, lower for latency. `memoryos eval-recall` measures that
trade rather than guessing at it.

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "CREATE INDEX ix_memory_chunks_embedding_hnsw "
        "ON memory_chunks USING hnsw (embedding vector_ip_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_memory_chunks_embedding_hnsw")
