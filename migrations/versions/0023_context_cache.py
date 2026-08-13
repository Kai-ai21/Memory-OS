"""context_cache: an assembled context, and the fingerprint that expires it

Revision ID: 0023
Revises: 0022
Create Date: 2026-08-13

**The only cache in this schema whose entries can be wrong rather than merely
stale**, and that difference shapes the table.

`embedding_cache` is content-addressed: an entry is a pure function of (model,
role, text), so a retained one is correct by construction and M1.7 keeps it
across a replay for exactly that reason. A context is a function of the *whole
corpus*. Ingest one file and every context built before it is a confident answer
whose evidence has moved — and nothing about the answer looks different.

So `cache_key` is a hash of the focus, the budget, and a fingerprint of the
corpus. A sync changes the fingerprint, every key changes with it, and no writer
anywhere has to know which focuses its change affected. That is deliberately
coarse: a per-focus dependency set would invalidate less and would have to be
maintained by every writer in the system, and getting it wrong means serving
context that silently omits the file somebody just edited. Over-invalidation
costs a re-assembly of a few hundred milliseconds.

`expires_at` is a second and weaker rule answering a different question. The
fingerprint handles staleness of content; the TTL handles staleness of intent. A
context assembled for a meeting three days ago is answering something nobody is
asking now.

`hit_count` is what M6.1 is judged on. Precomputing context for triggers nobody
reads burns compute continuously and produces mostly waste, so the hit rate is
the evidence for whether precomputation earns its cost. Counted in the database
on the read rather than in a log line, because a rate assembled from logs is a
rate nobody can query afterwards.

`payload` holds the rendered context rather than a list of ids. Ids would mean
re-reading and re-rendering every item on a hit — most of the cost the cache
exists to avoid — and would let a hit return text that no longer matches what
selection actually saw.

Classified **operational**, the fourth category M6.0 added. Nothing is rebuilt
from a cached context, no replay reproduces one, nobody wrote it, and dropping
the table costs a re-assembly. It is the clearest member that set has.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0023"
down_revision: str | Sequence[str] | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "context_cache",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        # Focus plus budget plus the corpus fingerprint, hashed. Unique because
        # it *is* the identity: two rows under one key would be two answers to
        # one question with nothing to choose between them.
        sa.Column("cache_key", sa.Text(), nullable=False),
        # Kept beside the key it is hashed into. A hash nobody can reverse is a
        # row nobody can debug.
        sa.Column("focus", sa.Text(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column(
            "built_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "hit_count", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_context_cache"),
        sa.UniqueConstraint("cache_key", name="uq_context_cache_key"),
        sa.CheckConstraint("length(btrim(focus)) > 0", name="ck_context_cache_focus"),
        sa.CheckConstraint("token_count >= 0", name="ck_context_cache_tokens"),
        sa.CheckConstraint("hit_count >= 0", name="ck_context_cache_hits"),
        # A row that expires before it was built is a clock or TTL bug, and it
        # would be invisible otherwise: the entry simply never serves a hit.
        sa.CheckConstraint(
            "expires_at > built_at", name="ck_context_cache_expiry_order"
        ),
    )
    # For the sweep that removes dead rows. The read path is served by the
    # unique index on `cache_key`; this is for the table most likely to
    # accumulate entries nobody will ever ask for again.
    op.create_index("ix_context_cache_expires", "context_cache", ["expires_at"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_context_cache_expires", table_name="context_cache")
    op.drop_table("context_cache")
