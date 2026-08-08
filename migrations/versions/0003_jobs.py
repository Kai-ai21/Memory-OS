"""jobs table

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-08

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_type", sa.Text(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("dedupe_key", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("priority", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default=sa.text("5"), nullable=False),
        sa.Column(
            "run_after",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("locked_by", sa.Text(), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_traceback", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_jobs_status",
        ),
        sa.CheckConstraint("attempts >= 0", name="ck_jobs_attempts_non_negative"),
        sa.CheckConstraint("max_attempts >= 1", name="ck_jobs_max_attempts_positive"),
        # A running job with no lease can never be reclaimed: the sweeper looks
        # for expired leases, and a null lease never expires.
        sa.CheckConstraint(
            "status <> 'running' OR (locked_by IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="ck_jobs_running_requires_lease",
        ),
    )

    # Mirrors the claim query's WHERE and ORDER BY exactly, so the planner can
    # take the ordering from the index rather than sorting. Partial, because in
    # a mature queue nearly every row is 'succeeded'; indexing those would be
    # dead weight that only ever grows.
    op.create_index(
        "ix_jobs_claim",
        "jobs",
        [sa.text("priority DESC"), "run_after"],
        postgresql_where=sa.text("status = 'pending'"),
    )

    # Idempotent enqueue: the same logical work cannot be queued twice while it
    # is still in flight. Once it settles, the key is free again.
    op.create_index(
        "uq_jobs_dedupe",
        "jobs",
        ["job_type", "dedupe_key"],
        unique=True,
        postgresql_where=sa.text("dedupe_key IS NOT NULL AND status IN ('pending', 'running')"),
    )

    # The sweeper's index.
    op.create_index(
        "ix_jobs_lease",
        "jobs",
        ["lease_expires_at"],
        postgresql_where=sa.text("status = 'running'"),
    )

    # Observability.
    op.create_index("ix_jobs_status_type", "jobs", ["status", "job_type"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_jobs_status_type", table_name="jobs")
    op.drop_index(
        "ix_jobs_lease", table_name="jobs", postgresql_where=sa.text("status = 'running'")
    )
    op.drop_index(
        "uq_jobs_dedupe",
        table_name="jobs",
        postgresql_where=sa.text("dedupe_key IS NOT NULL AND status IN ('pending', 'running')"),
    )
    op.drop_index(
        "ix_jobs_claim", table_name="jobs", postgresql_where=sa.text("status = 'pending'")
    )
    op.drop_table("jobs")
