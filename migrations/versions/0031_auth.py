"""M11.0: a real login for a single-user system.

Two tables, and the interesting decision is in the second one.

**`users`.** One row, ever. The single-user rule is enforced by `CreateUser`
rather than by a partial unique index on a constant, because M11.1 scopes the
existing corpus to a user and a schema that forbade a second row would have to
be migrated before that work could start. `email` is checked lowercase so a
login never has to guess the casing somebody typed on the day they made the
account.

**`sessions` stores a hash of the token, never the token.** The cookie carries
32 random bytes; this table carries their SHA-256. If the database leaks, stored
tokens would be live credentials and stored hashes are not — and it costs the
honest path nothing, because the server is handed the token on every request and
hashes it again. SHA-256 rather than Argon2 is correct here and only here: a
256-bit random token is not guessable, so the slowness that protects a password
would buy nothing and cost tens of milliseconds on every request.

Revocation is `revoked_at` rather than a delete. The row surviving is what lets
a logout be distinguishable from a session that never existed, and what a
future "your sessions" screen would read.

Both tables are `USER_AUTHORED`. Nothing in the ingestion log produces an
account or a login, so a replay that truncated them would lock the operator out
of their own system and leave no way back in but the CLI.

Revision ID: 0031
Revises: 0030
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0031"
down_revision: str | None = "0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

HEX64 = "^[0-9a-f]{64}$"


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.CheckConstraint("length(btrim(email)) > 0", name="ck_users_email_non_empty"),
        sa.CheckConstraint("email = lower(email)", name="ck_users_email_lowercase"),
        sa.CheckConstraint(
            "length(btrim(password_hash)) > 0", name="ck_users_password_hash_non_empty"
        ),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_sessions_user_id", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("token_hash", name="uq_sessions_token_hash"),
        sa.CheckConstraint(f"token_hash ~ '{HEX64}'", name="ck_sessions_token_hash_hex"),
        sa.CheckConstraint(
            "expires_at > created_at", name="ck_sessions_expiry_after_creation"
        ),
    )
    op.create_index("ix_sessions_user", "sessions", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_sessions_user", table_name="sessions")
    op.drop_table("sessions")
    op.drop_table("users")
