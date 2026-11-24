"""M12.2: a second walkable source kind, a notes vault.

Only the CHECK on `sources.kind` changes. A vault is walked by the same
machinery as a directory and its memories go through the same pipeline, so
nothing else in the schema knows the difference — which is the property this
milestone exists to test.

Revision ID: 0035
Revises: 0034
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0035"
down_revision: str | None = "0034"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Dropped and recreated: Postgres has no `ALTER CONSTRAINT` for a CHECK, and
    # the name must stay the one `alembic check` generates from `SourceKind`.
    op.drop_constraint("ck_sources_kind", "sources", type_="check")
    op.create_check_constraint(
        "ck_sources_kind", "sources", "kind IN ('filesystem', 'chat', 'upload', 'vault')"
    )


def downgrade() -> None:
    # Refused rather than handled, as 0027 and 0029 do: deleting a vault source
    # would delete its memories, and that is not a migration's decision to make.
    rows = op.get_bind().execute(
        sa.text("SELECT count(*) FROM sources WHERE kind = 'vault'")
    ).scalar_one()
    if rows:
        raise RuntimeError(
            f"{rows} vault source(s) exist and the pre-0035 schema cannot hold them. "
            "Delete them with `memoryos source delete`, then rerun."
        )
    op.drop_constraint("ck_sources_kind", "sources", type_="check")
    op.create_check_constraint(
        "ck_sources_kind", "sources", "kind IN ('filesystem', 'chat', 'upload')"
    )
