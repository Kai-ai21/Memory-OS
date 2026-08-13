"""external events: something happened outside, and work nobody asked for

Revision ID: 0022
Revises: 0021
Create Date: 2026-08-13

**The first table in this schema whose rows arrive rather than being written.**
Everything before it is the product of somebody using the system — a source they
registered, a decision they recorded, a query they judged. This is a plugin
firing on a keystroke, and the difference shows up in the indexes rather than in
the columns.

Phase 6 inverts the direction of the whole system. Through Phase 5 it is pull:
you ask, it answers, and a bad answer wastes one query. From here it is push:
something happens, the system decides work is worth doing, and does it unasked. A
bad push wastes compute continuously and teaches its reader to ignore the output.
M6.0 predicts nothing, so it cannot be wrong that way yet; what it can get wrong
is volume, and two of the four indexes below are the defence.

**`uq_events_pending_dedupe` is the milestone's actual content.** A partial
unique index on `(kind, dedupe_key)` restricted to `processed_at IS NULL`, so the
same editor-focus event fired ten times in a minute produces one unit of work.
Restricted to *unprocessed* rows deliberately: the same file focused again
tomorrow is genuinely new work, and an index over every row would refuse the
second focus forever. `dedupe_key IS NOT NULL` is in the predicate too — without
it, Postgres treats nulls as distinct and the index would be dead weight over
every keyless row.

**`ix_events_source_received` is what makes rate limiting affordable.** The limit
is counted against stored rows rather than held in a token bucket, because a
bucket in the API process is per replica and resets on deploy while the queue it
protects is shared. Counting costs one indexed range scan per POST and cannot be
wrong about what was actually accepted.

Bitemporal, and M1.1's rules apply unchanged. `occurred_at` is when the thing
happened; `received_at` is when this system heard about it. There is no
`occurred_at_source` because there is only one way an event gets its time — the
client asserts it — and a client that asserts nothing gets `received_at`, making
the two equal. That equality is the provenance: it means nobody said when.

`processed_at` stays null until a handler has run to completion, and is
deliberately not set at dispatch. Dispatch only enqueues; an event whose jobs are
all still pending has had nothing done about it, and marking it processed then
would make `events stats` report the speed of an INSERT.

**Classified operational** — a fourth category in `application/replay.py`, and
the one M1.7's retrospective said the two-set binary was already hiding. An event
is not source-of-truth (nothing rebuilds from it), not derived (no replay
reproduces a keystroke), and not user-authored (nobody wrote it). It is
discardable, which is a property none of the other three carry. `jobs` moves into
the same set, where it always belonged.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0022"
down_revision: str | Sequence[str] | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "events",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        # Which client emitted it. Free text rather than an enum: clients are
        # added by whoever writes a plugin, and an enum would mean a migration
        # before a new editor could say anything at all.
        sa.Column("source", sa.Text(), nullable=False),
        # Not validated beyond being an object. No handler reads a field of it
        # yet, and a schema invented before its first consumer exists is a
        # schema that will be wrong when one arrives.
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dedupe_key", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_events"),
        # Refused at the edge as well, with a message naming every valid kind.
        # The constraint is the floor: a row nothing subscribes to would sit
        # unprocessed forever, and a queue full of those cannot be told from a
        # queue that is behind.
        sa.CheckConstraint(
            "kind IN ('editor_opened', 'file_focused', 'meeting_upcoming', 'manual')",
            name="ck_events_kind",
        ),
        sa.CheckConstraint("length(btrim(source)) > 0", name="ck_events_source"),
    )
    op.create_index(
        "ix_events_kind_received", "events", ["kind", "received_at"], unique=False
    )
    op.create_index(
        "ix_events_source_received", "events", ["source", "received_at"], unique=False
    )
    op.create_index(
        "ix_events_unprocessed",
        "events",
        ["received_at"],
        unique=False,
        postgresql_where=sa.text("processed_at IS NULL"),
    )
    op.create_index(
        "uq_events_pending_dedupe",
        "events",
        ["kind", "dedupe_key"],
        unique=True,
        postgresql_where=sa.text("processed_at IS NULL AND dedupe_key IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "uq_events_pending_dedupe",
        table_name="events",
        postgresql_where=sa.text("processed_at IS NULL AND dedupe_key IS NOT NULL"),
    )
    op.drop_index(
        "ix_events_unprocessed",
        table_name="events",
        postgresql_where=sa.text("processed_at IS NULL"),
    )
    op.drop_index("ix_events_source_received", table_name="events")
    op.drop_index("ix_events_kind_received", table_name="events")
    op.drop_table("events")
