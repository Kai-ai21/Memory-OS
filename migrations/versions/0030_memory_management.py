"""M10.4: correcting, deleting and organising memories from the chat.

Four changes, and three of them exist to make a deletion honest.

**`ingestion_events.event_type` gains `item_purged`.** The log is append-only, so
a permanent deletion cannot be a missing row — it has to be a present one saying
the content was removed. That is the crypto-shredding tension Phase 1 named,
resolved in the only direction an append-only log allows: the event survives, and
records that bytes with this hash and this size were once observed at this key. The
memory, every version of it, its chunks, its vectors, its mentions, its graph
nodes, its transcript rows and the blob itself do not. Replay reads the purge and
skips the key entirely, including the observations before it, whose blobs are
deliberately gone.

**`chat_messages.corrects_message_id`.** M10.0 made turns immutable, so a
correction is a new row rather than an edit, and both stay visible: the original
marked superseded, the correction pointing back at it. What somebody believed
before they corrected it is the data Phase 5 reasons over, and an in-place edit
would delete exactly that. Both rows carry the same `external_key`, which is what
makes the corpus side a version bump — one item with a history — rather than two
items disagreeing.

**Two CHECK constraints on that column, and two indexes.** A message may not
correct itself, and an assistant turn may not correct anything: re-answering is
asking again, not correcting. The partial index on `external_key` is what a purge
finds its transcript rows by.

**`memory_tags`.** A tag is a link to a `CONCEPT` entity, not a taxonomy of its
own — `#postgres` resolves to the concept M3.2 already knows, so a tag connects to
everything that mentions Postgres. Only the link is stored here, keyed by *name*
on both ends and by nothing derived: `entities` is truncated by every replay and
refilled by extraction, so a foreign key into it would delete every tag anybody
had applied on the next rebuild; and `(source_id, external_key)` rather than
`memory_id` because a correction mints a new version, and a tag keyed on a version
would stop applying the moment somebody fixed a typo.

Revision ID: 0030
Revises: 0029
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0030"
down_revision: str | None = "0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Dropped and recreated rather than altered: Postgres has no `ALTER
    # CONSTRAINT` for a CHECK expression, and the name has to stay the same
    # because `alembic check` compares it against the one generated from
    # `EventType`.
    op.drop_constraint(
        "ck_ingestion_events_event_type", "ingestion_events", type_="check"
    )
    op.create_check_constraint(
        "ck_ingestion_events_event_type",
        "ingestion_events",
        "event_type IN ('artifact_observed', 'item_deleted', 'item_purged')",
    )

    op.add_column(
        "chat_messages",
        sa.Column("corrects_message_id", sa.Uuid(as_uuid=True), nullable=True),
    )
    # SET NULL rather than CASCADE. A purge deletes every turn sharing the purged
    # key, so the pair goes together in practice; if some other path ever removes
    # only the original, the correction is still a message somebody sent, and
    # deleting it would be this column reaching further than its meaning.
    op.create_foreign_key(
        "fk_chat_messages_corrects_message_id",
        "chat_messages",
        "chat_messages",
        ["corrects_message_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "ck_chat_messages_not_correcting_self",
        "chat_messages",
        "corrects_message_id IS NULL OR corrects_message_id <> id",
    )
    op.create_check_constraint(
        "ck_chat_messages_only_user_corrects",
        "chat_messages",
        "corrects_message_id IS NULL OR role = 'user'",
    )
    op.create_index(
        "ix_chat_messages_corrects",
        "chat_messages",
        ["corrects_message_id"],
        postgresql_where=sa.text("corrects_message_id IS NOT NULL"),
    )
    op.create_index(
        "ix_chat_messages_external_key",
        "chat_messages",
        ["external_key"],
        postgresql_where=sa.text("external_key IS NOT NULL"),
    )

    op.create_table(
        "memory_tags",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("source_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=False),
        sa.Column("tag", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_memory_tags"),
        # The one foreign key here, and it points at a table no replay truncates.
        # `external_key` is only unique within a source — two sources may each hold
        # a `notes/queue.md` — so the source has to be part of the identity.
        sa.ForeignKeyConstraint(
            ["source_id"], ["sources.id"], name="fk_memory_tags_source_id"
        ),
        sa.UniqueConstraint(
            "source_id", "external_key", "tag", name="uq_memory_tags_item_tag"
        ),
        sa.CheckConstraint("length(btrim(tag)) > 0", name="ck_memory_tags_tag_non_empty"),
        sa.CheckConstraint(
            "length(btrim(label)) > 0", name="ck_memory_tags_label_non_empty"
        ),
        # Both of these are the join to `entities.canonical_name` defending itself.
        # It is an equality, so one uppercase character or one surviving `#` written
        # by any path — psql included — is a tag that resolves to nothing, silently.
        sa.CheckConstraint("tag = lower(tag)", name="ck_memory_tags_tag_casefolded"),
        sa.CheckConstraint("tag NOT LIKE '#%'", name="ck_memory_tags_tag_without_sigil"),
    )
    op.create_index("ix_memory_tags_tag", "memory_tags", ["tag"])
    op.create_index("ix_memory_tags_item", "memory_tags", ["source_id", "external_key"])


def downgrade() -> None:
    op.drop_index("ix_memory_tags_item", table_name="memory_tags")
    op.drop_index("ix_memory_tags_tag", table_name="memory_tags")
    op.drop_table("memory_tags")

    op.drop_index("ix_chat_messages_external_key", table_name="chat_messages")
    op.drop_index("ix_chat_messages_corrects", table_name="chat_messages")
    op.drop_constraint(
        "ck_chat_messages_only_user_corrects", "chat_messages", type_="check"
    )
    op.drop_constraint(
        "ck_chat_messages_not_correcting_self", "chat_messages", type_="check"
    )
    op.drop_constraint(
        "fk_chat_messages_corrects_message_id", "chat_messages", type_="foreignkey"
    )
    # The corrections themselves survive as ordinary turns, which is the honest
    # outcome: the text of both versions is in the transcript either way, and the
    # pre-0030 schema simply cannot say which one superseded which. Dropping the
    # column loses the link, not the words.
    op.drop_column("chat_messages", "corrects_message_id")

    # Purge events cannot be represented before this revision, and unlike the
    # constraint above they are not a link — they are the only record that a
    # permanent deletion happened. Refused rather than deleted: dropping them would
    # make the log claim the content is still there, and a replay against that log
    # would try to rebuild memories whose blobs were deliberately shredded and fail
    # on the first one. 0029 refuses for the same class of reason.
    rows = op.get_bind().execute(
        sa.text("SELECT count(*) FROM ingestion_events WHERE event_type = 'item_purged'")
    ).scalar_one()
    if rows:
        raise RuntimeError(
            f"{rows} purge event(s) are in the log and the pre-0030 schema cannot "
            "hold them. They are the only record that a permanent deletion "
            "happened, and their blobs are gone, so this migration will not delete "
            "them for you — a replay against the resulting log would fail on the "
            "first purged artifact. Export the log first if you need this."
        )
    op.drop_constraint(
        "ck_ingestion_events_event_type", "ingestion_events", type_="check"
    )
    op.create_check_constraint(
        "ck_ingestion_events_event_type",
        "ingestion_events",
        "event_type IN ('artifact_observed', 'item_deleted')",
    )
