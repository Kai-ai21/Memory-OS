"""M10.2: a file dropped into the chat becomes a memory like everything else.

Two changes.

**`sources.kind` gains `upload`.** Not `filesystem`, and the distinction is load
bearing rather than tidy: a dropped file has no path this machine can walk back
to. The bytes arrived over HTTP, the browser's path means nothing here, and there
is nothing for a sync to observe a second time. Filing uploads under a filesystem
source would put rows in a source whose `config.root` cannot explain them — and
the first full sync of that root would find them absent and tombstone every one,
a deletion nobody performed arriving by way of the reconciliation that exists to
be correct.

**`chat_attachments` is a table, not a column.** A message may carry several
files and each one is its own memory: three files dropped together are three
documents that happen to share a moment, and merging them would make the content
hash a function of the batch, so re-uploading two of the three would mint a fourth
artifact nobody has.

It keys on `external_key` for the reason `chat_messages` does — `memories` is
derived, a replay truncates it and mints fresh ids, so an id is the one column
that cannot survive one. `content_hash` is stored beside it and is not redundant:
the artifact is shared by every upload of the same bytes, so the memory cannot say
whether *this* drop was the first, and `deduplicated` records the answer at the
moment it was known.

Revision ID: 0029
Revises: 0028
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0029"
down_revision: str | None = "0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_HEX64 = "^[0-9a-f]{64}$"


def upgrade() -> None:
    # Dropped and recreated rather than altered: Postgres has no `ALTER
    # CONSTRAINT` for a CHECK expression, and the name has to stay the same
    # because `alembic check` compares it against the one generated from
    # `SourceKind`.
    op.drop_constraint("ck_sources_kind", "sources", type_="check")
    op.create_check_constraint(
        "ck_sources_kind", "sources", "kind IN ('filesystem', 'chat', 'upload')"
    )

    # M10.1 asserted that a statement stores a memory and a question does not, as
    # a biconditional. An attachment-only turn breaks it: its files each become a
    # memory through `chat_attachments`, and the turn itself has no note, so
    # `external_key` is legitimately null on a `statement`.
    #
    # Weakened rather than moved to a trigger. The reverse direction now depends on
    # a row count in another table, which a CHECK cannot see, and a trigger that
    # could is business logic somewhere nobody looks. What survives is the half
    # whose violation would matter: a question must never store anything, because
    # that is the one that would put a question's own words into the retrieval set.
    op.drop_constraint(
        "ck_chat_messages_question_stores_nothing", "chat_messages", type_="check"
    )
    op.create_check_constraint(
        "ck_chat_messages_question_stores_nothing",
        "chat_messages",
        "intent <> 'question' OR external_key IS NULL",
    )

    op.create_table(
        "chat_attachments",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("message_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=True),
        sa.Column(
            "deduplicated", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chat_attachments"),
        # CASCADE to the message, which is the one place a cascade is right here:
        # an attachment row is *of* a turn and means nothing without it. The
        # memory it names is untouched either way — deleting the turn does not
        # delete the document, which is the whole reason the link is a key rather
        # than a foreign key.
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["chat_messages.id"],
            name="fk_chat_attachments_message_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "message_id", "ordinal", name="uq_chat_attachments_message_ordinal"
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_chat_attachments_ordinal_non_negative"),
        sa.CheckConstraint("byte_size >= 0", name="ck_chat_attachments_byte_size"),
        sa.CheckConstraint(
            "length(btrim(filename)) > 0", name="ck_chat_attachments_filename_non_empty"
        ),
        sa.CheckConstraint(
            f"content_hash ~ '{_HEX64}'", name="ck_chat_attachments_hash_hex"
        ),
    )
    op.create_index(
        "ix_chat_attachments_message", "chat_attachments", ["message_id", "ordinal"]
    )


def downgrade() -> None:
    op.drop_index("ix_chat_attachments_message", table_name="chat_attachments")
    op.drop_table("chat_attachments")

    # Upload sources go with the constraint, and their memories with them. Stated
    # and refused rather than handled: there is no path to re-read a dropped file
    # from, so deleting them destroys the only copy of the row — while leaving them
    # would leave memories in a source whose kind the constraint forbids. This
    # raises instead of choosing, exactly as 0027 does for chat.
    rows = op.get_bind().execute(
        sa.text("SELECT count(*) FROM sources WHERE kind = 'upload'")
    ).scalar_one()
    if rows:
        raise RuntimeError(
            f"{rows} upload source(s) exist and the pre-0029 schema cannot hold them. "
            "An uploaded file has no path to be re-read from, so this migration will "
            "not delete them for you. Export or drop them deliberately, then rerun."
        )
    op.drop_constraint(
        "ck_chat_messages_question_stores_nothing", "chat_messages", type_="check"
    )
    op.create_check_constraint(
        "ck_chat_messages_question_stores_nothing",
        "chat_messages",
        "intent IS NULL OR (intent = 'question') = (external_key IS NULL)",
    )

    op.drop_constraint("ck_sources_kind", "sources", type_="check")
    op.create_check_constraint(
        "ck_sources_kind", "sources", "kind IN ('filesystem', 'chat')"
    )
