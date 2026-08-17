"""M10.0: a source that nothing walks, and the transcript of what was typed.

Two changes, and the small one is the more interesting.

**`sources.kind` gains `chat`.** Every kind before this named something a
connector enumerates; `chat` names something that is pushed, one message at a
time, with no cursor and no second observation. It is a source anyway because
`memories.source_id` is not optional and every scope in Phase 4 is a source id —
a memory recorded some other way would be invisible to the timeline, to
`--source` on every command, and to the graph's `FROM_SOURCE` edge. Widening a
CHECK constraint is a smaller exception than a memory with no origin.

**`chat_messages` is the transcript, and is deliberately not the memory.** A
stored message becomes a `memories` row through the same pipeline a file does —
content hash, artifact, ingestion event, memory, chunks, embedding, entities —
and this table names it. What it adds is the two things the pipeline has nowhere
to put: how the message was *read* (statement, question, or both), and, for a
question, the answer that came back.

It names the memory by `external_key` rather than by a foreign key, and that is
the replay classification deciding a schema rather than a preference. `memories`
is derived; a full replay truncates it, and `TRUNCATE ... CASCADE` reaches every
table referencing it whatever set this one is classified in. `decision_evidence`
survives that with a snapshot taken before the truncation and re-linked after —
machinery built for *link* rows, which these are not, and which here would mean
deleting and reinserting the entire transcript on every rebuild. The key needs
none of it: the ingestion log records it, so the memory comes back carrying the
same one and the join finds it again with nothing having been preserved. Names
outlive ids, which is the argument M1.7 made for the golden set.

The answer being here is the part worth defending. M10.0 is explicit that
answers are not stored as memories, because generated text that can be retrieved
becomes evidence for the next generation, and a corpus that cites itself decays
in a way nothing reports. The guarantee is kept structurally rather than by
intention: this table has no content hash, no chunks, no embedding and no
tsvector, and no retriever joins to it. It is what makes the message list survive
a reload and what makes a follow-up question resolvable. That is a transcript,
not evidence.

`intent` is a stored column rather than something recomputed on read, and that
is the correction path working. The classifier is rules over a regex and will
misread things; the reading that was *acted on* has to stay recoverable, because
recomputing it later under a newer rule set would quietly rewrite history to
agree with the present.

Revision ID: 0027
Revises: 0026
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0027"
down_revision: str | None = "0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Dropped and recreated rather than altered: Postgres has no `ALTER
    # CONSTRAINT` for a CHECK expression, and the name has to stay the same
    # because `alembic check` compares it against the one generated from
    # `SourceKind`.
    op.drop_constraint("ck_sources_kind", "sources", type_="check")
    op.create_check_constraint(
        "ck_sources_kind", "sources", "kind IN ('filesystem', 'chat')"
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("answer_model", sa.Text(), nullable=True),
        sa.Column("answer_refused", sa.Boolean(), nullable=True),
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chat_messages"),
        sa.CheckConstraint(
            "intent IN ('statement', 'question', 'both')", name="ck_chat_messages_intent"
        ),
        sa.CheckConstraint(
            "length(btrim(text)) > 0", name="ck_chat_messages_text_non_empty"
        ),
        sa.CheckConstraint(
            "(intent = 'question') = (external_key IS NULL)",
            name="ck_chat_messages_question_stores_nothing",
        ),
        sa.CheckConstraint(
            "intent <> 'statement' OR answer IS NULL",
            name="ck_chat_messages_statement_is_not_answered",
        ),
        sa.CheckConstraint(
            "(answer IS NULL) = (answer_refused IS NULL)",
            name="ck_chat_messages_answer_verdict",
        ),
    )
    # The message list reads it descending and the conversational context reads
    # the last three turns. One index, both reads.
    op.create_index("ix_chat_messages_created_at", "chat_messages", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_chat_messages_created_at", table_name="chat_messages")
    op.drop_table("chat_messages")

    # Chat sources go with it, and so does every memory that came through one.
    # Stated rather than handled: there is no filesystem path to re-derive a
    # typed message from, so a downgrade that left the rows behind would leave
    # memories pointing at a source kind the constraint forbids, and one that
    # deletes them destroys the only copy. This raises instead of choosing.
    rows = op.get_bind().execute(
        sa.text("SELECT count(*) FROM sources WHERE kind = 'chat'")
    ).scalar_one()
    if rows:
        raise RuntimeError(
            f"{rows} chat source(s) exist and the pre-0027 schema cannot hold them. "
            "A typed message has no file to be re-read from, so this migration will "
            "not delete them for you. Export or drop them deliberately, then rerun."
        )
    op.drop_constraint("ck_sources_kind", "sources", type_="check")
    op.create_check_constraint("ck_sources_kind", "sources", "kind IN ('filesystem')")
