"""M10.1: conversations that persist, resume, and can be revisited.

**A session is a view, not a container**, and the schema says so by what it does
not do. No memory is created for a session, no message is merged into one, and
nothing that decides meaning — retrieval, the graph, the timeline — joins to
either table here. M10.0 stored each message as its own memory so that a thought
from Tuesday connects to one from last month through the entities they share; that
arrangement is untouched. What a session decides is what to draw in a list and
which three turns to carry into the next question.

**`chat_messages` goes from one row per exchange to one row per turn.** M10.0 put
the question and its answer on the same row, with `answer` and `answer_refused`
columns beside `text`. That was fine for a flat stream and stops being fine the
moment there is an order to keep: `ordinal` has to be a total order over a
conversation, and a row that is two turns cannot have one position. It also made
"the last three turns" a rule about how many halves of a row to count rather than
a `LIMIT`.

So this migration splits every existing row in two where it carried an answer, and
groups the lot into sessions by the same thirty-minute silence the application
uses. The data is real — it was typed — and it is the only copy of how those
messages were read, so it is transformed rather than dropped.

**`external_key` stays, and the milestone's own requirement is why.** M10.1 asks
that these rows survive a replay while the memories they point at are rebuilt, and
that the link key on nothing that changes. A memory id changes on every replay:
`memories` is derived, a replay truncates it, and every id is minted fresh — so an
id is exactly the wrong column, and `TRUNCATE memories CASCADE` would take this
table with it whatever set it is classified in. The external key is what the
ingestion log records, so the memory comes back carrying the same one and the join
finds it again having preserved nothing.

Revision ID: 0028
Revises: 0027
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0028"
down_revision: str | None = "0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The same silence `domain/sessions.py` measures. Duplicated as an interval
# literal rather than imported, because a migration has to keep meaning what it
# meant on the day it ran: importing the constant would make this backfill change
# retroactively the next time somebody tunes the gap.
_GAP = "30 minutes"


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("started_at", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("last_activity", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "message_count", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("archived_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_chat_sessions"),
        sa.CheckConstraint("message_count >= 0", name="ck_chat_sessions_count_non_negative"),
        sa.CheckConstraint(
            "last_activity >= started_at", name="ck_chat_sessions_activity_after_start"
        ),
    )
    op.create_index(
        "ix_chat_sessions_live",
        "chat_sessions",
        ["last_activity"],
        postgresql_where=sa.text("archived_at IS NULL"),
    )

    # --- the split ----------------------------------------------------------
    #
    # Built alongside the old table and swapped in, rather than altered in place.
    # The transformation is one row becoming two, which no sequence of ALTERs
    # expresses, and a temporary table means a failure halfway leaves the original
    # intact.
    op.create_table(
        "chat_messages_new",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("session_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=True),
        sa.Column("intent", sa.Text(), nullable=True),
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
    )

    # Sessions from the existing transcript, by the thirty-minute rule.
    #
    # `sum(...) OVER (ORDER BY created_at)` over a boolean-as-int is the standard
    # gaps-and-islands shape: each row that opens a session contributes 1, so the
    # running total is a session number that every row in the same run shares.
    op.execute(
        sa.text(
            f"""
            CREATE TEMPORARY TABLE _grouped AS
            WITH marked AS (
                SELECT id,
                       text,
                       intent,
                       answer,
                       answer_model,
                       answer_refused,
                       citations,
                       created_at,
                       external_key,
                       CASE
                           WHEN lag(created_at) OVER (ORDER BY created_at, id) IS NULL
                             OR created_at - lag(created_at) OVER (ORDER BY created_at, id)
                                >= INTERVAL '{_GAP}'
                           THEN 1 ELSE 0
                       END AS opens
                  FROM chat_messages
            )
            SELECT *,
                   sum(opens) OVER (ORDER BY created_at, id) AS session_number
              FROM marked
            """
        )
    )

    op.execute(
        sa.text(
            """
            CREATE TEMPORARY TABLE _sessions AS
            SELECT session_number,
                   gen_random_uuid() AS session_id,
                   min(created_at) AS started_at,
                   max(created_at) AS last_activity
              FROM _grouped
             GROUP BY session_number
            """
        )
    )

    op.execute(
        sa.text(
            """
            INSERT INTO chat_sessions
                   (id, title, started_at, last_activity, message_count)
            SELECT s.session_id,
                   -- The first user message, cut at 60 characters. A cruder rule
                   -- than `domain.sessions.title_for` — no clause boundary, no
                   -- ellipsis — and deliberately so: a backfill that reimplemented
                   -- the live rule would be a second copy of it, drifting from the
                   -- day it was written. These titles are re-derivable by hand and
                   -- nothing reads them but a human.
                   (SELECT left(g.text, 60)
                      FROM _grouped g
                     WHERE g.session_number = s.session_number
                     ORDER BY g.created_at, g.id
                     LIMIT 1),
                   s.started_at,
                   s.last_activity,
                   0
              FROM _sessions s
            """
        )
    )

    # Every old row becomes a user turn, and the ones that carried an answer
    # become an assistant turn as well. `ordinal` is assigned over the pair
    # ordering so the answer immediately follows its question.
    op.execute(
        sa.text(
            """
            WITH turns AS (
                SELECT s.session_id,
                       g.created_at,
                       g.id AS source_id,
                       0 AS half,
                       'user' AS role,
                       g.text AS content,
                       g.external_key,
                       g.intent,
                       NULL::text AS answer_model,
                       NULL::boolean AS answer_refused,
                       '[]'::jsonb AS citations
                  FROM _grouped g
                  JOIN _sessions s USING (session_number)
                 UNION ALL
                SELECT s.session_id,
                       g.created_at,
                       g.id AS source_id,
                       1 AS half,
                       'assistant' AS role,
                       g.answer AS content,
                       NULL::text AS external_key,
                       NULL::text AS intent,
                       g.answer_model,
                       g.answer_refused,
                       g.citations
                  FROM _grouped g
                  JOIN _sessions s USING (session_number)
                 WHERE g.answer IS NOT NULL AND btrim(g.answer) <> ''
            )
            INSERT INTO chat_messages_new
                   (id, session_id, role, content, ordinal, external_key, intent,
                    answer_model, answer_refused, citations, created_at)
            SELECT gen_random_uuid(),
                   session_id,
                   role,
                   content,
                   row_number() OVER (
                       PARTITION BY session_id ORDER BY created_at, source_id, half
                   ) - 1,
                   external_key,
                   intent,
                   answer_model,
                   answer_refused,
                   citations,
                   created_at
              FROM turns
            """
        )
    )

    op.execute(
        sa.text(
            """
            UPDATE chat_sessions s
               SET message_count = (
                       SELECT count(*) FROM chat_messages_new m WHERE m.session_id = s.id
                   )
            """
        )
    )

    op.drop_index("ix_chat_messages_created_at", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.rename_table("chat_messages_new", "chat_messages")

    # Constraints after the load rather than during it. The backfill above is
    # correct by construction, and adding them here means one validation pass over
    # finished data instead of a check per inserted row.
    op.create_primary_key("pk_chat_messages", "chat_messages", ["id"])
    op.create_foreign_key(
        "fk_chat_messages_session_id",
        "chat_messages",
        "chat_sessions",
        ["session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint(
        "uq_chat_messages_session_ordinal", "chat_messages", ["session_id", "ordinal"]
    )
    op.create_check_constraint(
        "ck_chat_messages_role", "chat_messages", "role IN ('user', 'assistant')"
    )
    op.create_check_constraint(
        "ck_chat_messages_ordinal_non_negative", "chat_messages", "ordinal >= 0"
    )
    op.create_check_constraint(
        "ck_chat_messages_content_non_empty", "chat_messages", "length(btrim(content)) > 0"
    )
    op.create_check_constraint(
        "ck_chat_messages_assistant_stores_nothing",
        "chat_messages",
        "role <> 'assistant' OR (intent IS NULL AND external_key IS NULL)",
    )
    op.create_check_constraint(
        "ck_chat_messages_user_is_not_an_answer",
        "chat_messages",
        "role <> 'user' OR (answer_model IS NULL AND answer_refused IS NULL "
        "AND citations = '[]'::jsonb)",
    )
    op.create_check_constraint(
        "ck_chat_messages_user_has_intent",
        "chat_messages",
        "(role = 'user') = (intent IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_chat_messages_question_stores_nothing",
        "chat_messages",
        "intent IS NULL OR (intent = 'question') = (external_key IS NULL)",
    )
    op.create_index("ix_chat_messages_session", "chat_messages", ["session_id", "ordinal"])
    op.create_index("ix_chat_messages_created_at", "chat_messages", ["created_at"])


def downgrade() -> None:
    """Back to one row per exchange, rejoining each answer to its question.

    Lossy in one specific way, stated rather than hidden: a session that was
    *explicitly* started inside the thirty-minute window is indistinguishable from
    a continuation once the sessions table is gone, so the boundary a person drew
    by hand is not recoverable. Everything else round-trips — an assistant turn
    goes back onto the user turn above it, and the ordering is preserved by
    `created_at`.
    """
    op.create_table(
        "chat_messages_old",
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
    )
    op.execute(
        sa.text(
            """
            INSERT INTO chat_messages_old
                   (id, text, intent, external_key, answer, answer_model,
                    answer_refused, citations, created_at)
            SELECT u.id,
                   u.content,
                   u.intent,
                   u.external_key,
                   a.content,
                   a.answer_model,
                   a.answer_refused,
                   coalesce(a.citations, '[]'::jsonb),
                   u.created_at
              FROM chat_messages u
              LEFT JOIN chat_messages a
                     ON a.session_id = u.session_id
                    AND a.ordinal = u.ordinal + 1
                    AND a.role = 'assistant'
             WHERE u.role = 'user'
            """
        )
    )

    op.drop_index("ix_chat_messages_created_at", table_name="chat_messages")
    op.drop_index("ix_chat_messages_session", table_name="chat_messages")
    op.drop_table("chat_messages")
    op.rename_table("chat_messages_old", "chat_messages")

    op.create_primary_key("pk_chat_messages", "chat_messages", ["id"])
    op.create_check_constraint(
        "ck_chat_messages_intent", "chat_messages", "intent IN ('statement', 'question', 'both')"
    )
    op.create_check_constraint(
        "ck_chat_messages_text_non_empty", "chat_messages", "length(btrim(text)) > 0"
    )
    op.create_check_constraint(
        "ck_chat_messages_question_stores_nothing",
        "chat_messages",
        "(intent = 'question') = (external_key IS NULL)",
    )
    op.create_check_constraint(
        "ck_chat_messages_statement_is_not_answered",
        "chat_messages",
        "intent <> 'statement' OR answer IS NULL",
    )
    op.create_check_constraint(
        "ck_chat_messages_answer_verdict",
        "chat_messages",
        "(answer IS NULL) = (answer_refused IS NULL)",
    )
    op.create_index("ix_chat_messages_created_at", "chat_messages", ["created_at"])

    op.drop_index("ix_chat_sessions_live", table_name="chat_sessions")
    op.drop_table("chat_sessions")
