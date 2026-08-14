"""surfacing_log: what was volunteered, what was refused, and what came back

Revision ID: 0024
Revises: 0023
Create Date: 2026-08-14

**One row per decision, not one row per interruption**, and that is the only
place this table departs from the shape the milestone specified. The reason is
the question a proactive system has to be able to answer: *why didn't it show me
anything?* A table of things that were shown cannot answer it. Silence would look
identical whether the gate refused, the corpus was empty, or the handler never
ran at all — and a system whose silence is unexplainable is one nobody can tell
apart from a broken one.

So `surfaced_at` is nullable and `reason` is not. A row with `surfaced_at IS
NULL` is a refusal, carrying the score it reached and the threshold it did not,
and `memoryos surfacing log` reads them back. The requested columns are all here
and mean what they were asked to mean; there are simply more of them.

Two CHECK constraints keep those columns from disagreeing:

* `reason = 'cleared'` exactly when `surfaced_at IS NOT NULL`. The verdict and
  the outcome are stored separately because the report needs both, and two
  columns that must agree will eventually not.
* Feedback only on something that was surfaced, and never both kinds. "Useful"
  and "dismissed" are the same click made two ways, and a row holding both is a
  row that would be counted twice in the dismissal rate — the one number this
  milestone is judged on.

**Classified user-authored, and that is the interesting call.** By its origin
this table is pure exhaust: nothing rebuilds from it, no replay reproduces a
decision the gate made, and by that test it belongs beside `events` and
`context_cache` in `OPERATIONAL_TABLES`. Two columns say otherwise.
`dismissed_at` and `acted_on_at` are a person having judged what they were shown,
which exists nowhere else — and `application/surfacing.py` *reads* them, to raise
the bar on a focus whose context keeps getting refused. A replay that truncated
this table would un-dismiss every dismissal and reset every adapted threshold to
its default, so the next trigger would surface exactly what somebody had already
told it not to. That is the failure this milestone exists to prevent, arriving by
way of the replay.

It is the same argument `patterns` settled in M5.3, one milestone on: most of the
table is derivable, two columns are not, and the two decide. The refusal rows
come along for the ride and cost a few bytes each.

`item_keys` is stored beside the hash rather than instead of it because the two
answer different questions. The hash is identity — is this exactly what you saw —
and the keys are what makes similarity computable: one item different is a
different hash and the same interruption, so suppression compares overlap. See
`domain/surfacing.overlap`.

No foreign key to `events`. The trigger id is recorded, but `events` is
operational and a replay truncates it, so an FK would either cascade away a
person's dismissal or block the truncation. The column is a breadcrumb, and a
dangling one is the honest cost of pointing at a discardable table.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0024"
down_revision: str | Sequence[str] | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_REASONS = (
    "cleared",
    "no_context",
    "nothing_new",
    "below_threshold",
    "dismissed",
    "already_surfaced",
)


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "surfacing_log",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        # What the decision was about. The corpus key of a file, or a meeting's
        # title. Free text for the same reason `context_cache.focus` is: it is
        # whatever the trigger said, and normalising it here would mean this
        # table and the cache disagreeing about what one focus is.
        sa.Column("focus", sa.Text(), nullable=False),
        sa.Column("context_hash", sa.Text(), nullable=False),
        sa.Column(
            "item_keys",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        # The best item that was not the file already open, kept so the log and
        # the web view can say *what* was surfaced without re-assembling a
        # context whose corpus fingerprint has since moved on.
        sa.Column("top_key", sa.Text(), nullable=True),
        sa.Column("top_title", sa.Text(), nullable=True),
        # What it scored and what it had to beat. Both stored, because the first
        # question anybody asks about a refusal is how close it came — and the
        # threshold is adaptive, so recomputing it later would give the answer
        # for today's feedback rather than the one the decision was made under.
        #
        # DOUBLE PRECISION rather than the REAL this schema uses for
        # confidences, and the difference is not fussiness. Those are opinions
        # on a 0-1 scale where the fourth decimal place is noise. These two are
        # the terms of a comparison somebody may want to re-run, and a stored
        # score that does not round-trip exactly would let the log disagree with
        # the decision it exists to record.
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        # Which event triggered this, or null when a person ran the gate by hand.
        sa.Column("trigger_kind", sa.Text(), nullable=True),
        sa.Column("trigger_id", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column(
            "decided_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("surfaced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dismissed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acted_on_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_surfacing_log"),
        sa.CheckConstraint("length(btrim(focus)) > 0", name="ck_surfacing_focus"),
        sa.CheckConstraint(
            "reason IN ('" + "', '".join(_REASONS) + "')", name="ck_surfacing_reason"
        ),
        # The verdict and the outcome cannot disagree.
        sa.CheckConstraint(
            "(reason = 'cleared') = (surfaced_at IS NOT NULL)",
            name="ck_surfacing_reason_matches_outcome",
        ),
        # Feedback belongs only to something that was actually shown.
        sa.CheckConstraint(
            "surfaced_at IS NOT NULL "
            "OR (dismissed_at IS NULL AND acted_on_at IS NULL)",
            name="ck_surfacing_feedback_needs_surfacing",
        ),
        # And it is one or the other. Both would be counted twice in the
        # dismissal rate, which is the number the milestone is judged on.
        sa.CheckConstraint(
            "NOT (dismissed_at IS NOT NULL AND acted_on_at IS NOT NULL)",
            name="ck_surfacing_one_verdict",
        ),
    )
    # The suppression lookup: everything surfaced for this focus lately. Partial,
    # because refusals are the majority of the table by design and none of them
    # suppress anything.
    op.create_index(
        "ix_surfacing_focus_surfaced",
        "surfacing_log",
        ["focus", "surfaced_at"],
        unique=False,
        postgresql_where=sa.text("surfaced_at IS NOT NULL"),
    )
    # For `surfacing log`, which reads the tail of every decision including the
    # refusals, and for the stats, which group by focus over the whole table.
    op.create_index("ix_surfacing_decided", "surfacing_log", ["decided_at"])
    op.create_index("ix_surfacing_focus", "surfacing_log", ["focus"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_surfacing_focus", table_name="surfacing_log")
    op.drop_index("ix_surfacing_decided", table_name="surfacing_log")
    op.drop_index(
        "ix_surfacing_focus_surfaced",
        table_name="surfacing_log",
        postgresql_where=sa.text("surfaced_at IS NOT NULL"),
    )
    op.drop_table("surfacing_log")
