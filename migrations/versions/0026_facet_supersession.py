"""M8.2: when a facet stopped being true, and why.

0025 could record that a facet had been *replaced* — `superseded_by` points at
the row that took over. It could not record the other half of the same event: a
facet whose evidence simply went away.

**That is the more common case, and it had nowhere to go.** A group of
assumptions is regrouped, a pattern is dismissed, three decisions become two —
and the deriver that proposed the facet stops proposing it. There is no
replacement statement, so `superseded_by` stays null, so the row stays live
forever asserting something nothing supports any more. The only alternatives
0025 left were to delete it or to keep it, and deleting a claim the system used
to make is the one thing M1.1 spent a phase arguing against.

So supersession becomes an event in its own right, with two columns:

* `superseded_at` — **when it stopped**, which nothing in 0025 recorded. The
  replacement's `created_at` was the closest available stand-in and it does not
  exist for a facet with no replacement. Every question M8.2 asks needs this
  column: `model diff --since` needs to know what changed inside a window, and
  `model stability` needs a lifetime, which is this minus `created_at`.
* `superseded_reason` — **why**, in the words the command prints. "Superseded"
  with no reason is the same non-answer "insufficient evidence" was in M8.0: it
  tells a reader that something happened and nothing about what would change it.
  Paired with `superseded_at` by a constraint, exactly as `dismissed_reason` is
  paired with `dismissed_at`, and for the reason that one gives — the reason is
  the part that survives being forgotten.

`superseded_by` stays, and stays nullable, and its nullability now means
something specific: **superseded with a replacement, or superseded by the
absence of its evidence.** The first is a revision, the second a withdrawal, and
`model timeline` distinguishes them by exactly this.

The two partial indexes move from `superseded_by IS NULL` to
`superseded_at IS NULL`, which is what "live" now means. Without that move a
withdrawn facet would keep appearing in `model show` — it has no replacement, so
`superseded_by` is null, so the old predicate calls it live. That is the bug this
migration exists to make unrepresentable, and it lives in an index rather than in
a query because the unique index is also what keeps re-derivation idempotent.

Revision ID: 0026
Revises: 0025
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# What a row superseded before this migration gets for a reason. It is not a
# good reason and it does not pretend to be one: the information was never
# captured, and writing a plausible reason here — "statement revised" — would be
# inventing a record of an event nobody observed, which is the failure mode this
# whole column exists to prevent.
_BACKFILL_REASON = "superseded before 0026; the reason was not recorded"


def upgrade() -> None:
    op.add_column(
        "user_model_facets",
        sa.Column(
            "superseded_at", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
    )
    op.add_column(
        "user_model_facets", sa.Column("superseded_reason", sa.Text(), nullable=True)
    )

    # Existing superseded rows: dated from their replacement, which is the best
    # available answer and within a transaction of the true one, since 0025's
    # `derive` retired the old row and inserted the new one in the same
    # statement block.
    op.execute(
        sa.text(
            """
            UPDATE user_model_facets AS old
               SET superseded_at = new.created_at,
                   superseded_reason = :reason
              FROM user_model_facets AS new
             WHERE old.superseded_by = new.id
               AND old.superseded_at IS NULL
            """
        ).bindparams(reason=_BACKFILL_REASON)
    )

    # A replacement implies supersession. The converse does not hold, and that
    # asymmetry is the whole point: a withdrawn facet is superseded with nothing
    # pointing forward.
    op.create_check_constraint(
        "ck_user_model_facets_supersession_dated",
        "user_model_facets",
        "superseded_by IS NULL OR superseded_at IS NOT NULL",
    )
    op.create_check_constraint(
        "ck_user_model_facets_supersession_paired",
        "user_model_facets",
        "(superseded_at IS NULL) = (superseded_reason IS NULL)",
    )

    # "Live" now means not superseded and not dismissed, where superseded is read
    # from the timestamp rather than from the pointer.
    op.drop_index("ix_user_model_facets_live", table_name="user_model_facets")
    op.create_index(
        "ix_user_model_facets_live",
        "user_model_facets",
        ["dimension"],
        postgresql_where=sa.text("superseded_at IS NULL AND dismissed_at IS NULL"),
    )
    op.drop_index("uq_user_model_facets_live_subject", table_name="user_model_facets")
    op.create_index(
        "uq_user_model_facets_live_subject",
        "user_model_facets",
        ["detector", "subject_key"],
        unique=True,
        postgresql_where=sa.text(
            "origin = 'derived' AND superseded_at IS NULL AND dismissed_at IS NULL"
        ),
    )
    # The timeline and the diff both scan by *when something happened* rather
    # than by dimension, and both of them read this column with a range
    # predicate.
    op.create_index(
        "ix_user_model_facets_superseded_at",
        "user_model_facets",
        ["superseded_at"],
        postgresql_where=sa.text("superseded_at IS NOT NULL"),
    )


def downgrade() -> None:
    # Withdrawn facets — superseded with no replacement — become live again on
    # the way down, because the old schema has no way to express them. That is a
    # real loss of information and the reason it is stated here: unwinding this
    # migration resurrects claims whose evidence had gone.
    op.drop_index(
        "ix_user_model_facets_superseded_at", table_name="user_model_facets"
    )
    op.drop_index("uq_user_model_facets_live_subject", table_name="user_model_facets")
    op.create_index(
        "uq_user_model_facets_live_subject",
        "user_model_facets",
        ["detector", "subject_key"],
        unique=True,
        postgresql_where=sa.text(
            "origin = 'derived' AND superseded_by IS NULL AND dismissed_at IS NULL"
        ),
    )
    op.drop_index("ix_user_model_facets_live", table_name="user_model_facets")
    op.create_index(
        "ix_user_model_facets_live",
        "user_model_facets",
        ["dimension"],
        postgresql_where=sa.text("superseded_by IS NULL AND dismissed_at IS NULL"),
    )
    op.drop_constraint(
        "ck_user_model_facets_supersession_paired", "user_model_facets"
    )
    op.drop_constraint(
        "ck_user_model_facets_supersession_dated", "user_model_facets"
    )
    op.drop_column("user_model_facets", "superseded_reason")
    op.drop_column("user_model_facets", "superseded_at")
