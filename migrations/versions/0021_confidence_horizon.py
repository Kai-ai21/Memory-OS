"""whether a decision's confidence was recorded before the answer was known

Revision ID: 0021
Revises: 0020
Create Date: 2026-08-13

**The column Phase 5's retrospective said should have been in M5.0.**

`decisions.confidence` is never updated, and M5.0's docstring says its whole
value is that it was recorded before the outcome was known. Immutability
guarantees the number did not *move*. It guarantees nothing about when it was
first written — a confidence reconstructed a week later is also never refreshed —
and a calibration table built on reconstructions measures hindsight while looking
exactly like one built on foresight. That is what M5.3 measured, and the only
thing that said so was a paragraph of prose in a README.

`confidence_horizon` makes it checkable. Written once at capture, never updated,
paired with `confidence` by a CHECK so a null number and a claimed horizon cannot
coexist.

**The backfill classifies every existing row as `hindsight`, and it does so from
a column that has been there since M1.1.** The rule in
`domain/patterns.classify_confidence` is applied retroactively to data already
stored, and the test that decides it is `decided_at_source`: `parsed` means the
date was read out of a document and `filesystem` means it came from an mtime. If
nobody wrote the date down at the time, nobody wrote the confidence down at the
time either — both came from the same act of reconstruction. All twelve
confidence-bearing decisions here are `parsed`, read out of the README by
`scripts/seed_decisions.py`, whose own docstring says *"the numbers here are what
the person who made the call believes they believed"*. This is the first time the
database agrees with it.

The window test in the application is deliberately *not* used here, and the
reason is worth recording: applied alone it classifies two of the twelve as
foresight, because they were decided fourteen hours before their row was written.
Both are reconstructions. A backfill that let those two through on a clock
technicality would leave the calibration table lying about exactly the two rows
nobody would think to check.

The four decisions with no confidence — all accepted extraction drafts, which the
M5.0 prompt is told to leave empty rather than guess — become `unknown`. That is
the pairing rule, not a judgement about them.

The consequence is deliberate and is the point: `patterns calibration` now has an
empty population and says why. A table nobody can act on is worse than no table,
because a reader acts on it anyway.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0021"
down_revision: str | Sequence[str] | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "decisions",
        sa.Column(
            "confidence_horizon",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'unknown'"),
        ),
    )

    # One statement over one table. No clock arithmetic and no window: the
    # provenance of the *date* decides, for the reason in the docstring above.
    #
    # A row that is `declared` and carries a confidence would become `foresight`
    # here, and there are none — but the CASE says so explicitly rather than
    # assuming, because a backfill written against what the table happens to
    # contain is a backfill that is wrong the first time it is re-run.
    op.execute(
        sa.text(
            """
            UPDATE decisions
               SET confidence_horizon = CASE
                     WHEN confidence IS NULL THEN 'unknown'
                     WHEN decided_at_source <> 'declared' THEN 'hindsight'
                     ELSE 'foresight'
                   END
            """
        )
    )

    op.create_check_constraint(
        "ck_decisions_horizon",
        "decisions",
        "confidence_horizon IN ('foresight', 'hindsight', 'unknown')",
    )
    op.create_check_constraint(
        "ck_decisions_horizon_pairing",
        "decisions",
        "(confidence IS NULL) = (confidence_horizon = 'unknown')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_decisions_horizon_pairing", "decisions", type_="check")
    op.drop_constraint("ck_decisions_horizon", "decisions", type_="check")
    op.drop_column("decisions", "confidence_horizon")
