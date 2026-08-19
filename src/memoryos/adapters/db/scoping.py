"""Telling Postgres who is asking, once, for every transaction.

**This file is the entire application-side cost of row-level security**, and
that is the argument for having chosen it. The policies added in migration 0032
compare `user_id` against `memos_current_user_id()`, which reads the
`app.current_user_id` GUC; something has to set that GUC. This sets it on every
transaction on every session, from a context variable, so no query anywhere else
in the codebase mentions scoping and none of them can forget to.

**Unset is fail-closed and deliberately so.** With no user in context the GUC is
set to the empty string, `memos_current_user_id()` returns NULL, and
`user_id = NULL` is NULL — so a `SELECT` matches nothing and an `INSERT` fails
the NOT NULL default. A connection that never says who it is gets an empty
system, not everybody's.

**A context variable rather than a parameter** because the alternative is
threading a user id through fifty-nine modules that have no other reason to know
about one — which is the same rewrite RLS was chosen to avoid. `contextvars`
are per-task, so concurrent requests on one event loop do not see each other's.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID

from sqlalchemy import event, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session, SessionTransaction

# `None` means "no user in context", which is a legitimate state — the login
# route reads `users` before it knows who is asking — and an unsafe one for
# anything scoped, which is why it resolves to an empty GUC rather than to a
# skipped policy.
CURRENT_USER_ID: ContextVar[UUID | None] = ContextVar("memos_current_user_id", default=None)


@contextmanager
def scoped_to(user_id: UUID | None) -> Iterator[None]:
    """Run a block as `user_id`.

    Resets on exit rather than leaving the value set, because a CLI process that
    handles two users in sequence would otherwise carry the first one into the
    second — and the symptom of that is data written to the wrong account, which
    no test that checks one user at a time will ever see.
    """
    token = CURRENT_USER_ID.set(user_id)
    try:
        yield
    finally:
        CURRENT_USER_ID.reset(token)


def current_user_id() -> UUID | None:
    return CURRENT_USER_ID.get()


@event.listens_for(Session, "after_begin")
def _apply_scope(
    session: Session, transaction: SessionTransaction, connection: Connection
) -> None:
    """Set the GUC as each transaction opens.

    **`after_begin` rather than a connection-level event**, and the difference
    matters with a pool: `set_config(..., true)` is `SET LOCAL`, scoped to the
    transaction, so it cannot leak to whoever borrows the connection next. A
    session-level `SET` would be exactly the bug this is meant to prevent, one
    connection at a time.

    Registered on `Session` globally rather than on a particular factory so that
    a session built anywhere — the API, the CLI, the worker, a test — is scoped
    without being told. The async sessions in this codebase are greenlet
    wrappers over these, so this covers them too.
    """
    user_id = CURRENT_USER_ID.get()
    connection.execute(
        text("SELECT set_config('app.current_user_id', :user_id, true)"),
        {"user_id": str(user_id) if user_id is not None else ""},
    )
