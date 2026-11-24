"""Reading encrypted content back as text, for every query, without being asked.

M11.2 encrypted `memories.content` and `memory_chunks.content` and then decrypted
them nowhere. Its module docstring explained why decryption could not be
transparent — a `TypeDecorator` runs inside result processing with no session,
and the data key lives in another table — and concluded that it would happen
"at named boundaries". None were named. Search excerpts, the reranker's input
and the passages sent to the model were all `enc:v1:…`, and nothing failed: the
cross-encoder scored base64 and the model politely declined to answer from it.

**This is the boundary, and it is the session.** An ORM `do_orm_execute` hook
runs inside the greenlet that drives every async query in this codebase, so it
*can* run a second query: one batched lookup of the data keys for whatever the
first query returned. That is the thing the type decorator could not do, and it
is what makes the transparent version possible after all — the same move M11.1
made for scoping, for the same reason. Fifty-nine modules read content, and a
boundary each of them has to remember is a boundary one of them forgets.

What it covers, exactly:

* **Whole entities** (`select(models.Memory)`, `session.get(...)`, lazy and
  refresh loads). The decrypted text is set with `set_committed_value`, so the
  instance is not dirty and a later flush cannot write plaintext back.
* **The content columns selected on their own** (`select(Memory.id,
  Memory.content)`). Cheapest when the row says whose content it is: `Memory.id` beside
  `Memory.content`, and `MemoryChunk.memory_id` or `MemoryChunk.id` beside
  `MemoryChunk.content`. A query that selects content and nothing that
  identifies it still works — the ciphertext, which carries a random nonce, is
  looked up instead — but pays for a table scan to do it.

What it cannot cover, and does not pretend to: **SQL that computes on content**
— `length(content)`, `content ~* …`, `ORDER BY length(content)`. Postgres never
sees the plaintext, so those predicates are evaluated over ciphertext and have
to move into Python wherever they appear.

**Writes are explicit and guarded.** Content enters the database in one place
(`NormalizeMemory`), which encrypts before the row is inserted so plaintext
never reaches a heap tuple or the WAL. A `before_flush` check refuses any
plaintext write to either column from a session that holds a cipher, so a
second write path added later fails its first test instead of quietly storing
text.

A session without a cipher passes plaintext through and refuses ciphertext.
Every session the application opens has one (`Database.from_url` and the replay
shadow are both given it by the container); the refusal is for a script that
opened its own engine and would otherwise print base64 as if it were content.
"""

from collections.abc import Iterable, Sequence
from typing import Any, cast
from uuid import UUID

import structlog
from sqlalchemy import Row, Select, event, select
from sqlalchemy.orm import ORMExecuteState, Session, UOWTransaction
from sqlalchemy.orm.attributes import get_history, set_committed_value
from sqlalchemy.sql.elements import Label

from memoryos.adapters.db import models
from memoryos.crypto import Cipher, DecryptionFailed, is_encrypted

logger = structlog.get_logger(__name__)

#: Where a session carries its cipher. Set through the session factory's `info`.
CIPHER = "memos.cipher"
#: Unwrapped data keys, per session, so a page of results costs one key query.
_KEYS = "memos.data_keys"
#: Session info flag: let plaintext content be flushed. For test fixtures that
#: seed rows as a pre-encryption corpus would hold them; never set in the app.
ALLOW_PLAINTEXT_WRITES = "memos.allow_plaintext_writes"
#: Execution option: return stored values untouched. For the code whose job is
#: the stored form itself — encrypting, re-keying, shredding.
KEEP_CIPHERTEXT = "memos_keep_ciphertext"

#: What a permanently deleted memory's text reads as, in the window between the
#: key being destroyed and the rows being purged. Not an error: the deletion was
#: the point, and a search page should not fail because one hit was shredded.
SHREDDED = ""


class CiphertextWithoutCipher(RuntimeError):
    """Encrypted content was read by a session that cannot decrypt it."""


class UnidentifiedContent(RuntimeError):
    """Encrypted content whose owner cannot be determined."""


class PlaintextContentWrite(RuntimeError):
    """A flush tried to store content unencrypted from a session that can encrypt."""


def session_info(
    cipher: Cipher | None, *, allow_plaintext_writes: bool = False
) -> dict[str, Any]:
    """The `info` a session factory needs for its sessions to decrypt."""
    info: dict[str, Any] = {} if cipher is None else {CIPHER: cipher}
    if allow_plaintext_writes:
        info[ALLOW_PLAINTEXT_WRITES] = True
    return info


def cipher_of(session: Session | Any) -> Cipher | None:
    """The cipher a session (sync or async) carries, if any."""
    cipher = session.info.get(CIPHER)
    return cipher if isinstance(cipher, Cipher) else None


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def _column(expr: Any) -> Any:
    """The table column under an attribute, an annotation or a label."""
    if hasattr(expr, "__clause_element__"):
        expr = expr.__clause_element__()
    while isinstance(expr, Label):
        expr = expr.element
    return expr


def _is_column(expr: Any, model: type, name: str) -> bool:
    # By table and name rather than by identity: an attribute, its annotated
    # column and a `.label()` of it are three different objects for one column,
    # and ORM attributes overload `==`, so they cannot be compared or hashed
    # safely as dict keys.
    column = _column(expr)
    return (
        getattr(column, "table", None) is model.__table__  # type: ignore[attr-defined]
        and getattr(column, "name", None) == name
    )


def _find(descriptions: Sequence[dict[str, Any]], model: type, name: str) -> int | None:
    for index, description in enumerate(descriptions):
        if _is_column(description.get("expr"), model, name):
            return index
    return None


@event.listens_for(Session, "do_orm_execute")
def _decrypt_results(state: ORMExecuteState) -> Any:
    if not state.is_select or state.execution_options.get(KEEP_CIPHERTEXT):
        return None
    if state.is_column_load:
        # A refresh or an expired attribute coming back for an instance already
        # in the session. The statement names only the columns being reloaded,
        # so there may be no id beside the content; the `refresh` event below
        # knows the instance and handles it instead.
        return None
    statement = state.statement
    if not isinstance(statement, Select):
        return None

    plan = _plan(statement.column_descriptions)
    if plan is None:
        return None

    frozen = state.invoke_statement().freeze()
    rows = frozen.data
    if not rows:
        return frozen()

    session = state.session
    changed = _decrypt_rows(session, plan, rows)
    if changed is None:
        return frozen()
    # Raw tuples, which is what a FrozenResult holds and rebuilds rows from.
    return frozen.with_new_rows(cast("Sequence[Row[Any]]", changed))()


class _Plan:
    """Which positions in a row hold content, and where each one's owner is."""

    __slots__ = ("chunk_content", "entities", "memory_content")

    def __init__(self) -> None:
        # (position, model) for whole Memory / MemoryChunk entities.
        self.entities: list[tuple[int, type]] = []
        # (content position, Memory.id position or None)
        self.memory_content: list[tuple[int, int | None]] = []
        # (content position, memory_id position or None, chunk id position or None)
        self.chunk_content: list[tuple[int, int | None, int | None]] = []


def _plan(descriptions: Sequence[dict[str, Any]]) -> _Plan | None:
    plan = _Plan()
    for index, description in enumerate(descriptions):
        expr = description.get("expr")
        if expr is models.Memory or expr is models.MemoryChunk:
            plan.entities.append((index, expr))
        elif _is_column(expr, models.Memory, "content"):
            plan.memory_content.append((index, _find(descriptions, models.Memory, "id")))
        elif _is_column(expr, models.MemoryChunk, "content"):
            plan.chunk_content.append(
                (
                    index,
                    _find(descriptions, models.MemoryChunk, "memory_id"),
                    _find(descriptions, models.MemoryChunk, "id"),
                )
            )

    if not (plan.entities or plan.memory_content or plan.chunk_content):
        return None
    return plan


class _Owners:
    """Whose content each encrypted value is, resolved in as few queries as possible.

    Cheapest first: an owner id selected in the same row; then a chunk id, one
    primary-key lookup for the page; then, for a query that selected content and
    nothing that identifies it, **the ciphertext itself**. Every envelope carries
    a random 96-bit nonce, so two rows never share one, and the stored value is
    as good an identifier as the id the query left out. That last lookup scans
    the table, which is the price of not selecting an id, and is why the code in
    `src/` selects one.
    """

    def __init__(self) -> None:
        self.ids: set[UUID] = set()
        self.chunk_ids: set[UUID] = set()
        self.memory_values: set[str] = set()
        self.chunk_values: set[str] = set()
        self.of_chunk: dict[UUID, UUID] = {}
        self.of_value: dict[str, UUID] = {}

    def resolve(self, session: Session) -> None:
        with session.no_autoflush:
            if self.chunk_ids:
                self.of_chunk = {
                    row[0]: row[1]
                    for row in session.execute(
                        select(models.MemoryChunk.id, models.MemoryChunk.memory_id).where(
                            models.MemoryChunk.id.in_(self.chunk_ids)
                        )
                    )
                }
            for column, owner, values in (
                (models.Memory.content, models.Memory.id, self.memory_values),
                (models.MemoryChunk.content, models.MemoryChunk.memory_id, self.chunk_values),
            ):
                if values:
                    self.of_value.update(
                        {
                            row[0]: row[1]
                            for row in session.execute(
                                select(column, owner)
                                .where(column.in_(values))
                                .execution_options(**{KEEP_CIPHERTEXT: True})
                            )
                        }
                    )
        self.ids.update(self.of_chunk.values())
        self.ids.update(self.of_value.values())

    def by_value(self, stored: str) -> UUID:
        owner = self.of_value.get(stored)
        if owner is None:
            raise UnidentifiedContent(
                "Encrypted content was read without an id, and no row holds that "
                "ciphertext any more, so there is no way to tell whose it is."
            )
        return owner


def _decrypt_rows(
    session: Session, plan: _Plan, rows: Sequence[Any]
) -> list[tuple[Any, ...]] | None:
    """Decrypt in place (entities) or into new rows (columns). None if no column changed."""
    # A single-entity result freezes to bare instances rather than rows.
    rows = [row if isinstance(row, (Row, tuple)) else (row,) for row in rows]

    # First pass: whose keys are needed, and which owners are not in the row.
    owners = _Owners()
    for row in rows:
        for index, _model in plan.entities:
            instance = row[index]
            if instance is not None and is_encrypted(instance.__dict__.get("content")):
                owners.ids.add(_owner_of(instance))
        for index, owner in plan.memory_content:
            if is_encrypted(row[index]):
                if owner is not None:
                    owners.ids.add(row[owner])
                else:
                    owners.memory_values.add(row[index])
        for index, memory_id, chunk_id in plan.chunk_content:
            if is_encrypted(row[index]):
                if memory_id is not None:
                    owners.ids.add(row[memory_id])
                elif chunk_id is not None:
                    owners.chunk_ids.add(row[chunk_id])
                else:
                    owners.chunk_values.add(row[index])

    if not (owners.ids or owners.chunk_ids or owners.memory_values or owners.chunk_values):
        return None

    cipher = cipher_of(session)
    if cipher is None:
        raise CiphertextWithoutCipher(
            "Encrypted content was read by a session with no cipher. Sessions "
            "from Container.build carry one; a hand-built engine needs "
            "`info=session_info(cipher)` on its session factory."
        )

    owners.resolve(session)
    keys = _data_keys(session, cipher, owners.ids)

    def plain(memory_id: UUID, stored: str) -> str:
        key = keys.get(memory_id)
        if key is None:
            logger.warning("encryption.read_shredded", memory_id=str(memory_id))
            return SHREDDED
        return cipher.decrypt(stored, key, memory_id)

    for row in rows:
        for index, _model in plan.entities:
            instance = row[index]
            if instance is None:
                continue
            stored = instance.__dict__.get("content")
            if is_encrypted(stored):
                set_committed_value(instance, "content", plain(_owner_of(instance), stored))

    if not (plan.memory_content or plan.chunk_content):
        return None

    rebuilt: list[tuple[Any, ...]] = []
    for row in rows:
        values = list(row)
        for index, owner in plan.memory_content:
            stored = values[index]
            if is_encrypted(stored):
                whose = values[owner] if owner is not None else owners.by_value(stored)
                values[index] = plain(whose, stored)
        for index, memory_id, chunk_id in plan.chunk_content:
            stored = values[index]
            if is_encrypted(stored):
                if memory_id is not None:
                    whose = values[memory_id]
                elif chunk_id is not None:
                    whose = owners.of_chunk[values[chunk_id]]
                else:
                    whose = owners.by_value(stored)
                values[index] = plain(whose, stored)
        rebuilt.append(tuple(values))
    return rebuilt


def _owner_of(instance: Any) -> UUID:
    if isinstance(instance, models.MemoryChunk):
        return instance.memory_id
    return instance.id  # type: ignore[no-any-return]


def _data_keys(session: Session, cipher: Cipher, memory_ids: Iterable[UUID]) -> dict[UUID, bytes]:
    """Unwrapped data keys, cached on the session. A destroyed key is absent."""
    cache: dict[UUID, bytes | None] = session.info.setdefault(_KEYS, {})
    missing = [memory_id for memory_id in memory_ids if memory_id not in cache]
    if missing:
        with session.no_autoflush:
            result = session.execute(
                select(models.MemoryKey.memory_id, models.MemoryKey.wrapped_key).where(
                    models.MemoryKey.memory_id.in_(missing)
                )
            )
            found = {row[0]: row[1] for row in result}
        for memory_id in missing:
            wrapped = found.get(memory_id)
            if memory_id not in found:
                raise DecryptionFailed(
                    f"Memory {memory_id} holds ciphertext but has no key row."
                )
            cache[memory_id] = None if wrapped is None else cipher.unwrap(wrapped, memory_id)
    return {
        memory_id: key
        for memory_id in memory_ids
        if (key := cache.get(memory_id)) is not None
    }


def forget_key(session: Session | Any, memory_id: UUID) -> None:
    """Drop a cached data key, after it has been destroyed or rotated."""
    cache = session.info.get(_KEYS)
    if cache is not None:
        cache.pop(memory_id, None)


def _on_refresh(target: Any, context: Any, attrs: Iterable[str] | None) -> None:
    """Decrypt one instance whose columns were just reloaded from the database."""
    if attrs is not None and "content" not in attrs:
        return
    options = getattr(context, "execution_options", None) or {}
    if options.get(KEEP_CIPHERTEXT):
        return
    stored = target.__dict__.get("content")
    if not is_encrypted(stored):
        return
    session = getattr(context, "session", None)
    if session is None:
        return
    cipher = cipher_of(session)
    if cipher is None:
        raise CiphertextWithoutCipher(
            "Encrypted content was refreshed by a session with no cipher."
        )
    owner = _owner_of(target)
    key = _data_keys(session, cipher, [owner]).get(owner)
    set_committed_value(
        target,
        "content",
        SHREDDED if key is None else cipher.decrypt(stored, key, owner),
    )


event.listen(models.Memory, "refresh", _on_refresh)
event.listen(models.MemoryChunk, "refresh", _on_refresh)


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


@event.listens_for(Session, "before_flush")
def _refuse_plaintext(session: Session, flush_context: UOWTransaction, instances: Any) -> None:
    if cipher_of(session) is None or session.info.get(ALLOW_PLAINTEXT_WRITES):
        return
    for instance in (*session.new, *session.dirty):
        if not isinstance(instance, (models.Memory, models.MemoryChunk)):
            continue
        added = get_history(instance, "content").added
        for value in added:
            if isinstance(value, str) and value and not is_encrypted(value):
                raise PlaintextContentWrite(
                    f"{type(instance).__name__} {instance.id} was about to be "
                    "stored unencrypted. Content is encrypted where it is written "
                    "(see NormalizeMemory); a new write path has to do the same."
                )
