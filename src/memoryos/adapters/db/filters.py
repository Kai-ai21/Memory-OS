"""What a chunk must satisfy to be returned, whichever index found it.

Shared by the vector and keyword stores rather than written twice. The two
retrievers disagree about everything except this: they must agree exactly about
which rows are *eligible*, or the same corpus answers the same filtered question
two different ways — and M2.2 will fuse their outputs, at which point a
disagreement here would show up as a ranking artefact rather than as the bug it
is.

`is_current` is unconditional and is the one that matters. A superseded version's
chunks describe text the item no longer says; surfacing them is a correctness
failure, not a ranking one.
"""

from sqlalchemy import ColumnElement

from memoryos.adapters.db import models
from memoryos.application.ports import SearchFilters


def memory_predicates(filters: SearchFilters) -> list[ColumnElement[bool]]:
    """Eligibility clauses over the joined `memories` row.

    Assumes the statement already joins `memories` to `memory_chunks`; every
    caller does, because none of these can be evaluated from a chunk alone.
    """
    clauses: list[ColumnElement[bool]] = [models.Memory.is_current.is_(True)]

    if not filters.include_deleted:
        clauses.append(models.Memory.deleted_at.is_(None))
    if filters.source_ids:
        clauses.append(models.Memory.source_id.in_(list(filters.source_ids)))
    if filters.kinds:
        clauses.append(models.Memory.kind.in_([kind.value for kind in filters.kinds]))
    if filters.occurred_after is not None:
        clauses.append(models.Memory.occurred_at >= filters.occurred_after)
    if filters.occurred_before is not None:
        clauses.append(models.Memory.occurred_at <= filters.occurred_before)

    return clauses
