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

from collections.abc import Sequence

from sqlalchemy import ColumnElement, func, select

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
    if filters.tags:
        clauses.append(_carries_every_tag(filters.tags))

    return clauses


def _carries_every_tag(tags: Sequence[str]) -> ColumnElement[bool]:
    """The memory's item is tagged with all of these. M10.4.

    A correlated `EXISTS` with a `HAVING` count, rather than a join, and the reason
    is what this predicate is used for: it is added to *both* retrievers'
    statements, one of which is an HNSW vector scan. A join to `memory_tags` would
    multiply that statement's rows by the number of tags per memory before the
    `LIMIT` applied, which changes what "the fifty nearest chunks" means. An
    `EXISTS` filters and does not fan out.

    Matched on `(source_id, external_key)` because that is what a tag is attached
    to — the item, not the version — which is what makes a tag survive a
    correction. Two sources may hold the same external key, so both halves are
    required.
    """
    tag_rows = models.MemoryTag
    return (
        select(func.count(func.distinct(tag_rows.tag)))
        .where(
            tag_rows.source_id == models.Memory.source_id,
            tag_rows.external_key == models.Memory.external_key,
            tag_rows.tag.in_(list(tags)),
        )
        .correlate(models.Memory)
        .scalar_subquery()
        == len(set(tags))
    )
