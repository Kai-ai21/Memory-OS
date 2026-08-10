"""Postgres full-text search over `memory_chunks.search_vector`.

The lexical retriever. Three choices are worth knowing about, and each has a
plausible alternative that is wrong here.

**`websearch_to_tsquery`, not `plainto_tsquery` or `to_tsquery`.** `to_tsquery`
takes an operator language and raises a syntax error on anything that is not
one — `SKIP LOCKED` with an unescaped colon, a query containing `&`, a stray
quote. Search input arrives from a URL bar and a text field, so a parser that
throws on ordinary text is a parser that turns a user's typo into a 500.
`plainto_tsquery` never throws but also never does anything but AND the words
together; `websearch_to_tsquery` is the one that both survives arbitrary input
and understands `"quoted phrases"` and `-exclusion`, which are the two operators
people actually type.

**`ts_rank_cd`, not `ts_rank`.** Cover density ranking accounts for how close the
matched terms sit to each other. For this corpus that is the difference between a
chunk containing `SKIP LOCKED` as a phrase and one that mentions skipping in a
docstring and locking forty lines later.

**The same eligibility clauses as the vector store**, imported rather than
retyped — see `db/filters.py`.

There is no over-fetch here, unlike `PgVectorStore`. That multiplier exists
because an approximate index picks its candidates before the filter is applied
and a restrictive filter can leave fewer than `k` standing. This query is exact:
the filter is part of it, and `LIMIT k` returns k rows if k rows qualify.
"""

from typing import Any

import structlog
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.filters import memory_predicates
from memoryos.application.ports import KeywordStore, ScoredChunk, SearchFilters

logger = structlog.get_logger(__name__)

# The text search configuration. English stemming and stop words, matching the
# generated column in migration 0010 — a query parsed under a different
# configuration than the one that built the vector matches nothing, silently.
TEXT_SEARCH_CONFIG = "english"


class PostgresKeywordStore(KeywordStore):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = session_factory

    async def search(
        self, query: str, *, k: int, filters: SearchFilters
    ) -> list[ScoredChunk]:
        if not query.strip() or k <= 0:
            # Saves a round trip on the empty case, and nothing more: a query of
            # nothing but stop words still reaches Postgres and still comes back
            # empty, because an empty tsquery matches no row. Both paths return
            # the same thing; only one of them is worth a database call.
            return []

        async with self._sessions() as session:
            rows = (await session.execute(self._query(query, k=k, filters=filters))).all()

        if not rows:
            # Distinguishable in the log from "found nothing relevant": a query
            # that reduced to no lexemes cannot match anything by construction,
            # and that is a different thing to explain to somebody than a corpus
            # that simply lacks the term.
            logger.info("keyword_store.empty", query_length=len(query), k=k)

        return [
            ScoredChunk(
                chunk_id=row[0],
                memory_id=row[1],
                ordinal=row[2],
                text=row[3],
                char_start=row[4],
                char_end=row[5],
                prefix_chars=row[6],
                metadata=dict(row[7] or {}),
                score=float(row[8]),
            )
            for row in rows
        ]

    def _query(self, query: str, *, k: int, filters: SearchFilters) -> Select[Any]:
        # Bound parameters, not interpolation — the query text is user input and
        # this is a place where that has to be said out loud.
        tsquery = func.websearch_to_tsquery(TEXT_SEARCH_CONFIG, query)
        # Written twice rather than lifted into a FROM-item. `websearch_to_tsquery`
        # is IMMUTABLE, so Postgres folds the two occurrences into one evaluation;
        # a lateral join would be the same plan with more SQL.
        score = func.ts_rank_cd(models.MemoryChunk.search_vector, tsquery)

        return (
            select(
                models.MemoryChunk.id,
                models.MemoryChunk.memory_id,
                models.MemoryChunk.ordinal,
                models.MemoryChunk.content,
                models.MemoryChunk.char_start,
                models.MemoryChunk.char_end,
                models.MemoryChunk.prefix_chars,
                models.MemoryChunk.meta,
                score.label("score"),
            )
            .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
            .where(
                # `@@` first, because it is the clause the GIN index answers.
                models.MemoryChunk.search_vector.op("@@")(tsquery),
                *memory_predicates(filters),
            )
            # `id` as the tie-break. `ts_rank_cd` produces exact ties readily —
            # two chunks with the same terms at the same density score
            # identically — and without a total order the same query returns the
            # same rows in a different sequence run to run, which would make the
            # evaluation harness report noise as change.
            .order_by(score.desc(), models.MemoryChunk.id)
            .limit(k)
        )
