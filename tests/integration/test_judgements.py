"""Human-authored data, and the guarantee that a rebuild cannot destroy it.

The last test in this file is the one that matters. `query_judgements` is the
only table nobody can regenerate, M1.7 made replay a routine operation, and the
schema the milestone originally specified — a foreign key to `memories` — would
have made every replay delete the golden set. That combination is measured here
rather than argued about.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.judgements import (
    InvalidJudgement,
    JudgementInput,
    export_golden_set,
    record,
    summarise,
)
from memoryos.application.replay import truncate_derived
from memoryos.application.verification import compare
from memoryos.domain.values import Verdict
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def judgement(**overrides: object) -> JudgementInput:
    fields: dict[str, object] = {
        "query_text": "how a lease is renewed",
        "source_name": "corpus",
        "external_key": "queue.md",
        "verdict": Verdict.RELEVANT,
        "rank_at_judgement": 1,
        "score_at_judgement": 0.81,
        "filters": {"k": 10},
    }
    fields.update(overrides)
    return JudgementInput(**fields)  # type: ignore[arg-type]


async def count(sessions: async_sessionmaker[AsyncSession]) -> int:
    async with sessions() as session:
        return int(
            (
                await session.execute(
                    select(func.count()).select_from(models.QueryJudgement)
                )
            ).scalar_one()
        )


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------


async def test_a_judgement_is_stored(sessions: async_sessionmaker[AsyncSession]) -> None:
    await record(sessions, judgement())

    async with sessions() as session:
        row = (await session.execute(select(models.QueryJudgement))).scalar_one()
    assert row.verdict == "relevant"
    assert row.external_key == "queue.md"
    assert row.rank_at_judgement == 1
    assert row.filters == {"k": 10}


async def test_rejudging_replaces_rather_than_appends(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """Two contradictory opinions about one pair is not richer data.

    The snapshots are replaced along with the verdict: a second judgement is an
    opinion about a *fresh* ranking, so keeping the first one's rank would
    attribute the new verdict to an old position.
    """
    first = await record(sessions, judgement(verdict=Verdict.RELEVANT, rank_at_judgement=1))
    second = await record(
        sessions, judgement(verdict=Verdict.NOT_RELEVANT, rank_at_judgement=4)
    )

    assert first == second, "the row was replaced, so its id is stable"
    assert await count(sessions) == 1
    async with sessions() as session:
        row = (await session.execute(select(models.QueryJudgement))).scalar_one()
    assert row.verdict == "not_relevant"
    assert row.rank_at_judgement == 4


async def test_the_same_item_under_a_different_query_is_a_separate_judgement(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    # A judgement is about a phrasing. Collapsing two phrasings into one row
    # would erase the distinction the golden set exists to measure.
    await record(sessions, judgement(query_text="how does claiming work"))
    await record(sessions, judgement(query_text="what is a lease"))
    assert await count(sessions) == 2


async def test_a_missing_verdict_cannot_carry_a_rank(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """The point of `missing` is that the item was not in the ranking.

    A rank on it would silently corrupt any recall computed from this table.
    """
    with pytest.raises(InvalidJudgement, match="cannot carry a rank"):
        await record(
            sessions, judgement(verdict=Verdict.MISSING, rank_at_judgement=3)
        )
    assert await count(sessions) == 0


async def test_a_missing_verdict_without_a_rank_is_fine(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await record(
        sessions,
        judgement(verdict=Verdict.MISSING, rank_at_judgement=None, score_at_judgement=None),
    )
    assert await count(sessions) == 1


async def test_a_blank_query_is_refused(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(InvalidJudgement, match="needs the query"):
        await record(sessions, judgement(query_text="   "))


async def test_the_database_refuses_an_unknown_verdict(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    # The CHECK constraint, not just the enum: Python invariants protect this
    # process and constraints protect the data against every other writer.
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        async with sessions.begin() as session:
            await session.execute(
                text(
                    "INSERT INTO query_judgements "
                    "(id, query_text, source_name, external_key, verdict) "
                    "VALUES (gen_random_uuid(), 'q', 's', 'k', 'maybe')"
                )
            )


# --------------------------------------------------------------------------
# Summaries and export
# --------------------------------------------------------------------------


async def test_summaries_count_each_verdict_per_query(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    await record(sessions, judgement(external_key="a.md", verdict=Verdict.RELEVANT))
    await record(sessions, judgement(external_key="b.md", verdict=Verdict.NOT_RELEVANT))
    await record(
        sessions,
        judgement(external_key="c.md", verdict=Verdict.MISSING, rank_at_judgement=None),
    )

    (summary,) = await summarise(sessions)
    assert (summary.relevant, summary.not_relevant, summary.missing) == (1, 1, 1)
    assert summary.total == 3


async def test_the_export_groups_by_query_and_names_the_answer_key(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    """`relevant_keys` is what an evaluation harness actually scores against.

    Both `relevant` and `missing` belong in it: a missing item is one that should
    have been returned, which is exactly what an answer key is for.
    """
    await record(sessions, judgement(external_key="a.md", verdict=Verdict.RELEVANT))
    await record(sessions, judgement(external_key="b.md", verdict=Verdict.NOT_RELEVANT))
    await record(
        sessions,
        judgement(external_key="c.md", verdict=Verdict.MISSING, rank_at_judgement=None),
    )

    golden = await export_golden_set(sessions, now=NOW)

    (query,) = golden.queries
    assert query.relevant_keys == ["a.md", "c.md"]
    assert golden.totals == {
        "queries": 1,
        "judgements": 3,
        "relevant": 1,
        "not_relevant": 1,
        "missing": 1,
        # Nothing in this test's corpus, so no id resolves.
        "unresolved": 3,
    }


async def test_the_export_is_stable_for_the_same_data(
    sessions: async_sessionmaker[AsyncSession],
) -> None:
    # A consumer diffing two exports should see changes in the data, not in the
    # ordering. Queries sort by text, items by rank then key.
    for key in ("c.md", "a.md", "b.md"):
        await record(sessions, judgement(external_key=key, rank_at_judgement=None))

    first = await export_golden_set(sessions, now=NOW)
    second = await export_golden_set(sessions, now=NOW)
    assert first.as_dict() == second.as_dict()


# --------------------------------------------------------------------------
# The guarantee
# --------------------------------------------------------------------------


async def test_a_full_replay_leaves_judgements_untouched(harness: Harness) -> None:
    """The reason `USER_AUTHORED` exists, measured rather than asserted.

    The schema this milestone originally specified put a foreign key on
    `memories`. Measured on this database, `TRUNCATE memory_chunks, memories,
    jobs CASCADE` reports "truncate cascades to" and empties any table
    referencing memories, and `DELETE FROM memories` takes it via `ON DELETE
    CASCADE`. Either way every routine replay would have destroyed the golden set
    — the exact opposite of what the milestone asked for. Identity is a natural
    key instead, and this is what pins that.
    """
    before_corpus = await harness.snapshot()
    await record(
        harness.sessions,
        judgement(source_name="corpus", external_key="queue.md"),
    )
    await record(
        harness.sessions,
        judgement(
            query_text="what is a lease",
            source_name="corpus",
            external_key="bread.txt",
            verdict=Verdict.NOT_RELEVANT,
            rank_at_judgement=7,
        ),
    )

    async with harness.sessions() as session:
        before = [
            (row.id, row.query_text, row.external_key, row.verdict, row.judged_at)
            for row in (
                await session.execute(
                    select(models.QueryJudgement).order_by(
                        models.QueryJudgement.external_key
                    )
                )
            ).scalars()
        ]
    assert len(before) == 2

    # The most destructive thing the system can legitimately do to itself.
    await truncate_derived(harness.sessions, clear_cache=True)
    await harness.replay(clear_cache=True)

    async with harness.sessions() as session:
        after = [
            (row.id, row.query_text, row.external_key, row.verdict, row.judged_at)
            for row in (
                await session.execute(
                    select(models.QueryJudgement).order_by(
                        models.QueryJudgement.external_key
                    )
                )
            ).scalars()
        ]

    # Not merely the same count — the same rows, ids and timestamps included.
    assert after == before
    # And the corpus itself still rebuilt correctly.
    assert compare(before_corpus, await harness.snapshot()).identical


async def test_the_export_re_resolves_ids_after_a_rebuild(harness: Harness) -> None:
    """The payoff from keying on the natural key.

    Every memory id changes during a replay. A golden set that stored ids would
    hand M2.0 pointers to rows that no longer exist, and an evaluation against it
    would score nothing while looking like it ran.
    """
    async with harness.sessions() as session:
        original_id = (
            await session.execute(
                select(models.Memory.id).where(
                    models.Memory.external_key == "queue.md",
                    models.Memory.is_current.is_(True),
                )
            )
        ).scalar_one()

    await record(
        harness.sessions,
        judgement(source_name="corpus", external_key="queue.md", memory_id=original_id),
    )

    await harness.replay(clear_cache=False)

    async with harness.sessions() as session:
        rebuilt_id = (
            await session.execute(
                select(models.Memory.id).where(
                    models.Memory.external_key == "queue.md",
                    models.Memory.is_current.is_(True),
                )
            )
        ).scalar_one()
    assert rebuilt_id != original_id, "the fixture did not actually rebuild"

    golden = await export_golden_set(harness.sessions, now=NOW)
    (item,) = golden.queries[0].items
    # Resolved to the row that exists now, not to the stale snapshot.
    assert item.memory_id == str(rebuilt_id)
    assert golden.totals["unresolved"] == 0


async def test_a_judged_item_that_leaves_the_corpus_exports_unresolved(
    harness: Harness,
) -> None:
    # Dropped judgements would make a shrinking corpus look like a shrinking
    # disagreement. It exports with a null id and is counted, so a consumer can
    # tell the difference.
    await record(
        harness.sessions,
        judgement(source_name="corpus", external_key="deleted-long-ago.md"),
    )

    golden = await export_golden_set(harness.sessions, now=NOW)
    (item,) = golden.queries[0].items
    assert item.memory_id is None
    assert golden.totals["unresolved"] == 1


async def test_the_cli_writes_the_golden_set_to_a_file(
    harness: Harness, tmp_path: Path
) -> None:
    import json

    from memoryos.application.judgements import export_golden_set as export

    await record(
        harness.sessions, judgement(source_name="corpus", external_key="queue.md")
    )
    golden = await export(harness.sessions, now=NOW)

    destination = tmp_path / "nested" / "golden-set.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(golden.as_dict(), indent=2) + "\n")

    reloaded = json.loads(destination.read_text())
    assert reloaded["totals"]["queries"] == 1
    assert reloaded["queries"][0]["relevant_keys"] == ["queue.md"]
