"""M4.3's four claims, and the third is the one the milestone rests on.

A parser that fires when it should not is the failure mode here, and it is
invisible from the outside: results move, nothing errors, and the reason is a
phrase nobody thought was a date. So the tests are weighted towards refusal —
that a month name in ordinary prose parses to nothing, and that a query with no
temporal signal produces byte-identical retrieval to M3.5.

"Byte-identical" is asserted against the feature's own off switch rather than
against a recorded fixture. A fixture would drift with the corpus and would only
prove the results match what they matched when it was written; running the same
query twice through the same code, once with parsing disabled, proves the parse
itself changed nothing.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.embed import EmbedMemory
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.search import SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.domain.jobs import JobType
from memoryos.domain.temporal_intent import IntentKind, parse_temporal_intent
from tests.integration.conftest import add_source
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

# Pinned, so "in August" resolves to the same window on every run and in every
# time zone. The parser reads the clock for relative phrases, and a test that
# let it read the real one would pass in August and fail in September.
NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)

JULY = datetime(2026, 7, 4, 9, 0, tzinfo=UTC)
AUGUST = datetime(2026, 8, 9, 9, 0, tzinfo=UTC)

LEASE_TEXT = (
    "The worker claims a task from the queue and holds a lease on it while the "
    "handler runs to completion. Renewing that lease is how a long task keeps "
    "its hold on the work it started. "
)


@dataclass(slots=True)
class Corpus:
    sessions: async_sessionmaker[AsyncSession]
    search: SearchMemories
    plain: SearchMemories

    async def dates(self) -> dict[str, datetime | None]:
        async with self.sessions() as session:
            return {
                key: occurred
                for key, occurred in await session.execute(
                    select(models.Memory.external_key, models.Memory.occurred_at)
                )
            }


@pytest.fixture
async def corpus(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> Corpus:
    """Two documents about the same subject, dated a month apart.

    Same subject on purpose: a range filter that works only because the excluded
    document was irrelevant anyway proves nothing. Both of these answer a query
    about leases, so a filter is the only thing that can separate them.
    """
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "july-notes.md").write_text("# July\n\n" + LEASE_TEXT * 4 + "\n")
    (root / "august-notes.md").write_text("# August\n\n" + LEASE_TEXT * 4 + "\n")

    source = await add_source(sessions, "temporal", root)
    blobs = FilesystemBlobStore(tmp_path / "blobs")
    embedder = FakeEmbedder()
    store = PgVectorStore(sessions, embedder)
    cache = PostgresEmbeddingCache(sessions)

    sync = SyncSource(sessions, FilesystemConnector(blobs), blobs)
    normalize = NormalizeMemory(sessions, blobs, build_parsers(), StructuralChunker(embedder))
    embed = EmbedMemory(sessions, embedder, cache)

    await sync(source.id, full=True)
    for job_type, handler in ((JobType.NORMALIZE_MEMORY, normalize), (JobType.EMBED_MEMORY, embed)):
        async with sessions() as session:
            targets = [
                UUID(row[0]["memory_id"])
                for row in await session.execute(
                    select(models.Job.payload).where(models.Job.job_type == job_type.value)
                )
            ]
        for memory_id in targets:
            await handler(memory_id)

    # The connector reads mtimes, which are all "now". Overwritten here because
    # the point of the fixture is two documents in two different months.
    async with sessions.begin() as session:
        for key, moment in (("july-notes.md", JULY), ("august-notes.md", AUGUST)):
            await session.execute(
                update(models.Memory)
                .where(models.Memory.external_key == key)
                .values(occurred_at=moment)
            )

    def build(*, temporal: bool) -> SearchMemories:
        return SearchMemories(
            sessions,
            embedder,
            store,
            PostgresKeywordStore(sessions),
            now=lambda: NOW,
            temporal_intent=temporal,
        )

    return Corpus(sessions=sessions, search=build(temporal=True), plain=build(temporal=False))


def test_a_month_name_in_ordinary_prose_is_not_a_date() -> None:
    """The trap, and the whole reason the preposition rule exists.

    `may` is a modal verb before it is a month, and it is the commonest month
    name in English prose by a wide margin. `march` is a verb, `august` is an
    adjective, `first` and `last` are positions in a structure. Every one of
    these is a plausible query against this corpus, and every one of them must
    parse to nothing.
    """
    not_temporal = [
        "what may cause a chunk to be dropped",
        "the May release notes",
        "why does the parser march through the tree",
        "the first argument to the chunker",
        "the last chunk of a memory",
        "how do I mark a result relevant",
        "an august decision about hashing",
        "how does the worker claim a job",
    ]
    for query in not_temporal:
        assert parse_temporal_intent(query, now=NOW) is None, query

    # And the same words *with* a preposition in front are dates, so the rule is
    # doing work rather than simply never firing.
    for query in ("what changed in May", "what landed during march", "since august"):
        intent = parse_temporal_intent(query, now=NOW)
        assert intent is not None and intent.is_range, query


async def test_an_explicit_range_becomes_a_hard_filter(corpus: Corpus) -> None:
    """A question about a month does not return the other month's answer.

    Both documents are about leases and the fake embedder scores them almost
    identically, so relevance cannot separate them. Only a filter can, which is
    what makes this an assertion about filtering rather than about ranking.
    """
    unfiltered = await corpus.search("holding a lease on a task", k=10)
    assert {hit.external_key for hit in unfiltered.hits} == {
        "july-notes.md",
        "august-notes.md",
    }
    assert unfiltered.temporal_intent is None

    filtered = await corpus.search("holding a lease on a task in July", k=10)

    assert filtered.temporal_filter_applied is True
    assert filtered.temporal_intent is not None
    assert "2026-07-01" in filtered.temporal_intent
    # The August document is not ranked lower. It is gone.
    assert [hit.external_key for hit in filtered.hits] == ["july-notes.md"]

    # And the other direction, so this is a filter rather than a preference for
    # whichever document happens to sort first.
    august = await corpus.search("holding a lease on a task in August", k=10)
    assert [hit.external_key for hit in august.hits] == ["august-notes.md"]

    # A range the corpus cannot satisfy returns nothing, and still says why —
    # this is the case where the interpretation matters most, because "no
    # results" and "no corpus" are otherwise indistinguishable.
    empty = await corpus.search("holding a lease on a task in January", k=10)
    assert empty.hits == []
    assert empty.temporal_filter_applied is True
    assert "2026-01-01" in (empty.temporal_intent or "")


async def test_a_query_with_no_temporal_signal_retrieves_identically(
    corpus: Corpus,
) -> None:
    """Intent `None` reproduces M3.5 exactly.

    Compared against the same code with parsing switched off rather than against
    a recorded fixture: a fixture proves the results equal what they equalled
    when it was written, and this proves the parse changed nothing. Every field
    that could carry a difference is compared — the ordering, the fused scores,
    and the whole breakdown — because a weight applied by accident moves scores
    without necessarily moving ranks.
    """
    query = "how does a worker hold on to the work it started"
    assert parse_temporal_intent(query, now=NOW) is None

    with_parsing = await corpus.search(query, k=10)
    without_parsing = await corpus.plain(query, k=10)

    assert with_parsing.temporal_intent is None
    assert with_parsing.temporal_filter_applied is False

    assert [hit.external_key for hit in with_parsing.hits] == [
        hit.external_key for hit in without_parsing.hits
    ]
    assert [hit.score for hit in with_parsing.hits] == [
        hit.score for hit in without_parsing.hits
    ]
    for left, right in zip(with_parsing.hits, without_parsing.hits, strict=True):
        assert [chunk.chunk_id for chunk in left.matched_chunks] == [
            chunk.chunk_id for chunk in right.matched_chunks
        ]
        assert [chunk.breakdown for chunk in left.matched_chunks] == [
            chunk.breakdown for chunk in right.matched_chunks
        ]


async def test_the_breakdown_carries_the_detected_intent(corpus: Corpus) -> None:
    """A reinterpreted query has to be visible on the result that came back.

    Both fields, because they answer different questions: "recently" is detected
    and changes a weight, "in August" is detected and removes rows. A reader who
    cannot tell those apart cannot tell a reordering from a truncation.
    """
    ranged = await corpus.search("what changed in August", k=10)
    chunks = [chunk for hit in ranged.hits for chunk in hit.matched_chunks]
    assert chunks, "the fixture must return something for this to assert on"
    for chunk in chunks:
        assert chunk.breakdown is not None
        assert chunk.breakdown.temporal_intent is not None
        assert "range" in chunk.breakdown.temporal_intent
        assert "august" in chunk.breakdown.temporal_intent
        assert chunk.breakdown.temporal_filter_applied is True
        # And it survives the trip through `as_dict`, which is what the API and
        # the UI actually read.
        assert chunk.breakdown.as_dict()["temporal_filter_applied"] is True

    # A relative phrase is detected and changes ranking *without* filtering.
    # The distinction is the point of carrying two fields rather than one.
    relative = await corpus.search("what was I working on recently", k=10)
    assert relative.temporal_intent is not None
    assert relative.temporal_intent.startswith("relative")
    assert relative.temporal_filter_applied is False
    assert len(relative.hits) == 2, "a preference must not drop anything"
    for hit in relative.hits:
        for chunk in hit.matched_chunks:
            assert chunk.breakdown is not None
            assert chunk.breakdown.temporal_filter_applied is False

    # Ordering reorders what relevance selected, and does not filter either.
    ordered = await corpus.search("the first version of the lease handling", k=10)
    assert ordered.temporal_intent is not None
    assert ordered.temporal_intent.startswith("ordering earliest")
    assert ordered.temporal_filter_applied is False
    assert [hit.external_key for hit in ordered.hits] == [
        "july-notes.md",
        "august-notes.md",
    ]

    latest = await corpus.search("the latest change to the lease handling", k=10)
    assert [hit.external_key for hit in latest.hits] == [
        "august-notes.md",
        "july-notes.md",
    ]


def test_the_parser_reads_the_preposition_rather_than_only_the_month() -> None:
    """`since March` is not `during March`, and the difference is the answer.

    Collapsing all four prepositions to "the month named" would look right on
    this corpus — everything in it sits inside one month — and would silently
    answer a different question on any corpus that did not.
    """
    since = parse_temporal_intent("what changed since March 2026", now=NOW)
    assert since is not None
    assert since.start == datetime(2026, 3, 1, tzinfo=UTC)
    assert since.end is None, "'since' is open at the future end"

    during = parse_temporal_intent("what changed in March 2026", now=NOW)
    assert during is not None
    assert during.start == datetime(2026, 3, 1, tzinfo=UTC)
    assert during.end == datetime(2026, 4, 1, tzinfo=UTC)

    before = parse_temporal_intent("what existed before March 2026", now=NOW)
    assert before is not None
    assert before.start is None, "'before' is open at the past end"
    assert before.end == datetime(2026, 3, 1, tzinfo=UTC)

    after = parse_temporal_intent("what landed after March 2026", now=NOW)
    assert after is not None
    assert after.start == datetime(2026, 4, 1, tzinfo=UTC), "after the month ends"

    # A month with no year resolves backwards, never forwards: a corpus records
    # what has happened, so a window in the future cannot contain anything.
    past = parse_temporal_intent("what changed in December", now=NOW)
    assert past is not None and past.start == datetime(2025, 12, 1, tzinfo=UTC)

    # A date that does not exist parses to nothing rather than being widened to
    # the month. Widening reads as charitable and is not: it would apply a hard
    # filter derived from a phrase the parser has just failed to read.
    assert parse_temporal_intent("what changed on 31 February", now=NOW) is None

    # A range still resolves when the day is real, so the guard above is about
    # impossible dates rather than about day parsing being broken.
    real = parse_temporal_intent("what changed on 28 February", now=NOW)
    assert real is not None and real.kind is IntentKind.RANGE
    assert real.start == datetime(2026, 2, 28, tzinfo=UTC)
