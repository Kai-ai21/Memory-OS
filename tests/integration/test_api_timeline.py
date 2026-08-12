"""The three endpoints M4.1's timeline is a client of.

The UI can only render what these promise, so what gets asserted here is what it
is allowed to assume: that the buckets are dense and their kind breakdown sums to
their count, that a click on a bar can be turned into an exact range, and that
every date arrives with the provenance that says what it is worth.

The first test is about routing rather than time, and it earns its place: FastAPI
matches routes in registration order, so `/memories/{memory_id}` will happily
claim `/memories/at` and answer 422 about a malformed UUID. Nothing about that
failure mentions the timeline.
"""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db.repositories import (
    SqlAlchemyArtifactRepository,
    SqlAlchemyMemoryRepository,
)
from memoryos.domain.entities import Memory, RawArtifact, Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash, MemoryKind, TimeProvenance
from tests.integration.conftest import add_source

pytestmark = pytest.mark.integration

JANUARY = datetime(2026, 1, 1, tzinfo=UTC)
ARTIFACT = RawArtifact(content_hash=ContentHash.of(b"timeline api fixture"), byte_size=20)


@pytest.fixture
async def corpus(
    sessions: async_sessionmaker[AsyncSession], tmp_path_factory: pytest.TempPathFactory
) -> Source:
    """Nine memories across four months, with February deliberately empty.

    Two kinds, because a stacked bar with one kind in it cannot show whether the
    stack is right, and two provenances, because the UI's whole job in this
    milestone is to tell them apart.
    """
    source = await add_source(sessions, "timeline", tmp_path_factory.mktemp("corpus"))
    async with sessions.begin() as session:
        await SqlAlchemyArtifactRepository(session).add(ARTIFACT)

    plan = [
        (4, MemoryKind.CODE, TimeProvenance.FILESYSTEM),
        (5, MemoryKind.CODE, TimeProvenance.FILESYSTEM),
        (6, MemoryKind.CODE, TimeProvenance.FILESYSTEM),
        (10, MemoryKind.NOTE, TimeProvenance.DECLARED),
        # Nothing in February. The gap the chart has to draw.
        (61, MemoryKind.CODE, TimeProvenance.FILESYSTEM),
        (62, MemoryKind.CODE, TimeProvenance.FILESYSTEM),
        (95, MemoryKind.NOTE, TimeProvenance.DECLARED),
        (96, MemoryKind.NOTE, TimeProvenance.DECLARED),
        (97, MemoryKind.NOTE, TimeProvenance.DECLARED),
    ]
    for day, kind, provenance in plan:
        async with sessions.begin() as session:
            await SqlAlchemyMemoryRepository(session).add_version(
                Memory(
                    id=new_id(),
                    source_id=source.id,
                    external_key=f"item-{day}.md",
                    content_hash=ARTIFACT.content_hash,
                    kind=kind,
                    occurred_at=JANUARY + timedelta(days=day),
                    occurred_at_source=provenance,
                )
            )
    return source


async def test_memories_at_is_not_shadowed_by_the_memory_detail_route(
    client: AsyncClient, corpus: Source
) -> None:
    """`/memories/at` reaches its own handler rather than the `{memory_id}` one.

    A literal segment registered after the parameterised one that matches it is
    dead, and dead in the most confusing way available: the request 422s about a
    UUID nobody asked for. Asserted on the status *and* the body shape, because a
    404 here would also be wrong and would look like an empty corpus.
    """
    response = await client.get(
        "/memories/at", params={"date": "2026-01-04", "window_days": 1}
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"start", "end", "total", "memories"}


async def test_timeline_buckets_are_dense_and_their_kinds_sum_to_their_counts(
    client: AsyncClient, corpus: Source
) -> None:
    response = await client.get("/timeline", params={"period": "month", "source": "timeline"})

    assert response.status_code == 200
    body = response.json()

    starts = [bucket["start"][:7] for bucket in body["buckets"]]
    # February is present with a count of zero rather than absent. A `GROUP BY`
    # cannot produce that row, which is why the layer generates it — and it is
    # the row the chart draws its hatch on.
    assert starts == ["2026-01", "2026-02", "2026-03", "2026-04"]
    assert [bucket["count"] for bucket in body["buckets"]] == [4, 0, 2, 3]
    assert body["buckets"][1]["by_kind"] == {}

    for bucket in body["buckets"]:
        assert sum(bucket["by_kind"].values()) == bucket["count"], (
            "the stacked bar and the number above it come from one GROUP BY "
            "and must agree"
        )
    assert body["total"] == sum(bucket["count"] for bucket in body["buckets"]) == 9

    # The profile travels with the histogram, so a caller cannot draw the chart
    # without having been told what its dates are worth.
    bands = {band["provenance"]: band["count"] for band in body["provenance"]}
    assert bands == {"filesystem": 5, "declared": 4, "unknown": 0}


async def test_a_bucket_can_be_turned_back_into_exactly_its_own_memories(
    client: AsyncClient, corpus: Source
) -> None:
    """What clicking a bar does, and that adjacent bars do not double-count.

    The window runs forward from the bucket's start for the bucket's own length,
    which is 31 days for January and 28 for February. A fixed 30 would leak the
    31st into the next bar and lose it from this one.
    """
    timeline = (
        await client.get("/timeline", params={"period": "month", "source": "timeline"})
    ).json()

    seen: list[str] = []
    for bucket in timeline["buckets"]:
        days = (
            datetime.fromisoformat(bucket["end"]) - datetime.fromisoformat(bucket["start"])
        ).days
        response = await client.get(
            "/memories/at",
            params={"date": bucket["start"], "window_days": days, "source": "timeline"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == bucket["count"], f"{bucket['start']} disagrees with its bar"
        seen.extend(memory["external_key"] for memory in body["memories"])

    # Every memory once, and only once: the windows tile the year rather than
    # overlapping at the boundaries.
    assert len(seen) == len(set(seen)) == 9

    january = (
        await client.get(
            "/memories/at",
            params={"date": "2026-01-01T00:00:00Z", "window_days": 31, "source": "timeline"},
        )
    ).json()
    # And every row carries its provenance, which is what the UI marks on.
    assert {memory["occurred_at_source"] for memory in january["memories"]} == {
        "filesystem",
        "declared",
    }


async def test_gaps_come_back_with_the_memories_on_either_side(
    client: AsyncClient, corpus: Source
) -> None:
    response = await client.get("/gaps", params={"min_days": 30, "source": "timeline"})

    assert response.status_code == 200
    gaps = response.json()
    assert len(gaps) == 2

    first = gaps[0]
    assert first["source_name"] == "timeline"
    assert round(first["days"]) == 51
    # Named on both sides, because "when did this stop" is not answerable from
    # two bare timestamps — the reader needs to know what was last touched.
    assert first["before"]["external_key"] == "item-10.md"
    assert first["after"]["external_key"] == "item-61.md"
    assert first["before"]["occurred_at_source"] == "declared"

    # And nothing at a threshold longer than any silence in the corpus.
    assert (
        await client.get("/gaps", params={"min_days": 90, "source": "timeline"})
    ).json() == []


async def test_an_unknown_source_is_a_404_rather_than_an_empty_chart(
    client: AsyncClient, corpus: Source
) -> None:
    """A typo in a source name must not look like a source with no history."""
    for path in ("/timeline", "/gaps"):
        response = await client.get(path, params={"source": "no-such-source"})
        assert response.status_code == 404
