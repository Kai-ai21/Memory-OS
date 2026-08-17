"""The chat endpoints, over the real app.

Only what the transport is responsible for: that a message posted as JSON comes
back with the classification the server made, that the transcript reads oldest
first, and that a status for a memory nobody has is a 404 rather than an empty
connection line. The behaviour underneath is `test_chat.py`'s.

There is no test here that posts a question. Doing so would construct a language
model from settings, and the deployment running this suite may not have a key —
which is the same reason `/answer` has no happy-path test at this level either.
"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.integration


async def test_a_statement_comes_back_classified_and_stored(client: AsyncClient) -> None:
    response = await client.post(
        "/chat", json={"text": "postgres full-text search is faster than I expected"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["intent"] == "statement"
    assert body["memory_id"] is not None
    assert body["answer"] is None
    # Null rather than false. A statement was not refused and was not answered,
    # and a client that read `false` here would draw it as an answered question.
    assert body["refused"] is None


async def test_the_transcript_reads_oldest_first(client: AsyncClient) -> None:
    for text in ("the first thing I typed", "the second thing I typed"):
        assert (await client.post("/chat", json={"text": text})).status_code == 201

    body = (await client.get("/chat")).json()

    assert [turn["text"] for turn in body] == [
        "the first thing I typed",
        "the second thing I typed",
    ]


async def test_an_empty_message_is_refused_rather_than_stored(client: AsyncClient) -> None:
    # Rejected by the schema before it reaches the use case, which is where a
    # length rule belongs. Whitespace-only gets there and is rejected by the use
    # case, so both routes are covered.
    assert (await client.post("/chat", json={"text": ""})).status_code == 422
    assert (await client.post("/chat", json={"text": "   \n "})).status_code == 422


async def test_status_reports_the_pipeline_before_anything_has_run(
    client: AsyncClient,
) -> None:
    """`searchable: false` is the whole point of this endpoint.

    The message is committed and the worker has not run, which is exactly the
    window the interface renders as `indexing…`. An endpoint that reported
    "stored" and left it there would let a UI imply a searchability that does
    not exist for another second or two.
    """
    posted = (
        await client.post("/chat", json={"text": "a thought about the job queue"})
    ).json()

    body = (await client.get(f"/chat/{posted['memory_id']}/status")).json()

    assert body["chunks"] == 0
    assert body["searchable"] is False
    assert body["extracted"] is False
    assert body["connections"] == []
    assert body["connected_memories"] == 0


async def test_status_for_a_memory_that_does_not_exist_is_a_404(
    client: AsyncClient,
) -> None:
    # "Nothing connects to this" and "this does not exist" are different
    # sentences, and an empty 200 would say the first when it meant the second.
    response = await client.get(
        "/chat/11111111-1111-7111-8111-111111111111/status"
    )
    assert response.status_code == 404
