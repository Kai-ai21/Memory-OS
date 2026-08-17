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
    assert body["session_id"] is not None
    # One message, not two. A statement is answered by nothing, so there is no
    # assistant turn to render.
    assert len(body["messages"]) == 1
    message = body["messages"][0]
    assert message["role"] == "user"
    assert message["intent"] == "statement"
    assert message["memory_id"] is not None
    assert message["ordinal"] == 0
    # Null rather than false. A statement was not refused and was not answered,
    # and a client that read `false` here would draw it as an answered question.
    assert message["refused"] is None


async def test_the_transcript_reads_oldest_first(client: AsyncClient) -> None:
    posted = [
        (await client.post("/chat", json={"text": text})).json()
        for text in ("the first thing I typed", "the second thing I typed")
    ]
    # Both landed in one conversation: nothing waited thirty minutes.
    assert posted[0]["session_id"] == posted[1]["session_id"]

    body = (await client.get(f"/chat/{posted[0]['session_id']}")).json()

    assert [turn["content"] for turn in body] == [
        "the first thing I typed",
        "the second thing I typed",
    ]
    assert [turn["ordinal"] for turn in body] == [0, 1]


async def test_the_session_list_carries_the_derived_title(client: AsyncClient) -> None:
    await client.post("/chat", json={"text": "postgres is doing the keyword half"})

    body = (await client.get("/chat/sessions")).json()

    assert len(body) == 1
    assert body[0]["title"] == "postgres is doing the keyword half"
    assert body[0]["message_count"] == 1
    assert body[0]["archived_at"] is None


async def test_archiving_hides_a_session_without_deleting_it(
    client: AsyncClient,
) -> None:
    """Hidden, and available by asking. Nothing is deleted, ever."""
    posted = (await client.post("/chat", json={"text": "a thought to put away"})).json()
    session_id = posted["session_id"]

    assert (
        await client.post(f"/chat/sessions/{session_id}/archive")
    ).status_code == 204

    assert (await client.get("/chat/sessions")).json() == []
    hidden = (await client.get("/chat/sessions?include_archived=true")).json()
    assert [row["id"] for row in hidden] == [session_id]
    assert hidden[0]["archived_at"] is not None

    # And the messages are still there, which is the difference between archiving
    # and deleting.
    assert len((await client.get(f"/chat/{session_id}")).json()) == 1

    assert (
        await client.post(f"/chat/sessions/{session_id}/archive?archived=false")
    ).status_code == 204
    assert [row["id"] for row in (await client.get("/chat/sessions")).json()] == [
        session_id
    ]


async def test_searching_within_a_session_filters_that_session_only(
    client: AsyncClient,
) -> None:
    """A substring over rows the reader has already seen, and nothing more.

    Deliberately not corpus search: this answers "where in this conversation did
    I say that", and running it through the embedder would return semantic
    neighbours from a conversation somebody can see all of.
    """
    posted = (
        await client.post("/chat", json={"text": "the reranker costs ten seconds"})
    ).json()
    session_id = posted["session_id"]
    await client.post(
        "/chat", json={"text": "the embedder costs seven", "session_id": session_id}
    )

    found = (await client.get(f"/chat/{session_id}?q=reranker")).json()

    assert [row["content"] for row in found] == ["the reranker costs ten seconds"]


async def test_archiving_a_session_that_does_not_exist_is_a_404(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/chat/sessions/11111111-1111-7111-8111-111111111111/archive"
    )
    assert response.status_code == 404


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
    memory_id = posted["messages"][0]["memory_id"]

    body = (await client.get(f"/chat/messages/{memory_id}/status")).json()

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
        "/chat/messages/11111111-1111-7111-8111-111111111111/status"
    )
    assert response.status_code == 404
