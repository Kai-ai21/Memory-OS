"""Correcting, deleting and organising memories, and one test that matters most.

**"Users can permanently delete memories" has been a stated guarantee since Phase
1, and until M10.4 nothing in this system could exercise it.** Tombstoning existed
because a full sync writes one when a file disappears; nothing could delete
anything on purpose, which made the guarantee a claim about the schema rather than
about the product.

So five claims, and the third one is the guardrail:

1. A correction creates version 2 and version 1 survives.
2. A permanent deletion removes the memory, its chunks, its mentions and its graph
   nodes.
3. **A permanently deleted memory's distinctive text returns nothing from search.**
4. Deleting a source removes its memories and nobody else's.
5. An export round-trips, version history included.

The third is not a restatement of the second. A deletion that removes rows and
leaves the content findable is not a deletion — the chunk could survive in the
vector index, or the tombstone filter could be doing the hiding while the row is
still there, and a row-count assertion cannot tell any of those apart. The only
question a person actually asks is "is it gone", and the only way to answer it is
to search for the text and get nothing.
"""

import json
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.db import models
from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.application import deletion, export, graph_projection
from memoryos.application import tags as tags_module
from memoryos.application.chat import Chat
from memoryos.application.graph_sync import SyncGraph
from memoryos.application.ports import SearchFilters
from memoryos.application.search import SearchMemories
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import (
    ContentHash,
    EntityType,
    EventType,
    GraphLabel,
    SearchMode,
)
from tests.integration.test_chat import build, drain
from tests.support.fakes import FakeEmbedder, InMemoryGraphStore

pytestmark = pytest.mark.integration

# A phrase that appears nowhere else in any fixture in this suite, so that a hit
# for it can only have come from the memory under test. "distinctive" is the
# milestone's own word and it has to be true of the string or the search test
# proves nothing.
DISTINCTIVE = "the marmalade quantifier rebalances every antimeridian ledger"


def searcher(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> SearchMemories:
    """The real search use case, both retrievers, over the test corpus.

    Hybrid rather than one half, because that is what the interface uses and
    because the two fail in opposite directions: a deletion that left the vector
    index intact would be caught by one and a deletion that left the tsvector
    intact by the other.
    """
    embedder = FakeEmbedder()
    return SearchMemories(
        sessions,
        embedder,
        PgVectorStore(sessions, embedder, default_ef_search=100),
        PostgresKeywordStore(sessions),
    )


async def store(
    chat: Chat, tmp_path: Path, sessions: async_sessionmaker[AsyncSession], text: str
) -> UUID:
    """Type a statement and run its pipeline. Returns the memory id."""
    exchange = await chat(text)
    assert exchange.user.memory_id is not None
    await drain(tmp_path, sessions)
    return exchange.user.memory_id


async def mention(
    sessions: async_sessionmaker[AsyncSession], memory_id: UUID, name: str
) -> UUID:
    """An entity mention on this memory's first chunk, written directly.

    Extraction is an LLM call per chunk and these tests are about deletion, not
    about extraction — so the row is written here, at a real offset in a real
    chunk, which is what the cascade has to reach. Faking the *shape* would be a
    test of a fake; this is the real row.
    """
    async with sessions.begin() as session:
        chunk = (
            await session.execute(
                select(models.MemoryChunk)
                .where(models.MemoryChunk.memory_id == memory_id)
                .order_by(models.MemoryChunk.ordinal)
                .limit(1)
            )
        ).scalars().one()
        entity_id = new_id()
        session.add(
            models.Entity(
                id=entity_id,
                name=name,
                canonical_name=name.casefold(),
                type=EntityType.CONCEPT.value,
            )
        )
        await session.flush()
        session.add(
            models.EntityMention(
                id=new_id(),
                entity_id=entity_id,
                memory_id=memory_id,
                chunk_id=chunk.id,
                char_start=0,
                char_end=min(len(name), chunk.char_end - chunk.char_start),
                extractor_version="test",
            )
        )
    return entity_id


# --------------------------------------------------------------------------
# 1. A correction creates version 2, with version 1 preserved
# --------------------------------------------------------------------------


async def test_a_correction_creates_version_two_and_keeps_version_one(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The corpus gets a history; the transcript keeps both turns.

    **Both halves are the claim.** A correction that replaced the text in place
    would leave the corpus correct and the record poorer: what somebody believed
    before they corrected it is exactly what Phase 5's reflections reason over, and
    an in-place edit deletes it silently.

    The corpus side is a *version*, not a second memory. Two memories saying
    different things about one thought would both be retrievable, and retrieval
    would then choose between them on similarity — which is the failure
    `is_current` exists to prevent.
    """
    chat = build(tmp_path, sessions)
    first = "postgres full-text search is faster than I expected"
    memory_id = await store(chat, tmp_path, sessions, first)

    async with sessions() as session:
        original = await session.get(models.Memory, memory_id)
        assert original is not None
        key, source_id = original.external_key, original.source_id
        message_id = (
            await session.execute(
                select(models.ChatMessage.id).where(
                    models.ChatMessage.external_key == key
                )
            )
        ).scalar_one()

    corrected = "postgres full-text search is slower than I expected"
    exchange = await chat.correct(message_id, corrected)
    await drain(tmp_path, sessions)

    async with sessions() as session:
        versions = (
            await session.execute(
                select(models.Memory)
                .where(
                    models.Memory.source_id == source_id,
                    models.Memory.external_key == key,
                )
                .order_by(models.Memory.version)
            )
        ).scalars().all()

        # Two versions of one item, under one external key. Not two items.
        assert [row.version for row in versions] == [1, 2]
        assert [row.is_current for row in versions] == [False, True]
        assert versions[0].id == memory_id
        assert versions[1].id == exchange.user.memory_id

        # Version 1's text is *still there*, which is the half an in-place edit
        # would have destroyed.
        assert versions[0].content is not None
        assert first in versions[0].content
        assert versions[1].content is not None
        assert corrected in versions[1].content

        # Both turns are in the transcript, linked, and the original is not
        # rewritten: its content is what it always said.
        turns = (
            await session.execute(
                select(models.ChatMessage)
                .where(models.ChatMessage.external_key == key)
                .order_by(models.ChatMessage.ordinal)
            )
        ).scalars().all()
        assert len(turns) == 2
        assert turns[0].content == first
        assert turns[0].corrects_message_id is None
        assert turns[1].content == corrected
        assert turns[1].corrects_message_id == turns[0].id

    # And the read path reports the supersession in both directions, so an
    # interface can dim the original without scanning the conversation for a
    # message that mentions it.
    from memoryos.application import chat as chat_use_case

    messages = await chat_use_case.messages(sessions, exchange.session_id)
    stored = [message for message in messages if message.external_key == key]
    assert stored[0].superseded_by == stored[1].id
    assert stored[1].corrects == stored[0].id

    # Only the current version is retrievable. A superseded version's chunks
    # describe text the item no longer says.
    hits = await searcher(tmp_path, sessions)(
        "postgres full-text search", k=10, mode=SearchMode.HYBRID
    )
    assert exchange.user.memory_id in {hit.memory_id for hit in hits.hits}
    assert memory_id not in {hit.memory_id for hit in hits.hits}


async def test_a_correction_reuses_the_chunks_it_did_not_change(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """M1.4's chunk adoption applies, so a trivial correction costs almost nothing.

    The milestone asks for this by name — "a trivial correction should cost
    nothing" — and the reason it holds is that nothing about the correction path is
    special: `ingest_item` records a new version and `normalize` matches the new
    text against the previous version's chunks, exactly as it does for a file
    somebody edited.

    Asserted as chunk *identity* across the two versions rather than as a timing.
    An adopted chunk is the same row moved across, so its content hash is
    unchanged, and a re-chunk that happened to be fast would still fail this.
    """
    chat = build(tmp_path, sessions)
    # Long enough to be several chunks, which is what makes adoption observable at
    # all: the chunker's window is 512 model tokens, so a four-paragraph note is one
    # chunk and there is nothing for a correction to leave alone. Twelve sections of
    # distinct prose, with the typo in the last one.
    sections = [
        f"## Section {index}\n\n"
        + " ".join(
            f"The {word} in section {index} is written out at length so that this "
            f"paragraph carries real tokens rather than a placeholder."
            for word in ("queue", "cursor", "projection", "boundary")
        )
        for index in range(12)
    ]
    long_note = "\n\n".join(
        ["# The queue", *sections, "One last line, with a typo: search is fastre."]
    )
    memory_id = await store(chat, tmp_path, sessions, long_note)

    async with sessions() as session:
        before = {
            (row.ordinal, row.content_hash)
            for row in (
                await session.execute(
                    select(models.MemoryChunk).where(
                        models.MemoryChunk.memory_id == memory_id
                    )
                )
            ).scalars()
        }
        key = (await session.get(models.Memory, memory_id)).external_key  # type: ignore[union-attr]
        message_id = (
            await session.execute(
                select(models.ChatMessage.id).where(
                    models.ChatMessage.external_key == key
                )
            )
        ).scalar_one()

    exchange = await chat.correct(message_id, long_note.replace("fastre", "faster"))
    await drain(tmp_path, sessions)

    async with sessions() as session:
        after = {
            (row.ordinal, row.content_hash)
            for row in (
                await session.execute(
                    select(models.MemoryChunk).where(
                        models.MemoryChunk.memory_id == exchange.user.memory_id
                    )
                )
            ).scalars()
        }

    shared = before & after
    # At least one chunk survived the correction unchanged. On a short note the
    # whole text is one chunk and there is nothing to adopt, which is why this
    # fixture is several paragraphs — and asserting "some" rather than "all but
    # one" keeps this a test of adoption rather than of the chunker's boundaries.
    assert shared, "no chunk was adopted; a trivial correction re-chunked everything"


# --------------------------------------------------------------------------
# 2. Permanent deletion removes memory, chunks, mentions and graph nodes
# --------------------------------------------------------------------------


async def test_permanent_deletion_removes_memory_chunks_mentions_and_graph_nodes(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Every version, and everything hanging off them.

    The graph half is checked against the *projection* rather than against a live
    Neo4j, and that is the stronger assertion rather than a convenience: the
    projection is a pure function of Postgres, so it is what the corpus now
    *implies*. A sync writing that projection cannot produce a node for a memory
    the projection does not contain — and `graph verify` is where a real graph is
    compared against it.
    """
    chat = build(tmp_path, sessions)
    memory_id = await store(chat, tmp_path, sessions, DISTINCTIVE)
    entity_id = await mention(sessions, memory_id, "marmalade")

    async with sessions() as session:
        row = await session.get(models.Memory, memory_id)
        assert row is not None
        key, source_id, content_hash = (
            row.external_key,
            row.source_id,
            row.content_hash,
        )

    # In the graph the corpus implies, before the deletion.
    projection = await graph_projection.read(sessions)
    assert memory_id in {node.memory_id for node in projection.memories}
    assert entity_id in {node.entity_id for node in projection.entities}

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    assert await blobs.exists(ContentHash(content_hash))

    scope = await deletion.scope_of_memory(sessions, memory_id)
    assert scope.memories == 1
    assert scope.chunks >= 1
    assert scope.mentions == 1
    # The entity's only mention is in what is being deleted, so the corpus will
    # stop knowing about it. Counted separately from `mentions` because "9 mentions
    # removed" and "2 entities gone" are different facts.
    assert scope.orphaned_entities == 1

    report = await deletion.purge_memory(sessions, blobs, memory_id)
    assert report.memories == 1
    assert report.chunks == scope.chunks
    assert report.mentions == 1
    assert report.blobs_shredded == 1
    assert report.blobs_surviving == ()

    async with sessions() as session:
        assert await session.get(models.Memory, memory_id) is None
        assert (
            await session.execute(
                select(func.count(models.Memory.id)).where(
                    models.Memory.source_id == source_id,
                    models.Memory.external_key == key,
                )
            )
        ).scalar_one() == 0
        # Chunks — and therefore vectors, which live on them — cascaded.
        assert (
            await session.execute(
                select(func.count(models.MemoryChunk.id)).where(
                    models.MemoryChunk.memory_id == memory_id
                )
            )
        ).scalar_one() == 0
        # Mentions cascaded, by both of their foreign keys.
        assert (
            await session.execute(
                select(func.count(models.EntityMention.id)).where(
                    models.EntityMention.memory_id == memory_id
                )
            )
        ).scalar_one() == 0
        # The transcript row went too, explicitly rather than by cascade: it holds
        # the message text a second time, so leaving it would keep the content in
        # the most visible place in the product.
        assert (
            await session.execute(
                select(func.count(models.ChatMessage.id)).where(
                    models.ChatMessage.external_key == key
                )
            )
        ).scalar_one() == 0

        # **The log keeps its record, and that is the crypto-shredding tension
        # rather than an oversight.** The observation is still there and a purge
        # event has been appended beside it.
        events = (
            await session.execute(
                select(models.IngestionEvent.event_type)
                .where(models.IngestionEvent.external_key == key)
                .order_by(models.IngestionEvent.seq)
            )
        ).scalars().all()
        assert events == [
            EventType.ARTIFACT_OBSERVED.value,
            EventType.ITEM_PURGED.value,
        ]

    # The bytes are gone. Without this the content is still on disk and a replay
    # would rebuild it.
    assert not await blobs.exists(ContentHash(content_hash))

    # And the graph the corpus implies no longer contains the memory or the entity
    # whose only mention it was.
    after = await graph_projection.read(sessions)
    assert memory_id not in {node.memory_id for node in after.memories}
    assert entity_id not in {node.entity_id for node in after.entities}
    assert not [
        edge
        for edge in after.edges
        if str(memory_id) in (edge.start.key, edge.end.key)
    ]


async def test_the_graph_sync_prunes_a_purged_memorys_nodes(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The projection is not enough on its own: something has to write it.

    A purge enqueues a `SYNC_GRAPH` job in the same transaction as the deletion, and
    the real sync is what runs. Its own "prune the scope, then project what Postgres
    implies" pass is what removes the node — not a special case for deletion, which
    is why the payload naming an id that no longer exists is exactly right.

    Run against an in-memory store rather than Neo4j, because the claim is that the
    prune happens, and `graph verify` is where a real graph is compared against the
    projection.
    """
    chat = build(tmp_path, sessions)
    memory_id = await store(chat, tmp_path, sessions, DISTINCTIVE)
    entity_id = await mention(sessions, memory_id, "marmalade")

    graph = InMemoryGraphStore()
    await graph_projection.write(graph, await graph_projection.read(sessions))
    assert await _has_node(graph, GraphLabel.MEMORY, memory_id)
    assert await _has_node(graph, GraphLabel.ENTITY, entity_id)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    await deletion.purge_memory(sessions, blobs, memory_id)

    # The job was queued in the deletion's own transaction, so there is no window in
    # which the rows are gone and the job that removes their nodes does not exist.
    async with sessions() as session:
        payloads = (
            await session.execute(
                select(models.Job.payload).where(
                    models.Job.job_type == JobType.SYNC_GRAPH.value
                )
            )
        ).all()
    assert payloads, "a purge queued no graph sync"

    sync = SyncGraph(sessions, graph)
    for (payload,) in payloads:
        await sync(payload)

    # The memory's node is gone, and so is the entity whose only mention it was —
    # the sync widened its scope to reach it, which is what `expand` is for.
    assert not await _has_node(graph, GraphLabel.MEMORY, memory_id)
    assert not await _has_node(graph, GraphLabel.ENTITY, entity_id)


async def _has_node(
    graph: InMemoryGraphStore, label: GraphLabel, key: UUID
) -> bool:
    """Whether the graph holds this node, read back through its own accessor.

    Through `all_nodes` rather than by reaching into the fake's dict, so this
    asserts what a reader of the graph would see — which is the only thing the
    projection promises.
    """
    return any(
        node.label is label and node.key == str(key)
        for node in await graph.all_nodes()
    )


# --------------------------------------------------------------------------
# 3. The one that matters: the text is unfindable
# --------------------------------------------------------------------------


async def test_a_purged_memorys_distinctive_text_returns_nothing_from_search(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """**A deletion that leaves the content findable is not a deletion.**

    This is the test the milestone singles out, and it is not a restatement of the
    row counts above. Those can all pass while the content is still reachable: a
    chunk could survive in the vector index under a memory row that is gone, the
    tombstone filter could be doing the hiding while everything is still stored, or
    the tsvector could match text nothing else references. A row count cannot
    distinguish any of those from a deletion.

    So this asks the only question a person actually asks — is it gone — in the only
    way that answers it. The phrase is searched for first, to prove the search can
    find it at all: an assertion that a query returns nothing is worthless unless
    the same query returned something a moment earlier.

    All three modes, because the two retrievers fail in opposite directions and a
    deletion that satisfied one is exactly the kind of half-deletion this exists to
    catch. `include_deleted=True` is asked as well, which is the strongest form of
    the question: not "is it filtered" but "is it there".
    """
    chat = build(tmp_path, sessions)
    memory_id = await store(chat, tmp_path, sessions, DISTINCTIVE)
    search = searcher(tmp_path, sessions)

    # It is findable. Every mode, so that "nothing comes back" later is a change
    # rather than a property of the fixture.
    for mode in (SearchMode.HYBRID, SearchMode.VECTOR, SearchMode.KEYWORD):
        found = await search(DISTINCTIVE, k=10, mode=mode)
        assert memory_id in {hit.memory_id for hit in found.hits}, mode

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    await deletion.purge_memory(sessions, blobs, memory_id)

    for mode in (SearchMode.HYBRID, SearchMode.VECTOR, SearchMode.KEYWORD):
        found = await search(DISTINCTIVE, k=10, mode=mode)
        assert found.hits == [], f"{mode} still returns the purged text"

    # And with the tombstone filter explicitly disabled, which is the difference
    # between "excluded from results" and "gone". A tombstoned memory would still
    # be returned here; a purged one has nothing to return.
    unfiltered = await search(
        DISTINCTIVE,
        k=10,
        mode=SearchMode.HYBRID,
        filters=SearchFilters(include_deleted=True),
    )
    assert unfiltered.hits == []

    # Not one chunk of that text is left anywhere in the corpus, whatever memory it
    # might be attached to. The narrowest form of the same question, asked of the
    # table rather than of the retriever.
    async with sessions() as session:
        assert (
            await session.execute(
                select(func.count(models.MemoryChunk.id)).where(
                    models.MemoryChunk.content.ilike(f"%{DISTINCTIVE}%")
                )
            )
        ).scalar_one() == 0
        assert (
            await session.execute(
                select(func.count(models.Memory.id)).where(
                    models.Memory.content.ilike(f"%{DISTINCTIVE}%")
                )
            )
        ).scalar_one() == 0


async def test_removing_from_view_hides_the_text_and_keeps_it(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The other level, and the difference between them, in one test.

    Tombstoning has to be *exactly* as unfindable as a purge through the ordinary
    search path and *entirely* recoverable underneath it. Both halves asserted
    together, because the two levels are only meaningfully different if each one
    delivers what it promises: hiding that left results visible would be
    pointless, and hiding that destroyed the bytes would make the recoverable
    level a lie.
    """
    chat = build(tmp_path, sessions)
    memory_id = await store(chat, tmp_path, sessions, DISTINCTIVE)
    search = searcher(tmp_path, sessions)
    blobs = FilesystemBlobStore(tmp_path / "blobs")

    await deletion.tombstone(sessions, memory_id)

    # Gone from search, by the filter every retrieval path shares.
    assert (await search(DISTINCTIVE, k=10, mode=SearchMode.HYBRID)).hits == []

    # And still entirely there underneath it, which is what makes it recoverable.
    async with sessions() as session:
        row = await session.get(models.Memory, memory_id)
        assert row is not None
        assert row.deleted_at is not None
        assert row.content is not None and DISTINCTIVE in row.content
        assert await blobs.exists(ContentHash(row.content_hash))

    # Restored through the ordinary version path rather than by clearing a column,
    # so a replay reproduces the result without knowing restoration exists.
    await deletion.restore(sessions, blobs, memory_id)
    await drain(tmp_path, sessions)
    restored = await search(DISTINCTIVE, k=10, mode=SearchMode.HYBRID)
    assert restored.hits, "a restored memory is not searchable again"


# --------------------------------------------------------------------------
# 4. Deleting a source removes only its memories
# --------------------------------------------------------------------------


async def test_deleting_a_source_removes_only_its_own_memories(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The most destructive operation in the product, scoped.

    Two sources, one deleted. The assertion that matters is the *negative* one:
    everything from the other source is byte-identical afterwards. A source
    deletion implemented with a predicate one clause short would pass every count
    on the deleted side and take the corpus with it.
    """
    chat = build(tmp_path, sessions)
    kept_id = await store(chat, tmp_path, sessions, "the chat source keeps this one")

    # A second source with its own items, registered and ingested the ordinary way.
    from memoryos.application.ingest import ingest_item
    from memoryos.application.ports import ObservedItem
    from memoryos.domain.entities import Source
    from memoryos.domain.values import SourceKind, TimeProvenance

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    other = Source(
        id=new_id(), kind=SourceKind.FILESYSTEM, name="notes", config={"root": "/tmp"}
    )
    async with sessions.begin() as session:
        from memoryos.adapters.db.repositories import SqlAlchemySourceRepository

        await SqlAlchemySourceRepository(session).add(other)

    doomed_keys = ["notes/one.md", "notes/two.md"]
    for key in doomed_keys:
        data = f"# {key}\n\n{DISTINCTIVE} in {key}".encode()

        async def read(payload: bytes = data) -> bytes:
            return payload

        async with sessions.begin() as session:
            await ingest_item(
                session,
                blobs,
                other,
                ObservedItem(
                    external_key=key,
                    content_hash=ContentHash.of(data),
                    byte_size=len(data),
                    media_type="text/markdown",
                    occurred_at=None,
                    occurred_at_source=TimeProvenance.UNKNOWN,
                    read_bytes=read,
                    fingerprint=None,
                ),
            )
    await drain(tmp_path, sessions)

    async with sessions() as session:
        kept_before = (
            await session.execute(
                select(models.Memory.id, models.Memory.content_hash, models.Memory.content)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Source.name != "notes")
                .order_by(models.Memory.external_key)
            )
        ).all()
        kept_chunks_before = (
            await session.execute(
                select(func.count(models.MemoryChunk.id))
                .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Source.name != "notes")
            )
        ).scalar_one()

    scope = await deletion.scope_of_source(sessions, other.id)
    assert len(scope.items) == 2
    assert scope.memories == 2

    report = await deletion.purge_source(sessions, blobs, other.id)
    assert report.items == 2
    assert report.memories == 2

    async with sessions() as session:
        # Nothing of the deleted source's is left.
        assert (
            await session.execute(
                select(func.count(models.Memory.id)).where(
                    models.Memory.source_id == other.id
                )
            )
        ).scalar_one() == 0

        # And everything else is exactly as it was — the same rows, the same
        # hashes, the same text, and the same number of chunks.
        kept_after = (
            await session.execute(
                select(models.Memory.id, models.Memory.content_hash, models.Memory.content)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Source.name != "notes")
                .order_by(models.Memory.external_key)
            )
        ).all()
        kept_chunks_after = (
            await session.execute(
                select(func.count(models.MemoryChunk.id))
                .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Source.name != "notes")
            )
        ).scalar_one()

    assert kept_after == kept_before
    assert kept_chunks_after == kept_chunks_before
    assert kept_id in {row[0] for row in kept_after}

    # The kept memory is still searchable. The strongest form of "only its own":
    # not merely present in a count, but still answering.
    found = await searcher(tmp_path, sessions)(
        "the chat source keeps this one", k=10, mode=SearchMode.HYBRID
    )
    assert kept_id in {hit.memory_id for hit in found.hits}

    # The source's registration survives, because its ingestion events hold a
    # foreign key to it and those events are the record that it existed. Reported
    # rather than silent.
    async with sessions() as session:
        assert await session.get(models.Source, other.id) is not None


# --------------------------------------------------------------------------
# 5. Export round-trips, including versions
# --------------------------------------------------------------------------


async def test_export_round_trips_including_version_history(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """What is exported is what is stored, versions and all.

    **An export that flattened to current state would lose what the system is
    for.** So the assertion is not "the file parses" but a field-by-field
    comparison against the database for *every* version — the superseded text
    included, since that is the half a flattening export would drop and the half
    Phase 4 and 5 rest on.

    A tombstoned item is exported with its deletion recorded rather than omitted:
    "this was here and I removed it from view" is a fact about the corpus.
    """
    chat = build(tmp_path, sessions)
    first = "the queue is a table so a memory cannot exist without its job"
    memory_id = await store(chat, tmp_path, sessions, first)
    hidden_id = await store(chat, tmp_path, sessions, "this one is removed from view")

    async with sessions() as session:
        key = (await session.get(models.Memory, memory_id)).external_key  # type: ignore[union-attr]
        message_id = (
            await session.execute(
                select(models.ChatMessage.id).where(
                    models.ChatMessage.external_key == key
                )
            )
        ).scalar_one()

    corrected = "the queue is a table so a memory cannot exist without the job that processes it"
    await chat.correct(message_id, corrected)
    await drain(tmp_path, sessions)
    await deletion.tombstone(sessions, hidden_id)

    async with sessions() as session:
        tagged = await deletion.item_of_memory(session, memory_id)
    await tags_module.apply(
        sessions,
        source_id=tagged.source_id,
        external_key=tagged.external_key,
        tags=tags_module.parse_required("#queue #idea"),
    )

    document = "".join([piece async for piece in export.to_json(sessions)])
    parsed = json.loads(document)

    assert parsed["meta"]["format"] == "memoryos-export"
    assert parsed["meta"]["includes_version_history"] is True

    exported = {entry["external_key"]: entry for entry in parsed["items"]}

    # Every version, in order, and every field matched against the database rather
    # than against another copy of the fixture.
    async with sessions() as session:
        rows = (
            await session.execute(
                select(models.Memory, models.Source.name)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .order_by(models.Memory.external_key, models.Memory.version)
            )
        ).all()

    assert rows, "nothing in the corpus to round-trip"
    seen = 0
    for row, source_name in rows:
        entry = exported[row.external_key]
        assert entry["source"] == source_name
        version = next(
            item for item in entry["versions"] if item["version"] == row.version
        )
        assert version["content"] == row.content
        assert version["content_hash"] == row.content_hash
        assert version["is_current"] == row.is_current
        assert version["title"] == row.title
        assert version["kind"] == row.kind
        assert version["occurred_at_source"] == row.occurred_at_source
        assert (version["deleted_at"] is not None) == (row.deleted_at is not None)
        seen += 1
    assert seen == len(rows)

    # The corrected item carries *both* versions, which is the point.
    history = exported[key]["versions"]
    assert [version["version"] for version in history] == [1, 2]
    assert first in history[0]["content"]
    assert corrected in history[1]["content"]
    assert history[0]["is_current"] is False and history[1]["is_current"] is True

    # Tags travel with it, as typed.
    assert set(exported[key]["tags"]) == {"#idea", "#queue"}

    # And the tombstoned item is present with its deletion recorded, rather than
    # silently dropped.
    async with sessions() as session:
        hidden_key = (await session.get(models.Memory, hidden_id)).external_key  # type: ignore[union-attr]
    hidden = exported[hidden_key]
    assert hidden["versions"][-1]["deleted_at"] is not None

    # Markdown renders the same corpus and says what it cannot carry, rather than
    # implying completeness.
    readable = "".join([piece async for piece in export.to_markdown(sessions)])
    assert first in readable
    assert corrected in readable
    assert "(superseded)" in readable and "(current)" in readable
    assert "`--format json` is the one to use" in readable


async def test_a_purged_item_is_absent_from_an_export_entirely(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Not "reported as deleted" — absent.

    An export that recorded the purge would leak the thing the purge removed: the
    key, and with it the fact that this particular text was here. The log holds the
    event; the export is of the corpus.
    """
    chat = build(tmp_path, sessions)
    memory_id = await store(chat, tmp_path, sessions, DISTINCTIVE)
    async with sessions() as session:
        key = (await session.get(models.Memory, memory_id)).external_key  # type: ignore[union-attr]

    await deletion.purge_memory(
        sessions, FilesystemBlobStore(tmp_path / "blobs"), memory_id
    )

    document = "".join([piece async for piece in export.to_json(sessions)])
    assert DISTINCTIVE not in document
    assert key not in document


# --------------------------------------------------------------------------
# Tagging
# --------------------------------------------------------------------------


async def test_a_tag_is_a_concept_entity_and_connects_through_it(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The whole argument for tags not being their own table.

    A tag resolves to the `CONCEPT` row an extractor would have written for the
    same word — so tagging one memory `#marmalade` connects it to every memory that
    *mentions* marmalade, which is what a separate tag vocabulary could not do.
    """
    chat = build(tmp_path, sessions)
    mentions_it = await store(
        chat, tmp_path, sessions, "marmalade came up in the meeting about ledgers"
    )
    # The concept, as extraction would have created it, on the earlier memory.
    entity_id = await mention(sessions, mentions_it, "marmalade")

    tagged = await store(chat, tmp_path, sessions, "a thought with no shared words")
    async with sessions() as session:
        item = await deletion.item_of_memory(session, tagged)

    report = await tags_module.apply(
        sessions,
        source_id=item.source_id,
        external_key=item.external_key,
        tags=tags_module.parse_required("#marmalade"),
    )
    # It joined the existing concept rather than creating a second one, which is the
    # measurable form of "one vocabulary".
    assert report.applied and report.entities_created == 0

    async with sessions() as session:
        assert (
            await session.execute(
                select(func.count(models.Entity.id)).where(
                    models.Entity.canonical_name == "marmalade"
                )
            )
        ).scalar_one() == 1

    # And the connection line names it: the tagged memory now reaches the earlier
    # one through a word it does not contain.
    from memoryos.application import chat as chat_use_case

    status = await chat_use_case.status(sessions, tagged)
    assert status is not None
    assert entity_id in {connection.entity_id for connection in status.connections}
    assert status.connected_memories >= 1


async def test_a_tag_filters_search_conjunctively(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Two tags narrow, and an untagged memory is excluded.

    Conjunctive because a second filter is somebody narrowing — returning the union
    would return more than either tag alone, which is the opposite of what adding a
    filter is for.
    """
    chat = build(tmp_path, sessions)
    both = await store(chat, tmp_path, sessions, "postgres and the queue, together")
    one = await store(chat, tmp_path, sessions, "postgres on its own")
    await store(chat, tmp_path, sessions, "postgres with no tags at all")

    for memory_id, spec in ((both, "#queue #idea"), (one, "#queue")):
        async with sessions() as session:
            item = await deletion.item_of_memory(session, memory_id)
        await tags_module.apply(
            sessions,
            source_id=item.source_id,
            external_key=item.external_key,
            tags=tags_module.parse_required(spec),
        )

    search = searcher(tmp_path, sessions)
    queue = await search(
        "postgres", k=10, mode=SearchMode.HYBRID, filters=SearchFilters(tags=["queue"])
    )
    assert {hit.memory_id for hit in queue.hits} == {both, one}

    narrowed = await search(
        "postgres",
        k=10,
        mode=SearchMode.HYBRID,
        filters=SearchFilters(tags=["queue", "idea"]),
    )
    assert {hit.memory_id for hit in narrowed.hits} == {both}


async def test_a_tag_survives_a_correction(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Keyed on the item, not the version, and this is why it matters.

    A tag keyed on `memory_id` would silently stop applying the moment somebody
    fixed a typo — the correction mints a new version, and the tag would be left
    pointing at the superseded one.
    """
    chat = build(tmp_path, sessions)
    memory_id = await store(chat, tmp_path, sessions, "a thought worth filing")
    async with sessions() as session:
        item = await deletion.item_of_memory(session, memory_id)
        message_id = (
            await session.execute(
                select(models.ChatMessage.id).where(
                    models.ChatMessage.external_key == item.external_key
                )
            )
        ).scalar_one()

    await tags_module.apply(
        sessions,
        source_id=item.source_id,
        external_key=item.external_key,
        tags=tags_module.parse_required("#idea"),
    )
    exchange = await chat.correct(message_id, "a thought worth filing, corrected")
    await drain(tmp_path, sessions)

    search = searcher(tmp_path, sessions)
    found = await search(
        "a thought worth filing",
        k=10,
        mode=SearchMode.HYBRID,
        filters=SearchFilters(tags=["idea"]),
    )
    assert {hit.memory_id for hit in found.hits} == {exchange.user.memory_id}
