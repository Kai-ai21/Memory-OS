"""Files dropped into the chat, and the four things that must not go wrong.

Content addressing has to hold — the same file twice is one artifact and two
memories — the door has to reject readably, a document the parser refuses has to
*say so to the person*, and the whole write has to be one transaction.

The third is the one worth the most. A scanned PDF is a file somebody handed over
in good faith; the parser is right to refuse it, and the only outcome worse than
refusing is refusing quietly, because then they believe it was filed.
"""

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.db import models
from memoryos.application import chat as chat_use_case
from memoryos.application.attachments import (
    MAX_FILE_BYTES,
    EmptyFile,
    FileTooLarge,
    UnsupportedMediaType,
    Upload,
)
from memoryos.application.chat import Stage
from memoryos.application.ingest import ingest_item
from memoryos.application.ports import ObservedItem
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobStatus, JobType
from memoryos.domain.values import ContentHash, SourceKind, TimeProvenance
from tests.integration.test_chat import build, drain

pytestmark = pytest.mark.integration

NOTES = b"# Vendor proposal\n\nThe pricing on page four is the part that matters.\n"


def upload(name: str, data: bytes, media_type: str | None = "text/markdown") -> Upload:
    """One file, streamed in small pieces.

    Deliberately chunked rather than yielded whole, so the test exercises the
    multi-chunk path through `put_stream` — a single-yield stream would pass
    against an implementation that only ever handled one.
    """

    async def stream() -> AsyncIterator[bytes]:
        for start in range(0, len(data), 16):
            yield data[start : start + 16]

    return Upload(filename=name, media_type=media_type, stream=stream())


async def test_the_same_file_twice_is_one_artifact_and_two_memories(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Content addressing, unchanged, and said out loud.

    Identity has been a function of content since M1.1, so this needs no new
    mechanism — what M10.2 adds is *telling somebody*. A silent success looks
    identical to a re-upload that did nothing, and `deduplicated` is the field that
    distinguishes them. It is recorded rather than derived because it stops being
    derivable the moment a second upload exists: from then on the artifact looks
    like it was always there.

    Two memories rather than one, and that is the right answer. The same document
    handed over twice is two acts of handing it over, in two conversations, perhaps
    with two different notes — and collapsing them would mean the second note
    pointed at a memory belonging to the first.
    """
    chat = build(tmp_path, sessions)

    first = await chat.attach([upload("proposal.md", NOTES)])
    second = await chat.attach([upload("proposal.md", NOTES)])

    assert first.user.attachments[0].deduplicated is False
    assert second.user.attachments[0].deduplicated is True, (
        "the second upload found the bytes already in the corpus"
    )

    async with sessions() as session:
        artifacts = (
            await session.execute(
                select(func.count(models.RawArtifact.content_hash)).where(
                    models.RawArtifact.content_hash
                    == first.user.attachments[0].content_hash
                )
            )
        ).scalar_one()
        memories = (
            await session.execute(
                select(func.count(models.Memory.id))
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Source.kind == SourceKind.UPLOAD.value)
            )
        ).scalar_one()

    assert artifacts == 1, "one set of bytes"
    assert memories == 2, "two documents pointing at it"
    # And two distinct external keys, because two files can share a name — a
    # name-derived key would have made the second upload a new *version* of the
    # first, which is a claim about an edit that did not happen.
    assert (
        first.user.attachments[0].external_key != second.user.attachments[0].external_key
    )
    assert first.user.attachments[0].memory_id != second.user.attachments[0].memory_id


async def test_an_unsupported_type_is_rejected_naming_what_is_supported(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The rejection has to be actionable, which means listing the alternatives.

    A caller is usually one extension away. The same argument `UnknownSource`
    makes for listing the source names: a generic refusal makes somebody guess,
    and guessing at a media-type allow-list is guessing at a list they cannot see.
    """
    chat = build(tmp_path, sessions)

    with pytest.raises(UnsupportedMediaType) as raised:
        await chat.attach([upload("holiday.heic", b"\x00\x01\x02", "image/heic")])

    message = str(raised.value)
    assert "holiday.heic" in message
    assert "image/heic" in message
    # Names the formats rather than saying "unsupported type".
    assert ".pdf" in message
    assert ".md" in message

    # And nothing was written. Validation happens before the transaction opens,
    # which is what makes a rejected upload cost a discarded temp file and no rows.
    async with sessions() as session:
        assert (
            await session.execute(select(func.count(models.ChatAttachment.id)))
        ).scalar_one() == 0
        assert (
            await session.execute(select(func.count(models.ChatMessage.id)))
        ).scalar_one() == 0


async def test_a_scanned_pdf_surfaces_a_readable_failure(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The one that matters most: a refusal the person actually sees.

    A PDF with no text layer is a real document that this system genuinely cannot
    read, and M1.4's parser correctly raises `PermanentError` rather than storing an
    empty memory. The failure mode M10.2 exists to prevent is that error living
    only in `jobs.last_error` — a dead-lettered attachment nobody hears about is
    the worst outcome, because they believe it worked.

    So the assertion is on what `status` reports, in the parser's own words, and on
    the stage being `failed` rather than a `parsing` that never ends.
    """
    scanned = _scanned_pdf()
    chat = build(tmp_path, sessions)

    exchange = await chat.attach(
        [upload("scan.pdf", scanned, "application/pdf")],
        note="this is the vendor's proposal",
    )
    memory_id = exchange.user.attachments[0].memory_id
    assert memory_id is not None

    # Before the worker: parsing, not failed and not indexed.
    waiting = await chat_use_case.status(sessions, memory_id)
    assert waiting is not None
    assert waiting.stage is Stage.PARSING
    assert waiting.failure is None

    # The worker tries and gives up. Marked failed the way the real worker marks
    # it — `PermanentError` goes straight to the dead letter without burning
    # retries, which is `handlers.py`'s decision and not this test's.
    await _fail_normalization(sessions, tmp_path, memory_id)

    failed = await chat_use_case.status(sessions, memory_id)
    assert failed is not None
    assert failed.stage is Stage.FAILED
    assert failed.failure is not None
    # The parser's sentence, verbatim. It names the file, counts the characters it
    # found, counts the pages it looked at, and says what the file probably *is* —
    # which is the difference between a reader knowing to run OCR and a reader
    # opening a support ticket. Nothing in this layer paraphrases it.
    reason = failed.failure.lower()
    assert "yielded 0 characters" in reason
    assert "scanned" in reason
    assert "ocr" in reason
    assert "scan.pdf" in reason

    # The note is unaffected and is its own memory. Context about a document
    # frequently outlives the document, and a file this system cannot read is
    # exactly the case where that matters most.
    assert exchange.user.memory_id is not None
    note = await chat_use_case.status(sessions, exchange.user.memory_id)
    assert note is not None
    assert note.stage is not Stage.FAILED


async def test_the_upload_and_its_normalization_job_commit_together(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """A rollback leaves no memory, no job, no turn and no attachment row.

    The queue is a table precisely so a memory cannot exist without the job that
    processes it, and an attachment row joins that guarantee rather than sitting
    beside it: a turn claiming to have attached a document whose memory rolled back
    would be a file the interface says it kept and search cannot find.

    Driven through the same `ingest_item` the attach path uses, with a failure
    injected after every write — going through `attach` and contriving a database
    error would test the same boundary less legibly.
    """
    chat = build(tmp_path, sessions)
    first = await chat.attach([upload("kept.md", NOTES)])
    before = await _counts(sessions)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    async with sessions() as session:
        source = (
            await session.execute(
                select(models.Source).where(
                    models.Source.kind == SourceKind.UPLOAD.value
                )
            )
        ).scalars().one()

    from memoryos.application.attachments import external_key_for

    doomed = b"# Lost\n\nThis document does not survive the transaction.\n"
    with pytest.raises(DeliberateFailure):
        async with sessions.begin() as session:
            recorded = await ingest_item(
                session,
                blobs,
                Source(
                    id=source.id,
                    kind=SourceKind.UPLOAD,
                    name=source.name,
                    config=source.config,
                ),
                _item(
                    external_key_for("lost.md", first.user.created_at, new_id()),
                    doomed,
                ),
            )
            assert recorded is not None
            message = models.ChatMessage(
                id=new_id(),
                session_id=first.session_id,
                role="user",
                content="lost.md",
                ordinal=99,
                intent="statement",
            )
            session.add(message)
            # Flushed first, for the reason `Chat.attach` flushes: the attachment
            # holds a foreign key to this row and the unit of work would otherwise
            # insert the child before its parent.
            await session.flush()
            session.add(
                models.ChatAttachment(
                    id=new_id(),
                    message_id=message.id,
                    ordinal=0,
                    filename="lost.md",
                    external_key="never/lost.md",
                    content_hash=ContentHash.of(doomed).value,
                    byte_size=len(doomed),
                    media_type="text/markdown",
                )
            )
            await session.flush()
            raise DeliberateFailure("after the memory, the job, the turn and the row")

    assert await _counts(sessions) == before, (
        "the memory, its normalization job, the turn and the attachment row went "
        "together"
    )


async def test_a_note_and_several_files_are_one_turn_and_several_memories(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """One message, three memories, and the note is one of them.

    The note being its own memory is the interesting half. "This is the vendor's
    proposal" is a thought somebody had rather than a document somebody sent, and
    storing it inside the file's memory would make it unfindable except by reading
    the file — which is exactly backwards, because the note is often the more
    valuable of the two.
    """
    chat = build(tmp_path, sessions)

    exchange = await chat.attach(
        [upload("a.md", b"# A\n\nfirst\n"), upload("b.md", b"# B\n\nsecond\n")],
        note="this is the vendor's proposal and its addendum",
    )
    await drain(tmp_path, sessions)

    assert len(exchange.user.attachments) == 2
    assert exchange.user.memory_id is not None, "the note is its own memory"

    # One turn in the transcript, carrying both files.
    drawn = await chat_use_case.messages(sessions, exchange.session_id)
    assert len(drawn) == 1
    assert [item.filename for item in drawn[0].attachments] == ["a.md", "b.md"]
    assert all(item.memory_id is not None for item in drawn[0].attachments)

    # Three memories: the note and the two files, every one of them searchable on
    # its own merits.
    for memory_id in [
        exchange.user.memory_id,
        *(item.memory_id for item in drawn[0].attachments),
    ]:
        assert memory_id is not None
        found = await chat_use_case.status(sessions, memory_id)
        assert found is not None
        assert found.stage is Stage.INDEXED, memory_id


async def test_the_ceiling_is_enforced_while_streaming(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """Mid-stream, not from `Content-Length`, which a client states.

    A 5GB body must stop costing disk at 50MB rather than after all of it has been
    written, so the bound is a generator between the socket and the blob store
    rather than a check afterwards.
    """
    chat = build(tmp_path, sessions)

    async def torrent() -> AsyncIterator[bytes]:
        block = b"x" * (1024 * 1024)
        for _ in range(MAX_FILE_BYTES // len(block) + 2):
            yield block

    with pytest.raises(FileTooLarge) as raised:
        await chat.attach(
            [Upload(filename="huge.txt", media_type="text/plain", stream=torrent())]
        )
    # Names the alternative that has no ceiling, rather than only the limit.
    assert "source" in str(raised.value)

    with pytest.raises(EmptyFile):
        await chat.attach([upload("nothing.md", b"")])


class DeliberateFailure(RuntimeError):
    """Something going wrong after every row was written and before it committed."""


def _item(key: str, data: bytes) -> "ObservedItem":
    async def read() -> bytes:
        return data

    return ObservedItem(
        external_key=key,
        content_hash=ContentHash.of(data),
        byte_size=len(data),
        media_type="text/markdown",
        occurred_at=None,
        occurred_at_source=TimeProvenance.UNKNOWN,
        read_bytes=read,
    )


async def _counts(sessions: async_sessionmaker[AsyncSession]) -> tuple[int, ...]:
    """Memories, normalization jobs, turns, attachment rows.

    Four numbers rather than one, because the invariant is that they move
    together: a rollback that left any of them behind is the failure.
    """
    statements = (
        select(func.count(models.Memory.id)),
        select(func.count(models.Job.id)).where(
            models.Job.job_type == JobType.NORMALIZE_MEMORY.value
        ),
        select(func.count(models.ChatMessage.id)),
        select(func.count(models.ChatAttachment.id)),
    )
    counted: list[int] = []
    async with sessions() as session:
        for statement in statements:
            counted.append(int((await session.execute(statement)).scalar_one()))
    return tuple(counted)


async def _fail_normalization(
    sessions: async_sessionmaker[AsyncSession], tmp_path: Path, memory_id: UUID
) -> None:
    """Run the real parser, and record its refusal the way the worker would.

    The error text is the parser's own — this does not invent a message — because
    the sentence a reader sees has to be the one the code that refused the file
    wrote. A test that supplied its own would pass while the real failure said
    something useless.
    """
    from memoryos.adapters.chunking.structural import StructuralChunker
    from memoryos.adapters.parsers.registry import build_default_registry
    from memoryos.application.normalize import NormalizeMemory
    from memoryos.domain.jobs import PermanentError
    from tests.support.fakes import FakeEmbedder

    normalize = NormalizeMemory(
        sessions,
        FilesystemBlobStore(tmp_path / "blobs"),
        build_default_registry(),
        StructuralChunker(FakeEmbedder()),
    )
    try:
        await normalize(memory_id)
    except PermanentError as exc:
        reason = str(exc)
    else:  # pragma: no cover - the fixture PDF has no text layer
        raise AssertionError("the scanned PDF was parsed, which it should not be")

    async with sessions.begin() as session:
        job = (
            await session.execute(
                select(models.Job).where(
                    models.Job.job_type == JobType.NORMALIZE_MEMORY.value,
                    models.Job.payload["memory_id"].astext == str(memory_id),
                )
            )
        ).scalars().one()
        job.status = JobStatus.FAILED.value
        job.last_error = reason


def _scanned_pdf() -> bytes:
    """A one-page PDF containing an image and no text layer.

    Hand-assembled rather than generated by a library, because the property under
    test is precisely "no text objects" and a generator would have to be trusted
    not to add any. This is a minimal valid PDF: one page, one XObject drawn onto
    it, and nothing a text extractor can find.
    """
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length 44 >>\nstream\nq 200 0 0 200 0 0 cm /Im0 Do Q\nendstream",
        b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace "
        b"/DeviceGray /BitsPerComponent 8 /Length 1 >>\nstream\n\xff\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{start}\n%%EOF\n"
    ).encode()
    return bytes(out)
