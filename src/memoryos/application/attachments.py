"""A file dropped into the chat, recorded the way a walked file is.

**One pass over the bytes.** `BlobStore.put_stream` hashes while it writes, which
is the pattern M1.4 built it for and the reason nothing here ever holds a file in
memory: a 50MB PDF read into a `bytes` before hashing costs 50MB of the API
process per concurrent upload, and the second pass to store it costs the read
again. The hash is not known until the last byte has gone by, which is why the
destination name cannot be chosen first — see the adapter.

**Content addressing applies unchanged, and the response says so.** The same PDF
uploaded twice stores its bytes once and produces two memories pointing at one
artifact, because identity is a function of content and always has been. What is
new is telling the person: a silent success looks identical to a re-upload that
did nothing, and "already in memory, linked" is the sentence that distinguishes
them. `deduplicated` is recorded per attachment rather than derived, because it
stops being derivable the moment a second upload of the same file exists.

**The allow-list is composed from the parsers, not restated.** Every suffix and
media type here comes from the module that handles it, so a file cannot be
accepted at the door and rejected by the pipeline, or the reverse. `TextParser`
accepts everything — it is the registry's catch-all — so the door reads its
*named* vocabulary instead: "anything at all" is not a policy a door can have.

Nothing downstream of `ingest_item` knows an upload happened. Same artifact table,
same event log, same normalization job, same chunker, same embedder.
"""

import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import structlog
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import (
    SqlAlchemyArtifactRepository,
    SqlAlchemySourceRepository,
)
from memoryos.adapters.parsers import code, markdown, pdf, text
from memoryos.application.ingest import ingest_item
from memoryos.application.ports import BlobStore, ObservedItem
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash, SourceKind, TimeProvenance

logger = structlog.get_logger(__name__)

# The one upload source, by name, matching how the chat source is a singleton.
UPLOAD_SOURCE_NAME = "uploads"

# 50MB. A ceiling rather than a measurement: the pipeline has no trouble with a
# larger file, and the reason to stop somewhere is that an unbounded multipart
# body is an unbounded write to somebody's disk. Enforced while streaming rather
# than from `Content-Length`, which a client states and can be wrong about.
MAX_FILE_BYTES = 50 * 1024 * 1024

# Read size. 64KiB is large enough that the syscall overhead disappears and small
# enough that ten concurrent uploads are still under a megabyte of buffers.
CHUNK_BYTES = 64 * 1024

# What may be dropped, composed from the parsers that claim it. One definition
# per format, read here and by `ParserRegistry.for_item`.
ALLOWED_SUFFIXES: tuple[str, ...] = (
    *markdown.SUFFIXES,
    *code.SUFFIXES,
    *pdf.SUFFIXES,
    *text.SUFFIXES,
)
ALLOWED_MEDIA_TYPES: frozenset[str] = frozenset(
    {*markdown.MEDIA_TYPES, *pdf.MEDIA_TYPES, *text.MEDIA_TYPES}
)

# Filenames are kept verbatim for display and sanitised for identity. Anything
# that is not a word character, a dot or a dash becomes an underscore, and path
# separators go with it: the external key is not a path, but it is read by people
# and printed in citations, and a key containing `../` is a key that looks like an
# instruction.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class UnsupportedMediaType(ValueError):
    """A file no parser handles.

    Carries the supported list, because the caller is usually one extension away
    and a generic rejection makes them guess. The same reasoning `UnknownSource`
    uses for listing the source names.
    """

    def __init__(self, filename: str, media_type: str | None) -> None:
        described = ", ".join(sorted(ALLOWED_SUFFIXES))
        super().__init__(
            f"{filename!r} ({media_type or 'unknown type'}) is not something this "
            f"system can read. Supported: {described}."
        )
        self.filename = filename
        self.media_type = media_type
        self.supported = ALLOWED_SUFFIXES


class FileTooLarge(ValueError):
    """More bytes than the ceiling allows, discovered while streaming."""

    def __init__(self, filename: str, limit: int = MAX_FILE_BYTES) -> None:
        super().__init__(
            f"{filename!r} is larger than the {limit // (1024 * 1024)}MB limit. "
            f"Point a source at the directory it lives in instead — that path "
            f"streams from disk and has no ceiling."
        )
        self.filename = filename
        self.limit = limit


class EmptyFile(ValueError):
    """Zero bytes. There is nothing here to parse, chunk or retrieve."""


@dataclass(frozen=True, slots=True)
class Upload:
    """One file, described without being read.

    `stream` is an async iterator rather than bytes, which is the whole point:
    nothing between the socket and the blob store materialises the file.
    """

    filename: str
    media_type: str | None
    stream: AsyncIterator[bytes]


@dataclass(frozen=True, slots=True)
class Stored:
    """What one upload became."""

    filename: str
    external_key: str
    content_hash: ContentHash
    byte_size: int
    media_type: str | None
    memory_id: UUID
    # True when these bytes were already in the corpus. The interesting field:
    # "already in memory, linked" is worth saying and a silent success is not.
    deduplicated: bool


def supported(filename: str, media_type: str | None) -> bool:
    """Whether any parser claims this file.

    Suffix *or* media type, because both are unreliable alone: browsers send
    `application/octet-stream` for a `.md` often enough that a media-type-only
    door would reject markdown, and a file with no extension can still be
    correctly typed by the client.
    """
    if filename.lower().endswith(ALLOWED_SUFFIXES):
        return True
    return (media_type or "").split(";")[0].strip().lower() in ALLOWED_MEDIA_TYPES


def external_key_for(filename: str, at: datetime, upload_id: UUID) -> str:
    """The durable identity of an uploaded file.

    `2026-08-17/proposal.pdf#0198…` — dated for the reader, named for
    recognition, and disambiguated by an id because **two files can share a
    name**. Deriving the key from the name alone would make this month's
    `invoice.pdf` a new version of last month's, which is the wrong claim: they
    are two documents, and a version chain implies an edit that did not happen.

    Not derived from the content hash either, and that is the opposite mistake:
    identical bytes uploaded twice are two uploads of one document, and a
    content-derived key would collapse them into a single memory whose external
    key belongs to whichever upload happened first.
    """
    safe = _UNSAFE.sub("_", filename.strip()) or "file"
    return f"{at:%Y-%m-%d}/{safe}#{upload_id}"


class StoreUpload:
    """Stream a file to the blob store, then record it in the caller's transaction.

    Two phases, and the split is forced by what each one is. Writing bytes is I/O
    that can take a minute and must not hold a database transaction open; writing
    the rows is a transaction that must include the normalization job. So the blob
    lands first and the rows follow — which is the same order `ingest_item`
    already documents for the sync path, and safe for the same reason: a blob with
    no artifact row is invisible garbage, while an artifact with no blob is a
    dangling promise the pipeline would trip over.
    """

    def __init__(
        self, session_factory: async_sessionmaker[AsyncSession], blob_store: BlobStore
    ) -> None:
        self._sessions = session_factory
        self._blobs = blob_store

    async def stage(self, upload: Upload) -> tuple[ContentHash, int]:
        """Stream the bytes in, hashing as they go. No transaction is held.

        Raises before writing anything for an unsupported type, and mid-stream for
        one that exceeds the ceiling — the second check has to be mid-stream
        because `Content-Length` is a claim the client makes and a truncated or
        lying one is exactly the case a limit exists for.
        """
        if not supported(upload.filename, upload.media_type):
            raise UnsupportedMediaType(upload.filename, upload.media_type)

        content_hash, byte_size = await self._blobs.put_stream(
            _bounded(upload.stream, upload.filename)
        )
        if byte_size == 0:
            raise EmptyFile(
                f"{upload.filename!r} is empty, so there is nothing to read from it"
            )
        logger.info(
            "upload.staged",
            filename=upload.filename,
            byte_size=byte_size,
            content_hash=content_hash.value,
        )
        return content_hash, byte_size

    async def record(
        self,
        session: AsyncSession,
        source: Source,
        upload: Upload,
        content_hash: ContentHash,
        byte_size: int,
        at: datetime,
    ) -> Stored:
        """Record one staged file in the caller's transaction.

        The caller owns the transaction because the message row and every
        attachment on it commit together with the memories and their normalization
        jobs — a turn that claims to have attached a file whose memory rolled back
        would be a document the interface says it kept and search cannot find.
        """
        known = await SqlAlchemyArtifactRepository(session).exists(content_hash)

        upload_id = new_id()
        key = external_key_for(upload.filename, at, upload_id)
        recorded = await ingest_item(
            session,
            self._blobs,
            source,
            ObservedItem(
                external_key=key,
                content_hash=content_hash,
                byte_size=byte_size,
                media_type=upload.media_type,
                # When it was handed over, which is the only honest thing this
                # system knows about a dropped file. The browser's `lastModified`
                # is a clock on somebody else's machine and is not carried here;
                # M10.0's chat messages are `DECLARED` because somebody pressed
                # enter, and so is this.
                occurred_at=at,
                occurred_at_source=TimeProvenance.DECLARED,
                # Omitted, which is what `None` means: `stage` streamed the bytes
                # into the blob store while hashing them, so there is nothing left
                # to read and no path to read it from. M10.2 passed a function
                # that raised; the field is optional as of M10.3 and this says the
                # true thing instead.
                fingerprint=None,
            ),
        )
        if recorded is None:
            raise UnstorableUpload(
                f"{upload.filename!r} produced no memory; its external key {key!r} "
                f"was already current, which cannot happen for a fresh upload id"
            )

        return Stored(
            filename=upload.filename,
            external_key=key,
            content_hash=content_hash,
            byte_size=byte_size,
            media_type=upload.media_type,
            memory_id=recorded.memory_id,
            deduplicated=known,
        )

    async def source(self) -> Source:
        """The upload source, created on first use.

        Not cached, for the reason the chat source is not: one indexed lookup on a
        unique index, against a cached id that outlives its row in exactly the
        situation where that hurts — a truncated test database, or a replay under a
        long-lived process.
        """
        async with self._sessions.begin() as session:
            await session.execute(
                pg_insert(models.Source)
                .values(
                    id=new_id(),
                    kind=SourceKind.UPLOAD.value,
                    name=UPLOAD_SOURCE_NAME,
                    # No root, no include globs, no cursor. There is nothing to
                    # walk and nothing to resume from, which is what a pushed
                    # source looks like — and is why `upload` is not `filesystem`.
                    config={},
                    cursor={},
                )
                .on_conflict_do_nothing(constraint="uq_sources_kind_name")
            )
            found = await SqlAlchemySourceRepository(session).get_by_name(
                SourceKind.UPLOAD, UPLOAD_SOURCE_NAME
            )
        assert found is not None  # just inserted, or already there
        return found


class UnstorableUpload(RuntimeError):
    """An upload the ingest path declined to record."""


async def _bounded(stream: AsyncIterator[bytes], filename: str) -> AsyncIterator[bytes]:
    """Pass bytes through, refusing past the ceiling.

    A generator rather than a check afterwards, so a 5GB body stops costing disk
    at 50MB rather than after it has all been written. The partial temp file is
    discarded by `put_stream`'s own cleanup when this raises.
    """
    seen = 0
    async for chunk in stream:
        seen += len(chunk)
        if seen > MAX_FILE_BYTES:
            raise FileTooLarge(filename)
        yield chunk


def attachment_rows(
    message_id: UUID, stored: Sequence[Stored]
) -> list[models.ChatAttachment]:
    """The attachment rows for one message, in the order the files arrived."""
    return [
        models.ChatAttachment(
            id=new_id(),
            message_id=message_id,
            ordinal=ordinal,
            filename=item.filename,
            external_key=item.external_key,
            content_hash=item.content_hash.value,
            byte_size=item.byte_size,
            media_type=item.media_type,
            deduplicated=item.deduplicated,
        )
        for ordinal, item in enumerate(stored)
    ]
