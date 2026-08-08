"""Content-addressed blob storage on a local filesystem."""

import asyncio
import hashlib
import os
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import structlog

from memoryos.application.ports import BlobStore
from memoryos.domain.values import ContentHash

logger = structlog.get_logger(__name__)

# Two levels of two hex characters: 256 top-level directories, 65,536 leaves.
# Without the fan-out, a mature store is a single directory holding hundreds of
# thousands of entries, which degrades `readdir` badly on most filesystems and
# makes the directory impossible to inspect by hand when something is wrong.
_FANOUT_WIDTH = 2
_FANOUT_DEPTH = 2

# BLAKE2b truncated to 256 bits, matching ContentHash.of.
_DIGEST_SIZE = 32

# Staging directory for streamed writes, inside the blob root so that the final
# os.replace stays within one filesystem and therefore stays atomic.
_STAGING = "incoming"


class BlobNotFound(KeyError):
    """No bytes are stored under that hash."""


class FilesystemBlobStore(BlobStore):
    """Blobs under `<root>/ab/cd/abcd...`.

    Every method is blocking work pushed onto a thread. File I/O does not
    release the event loop on its own, and a sync reading thousands of files
    would otherwise stall every other coroutine in the process.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, content_hash: ContentHash) -> Path:
        digest = content_hash.value
        parts = [
            digest[i * _FANOUT_WIDTH : (i + 1) * _FANOUT_WIDTH] for i in range(_FANOUT_DEPTH)
        ]
        return self._root.joinpath(*parts, digest)

    async def put(self, content_hash: ContentHash, data: bytes) -> None:
        await asyncio.to_thread(self._put_sync, content_hash, data)

    async def put_stream(self, chunks: AsyncIterator[bytes]) -> tuple[ContentHash, int]:
        """Hash and store in a single pass over the source.

        The destination path is a function of the content, so it is not known
        until the last byte has gone by. The bytes land in a temp file inside
        the blob root while the digest accumulates; only then is the final name
        known and the file moved into place — the same atomic `os.replace` that
        `put` uses, for the same reason.

        If the hash turns out to be one already stored, the temp file is
        discarded. That is the cost of this approach: unchanged files churn a
        little disk. It buys back an entire read of every changed file, and it
        leaves the bytes local for whatever needs them next.
        """
        digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
        size = 0

        handle, temp_name = await asyncio.to_thread(self._open_temp)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                async for chunk in chunks:
                    digest.update(chunk)
                    size += len(chunk)
                    await asyncio.to_thread(stream.write, chunk)
                await asyncio.to_thread(_flush_and_sync, stream)

            content_hash = ContentHash(digest.hexdigest())
            await asyncio.to_thread(self._commit_temp, temp_path, content_hash)
            return content_hash, size
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    async def get(self, content_hash: ContentHash) -> bytes:
        return await asyncio.to_thread(self._get_sync, content_hash)

    async def exists(self, content_hash: ContentHash) -> bool:
        return await asyncio.to_thread(self.path_for(content_hash).is_file)

    async def delete(self, content_hash: ContentHash) -> None:
        await asyncio.to_thread(self._delete_sync, content_hash)

    def _open_temp(self) -> tuple[int, str]:
        staging = self._root / _STAGING
        staging.mkdir(parents=True, exist_ok=True)
        return tempfile.mkstemp(dir=staging, prefix="stream-")

    def _commit_temp(self, temp_path: Path, content_hash: ContentHash) -> None:
        target = self.path_for(content_hash)
        if target.is_file():
            # Already stored. Same hash means same bytes, so there is nothing
            # to gain from rewriting it.
            temp_path.unlink(missing_ok=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_path, target)

    def _put_sync(self, content_hash: ContentHash, data: bytes) -> None:
        """Write atomically, or not at all.

        The bytes go to a temp file in the destination directory, are fsynced,
        and are then moved into place with `os.replace`, which is atomic within
        a filesystem. A crash at any point leaves either no file or the
        complete file — never a truncated one sitting at a path that claims to
        be a particular hash. That claim is the whole basis of content
        addressing, so a partial file there would be a lie the system would go
        on believing.
        """
        target = self.path_for(content_hash)
        if target.is_file():
            # Same hash, same bytes. Rewriting buys nothing.
            return

        target.parent.mkdir(parents=True, exist_ok=True)

        handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        temp_path = Path(temp_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, target)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    def _get_sync(self, content_hash: ContentHash) -> bytes:
        try:
            return self.path_for(content_hash).read_bytes()
        except FileNotFoundError as exc:
            raise BlobNotFound(content_hash.value) from exc

    def _delete_sync(self, content_hash: ContentHash) -> None:
        self.path_for(content_hash).unlink(missing_ok=True)


def _flush_and_sync(stream: object) -> None:
    stream.flush()  # type: ignore[attr-defined]
    os.fsync(stream.fileno())  # type: ignore[attr-defined]
