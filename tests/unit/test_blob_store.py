"""The blob store. No database, so this lives in the fast tier."""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from memoryos.adapters.blobs.filesystem import BlobNotFound, FilesystemBlobStore
from memoryos.domain.values import ContentHash

HELLO = b"hello blobs"
DIGEST = ContentHash.of(HELLO)


@pytest.fixture
def store(tmp_path: Path) -> FilesystemBlobStore:
    return FilesystemBlobStore(tmp_path / "blobs")


async def test_put_then_get_round_trips(store: FilesystemBlobStore) -> None:
    await store.put(DIGEST, HELLO)
    assert await store.get(DIGEST) == HELLO


async def test_exists_reflects_what_was_written(store: FilesystemBlobStore) -> None:
    assert await store.exists(DIGEST) is False
    await store.put(DIGEST, HELLO)
    assert await store.exists(DIGEST) is True


async def test_get_raises_for_an_unknown_hash(store: FilesystemBlobStore) -> None:
    with pytest.raises(BlobNotFound):
        await store.get(DIGEST)


async def test_put_twice_is_a_no_op(store: FilesystemBlobStore) -> None:
    # Ingestion retries. Writing the same hash again must be free and must not
    # disturb what is already there.
    await store.put(DIGEST, HELLO)
    first_mtime = store.path_for(DIGEST).stat().st_mtime_ns

    await store.put(DIGEST, HELLO)

    assert store.path_for(DIGEST).stat().st_mtime_ns == first_mtime
    assert await store.get(DIGEST) == HELLO


async def test_the_path_fans_out_on_the_hash_prefix(store: FilesystemBlobStore) -> None:
    # One directory holding hundreds of thousands of entries degrades readdir
    # badly and is impossible to inspect by hand. Two levels of two hex
    # characters gives 65,536 leaves.
    path = store.path_for(DIGEST)
    digest = DIGEST.value

    assert path.parent.name == digest[2:4]
    assert path.parent.parent.name == digest[0:2]
    assert path.name == digest
    assert path.relative_to(store.root).as_posix() == f"{digest[0:2]}/{digest[2:4]}/{digest}"


async def test_delete_removes_the_blob(store: FilesystemBlobStore) -> None:
    await store.put(DIGEST, HELLO)
    await store.delete(DIGEST)
    assert await store.exists(DIGEST) is False


async def test_delete_is_forgiving_of_a_missing_blob(store: FilesystemBlobStore) -> None:
    await store.delete(DIGEST)


async def test_a_crash_before_the_replace_leaves_nothing_at_the_final_path(
    store: FilesystemBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason writes go through a temp file and os.replace.

    The path names a hash, so a truncated file sitting there would be a lie the
    system would go on believing — every later read would return bytes that do
    not hash to the name they are stored under. os.replace is atomic within a
    filesystem, so the only two possible outcomes are "no file" and "the whole
    file".
    """

    def explode(src: object, dst: object) -> None:
        raise OSError("crash between write and replace")

    monkeypatch.setattr(os, "replace", explode)

    with pytest.raises(OSError, match="crash between write and replace"):
        await store.put(DIGEST, HELLO)

    assert await store.exists(DIGEST) is False
    assert store.path_for(DIGEST).exists() is False


async def test_a_failed_write_leaves_no_temp_file_behind(
    store: FilesystemBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(src: object, dst: object) -> None:
        raise OSError("crash")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError, match="crash"):
        await store.put(DIGEST, HELLO)

    leftovers = list(store.path_for(DIGEST).parent.glob(".tmp-*"))
    assert leftovers == []


async def test_distinct_content_lands_in_distinct_paths(
    store: FilesystemBlobStore,
) -> None:
    other = ContentHash.of(b"different bytes")
    await store.put(DIGEST, HELLO)
    await store.put(other, b"different bytes")

    assert store.path_for(DIGEST) != store.path_for(other)
    assert await store.get(DIGEST) == HELLO
    assert await store.get(other) == b"different bytes"


# --------------------------------------------------------------------------
# Streaming writes
# --------------------------------------------------------------------------


async def chunks_of(data: bytes, size: int = 4) -> AsyncIterator[bytes]:
    for start in range(0, len(data), size):
        yield data[start : start + size]


async def test_put_stream_returns_the_hash_and_size(store: FilesystemBlobStore) -> None:
    content_hash, byte_size = await store.put_stream(chunks_of(HELLO))

    # The same digest the one-shot path would have produced: the hash is a
    # function of the bytes, not of how they arrived.
    assert content_hash == DIGEST
    assert byte_size == len(HELLO)
    assert await store.get(content_hash) == HELLO


async def test_put_stream_handles_an_empty_stream(store: FilesystemBlobStore) -> None:
    content_hash, byte_size = await store.put_stream(chunks_of(b""))

    assert byte_size == 0
    assert content_hash == ContentHash.of(b"")
    assert await store.get(content_hash) == b""


async def test_put_stream_is_idempotent(store: FilesystemBlobStore) -> None:
    first_hash, _ = await store.put_stream(chunks_of(HELLO))
    mtime = store.path_for(first_hash).stat().st_mtime_ns

    second_hash, _ = await store.put_stream(chunks_of(HELLO))

    assert second_hash == first_hash
    # Already stored, so the streamed copy was discarded rather than replacing
    # a file that already holds exactly these bytes.
    assert store.path_for(first_hash).stat().st_mtime_ns == mtime


async def test_put_stream_leaves_no_staging_files_behind(
    store: FilesystemBlobStore,
) -> None:
    await store.put_stream(chunks_of(HELLO))
    await store.put_stream(chunks_of(HELLO))
    await store.put_stream(chunks_of(b"other content"))

    staging = store.root / "incoming"
    assert list(staging.iterdir()) == []


async def test_a_failure_mid_stream_stores_nothing(store: FilesystemBlobStore) -> None:
    async def explodes() -> AsyncIterator[bytes]:
        yield b"partial"
        raise OSError("disk went away")

    with pytest.raises(OSError, match="disk went away"):
        await store.put_stream(explodes())

    # Nothing was committed, because the name is only known once every byte has
    # been seen — a partial stream never produces a path to write to.
    assert list((store.root / "incoming").iterdir()) == []
    assert list(store.root.rglob("*")) == [store.root / "incoming"]


async def test_chunk_boundaries_do_not_affect_the_hash(
    store: FilesystemBlobStore,
) -> None:
    data = b"a" * 1000
    first, _ = await store.put_stream(chunks_of(data, size=7))
    second, _ = await store.put_stream(chunks_of(data, size=999))

    assert first == second == ContentHash.of(data)
