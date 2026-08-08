"""The filesystem connector: walk a directory, describe what is in it.

It walks, hashes, and yields. It does not parse, chunk, or embed — every source
has a different walking problem and an identical downstream pipeline, and
keeping that line sharp is what makes the second connector cheap.
"""

import asyncio
import fnmatch
import hashlib
import mimetypes
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import structlog

from memoryos.application.ports import Connector, ObservedItem
from memoryos.domain.entities import Source
from memoryos.domain.values import ContentHash, SourceKind, TimeProvenance

logger = structlog.get_logger(__name__)

# Files are hashed in chunks so that memory use is independent of file size. A
# 2GB file must not become a 2GB allocation just to learn its digest.
_HASH_CHUNK_BYTES = 64 * 1024

DEFAULT_INCLUDE = ["**/*.md", "**/*.txt", "**/*.py", "**/*.pdf"]
DEFAULT_EXCLUDE = [
    "**/.git/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/.DS_Store",
]
DEFAULT_MAX_FILE_BYTES = 50 * 1024 * 1024

# Cursor keys. The cursor is opaque to everything but this connector.
CURSOR_SEEN = "seen"


@dataclass(frozen=True, slots=True)
class FilesystemConfig:
    root: Path
    include: list[str] = field(default_factory=lambda: list(DEFAULT_INCLUDE))
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    follow_symlinks: bool = False

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "FilesystemConfig":
        raw_root = config.get("root")
        if not raw_root:
            raise ValueError("filesystem source config requires a 'root'")
        return cls(
            # Resolved once, here, so every later comparison is against a real
            # path with symlinks and `..` already collapsed.
            root=Path(raw_root).expanduser().resolve(),
            include=list(config.get("include") or DEFAULT_INCLUDE),
            exclude=list(config.get("exclude") or DEFAULT_EXCLUDE),
            max_file_bytes=int(config.get("max_file_bytes") or DEFAULT_MAX_FILE_BYTES),
            follow_symlinks=bool(config.get("follow_symlinks", False)),
        )


def matches(external_key: str, patterns: list[str]) -> bool:
    """Whether a relative key matches any glob.

    Each pattern is also tried without a leading `**/`, because `**/*.md` is
    the natural way to write "markdown anywhere" and `fnmatch` does not treat
    `**` specially — without this, a file at the root of the tree would not
    match its own include pattern.
    """
    for pattern in patterns:
        if fnmatch.fnmatch(external_key, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatch(external_key, pattern[3:]):
            return True
    return False


def to_external_key(path: Path, root: Path) -> str:
    """The POSIX-normalised path relative to root.

    Relative, because the identity of an item must survive the directory being
    moved or the same tree being synced from a different machine.
    """
    return PurePosixPath(path.relative_to(root)).as_posix()


def escapes_root(path: Path, root: Path) -> bool:
    """Whether a resolved path lies outside the source root.

    Checked after resolution rather than before, because that is the only point
    at which a symlink has stopped hiding where it actually goes.
    """
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return True
    return False


def hash_file(path: Path) -> tuple[ContentHash, int]:
    """Stream the file through BLAKE2b, returning the digest and byte size."""
    digest = hashlib.blake2b(digest_size=32)
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return ContentHash(digest.hexdigest()), size


def fingerprint(stat: os.stat_result) -> list[Any]:
    """The cheap change filter: (mtime_ns, size).

    Never the authority on whether content changed. Copying a file resets
    mtime, some editors preserve it, and sync tools rewrite it wholesale. This
    only decides which files are worth hashing; the hash decides what actually
    changed.
    """
    return [stat.st_mtime_ns, stat.st_size]


class FilesystemConnector(Connector):
    kind: SourceKind = SourceKind.FILESYSTEM

    async def observe(self, source: Source, *, full: bool) -> AsyncIterator[ObservedItem]:
        config = FilesystemConfig.from_dict(source.config)
        seen: dict[str, list[Any]] = dict(source.cursor.get(CURSOR_SEEN) or {})

        log = logger.bind(source=source.name, root=str(config.root), full=full)
        if not config.root.is_dir():
            raise FileNotFoundError(f"source root is not a directory: {config.root}")

        for path, stat in await asyncio.to_thread(self._collect, config, log):
            external_key = to_external_key(path, config.root)

            if not full and seen.get(external_key) == fingerprint(stat):
                # Unchanged by the cheap filter, so not worth reading. A full
                # sync ignores the filter entirely: it has to look at
                # everything to be able to tell what has gone missing.
                continue

            try:
                content_hash, byte_size = await asyncio.to_thread(hash_file, path)
            except OSError as exc:
                log.warning("file.skipped", key=external_key, reason=str(exc))
                continue

            yield ObservedItem(
                external_key=external_key,
                content_hash=content_hash,
                byte_size=byte_size,
                media_type=mimetypes.guess_type(path.name)[0],
                # mtime is weak evidence of when something happened, but it is
                # evidence. Recorded honestly with its provenance rather than
                # defaulted to now(), which would be a fabrication.
                occurred_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                occurred_at_source=TimeProvenance.FILESYSTEM,
                read_bytes=_reader(path),
                fingerprint=fingerprint(stat),
            )

    def _collect(
        self, config: FilesystemConfig, log: structlog.BoundLogger
    ) -> list[tuple[Path, os.stat_result]]:
        """Walk the tree once, returning the files worth considering.

        Runs on a thread: `os.walk` on a large tree is a long blocking call.
        """
        return list(self._walk(config, log))

    def _walk(
        self, config: FilesystemConfig, log: structlog.BoundLogger
    ) -> Iterator[tuple[Path, os.stat_result]]:
        root = config.root

        for dirpath, dirnames, filenames in os.walk(
            root, followlinks=config.follow_symlinks
        ):
            current = Path(dirpath)

            # Prune excluded directories in place, so os.walk never descends
            # into them. Filtering after the fact would still pay the cost of
            # walking .git and node_modules.
            dirnames[:] = [
                name
                for name in dirnames
                if not self._excluded_dir(current / name, root, config)
            ]

            for name in sorted(filenames):
                path = current / name
                external_key = to_external_key(path, root)

                if matches(external_key, config.exclude):
                    continue
                if not matches(external_key, config.include):
                    continue
                # A symlink out of the tree is the case that turns a directory
                # sync into a whole-disk sync, so the check is on the resolved
                # path and applies whether or not following is enabled.
                if escapes_root(path, root):
                    log.warning("file.skipped", key=external_key, reason="escapes root")
                    continue

                try:
                    stat = path.stat()
                except OSError as exc:
                    # Broken symlink, permission denied, a file that vanished
                    # mid-walk. One bad file must not abort a sync of ten
                    # thousand good ones.
                    log.warning("file.skipped", key=external_key, reason=str(exc))
                    continue

                if not path.is_file():
                    continue
                if stat.st_size > config.max_file_bytes:
                    log.warning(
                        "file.skipped",
                        key=external_key,
                        reason="over max_file_bytes",
                        byte_size=stat.st_size,
                        limit=config.max_file_bytes,
                    )
                    continue

                yield path, stat

    def _excluded_dir(self, path: Path, root: Path, config: FilesystemConfig) -> bool:
        key = to_external_key(path, root)
        # Directory patterns are written as `**/.git/**`, which describes the
        # contents rather than the directory itself, so both are tried.
        return matches(key, config.exclude) or matches(f"{key}/", config.exclude)


def _reader(path: Path) -> Callable[[], Awaitable[bytes]]:
    async def read() -> bytes:
        return await asyncio.to_thread(path.read_bytes)

    return read
