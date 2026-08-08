"""The connector's walking rules. Real directories under tmp_path, no database.

The filesystem is not mocked. We own neither `pathlib` nor the OS, and a mock
of either would be asserting that our beliefs about them are self-consistent
rather than that they are true.
"""

import os
from pathlib import Path

import pytest

from memoryos.adapters.connectors.filesystem import (
    DEFAULT_EXCLUDE,
    FilesystemConfig,
    FilesystemConnector,
    escapes_root,
    fingerprint,
    matches,
    to_external_key,
)
from memoryos.application.ports import ObservedItem
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import SourceKind, TimeProvenance


def make_source(root: Path, **config: object) -> Source:
    return Source(
        id=new_id(),
        kind=SourceKind.FILESYSTEM,
        name="fixture",
        config={"root": str(root), **config},
    )


async def observe_all(source: Source, *, full: bool = True) -> list[ObservedItem]:
    return [item async for item in FilesystemConnector().observe(source, full=full)]


# --------------------------------------------------------------------------
# Glob matching
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["notes.md", "a/notes.md", "a/b/c/notes.md"],
)
def test_include_matches_at_every_depth(key: str) -> None:
    # `**/*.md` is the natural way to write "markdown anywhere", including at
    # the root, where a literal reading of the pattern would not match.
    assert matches(key, ["**/*.md"]) is True


def test_include_does_not_match_a_different_suffix() -> None:
    assert matches("notes.rst", ["**/*.md", "**/*.txt"]) is False


@pytest.mark.parametrize(
    "key",
    [
        ".git/config",
        "a/.git/config",
        "a/b/.git/objects/ab/cdef",
        "node_modules/pkg/index.js",
        "a/node_modules/pkg/index.js",
        "src/__pycache__/mod.cpython-312.pyc",
        ".venv/lib/python3.12/site-packages/x.py",
    ],
)
def test_default_excludes_catch_nested_directories(key: str) -> None:
    assert matches(key, DEFAULT_EXCLUDE) is True


def test_ds_store_is_excluded_at_the_root_too() -> None:
    assert matches(".DS_Store", DEFAULT_EXCLUDE) is True
    assert matches("a/b/.DS_Store", DEFAULT_EXCLUDE) is True


def test_a_normal_file_is_not_excluded() -> None:
    assert matches("src/memoryos/cli.py", DEFAULT_EXCLUDE) is False


# --------------------------------------------------------------------------
# Key normalisation and root containment
# --------------------------------------------------------------------------


def test_external_key_is_relative_and_posix(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "notes.md"
    # Relative, because identity has to survive the directory being moved or
    # the same tree being synced from another machine.
    assert to_external_key(nested, tmp_path) == "a/b/notes.md"


def test_external_key_of_a_root_level_file(tmp_path: Path) -> None:
    assert to_external_key(tmp_path / "notes.md", tmp_path) == "notes.md"


def test_a_path_inside_root_does_not_escape(tmp_path: Path) -> None:
    inside = tmp_path / "a" / "notes.md"
    inside.parent.mkdir(parents=True)
    inside.write_text("x")
    assert escapes_root(inside, tmp_path) is False


def test_a_symlink_out_of_the_tree_escapes_after_resolution(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.md").write_text("not yours")

    link = root / "escape.md"
    link.symlink_to(outside / "secret.md")

    # Before resolution the path looks like it is inside root. Resolution is the
    # only point at which the symlink stops hiding where it actually goes.
    assert escapes_root(link, root) is True


def test_a_dotdot_path_escapes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "sibling.md").write_text("x")
    assert escapes_root(root / ".." / "sibling.md", root) is True


async def test_the_walk_skips_a_symlink_that_escapes_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.md").write_text("not yours")
    (root / "kept.md").write_text("mine")
    (root / "escape.md").symlink_to(outside / "secret.md")

    keys = [item.external_key for item in await observe_all(make_source(root))]

    assert keys == ["kept.md"]


# --------------------------------------------------------------------------
# The (mtime, size) filter
# --------------------------------------------------------------------------


def test_fingerprint_is_mtime_and_size(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("hello")
    stat = path.stat()
    assert fingerprint(stat) == [stat.st_mtime_ns, stat.st_size]


def test_fingerprint_changes_when_content_changes(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("hello")
    before = fingerprint(path.stat())

    os.utime(path, ns=(0, 0))
    path.write_text("hello there")

    assert fingerprint(path.stat()) != before


def test_fingerprint_is_unchanged_by_reading(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("hello")
    before = fingerprint(path.stat())
    path.read_text()
    assert fingerprint(path.stat()) == before


async def test_an_incremental_walk_skips_files_the_filter_says_are_unchanged(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.md").write_text("alpha")
    (tmp_path / "b.md").write_text("beta")

    source = make_source(tmp_path)
    first = await observe_all(source, full=True)
    assert sorted(item.external_key for item in first) == ["a.md", "b.md"]

    # Feed back what a sync would have stored in the cursor.
    cursor = {"seen": {item.external_key: item.fingerprint for item in first}}
    primed = Source(
        id=source.id,
        kind=source.kind,
        name=source.name,
        config=source.config,
        cursor=cursor,
    )

    assert await observe_all(primed, full=False) == []

    # Touching one file brings exactly that one back into view.
    (tmp_path / "b.md").write_text("beta changed")
    changed = await observe_all(primed, full=False)
    assert [item.external_key for item in changed] == ["b.md"]


async def test_a_full_walk_ignores_the_filter_entirely(tmp_path: Path) -> None:
    # A full sync has to look at everything, because its other job is noticing
    # what has gone missing, and you cannot compare against a set you did not
    # build.
    (tmp_path / "a.md").write_text("alpha")
    source = make_source(tmp_path)
    first = await observe_all(source, full=True)

    primed = Source(
        id=source.id,
        kind=source.kind,
        name=source.name,
        config=source.config,
        cursor={"seen": {item.external_key: item.fingerprint for item in first}},
    )

    assert [item.external_key for item in await observe_all(primed, full=True)] == ["a.md"]


# --------------------------------------------------------------------------
# Walking rules
# --------------------------------------------------------------------------


async def test_the_walk_reports_hash_size_and_provenance(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hello")

    (item,) = await observe_all(make_source(tmp_path))

    assert item.external_key == "notes.md"
    assert item.byte_size == 5
    assert item.content_hash.value == __import__("hashlib").blake2b(
        b"hello", digest_size=32
    ).hexdigest()
    assert item.media_type == "text/markdown"
    # mtime is weak evidence, recorded honestly rather than defaulted to now().
    assert item.occurred_at is not None
    assert item.occurred_at_source is TimeProvenance.FILESYSTEM


async def test_read_bytes_is_lazy_and_returns_the_content(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("hello")
    (item,) = await observe_all(make_source(tmp_path))
    assert await item.read_bytes() == b"hello"


async def test_oversized_files_are_skipped_without_aborting(tmp_path: Path) -> None:
    (tmp_path / "small.md").write_text("x")
    (tmp_path / "huge.md").write_text("y" * 5000)

    source = make_source(tmp_path, max_file_bytes=1000)
    keys = [item.external_key for item in await observe_all(source)]

    assert keys == ["small.md"]


async def test_excluded_directories_are_never_descended_into(tmp_path: Path) -> None:
    (tmp_path / "kept.md").write_text("x")
    for excluded in [".git", "node_modules", "__pycache__", ".venv"]:
        directory = tmp_path / excluded
        directory.mkdir()
        (directory / "inside.md").write_text("x")

    keys = [item.external_key for item in await observe_all(make_source(tmp_path))]

    assert keys == ["kept.md"]


async def test_files_outside_the_include_list_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "kept.md").write_text("x")
    (tmp_path / "ignored.bin").write_bytes(b"\x00")

    keys = [item.external_key for item in await observe_all(make_source(tmp_path))]

    assert keys == ["kept.md"]


async def test_a_missing_root_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a directory"):
        await observe_all(make_source(tmp_path / "nope"))


def test_config_requires_a_root() -> None:
    with pytest.raises(ValueError, match="requires a 'root'"):
        FilesystemConfig.from_dict({})


def test_config_resolves_the_root(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    config = FilesystemConfig.from_dict({"root": str(nested / ".." / "b")})
    assert config.root == nested.resolve()
