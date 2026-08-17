"""Where a relative `blob_root` resolves to, and why it must not be `cwd`.

M10.3's report flagged the sharp edge these tests pin down. The API was started
once from `web/` while the worker ran from the repo root, and the two processes
read the same `MEMOS_BLOB_ROOT=./var/blobs` and disagreed about which directory it
named: the API wrote artifacts into `web/var/blobs`, the worker looked in
`var/blobs`, and a normalization job dead-lettered on a blob that had been stored
successfully. Nothing failed at the point of the mistake, which is what made it
cost an evening.

M1.7 met the same edge from the other direction — `replay` run from a
subdirectory resolved the default to an empty path and truncated 119 memories
before failing on the first document — and answered it with a preflight check.
The preflight is still right and still there; it turns a destroyed corpus into a
refusal. What it cannot do is make two processes agree, because both of them are
individually consistent. Only the anchor does that.
"""

import os
from pathlib import Path

import pytest

from memoryos.config import PROJECT_ROOT, Settings


def test_the_project_root_is_the_tree_rather_than_the_shell() -> None:
    """The anchor is found by walking up from the module, not from `cwd`.

    Which is the whole point: `cwd` is the input that varies between the two
    processes, so anything derived from it cannot be what they agree on.
    """
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_a_relative_blob_root_does_not_follow_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The M10.3 failure, reproduced as the two readings it produced.

    Asserted as an equality between two `Settings()` built from different
    directories rather than against a literal path, because the property that
    matters is *agreement*. A test naming one expected absolute path would still
    pass if both readings moved together, and moving together is all this needs
    to do.
    """
    monkeypatch.setenv("MEMOS_BLOB_ROOT", "./var/blobs")

    before = os.getcwd()
    try:
        from_root = Settings().blob_root
        os.chdir(tmp_path)
        from_elsewhere = Settings().blob_root
    finally:
        os.chdir(before)

    assert from_root == from_elsewhere
    assert from_root.is_absolute()
    # And it is the repo's own `var/blobs`, not the temporary directory the second
    # reading was taken from — which is exactly what `web/var/blobs` was.
    assert from_root == PROJECT_ROOT / "var" / "blobs"
    assert tmp_path not in from_elsewhere.parents


def test_an_absolute_blob_root_is_left_exactly_as_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Naming a path outright is the one case where nobody has to guess.

    The test suite points this at temporary directories and a real deployment
    points it at a volume; rewriting either would be the anchor overreaching from
    "resolve what is ambiguous" into "decide where blobs live".
    """
    monkeypatch.setenv("MEMOS_BLOB_ROOT", str(tmp_path / "blobs"))
    assert Settings().blob_root == tmp_path / "blobs"


def test_the_env_file_is_read_from_the_tree_too(tmp_path: Path) -> None:
    """`env_file` was the other half of the same failure.

    A cwd-relative `.env` run from `web/` is not found at all, so every setting
    silently falls back to its default — including the database URL. That is a
    worse version of the same bug: the blob root at least differed visibly once
    somebody went looking for the file.
    """
    before = os.getcwd()
    try:
        os.chdir(tmp_path)
        # No `.env` here. Read anyway, from the tree.
        assert Settings().model_config["env_file"] == PROJECT_ROOT / ".env"
    finally:
        os.chdir(before)
