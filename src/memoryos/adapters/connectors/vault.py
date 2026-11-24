"""The vault connector: a folder of notes, dated the way notes date themselves.

M12.2, and deliberately small. A vault — Obsidian, Logseq, or a plain folder of
Markdown — is walked exactly as a directory is: the same exclusion pruning, the
same `(mtime, size)` cursor, the same one-pass hash-into-the-blob-store, the
same deletion sweep (which belongs to `SyncSource`, not to any connector).
Everything here is the one question a filesystem cannot answer well: **when was
this written?**

The filesystem's answer is mtime, and on this project's own corpus that answer
is the whole reason Phase 4 to 8 are starved: a checkout resets every mtime at
once, so 282 memories have one date between them. A notes vault usually knows
better, in one of two ways, and each is recorded with the provenance it earns:

* **Front matter** — `date`, `created` or `updated`, in that order of
  preference. The author wrote it, so it is `DECLARED`. `date` first because
  it is what a note says it is *about*; `created` next because it is when the
  note began; `updated` last, because a date somebody last touched a file is
  the weakest of the three, though still stated rather than guessed.
* **A daily note's file name** — `2024-03-15.md`, or Logseq's
  `journals/2024_03_15.md`. Not stated in the note, but read off something the
  author named on purpose, so `PARSED` rather than `DECLARED`.

Otherwise mtime, marked `FILESYSTEM`, exactly as the filesystem connector does.

A declared date with no time zone is taken as UTC. A note that says
`date: 2024-03-15` has said which day, not which instant, and midnight UTC is
the convention this schema already uses for day-grain dates.

`[[wiki links]]` are not handled here. They are a property of the *text* — a
link means the same thing in a note synced from a vault as in a README synced
from a repository — so they are found where entities are found, by the
extractor (`adapters/extraction/links.py`), and this connector stays a walk.
"""

import asyncio
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import frontmatter
import structlog

from memoryos.adapters.connectors.filesystem import (
    DEFAULT_EXCLUDE,
    DEFAULT_MAX_FILE_BYTES,
    FilesystemConfig,
    FilesystemConnector,
)
from memoryos.domain.values import SourceKind, TimeProvenance

logger = structlog.get_logger(__name__)

VAULT_INCLUDE = ["**/*.md", "**/*.markdown"]
VAULT_EXCLUDE = [
    *DEFAULT_EXCLUDE,
    # The application's own state, not notes: settings, plugins, workspace
    # layout, and the trash a vault keeps instead of deleting.
    "**/.obsidian/**",
    "**/.trash/**",
    "**/logseq/bak/**",
    "**/logseq/.recycle/**",
]

#: Front-matter keys that date a note, most specific first.
DATE_KEYS = ("date", "created", "updated")

#: Enough for any front matter a person writes; a note whose front matter runs
#: past this is not one whose date is worth a second full read.
_HEAD_BYTES = 64 * 1024

#: `2024-03-15`, `2024_03_15`, optionally followed by a title.
_DAILY_NOTE = re.compile(r"^(\d{4})[-_](\d{2})[-_](\d{2})(?:$|[^\d])")

#: Outside this range a "date" is a typo or a placeholder, not a date.
_EARLIEST_YEAR = 1900
_LATEST_YEAR = 2100


def default_config(root: Path) -> dict[str, Any]:
    """What `source add --kind vault` stores when nothing else is asked for."""
    return {
        "root": str(root),
        "include": list(VAULT_INCLUDE),
        "exclude": list(VAULT_EXCLUDE),
        "max_file_bytes": DEFAULT_MAX_FILE_BYTES,
        "follow_symlinks": False,
    }


class VaultConnector(FilesystemConnector):
    """A filesystem walk whose items are dated from the notes themselves."""

    kind: SourceKind = SourceKind.VAULT

    def _config(self, raw: dict[str, Any]) -> FilesystemConfig:
        return FilesystemConfig.from_dict(
            {
                **raw,
                "include": raw.get("include") or VAULT_INCLUDE,
                "exclude": raw.get("exclude") or VAULT_EXCLUDE,
            }
        )

    async def _dated(
        self, path: Path, external_key: str, stat: Any
    ) -> tuple[datetime | None, TimeProvenance]:
        declared = await asyncio.to_thread(declared_date, path)
        if declared is not None:
            return declared, TimeProvenance.DECLARED
        parsed = date_from_name(external_key)
        if parsed is not None:
            return parsed, TimeProvenance.PARSED
        return await super()._dated(path, external_key, stat)


def declared_date(path: Path) -> datetime | None:
    """The date a note's front matter states, or None."""
    try:
        with path.open("rb") as handle:
            head = handle.read(_HEAD_BYTES)
    except OSError:
        return None
    return date_from_front_matter(head.decode("utf-8", errors="replace"))


def date_from_front_matter(text: str) -> datetime | None:
    if not text.startswith("---"):
        return None
    try:
        metadata = frontmatter.loads(text).metadata
    except Exception as exc:
        # Broken front matter is the author's typo, not a reason to drop the
        # note. It falls through to the next way of dating it.
        logger.info("vault.front_matter_unreadable", error=str(exc))
        return None
    for key in DATE_KEYS:
        value = _as_datetime(metadata.get(key))
        if value is not None:
            return value
    return None


def date_from_name(external_key: str) -> datetime | None:
    """A daily note's date, from its file name."""
    match = _DAILY_NOTE.match(Path(external_key).stem)
    if match is None:
        return None
    try:
        day = date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return None
    return _bounded(datetime(day.year, day.month, day.day, tzinfo=UTC))


def _as_datetime(value: Any) -> datetime | None:
    """YAML gives a date, a datetime, or a string. Anything else is not a date."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, date):
        moment = datetime(value.year, value.month, value.day)
    elif isinstance(value, str) and value.strip():
        text = value.strip().strip("'\"")
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return _bounded(moment)


def _bounded(moment: datetime) -> datetime | None:
    return moment if _EARLIEST_YEAR <= moment.year <= _LATEST_YEAR else None
