"""The vault connector, through the same `SyncSource` every source uses (M12.2).

The four the milestone asks for, and two it implies: that the walk keeps a
vault's own machinery (`.obsidian/`, the trash) out of the corpus, and that a
`[[link]]` lands on an entity that already exists rather than making a second
one with a guessed type.
"""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.connectors.registry import ConnectorsByKind
from memoryos.adapters.connectors.vault import VaultConnector, default_config
from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.extraction.links import LinkedExtractor
from memoryos.adapters.parsers.registry import build_default_registry
from memoryos.application.extraction import ExtractEntities
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.sync import SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import EntityType, SourceKind, TimeProvenance
from tests.support.fakes import FakeEmbedder, FakeEntityExtractor

pytestmark = pytest.mark.integration

#: A fixed mtime, so a note dated from the filesystem has a date the test knows.
MTIME = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)


def write(root: Path, key: str, text: str) -> Path:
    path = root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (MTIME.timestamp(), MTIME.timestamp()))
    return path


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    write(
        root,
        "projects/harbour.md",
        "---\ndate: 2024-03-15\nupdated: 2024-09-01\n---\n"
        "# Harbour\n\nWe chose [[Postgres]] over [[DynamoDB]] for the ledger.\n",
    )
    write(
        root,
        "people/ada.md",
        "---\ncreated: 2023-11-02T09:30:00\n---\nAda runs the [[Harbour]] review.\n",
    )
    write(root, "journals/2024_02_07.md", "Met Ada about [[Harbour]] budgets.\n")
    write(root, "inbox/undated.md", "No front matter here, only an mtime.\n")
    write(root, "inbox/broken.md", "---\ndate: [unclosed\n---\nBroken YAML.\n")
    # Vault machinery and non-notes: none of these is a memory.
    write(root, ".obsidian/workspace.md", "# layout state\n")
    write(root, ".trash/old.md", "deleted note\n")
    write(root, "attachments/scan.txt", "plain text is not a note in a vault\n")
    return root


class Vault:
    def __init__(
        self,
        root: Path,
        source: Source,
        sessions: async_sessionmaker[AsyncSession],
        sync: SyncSource,
    ) -> None:
        self.root = root
        self.source = source
        self.sessions = sessions
        self.sync = sync

    async def scalar(self, stmt: object) -> object:
        async with self.sessions() as session:
            return (await session.execute(stmt)).scalar_one()  # type: ignore[call-overload]

    async def memories(self) -> dict[str, models.Memory]:
        async with self.sessions() as session:
            rows = (
                (
                    await session.execute(
                        select(models.Memory).where(
                            models.Memory.source_id == self.source.id,
                            models.Memory.is_current.is_(True),
                        )
                    )
                )
                .scalars()
                .all()
            )
        return {row.external_key: row for row in rows}


@pytest.fixture
async def synced(vault: Path, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]) -> Vault:
    source = Source(id=new_id(), kind=SourceKind.VAULT, name="notes", config=default_config(vault))
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)
    blobs = FilesystemBlobStore(tmp_path / "blobs")
    # Through the dispatcher the container uses, so the test also proves that a
    # vault source reaches the vault connector without `SyncSource` knowing.
    connectors = ConnectorsByKind(
        {
            SourceKind.FILESYSTEM: FilesystemConnector(blobs),
            SourceKind.VAULT: VaultConnector(blobs),
        }
    )
    handle = Vault(vault, source, sessions, SyncSource(sessions, connectors, blobs))
    await handle.sync(source.id, full=True)
    return handle


async def reload(handle: Vault) -> Source:
    async with handle.sessions() as session:
        found = await SqlAlchemySourceRepository(session).get(handle.source.id)
    assert found is not None
    return found


# --------------------------------------------------------------------------
# The four
# --------------------------------------------------------------------------


async def test_an_incremental_sync_with_no_changes_does_nothing(synced: Vault) -> None:
    events_before = await synced.scalar(select(func.count()).select_from(models.IngestionEvent))
    jobs_before = await synced.scalar(select(func.count()).select_from(models.Job))

    report = await synced.sync(synced.source.id, full=False)

    # Not even observed: the `(mtime, size)` cursor filters every file before a
    # byte is read, which is what makes re-running it nearly free.
    assert report.observed == 0
    assert (report.new, report.modified, report.deleted) == (0, 0, 0)
    assert (
        await synced.scalar(select(func.count()).select_from(models.IngestionEvent))
        == events_before
    )
    assert await synced.scalar(select(func.count()).select_from(models.Job)) == jobs_before


async def test_identical_content_from_two_places_is_stored_once(
    synced: Vault,
) -> None:
    text = "The same paragraph, pasted into two notes.\n"
    write(synced.root, "a/copy.md", text)
    write(synced.root, "b/copy.md", text)

    await synced.sync(synced.source.id, full=False)

    memories = await synced.memories()
    assert memories["a/copy.md"].content_hash == memories["b/copy.md"].content_hash
    artifacts = await synced.scalar(
        select(func.count())
        .select_from(models.RawArtifact)
        .where(models.RawArtifact.content_hash == memories["a/copy.md"].content_hash)
    )
    assert artifacts == 1


async def test_each_date_is_recorded_with_the_provenance_it_earned(synced: Vault) -> None:
    memories = await synced.memories()

    def dated(key: str) -> tuple[datetime | None, str]:
        row = memories[key]
        return row.occurred_at, row.occurred_at_source

    # `date` wins over `updated`: what a note is about, not when it was touched.
    assert dated("projects/harbour.md") == (
        datetime(2024, 3, 15, tzinfo=UTC),
        TimeProvenance.DECLARED.value,
    )
    assert dated("people/ada.md") == (
        datetime(2023, 11, 2, 9, 30, tzinfo=UTC),
        TimeProvenance.DECLARED.value,
    )
    # A daily note's name is its date, read off rather than stated.
    assert dated("journals/2024_02_07.md") == (
        datetime(2024, 2, 7, tzinfo=UTC),
        TimeProvenance.PARSED.value,
    )
    # Nothing stated, nothing named: the mtime, and it says so.
    assert dated("inbox/undated.md") == (MTIME, TimeProvenance.FILESYSTEM.value)
    # Broken front matter is a typo, not a lost note.
    assert dated("inbox/broken.md") == (MTIME, TimeProvenance.FILESYSTEM.value)


async def test_a_deletion_is_detected_on_a_full_sweep_and_only_then(synced: Vault) -> None:
    (synced.root / "inbox/undated.md").unlink()

    incremental = await synced.sync(synced.source.id, full=False)
    assert incremental.deleted == 0
    assert (await synced.memories())["inbox/undated.md"].deleted_at is None

    full = await synced.sync(synced.source.id, full=True)
    assert full.deleted == 1
    assert (await synced.memories())["inbox/undated.md"].deleted_at is not None
    # And the cursor forgets it, so it is not carried as dead weight.
    assert "inbox/undated.md" not in (await reload(synced)).cursor["seen"]


# --------------------------------------------------------------------------
# What the four imply
# --------------------------------------------------------------------------


async def test_the_vaults_own_machinery_stays_out(synced: Vault) -> None:
    keys = set(await synced.memories())
    assert keys == {
        "projects/harbour.md",
        "people/ada.md",
        "journals/2024_02_07.md",
        "inbox/undated.md",
        "inbox/broken.md",
    }


async def test_configured_include_and_exclude_are_respected(
    vault: Path, tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    config = {
        **default_config(vault),
        "include": ["**/*.md", "**/*.txt"],
        "exclude": ["**/inbox/**"],
    }
    source = Source(id=new_id(), kind=SourceKind.VAULT, name="custom", config=config)
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)
    blobs = FilesystemBlobStore(tmp_path / "blobs-custom")
    await SyncSource(sessions, VaultConnector(blobs), blobs)(source.id, full=True)

    async with sessions() as session:
        keys = set(
            (
                await session.execute(
                    select(models.Memory.external_key).where(models.Memory.source_id == source.id)
                )
            ).scalars()
        )
    assert "attachments/scan.txt" in keys
    assert not any(key.startswith("inbox/") for key in keys)
    # A replaced exclude list is exactly what is applied, so `.obsidian/` is
    # back in: the configuration is the rule, not a suggestion on top of one.
    assert ".obsidian/workspace.md" in keys


async def test_a_link_lands_on_the_entity_that_already_has_that_name(
    synced: Vault, tmp_path: Path
) -> None:
    """`[[Harbour]]` does not say it is a project. An existing entity does."""
    blobs = FilesystemBlobStore(tmp_path / "blobs")
    normalize = NormalizeMemory(
        synced.sessions, blobs, build_default_registry(), StructuralChunker(FakeEmbedder())
    )
    memories = await synced.memories()
    for row in memories.values():
        await normalize(row.id)

    # The "model" knows Harbour is a project, from the note that is about it.
    model = FakeEntityExtractor(pattern=r"\bHarbour\b", entity_type=EntityType.PROJECT)
    await ExtractEntities(synced.sessions, LinkedExtractor(model))(
        memories["projects/harbour.md"].id
    )
    # The other two only link to it, read with no model at all: the link alone
    # carries no type, so the type has to come from what already exists.
    links_only = ExtractEntities(synced.sessions, LinkedExtractor(None))
    await links_only(memories["people/ada.md"].id)
    await links_only(memories["journals/2024_02_07.md"].id)

    async with synced.sessions() as session:
        harbour = (
            await session.execute(
                select(models.Entity.type, func.count(models.EntityMention.id))
                .join(models.EntityMention, models.EntityMention.entity_id == models.Entity.id)
                .where(models.Entity.canonical_name == "harbour")
                .group_by(models.Entity.type)
            )
        ).all()
        postgres = (
            (
                await session.execute(
                    select(models.Entity.type).where(models.Entity.canonical_name == "postgres")
                )
            )
            .scalars()
            .all()
        )

    # One entity, a project, mentioned from all three notes — not a project
    # plus a "concept" that the links invented.
    assert [tuple(row) for row in harbour] == [(EntityType.PROJECT.value, 3)]
    # A link to a name nobody has typed yet falls back to concept.
    assert postgres == [EntityType.CONCEPT.value]
