"""Command line entry point.

argparse, deliberately. The moment a CLI framework is in the tree, every later
command gets written in it; the commands here do not need one.
"""

import argparse
import asyncio
import json
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select

from memoryos.adapters.connectors.filesystem import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    DEFAULT_MAX_FILE_BYTES,
)
from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.application.rechunk import enqueue_rechunk, find_stale
from memoryos.application.worker import Worker, WorkerConfig
from memoryos.config import Settings, get_settings
from memoryos.container import Container
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import SourceKind
from memoryos.logging import configure_logging


async def run_worker(settings: Settings, *, lease_seconds: float, drain: bool) -> None:
    container = Container.build(settings)
    try:
        worker = Worker(
            queue=container.queue,
            registry=container.registry(),
            session_factory=container.database.session_factory,
            config=WorkerConfig(lease=timedelta(seconds=lease_seconds)),
        )
        await worker.run(drain=drain)
    finally:
        await container.dispose()


async def add_source(settings: Settings, *, kind: str, name: str, root: Path) -> int:
    container = Container.build(settings)
    try:
        async with container.database.session_factory.begin() as session:
            repository = SqlAlchemySourceRepository(session)
            source_kind = SourceKind(kind)
            if await repository.get_by_name(source_kind, name) is not None:
                print(f"source {name!r} already exists")
                return 1

            source = Source(
                id=new_id(),
                kind=source_kind,
                name=name,
                config={
                    "root": str(Path(root).expanduser().resolve()),
                    "include": DEFAULT_INCLUDE,
                    "exclude": DEFAULT_EXCLUDE,
                    "max_file_bytes": DEFAULT_MAX_FILE_BYTES,
                    "follow_symlinks": False,
                },
            )
            await repository.add(source)
        print(f"added source {name!r} ({source.id}) at {source.config['root']}")
    finally:
        await container.dispose()
    return 0


async def list_sources(settings: Settings) -> int:
    container = Container.build(settings)
    try:
        async with container.database.session_factory() as session:
            rows = (
                (await session.execute(select(models.Source).order_by(models.Source.name)))
                .scalars()
                .all()
            )
        if not rows:
            print("no sources")
        for row in rows:
            print(
                f"{row.name:20} {row.kind:12} {row.config.get('root', '')}\n"
                f"{'':20} id={row.id} last_sync={row.last_sync_at} "
                f"last_full_sync={row.last_full_sync_at}"
            )
    finally:
        await container.dispose()
    return 0


async def run_rechunk(
    settings: Settings, *, source: str | None, stale_version: str | None, dry_run: bool
) -> int:
    container = Container.build(settings)
    try:
        current = container.normalize().chunker_version
        stale = await find_stale(
            container.database.session_factory,
            current_version=current,
            source=source,
            stale_version=stale_version,
        )

        print(f"current chunker: {current}")
        print(f"stale memories:  {len(stale)}")
        for memory in stale[:20]:
            print(f"  {memory.external_key}")
        if len(stale) > 20:
            print(f"  ... and {len(stale) - 20} more")

        if dry_run:
            print("dry run; nothing enqueued")
            return 0

        enqueued = await enqueue_rechunk(container.database.session_factory, stale)
        print(f"enqueued: {enqueued}")
    finally:
        await container.dispose()
    return 0


async def run_sync(settings: Settings, *, name: str, full: bool) -> int:
    container = Container.build(settings)
    try:
        async with container.database.session_factory() as session:
            source = await SqlAlchemySourceRepository(session).get_by_name(
                SourceKind.FILESYSTEM, name
            )
        if source is None:
            print(f"no source named {name!r}")
            return 1

        report = await container.sync()(source.id, full=full)
        print(json.dumps(report.as_dict(), indent=2))
    finally:
        await container.dispose()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memoryos", description="Memory Intelligence OS")
    commands = parser.add_subparsers(dest="command", required=True)

    worker = commands.add_parser("worker", help="drain the job queue")
    worker.add_argument(
        "--lease-seconds",
        type=float,
        default=WorkerConfig().lease.total_seconds(),
        help="how long a claim is held without a heartbeat",
    )
    worker.add_argument(
        "--drain",
        action="store_true",
        help="exit once the queue has been empty for a few polls, instead of running forever",
    )

    source = commands.add_parser("source", help="manage sources")
    source_commands = source.add_subparsers(dest="source_command", required=True)

    source_add = source_commands.add_parser("add", help="register a source")
    source_add.add_argument("--kind", default=SourceKind.FILESYSTEM.value)
    source_add.add_argument("--name", required=True)
    source_add.add_argument("--root", required=True, type=Path)

    source_commands.add_parser("list", help="list registered sources")

    sync = commands.add_parser("sync", help="sync a source now")
    sync.add_argument("--source", required=True, help="source name")
    sync.add_argument(
        "--full",
        action="store_true",
        help="walk everything and reconcile deletions, rather than only what changed",
    )

    rechunk = commands.add_parser(
        "rechunk", help="re-normalize memories whose chunks are stale"
    )
    rechunk.add_argument("--source", help="limit to one source by name")
    rechunk.add_argument(
        "--chunker-version",
        dest="chunker_version",
        help="target this exact version instead of everything that is not current",
    )
    rechunk.add_argument(
        "--dry-run", action="store_true", help="report what would be enqueued"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings)

    if args.command == "worker":
        asyncio.run(
            run_worker(settings, lease_seconds=args.lease_seconds, drain=args.drain)
        )
        return 0

    if args.command == "source":
        if args.source_command == "add":
            return asyncio.run(
                add_source(settings, kind=args.kind, name=args.name, root=args.root)
            )
        return asyncio.run(list_sources(settings))

    if args.command == "sync":
        return asyncio.run(run_sync(settings, name=args.source, full=args.full))

    if args.command == "rechunk":
        return asyncio.run(
            run_rechunk(
                settings,
                source=args.source,
                stale_version=args.chunker_version,
                dry_run=args.dry_run,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
