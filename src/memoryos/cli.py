"""Command line entry point.

argparse, deliberately. One command does not justify a dependency, and the
moment a CLI framework is in the tree, every later command is written in it.
"""

import argparse
import asyncio
from datetime import timedelta

from memoryos.adapters.db.engine import Database
from memoryos.adapters.db.job_queue import PostgresJobQueue
from memoryos.application.jobs.handlers import build_default_registry
from memoryos.application.worker import Worker, WorkerConfig
from memoryos.config import Settings, get_settings
from memoryos.logging import configure_logging


async def run_worker(settings: Settings, *, lease_seconds: float, drain: bool) -> None:
    database = Database.from_url(settings.database_url, echo=settings.db_echo)
    try:
        worker = Worker(
            queue=PostgresJobQueue(database.session_factory),
            registry=build_default_registry(),
            session_factory=database.session_factory,
            config=WorkerConfig(lease=timedelta(seconds=lease_seconds)),
        )
        await worker.run(drain=drain)
    finally:
        await database.dispose()


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


if __name__ == "__main__":
    raise SystemExit(main())
