"""Command line entry point.

argparse, deliberately. The moment a CLI framework is in the tree, every later
command gets written in it; the commands here do not need one.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from memoryos.adapters.connectors.filesystem import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    DEFAULT_MAX_FILE_BYTES,
)
from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.application.backfill import (
    enqueue_embedding,
    find_unembedded,
    gather_stats,
)
from memoryos.application.doctor import run_doctor
from memoryos.application.evaluate import (
    compare as compare_runs,
)
from memoryos.application.evaluate import (
    evaluate,
    format_report,
    format_verbose,
)
from memoryos.application.evaluation import format_table, measure_recall
from memoryos.application.golden import load_golden_set
from memoryos.application.judgements import export_golden_set
from memoryos.application.ports import SearchFilters
from memoryos.application.rechunk import enqueue_rechunk, find_stale
from memoryos.application.replay import PartialShadowReplay, ReplayScope, ReplayStage
from memoryos.application.verification import compare, snapshot
from memoryos.application.worker import Worker, WorkerConfig
from memoryos.config import Settings, get_settings
from memoryos.container import Container
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.values import SearchMode, SourceKind
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


async def run_embed(
    settings: Settings, *, source: str | None, dry_run: bool, stale_only: bool = False
) -> int:
    container = Container.build(settings)
    try:
        model_id = container.embedder.model_id
        pending = await find_unembedded(
            container.database.session_factory,
            model_id=model_id,
            source=source,
            stale_only=stale_only,
        )

        print(f"target model:      {model_id}")
        print(f"memories pending:  {len(pending)}")
        print(f"chunks pending:    {sum(memory.chunks for memory in pending)}")
        for memory in pending[:20]:
            print(f"  {memory.external_key} ({memory.chunks} chunks)")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")

        if dry_run:
            print("dry run; nothing enqueued")
            return 0

        enqueued = await enqueue_embedding(container.database.session_factory, pending)
        print(f"enqueued: {enqueued}")
    finally:
        await container.dispose()
    return 0


async def run_search(
    settings: Settings,
    *,
    query: str,
    k: int,
    source: str | None,
    exact: bool,
    mode: SearchMode,
) -> int:
    container = Container.build(settings)
    try:
        filters = SearchFilters()
        if source is not None:
            async with container.database.session_factory() as session:
                source_ids = list(
                    (
                        await session.execute(
                            select(models.Source.id).where(models.Source.name == source)
                        )
                    ).scalars()
                )
            if not source_ids:
                print(f"no source named {source!r}")
                return 1
            filters = SearchFilters(source_ids=source_ids)

        result = await container.search()(
            query, k=k, filters=filters, exact=exact, mode=mode
        )

        if mode is SearchMode.KEYWORD:
            described = "keyword (ts_rank_cd)"
        else:
            described = "exact" if exact else f"ann (ef_search={settings.hnsw_ef_search})"
        print(f'query: {result.query!r}   [{described}]')
        print(
            f"timing: embed {result.timing.embed_ms}ms  "
            f"search {result.timing.search_ms}ms  total {result.timing.total_ms}ms\n"
        )
        if not result.hits:
            print("no results")
        for rank, hit in enumerate(result.hits, start=1):
            best = max(hit.matched_chunks, key=lambda chunk: chunk.score)
            excerpt = " ".join(best.text.split())[:160]
            print(f"{rank}. {hit.score:.4f}  {hit.external_key}")
            print(f"     chunks matched: {len(hit.matched_chunks)}  best: #{best.ordinal}")
            print(f"     {excerpt}\n")
    finally:
        await container.dispose()
    return 0


async def run_eval_recall(
    settings: Settings, *, queries: int, k: int, ef_search_values: list[int]
) -> int:
    container = Container.build(settings)
    try:
        rows = await measure_recall(
            container.database.session_factory,
            container.embedder,
            container.vectors,
            queries=queries,
            k=k,
            ef_search_values=ef_search_values,
        )
        if not rows:
            print("no embedded chunks to evaluate")
            return 1
        print(f"queries: {rows[0].queries}   k: {k}\n")
        print(format_table(rows, k))
        print(
            "\nself@1 is the fraction of queries where the chunk used as the query "
            "came back first;\nanything well below 1.0 points at the pipeline, not the "
            "index."
        )
    finally:
        await container.dispose()
    return 0


async def run_evaluate(
    settings: Settings,
    *,
    golden_path: Path,
    k: int,
    json_path: Path | None,
    query: str | None,
    verbose: bool,
    compare_path: Path | None,
    worst: int,
    mode: SearchMode,
) -> int:
    """Score the golden set through the ordinary search path.

    Exits non-zero only when the run could not happen — an empty golden set, a
    missing export, a `--query` that names nothing. A *low score* is a result,
    not a failure, and making the command fail on one would mean every later
    milestone starts by disabling it.
    """
    if not golden_path.exists():
        print(
            f"no golden set at {golden_path}\n"
            f"run: memoryos export-golden-set --output {golden_path}"
        )
        return 1

    container = Container.build(settings)
    try:
        sessions = container.database.session_factory
        golden = await load_golden_set(golden_path, sessions)
        if query is not None:
            selected = golden.select(query)
            if not selected.queries:
                print(f"no golden query matching {query!r}")
                return 1
            golden = selected

        if not golden.queries:
            print("no scoreable queries in the golden set")
            for item in golden.excluded:
                print(f"  excluded: {item.query_text}  ({item.reason})")
            return 1

        run = await evaluate(
            golden, container.search(), sessions, k=k, now=datetime.now(UTC), mode=mode
        )
        print(format_report(run, worst=worst))

        if verbose:
            print()
            print(format_verbose(run))

        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(run.as_dict(), indent=2) + "\n")
            print(f"\nwrote {json_path}")

        if compare_path is not None:
            if not compare_path.exists():
                print(f"\nno baseline at {compare_path}")
                return 1
            print()
            print(f"compared against {compare_path}")
            print(compare_runs(json.loads(compare_path.read_text()), run).render())
    finally:
        await container.dispose()
    return 0


async def run_doctor_command(settings: Settings) -> int:
    container = Container.build(settings)
    try:
        report = await run_doctor(container.database.session_factory, container.embedder)
        print(f"model:   {container.embedder.model_id}")
        print(f"window:  {container.embedder.max_sequence_tokens} tokens")
        print(f"chunker: {container.chunker.version}\n")
        for finding in report.findings:
            mark = "ok  " if finding.healthy else "FAIL"
            print(f"[{mark}] {finding.check}: {finding.count}")
            if not finding.healthy:
                print(f"        {finding.detail}")
                for example in finding.examples:
                    print(f"        - {example}")
        print()
        print("healthy" if report.healthy else "problems found")
    finally:
        await container.dispose()
    return 0 if report.healthy else 1


async def run_stats(settings: Settings) -> int:
    container = Container.build(settings)
    try:
        stats = await gather_stats(container.database.session_factory)
        print(f"memories          {stats.memories} ({stats.current_memories} current)")
        print(f"chunks            {stats.chunks}")
        print(
            f"embedded chunks   {stats.embedded_chunks} "
            f"({stats.coverage:.1%} coverage)"
        )
        print(f"cache entries     {stats.cache_entries}")
        if stats.chunks:
            # How much of the corpus the cache spared. Distinct from the
            # per-run hit rate, which the embed job logs as it goes.
            reuse = 1 - (stats.cache_entries / stats.chunks) if stats.chunks else 0
            print(f"cache reuse       {reuse:.1%} of chunks shared a cached vector")
        for model, count in sorted(stats.models.items()):
            print(f"  {model or '(none)'}: {count}")
    finally:
        await container.dispose()
    return 0


async def run_replay(
    settings: Settings,
    *,
    scope: ReplayScope,
    into_shadow: bool,
    clear_cache: bool,
) -> int:
    container = Container.build(settings)
    try:
        print(f"scope:  {scope.describe()}")
        print(f"cache:  {'cleared' if clear_cache else 'kept (content-addressed)'}")
        print(f"target: {'shadow schema, swapped in' if into_shadow else 'in place'}\n")

        report = await container.replay()(
            scope, into_shadow=into_shadow, clear_cache=clear_cache
        )
        print(json.dumps(report.as_dict(), indent=2))
    except (LookupError, PartialShadowReplay) as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    return 0


async def run_verify_replay(
    settings: Settings, *, sample: int | None, clear_cache: bool
) -> int:
    """Rebuild into a shadow schema and prove it matches, or say exactly how not.

    The live corpus is never modified: the workspace is built, compared, and
    dropped. A non-zero exit is the whole point — a verification that cannot fail
    is not a verification, which is why one of this milestone's tests corrupts a
    chunk on purpose and requires this command to notice.
    """
    container = Container.build(settings)
    try:
        sessions = container.database.session_factory
        before = await snapshot(sessions, sample=sample)
        if not before.memories:
            print("nothing to verify: the corpus is empty")
            return 1

        print(f"snapshot: {before.counts}")
        if sample is not None:
            print(f"sampled:  {sample} memories")
        print("rebuilding into a shadow schema (the live tables are not touched)\n")

        async with container.replay().rebuild_into_shadow(
            clear_cache=clear_cache
        ) as (report, shadow_sessions):
            after = await snapshot(shadow_sessions, sample=sample)

        result = compare(before, after)
        print(json.dumps(report.as_dict(), indent=2))
        print()
        print(result.render())
        print()
        print("identical" if result.identical else "MISMATCH: the rebuild differs")
        return 0 if result.identical else 1
    except LookupError as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()


async def run_export_golden_set(settings: Settings, *, output: Path) -> int:
    """Write the labelled data to a file. M2.0's direct input.

    Ids are re-resolved from natural keys as it writes, so an export taken after
    a rebuild still points at rows that exist. An item that has left the corpus
    exports with a null id and is counted as unresolved rather than dropped —
    silently omitting it would make a shrinking corpus look like a shrinking
    disagreement.
    """
    container = Container.build(settings)
    try:
        golden = await export_golden_set(
            container.database.session_factory, now=datetime.now(UTC)
        )
        payload = golden.as_dict()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

        totals = golden.totals
        print(f"wrote {output}")
        print(f"queries       {totals['queries']}")
        print(f"judgements    {totals['judgements']}")
        print(
            f"  relevant    {totals['relevant']}\n"
            f"  not relevant{totals['not_relevant']:>3}\n"
            f"  missing     {totals['missing']}"
        )
        if totals["unresolved"]:
            print(
                f"unresolved    {totals['unresolved']} judged items are no longer "
                f"in the corpus"
            )
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

    embed = commands.add_parser(
        "embed", help="enqueue embedding for memories with unembedded chunks"
    )
    embed.add_argument("--source", help="limit to one source by name")
    embed.add_argument(
        "--dry-run", action="store_true", help="report what would be enqueued"
    )

    reembed = commands.add_parser(
        "reembed", help="enqueue re-embedding for chunks from a different model"
    )
    reembed.add_argument(
        "--model",
        dest="model",
        help="target model id; defaults to the configured one",
    )
    reembed.add_argument("--source", help="limit to one source by name")
    reembed.add_argument(
        "--dry-run", action="store_true", help="report what would be enqueued"
    )

    commands.add_parser("stats", help="report corpus and embedding coverage")
    commands.add_parser(
        "doctor", help="check the corpus for silently-degrading conditions"
    )

    search = commands.add_parser("search", help="semantic search over memories")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=10, help="how many memories to return")
    search.add_argument("--source", help="limit to one source by name")
    search.add_argument(
        "--exact",
        action="store_true",
        help="sequential scan instead of the index, for spot-checking what it missed",
    )
    search.add_argument(
        "--mode",
        choices=[value.value for value in SearchMode],
        default=SearchMode.VECTOR.value,
        help="which retriever answers: 'vector' embeds the query, 'keyword' matches terms",
    )

    replay = commands.add_parser(
        "replay", help="rebuild the derived tables from the event log and blobs"
    )
    scope_group = replay.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--from-beginning",
        dest="from_beginning",
        action="store_true",
        help="replay the whole log; the full proof",
    )
    scope_group.add_argument("--source", help="replay one connector's corpus")
    scope_group.add_argument(
        "--since",
        type=int,
        metavar="SEQ",
        help="replay only events after this log position",
    )
    replay.add_argument(
        "--stage",
        choices=[stage.value for stage in ReplayStage if stage is not ReplayStage.ALL],
        help=(
            "keep upstream artifacts and redo from here: 'normalize' re-chunks and "
            "re-embeds, 'embed' keeps chunk rows and only recomputes vectors"
        ),
    )
    replay.add_argument(
        "--into-shadow",
        dest="into_shadow",
        action="store_true",
        help=(
            "build into a separate schema and swap it in, instead of in place; "
            "requires the whole log at stage 'all', because a swap replaces the "
            "derived tables rather than merging into them"
        ),
    )
    replay.add_argument(
        "--clear-cache",
        dest="clear_cache",
        action="store_true",
        help=(
            "recompute every vector instead of reusing the content-addressed "
            "cache; slower, and the stronger check"
        ),
    )

    verify = commands.add_parser(
        "verify-replay",
        help="rebuild into a shadow schema and prove it matches the live corpus",
    )
    verify.add_argument(
        "--sample",
        type=int,
        help="compare only the first N memories, for a corpus too large to do whole",
    )
    verify.add_argument(
        "--clear-cache",
        dest="clear_cache",
        action="store_true",
        help="recompute every vector during the rebuild rather than reusing cached ones",
    )

    golden = commands.add_parser(
        "export-golden-set",
        help="write the captured judgements to a file, ids re-resolved",
    )
    golden.add_argument(
        "--output",
        type=Path,
        default=Path("var/golden-set.json"),
        help="where to write the JSON",
    )

    eval_recall = commands.add_parser(
        "eval-recall", help="measure index recall against an exhaustive scan"
    )
    eval_recall.add_argument("--queries", type=int, default=50)
    eval_recall.add_argument("-k", "--k", dest="k", type=int, default=10)
    eval_recall.add_argument(
        "--ef-search",
        dest="ef_search",
        default="40,100,200,400",
        help="comma-separated ef_search values to compare",
    )

    evaluate_command = commands.add_parser(
        "evaluate", help="score the golden set through the ordinary search path"
    )
    evaluate_command.add_argument("-k", "--k", dest="k", type=int, default=10)
    evaluate_command.add_argument(
        "--golden",
        type=Path,
        default=Path("var/golden-set.json"),
        help="the export to score against",
    )
    evaluate_command.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="write the full result here, for comparison across runs",
    )
    evaluate_command.add_argument(
        "--query", help="score only the golden query with this exact text"
    )
    evaluate_command.add_argument(
        "--verbose",
        action="store_true",
        help="print every ranking with a relevant/not marker, to diagnose a bad score",
    )
    evaluate_command.add_argument(
        "--compare",
        dest="compare_path",
        type=Path,
        help="a previous --json run; prints per-metric deltas and any MRR regression",
    )
    evaluate_command.add_argument(
        "--worst",
        type=int,
        default=3,
        help="how many of the worst queries by MRR to list",
    )
    evaluate_command.add_argument(
        "--mode",
        choices=[value.value for value in SearchMode],
        default=SearchMode.VECTOR.value,
        help="which retriever to score; the same golden set judges both",
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

    if args.command == "embed":
        return asyncio.run(
            run_embed(settings, source=args.source, dry_run=args.dry_run)
        )

    if args.command == "reembed":
        if args.model:
            settings = settings.model_copy(update={"embedding_model": args.model})
        return asyncio.run(
            run_embed(
                settings, source=args.source, dry_run=args.dry_run, stale_only=True
            )
        )

    if args.command == "stats":
        return asyncio.run(run_stats(settings))

    if args.command == "doctor":
        return asyncio.run(run_doctor_command(settings))

    if args.command == "search":
        return asyncio.run(
            run_search(
                settings,
                query=args.query,
                k=args.k,
                source=args.source,
                exact=args.exact,
                mode=SearchMode(args.mode),
            )
        )

    if args.command == "eval-recall":
        values = [int(value) for value in args.ef_search.split(",") if value.strip()]
        return asyncio.run(
            run_eval_recall(settings, queries=args.queries, k=args.k, ef_search_values=values)
        )

    if args.command == "evaluate":
        return asyncio.run(
            run_evaluate(
                settings,
                golden_path=args.golden,
                k=args.k,
                json_path=args.json_path,
                query=args.query,
                verbose=args.verbose,
                compare_path=args.compare_path,
                worst=args.worst,
                mode=SearchMode(args.mode),
            )
        )

    if args.command == "replay":
        return asyncio.run(
            run_replay(
                settings,
                scope=ReplayScope(
                    source_name=args.source,
                    after_seq=args.since or 0,
                    stage=ReplayStage(args.stage) if args.stage else ReplayStage.ALL,
                ),
                into_shadow=args.into_shadow,
                clear_cache=args.clear_cache,
            )
        )

    if args.command == "verify-replay":
        return asyncio.run(
            run_verify_replay(
                settings, sample=args.sample, clear_cache=args.clear_cache
            )
        )

    if args.command == "export-golden-set":
        return asyncio.run(run_export_golden_set(settings, output=args.output))

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
