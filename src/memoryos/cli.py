"""Command line entry point.

argparse, deliberately. The moment a CLI framework is in the tree, every later
command gets written in it; the commands here do not need one.
"""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from memoryos.adapters.connectors.filesystem import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    DEFAULT_MAX_FILE_BYTES,
)
from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.graph.schema import SCHEMA_VERSION
from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.application.answer_eval import evaluate_answers, load_refusal_queries
from memoryos.application.backfill import (
    enqueue_embedding,
    find_extraction_targets,
    find_unembedded,
    gather_stats,
)
from memoryos.application.citations import ExplainedHit, explain_hits
from memoryos.application.doctor import GraphStatus, run_doctor
from memoryos.application.entity_stats import gather_entity_stats
from memoryos.application.evaluate import (
    compare as compare_runs,
)
from memoryos.application.evaluate import (
    evaluate,
    format_report,
    format_stability,
    format_verbose,
)
from memoryos.application.evaluation import format_table, measure_recall
from memoryos.application.extraction import ExtractEntities, ExtractReport
from memoryos.application.golden import load_golden_set
from memoryos.application.importance import recompute_importance
from memoryos.application.judgements import export_golden_set
from memoryos.application.ports import ScoreBreakdown, SearchFilters
from memoryos.application.rechunk import enqueue_rechunk, find_stale
from memoryos.application.replay import PartialShadowReplay, ReplayScope, ReplayStage
from memoryos.application.tuning import (
    COARSE,
    FINE,
    RESOLUTION_FLOOR,
    baseline_row,
    collect_candidates,
    format_grid,
    score_grid,
)
from memoryos.application.verification import compare, snapshot
from memoryos.application.verify_citations import verify_citations
from memoryos.application.worker import Worker, WorkerConfig
from memoryos.config import Settings, get_settings
from memoryos.container import Container
from memoryos.domain.backoff import compute_backoff
from memoryos.domain.entities import Source
from memoryos.domain.fusion import DEFAULT_RRF_K
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import PermanentError, TransientError
from memoryos.domain.values import DEFAULT_SEARCH_MODE, SearchMode, SourceKind
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
    rerank: bool,
    explain: bool,
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
            query, k=k, filters=filters, exact=exact, mode=mode, rerank=rerank
        )

        if mode is SearchMode.KEYWORD:
            described = "keyword (ts_rank_cd)"
        elif mode is SearchMode.HYBRID:
            described = f"hybrid (rrf k={DEFAULT_RRF_K})"
        else:
            described = "exact" if exact else f"ann (ef_search={settings.hnsw_ef_search})"
        print(f'query: {result.query!r}   [{described}]')
        print(
            f"timing: embed {result.timing.embed_ms}ms  "
            f"search {result.timing.search_ms}ms  "
            f"rerank {result.timing.rerank_ms}ms  "
            f"total {result.timing.total_ms}ms\n"
        )
        explanations = None
        if explain:
            explanations = await explain_hits(
                container.database.session_factory,
                result.hits,
                weights=container.weights(),
                rrf_k=DEFAULT_RRF_K,
            )

        if not result.hits:
            print("no results")
        for rank, hit in enumerate(result.hits, start=1):
            best = max(hit.matched_chunks, key=lambda chunk: chunk.score)
            excerpt = " ".join(best.text.split())[:160]
            print(f"{rank}. {hit.score:.4f}  {hit.external_key}")
            print(f"     chunks matched: {len(hit.matched_chunks)}  best: #{best.ordinal}")
            if explanations is not None:
                _print_explanation(explanations[rank - 1])
            # Where the score came from. Under hybrid the fused number is
            # meaningless on its own — 0.0328 says nothing until you know it is
            # rank 2 in one retriever and rank 4 in the other.
            if best.breakdown is not None:
                print(f"     {_describe(best.breakdown)}")
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


def _print_explanation(explained: ExplainedHit) -> None:
    """The breakdown for one result: why it ranked, and what it quotes."""
    explanation = explained.explanation
    print(f"     {explanation.why}")
    for item in explanation.contributions:
        bar = "#" * max(1, round(item.share * 20))
        score = "     -" if item.score is None else f"{item.score:>6.3f}"
        print(
            f"       {item.name:<11} rank {item.rank:>3}  score {score}  "
            f"w={item.weight:<4g} {item.share:>5.1%} {bar}"
        )
    if explanation.rerank_score is not None:
        print(f"       {'reranked':<11} score {explanation.rerank_score:>6.3f}")

    for citation in explained.citations[:2]:
        print(f"     cite {citation.locator}")
        if citation.definition:
            print(f"          in {citation.definition}()")
        context = citation.context
        if context is not None:
            before = " ".join(context.text[: context.span_start].split())[-60:]
            span = " ".join(context.span.split())[:110]
            after = " ".join(context.text[context.span_end :].split())[:40]
            print(f"          …{before} [[{span}]] {after}…")


def _describe(breakdown: ScoreBreakdown) -> str:
    """One line of provenance: which retriever ranked this where."""
    parts = []
    if breakdown.vector_rank is not None:
        parts.append(f"vector #{breakdown.vector_rank} ({breakdown.vector_score:.4f})")
    if breakdown.keyword_rank is not None:
        parts.append(f"keyword #{breakdown.keyword_rank} ({breakdown.keyword_score:.4f})")
    return "from: " + ("  ".join(parts) if parts else "nothing")


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
    repeat: int,
    rerank: bool,
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

        if repeat > 1:
            runs = [
                await evaluate(
                    golden,
                    container.search(),
                    sessions,
                    k=k,
                    now=datetime.now(UTC),
                    mode=mode,
                    rerank=rerank,
                )
                for _ in range(repeat)
            ]
            print(format_report(runs[-1], worst=worst))
            print()
            print(format_stability(runs))
            return 0

        run = await evaluate(
            golden,
            container.search(),
            sessions,
            k=k,
            now=datetime.now(UTC),
            mode=mode,
            rerank=rerank,
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


async def run_ask(
    settings: Settings, *, question: str, k: int, show_context: bool
) -> int:
    """Answer in prose, grounded in retrieved memories.

    Exits non-zero when the answer is not fully grounded — an ungrounded answer
    is a defect even when it reads well, and a script piping this somewhere
    should be able to tell without parsing prose.
    """
    container = Container.build(settings)
    try:
        try:
            result = await container.answer()(question, k=k)
        except MissingApiKey as exc:
            print(str(exc))
            return 2
        except (TransientError, PermanentError) as exc:
            print(f"the language model could not answer: {exc}")
            return 1

        verification = result.verification
        print(f"Q: {result.question}\n")
        print(verification.marked() if not verification.grounded else result.answer)
        print()

        if result.citations:
            print("citations")
            for explained in result.citations:
                for citation in explained.citations[:1]:
                    print(f"  {citation.locator}")
                    if citation.definition:
                        print(f"      in {citation.definition}()")
                    print(f"      {' '.join(citation.excerpt.split())[:150]}")
        else:
            print("citations: none — the answer cited no passage")

        print()
        print(
            f"grounding: {verification.citation_rate:.0%} of "
            f"{verification.factual_sentences} factual sentences cited"
            + (
                f", {len(verification.hallucinated_indices)} invented citation(s) "
                f"{verification.hallucinated_indices}"
                if verification.hallucinated_indices
                else ""
            )
            + (" · refusal" if verification.is_refusal else "")
        )
        dropped = len(result.context.dropped)
        print(
            f"context:   {len(result.context.passages)} passages, "
            f"{result.context.tokens_used}/{result.context.token_budget} tokens"
            + (f", {dropped} dropped for budget" if dropped else "")
        )
        timing = result.timing.as_dict()
        print(
            "timing:    "
            + "  ".join(f"{name.removesuffix('_ms')} {value}ms" for name, value in timing.items())
        )

        if show_context:
            print("\npassages sent to the model")
            for passage in result.context.passages:
                print(f"  {passage.label}  ({passage.tokens} tokens)")
    finally:
        await container.dispose()
    return 0 if verification.grounded else 1


async def run_eval_answers(
    settings: Settings, *, golden_path: Path, refusals_path: Path, k: int, json_path: Path | None
) -> int:
    """Run every golden question and every out-of-corpus question through `/answer`.

    Exits non-zero when any out-of-corpus question was answered rather than
    declined, or when any answer cited an index it was never given. Those are
    fabrications, and a measurement that reported them at exit 0 would be a
    measurement nobody acts on.
    """
    if not golden_path.exists():
        print(f"no golden set at {golden_path}")
        return 1

    container = Container.build(settings)
    try:
        sessions = container.database.session_factory
        golden = await load_golden_set(golden_path, sessions)
        refusals = load_refusal_queries(refusals_path)
        try:
            ask = container.answer()
        except MissingApiKey as exc:
            print(str(exc))
            return 2

        print(
            f"asking {len(golden.queries)} corpus questions and "
            f"{len(refusals)} out-of-corpus questions\n"
        )
        report = await evaluate_answers(
            ask,
            questions=[query.query_text for query in golden.queries],
            refusals=refusals,
            k=k,
        )
        print(report.render())

        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
            print(f"\nwrote {json_path}")
    finally:
        await container.dispose()

    return 0 if report.refusal_rate == 1.0 and report.hallucinated_rate == 0.0 else 1


async def run_verify_citations(
    settings: Settings, *, golden_path: Path, k: int, everything: bool
) -> int:
    """Assert every citation points at the text it claims to.

    Non-zero on any mismatch, and that is the point: this is the standing check
    that would have caught M1.4a's offset bug on the day it landed instead of a
    milestone and a half later.
    """
    container = Container.build(settings)
    try:
        sessions = container.database.session_factory
        memory_ids: list[UUID] | None = None
        scope = "every current chunk in the corpus"

        if not everything:
            if not golden_path.exists():
                print(f"no golden set at {golden_path}; use --all to sweep the corpus")
                return 1
            golden = await load_golden_set(golden_path, sessions)
            search = container.search()
            found: set[UUID] = set()
            for query in golden.queries:
                result = await search(query.query_text, k=k)
                found.update(hit.memory_id for hit in result.hits)
            memory_ids = sorted(found, key=str)
            scope = (
                f"the {len(memory_ids)} memories retrieved by "
                f"{len(golden.queries)} golden queries"
            )

        print(f"checking {scope}\n")
        report = await verify_citations(sessions, memory_ids=memory_ids)
        print(report.render())
        print()
        print(
            "every citation points at the text it claims"
            if report.ok
            else "MISMATCH: at least one citation would quote the wrong text"
        )
    finally:
        await container.dispose()
    return 0 if report.ok else 1


async def run_recompute_importance(settings: Settings) -> int:
    """Fill the importance column from observable evidence.

    Deliberately a command rather than a step in the pipeline: two of its three
    inputs are properties of an item's history, so computing it at ingest would
    freeze them at first sight and never correct them.
    """
    container = Container.build(settings)
    try:
        report = await recompute_importance(
            container.database.session_factory, now=datetime.now(UTC)
        )
        print(json.dumps(report.as_dict(), indent=2))
        print(
            "\na proxy over chunk count, revision count and edit freshness — not a "
            "judgement about what matters, which is Phase 5"
        )
    finally:
        await container.dispose()
    return 0


async def run_tune_weights(
    settings: Settings, *, golden_path: Path, k: int, grid_name: str, floor: float, top: int
) -> int:
    """Grid-search fusion weights against the golden set."""
    if not golden_path.exists():
        print(f"no golden set at {golden_path}")
        return 1

    container = Container.build(settings)
    try:
        sessions = container.database.session_factory
        golden = await load_golden_set(golden_path, sessions)
        if not golden.queries:
            print("no scoreable queries in the golden set")
            return 1

        grid = COARSE if grid_name == "coarse" else FINE
        combinations = len(grid["recency"]) * len(grid["importance"])
        print(
            f"queries: {len(golden.queries)}   k={k}   grid={grid_name} "
            f"({combinations} combinations)\n"
        )

        candidates = await collect_candidates(golden, container.search(), sessions, k=k)
        rows = score_grid(
            golden, candidates, k=k, grid=grid, base=container.weights()
        )
        print(format_grid(rows, baseline=baseline_row(rows), floor=floor, top=top))
    finally:
        await container.dispose()
    return 0


def print_graph_status(status: GraphStatus, uri: str) -> None:
    """The graph section of `doctor`.

    "degraded" rather than "FAIL" when unreachable, and the word is chosen to
    match what `/health/ready` returns for the same condition. A reader who sees
    one and then the other should not have to work out whether they mean the
    same thing.
    """
    if not status.reachable:
        print(f"[degr] graph: unreachable at {uri}")
        print(f"        {status.error}")
        print("        retrieval and answering are unaffected; the graph is a projection")
        return

    if status.error is not None:
        print(f"[FAIL] graph: reachable at {uri} but not queryable")
        print(f"        {status.error}")
        return

    applied = "not applied" if status.schema_version is None else str(status.schema_version)
    mark = "ok  " if status.current else "FAIL"
    print(f"[{mark}] graph schema: {applied} (code expects {status.expected_version})")
    if not status.current:
        print("        the database was written under a different set of constraints")
    total = sum(status.counts.values())
    print(f"[ok  ] graph nodes: {total}")
    for label, count in sorted(status.counts.items()):
        print(f"        {label}: {count}")


async def run_doctor_command(settings: Settings) -> int:
    container = Container.build(settings)
    try:
        report = await run_doctor(
            container.database.session_factory,
            container.embedder,
            graph=container.graph,
            graph_schema_version=SCHEMA_VERSION,
        )
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
        if report.graph is not None:
            print()
            print_graph_status(report.graph, container.settings.neo4j_uri)
        print()
        print("healthy" if report.healthy else "problems found")
    finally:
        await container.dispose()
    return 0 if report.healthy else 1


async def run_extract_entities(
    settings: Settings, *, source: str | None, limit: int | None, dry_run: bool
) -> int:
    """Extract entities across the corpus, memory by memory.

    Sequential rather than concurrent, deliberately. The free tier's limit is
    requests per minute, so parallelism buys nothing but 429s — and the worker's
    backoff would then serialise them anyway, with the retries costing real
    requests against the daily cap.
    """
    container = Container.build(settings)
    try:
        targets = await find_extraction_targets(
            container.database.session_factory,
            extractor_version=_extractor_version(container),
            source=source,
            limit=limit,
        )

        print(f"extractor: {_extractor_version(container)}")
        print(f"provider:  {settings.llm_provider}")
        print(f"memories pending: {len(targets)}")
        print(f"chunks pending:   {sum(target.chunks for target in targets)}")
        for target in targets[:20]:
            print(f"  {target.external_key} ({target.chunks} chunks)")
        if len(targets) > 20:
            print(f"  ... and {len(targets) - 20} more")

        if dry_run:
            print("\ndry run; no model calls made, nothing written")
            return 0
        if not targets:
            print("\nnothing to do")
            return 0

        extractor = container.extractor()
        extract = ExtractEntities(
            container.database.session_factory, extractor, container.graph
        )

        started = time.monotonic()
        entities = mentions = failed = 0
        for index, target in enumerate(targets, start=1):
            try:
                report = await _extract_with_backoff(extract, target.memory_id)
            except (TransientError, PermanentError) as exc:
                # Reported and stepped over rather than aborting the run. One
                # memory the model refuses to process must not cost the corpus
                # every extraction after it.
                failed += 1
                print(f"[{index}/{len(targets)}] FAILED {target.external_key}: {exc}")
                continue
            entities += report.entities
            mentions += report.mentions
            print(
                f"[{index}/{len(targets)}] {target.external_key}: "
                f"{report.mentions} mentions, {report.entities} new entities"
                + ("" if report.projected else "  (not projected)")
            )

        elapsed = time.monotonic() - started
        stats = extractor.stats
        print(
            f"\nnew entities {entities}   mentions {mentions}   failed {failed}\n"
            f"api calls    {stats.calls} ({stats.retries} retries)\n"
            f"dropped      {stats.dropped_not_found} not in text, "
            f"{stats.dropped_low_confidence} low confidence, "
            f"{stats.dropped_bad_type} bad type, of {stats.returned} returned\n"
            f"prompt chars {stats.prompt_chars}   response chars {stats.response_chars}\n"
            f"wall clock   {elapsed:.1f}s"
        )
    finally:
        await container.dispose()
    return 1 if failed else 0


def _extractor_version(container: Container) -> str:
    return container.extractor().version


# How many times the command waits out a rate limit before giving up on a
# memory. Five with exponential backoff is roughly ten minutes of patience,
# which is longer than any free-tier window this has met.
EXTRACT_MAX_ATTEMPTS = 5


async def _extract_with_backoff(
    extract: ExtractEntities, memory_id: UUID
) -> ExtractReport:
    """Run one extraction, waiting out rate limits.

    The worker has backoff and this command does not go through the worker, so
    without this a single 429 marks a memory failed and moves on — and on a free
    tier a 429 is the expected steady state, not an exception. A corpus run
    would end with a scatter of missing memories and an exit code of 1, all of
    which a ten-second wait would have avoided.

    `PermanentError` is deliberately not retried here: the adapter has already
    made that judgement, including its own one retry for malformed JSON.
    """
    for attempt in range(EXTRACT_MAX_ATTEMPTS):
        try:
            return await extract(memory_id)
        except TransientError as exc:
            if attempt == EXTRACT_MAX_ATTEMPTS - 1:
                raise
            delay = compute_backoff(attempt)
            print(f"    rate limited ({exc}); waiting {delay:.0f}s")
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def run_entity_stats(settings: Settings, *, top: int) -> int:
    container = Container.build(settings)
    try:
        stats = await gather_entity_stats(container.database.session_factory, top=top)
        print(f"entities  {stats.entities}")
        print(f"mentions  {stats.mentions}\n")
        for entity_type, count in stats.by_type.items():
            print(f"  {entity_type:14} {count}")
        print(f"\ntop {len(stats.top)} by mentions")
        for name, entity_type, count in stats.top:
            print(f"  {count:5}  {entity_type:14} {name}")

        print(
            f"\nduplicate groups {len(stats.duplicate_groups)}   "
            f"surplus rows {stats.duplicate_surplus} "
            f"({stats.duplicate_rate:.1%} of entities)"
        )
        for group in stats.duplicate_groups[:20]:
            print(f"  {group.mentions:5}  {group.key}: {', '.join(group.names)}")
        for version, count in stats.extractor_versions.items():
            print(f"\nextractor {version}: {count} mentions")
    finally:
        await container.dispose()
    return 0


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
    # An `eval_exclude` already in the target file wins over the defaults. The
    # list is meant to be edited next to the data it applies to, and an export
    # that silently reverted a hand-tuned exclusion would quietly change what
    # every later measurement means.
    existing: list[str] | None = None
    if output.exists():
        previous = json.loads(output.read_text())
        if isinstance(previous.get("eval_exclude"), list):
            existing = [str(pattern) for pattern in previous["eval_exclude"]]

    container = Container.build(settings)
    try:
        golden = await export_golden_set(
            container.database.session_factory,
            now=datetime.now(UTC),
            eval_exclude=existing,
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

    extract_parser = commands.add_parser(
        "extract-entities", help="extract entities from memories using the configured LLM"
    )
    extract_parser.add_argument("--source", help="limit to one source by name")
    extract_parser.add_argument(
        "--limit", type=int, help="stop after this many memories"
    )
    extract_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be extracted without calling the model",
    )

    entity_stats_parser = commands.add_parser(
        "entity-stats", help="report entities, mentions, and the duplicate problem"
    )
    entity_stats_parser.add_argument(
        "--top", type=int, default=20, help="how many entities to list by mention count"
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
        "--explain",
        action="store_true",
        help="print citations and why each result ranked where it did",
    )
    search.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="skip the cross-encoder and return the fused ordering",
    )
    search.add_argument(
        "--mode",
        choices=[value.value for value in SearchMode],
        default=DEFAULT_SEARCH_MODE.value,
        help=(
            "which retriever answers: 'hybrid' fuses both with RRF, 'vector' embeds "
            "the query, 'keyword' matches terms"
        ),
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

    ask = commands.add_parser(
        "ask", help="answer a question in prose, grounded in retrieved memories"
    )
    ask.add_argument("question")
    ask.add_argument("-k", "--k", dest="k", type=int, default=10)
    ask.add_argument(
        "--show-context",
        action="store_true",
        help="list the passages that were sent to the model",
    )

    eval_answers = commands.add_parser(
        "eval-answers",
        help="ask every golden question plus out-of-corpus ones, and report grounding",
    )
    eval_answers.add_argument("-k", "--k", dest="k", type=int, default=10)
    eval_answers.add_argument(
        "--golden", type=Path, default=Path("var/golden-set.json")
    )
    eval_answers.add_argument(
        "--refusals",
        type=Path,
        default=Path("var/refusal-queries.json"),
        help="questions the corpus cannot answer; the system must decline them",
    )
    eval_answers.add_argument("--json", dest="json_path", type=Path)

    verify_citations_command = commands.add_parser(
        "verify-citations",
        help="assert every chunk's offsets point at the text the chunk claims",
    )
    verify_citations_command.add_argument(
        "--all",
        dest="everything",
        action="store_true",
        help=(
            "sweep every current chunk rather than only those the golden "
            "queries retrieve"
        ),
    )
    verify_citations_command.add_argument(
        "--golden",
        type=Path,
        default=Path("var/golden-set.json"),
        help="the golden set whose results are checked",
    )
    verify_citations_command.add_argument("-k", "--k", dest="k", type=int, default=10)

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
        default=DEFAULT_SEARCH_MODE.value,
        help="which retriever to score; the same golden set judges all three",
    )
    commands.add_parser(
        "recompute-importance",
        help="score every memory's importance from chunk count, revisions and freshness",
    )

    tune = commands.add_parser(
        "tune-weights", help="grid-search fusion weights against the golden set"
    )
    tune.add_argument("-k", "--k", dest="k", type=int, default=10)
    tune.add_argument(
        "--golden", type=Path, default=Path("var/golden-set.json"), help="the export to score"
    )
    tune.add_argument("--grid", choices=["coarse", "fine"], default="coarse")
    tune.add_argument("--top", type=int, default=5, help="how many combinations to list")
    tune.add_argument(
        "--floor",
        type=float,
        default=RESOLUTION_FLOOR,
        help=(
            "the smallest difference this harness can detect, from M2.3a. Gains "
            "below it are reported as noise rather than as results"
        ),
    )

    evaluate_command.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="score the fused ordering without the cross-encoder, to measure its effect",
    )
    evaluate_command.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "run the evaluation N times and report each metric's standard "
            "deviation; the floor below which a difference is not evidence"
        ),
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

    if args.command == "extract-entities":
        return asyncio.run(
            run_extract_entities(
                settings,
                source=args.source,
                limit=args.limit,
                dry_run=args.dry_run,
            )
        )

    if args.command == "entity-stats":
        return asyncio.run(run_entity_stats(settings, top=args.top))

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
                rerank=args.rerank,
                explain=args.explain,
            )
        )

    if args.command == "eval-recall":
        values = [int(value) for value in args.ef_search.split(",") if value.strip()]
        return asyncio.run(
            run_eval_recall(settings, queries=args.queries, k=args.k, ef_search_values=values)
        )

    if args.command == "ask":
        return asyncio.run(
            run_ask(
                settings,
                question=args.question,
                k=args.k,
                show_context=args.show_context,
            )
        )

    if args.command == "eval-answers":
        return asyncio.run(
            run_eval_answers(
                settings,
                golden_path=args.golden,
                refusals_path=args.refusals,
                k=args.k,
                json_path=args.json_path,
            )
        )

    if args.command == "verify-citations":
        return asyncio.run(
            run_verify_citations(
                settings, golden_path=args.golden, k=args.k, everything=args.everything
            )
        )

    if args.command == "recompute-importance":
        return asyncio.run(run_recompute_importance(settings))

    if args.command == "tune-weights":
        return asyncio.run(
            run_tune_weights(
                settings,
                golden_path=args.golden,
                k=args.k,
                grid_name=args.grid,
                floor=args.floor,
                top=args.top,
            )
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
                repeat=args.repeat,
                rerank=args.rerank,
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
