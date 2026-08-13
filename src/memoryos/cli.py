"""Command line entry point.

argparse, deliberately. The moment a CLI framework is in the tree, every later
command gets written in it; the commands here do not need one.
"""

import argparse
import asyncio
import json
import math
import textwrap
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.connectors.filesystem import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    DEFAULT_MAX_FILE_BYTES,
)
from memoryos.adapters.db import models
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.extraction.llm import ExtractionStats
from memoryos.adapters.graph.schema import SCHEMA_VERSION
from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.application import (
    assumption_groups,
    assumption_suggest,
    assumptions,
    decision_suggest,
    decisions,
    events,
    evolution,
    graph_projection,
    graph_sync,
    graph_verify,
    outcome_suggest,
    outcomes,
    patterns,
    reflections,
    temporal,
)
from memoryos.application.answer_eval import evaluate_answers, load_refusal_queries
from memoryos.application.backfill import (
    enqueue_embedding,
    find_extraction_targets,
    find_relationship_targets,
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
from memoryos.application.merge_admin import find_entity, list_merges
from memoryos.application.ports import ScoreBreakdown, SearchFilters
from memoryos.application.rechunk import enqueue_rechunk, find_stale
from memoryos.application.relationships import ExtractRelationships, RelationshipReport
from memoryos.application.replay import PartialShadowReplay, ReplayScope, ReplayStage
from memoryos.application.resolution import DEFAULT_THRESHOLD, MergeCandidate
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
from memoryos.domain.backoff import wait_for
from memoryos.domain.entities import Source
from memoryos.domain.events import Event, EventKind
from memoryos.domain.fusion import DEFAULT_RRF_K
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobStatus, JobType, PermanentError, TransientError
from memoryos.domain.patterns import (
    DEFAULT_MIN_SUPPORT,
    REFLECTION_MIN_CONFIDENCE,
)
from memoryos.domain.values import (
    DEFAULT_SEARCH_MODE,
    AssumptionVerdict,
    DecisionStatus,
    EvidenceKind,
    EvidenceRelation,
    MergeStatus,
    MergeStrategy,
    OutcomeVerdict,
    PatternKind,
    PatternRelation,
    Period,
    SearchMode,
    SourceKind,
    SuggestionStatus,
    TimeProvenance,
)
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

        # Printed once, above the results, and only when something fired. A
        # query reinterpreted as temporal changes what comes back; if the reader
        # cannot see the interpretation, an empty result set looks like an empty
        # corpus. Read off the result rather than off a hit, so it survives the
        # case where a filter left nothing to read it from.
        if result.temporal_intent is not None:
            applied = (
                "  [hard filter applied]"
                if result.temporal_filter_applied
                else "  [ranking only]"
            )
            print(f"read as temporal: {result.temporal_intent}{applied}\n")

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
    """One line of provenance: which ranking placed this where, and how.

    The graph line carries its route, and that is not a nicety. Expansion is the
    one ranking that introduces a result rather than reordering one, so a
    graph-promoted hit may share no word with the query — and "the graph found it"
    is not something a reader can check. `queue -> SKIP LOCKED -> worker` is.
    """
    parts = []
    if breakdown.vector_rank is not None:
        parts.append(f"vector #{breakdown.vector_rank} ({breakdown.vector_score:.4f})")
    if breakdown.keyword_rank is not None:
        parts.append(f"keyword #{breakdown.keyword_rank} ({breakdown.keyword_score:.4f})")
    if breakdown.graph_rank is not None:
        route = " -> ".join(breakdown.graph_path or ())
        parts.append(f"graph #{breakdown.graph_rank} via {route or 'an unnamed route'}")
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
        # Every axis, not two of them. M3.5 added a third and this line kept
        # reporting the size of the grid it used to be — a wrong number in the one
        # place a reader looks to know how much of the space was searched.
        combinations = math.prod(len(values) for values in grid.values())
        print(
            f"queries: {len(golden.queries)}   k={k}   grid={grid_name} "
            f"({combinations} combinations)\n"
        )

        candidates = await collect_candidates(golden, container.search(), sessions, k=k)
        # Before the grid, because the grid cannot be read without it: a graph row
        # that scores the same as the baseline means one of two very different
        # things, and only expansion coverage says which.
        reached = [found for found in candidates if found.graph and found.graph.chunks]
        introduced = sum(len(found.graph.chunks) for found in reached if found.graph)
        print(
            f"graph expansion: candidates for {len(reached)}/{len(candidates)} "
            f"queries, {introduced} chunks in total"
        )
        if not reached:
            print(
                "  nothing to fuse — every graph row below will equal the baseline. "
                "Check extraction coverage with `memoryos entity-stats`.\n"
            )
        else:
            print()

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
            # Three marks, not two. An advisory with a non-zero count is neither
            # a pass nor a failure: it names a capability nobody has exercised,
            # and printing `ok` beside a corpus with no entity extraction at all
            # is how that state stayed invisible through two full replays.
            if finding.advisory and finding.count:
                mark = "note"
            elif finding.healthy:
                mark = "ok  "
            else:
                mark = "FAIL"
            print(f"[{mark}] {finding.check}: {finding.count}")
            if mark != "ok  ":
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
            container.database.session_factory, extractor, container.queue
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
                + ("" if report.sync_enqueued else "  (no graph sync queued)")
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
            # The provider's own number when it gave one. See `backoff.wait_for`.
            delay = wait_for(exc, attempt)
            print(f"    rate limited ({exc}); waiting {delay:.0f}s")
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def run_extract_relationships(
    settings: Settings, *, source: str | None, limit: int | None, dry_run: bool
) -> int:
    """Extract relationships across the corpus, memory by memory.

    Sequential for the reason entity extraction is: the free tier limits
    requests per minute, so parallelism buys 429s that the backoff then
    serialises anyway — at the cost of the retries.
    """
    container = Container.build(settings)
    try:
        extract = container.relationships()
        targets = await find_relationship_targets(
            container.database.session_factory,
            extractor_version=extract.version,
            source=source,
            limit=limit,
        )

        print(f"extractor: {extract.version}")
        print(f"provider:  {settings.llm_provider}")
        print(f"memories pending: {len(targets)}")
        for target in targets[:20]:
            print(f"  {target.external_key} ({target.chunks} chunks with entities)")
        if len(targets) > 20:
            print(f"  ... and {len(targets) - 20} more")

        if dry_run:
            print("\ndry run; no model calls made, nothing written")
            return 0
        if not targets:
            print("\nnothing to do")
            return 0

        started = time.monotonic()
        stored = queued = failed = 0
        for index, target in enumerate(targets, start=1):
            try:
                report = await _relationships_with_backoff(extract, target.memory_id)
            except (TransientError, PermanentError) as exc:
                failed += 1
                print(f"[{index}/{len(targets)}] FAILED {target.external_key}: {exc}")
                continue
            stored += report.relationships
            queued += int(report.sync_enqueued)
            print(
                f"[{index}/{len(targets)}] {target.external_key}: "
                f"{report.relationships} relationships "
                f"from {report.chunks_considered} chunks"
            )

        elapsed = time.monotonic() - started
        stats = container_stats(extract)
        print(
            f"\nrelationships {stored}   graph syncs queued {queued}   failed {failed}\n"
            f"api calls     {stats.calls} ({stats.retries} retries)\n"
            f"dropped       {stats.dropped_unknown_entity} unknown entity, "
            f"{stats.dropped_low_confidence} low confidence, "
            f"{stats.dropped_bad_type} bad predicate, "
            f"{stats.dropped_self} self-referential, of {stats.returned} returned\n"
            f"prompt chars  {stats.prompt_chars}   response chars {stats.response_chars}\n"
            f"wall clock    {elapsed:.1f}s"
        )
    finally:
        await container.dispose()
    return 1 if failed else 0


def container_stats(extract: ExtractRelationships) -> ExtractionStats:
    """The adapter's counters, or empty ones if it keeps none."""
    stats = getattr(extract, "_extractor", None)
    return getattr(stats, "stats", ExtractionStats())


async def _relationships_with_backoff(
    extract: ExtractRelationships, memory_id: UUID
) -> RelationshipReport:
    """One extraction, waiting out rate limits. See `_extract_with_backoff`."""
    for attempt in range(EXTRACT_MAX_ATTEMPTS):
        try:
            return await extract(memory_id)
        except TransientError as exc:
            if attempt == EXTRACT_MAX_ATTEMPTS - 1:
                raise
            # The provider's own number when it gave one. See `backoff.wait_for`.
            delay = wait_for(exc, attempt)
            print(f"    rate limited ({exc}); waiting {delay:.0f}s")
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def run_resolve_entities(
    settings: Settings, *, dry_run: bool, threshold: float, limit: int
) -> int:
    container = Container.build(settings)
    try:
        resolve = container.resolve(threshold=threshold)

        candidates = await resolve.propose()
        auto = [c for c in candidates if resolve.would_auto_merge(c)]
        review = [c for c in candidates if not resolve.would_auto_merge(c)]

        print(f"threshold {threshold}   candidates {len(candidates)}")
        print(f"would auto-merge {len(auto)}   would queue {len(review)}\n")
        for candidate in candidates[:limit]:
            mark = "MERGE " if resolve.would_auto_merge(candidate) else "review"
            print(
                f"  [{mark}] {candidate.confidence:.3f} "
                f"{candidate.strategy.value:9} {candidate.evidence}"
            )
        if len(candidates) > limit:
            print(f"  ... and {len(candidates) - limit} more")

        if dry_run:
            print("\ndry run; nothing merged, nothing queued")
            return 0

        report = await resolve()
        print(
            f"\nauto-merged {report.auto_merged}   queued {report.pending}   "
            f"already queued {report.already_pending}\n"
            f"mentions moved {report.mentions_moved}   "
            f"took {report.duration_ms}ms"
        )
        for strategy, count in sorted(report.by_strategy.items()):
            print(f"  {strategy:10} {count} candidates")

        # Queued, not written. Every merge enqueued a `SYNC_GRAPH` job naming its
        # winner and loser as it was applied, so the projection catches up when a
        # worker runs — and the loser's node is pruned rather than left behind,
        # which an upsert-only update could never do.
        #
        # M3.2 rebuilt the whole projection here instead, which was correct and
        # also a use case writing to Neo4j. `graph rebuild` is still the answer to
        # divergence; it is no longer the answer to a merge.
        pending = await _pending_graph_syncs(container.database.session_factory)
        print(f"  queued {pending} graph syncs; run `memoryos worker --drain`")
    finally:
        await container.dispose()
    return 0


async def _pending_graph_syncs(sessions: async_sessionmaker[AsyncSession]) -> int:
    """How many projection updates are waiting for a worker."""
    async with sessions() as session:
        return int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(models.Job)
                    .where(
                        models.Job.job_type == JobType.SYNC_GRAPH.value,
                        models.Job.status.in_(
                            [JobStatus.PENDING.value, JobStatus.RUNNING.value]
                        ),
                    )
                )
            ).scalar_one()
        )


async def run_list_merges(
    settings: Settings, *, pending: bool, strategy: str | None, limit: int
) -> int:
    container = Container.build(settings)
    try:
        rows = await list_merges(
            container.database.session_factory,
            status=MergeStatus.PENDING if pending else None,
            strategy=strategy,
            limit=limit,
        )
        if not rows:
            print("no merges" + (" pending review" if pending else ""))
            return 0
        for row in rows:
            print(
                f"{row.id}  {row.status:8} {row.strategy:9} {row.confidence:.3f}  "
                f"{row.loser_name!r} -> {row.winner_name!r} ({row.entity_type})"
            )
            print(f"    {row.evidence}")
    finally:
        await container.dispose()
    return 0


async def run_manual_merge(settings: Settings, *, winner: str, loser: str) -> int:
    """Merge two entities by hand. The reviewer's verdict on a pending pair.

    Recorded as `manual` with confidence 1.0, because a person looked at it.
    That outranks every automatic strategy and is the point of having a queue.
    """
    container = Container.build(settings)
    try:
        try:
            winner_id = await find_entity(container.database.session_factory, winner)
            loser_id = await find_entity(container.database.session_factory, loser)
        except LookupError as exc:
            print(str(exc))
            return 1

        resolve = container.resolve()
        moved = await resolve.apply(
            MergeCandidate(
                left_id=winner_id,
                right_id=loser_id,
                strategy=MergeStrategy.MANUAL,
                confidence=1.0,
                evidence=f"merged by hand: {loser!r} into {winner!r}",
            )
        )
        if moved is None:
            print("nothing to do: they are already the same entity")
            return 0
        print(f"merged; {moved} mentions moved")
        print("queued a graph sync; run `memoryos worker --drain`")
    finally:
        await container.dispose()
    return 0


async def run_unmerge(settings: Settings, *, merge_id: str) -> int:
    container = Container.build(settings)
    try:
        resolve = container.resolve()
        try:
            restored = await resolve.revert(UUID(merge_id))
        except (LookupError, ValueError) as exc:
            print(str(exc))
            return 1
        print(f"reverted; {restored} mentions restored")
        print("queued a graph sync; run `memoryos worker --drain`")
    finally:
        await container.dispose()
    return 0


async def run_graph_rebuild(settings: Settings, *, dry_run: bool) -> int:
    """Clear the projection and build it again from Postgres.

    **There is no shadow swap, and it is not for want of trying.** The Postgres
    equivalent builds the replacement in a second schema and moves the tables in
    one transaction, so the live corpus is never unavailable. Neo4j Community
    Edition offers nowhere to build the replacement: it supports exactly one user
    database, and `CREATE DATABASE`, `CREATE ALIAS` and `CREATE COMPOSITE DATABASE`
    are all refused outright — "Unsupported administration command", by edition
    rather than by permissions. Nor can it be faked inside the one database with a
    parallel set of labels: Cypher has no operation that renames a label or a
    relationship type, so the "swap" would be a write over every node and edge,
    which is the rebuild again with an extra copy of the data.

    So this accepts downtime, and says how much: the graph is empty between the
    clear and the last write. On this corpus that is under a second. It grows with
    the corpus, which is the honest argument for the incremental sync existing at
    all — see `application/graph_sync.py`.

    `--dry-run` reads the projection and prints what it would write, touching
    nothing. That is genuinely useful before a rebuild rather than a courtesy: the
    counts it prints are what the graph *owes*, so a `--dry-run` beside
    `graph verify` is how you tell "the projection is behind" from "Postgres has
    less in it than you thought".
    """
    container = Container.build(settings)
    try:
        sessions = container.database.session_factory
        if dry_run:
            projection = await graph_projection.read(sessions)
            print("dry run; the graph was not touched\n")
            print(f"would project {projection.nodes} nodes, {len(projection.edges)} edges")
            for name, count in projection.counts.items():
                print(f"  {name:<12} {count}")
            return 0

        before = await container.graph.counts_by_label()
        print(f"before: {sum(before.values())} nodes  {before or '{}'}")
        print("clearing the projection — the graph answers nothing until this finishes")
        print("(Neo4j Community has one database, so there is nowhere to swap in)\n")

        started = time.monotonic()
        projection = await graph_projection.rebuild(sessions, container.graph)
        elapsed_ms = int((time.monotonic() - started) * 1000)

        after = await container.graph.counts_by_label()
        print(f"after:  {sum(after.values())} nodes")
        for name, count in projection.counts.items():
            print(f"  {name:<12} {count}")
        print(f"\n{elapsed_ms}ms, of which the graph was unavailable for all of it")
    finally:
        await container.dispose()
    return 0


async def run_graph_verify(settings: Settings) -> int:
    """Report every way the projection differs from Postgres. Non-zero on any.

    The same requirement `verify-replay` carries: a check that cannot fail is not
    a check. One of this milestone's tests corrupts a node's name — a change no
    count would see — and requires this to exit non-zero.

    Nothing is repaired here. `graph rebuild` is the repair, and keeping them
    separate is what leaves anybody able to answer how often the projection
    actually diverges.
    """
    container = Container.build(settings)
    try:
        divergence = await graph_verify.verify(
            container.database.session_factory, container.graph
        )
        print(divergence.render())
        print()
        if divergence.identical:
            print("identical: the projection matches Postgres")
            return 0
        print("DIVERGED: Postgres is the system of record — run `memoryos graph rebuild`")
        return 1
    finally:
        await container.dispose()


async def run_graph_sync(settings: Settings, *, memories: list[str], entities: list[str]) -> int:
    """Run one sync inline, with the payload a job would have carried.

    For diagnosing a neighbourhood rather than for routine use: the routine path is
    the `SYNC_GRAPH` job that extraction and resolution enqueue. Inline, because
    somebody debugging a single memory's projection should not have to start a
    worker to see what happens.
    """
    container = Container.build(settings)
    try:
        try:
            payload = graph_sync.payload_for(
                memory_ids=[UUID(value) for value in memories],
                entity_ids=[UUID(value) for value in entities],
            )
        except ValueError as exc:
            print(f"not an id: {exc}")
            return 1
        if not payload["memory_ids"] and not payload["entity_ids"]:
            print("nothing to sync: name at least one --memory or --entity")
            return 1

        report = await container.graph_sync()(payload)
        print(json.dumps(report.as_dict(), indent=2))
    except PermanentError as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    return 0


async def run_evolution(
    settings: Settings, *, source: str, path: str, summarize: bool, refresh: bool
) -> int:
    """One item's history: when, what changed, and whether it was rechunked.

    The summaries are opt-out rather than opt-in, because a history with no
    account of what changed is a list of hashes. They are cached on the version
    pair, so the second run of this command spends nothing.
    """
    container = Container.build(settings)
    sessions = container.database.session_factory
    try:
        async with sessions() as session:
            source_id = (
                await session.execute(
                    select(models.Source.id).where(models.Source.name == source)
                )
            ).scalars().first()
        if source_id is None:
            print(f"no source named {source!r}")
            return 1

        history = await evolution.version_history(sessions, source_id, path)
        if not history:
            print(f"no memory at {path!r} in source {source!r}")
            return 1

        print(f"{path}  ({len(history)} version{'s' if len(history) != 1 else ''})")
        # Stated once, here, rather than implied by a column of zeroes further
        # down. The rule is not obvious and the number is misleading without it.
        if any(not version.holds_chunks for version in history[:-1]):
            print(
                "  note: superseded versions hold no chunks — M1.4 deletes them when "
                "the next version is written, so a chunk delta against one is not\n"
                "        a measurement of chunking."
            )

        summarizer = None
        if summarize:
            try:
                summarizer = evolution.SummarizeChange(sessions, container.language_model())
            except MissingApiKey as exc:
                print(f"\n  (no change summaries: {exc})")

        for index, version in enumerate(history):
            print()
            print(
                f"  v{version.version}"
                f"{'  [current]' if version.is_current else ''}"
                f"{'  [tombstoned]' if version.deleted_at else ''}"
            )
            print(f"    occurred   {_stamp(version.occurred_at)} ({version.occurred_at_source})")
            print(f"    ingested   {_stamp(version.ingested_at)}")
            print(
                f"    content    {version.characters} chars, "
                f"{version.chunks} chunk{'s' if version.chunks != 1 else ''}"
                f"{'' if version.holds_chunks else ' (discarded)'}"
            )
            normalized = (version.normalized_hash or "-")[:8]
            print(f"    hashes     {version.content_hash[:8]} / {normalized}")
            print(f"    change     {version.summary_of_change}")

            if index == 0:
                continue

            diff = await evolution.diff_versions(sessions, history[index - 1].id, version.id)
            delta = diff.chunk_delta
            print(
                f"    diff       +{diff.added_chars} -{diff.removed_chars} chars "
                f"in {len(diff.spans)} span{'s' if len(diff.spans) != 1 else ''}, "
                f"chunk delta {delta if delta is not None else 'n/a'}"
            )
            if diff.affected_chunks:
                named = [
                    f"#{chunk.ordinal}"
                    + (f" ({chunk.definition})" if chunk.definition else "")
                    for chunk in diff.affected_chunks
                ]
                print(f"    touches    {len(named)} chunks: {', '.join(named[:8])}")
                if len(named) > 8:
                    print(f"               … and {len(named) - 8} more")

            if summarizer is not None:
                try:
                    summary = await summarizer(diff, refresh=refresh)
                except (TransientError, PermanentError) as exc:
                    print(f"    summary    (unavailable: {exc})")
                    continue
                mark = "" if summary.grounding.grounded else "  [ungrounded]"
                origin = "cached" if summary.cached else summary.model_id
                print(f"    summary    {summary.text}{mark}")
                print(f"               ({origin})")
                if summary.grounding.unsupported:
                    # Named, not hidden. These are the terms the summary used
                    # that the diff does not contain.
                    print(
                        "               not in the diff: "
                        f"{', '.join(summary.grounding.unsupported)}"
                    )
                if summary.grounding.context_only:
                    # Not an error. The place to look when a summary describes
                    # something that did not actually change.
                    print(
                        "               context only: "
                        f"{', '.join(summary.grounding.context_only)}"
                    )
    finally:
        await container.dispose()
    return 0


def _stamp(moment: datetime | None) -> str:
    return "—" if moment is None else f"{moment:%Y-%m-%d %H:%M:%S}"


def parse_moment(value: str, *, name: str) -> datetime:
    """A date or a timestamp on the command line, as an instant in UTC.

    A bare `2026-08-01` means midnight UTC that morning, not midnight wherever
    the terminal happens to be. The alternative — resolving it against the local
    zone — makes `timeline --from` return different rows on two laptops looking
    at the same corpus, and the difference is a few hours, which is exactly the
    size of error nobody notices.
    """
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"{name} must be a date or timestamp like 2026-08-01 "
            f"or 2026-08-01T14:30:00Z, got {value!r}"
        ) from None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


async def _source_id_named(
    sessions: async_sessionmaker[AsyncSession], name: str
) -> UUID | None:
    async with sessions() as session:
        return (
            await session.execute(
                select(models.Source.id).where(models.Source.name == name)
            )
        ).scalars().first()


def _print_provenance(bands: list[temporal.ProvenanceBand], total: int) -> None:
    """The assessment, above every timeline that is drawn from it."""
    print("occurred_at provenance")
    for band in bands:
        share = f"{band.count / total:6.1%}" if total else f"{'-':>6}"
        span = (
            f"{band.earliest:%Y-%m-%d} .. {band.latest:%Y-%m-%d}"
            if band.earliest is not None and band.latest is not None
            else "-"
        )
        print(f"  {band.provenance.value:11} {band.count:6}  {share}  {span}")


async def _print_backfill(
    sessions: async_sessionmaker[AsyncSession],
    total: int,
    *,
    source_id: UUID | None,
    threshold: timedelta = timedelta(days=1),
) -> None:
    """How far the corpus's world-time trails its ingestion-time.

    Printed with the provenance rather than behind its own command, because it
    answers the other half of the same question. Provenance says how the dates
    were derived; this says whether the system watched the content happen or was
    pointed at a pile of it afterwards — and a corpus that was assembled rather
    than accumulated has a timeline meaning something different from one that
    grew.
    """
    lagging = await temporal.out_of_order(sessions, threshold, source_id=source_id)
    share = f" of {total}" if total else ""
    print(
        f"\nbackfill lag over {threshold.days}d: {len(lagging)}{share} memories"
    )
    if lagging:
        # Ordered by lag descending, so the head is the largest.
        worst = lagging[0]
        assert worst.ingested_at is not None and worst.occurred_at is not None
        span = worst.ingested_at - worst.occurred_at
        hours = span.seconds // 3600
        print(f"  longest {span.days}d {hours}h  {worst.external_key}")


def _histogram(buckets: list[temporal.Bucket], *, width: int = 46) -> None:
    """A bar per period, scaled to the largest.

    Crude on purpose. The question it answers is whether the corpus has any
    shape at all, and a wrong answer to that is visible at this resolution.
    """
    peak = max((bucket.count for bucket in buckets), default=0)
    for bucket in buckets:
        bar = "#" * round(bucket.count / peak * width) if peak else ""
        print(f"  {bucket.start:%Y-%m-%d}  {bucket.count:6}  {bar}")


async def run_timeline(
    settings: Settings,
    *,
    start: str | None,
    end: str | None,
    period: Period,
    source: str | None,
) -> int:
    container = Container.build(settings)
    sessions = container.database.session_factory
    try:
        source_id = None
        if source is not None:
            source_id = await _source_id_named(sessions, source)
            if source_id is None:
                print(f"no source named {source!r}")
                return 1

        bands = await temporal.provenance_profile(sessions, source_id=source_id)
        total = sum(band.count for band in bands)
        _print_provenance(bands, total)
        await _print_backfill(sessions, total, source_id=source_id)

        observed = temporal.observed_bounds(bands)
        if observed is None:
            print("\nnothing dated: no timeline to draw")
            return 0

        # The window defaults to what the corpus actually covers, so that the
        # first run of this command needs no arguments and cannot silently draw
        # an empty chart because the default window missed the data.
        window_start = parse_moment(start, name="--from") if start else observed[0]
        window_end = (
            parse_moment(end, name="--to")
            if end
            else temporal.advance(observed[1], period)
        )
        if window_end <= window_start:
            print(f"empty window: --from {window_start} is not before --to {window_end}")
            return 1

        buckets = await temporal.activity_by_period(
            sessions, window_start, window_end, period=period, source_id=source_id
        )
        counted = sum(bucket.count for bucket in buckets)
        print(
            f"\n{counted} memories by {period.value}, "
            f"{window_start:%Y-%m-%d} to {window_end:%Y-%m-%d} "
            f"({len(buckets)} periods, {sum(1 for b in buckets if b.count)} non-empty)"
        )
        _histogram(buckets)
    except ValueError as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    return 0


async def run_gaps(settings: Settings, *, min_days: float, source: str | None) -> int:
    container = Container.build(settings)
    sessions = container.database.session_factory
    try:
        async with sessions() as session:
            rows = (
                await session.execute(
                    select(models.Source.id, models.Source.name).order_by(
                        models.Source.name
                    )
                )
            ).all()
        if source is not None:
            rows = [row for row in rows if row[1] == source]
            if not rows:
                print(f"no source named {source!r}")
                return 1

        min_gap = timedelta(days=min_days)
        found = 0
        for source_id, name in rows:
            gaps = await temporal.find_gaps(
                sessions, temporal.SourceScope(source_id), min_gap=min_gap
            )
            found += len(gaps)
            print(f"{name}: {len(gaps)} gaps of {min_days:g} days or more")
            for gap in gaps:
                days = gap.duration.total_seconds() / 86400
                print(
                    f"  {gap.start:%Y-%m-%d %H:%M} -> {gap.end:%Y-%m-%d %H:%M}  "
                    f"{days:.1f}d"
                )
                print(f"    before: {gap.before.external_key}")
                print(f"    after:  {gap.after.external_key}")
        if not found:
            # The absence is the result, and it has two readings that a bare
            # empty list would not distinguish for the reader.
            print("\nno gaps: either the activity is continuous, or it is too "
                  "short a span to contain one")
    finally:
        await container.dispose()
    return 0


async def run_as_of(settings: Settings, *, moment: str) -> int:
    container = Container.build(settings)
    try:
        query_time = parse_moment(moment, name="DATE")
        view = await temporal.as_of(container.database.session_factory, query_time)
        print(f"as of {view.query_time:%Y-%m-%d %H:%M:%S %Z}")
        print(f"  memories        {view.count}")
        latest = view.latest_ingested_at
        stamp = "-" if latest is None else f"{latest:%Y-%m-%d %H:%M:%S}"
        print(f"  latest ingest   {stamp}")
        for kind, count in view.by_kind().items():
            print(f"    {kind:10} {count}")
        if view.count == 0:
            print("\n  the system knew nothing at this time")
    except ValueError as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    return 0


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


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


def _prompt(question: str, *, default: str = "") -> str:
    """One line from the person at the terminal.

    A thin wrapper so the interactive capture reads as a conversation in the
    source, and so the whole of it can be driven from a test by patching one
    function rather than by faking a terminal.
    """
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer or default


def _prompt_list(question: str, *, allow_none: bool = True) -> list[str]:
    """Repeated answers to the same question, one at a time, ending on a blank.

    One at a time rather than a comma-separated line, and that is the whole
    reason `--interactive` exists rather than more flags. Asked for "your
    assumptions" in one field, people write one sentence; asked the same
    question five times, they produce the third and fourth ones, which are the
    ones they had not noticed they were making.
    """
    answers: list[str] = []
    while True:
        answer = input(f"{question} (blank to finish): ").strip()
        if not answer:
            break
        answers.append(answer)
    if not answers and not allow_none:
        print("  (none recorded)")
    return answers


def _prompt_confidence(question: str) -> float | None:
    """A probability, or nothing. An unanswerable question gets no answer.

    Refusing to default is the point. Zero and 0.5 are both real claims, and a
    field silently filled with either would make the calibration M5.2 measures
    a measurement of this function.
    """
    while True:
        raw = input(f"{question} (0-1, blank to skip): ").strip()
        if not raw:
            return None
        try:
            value = float(raw)
        except ValueError:
            print("  a number between 0 and 1, or blank")
            continue
        if 0.0 <= value <= 1.0:
            return value
        print("  a number between 0 and 1, or blank")


def _interactive(draft: decisions.DecisionDraft) -> decisions.DecisionDraft:
    """Fill in everything the flags did not, one question at a time.

    The order is deliberate: alternatives before reasoning, because naming what
    lost changes what somebody writes about why the winner won; assumptions
    last, because they are the hardest and asking for them first stops people
    finishing the form at all.
    """
    question = draft.question or _prompt("What was being decided")
    chosen = draft.chosen or _prompt("What was chosen")

    print("\nWhat else was on the table? Each one, then why it lost.")
    options = list(draft.options)
    while True:
        description = input("  option (blank to finish): ").strip()
        if not description:
            break
        why = input("    rejected because: ").strip() or None
        options.append(decisions.OptionInput(description=description, rejected_because=why))

    reasoning = draft.reasoning or _prompt("\nWhy this one") or None
    confidence = (
        draft.confidence
        if draft.confidence is not None
        else _prompt_confidence("How confident are you, right now")
    )
    expected = draft.expected_outcome or _prompt("What do you expect to happen") or None

    print(
        "\nWhat has to be true for this to be the right call? One at a time.\n"
        "  ('none' is an answer — a decision that rests on nothing is a finding.)"
    )
    statements = _prompt_list("  assumption")
    assumptions = [decisions.AssumptionInput(statement=text) for text in statements]

    return decisions.DecisionDraft(
        question=question,
        chosen=chosen,
        reasoning=reasoning,
        confidence=confidence,
        expected_outcome=expected,
        options=tuple(options),
        assumptions=tuple(assumptions),
        evidence=draft.evidence,
    )


async def run_decide(
    settings: Settings,
    *,
    question: str,
    chosen: str,
    reasoning: str | None,
    confidence: float | None,
    expected: str | None,
    options: list[str],
    assumptions: list[str],
    evidence: list[str],
    decided: str | None,
    interactive: bool,
) -> int:
    """Record one decision by hand. The primary path, and deliberately so.

    An `--option` may carry its rejection after a `::` — `--option "Celery::a
    broker cannot share the transaction"` — which keeps the alternative and the
    reason it lost in one argument. Splitting them across two flags would let a
    reason be attached to the wrong option by miscounting.
    """
    draft = decisions.DecisionDraft(
        question=question,
        chosen=chosen,
        reasoning=reasoning,
        confidence=confidence,
        expected_outcome=expected,
        options=tuple(_parse_option(value) for value in options),
        assumptions=tuple(
            decisions.AssumptionInput(statement=value) for value in assumptions
        ),
        evidence=tuple(_parse_evidence(value) for value in evidence),
    )
    if interactive:
        draft = _interactive(draft)

    # `declared` when a date was typed, `declared` when it defaults to now: both
    # are a person asserting when this happened, which is exactly what M1.1's
    # `declared` means. Nothing here reads a file's mtime, so no other
    # provenance is reachable from this command.
    when = parse_moment(decided, name="--decided") if decided else datetime.now(UTC)

    container = Container.build(settings)
    try:
        decision_id = await decisions.record(
            container.database.session_factory,
            draft,
            decided_at=when,
            decided_at_source=TimeProvenance.DECLARED,
        )
    except decisions.InvalidDecision as exc:
        print(f"refused: {exc}")
        return 1
    except decisions.UnresolvedEvidence as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        await container.dispose()

    print(f"recorded {decision_id}")
    return 0


def _parse_option(value: str) -> decisions.OptionInput:
    description, _, why = value.partition("::")
    return decisions.OptionInput(
        description=description.strip(), rejected_because=why.strip() or None
    )


def _parse_evidence(value: str) -> decisions.EvidenceInput:
    """`source:path[#ordinal][::relation]` — the natural key, as a person types it."""
    body, _, relation = value.partition("::")
    locator, _, ordinal = body.partition("#")
    source_name, _, external_key = locator.partition(":")
    if not source_name or not external_key:
        raise SystemExit(
            f"evidence must look like source:path[#chunk][::relation], got {value!r}"
        )
    try:
        chosen_relation = (
            EvidenceRelation(relation.strip()) if relation.strip() else EvidenceRelation.INFORMED
        )
    except ValueError as exc:
        allowed = ", ".join(member.value for member in EvidenceRelation)
        raise SystemExit(f"relation must be one of {allowed}") from exc
    return decisions.EvidenceInput(
        source_name=source_name.strip(),
        external_key=external_key.strip(),
        relation=chosen_relation,
        chunk_ordinal=int(ordinal) if ordinal.strip() else None,
    )


async def run_decisions_list(
    settings: Settings, *, status: str | None, limit: int
) -> int:
    container = Container.build(settings)
    try:
        rows = await decisions.list_decisions(
            container.database.session_factory,
            status=DecisionStatus(status) if status else None,
            limit=limit,
        )
    finally:
        await container.dispose()

    if not rows:
        print("no decisions recorded")
        return 0

    print(f"{len(rows)} decision(s)\n")
    for row in rows:
        confidence = "  —  " if row.confidence is None else f"{row.confidence:.2f}"
        # The date carries its provenance the way M4.1's timeline does: a `~`
        # for anything that was not asserted by a person.
        marker = "" if row.decided_at_source is TimeProvenance.DECLARED else "~"
        print(f"{row.id}  {row.status.value:8} conf {confidence}  {marker}{_stamp(row.decided_at)}")
        print(f"     {row.question}")
        print(f"     → {row.chosen}")
        # The counts are the point of the list. A decision with no assumptions
        # is one M5.2 has nothing to evaluate.
        print(
            f"     {row.options} options   {row.assumptions} assumptions   "
            f"{row.evidence} evidence\n"
        )
    return 0


async def run_decisions_show(settings: Settings, *, decision_id: str) -> int:
    container = Container.build(settings)
    try:
        detail = await decisions.show(
            container.database.session_factory, UUID(decision_id)
        )
        # Composed here rather than inside `decisions.show`, so the two modules
        # stay independent: outcomes read decisions, and nothing reads back.
        recorded = await outcomes.for_decision(
            container.database.session_factory, UUID(decision_id)
        )
    except decisions.UnknownDecision as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()

    marker = "" if detail.decided_at_source is TimeProvenance.DECLARED else "~"
    print(f"{detail.question}\n")
    print(f"  chosen      {detail.chosen}")
    print(f"  status      {detail.status.value}")
    print(
        f"  decided     {marker}{_stamp(detail.decided_at)}  "
        f"({detail.decided_at_source.value})"
    )
    if detail.confidence is not None:
        print(f"  confidence  {detail.confidence:.2f} at the time")
    if detail.reasoning:
        print(f"\n  why         {detail.reasoning}")
    if detail.expected_outcome:
        print(f"  expected    {detail.expected_outcome}")

    print("\n  options")
    for option in detail.options:
        mark = "✓" if option.was_chosen else "·"
        print(f"    {mark} {option.description}")
        if option.rejected_because:
            print(f"        rejected: {option.rejected_because}")

    print("\n  assumptions")
    if not detail.assumptions:
        # Said out loud rather than left as an empty heading. This is the field
        # M5.2 evaluates, so its absence is the most important thing on screen.
        print("    none recorded — nothing here for M5.2 to evaluate")
    for assumption in detail.assumptions:
        held = "?" if assumption.held is None else ("held" if assumption.held else "broke")
        confidence = "" if assumption.confidence is None else f" ({assumption.confidence:.2f})"
        print(f"    [{held}] {assumption.statement}{confidence}")

    print("\n  evidence")
    if not detail.evidence:
        print("    none linked")
    for item in detail.evidence:
        where = (
            f"{item.external_key}#{item.chunk_ordinal}"
            if item.chunk_ordinal is not None
            else item.external_key
        )
        print(f"    {item.relation.value:11} {item.source_name}:{where}")

    _print_outcomes(recorded)
    return 0


async def run_decisions_edit(
    settings: Settings,
    *,
    decision_id: str,
    question: str | None,
    chosen: str | None,
    reasoning: str | None,
    expected: str | None,
    status: str | None,
    options: list[str],
    assumptions: list[str],
) -> int:
    container = Container.build(settings)
    try:
        await decisions.edit(
            container.database.session_factory,
            UUID(decision_id),
            decisions.DecisionEdit(
                question=question,
                chosen=chosen,
                reasoning=reasoning,
                expected_outcome=expected,
                status=DecisionStatus(status) if status else None,
                options=(
                    tuple(_parse_option(value) for value in options) if options else None
                ),
                assumptions=(
                    tuple(
                        decisions.AssumptionInput(statement=value)
                        for value in assumptions
                    )
                    if assumptions
                    else None
                ),
            ),
        )
    except decisions.UnknownDecision as exc:
        print(str(exc))
        return 1
    except decisions.InvalidDecision as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        await container.dispose()
    print(f"updated {decision_id}")
    return 0


async def run_decisions_link(
    settings: Settings, *, decision_id: str, evidence: list[str]
) -> int:
    container = Container.build(settings)
    linked = 0
    try:
        for value in evidence:
            try:
                await decisions.link_evidence(
                    container.database.session_factory,
                    UUID(decision_id),
                    _parse_evidence(value),
                )
            except decisions.UnresolvedEvidence as exc:
                print(f"skipped: {exc}")
                continue
            linked += 1
    except decisions.UnknownDecision as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    print(f"linked {linked} of {len(evidence)}")
    return 0


async def run_outcome(
    settings: Settings,
    *,
    decision_id: str,
    verdict: str,
    description: str,
    observed: str | None,
    evidence: list[str],
) -> int:
    """Record what happened, as somebody who watched it happen.

    `declared`, confidence 1.0, and neither is a flag. Saying you observed
    something *is* certainty about the observation, and a `--confidence 0.7` on
    this command would be somebody hedging testimony into the one column M5.3
    uses to tell testimony from a model's guess. A reading you are unsure about
    belongs in the suggestion queue, which is what `outcomes suggest` fills.
    """
    when = parse_moment(observed, name="--observed") if observed else datetime.now(UTC)
    container = Container.build(settings)
    try:
        outcome_id = await outcomes.record(
            container.database.session_factory,
            UUID(decision_id),
            outcomes.OutcomeDraft(
                description=description,
                verdict=OutcomeVerdict(verdict),
                evidence=tuple(
                    _parse_outcome_evidence(value) for value in evidence
                ),
            ),
            observed_at=when,
            observed_at_source=TimeProvenance.DECLARED,
            evidence_kind=EvidenceKind.DECLARED,
        )
    except decisions.UnknownDecision as exc:
        print(str(exc))
        return 1
    except (outcomes.InvalidOutcome, outcomes.UnresolvedEvidence) as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        await container.dispose()
    print(f"recorded {outcome_id}")
    return 0


def _parse_outcome_evidence(value: str) -> outcomes.OutcomeEvidenceInput:
    """`source:path[#ordinal]` — the same natural key `decisions link` takes.

    No relation here, unlike decision evidence: an outcome's evidence has only
    one relation to it, which is that it shows the outcome happened. The
    informed/records distinction is about what came before a decision, and
    everything here is after one by construction.
    """
    locator, _, ordinal = value.partition("#")
    source_name, _, external_key = locator.partition(":")
    if not source_name or not external_key:
        raise SystemExit(f"evidence must look like source:path[#chunk], got {value!r}")
    return outcomes.OutcomeEvidenceInput(
        source_name=source_name.strip(),
        external_key=external_key.strip(),
        chunk_ordinal=int(ordinal) if ordinal.strip() else None,
    )


def _print_outcomes(rows: list[outcomes.OutcomeRow]) -> None:
    """The outcome block of `decisions show`."""
    print("\n  outcomes")
    if not rows:
        print("    none recorded — not the same as 'too early', which is a verdict")
        return
    for row in rows:
        marker = "" if row.observed_at_source is TimeProvenance.DECLARED else "~"
        confidence = "" if row.confidence is None else f" conf {row.confidence:.2f}"
        # The evidence kind is printed on every line rather than only when it is
        # inferred. A reader scanning a list has to be able to see which of
        # these somebody watched happen.
        print(
            f"    [{row.verdict.value:9}] {row.evidence_kind.value:8} "
            f"{marker}{_stamp(row.observed_at)}{confidence}"
        )
        print(f"        {row.description}")
        for item in row.evidence:
            where = (
                f"{item.external_key}#{item.chunk_ordinal}"
                if item.chunk_ordinal is not None
                else item.external_key
            )
            print(f"        · {item.source_name}:{where}")


async def run_outcomes_suggest(
    settings: Settings,
    *,
    decision_id: str | None,
    window_days: float | None,
    limit: int,
) -> int:
    container = Container.build(settings)
    try:
        suggest = outcome_suggest.SuggestOutcomes(
            container.database.session_factory, container.language_model()
        )
        report = await suggest(
            decision_id=UUID(decision_id) if decision_id else None,
            window_days=window_days,
            limit=limit,
        )
    except decisions.UnknownDecision as exc:
        print(str(exc))
        return 1
    except MissingApiKey as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()

    print(f"decisions examined:        {report.decisions}")
    if window_days is None:
        print(
            f"window:                   derived per decision from its confidence, "
            f"{outcome_suggest.MIN_WINDOW_DAYS:.0f}-"
            f"{outcome_suggest.MAX_WINDOW_DAYS:.0f} days "
            f"({outcome_suggest.DEFAULT_WINDOW_DAYS:.0f} where none was recorded)"
        )
        # Said out loud every run. A derived number that nobody sees is a
        # constant with extra steps, and this one is a guess rather than a
        # measurement.
        print("                          — a heuristic, not a measurement")
    else:
        print(f"window:                   {window_days:.0f} days (overridden)")
    print(f"candidates in window:      {report.candidates}")
    print(f"model calls:               {report.calls}")
    print(f"judged an outcome:         {report.judged_yes}")
    print(f"judged not:                {report.judged_no}")
    print(f"judged unsure:             {report.judged_unsure}")
    print(f"below confidence floor:    {report.below_confidence}")
    print(f"unparseable responses:     {report.unparseable}")
    print(f"queued for review:         {report.proposed}")
    print(f"already queued:            {report.duplicates}")
    # The two numbers that explain an empty run, and they explain it very
    # differently: no candidates is a corpus with nothing after these decisions,
    # no entity coverage is a filter that could not be applied at all.
    print(f"decisions with no candidate:      {report.decisions_without_candidates}")
    print(f"decisions with no entity coverage: {report.decisions_without_entity_coverage}")
    if report.decisions_without_entity_coverage:
        print(
            "\nAn 'unavailable' entity filter means nothing has been extracted for "
            "that decision's evidence, so candidates were found by time alone. "
            "Run `memoryos extract-entities`; `memoryos doctor` reports the "
            "coverage."
        )
    if report.proposed:
        print("\nNothing committed. Review with `memoryos outcomes review`.")
    return 0


async def run_outcomes_review(
    settings: Settings, *, status: str | None, limit: int, show_passage: bool
) -> int:
    container = Container.build(settings)
    try:
        rows = await outcome_suggest.list_suggestions(
            container.database.session_factory,
            status=SuggestionStatus(status) if status else None,
            limit=limit,
        )
    finally:
        await container.dispose()

    if not rows:
        print("nothing in the outcome review queue")
        return 0

    print(f"{len(rows)} suggestion(s)\n")
    for row in rows:
        print(f"{row.id}  {row.status.value}")
        print(f"     decision   {row.decision_question}")
        print(f"     candidate  {row.source_name}:{row.external_key}")
        # The temporal gap, stated. The entire claim is that one thing followed
        # another closely enough to be connected, so the number that claim rests
        # on belongs on screen rather than inside a score.
        print(
            f"     gap        {outcome_suggest.describe_gap(row.gap_days)} after the "
            f"decision (window {row.window_days:.0f}d)"
        )
        shared = ", ".join(row.shared_entities) if row.shared_entities else "—"
        print(f"     entities   {row.entity_filter}: {shared}")
        print(f"     verdict    {row.draft.verdict.value}")
        print(f"     says       {row.draft.description}")
        if row.draft.rationale:
            print(f"     because    {row.draft.rationale}")
        if show_passage:
            excerpt = " ".join(row.source_text.split())[:400]
            print(f"     ┃  {excerpt}")
        print()
    print("Accept with `outcomes accept <id>`, reject with `outcomes reject <id>`.")
    return 0


async def run_outcomes_accept(settings: Settings, *, suggestion_id: str) -> int:
    container = Container.build(settings)
    try:
        outcome_id = await outcome_suggest.accept(
            container.database.session_factory, UUID(suggestion_id)
        )
    except (decisions.UnknownDecision, outcome_suggest.AlreadyReviewed) as exc:
        print(str(exc))
        return 1
    except outcomes.InvalidOutcome as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        await container.dispose()
    print(f"recorded {outcome_id}")
    # Said on every accept. An accepted suggestion is still a model's reading,
    # and the one thing `evidence_kind` exists for is that nothing later
    # mistakes it for testimony.
    print("Recorded as `inferred` — accepting is not the same as having watched it.")
    return 0


async def run_outcomes_reject(settings: Settings, *, suggestion_id: str) -> int:
    container = Container.build(settings)
    try:
        await outcome_suggest.reject(
            container.database.session_factory, UUID(suggestion_id)
        )
    except (decisions.UnknownDecision, outcome_suggest.AlreadyReviewed) as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    print("rejected; the pair will not be proposed again")
    return 0


async def run_outcomes_rate(settings: Settings) -> int:
    container = Container.build(settings)
    try:
        rate = await outcomes.success_rate(container.database.session_factory)
    finally:
        await container.dispose()

    print(f"worked     {rate.worked}")
    print(f"failed     {rate.failed}")
    print(f"mixed      {rate.mixed}")
    print(f"too early  {rate.too_early}   (excluded from the rate)")
    print(f"undecided  {rate.undecided}   (no outcome recorded at all)")
    if rate.rate is None:
        # Not 0%. A corpus where every decision is too early has no success rate
        # rather than a bad one, and printing 0.0% would be the interface
        # inventing a verdict.
        print("\nno resolved outcomes, so there is no success rate to report")
    else:
        print(f"\nrate       {rate.rate:.0%} of {rate.resolved} resolved")
    return 0


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------


def _print_pattern(row: patterns.PatternRow, *, full: bool) -> None:
    confidence = "—" if row.confidence is None else f"{row.confidence:.2f}"
    span = "" if row.span_days is None else f", spanning {row.span_days:.0f} days"
    print(f"{row.id}  [{row.kind.value}]  confidence {confidence}")
    print(f"     {row.statement}")
    print(
        f"     {row.support_count} supporting, "
        f"{row.contradiction_count} contradicting{span}"
    )
    if row.dismissed_at:
        print(f"     DISMISSED: {row.dismissed_reason}")
    if not full:
        return
    # Both lists, at the same weight, always. Counter-evidence printed shorter
    # or last is how a tool becomes a flatterer.
    for label, items in (
        ("supports", row.supporting),
        ("contradicts", row.contradicting),
    ):
        print(f"\n     {label} ({len(items)})")
        for item in items:
            print(f"       {_stamp(item.decided_at)}  {item.decision_question}")
            if item.note:
                print(f"         {item.note}")


async def run_patterns_discover(
    settings: Settings, *, min_support: int, window_days: float
) -> int:
    container = Container.build(settings)
    try:
        report = await patterns.discover(
            container.database.session_factory,
            min_support=min_support,
            window_days=window_days,
        )
    finally:
        await container.dispose()

    print(f"candidates considered:  {report.candidates}")
    print(f"minimum support:        {min_support} distinct decisions")
    print(f"emitted:                {report.emitted}")
    print(f"updated:                {report.updated}")
    print(f"below support:          {report.below_support}")
    print(f"outweighed by counter-evidence: {report.outweighed_by_counter_evidence}")
    print(f"within sampling noise:  {report.within_noise}")
    print(f"skipped (dismissed):    {report.skipped_dismissed}")

    if report.silent:
        # Printed whether or not anything was emitted. A detector that could not
        # run is a fact about the corpus, and leaving it out would make the
        # machinery look busier than it was.
        print("\ndetectors with nothing to propose:")
        for name, reason in report.silent:
            print(f"  {name}: {reason}")

    if not report.emitted and not report.updated:
        # The correct result on a small corpus, and it has to read like one.
        # "No patterns" alone is indistinguishable from a broken detector.
        print("\nno patterns with sufficient support.\n")
        print("what was considered, and why each was not emitted:")
        for candidate in report.considered:
            support = len(candidate.supporting)
            counter = len(candidate.contradicting)
            if candidate.rejected_because is not None:
                reason = candidate.rejected_because
            elif support < min_support:
                reason = (
                    f"{support} supporting decision(s), below the minimum of "
                    f"{min_support}"
                )
            elif support <= counter:
                reason = f"{counter} contradicting against {support} supporting"
            else:
                reason = "emitted"
            print(f"\n  [{candidate.kind.value}] {candidate.statement}")
            print(f"    → {reason}")
    return 0


async def run_patterns_list(
    settings: Settings, *, kind: str | None, include_dismissed: bool
) -> int:
    container = Container.build(settings)
    try:
        rows = await patterns.list_patterns(
            container.database.session_factory,
            kind=PatternKind(kind) if kind else None,
            include_dismissed=include_dismissed,
        )
        totals = await patterns.counts(container.database.session_factory)
    finally:
        await container.dispose()

    if not rows:
        print("no patterns recorded")
        print(
            "\nRun `memoryos patterns discover`. Nothing clearing the bar is a "
            "result,\nnot a failure — see the reasons that command prints."
        )
        return 0
    print(f"{len(rows)} pattern(s)   ({totals['dismissed']} dismissed)\n")
    for row in rows:
        _print_pattern(row, full=False)
        print()
    return 0


async def run_patterns_show(settings: Settings, *, pattern_id: str) -> int:
    container = Container.build(settings)
    try:
        row = await patterns.show(
            container.database.session_factory, UUID(pattern_id)
        )
    except patterns.UnknownPattern as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    _print_pattern(row, full=True)
    return 0


async def run_patterns_dismiss(
    settings: Settings, *, pattern_id: str, reason: str
) -> int:
    container = Container.build(settings)
    try:
        await patterns.dismiss(
            container.database.session_factory, UUID(pattern_id), reason=reason
        )
    except patterns.UnknownPattern as exc:
        print(str(exc))
        return 1
    except ValueError as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        await container.dispose()
    print("dismissed; discovery will not propose this subject again")
    return 0


async def run_patterns_calibration(settings: Settings) -> int:
    """The calibration table, printed whether or not anything clears the bar.

    This is the output worth reading even — especially — when discovery is
    silent. "No patterns found" and "here are the bands, and every stated
    confidence falls inside what its sample supports" are the same result, and
    only the second one lets a reader see how far from a finding it was.
    """
    container = Container.build(settings)
    try:
        report = await patterns.calibration(container.database.session_factory)
    finally:
        await container.dispose()

    if report.excluded:
        # Printed first and unconditionally. An empty table below this line means
        # something entirely different from an empty table without it.
        print(
            f"excluded from every band: {report.excluded_decisions} decision(s) and "
            f"{report.excluded_assumptions} assumption(s)\n"
            f"  their confidence was reconstructed after the fact, and hindsight "
            f"cannot\n  measure foresight at any weight — see "
            f"domain/values.ConfidenceHorizon"
        )

    if not report.bands:
        print("\nnothing to calibrate: no confidence recorded before its outcome")
        return 0

    for detector, values in sorted(report.bands.items()):
        noun = "decisions" if detector.startswith("decision") else "assumptions"
        print(f"\n{noun} by stated confidence")
        print(f"  {'band':<12} {'n':>3}  {'stated':>7}  {'actual':>7}  95% CI")
        for band in values:
            marker = "  <-- outside the interval" if band.miscalibrated else ""
            print(
                f"  {band.low:.2f}-{band.high:.2f}  {band.interval.n:>3}  "
                f"{band.stated:>7.2f}  {band.interval.observed:>7.0%}  "
                f"{band.interval.low:.0%}-{band.interval.high:.0%}{marker}"
            )
    print(
        "\nA band is a finding only when the stated confidence falls outside the "
        "interval\nits sample supports. Anything inside is consistent with being "
        "exactly as\nreliable as claimed — see domain/patterns.py."
    )
    print(
        "\nCalibration is only meaningful when the confidence was written down "
        "before the\noutcome was known. Nothing in the schema records whether it "
        "was."
    )
    return 0


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------


def _print_event(event: Event) -> None:
    state = "done" if event.processed_at else "PENDING"
    lag = event.delivery_lag
    lag_note = "" if lag is None else f"  (+{lag.total_seconds():.1f}s to arrive)"
    keys = ",".join(sorted(event.payload)) or "-"
    print(
        f"{_stamp(event.received_at)}  {state:<7}  {event.kind.value:<16} "
        f"{event.source:<12}  {keys}{lag_note}"
    )


async def run_events_tail(
    settings: Settings, *, kind: str | None, limit: int, follow: bool
) -> int:
    """The event stream, oldest first, optionally following.

    `--follow` polls rather than listening on `NOTIFY`, and the reason is the
    same one that made the queue a table: a poll is one indexed query against
    rows that are already committed, while a listener is a second delivery path
    that can be connected while the transaction that would have notified it
    rolls back. At one query a second against a partial index this costs
    nothing, and it cannot show an event that is not in the table.
    """
    container = Container.build(settings)
    sessions = container.database.session_factory
    try:
        rows = await events.tail(
            sessions, kind=EventKind(kind) if kind else None, limit=limit
        )
        if not rows and not follow:
            print("no events")
            print(
                "\nPost one: curl -X POST localhost:8000/events -H "
                "'Content-Type: application/json' \\\n"
                "  -d '{\"kind\":\"manual\",\"source\":\"cli\",\"payload\":{}}'"
            )
            return 0
        for event in rows:
            _print_event(event)
        if not follow:
            return 0

        # Watermarked on `received_at` rather than re-reading the window, so a
        # long-running follow does not re-render what it has already printed and
        # does not grow more expensive the longer it runs.
        since = rows[-1].received_at if rows else datetime.now(UTC)
        while True:
            await asyncio.sleep(1.0)
            fresh = await events.tail(sessions, kind=EventKind(kind) if kind else None,
                                      limit=200, since=since)
            for event in fresh:
                _print_event(event)
            if fresh and fresh[-1].received_at is not None:
                since = fresh[-1].received_at
    except KeyboardInterrupt:
        return 0
    finally:
        await container.dispose()


async def run_events_stats(settings: Settings) -> int:
    """Counts per kind, and the latency M6.1 depends on.

    **The mean from `received_at` to `processed_at` is the number this milestone
    exists to produce.** Everything else here is a row count. M6.1 puts context
    assembly behind these triggers, and context that arrives after the meeting
    starts is worth nothing — so whether a Postgres queue is adequate for a push
    system is a question this number answers rather than an architecture opinion.
    """
    container = Container.build(settings)
    try:
        report = await events.stats(container.database.session_factory)
    finally:
        await container.dispose()

    if not report.by_kind:
        print("no events recorded")
        return 0

    print(f"  {'kind':<16} {'total':>6} {'done':>6} {'pending':>8}  {'latency':>9}  arrival")
    for row in report.by_kind:
        latency = "—" if row.mean_latency_seconds is None else f"{row.mean_latency_seconds:.3f}s"
        lag = (
            "—"
            if row.mean_delivery_lag_seconds is None
            else f"{row.mean_delivery_lag_seconds:.3f}s"
        )
        print(
            f"  {row.kind.value:<16} {row.total:>6} {row.processed:>6} "
            f"{row.pending:>8}  {latency:>9}  {lag}"
        )

    overall = report.mean_latency_seconds
    print(f"\n  {'total':<16} {report.total:>6} {report.processed:>6} {report.pending:>8}")
    if overall is None:
        print(
            "\nNothing has been processed, so there is no latency to report. Run "
            "`memoryos worker --drain`."
        )
    else:
        # Weighted by how many each kind processed. A plain mean of the per-kind
        # means would let one rare kind outweigh two hundred common ones.
        print(f"\n  mean received -> processed: {overall:.3f}s (weighted by kind)")
    print(
        "\nArrival is occurred_at -> received_at, kept separate on purpose: a slow "
        "queue is\nthis system's fault and a slow delivery is the network's."
    )
    return 0


# --------------------------------------------------------------------------
# Reflections
# --------------------------------------------------------------------------


def _print_reflection(row: reflections.ReflectionRow, *, full: bool = True) -> None:
    rate = "—" if row.citation_rate is None else f"{row.citation_rate:.0%}"
    print(f"{row.id}  cited {rate}  {row.model_id}  {_stamp(row.generated_at)}")
    if row.dismissed_at:
        print(f"     DISMISSED: {row.dismissed_reason}")
    elif row.acknowledged_at:
        print(f"     acknowledged {_stamp(row.acknowledged_at)}")
    print()
    for line in textwrap.wrap(row.text, width=78):
        print(f"  {line}")
    print()
    print(
        f"     from: {row.pattern_statement}"
    )
    print(
        f"     {row.support_count} supporting, {row.contradiction_count} contradicting"
    )
    if not full:
        return
    # The citations, resolved to the decisions they point at. Printed with the
    # marker so a reader can follow `[2]` out of the prose and into the record
    # it came from — which is the whole difference between a reflection and a
    # horoscope.
    print("\n     citations")
    for citation in row.citations:
        side = "for " if citation.relation is PatternRelation.SUPPORTS else "against"
        print(f"       [{citation.marker}] {side}  {citation.decision_question}")
        print(f"             {citation.decision_id}")
    uncited = row.uncited
    if uncited:
        # Flagged, never removed. A sentence deleted from the middle of a
        # paragraph leaves prose that reads as complete and is not.
        print(f"\n     {len(uncited)} sentence(s) carrying no citation:")
        for sentence in uncited:
            print(f"       {sentence}")


async def run_reflect(
    settings: Settings,
    *,
    pattern_id: str | None,
    threshold: float,
    regenerate: bool,
) -> int:
    """Generate reflections for whatever cleared the bar, and say so when nothing did.

    **Printing what would be needed is the correct output of this command, not
    its failure mode.** A pattern below the confidence bar produces no reflection
    at all — not a hedged one — and the model is never called for it, so there is
    no fluent paragraph anywhere in the process to be tempted by. What a reader
    gets instead is the arithmetic: how much more agreeing evidence it would take.
    """
    container = Container.build(settings)
    try:
        report = await reflections.reflect(
            container.database.session_factory,
            container.language_model(),
            pattern_id=UUID(pattern_id) if pattern_id else None,
            threshold=threshold,
            regenerate=regenerate,
        )
    except MissingApiKey as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        await container.dispose()

    print(f"patterns considered:    {report.considered}")
    print(f"threshold:              {threshold:.2f} confidence")
    print(f"reflections written:    {report.written}")
    print(f"below the threshold:    {len(report.refused)}")
    print(f"generated and rejected: {len(report.rejected)}")
    print(f"skipped (dismissed):    {report.skipped_dismissed}")
    print(f"skipped (already have one): {report.skipped_existing}")

    for reflection in report.reflections:
        print()
        print("-" * 78)
        if reflection.text:
            for line in textwrap.wrap(reflection.text, width=78):
                print(f"  {line}")
        if reflection.check is not None:
            print(f"\n  citation rate {reflection.check.citation_rate:.0%}")
            for sentence in reflection.check.uncited:
                print(f"  UNCITED: {sentence}")

    if report.rejected:
        # The model was called and what came back could not be stored. Louder
        # than a refusal, because this is the guardrail catching something.
        print("\ngenerated and thrown away:")
        for reflection in report.rejected:
            print(f"  {reflection.statement}")
            print(f"    → {reflection.rejected_because}")

    if report.refused:
        print("\nno reflection, and what each would need:")
        for reflection in report.refused:
            print(f"\n  {reflection.statement}")
            print(f"    → {reflection.refused_because}")
            print(f"    → would need {reflection.needed}")

    if not report.considered:
        print(
            "\nNo patterns to reflect on. Run `memoryos patterns discover` first;\n"
            "nothing clearing that bar is a result rather than a failure."
        )
    elif not report.written:
        print(
            "\nNothing was written, and that is the intended output rather than an\n"
            "error. A behavioural claim in prose is the riskiest thing this system\n"
            "emits, so it is refused before the model is called rather than hedged\n"
            "afterwards."
        )
    return 0


async def run_reflections_list(settings: Settings, *, include_dismissed: bool) -> int:
    container = Container.build(settings)
    try:
        rows = await reflections.list_reflections(
            container.database.session_factory, include_dismissed=include_dismissed
        )
    finally:
        await container.dispose()

    if not rows:
        print("no reflections")
        print(
            "\nRun `memoryos reflect --all`. Nothing clearing the bar is a result,\n"
            "not a failure — that command prints what each pattern would need."
        )
        return 0
    for row in rows:
        _print_reflection(row)
        print()
    return 0


async def run_reflections_acknowledge(settings: Settings, *, reflection_id: str) -> int:
    container = Container.build(settings)
    try:
        await reflections.acknowledge(
            container.database.session_factory, UUID(reflection_id)
        )
    except reflections.UnknownReflection as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    print("acknowledged; read, which is not the same as agreed with")
    return 0


async def run_reflections_dismiss(
    settings: Settings, *, reflection_id: str, reason: str
) -> int:
    container = Container.build(settings)
    try:
        await reflections.dismiss(
            container.database.session_factory, UUID(reflection_id), reason=reason
        )
    except reflections.UnknownReflection as exc:
        print(str(exc))
        return 1
    except ValueError as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        await container.dispose()
    print(
        "dismissed; this claim will not be shown again and its pattern will not be\n"
        "reflected on again"
    )
    return 0


# --------------------------------------------------------------------------
# Assumptions
# --------------------------------------------------------------------------


def _verdict_of(raw: str) -> AssumptionVerdict:
    """`--held true|false|partially`, in the words a person would type.

    `true` and `false` are accepted because the milestone's own interface says
    so and because they are what somebody reaches for; `partially` has no
    boolean spelling, which is the point of it existing.
    """
    normalised = raw.strip().lower()
    if normalised in ("true", "yes", "held"):
        return AssumptionVerdict.HELD
    if normalised in ("false", "no", "failed", "broke"):
        return AssumptionVerdict.FAILED
    if normalised in ("partial", "partially", "mixed"):
        return AssumptionVerdict.PARTIALLY
    raise SystemExit(
        f"--held takes true, false or partially, got {raw!r}. 'partially' is "
        f"there because almost nothing anybody assumes is cleanly right or wrong"
    )


def _print_assumption(row: assumptions.AssumptionRow, *, index: int | None = None) -> None:
    prefix = f"{index}. " if index is not None else ""
    verdict = "unevaluated" if row.held is None else row.held.value
    confidence = "" if row.confidence is None else f"  (held at {row.confidence:.2f})"
    print(f"{prefix}{row.id}")
    print(f"     {row.statement}{confidence}")
    print(f"     verdict    {verdict}")
    # The decision and its outcome, because an assumption read away from the
    # choice it served is a sentence with its subject removed.
    print(f"     decision   {row.decision_question}")
    outcome = "none recorded" if row.outcome_verdict is None else row.outcome_verdict.value
    print(f"     outcome    {outcome}")
    if row.group_label:
        print(f"     group      {row.group_label}")
    if row.note:
        print(f"     note       {row.note}")
    for item in row.evidence:
        print(f"     · {item.source_name}:{item.external_key}")


async def run_assumptions_review(
    settings: Settings, *, decision_id: str | None, unevaluated: bool, limit: int
) -> int:
    """Walk assumptions with the context needed to judge them.

    Prints rather than prompts. An interactive loop would be a nicer demo and a
    worse tool: evaluating an assumption honestly means going and reading
    something, and a prompt that sits waiting is a prompt somebody answers from
    memory to make it go away.
    """
    container = Container.build(settings)
    try:
        rows = await assumptions.list_assumptions(
            container.database.session_factory,
            decision_id=UUID(decision_id) if decision_id else None,
            unevaluated_only=unevaluated,
            limit=limit,
        )
    finally:
        await container.dispose()

    if not rows:
        print("no assumptions match")
        return 0

    print(f"{len(rows)} assumption(s)\n")
    for index, row in enumerate(rows, start=1):
        _print_assumption(row, index=index)
        print()
    print(
        "Evaluate with `memoryos assumption <id> --held true|false|partially "
        '--note "..."`.'
    )
    print("Find supporting memories with `memoryos assumptions suggest`.")
    return 0


async def run_assumption_evaluate(
    settings: Settings,
    *,
    assumption_id: str,
    held: str,
    note: str | None,
    evidence: list[str],
) -> int:
    container = Container.build(settings)
    try:
        await assumptions.evaluate(
            container.database.session_factory,
            UUID(assumption_id),
            _verdict_of(held),
            note=note,
            evidence=[_parse_assumption_evidence(value) for value in evidence],
        )
    except assumptions.UnknownAssumption as exc:
        print(str(exc))
        return 1
    except assumptions.UnresolvedEvidence as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        await container.dispose()
    print("recorded")
    return 0


def _parse_assumption_evidence(value: str) -> assumptions.EvidenceInput:
    """`source:path[#ordinal]` — the natural key, as everywhere else in Phase 5."""
    locator, _, ordinal = value.partition("#")
    source_name, _, external_key = locator.partition(":")
    if not source_name or not external_key:
        raise SystemExit(f"evidence must look like source:path[#chunk], got {value!r}")
    return assumptions.EvidenceInput(
        source_name=source_name.strip(),
        external_key=external_key.strip(),
        chunk_ordinal=int(ordinal) if ordinal.strip() else None,
    )


async def run_assumptions_suggest(
    settings: Settings, *, decision_id: str | None, limit: int
) -> int:
    container = Container.build(settings)
    try:
        suggest = assumption_suggest.SuggestAssumptionEvidence(
            container.database.session_factory, container.search()
        )
        report = await suggest(
            decision_id=UUID(decision_id) if decision_id else None, limit=limit
        )
    finally:
        await container.dispose()

    print(f"assumptions examined:     {report.assumptions}")
    print(f"passages retrieved:       {report.retrieved}")
    print(f"dropped (before decision): {report.dropped_before_decision}")
    print(f"with evidence to show:    {report.with_evidence}")
    print(f"no entity coverage:       {report.without_entity_coverage}\n")

    for proposal in report.proposals:
        if not proposal.evidence:
            continue
        print(f"{proposal.assumption.id}")
        print(f"     {proposal.assumption.statement}")
        print(f"     decision   {proposal.assumption.decision_question}")
        print(f"     entities   {proposal.entity_filter}")
        for item in proposal.evidence:
            print(f"     · {item.source_name}:{item.external_key}  [{item.why}]")
            print(f"       {item.excerpt[:200]}")
        print()

    # Said on every run. This is the one proposal path in Phase 5 with no model
    # in it, and the reason is worth repeating where somebody will read it.
    print(
        "Nothing here is a verdict. These are passages that bear on the "
        "assumption;\nwhether it held is yours to say."
    )
    return 0


async def run_assumptions_group(settings: Settings, *, dry_run: bool) -> int:
    container = Container.build(settings)
    try:
        group = assumption_groups.GroupAssumptions(
            container.database.session_factory, container.embedder
        )
        report = await group(dry_run=dry_run)
    finally:
        await container.dispose()

    print(f"assumptions compared: {report.assumptions} ({report.compared} pairs)")
    print(f"auto threshold:       {assumption_groups.AUTO_THRESHOLD}")
    print(f"review floor:         {assumption_groups.REVIEW_FLOOR}")
    print(f"grouped:              {report.auto_grouped} into {report.groups_created}")
    print(f"queued for review:    {report.queued}")
    print(f"already queued:       {report.already_queued}")

    if not report.auto_grouped and not report.queued and report.near_misses:
        # The number that matters when nothing groups. "0 groups" does not
        # distinguish a corpus that came close from one that is nowhere near,
        # and those call for different next steps.
        print("\nnothing cleared the floor. The closest pairs were:")
        for score, left, right in report.near_misses:
            print(f"  {score:.3f}  {left[:60]!r}")
            print(f"         {right[:60]!r}")
    if dry_run:
        print("\ndry run; nothing written")
    return 0


async def run_assumptions_candidates(
    settings: Settings, *, status: str | None, limit: int
) -> int:
    container = Container.build(settings)
    try:
        rows = await assumption_groups.list_candidates(
            container.database.session_factory,
            status=MergeStatus(status) if status else None,
            limit=limit,
        )
    finally:
        await container.dispose()

    if not rows:
        print("nothing in the grouping queue")
        return 0
    print(f"{len(rows)} pair(s)\n")
    for row in rows:
        print(f"{row.id}  {row.status.value}  cosine {row.similarity:.3f}")
        print(f"     A  {row.left_statement}")
        print(f"        from: {row.left_question}")
        print(f"     B  {row.right_statement}")
        print(f"        from: {row.right_question}\n")
    print("Group with `assumptions accept <id>`, separate with `assumptions reject <id>`.")
    return 0


async def run_assumptions_accept(settings: Settings, *, candidate_id: str) -> int:
    container = Container.build(settings)
    try:
        group_id = await assumption_groups.accept(
            container.database.session_factory, UUID(candidate_id)
        )
    except (assumption_groups.UnknownCandidate, assumption_groups.AlreadyReviewed) as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    print(f"grouped into {group_id}")
    return 0


async def run_assumptions_reject(settings: Settings, *, candidate_id: str) -> int:
    container = Container.build(settings)
    try:
        await assumption_groups.reject(
            container.database.session_factory, UUID(candidate_id)
        )
    except (assumption_groups.UnknownCandidate, assumption_groups.AlreadyReviewed) as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    print("separated; the pair will not be proposed again")
    return 0


async def run_assumptions_stats(settings: Settings) -> int:
    container = Container.build(settings)
    try:
        report = await assumptions.stats(container.database.session_factory)
    finally:
        await container.dispose()

    print(f"total          {report.total}")
    print(f"evaluated      {report.evaluated}")
    print(f"unevaluated    {report.unevaluated}   (in neither half of any rate)")
    print(f"  held         {report.held}")
    print(f"  failed       {report.failed}")
    print(f"  partially    {report.partially}")
    if report.hold_rate is None:
        print("\nnothing evaluated, so there is no hold rate to report")
    else:
        print(f"\nhold rate      {report.hold_rate:.0%} of {report.evaluated} evaluated")

    # The line the milestone is actually for. A group of four with a 25% hold
    # rate is a finding about how somebody estimates; the corpus-wide rate
    # mostly reflects which assumptions were easy to check.
    print("\nrecurring assumptions (groups with more than one member)")
    recurring = report.recurring
    if not recurring:
        print("  none. Every assumption in this corpus is held once, so there is")
        print("  no recurrence for M5.3 to find a pattern in.")
        return 0
    for group in recurring:
        rate = "—" if group.hold_rate is None else f"{group.hold_rate:.0%}"
        print(
            f"  [{group.members} members, {group.evaluated} evaluated, "
            f"hold rate {rate}]  {group.label}"
        )
        for statement in group.statements:
            print(f"      · {statement}")
    return 0


async def run_decisions_suggest(
    settings: Settings, *, source: str | None, limit: int
) -> int:
    """Propose drafts from the corpus. Never writes a decision."""
    container = Container.build(settings)
    try:
        suggest = decision_suggest.SuggestDecisions(
            container.database.session_factory, container.language_model()
        )
        report = await suggest(source=source, limit=limit)
    except MissingApiKey as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()

    print(f"passages examined: {report.passages}")
    print(f"model calls:       {report.calls}")
    print(f"queued for review: {report.proposed}")
    # Both refusals are reported because both are the extractor being wrong in
    # ways a reviewer would otherwise have to find by hand.
    print(f"refused (no alternatives): {report.rejected_no_alternatives}")
    print(f"unparseable responses:     {report.unparseable}")
    print(f"already queued:            {report.duplicates}")
    if report.proposed:
        print("\nNothing has been committed. Review with `memoryos decisions review`.")
    return 0


async def run_decisions_review(
    settings: Settings, *, status: str | None, limit: int, show_passage: bool
) -> int:
    container = Container.build(settings)
    try:
        rows = await decision_suggest.list_suggestions(
            container.database.session_factory,
            status=SuggestionStatus(status) if status else None,
            limit=limit,
        )
    finally:
        await container.dispose()

    if not rows:
        print("nothing in the review queue")
        return 0

    totals = decision_suggest.summarise_drafts(rows)
    print(
        f"{len(rows)} suggestion(s)   "
        f"{totals['with_reasoning']} with reasoning   "
        f"{totals['with_confidence']} with confidence   "
        f"{totals['with_assumptions']} with assumptions\n"
    )
    for row in rows:
        where = (
            f"{row.external_key}#{row.chunk_ordinal}"
            if row.chunk_ordinal is not None
            else row.external_key
        )
        print(f"{row.id}  {row.status.value}  {row.source_name}:{where}")
        print(f"     Q  {row.draft.question}")
        print(f"     →  {row.draft.chosen}")
        for option in row.draft.options:
            print(f"     ·  {option.description}")
            if option.rejected_because:
                print(f"           rejected: {option.rejected_because}")
        if row.draft.reasoning:
            print(f"     why {row.draft.reasoning}")
        if show_passage:
            # The passage beside the draft, which is the whole point of the
            # queue: accepting has to be a judgement about evidence rather than
            # about how well the draft reads.
            excerpt = " ".join(row.source_text.split())[:400]
            print(f"     ┃  {excerpt}")
        print()
    print("Accept with `decisions accept <id>`, reject with `decisions reject <id>`.")
    return 0


async def run_decisions_accept(settings: Settings, *, suggestion_id: str) -> int:
    container = Container.build(settings)
    try:
        decision_id = await decision_suggest.accept(
            container.database.session_factory, UUID(suggestion_id)
        )
    except (decisions.UnknownDecision, decision_suggest.AlreadyReviewed) as exc:
        print(str(exc))
        return 1
    except decisions.InvalidDecision as exc:
        print(f"refused: {exc}")
        return 1
    finally:
        await container.dispose()
    print(f"recorded {decision_id}")
    print("Review it with `decisions show`; confidence and assumptions are yours to add.")
    return 0


async def run_decisions_reject(settings: Settings, *, suggestion_id: str) -> int:
    container = Container.build(settings)
    try:
        await decision_suggest.reject(
            container.database.session_factory, UUID(suggestion_id)
        )
    except (decisions.UnknownDecision, decision_suggest.AlreadyReviewed) as exc:
        print(str(exc))
        return 1
    finally:
        await container.dispose()
    print("rejected; the passage will not be proposed again")
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

    rel_parser = commands.add_parser(
        "extract-relationships",
        help="extract typed relationships between resolved entities",
    )
    rel_parser.add_argument("--source", help="limit to one source by name")
    rel_parser.add_argument("--limit", type=int, help="stop after this many memories")
    rel_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be extracted without calling the model",
    )

    resolve_parser = commands.add_parser(
        "resolve-entities", help="merge entities that refer to the same thing"
    )
    resolve_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print proposed merges with evidence, changing nothing",
    )
    resolve_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"auto-merge at or above this confidence (default {DEFAULT_THRESHOLD})",
    )
    resolve_parser.add_argument(
        "--limit", type=int, default=40, help="how many proposals to print"
    )

    entity_parser = commands.add_parser("entity", help="inspect and edit entity merges")
    entity_commands = entity_parser.add_subparsers(dest="entity_command", required=True)

    merges_parser = entity_commands.add_parser("merges", help="list the merge ledger")
    merges_parser.add_argument(
        "--pending", action="store_true", help="only the review queue"
    )
    merges_parser.add_argument("--strategy", help="filter by strategy")
    merges_parser.add_argument("--limit", type=int, default=50)

    merge_parser = entity_commands.add_parser("merge", help="merge two entities by hand")
    merge_parser.add_argument("winner", help="id or name of the entity to keep")
    merge_parser.add_argument("loser", help="id or name of the entity to merge away")

    unmerge_parser = entity_commands.add_parser("unmerge", help="undo a merge")
    unmerge_parser.add_argument("merge_id", help="id from `entity merges`")

    graph_parser = commands.add_parser("graph", help="the Neo4j projection")
    graph_commands = graph_parser.add_subparsers(dest="graph_command", required=True)

    graph_rebuild = graph_commands.add_parser(
        "rebuild", help="clear the graph and re-project everything from Postgres"
    )
    graph_rebuild.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be projected without touching the graph",
    )

    graph_commands.add_parser(
        "verify", help="compare the graph against Postgres; exits non-zero on divergence"
    )

    graph_sync_parser = graph_commands.add_parser(
        "sync", help="re-project one neighbourhood, the way the job does"
    )
    graph_sync_parser.add_argument(
        "--memory", action="append", default=[], metavar="ID", help="a memory id"
    )
    graph_sync_parser.add_argument(
        "--entity", action="append", default=[], metavar="ID", help="an entity id"
    )

    entity_stats_parser = commands.add_parser(
        "entity-stats", help="report entities, mentions, and the duplicate problem"
    )
    entity_stats_parser.add_argument(
        "--top", type=int, default=20, help="how many entities to list by mention count"
    )

    timeline = commands.add_parser(
        "timeline", help="activity per period, by when things happened"
    )
    # `--from` is a Python keyword, so the destination is named rather than
    # derived; the flag itself is what a person would type.
    timeline.add_argument(
        "--from", dest="start", help="ISO date or timestamp, default the earliest dated memory"
    )
    timeline.add_argument(
        "--to", dest="end", help="ISO date or timestamp, exclusive, default just past the latest"
    )
    timeline.add_argument(
        "--period",
        choices=[value.value for value in Period],
        default=Period.MONTH.value,
        help="the calendar grain each bar covers",
    )
    timeline.add_argument("--source", help="limit to one source by name")

    gaps = commands.add_parser(
        "gaps", help="stretches with activity either side and none during"
    )
    gaps.add_argument(
        "--min-days",
        type=float,
        default=30.0,
        help="the shortest silence worth reporting",
    )
    gaps.add_argument("--source", help="limit to one source by name")

    as_of_parser = commands.add_parser(
        "as-of", help="what the system had ingested at a past instant"
    )
    as_of_parser.add_argument("date", help="ISO date or timestamp, interpreted as UTC")

    evolution_parser = commands.add_parser(
        "evolution", help="how one item changed across its versions"
    )
    evolution_parser.add_argument("source", help="source name")
    evolution_parser.add_argument("path", help="external key, e.g. README.md")
    evolution_parser.add_argument(
        "--no-summary",
        dest="summarize",
        action="store_false",
        help="skip the language model and print the diffs alone",
    )
    evolution_parser.add_argument(
        "--refresh",
        action="store_true",
        help="regenerate cached summaries instead of reading them",
    )

    decide = commands.add_parser(
        "decide", help="record a decision: what was chosen, what else, and why"
    )
    decide.add_argument("--question", default="", help="what was being decided")
    decide.add_argument("--chosen", default="", help="what was picked")
    decide.add_argument("--reasoning", help="why")
    decide.add_argument(
        "--confidence",
        type=float,
        help="how confident you are right now, 0 to 1. Never refreshed later",
    )
    decide.add_argument("--expected", help="what you expect to happen")
    decide.add_argument(
        "--option",
        dest="options",
        action="append",
        default=[],
        metavar="TEXT[::WHY]",
        help=(
            "an alternative that was considered, optionally with the reason it "
            "lost after '::'. At least one is required: a decision with no "
            "alternatives is a description"
        ),
    )
    decide.add_argument(
        "--assumption",
        dest="assumptions",
        action="append",
        default=[],
        metavar="TEXT",
        help="something that has to be true for this to be right; repeatable",
    )
    decide.add_argument(
        "--evidence",
        dest="evidence",
        action="append",
        default=[],
        metavar="SOURCE:PATH[#CHUNK][::RELATION]",
        help=(
            "a memory that informed it, records it, or contradicts it; "
            "relation defaults to 'informed'"
        ),
    )
    decide.add_argument(
        "--decided", help="ISO date or timestamp; defaults to now, provenance 'declared'"
    )
    decide.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "ask for options, reasoning, confidence, expected outcome and "
            "assumptions one at a time"
        ),
    )

    decisions_parser = commands.add_parser(
        "decisions", help="list, inspect, edit and review decisions"
    )
    decisions_commands = decisions_parser.add_subparsers(
        dest="decisions_command", required=True
    )

    decisions_list = decisions_commands.add_parser("list", help="every decision, newest first")
    decisions_list.add_argument(
        "--status", choices=[value.value for value in DecisionStatus]
    )
    decisions_list.add_argument("--limit", type=int, default=100)

    decisions_show = decisions_commands.add_parser(
        "show", help="one decision, with its options, assumptions and evidence"
    )
    decisions_show.add_argument("decision_id")

    decisions_edit = decisions_commands.add_parser("edit", help="amend one decision")
    decisions_edit.add_argument("decision_id")
    decisions_edit.add_argument("--question")
    decisions_edit.add_argument("--chosen")
    decisions_edit.add_argument("--reasoning")
    decisions_edit.add_argument("--expected")
    decisions_edit.add_argument(
        "--status", choices=[value.value for value in DecisionStatus]
    )
    decisions_edit.add_argument(
        "--option",
        dest="options",
        action="append",
        default=[],
        metavar="TEXT[::WHY]",
        help="replaces the whole option list when given",
    )
    decisions_edit.add_argument(
        "--assumption",
        dest="assumptions",
        action="append",
        default=[],
        metavar="TEXT",
        help="replaces the whole assumption list when given",
    )
    # Deliberately no `--confidence` and no `--decided`. Both are records of
    # what somebody believed at a moment, and an edit that moved either would
    # make the calibration M5.2 measures a measurement of hindsight.

    decisions_link = decisions_commands.add_parser(
        "link", help="attach memories to a decision as evidence"
    )
    decisions_link.add_argument("decision_id")
    decisions_link.add_argument(
        "--evidence",
        dest="evidence",
        action="append",
        default=[],
        required=True,
        metavar="SOURCE:PATH[#CHUNK][::RELATION]",
    )

    decisions_suggest = decisions_commands.add_parser(
        "suggest",
        help="propose drafts from the corpus into the review queue; commits nothing",
    )
    decisions_suggest.add_argument("--source", help="limit to one source by name")
    decisions_suggest.add_argument(
        "--limit", type=int, default=20, help="how many passages to examine"
    )

    decisions_review = decisions_commands.add_parser(
        "review", help="the suggestion queue, with the passage each draft came from"
    )
    decisions_review.add_argument(
        "--status",
        choices=[value.value for value in SuggestionStatus],
        default=SuggestionStatus.PENDING.value,
    )
    decisions_review.add_argument("--limit", type=int, default=50)
    decisions_review.add_argument(
        "--no-passage",
        dest="show_passage",
        action="store_false",
        help="hide the source passage; it is shown by default because it is the evidence",
    )

    decisions_accept = decisions_commands.add_parser(
        "accept", help="turn one suggestion into a decision, keeping its passage as evidence"
    )
    decisions_accept.add_argument("suggestion_id")

    decisions_reject = decisions_commands.add_parser(
        "reject", help="mark a suggestion as not a decision; the row stays"
    )
    decisions_reject.add_argument("suggestion_id")

    outcome_parser = commands.add_parser(
        "outcome", help="record what happened after a decision, as somebody who saw it"
    )
    outcome_parser.add_argument("decision_id")
    outcome_parser.add_argument(
        "--verdict",
        required=True,
        choices=[value.value for value in OutcomeVerdict],
        help="'too_early' is a real answer: most decisions have no outcome yet",
    )
    outcome_parser.add_argument("--description", required=True, help="what happened")
    outcome_parser.add_argument(
        "--observed", help="ISO date or timestamp; defaults to now, provenance 'declared'"
    )
    outcome_parser.add_argument(
        "--evidence",
        dest="evidence",
        action="append",
        default=[],
        metavar="SOURCE:PATH[#CHUNK]",
        help="a memory showing it happened; repeatable",
    )
    # Deliberately no `--confidence` and no `--kind`. This command is the
    # declared path: it records confidence 1.0 because saying you observed
    # something is what certainty about an observation means, and a hedged
    # verdict belongs in the suggestion queue instead.

    outcomes_parser = commands.add_parser(
        "outcomes", help="propose, review and summarise outcomes"
    )
    outcomes_commands = outcomes_parser.add_subparsers(
        dest="outcomes_command", required=True
    )

    outcomes_suggest = outcomes_commands.add_parser(
        "suggest",
        help="find candidate outcomes with the temporal layer; commits nothing",
    )
    outcomes_suggest.add_argument(
        "--decision", dest="decision", help="one decision id instead of every open one"
    )
    outcomes_suggest.add_argument(
        "--window-days",
        dest="window_days",
        type=float,
        help=(
            "override the per-decision window. By default it is derived from the "
            "decision's own confidence — a heuristic, not a measurement"
        ),
    )
    outcomes_suggest.add_argument(
        "--limit", type=int, default=10, help="candidates per decision"
    )

    outcomes_review = outcomes_commands.add_parser(
        "review", help="the queue, with the temporal gap and shared entities stated"
    )
    outcomes_review.add_argument(
        "--status",
        choices=[value.value for value in SuggestionStatus],
        default=SuggestionStatus.PENDING.value,
    )
    outcomes_review.add_argument("--limit", type=int, default=50)
    outcomes_review.add_argument(
        "--no-passage",
        dest="show_passage",
        action="store_false",
        help="hide the candidate's text; it is shown by default because it is the evidence",
    )

    outcomes_accept = outcomes_commands.add_parser(
        "accept", help="write the outcome; recorded as 'inferred', never as observed"
    )
    outcomes_accept.add_argument("suggestion_id")

    outcomes_reject = outcomes_commands.add_parser(
        "reject", help="mark a candidate as not an outcome; the row stays"
    )
    outcomes_reject.add_argument("suggestion_id")

    outcomes_commands.add_parser(
        "rate", help="worked/failed/mixed, with too_early and undecided outside the rate"
    )

    assumption_parser = commands.add_parser(
        "assumption", help="record whether one assumption held"
    )
    assumption_parser.add_argument("assumption_id")
    assumption_parser.add_argument(
        "--held",
        required=True,
        help=(
            "true, false or partially. 'partially' exists because almost "
            "nothing anybody assumes is cleanly right or wrong"
        ),
    )
    assumption_parser.add_argument("--note", help="why you reached that verdict")
    assumption_parser.add_argument(
        "--evidence",
        dest="evidence",
        action="append",
        default=[],
        metavar="SOURCE:PATH[#CHUNK]",
        help="a memory you actually used; repeatable",
    )

    assumptions_parser = commands.add_parser(
        "assumptions", help="review, group and report on assumptions"
    )
    assumptions_commands = assumptions_parser.add_subparsers(
        dest="assumptions_command", required=True
    )

    assumptions_review = assumptions_commands.add_parser(
        "review", help="walk assumptions with their decision and outcome"
    )
    assumptions_review.add_argument("--decision", dest="decision")
    assumptions_review.add_argument(
        "--unevaluated",
        action="store_true",
        help="only the ones nobody has judged yet",
    )
    assumptions_review.add_argument("--limit", type=int, default=200)

    assumptions_suggest = assumptions_commands.add_parser(
        "suggest",
        help="find memories bearing on an assumption; proposes evidence, never a verdict",
    )
    assumptions_suggest.add_argument("--decision", dest="decision")
    assumptions_suggest.add_argument("--limit", type=int, default=20)

    assumptions_group = assumptions_commands.add_parser(
        "group", help="cluster assumptions that say the same thing"
    )
    assumptions_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="report what would group and what would queue, writing nothing",
    )

    assumptions_candidates = assumptions_commands.add_parser(
        "candidates", help="pairs the embedder was unsure about"
    )
    assumptions_candidates.add_argument(
        "--status",
        choices=[value.value for value in MergeStatus],
        default=MergeStatus.PENDING.value,
    )
    assumptions_candidates.add_argument("--limit", type=int, default=50)

    assumptions_accept = assumptions_commands.add_parser(
        "accept", help="put both assumptions of a pair in one group"
    )
    assumptions_accept.add_argument("candidate_id")

    assumptions_reject = assumptions_commands.add_parser(
        "reject", help="mark a pair as not the same belief; the row stays"
    )
    assumptions_reject.add_argument("candidate_id")

    assumptions_commands.add_parser(
        "stats",
        help="totals, hold rate, and every group with more than one member",
    )

    patterns_parser = commands.add_parser(
        "patterns", help="behavioural patterns across decisions, with their evidence"
    )
    patterns_commands = patterns_parser.add_subparsers(
        dest="patterns_command", required=True
    )

    patterns_discover = patterns_commands.add_parser(
        "discover", help="run every detector; emits only what clears the bar"
    )
    patterns_discover.add_argument(
        "--min-support",
        dest="min_support",
        type=int,
        default=DEFAULT_MIN_SUPPORT,
        help=(
            f"distinct decisions a pattern needs (default {DEFAULT_MIN_SUPPORT}). "
            f"Raise it to be stricter; lowering it to produce output is how a "
            f"tool starts inventing findings"
        ),
    )
    patterns_discover.add_argument(
        "--window-days",
        dest="window_days",
        type=float,
        default=90.0,
        help="the timing detector's threshold for 'later than expected'",
    )

    patterns_list = patterns_commands.add_parser("list", help="patterns found so far")
    patterns_list.add_argument(
        "--kind", choices=[value.value for value in PatternKind]
    )
    patterns_list.add_argument(
        "--include-dismissed", dest="include_dismissed", action="store_true"
    )

    patterns_show = patterns_commands.add_parser(
        "show", help="one pattern, with its supporting and contradicting evidence"
    )
    patterns_show.add_argument("pattern_id")

    patterns_dismiss = patterns_commands.add_parser(
        "dismiss", help="reject a pattern permanently; discovery will not re-propose it"
    )
    patterns_dismiss.add_argument("pattern_id")
    patterns_dismiss.add_argument("--reason", required=True)

    patterns_commands.add_parser(
        "calibration",
        help="stated confidence against actual verdicts, with the interval each supports",
    )

    # The verb the milestone names, kept as its own top-level command rather than
    # `patterns reflect`: generating prose about somebody is a different act from
    # counting their decisions, and it should not read as a subcommand of it.
    events_parser = commands.add_parser(
        "events", help="the external event stream, and how fast it is drained"
    )
    events_commands = events_parser.add_subparsers(
        dest="events_command", required=True
    )
    events_tail = events_commands.add_parser(
        "tail", help="recent events, oldest first"
    )
    events_tail.add_argument("--kind", choices=[member.value for member in EventKind])
    events_tail.add_argument("--limit", type=int, default=20)
    events_tail.add_argument(
        "--follow",
        action="store_true",
        help="poll for new events once a second until interrupted",
    )
    events_commands.add_parser(
        "stats",
        help="events by kind, processed against pending, and the mean latency",
    )

    reflect_parser = commands.add_parser(
        "reflect",
        help="describe a pattern in prose, with citations; refuses below the bar",
    )
    reflect_parser.add_argument(
        "--pattern", dest="pattern_id", help="one pattern by id"
    )
    reflect_parser.add_argument(
        "--all",
        dest="all_patterns",
        action="store_true",
        help="every pattern that clears the confidence bar",
    )
    reflect_parser.add_argument(
        "--min-confidence",
        dest="threshold",
        type=float,
        default=REFLECTION_MIN_CONFIDENCE,
        help=(
            f"the confidence a pattern needs before anything is written about it "
            f"(default {REFLECTION_MIN_CONFIDENCE:.2f}). Deliberately above the bar "
            f"for the pattern itself. Lowering it to produce output is exactly how "
            f"a tool starts writing horoscopes"
        ),
    )
    reflect_parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "replace an existing reflection. Never overrides a dismissal: a "
            "rejection a re-run undid would not be a rejection"
        ),
    )

    reflections_parser = commands.add_parser(
        "reflections", help="reflections already written, with their citations"
    )
    reflections_commands = reflections_parser.add_subparsers(
        dest="reflections_command", required=True
    )
    reflections_list = reflections_commands.add_parser(
        "list", help="every reflection, with the decisions it cites"
    )
    reflections_list.add_argument(
        "--include-dismissed", dest="include_dismissed", action="store_true"
    )
    reflections_ack = reflections_commands.add_parser(
        "acknowledge", help="record that you read it; not agreement"
    )
    reflections_ack.add_argument("reflection_id")
    reflections_dismiss = reflections_commands.add_parser(
        "dismiss",
        help="\"this is wrong about me\"; stops it being shown or regenerated",
    )
    reflections_dismiss.add_argument("reflection_id")
    reflections_dismiss.add_argument("--reason", required=True)

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

    if args.command == "extract-relationships":
        return asyncio.run(
            run_extract_relationships(
                settings, source=args.source, limit=args.limit, dry_run=args.dry_run
            )
        )

    if args.command == "resolve-entities":
        return asyncio.run(
            run_resolve_entities(
                settings,
                dry_run=args.dry_run,
                threshold=args.threshold,
                limit=args.limit,
            )
        )

    if args.command == "graph":
        if args.graph_command == "rebuild":
            return asyncio.run(run_graph_rebuild(settings, dry_run=args.dry_run))
        if args.graph_command == "verify":
            return asyncio.run(run_graph_verify(settings))
        return asyncio.run(
            run_graph_sync(settings, memories=args.memory, entities=args.entity)
        )

    if args.command == "entity":
        if args.entity_command == "merges":
            return asyncio.run(
                run_list_merges(
                    settings,
                    pending=args.pending,
                    strategy=args.strategy,
                    limit=args.limit,
                )
            )
        if args.entity_command == "merge":
            return asyncio.run(
                run_manual_merge(settings, winner=args.winner, loser=args.loser)
            )
        if args.entity_command == "unmerge":
            return asyncio.run(run_unmerge(settings, merge_id=args.merge_id))

    if args.command == "entity-stats":
        return asyncio.run(run_entity_stats(settings, top=args.top))

    if args.command == "timeline":
        return asyncio.run(
            run_timeline(
                settings,
                start=args.start,
                end=args.end,
                period=Period(args.period),
                source=args.source,
            )
        )

    if args.command == "gaps":
        return asyncio.run(
            run_gaps(settings, min_days=args.min_days, source=args.source)
        )

    if args.command == "as-of":
        return asyncio.run(run_as_of(settings, moment=args.date))

    if args.command == "evolution":
        return asyncio.run(
            run_evolution(
                settings,
                source=args.source,
                path=args.path,
                summarize=args.summarize,
                refresh=args.refresh,
            )
        )

    if args.command == "decide":
        return asyncio.run(
            run_decide(
                settings,
                question=args.question,
                chosen=args.chosen,
                reasoning=args.reasoning,
                confidence=args.confidence,
                expected=args.expected,
                options=args.options,
                assumptions=args.assumptions,
                evidence=args.evidence,
                decided=args.decided,
                interactive=args.interactive,
            )
        )

    if args.command == "decisions":
        if args.decisions_command == "list":
            return asyncio.run(
                run_decisions_list(settings, status=args.status, limit=args.limit)
            )
        if args.decisions_command == "show":
            return asyncio.run(
                run_decisions_show(settings, decision_id=args.decision_id)
            )
        if args.decisions_command == "edit":
            return asyncio.run(
                run_decisions_edit(
                    settings,
                    decision_id=args.decision_id,
                    question=args.question,
                    chosen=args.chosen,
                    reasoning=args.reasoning,
                    expected=args.expected,
                    status=args.status,
                    options=args.options,
                    assumptions=args.assumptions,
                )
            )
        if args.decisions_command == "link":
            return asyncio.run(
                run_decisions_link(
                    settings, decision_id=args.decision_id, evidence=args.evidence
                )
            )
        if args.decisions_command == "suggest":
            return asyncio.run(
                run_decisions_suggest(settings, source=args.source, limit=args.limit)
            )
        if args.decisions_command == "review":
            return asyncio.run(
                run_decisions_review(
                    settings,
                    status=args.status,
                    limit=args.limit,
                    show_passage=args.show_passage,
                )
            )
        if args.decisions_command == "accept":
            return asyncio.run(
                run_decisions_accept(settings, suggestion_id=args.suggestion_id)
            )
        if args.decisions_command == "reject":
            return asyncio.run(
                run_decisions_reject(settings, suggestion_id=args.suggestion_id)
            )

    if args.command == "outcome":
        return asyncio.run(
            run_outcome(
                settings,
                decision_id=args.decision_id,
                verdict=args.verdict,
                description=args.description,
                observed=args.observed,
                evidence=args.evidence,
            )
        )

    if args.command == "outcomes":
        if args.outcomes_command == "suggest":
            return asyncio.run(
                run_outcomes_suggest(
                    settings,
                    decision_id=args.decision,
                    window_days=args.window_days,
                    limit=args.limit,
                )
            )
        if args.outcomes_command == "review":
            return asyncio.run(
                run_outcomes_review(
                    settings,
                    status=args.status,
                    limit=args.limit,
                    show_passage=args.show_passage,
                )
            )
        if args.outcomes_command == "accept":
            return asyncio.run(
                run_outcomes_accept(settings, suggestion_id=args.suggestion_id)
            )
        if args.outcomes_command == "reject":
            return asyncio.run(
                run_outcomes_reject(settings, suggestion_id=args.suggestion_id)
            )
        if args.outcomes_command == "rate":
            return asyncio.run(run_outcomes_rate(settings))

    if args.command == "patterns":
        if args.patterns_command == "discover":
            return asyncio.run(
                run_patterns_discover(
                    settings,
                    min_support=args.min_support,
                    window_days=args.window_days,
                )
            )
        if args.patterns_command == "list":
            return asyncio.run(
                run_patterns_list(
                    settings,
                    kind=args.kind,
                    include_dismissed=args.include_dismissed,
                )
            )
        if args.patterns_command == "show":
            return asyncio.run(run_patterns_show(settings, pattern_id=args.pattern_id))
        if args.patterns_command == "dismiss":
            return asyncio.run(
                run_patterns_dismiss(
                    settings, pattern_id=args.pattern_id, reason=args.reason
                )
            )
        if args.patterns_command == "calibration":
            return asyncio.run(run_patterns_calibration(settings))

    if args.command == "events":
        if args.events_command == "tail":
            return asyncio.run(
                run_events_tail(
                    settings,
                    kind=args.kind,
                    limit=args.limit,
                    follow=args.follow,
                )
            )
        if args.events_command == "stats":
            return asyncio.run(run_events_stats(settings))

    if args.command == "reflect":
        if not args.all_patterns and args.pattern_id is None:
            # Neither flag is not "everything". Generating prose about a person
            # is the one operation here that should never happen because a
            # command was run without arguments.
            print("give --pattern <id> or --all")
            return 2
        return asyncio.run(
            run_reflect(
                settings,
                pattern_id=args.pattern_id,
                threshold=args.threshold,
                regenerate=args.regenerate,
            )
        )

    if args.command == "reflections":
        if args.reflections_command == "list":
            return asyncio.run(
                run_reflections_list(
                    settings, include_dismissed=args.include_dismissed
                )
            )
        if args.reflections_command == "acknowledge":
            return asyncio.run(
                run_reflections_acknowledge(
                    settings, reflection_id=args.reflection_id
                )
            )
        if args.reflections_command == "dismiss":
            return asyncio.run(
                run_reflections_dismiss(
                    settings,
                    reflection_id=args.reflection_id,
                    reason=args.reason,
                )
            )

    if args.command == "assumption":
        return asyncio.run(
            run_assumption_evaluate(
                settings,
                assumption_id=args.assumption_id,
                held=args.held,
                note=args.note,
                evidence=args.evidence,
            )
        )

    if args.command == "assumptions":
        if args.assumptions_command == "review":
            return asyncio.run(
                run_assumptions_review(
                    settings,
                    decision_id=args.decision,
                    unevaluated=args.unevaluated,
                    limit=args.limit,
                )
            )
        if args.assumptions_command == "suggest":
            return asyncio.run(
                run_assumptions_suggest(
                    settings, decision_id=args.decision, limit=args.limit
                )
            )
        if args.assumptions_command == "group":
            return asyncio.run(run_assumptions_group(settings, dry_run=args.dry_run))
        if args.assumptions_command == "candidates":
            return asyncio.run(
                run_assumptions_candidates(
                    settings, status=args.status, limit=args.limit
                )
            )
        if args.assumptions_command == "accept":
            return asyncio.run(
                run_assumptions_accept(settings, candidate_id=args.candidate_id)
            )
        if args.assumptions_command == "reject":
            return asyncio.run(
                run_assumptions_reject(settings, candidate_id=args.candidate_id)
            )
        if args.assumptions_command == "stats":
            return asyncio.run(run_assumptions_stats(settings))

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
