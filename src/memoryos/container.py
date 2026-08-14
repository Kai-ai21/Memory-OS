"""Composition root.

One place assembles the object graph, so the CLI, the API, and the worker all
run against the same wiring rather than three slightly different versions of it.
"""

from dataclasses import dataclass

import structlog

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.db.engine import Database
from memoryos.adapters.db.job_queue import PostgresJobQueue
from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.shadow import PostgresShadowSchema
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.embedding.sentence_transformers import (
    SentenceTransformerEmbedder,
    build_embedder,
)
from memoryos.adapters.extraction.llm import LlmEntityExtractor
from memoryos.adapters.graph.neo4j_store import Neo4jGraphStore
from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.adapters.llm.gemini import GeminiLanguageModel
from memoryos.adapters.llm.groq import GroqLanguageModel
from memoryos.adapters.parsers.registry import ParserRegistry
from memoryos.adapters.parsers.registry import build_default_registry as build_parser_registry
from memoryos.adapters.reranking.cross_encoder import CrossEncoderReranker
from memoryos.application.agent.library import build_registry as build_tools
from memoryos.application.agent.planner import MultiHopPlanner
from memoryos.application.agent.tools import ToolRegistry
from memoryos.application.agent.verify import VerifiedAgent
from memoryos.application.answering import AnswerQuestion
from memoryos.application.context_engine import (
    AssembleContext,
    ContextRequest,
)
from memoryos.application.embed import EmbedMemory
from memoryos.application.events import (
    ContextEventHandler,
    EventBus,
    LogEventHandler,
    build_default_bus,
    focus_of,
)
from memoryos.application.extraction import ExtractEntities
from memoryos.application.graph_expand import ExpandThroughGraph
from memoryos.application.graph_sync import SyncGraph
from memoryos.application.jobs.handlers import build_default_registry
from memoryos.application.jobs.registry import HandlerRegistry
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.ports import (
    Chunker,
    Embedder,
    LanguageModel,
    supports_tools,
)
from memoryos.application.relationships import ExtractRelationships
from memoryos.application.replay import ReplayCorpus
from memoryos.application.resolution import DEFAULT_THRESHOLD, ResolveEntities
from memoryos.application.search import FusionWeights, SearchMemories
from memoryos.application.surfacing import build_handler as build_surfacing_handler
from memoryos.application.sync import SyncSource
from memoryos.config import Settings
from memoryos.domain.events import Event

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class Container:
    settings: Settings
    database: Database
    blobs: FilesystemBlobStore
    connector: FilesystemConnector
    queue: PostgresJobQueue
    parsers: ParserRegistry
    embedder: SentenceTransformerEmbedder
    cache: PostgresEmbeddingCache
    vectors: PgVectorStore
    keywords: PostgresKeywordStore
    reranker: CrossEncoderReranker | None
    chunker: StructuralChunker
    graph: Neo4jGraphStore

    @classmethod
    def build(cls, settings: Settings) -> "Container":
        database = Database.from_url(settings.database_url, echo=settings.db_echo)
        blobs = FilesystemBlobStore(settings.blob_root)
        embedder = build_embedder(settings)
        # Sized from the model's real window, not from a number somebody chose.
        chunker = StructuralChunker(embedder)
        _assert_window_alignment(chunker, embedder)
        return cls(
            settings=settings,
            database=database,
            blobs=blobs,
            connector=FilesystemConnector(blobs),
            queue=PostgresJobQueue(database.session_factory),
            parsers=build_parser_registry(),
            embedder=embedder,
            cache=PostgresEmbeddingCache(database.session_factory),
            chunker=chunker,
            vectors=PgVectorStore(
                database.session_factory,
                embedder,
                default_ef_search=settings.hnsw_ef_search,
            ),
            # No model, no configuration, nothing to warm up: the index is
            # maintained by Postgres from a generated column, so the lexical
            # half of retrieval costs a session factory.
            keywords=PostgresKeywordStore(database.session_factory),
            # Constructed even when disabled — construction loads nothing, and a
            # None here is what `rerank_enabled=false` means downstream.
            reranker=(
                CrossEncoderReranker(
                    settings.reranker_model, cache_dir=settings.hf_home
                )
                if settings.rerank_enabled
                else None
            ),
            # Always constructed, never connected here. Building a driver is
            # local work — it validates the URI and allocates an empty pool —
            # so a machine with no Neo4j running still gets a working container
            # and a working CLI. The graph is the only thing that stops working,
            # which is the whole point of it being a projection.
            graph=Neo4jGraphStore(
                settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
            ),
        )

    def registry(self) -> HandlerRegistry:
        return build_default_registry(
            session_factory=self.database.session_factory,
            connector=self.connector,
            blob_store=self.blobs,
            embedder=self.embedder,
            cache=self.cache,
            chunker=self.chunker,
            batch_size=self.settings.embedding_batch_size,
            extractor=self.optional_extractor(),
            graph=self.graph,
            # The queue is wired into the registry as a *collaborator*, not only
            # as the thing the worker drains: extraction and relationship
            # extraction enqueue a `SYNC_GRAPH` job rather than writing to Neo4j.
            queue=self.queue,
            # M6.0. The same bus the API dispatches through, built from the same
            # function: the API writes a handler *name* into a job payload and
            # the worker looks it up, so two independently-assembled lists would
            # show up as jobs no worker can run — which is exactly what happened
            # the first time this milestone was run end to end, and what the
            # `handle_event` registration test now pins.
            bus=self.event_bus(),
            # M6.2. Built here rather than lazily, unlike the event handler's:
            # a worker registering this type will run it, so there is nothing to
            # defer and a failure to construct should surface at startup rather
            # than on the first job.
            assemble=self.assemble_context(),
        )

    def assemble_context(self) -> AssembleContext:
        """The context engine, wired to every phase that contributes to it.

        The embedder doubles as the token counter, for M2.6's reason: it is the
        counter the chunker already sizes against, so the budget is spent in the
        same unit the corpus was built in.
        """
        return AssembleContext(
            self.database.session_factory,
            self.search(),
            self.embedder,
            self.embedder,
            expand=self.graph_expansion(),
        )

    def event_bus(self) -> EventBus:
        """The event bus, with every handler this build knows about.

        Built per call rather than cached, and that is safe because the bus holds
        no state at all beyond its subscriptions — it enqueues into a transaction
        the caller owns rather than running
        anything, so two instances cannot disagree about work in flight.
        """
        return build_default_bus(
            [
                LogEventHandler(),
                # Assembly is built lazily, inside the handler's call, because
                # constructing it loads the embedder — and a deployment that only
                # receives events should not pay for a model it may never use.
                ContextEventHandler(_LazyAssembler(self)),
                # M6.3, and a separate handler rather than a branch inside the
                # one above. They want opposite things from a failure: assembly
                # that raises should retry with M1.2's backoff, and a gate that
                # raises must not take the assembly down with it, because the
                # cached context is the useful half and the interruption is the
                # speculative one. One job each is what makes that true.
                build_surfacing_handler(
                    self.database.session_factory,
                    self.assemble_context,
                    focus_of,
                ),
            ]
        )

    def optional_extractor(self) -> LlmEntityExtractor | None:
        """The extractor, or None when no API key is configured.

        The worker builds the whole registry at startup, so a raising
        constructor here would stop a keyless deployment from running *any*
        job — sync, normalize and embed included, none of which need a model.
        Registering nothing instead leaves extraction jobs failing permanently
        with "no handler", which is both true and diagnosable, while everything
        upstream of extraction keeps working.
        """
        try:
            return self.extractor()
        except MissingApiKey as exc:
            logger.warning("container.extraction_unavailable", error=str(exc))
            return None

    def extractor(self) -> LlmEntityExtractor:
        """Entity extraction over whichever provider is configured.

        Built here rather than at `build` for the reason `answer` is: it
        constructs a language model, which raises without an API key, and a
        deployment with no key must still have working ingestion and search.
        """
        return LlmEntityExtractor(
            self.language_model(),
            batch_size=self.settings.extraction_batch_size,
            max_tokens=self.settings.extraction_max_tokens,
        )

    def relationships(self) -> ExtractRelationships:
        return ExtractRelationships(
            self.database.session_factory, self.extractor(), self.queue
        )

    def extract(self) -> ExtractEntities:
        return ExtractEntities(
            self.database.session_factory, self.extractor(), self.queue
        )

    def resolve(self, *, threshold: float = DEFAULT_THRESHOLD) -> ResolveEntities:
        """Entity resolution, with the queue that announces what it merged.

        Built here rather than in the CLI so that the enqueue cannot be forgotten
        by one caller: a merge that nobody announced leaves the graph asserting an
        entity Postgres has stopped having, and the only thing that would notice
        is `graph verify`.
        """
        return ResolveEntities(
            self.database.session_factory,
            self.embedder,
            self.queue,
            threshold=threshold,
        )

    def graph_sync(self) -> SyncGraph:
        return SyncGraph(self.database.session_factory, self.graph)

    def sync(self) -> SyncSource:
        return SyncSource(self.database.session_factory, self.connector, self.blobs)

    def normalize(self) -> NormalizeMemory:
        return NormalizeMemory(
            self.database.session_factory, self.blobs, self.parsers, self.chunker
        )

    def search(self, weights: FusionWeights | None = None) -> SearchMemories:
        return SearchMemories(
            self.database.session_factory,
            self.embedder,
            self.vectors,
            self.keywords,
            weights or self.weights(),
            self.reranker,
            rerank_candidates=self.settings.rerank_candidates,
            expand=self.graph_expansion(),
            seed_memories=self.settings.graph_seed_memories,
            temporal_intent=self.settings.temporal_intent_enabled,
            temporal_recency_weight=self.settings.temporal_recency_weight,
        )

    def graph_expansion(self) -> ExpandThroughGraph:
        """M3.5's expansion, wired to the same graph store everything else uses.

        Always constructed, never connected here — the same reasoning as
        `Container.build`'s graph store. An unreachable Neo4j makes the expansion
        return nothing, which is a hybrid search, rather than making search fail.
        """
        return ExpandThroughGraph(
            self.database.session_factory,
            self.graph,
            depth=self.settings.graph_depth,
            hub_ratio=self.settings.graph_hub_ratio,
            seed_memories=self.settings.graph_seed_memories,
        )

    def answer(self) -> AnswerQuestion:
        """The grounded-answer use case.

        The language model is constructed here rather than at `build`, so a
        deployment with no key still has a working search — `MissingApiKey`
        surfaces when somebody asks a question, which is the only operation
        that needs one.

        The embedder doubles as the token counter: it is the `TokenCounter` the
        chunker already sizes against, so the budget is counted in the same
        unit the corpus was built in.
        """
        return AnswerQuestion(
            self.database.session_factory,
            self.search(),
            self.language_model(),
            self.embedder,
            weights=self.weights(),
            token_budget=self.settings.answer_token_budget,
        )

    def language_model(self) -> LanguageModel:
        return build_language_model(self.settings)

    def tools(self) -> ToolRegistry:
        """The six tools, registered in the order the model reads them.

        Built here rather than in the agent, so the tools are wired from the same
        object graph everything else is: `search_memories` is the same
        `SearchMemories` the CLI and the API call, with the same weights and the
        same reranker. A tool that constructed its own would be a second
        configuration nobody knew existed until the two disagreed.
        """
        registry = ToolRegistry()
        for tool in build_tools(
            sessions=self.database.session_factory,
            search=self.search(),
            expand=self.graph_expansion(),
            weights=self.weights(),
        ):
            registry.register(tool)
        return registry

    def agent(self) -> VerifiedAgent:
        """M7.1's loop with M7.2's guardrail attached, and no way past it.

        **The verified agent is the only agent this container hands out.** A
        caller cannot ask for the unverified planner, because the one thing an
        ungrounded answer must not depend on is whoever is calling remembering to
        check it — and the CLI, the API and anything after them all come through
        here.

        The check is here rather than at the first failed call, because the
        failure it prevents is unreadable: a provider without tool support
        answers the question from its training data, fluently, and nothing in the
        response says a tool was never offered. Refusing at construction turns
        that into one sentence naming the setting to change.

        The embedder doubles as the token counter *and* as the support judge, for
        the reason `answer` gives: compaction sizes findings in the same unit the
        corpus was chunked in, and a claim is measured against retrieved text
        with the same function that decided the text was retrievable.
        """
        model = self.language_model()
        if supports_tools(model):
            planner = MultiHopPlanner(
                model,
                self.tools(),
                self.embedder,
                max_hops=self.settings.agent_max_hops,
                finding_budget=self.settings.agent_finding_budget,
                max_tokens=self.settings.agent_max_tokens,
            )
            return VerifiedAgent(
                planner,
                self.database.session_factory,
                self.embedder,
                # The same cross-encoder search reranks with, doing the same
                # thing for a different question: there it decides which of ten
                # passages answers a query best, here whether any of twelve says
                # what a sentence claims. `None` when reranking is off, which
                # `judged_by` reports rather than hides.
                self.reranker,
                min_support=self.settings.agent_min_support,
            )
        raise ToolsUnsupported(
            f"{model.model_id} cannot call tools, so the agent has nothing to "
            "drive. Set MEMOS_LLM_PROVIDER to a provider whose adapter "
            "implements `converse` — groq and gemini both do."
        )

    def weights(self) -> FusionWeights:
        """Fusion weights from settings, so `MEMOS_WEIGHT_*` reaches every caller."""
        return FusionWeights(
            vector=self.settings.weight_vector,
            keyword=self.settings.weight_keyword,
            recency=self.settings.weight_recency,
            importance=self.settings.weight_importance,
            graph=self.settings.weight_graph,
        )

    def embed(self) -> EmbedMemory:
        return EmbedMemory(
            self.database.session_factory,
            self.embedder,
            self.cache,
            self.settings.embedding_batch_size,
        )

    def replay(self) -> ReplayCorpus:
        """The replay use case, built to work through whichever tables it is given.

        `NormalizeMemory` and `EmbedMemory` are constructed per session factory
        rather than reused, because a shadow rebuild writes through a different
        one — same use cases, different tables. Anything holding a factory from
        construction would quietly write the live tables during a shadow replay,
        which is the one mistake this design has to make impossible.
        """
        return ReplayCorpus(
            self.database.session_factory,
            make_normalize=lambda sessions: NormalizeMemory(
                sessions,
                self.blobs,
                self.parsers,
                self.chunker,
                # Replay embeds inline immediately afterwards, so a queued job
                # would be work nobody needs doing twice.
                enqueue_followup=False,
            ),
            make_embed=lambda sessions: EmbedMemory(
                sessions, self.embedder, PostgresEmbeddingCache(sessions),
                self.settings.embedding_batch_size,
                # A rebuild must not queue paid work. Extraction is the first
                # stage in this pipeline that costs money per run, and a replay
                # — which exists partly so it can be run routinely to verify the
                # corpus — would otherwise enqueue one model call per memory
                # every time it ran.
                enqueue_followup=False,
            ),
            make_shadow=lambda: PostgresShadowSchema(
                self.settings.database_url, echo=self.settings.db_echo
            ),
            blobs=self.blobs,
            # A full replay clears the graph along with the derived tables. See
            # `ReplayCorpus._clear_graph` for why it is not simply another name
            # in `DERIVED_TABLES`.
            graph=self.graph,
        )

    async def dispose(self) -> None:
        await self.database.dispose()
        await self.graph.close()


def build_language_model(settings: Settings) -> LanguageModel:
    """The provider named by `MEMOS_LLM_PROVIDER`.

    The whole of M2.6a is this function. Everything above it — `AnswerQuestion`,
    the grounding checks, the citation verifier, the CLI — takes a
    `LanguageModel` and cannot tell which one it got, which is what M2.6 put the
    thing behind a Protocol for.

    A module-level function rather than only a method, because selecting a
    provider is a pure function of settings and needs none of the rest of the
    object graph. That is also what makes it testable: `Container.build`
    constructs the embedder, which reaches HuggingFace and takes seconds, so a
    test that went through it would be neither fast nor offline.

    The key is checked by the adapter's own constructor rather than here, so the
    error names the variable the *selected* provider needs. Checking both keys in
    one place would either demand a Gemini key from a Groq deployment or produce
    a message listing two variables and leave the reader to work out which one
    applies to them.
    """
    if settings.llm_provider == "gemini":
        return GeminiLanguageModel(
            settings.gemini_api_key, model_name=settings.llm_model
        )
    return GroqLanguageModel(settings.groq_api_key, model_name=settings.groq_model)


class ToolsUnsupported(RuntimeError):
    """The configured provider cannot be given tools.

    Its own type rather than a `ValueError`, so a caller can tell "this
    deployment has no agent" from "this question was malformed" — the first is a
    setting and the second is a bug.
    """


class WindowMisalignment(RuntimeError):
    """Chunks can be larger than the model will read."""


def _assert_window_alignment(chunker: Chunker, embedder: Embedder) -> None:
    """Refuse to start if chunks could exceed what the model reads.

    The most important line in this milestone. Parameters drift, models get
    swapped, and neither the chunker nor the embedder complains when they stop
    agreeing — the text past the window is simply discarded, silently, and
    retrieval degrades in a way that looks like a model quality problem.

    An assertion does not drift. A future swap to a smaller-window model breaks
    the build instead.
    """
    if chunker.max_tokens > embedder.max_sequence_tokens:
        raise WindowMisalignment(
            f"chunker produces up to {chunker.max_tokens} tokens but "
            f"{embedder.model_id} reads only {embedder.max_sequence_tokens}. "
            f"Everything past the window would be discarded before embedding. "
            f"Size the chunker from the model: "
            f"ChunkerConfig.for_window({embedder.max_sequence_tokens})."
        )


class _LazyAssembler:
    """Builds the context engine on first use, not at subscription time.

    `Container.event_bus()` is called wherever the bus is needed — including by
    the API process, which subscribes handlers on every request that dispatches
    one. Constructing `AssembleContext` eagerly there would load the embedder
    into a process that may never assemble anything, and would do it on the
    request path.

    Holding the container rather than the engine also means the handler picks up
    whatever the container is configured with at the moment it runs, which is the
    behaviour a worker restarting after a settings change should have.
    """

    def __init__(self, container: "Container") -> None:
        self._container = container

    async def __call__(self, focus: str, trigger: Event) -> None:
        assemble = self._container.assemble_context()
        await assemble(ContextRequest(focus=focus, trigger=trigger))
