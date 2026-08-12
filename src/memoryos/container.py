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
from memoryos.application.answering import AnswerQuestion
from memoryos.application.embed import EmbedMemory
from memoryos.application.extraction import ExtractEntities
from memoryos.application.graph_expand import ExpandThroughGraph
from memoryos.application.graph_sync import SyncGraph
from memoryos.application.jobs.handlers import build_default_registry
from memoryos.application.jobs.registry import HandlerRegistry
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.ports import Chunker, Embedder, LanguageModel
from memoryos.application.relationships import ExtractRelationships
from memoryos.application.replay import ReplayCorpus
from memoryos.application.resolution import DEFAULT_THRESHOLD, ResolveEntities
from memoryos.application.search import FusionWeights, SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.config import Settings

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
