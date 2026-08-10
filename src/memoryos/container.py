"""Composition root.

One place assembles the object graph, so the CLI, the API, and the worker all
run against the same wiring rather than three slightly different versions of it.
"""

from dataclasses import dataclass

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
from memoryos.adapters.llm.gemini import GeminiLanguageModel
from memoryos.adapters.parsers.registry import ParserRegistry
from memoryos.adapters.parsers.registry import build_default_registry as build_parser_registry
from memoryos.adapters.reranking.cross_encoder import CrossEncoderReranker
from memoryos.application.answering import AnswerQuestion
from memoryos.application.embed import EmbedMemory
from memoryos.application.jobs.handlers import build_default_registry
from memoryos.application.jobs.registry import HandlerRegistry
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.ports import Chunker, Embedder
from memoryos.application.replay import ReplayCorpus
from memoryos.application.search import FusionWeights, SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.config import Settings


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
        )

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
            GeminiLanguageModel(
                self.settings.gemini_api_key, model_name=self.settings.llm_model
            ),
            self.embedder,
            weights=self.weights(),
            token_budget=self.settings.answer_token_budget,
        )

    def weights(self) -> FusionWeights:
        """Fusion weights from settings, so `MEMOS_WEIGHT_*` reaches every caller."""
        return FusionWeights(
            vector=self.settings.weight_vector,
            keyword=self.settings.weight_keyword,
            recency=self.settings.weight_recency,
            importance=self.settings.weight_importance,
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
            ),
            make_shadow=lambda: PostgresShadowSchema(
                self.settings.database_url, echo=self.settings.db_echo
            ),
            blobs=self.blobs,
        )

    async def dispose(self) -> None:
        await self.database.dispose()


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
