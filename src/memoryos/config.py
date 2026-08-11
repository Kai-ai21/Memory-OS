from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from memoryos.adapters.embedding.sentence_transformers import (
    DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL,
)
from memoryos.adapters.llm.gemini import DEFAULT_MODEL as DEFAULT_LLM_MODEL
from memoryos.adapters.reranking.cross_encoder import (
    DEFAULT_MODEL as DEFAULT_RERANKER_MODEL,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MEMOS_", extra="ignore"
    )

    environment: str = "local"
    database_url: str = "postgresql+asyncpg://memos:memos@localhost:5433/memos"
    # Where the test suite writes. Its own database rather than its own
    # isolation strategy: the integration tests truncate every table, which is
    # the only strategy that survives code under test committing, and pointing
    # that at the development database destroys a working corpus. It did, three
    # times during M2.0a.
    #
    # Compose creates it; `MEMOS_ENVIRONMENT=test` selects it. CI sets neither
    # and is unaffected — its database is disposable, and a second one there
    # would only be a second thing to migrate.
    test_database_url: str = "postgresql+asyncpg://memos:memos@localhost:5433/memos_test"
    db_echo: bool = False
    # Where artifact bytes live. Local directory for now; the BlobStore port is
    # what lets this become object storage without a use case changing.
    blob_root: Path = Path("./var/blobs")
    # Where HuggingFace caches model weights. Pinned explicitly so that a local
    # run and a CI run agree on the location, which is what makes the CI cache
    # key mean anything.
    hf_home: Path = Path("./var/hf")
    # Sourced from the adapter rather than repeated here. Two copies of a
    # model name is how the CLI ended up on a different model from the tests.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = 32
    # HNSW search width per query. Higher recall, higher latency; measured
    # rather than guessed by `memoryos eval-recall`.
    hnsw_ef_search: int = 100
    # How much each ranking counts in the fusion.
    #
    # **Both signals default to zero, and that is the M2.3b result rather than a
    # placeholder.** A 97-combination grid search over the 41-query golden set
    # found that recency monotonically *lowers* nDCG at every importance level
    # it was tried with — 0.735 at weight 0, 0.731 at 0.15, 0.721 at 0.30, 0.707
    # at 0.60 — and that the best importance weight, 0.10, gains 0.0109, which
    # is below the 0.0122 resolution floor M2.3a measured. A gain under the
    # floor is not evidence, so shipping it would be shipping noise with a
    # decimal point.
    #
    # The machinery stays and the weights stay tunable: `MEMOS_WEIGHT_RECENCY`
    # and `MEMOS_WEIGHT_IMPORTANCE` turn them on for a corpus where the answer
    # may differ. This one is a repository of explanatory prose, where when a
    # file was last edited says almost nothing about whether it answers a
    # question about the design.
    weight_vector: float = 1.0
    weight_keyword: float = 1.0
    weight_recency: float = 0.0
    weight_importance: float = 0.0
    # The cross-encoder that rescores the shortlist. Sourced from the adapter
    # for the same reason the embedder is: two copies of a model name is how the
    # CLI ended up running a different model from the tests.
    reranker_model: str = DEFAULT_RERANKER_MODEL
    # How many fused chunks reach the expensive model. The whole retrieve-then-
    # rerank trade lives in this number: every candidate costs a forward pass,
    # and the shortlist is the only thing standing between a 30ms search and a
    # half-second one.
    #
    # **25 rather than the obvious 50, and it is faster *and* more accurate.**
    # Measured on the 41-query golden set: nDCG 0.761 at 50, 0.788 at 25, 0.781
    # at 15, against 0.718 with reranking off — while p95 latency falls from
    # 473ms to 280ms. A deeper shortlist lets the cross-encoder promote a chunk
    # fusion ranked fortieth into the top ten, and at that depth the model is
    # not reliable enough to be trusted over fusion's own judgement. Bounding
    # the shortlist bounds how far a candidate can jump.
    rerank_candidates: int = 25
    rerank_enabled: bool = True
    # Answering only. Absent, retrieval and search work exactly as before and
    # `ask` reports that it needs a key rather than failing obscurely.
    gemini_api_key: str | None = None
    llm_model: str = DEFAULT_LLM_MODEL
    # Tokens of *passages*. The system prompt and question add a few hundred
    # more. Deliberately well inside the model's window: a fuller prompt is not
    # a better one, because the instruction to refuse competes with every extra
    # passage that looks vaguely relevant.
    answer_token_budget: int = 6000
    answer_max_tokens: int = 1024
    # The graph projection. Bolt rather than HTTP: it is what the official
    # driver speaks, and the browser on 7474 is for reading traversals by eye.
    #
    # No key here is optional the way `gemini_api_key` is, and that difference is
    # deliberate: an absent Gemini key means answering cannot work, while an
    # unreachable Neo4j means only that the graph is unavailable. So these carry
    # working local defaults and the *reachability* — not the configuration — is
    # what everything downstream degrades on.
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "memoryos"
    log_level: str = "INFO"
    log_json: bool = False
    # Browser origins allowed to call this API. Empty by default, which means no
    # CORS middleware is installed at all — a browser cannot reach the API from a
    # page unless somebody deliberately says which page.
    #
    # A list rather than a string, and never `["*"]`: a wildcard on an API that
    # reads a private corpus means any page the operator visits can search it.
    # `create_app` refuses a wildcard outright rather than trusting this comment.
    cors_origins: list[str] = []

    @model_validator(mode="after")
    def _redirect_the_test_environment(self) -> "Settings":
        """Under `MEMOS_ENVIRONMENT=test`, `database_url` *is* the test database.

        Resolved here rather than at each call site so that everything reading
        `database_url` — the container, Alembic's env.py, the shadow workspace —
        agrees without knowing the rule exists. A test run that reached the
        development database through one forgotten call site would truncate it,
        which is the failure this exists to prevent.
        """
        if self.environment == "test":
            self.database_url = self.test_database_url
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
