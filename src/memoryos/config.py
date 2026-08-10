from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from memoryos.adapters.embedding.sentence_transformers import (
    DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL,
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
