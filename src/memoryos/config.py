from functools import lru_cache
from pathlib import Path

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
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
