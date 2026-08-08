from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
