from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MEMOS_", extra="ignore"
    )

    environment: str = "local"
    database_url: str = "postgresql+asyncpg://memos:memos@localhost:5433/memos"
    db_echo: bool = False
    log_level: str = "INFO"
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
