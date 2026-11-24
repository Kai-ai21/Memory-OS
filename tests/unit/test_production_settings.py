"""Production refuses to start with a gap, and says what the gap is (M12.3)."""

import base64

import pytest
from pydantic import ValidationError

from memoryos.config import Settings

KEY = base64.b64encode(b"k" * 32).decode()
SAFE = {
    "environment": "production",
    "master_key": KEY,
    "database_url": "postgresql+asyncpg://memos_app:s3cret@postgres:5432/memos",
    "database_admin_url": "postgresql+asyncpg://memos:0wner@postgres:5432/memos",
    "neo4j_password": "a-real-neo4j-password",
    "cors_origins": ["https://memo.example.com"],
}


def production(**overrides: object) -> Settings:
    # `_env_file=None`: the developer's `.env` must not decide this test.
    return Settings(_env_file=None, **{**SAFE, **overrides})  # type: ignore[call-arg, arg-type]


def test_a_complete_production_configuration_starts() -> None:
    settings = production()
    assert settings.is_production
    assert settings.session_cookie_secure
    # Structured logs, whatever was asked for.
    assert settings.log_json


@pytest.mark.parametrize(
    ("override", "complaint"),
    [
        ({"master_key": ""}, "MEMOS_MASTER_KEY"),
        (
            {"database_url": "postgresql+asyncpg://memos_app:memos_app@localhost:5433/memos"},
            "MEMOS_DATABASE_URL is the development default",
        ),
        (
            {"database_admin_url": "postgresql+asyncpg://memos:x@localhost:5432/memos"},
            "points at localhost",
        ),
        ({"neo4j_password": "memoryos"}, "MEMOS_NEO4J_PASSWORD"),
        ({"cors_origins": []}, "MEMOS_CORS_ORIGINS is empty"),
        ({"cors_origins": ["http://memo.example.com"]}, "every origin is https"),
        ({"db_echo": True}, "MEMOS_DB_ECHO"),
    ],
)
def test_each_gap_is_a_refusal_not_a_warning(override: dict[str, object], complaint: str) -> None:
    with pytest.raises(ValidationError, match="refusing to start in production") as raised:
        production(**override)
    assert complaint in str(raised.value)


def test_every_gap_is_listed_at_once() -> None:
    with pytest.raises(ValidationError) as raised:
        production(master_key="", neo4j_password="memoryos", cors_origins=[])
    message = str(raised.value)
    assert "MEMOS_MASTER_KEY" in message
    assert "MEMOS_NEO4J_PASSWORD" in message
    assert "MEMOS_CORS_ORIGINS" in message


def test_memos_env_names_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MEMOS_ENVIRONMENT", raising=False)
    monkeypatch.setenv("MEMOS_ENV", "development")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.environment == "development"
    # A laptop over plain HTTP: a `Secure` cookie would never come back.
    assert not settings.session_cookie_secure


def test_an_unknown_environment_is_refused() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="prod")  # type: ignore[call-arg, arg-type]


def test_registration_is_off_unless_turned_on() -> None:
    assert Settings(_env_file=None).allow_registration is False  # type: ignore[call-arg]
