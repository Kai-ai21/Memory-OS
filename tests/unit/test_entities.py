from datetime import UTC, datetime

import pytest

from memoryos.domain.entities import IngestionEvent, Memory, MemoryChunk, RawArtifact
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash, EventType, MemoryKind, TimeProvenance

HASH = ContentHash.of(b"memory intelligence os")
WHEN = datetime(2023, 4, 1, 12, 0, tzinfo=UTC)


def make_memory(**overrides: object) -> Memory:
    fields: dict[str, object] = {
        "id": new_id(),
        "source_id": new_id(),
        "external_key": "notes/example.md",
        "content_hash": HASH,
        "kind": MemoryKind.NOTE,
        "occurred_at": WHEN,
        "occurred_at_source": TimeProvenance.FILESYSTEM,
    }
    fields.update(overrides)
    return Memory(**fields)  # type: ignore[arg-type]


def make_chunk(**overrides: object) -> MemoryChunk:
    fields: dict[str, object] = {
        "id": new_id(),
        "memory_id": new_id(),
        "ordinal": 0,
        "content": "hello",
        "token_count": 2,
        "char_start": 0,
        "char_end": 5,
        "chunker_version": "v1",
        "content_hash": HASH,
    }
    fields.update(overrides)
    return MemoryChunk(**fields)  # type: ignore[arg-type]


def test_new_ids_are_time_ordered() -> None:
    # UUIDv7's leading timestamp is what keeps primary-key inserts sequential.
    ids = [str(new_id()) for _ in range(50)]
    assert ids == sorted(ids)


def test_memory_accepts_a_known_time_with_provenance() -> None:
    memory = make_memory()
    assert memory.occurred_at == WHEN
    assert memory.version == 1
    assert memory.is_current is True


def test_memory_accepts_an_unknown_time_with_unknown_provenance() -> None:
    memory = make_memory(occurred_at=None, occurred_at_source=TimeProvenance.UNKNOWN)
    assert memory.occurred_at is None


def test_memory_rejects_null_time_with_known_provenance() -> None:
    with pytest.raises(ValueError, match="occurred_at and occurred_at_source disagree"):
        make_memory(occurred_at=None, occurred_at_source=TimeProvenance.DECLARED)


def test_memory_rejects_known_time_with_unknown_provenance() -> None:
    with pytest.raises(ValueError, match="occurred_at and occurred_at_source disagree"):
        make_memory(occurred_at=WHEN, occurred_at_source=TimeProvenance.UNKNOWN)


@pytest.mark.parametrize("version", [0, -1])
def test_memory_rejects_version_below_one(version: int) -> None:
    with pytest.raises(ValueError, match="version must be >= 1"):
        make_memory(version=version)


@pytest.mark.parametrize("importance", [-0.001, 1.001, -1.0, 2.0])
def test_memory_rejects_importance_outside_the_unit_interval(importance: float) -> None:
    with pytest.raises(ValueError, match="importance must be within"):
        make_memory(importance=importance)


@pytest.mark.parametrize("importance", [0.0, 0.5, 1.0, None])
def test_memory_accepts_importance_within_the_unit_interval(importance: float | None) -> None:
    assert make_memory(importance=importance).importance == importance


@pytest.mark.parametrize(
    ("char_start", "char_end"),
    [pytest.param(5, 5, id="equal"), pytest.param(5, 4, id="inverted")],
)
def test_chunk_rejects_a_non_positive_span(char_start: int, char_end: int) -> None:
    with pytest.raises(ValueError, match="char_end must be > char_start"):
        make_chunk(char_start=char_start, char_end=char_end)


def test_chunk_rejects_a_negative_ordinal() -> None:
    with pytest.raises(ValueError, match="ordinal must be >= 0"):
        make_chunk(ordinal=-1)


def test_chunk_rejects_a_negative_char_start() -> None:
    with pytest.raises(ValueError, match="char_start must be >= 0"):
        make_chunk(char_start=-1, char_end=3)


@pytest.mark.parametrize("token_count", [0, -1])
def test_chunk_rejects_a_non_positive_token_count(token_count: int) -> None:
    with pytest.raises(ValueError, match="token_count must be > 0"):
        make_chunk(token_count=token_count)


def test_artifact_rejects_a_negative_byte_size() -> None:
    with pytest.raises(ValueError, match="byte_size must be >= 0"):
        RawArtifact(content_hash=HASH, byte_size=-1)


def test_event_rejects_an_empty_external_key() -> None:
    with pytest.raises(ValueError, match="external_key must not be empty"):
        IngestionEvent(
            id=new_id(),
            event_type=EventType.ARTIFACT_OBSERVED,
            source_id=new_id(),
            external_key="",
            occurred_at_source=TimeProvenance.UNKNOWN,
        )
