import pytest

from memoryos.domain.values import ContentHash, EventType, MemoryKind, TimeProvenance

# BLAKE2b-256 of the empty string. A published test vector, so this asserts the
# algorithm and digest size, not merely that the code agrees with itself.
BLAKE2B_256_OF_EMPTY = "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"


def test_of_matches_a_known_blake2b_256_vector() -> None:
    assert ContentHash.of(b"").value == BLAKE2B_256_OF_EMPTY


def test_of_is_stable_across_calls() -> None:
    assert ContentHash.of(b"memory intelligence os") == ContentHash.of(b"memory intelligence os")


def test_of_produces_a_valid_hash_for_arbitrary_bytes() -> None:
    digest = ContentHash.of(bytes(range(256)))
    assert len(digest.value) == 64


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("a" * 63, id="too-short"),
        pytest.param("a" * 65, id="too-long"),
        pytest.param("", id="empty"),
        pytest.param("g" * 64, id="non-hex"),
        pytest.param("A" * 64, id="uppercase"),
        pytest.param(BLAKE2B_256_OF_EMPTY.upper(), id="uppercase-real-digest"),
        pytest.param(f"{'a' * 64}\n", id="trailing-newline"),
        pytest.param(f" {'a' * 63}", id="leading-space"),
    ],
)
def test_rejects_malformed_input(value: str) -> None:
    with pytest.raises(ValueError, match="64 lowercase hex characters"):
        ContentHash(value)


def test_is_frozen() -> None:
    digest = ContentHash.of(b"x")
    with pytest.raises(AttributeError):
        digest.value = "b" * 64  # type: ignore[misc]


def test_enum_values_are_the_lowercase_member_names() -> None:
    # The database CHECK constraints are written against these literals.
    assert TimeProvenance.UNKNOWN.value == "unknown"
    assert MemoryKind.NOTE.value == "note"
    assert EventType.ARTIFACT_OBSERVED.value == "artifact_observed"
