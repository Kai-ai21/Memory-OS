"""How a provider failure is classified, and why it matters.

The worker already knows how to retry `TransientError` with growing backoff and
how to dead-letter `PermanentError`. So the only thing the adapter has to get
right is which is which — and getting it wrong is expensive in both directions.
A rate limit classified permanent dead-letters a job that a ten-second wait
would have fixed; a safety block classified transient burns the whole attempt
budget re-asking a question the model has already declined.

No network. `_classify` is the decision, and it is reachable directly.
"""

import pytest

from memoryos.adapters.llm.gemini import GeminiLanguageModel, MissingApiKey, _classify
from memoryos.domain.jobs import PermanentError, TransientError


@pytest.mark.parametrize(
    "message",
    [
        "429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric",
        "rate limit reached for gemini-2.0-flash",
        "503 Service Unavailable",
        "500 Internal error encountered",
        "Deadline exceeded while waiting for response",
    ],
)
def test_rate_limits_and_outages_are_transient(message: str) -> None:
    """The free tier's quota is the expected steady state, not an exception."""
    classified = _classify(RuntimeError(message))

    assert isinstance(classified, TransientError)
    assert "unavailable" in str(classified)


@pytest.mark.parametrize(
    "message",
    [
        "Response was blocked due to SAFETY",
        "content prohibited by policy",
    ],
)
def test_a_safety_block_is_permanent(message: str) -> None:
    """Retrying returns the same block, and the answer has to say so."""
    assert isinstance(_classify(RuntimeError(message)), PermanentError)


def test_an_unclassified_failure_is_permanent() -> None:
    """Retrying something nobody has classified burns the attempt budget."""
    assert isinstance(_classify(ValueError("malformed request")), PermanentError)


def test_a_missing_key_fails_at_construction_rather_than_at_the_question() -> None:
    """A misconfigured deployment should break at startup, not on a user."""
    with pytest.raises(MissingApiKey, match=r"aistudio\.google\.com"):
        GeminiLanguageModel(None)
    with pytest.raises(MissingApiKey):
        GeminiLanguageModel("")

    # And the message says search still works, because it does.
    try:
        GeminiLanguageModel(None)
    except MissingApiKey as exc:
        assert "Retrieval and search work without one" in str(exc)
