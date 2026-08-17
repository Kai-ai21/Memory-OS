"""How a provider failure is classified, and why it matters.

The worker already knows how to retry `TransientError` with growing backoff and
how to dead-letter `PermanentError`. So the only thing the adapter has to get
right is which is which — and getting it wrong is expensive in both directions.
A rate limit classified permanent dead-letters a job that a ten-second wait
would have fixed; a safety block classified transient burns the whole attempt
budget re-asking a question the model has already declined.

No network. `_classify` is the decision, and it is reachable directly.
"""

import httpx
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


def test_a_withdrawn_model_is_its_own_permanent_error() -> None:
    """The one permanent failure a batch loop should stop on rather than skip.

    Every other `PermanentError` is about one request. This one is about the
    configuration, so it will refuse item twenty-seven exactly as it refused item
    one — and M10.1 watched `extract-entities` print the same 404 twenty-six times
    and exit 0, which reads like a corpus full of difficult documents rather than
    one stale setting.
    """
    import groq as groq_sdk

    from memoryos.adapters.llm.errors import ModelNotAvailable
    from memoryos.adapters.llm.groq import _classify as classify_groq

    response = httpx.Response(
        404,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        json={"error": {"message": "The model `gone-70b` does not exist"}},
    )
    classified = classify_groq(
        groq_sdk.NotFoundError("not found", response=response, body=None),
        model="gone-70b",
    )

    assert isinstance(classified, ModelNotAvailable)
    # Still a PermanentError, so every existing `except PermanentError` keeps
    # working and nothing has to learn this class exists to stay correct.
    assert isinstance(classified, PermanentError)
    assert classified.model_id == "gone-70b"
    # The guidance names the variable to change, which is the whole point: "the
    # model does not exist" is the provider's true and useless sentence.
    assert "MEMOS_GROQ_MODEL" in classified.guidance
    assert "gone-70b" in classified.guidance
