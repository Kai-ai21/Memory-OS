"""The second `LanguageModel`, and the port's claim that a swap is an adapter.

No network. The Groq client is mocked, because this project does not own it — a
real call here would be slow, would need a key in CI, and would fail for reasons
that have nothing to do with the code under test.

What is deliberately *not* mocked is the error taxonomy. The exceptions raised
below are the SDK's own classes, constructed the way the SDK constructs them, so
these assert against Groq's real types rather than against a belief about them.

Three tests, which is the milestone's budget. The provider selection goes
through `build_language_model` rather than `Container.build`, because building a
container constructs the embedder — seconds of work and a HuggingFace round trip
that has nothing to do with which language model was chosen.
"""

from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest

from memoryos.adapters.llm.gemini import GeminiLanguageModel
from memoryos.adapters.llm.groq import GroqLanguageModel
from memoryos.config import Settings
from memoryos.container import build_language_model
from memoryos.domain.jobs import PermanentError, TransientError


class FakeCompletions:
    """Stands in for `client.chat.completions`.

    Either raises what it was given or returns a response shaped like the SDK's:
    `choices[0].message.content` and `choices[0].finish_reason`.
    """

    def __init__(
        self, *, raises: Exception | None = None, content: str | None = None
    ) -> None:
        self._raises = raises
        self._content = content
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content),
                    finish_reason="stop",
                )
            ]
        )


def model_with(completions: FakeCompletions) -> GroqLanguageModel:
    """A Groq adapter wired to a fake client, bypassing the lazy constructor."""
    model = GroqLanguageModel("test-key")
    model._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=completions)
    )
    return model


def rate_limit() -> groq.RateLimitError:
    """A real 429, built the way the SDK builds one."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.RateLimitError(
        "rate limit reached", response=httpx.Response(429, request=request), body=None
    )


def test_the_provider_is_selected_from_settings() -> None:
    """The whole milestone, in one assertion each way.

    Selection is the only place that knows a provider exists. Everything above
    it takes a `LanguageModel` and cannot tell which one it got, which is what
    M2.6 bought by putting this behind a Protocol.
    """
    groq_model = build_language_model(
        Settings(llm_provider="groq", groq_api_key="test-key")
    )
    gemini_model = build_language_model(
        Settings(llm_provider="gemini", gemini_api_key="test-key")
    )

    assert isinstance(groq_model, GroqLanguageModel)
    assert isinstance(gemini_model, GeminiLanguageModel)


async def test_a_rate_limit_is_transient() -> None:
    """429 is the expected steady state on a free tier, not an exception.

    Classified permanent it would dead-letter a job that a ten-second wait would
    have fixed — and on a free tier that is most jobs. The mapping has to match
    Gemini's, because the worker's backoff is downstream of both and knows about
    neither.
    """
    model = model_with(FakeCompletions(raises=rate_limit()))

    with pytest.raises(TransientError, match="rate limited"):
        await model.complete("system", "user")


async def test_an_empty_response_is_permanent() -> None:
    """The model returned successfully and said nothing.

    Presenting that as an answer is worse than failing, and re-asking tends to
    produce another empty completion.
    """
    model = model_with(FakeCompletions(content=""))

    with pytest.raises(PermanentError, match="returned no text"):
        await model.complete("system", "user")
