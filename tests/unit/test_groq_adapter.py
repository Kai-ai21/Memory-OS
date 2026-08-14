"""The second `LanguageModel`, and the port's claim that a swap is an adapter.

No network. The Groq client is mocked, because this project does not own it — a
real call here would be slow, would need a key in CI, and would fail for reasons
that have nothing to do with the code under test.

What is deliberately *not* mocked is the error taxonomy. The exceptions raised
below are the SDK's own classes, constructed the way the SDK constructs them, so
these assert against Groq's real types rather than against a belief about them.

M2.6a's budget was three; the fourth was proposed there, deferred, and added by
M3.1 step 0b. The provider selection goes through `build_language_model` rather
than `Container.build`, because building a container constructs the embedder —
seconds of work and a HuggingFace round trip that has nothing to do with which
language model was chosen.
"""

import json
from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest

from memoryos.adapters.llm.gemini import GeminiLanguageModel
from memoryos.adapters.llm.groq import GroqLanguageModel
from memoryos.application.agent.tools import ToolSpec
from memoryos.application.ports import ToolCall, ToolExchange, supports_tools
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


async def test_both_prompts_reach_the_request_at_zero_temperature() -> None:
    """The translation the port hides, pinned.

    Gemini takes the system prompt as its own `system_instruction` field; Groq
    takes it as the first message. The port hands over two strings either way,
    so the mapping is this adapter's private business — and a system prompt
    dropped in that translation is the defect worth a test, because nothing
    would fail. The call would succeed, the answer would come back fluent, and
    the instruction to stay inside the passages and refuse otherwise would
    simply not have been sent. M2.6's entire guardrail lives in that string.

    Temperature is asserted for the same reason it is set: the job is to restate
    what the passages say, and sampling in a grounded answer is another word for
    drift away from the source.
    """
    completions = FakeCompletions(content="an answer")
    model = model_with(completions)

    assert await model.complete("be grounded", "the question", max_tokens=64) == "an answer"

    sent = completions.calls[0]
    assert sent["messages"] == [
        {"role": "system", "content": "be grounded"},
        {"role": "user", "content": "the question"},
    ]
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 64
    assert sent["model"] == model.model_id


# --------------------------------------------------------------------------
# M7.0: tool calling, and the replay shape that does not travel between providers
# --------------------------------------------------------------------------


class FakeToolCompletions(FakeCompletions):
    """A `create` that can also return tool calls, shaped like the SDK's."""

    def __init__(self, *, tool_calls: list[Any] | None = None, content: str | None = None):
        super().__init__(content=content)
        self._tool_calls = tool_calls

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self._content, tool_calls=self._tool_calls
                    ),
                    finish_reason="tool_calls" if self._tool_calls else "stop",
                )
            ]
        )


def _call(name: str, arguments: str, call_id: str = "call_1") -> Any:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


SPEC = ToolSpec(
    name="search_memories",
    description="Find passages.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)


async def test_a_tool_call_comes_back_parsed() -> None:
    completions = FakeToolCompletions(
        tool_calls=[_call("search_memories", '{"query": "leases"}')]
    )
    model = model_with(completions)

    turn = await model.converse("system", "how do leases work", tools=[SPEC])

    assert turn.wants_tools
    assert turn.tool_calls[0].name == "search_memories"
    assert turn.tool_calls[0].arguments == {"query": "leases"}
    # The provider's own id, carried unchanged: Groq requires it back on the
    # result so the two can be paired.
    assert turn.tool_calls[0].id == "call_1"
    # The schema is passed through untouched — verified against the live API,
    # and pinned here so a later "cleanup" cannot start rewriting it.
    sent = completions.calls[0]["tools"][0]["function"]
    assert sent["parameters"] == SPEC.parameters
    assert completions.calls[0]["tool_choice"] == "auto"


async def test_arguments_that_are_not_json_become_no_arguments() -> None:
    """**Not raised**, because the registry's validation is what should answer.

    A model that emitted broken JSON gets "you sent no arguments" from the tool's
    own schema check, which is a sentence it can act on. Raising here would abort
    the turn over a mistake the next turn could fix.
    """
    model = model_with(
        FakeToolCompletions(tool_calls=[_call("search_memories", "{not json")])
    )

    turn = await model.converse("system", "anything", tools=[SPEC])

    assert turn.tool_calls[0].arguments == {}


async def test_a_completed_call_is_replayed_in_the_shape_groq_requires() -> None:
    """**The one part of tool calling that does not travel between providers.**

    Groq needs the assistant message that *requested* the call, then a separate
    message with `role="tool"` carrying the matching `tool_call_id`. Sending the
    result without the request is rejected, and the error says only that the
    message order is wrong — which is why the shape is asserted here rather than
    discovered at three in the morning. Gemini pairs the same exchange by
    function *name* instead; see `_gemini_schema`'s neighbours in that adapter.
    """
    completions = FakeToolCompletions(content="Leases expire after 30 seconds.")
    model = model_with(completions)
    call = ToolCall(id="call_7", name="search_memories", arguments={"query": "leases"})

    turn = await model.converse(
        "system",
        "how do leases work",
        tools=[SPEC],
        exchanges=[ToolExchange(call=call, result="worker.py: lease = 30s")],
    )

    assert turn.text == "Leases expire after 30 seconds."
    messages = completions.calls[0]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    requested = messages[2]["tool_calls"][0]
    assert requested["id"] == "call_7"
    assert requested["function"]["name"] == "search_memories"
    # Arguments go back as the JSON string the API expects, not as a dict.
    assert json.loads(requested["function"]["arguments"]) == {"query": "leases"}
    assert messages[3]["tool_call_id"] == "call_7"
    assert messages[3]["content"] == "worker.py: lease = 30s"


async def test_a_turn_with_neither_text_nor_a_call_is_permanent() -> None:
    """There is nothing to retry towards, exactly as for an empty completion."""
    model = model_with(FakeToolCompletions(content=None, tool_calls=[]))

    with pytest.raises(PermanentError, match="neither text nor a tool call"):
        await model.converse("system", "anything", tools=[SPEC])


def test_both_adapters_are_tool_calling_models() -> None:
    """The capability check the container gates the agent on.

    Structural, so an adapter that grows `converse` becomes usable with nothing
    to remember to set — and one that loses it stops being offered rather than
    failing at the first question.
    """
    groq_model = GroqLanguageModel("key")
    gemini_model = GeminiLanguageModel("key")

    assert supports_tools(groq_model)
    assert supports_tools(gemini_model)
    assert not supports_tools(object())
