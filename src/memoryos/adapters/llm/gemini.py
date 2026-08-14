"""Google Gemini, on the free tier.

The client is constructed lazily and held, the same discipline as the two local
models — but for a different reason. Those load hundreds of megabytes of
weights; this one holds a connection pool, and constructing it per request would
be a new TLS handshake on every answer.

**The failure taxonomy is the part worth reading.** The worker already knows how
to retry `TransientError` with backoff and how to dead-letter `PermanentError`,
so the only thing this adapter has to get right is which is which:

* **Rate limits and timeouts are transient.** The free tier's quota is the
  expected steady state, not an exception, and the existing backoff is exactly
  the right response.
* **A safety block is permanent.** Retrying returns the same block, and it is
  not a failure of this system — the model declined, and an answer has to say so
  rather than silently return nothing.
* **An empty response is permanent.** The model returned successfully and said
  nothing. Presenting that as an answer is worse than failing, and retrying an
  empty completion tends to produce another one.

A missing API key raises at construction rather than at the first request, so a
misconfigured deployment fails at startup instead of on a user's question.
"""

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import structlog

from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.application.ports import (
    LanguageModel,
    ModelTurn,
    ToolCall,
    ToolExchange,
)
from memoryos.domain.jobs import PermanentError, TransientError

__all__ = ["DEFAULT_MODEL", "GeminiLanguageModel", "MissingApiKey"]

if TYPE_CHECKING:  # pragma: no cover
    from google import genai
    from google.genai.types import GenerateContentConfigDict

    from memoryos.application.agent.tools import ToolSpec

logger = structlog.get_logger(__name__)

# Free tier, and fast enough that generation is not the dominant cost of an
# answer. Flash rather than Pro deliberately: this task is extraction and
# summary over supplied passages, not reasoning, and the guardrails do the work
# a larger model would otherwise be asked to do on trust.
#
# **`gemini-2.0-flash` was retired and this was a dead default.** M7.0 went to
# check whether this provider supports tool calling and got a 404 back saying
# the model "is no longer available" — which nothing noticed, because the
# configured provider is Groq and no test calls Gemini. Retrieval-only
# deployments never touch it either. Bumped to the current Flash, and the
# lesson is the one M2.6a already wrote down about Groq: a model id copied from
# documentation is a plausible-looking string, and the only thing that tells you
# it has expired is a real call.
DEFAULT_MODEL = "gemini-2.5-flash"

# Substrings that mark a provider error as worth retrying. Matched on the
# message because the SDK raises a single exception type for most HTTP failures
# and the status code is not always exposed as a field.
_TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "resource_exhausted",
    "quota",
    "503",
    "unavailable",
    "500",
    "internal",
    "deadline",
    "timeout",
)


class GeminiLanguageModel(LanguageModel):
    def __init__(
        self,
        api_key: str | None,
        *,
        model_name: str = DEFAULT_MODEL,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise MissingApiKey(
                "MEMOS_GEMINI_API_KEY is not set. Answering needs a language "
                "model; get a free key at https://aistudio.google.com/apikey. "
                "Retrieval and search work without one."
            )
        self._api_key = api_key
        self._model_name = model_name
        self._timeout = timeout_seconds
        self._client: genai.Client | None = None

    @property
    def model_id(self) -> str:
        return self._model_name

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 1024
    ) -> str:
        client = self._load()
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self._model_name,
                    contents=user,
                    config={
                        "system_instruction": system,
                        "max_output_tokens": max_tokens,
                        # Zero, because the job is to restate what the passages
                        # say. Sampling buys variety, and variety in a grounded
                        # answer is another word for drift away from the source.
                        "temperature": 0.0,
                    },
                ),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            raise TransientError(
                f"{self._model_name} did not respond within {self._timeout}s"
            ) from exc
        except Exception as exc:
            raise _classify(exc) from exc

        text = getattr(response, "text", None)
        if not text or not text.strip():
            # Distinguish the two ways a response arrives empty, because one is
            # the model declining and the other is a truncated generation, and
            # a reader needs to know which.
            reason = _finish_reason(response)
            raise PermanentError(
                f"{self._model_name} returned no text"
                + (f" ({reason})" if reason else "")
            )

        logger.info(
            "llm.completed",
            model=self._model_name,
            prompt_chars=len(system) + len(user),
            answer_chars=len(text),
        )
        return str(text)

    async def converse(
        self,
        system: str,
        user: str,
        *,
        tools: Sequence["ToolSpec"] = (),
        exchanges: Sequence[ToolExchange] = (),
        max_tokens: int = 1024,
    ) -> ModelTurn:
        """One turn, with tools offered and any completed calls replayed.

        **Gemini rejects `additionalProperties`, and that is the whole schema
        difference.** Measured against the live API rather than read from a
        specification: the pydantic-generated schema is accepted with `title`,
        `default`, `minimum` and `maximum` intact, and refused outright with a
        400 — `Unknown name "additional_properties"` — while that one key is
        present. So `_gemini_schema` strips exactly it, and nothing else. A
        broader strip would be guessing, and would quietly drop the bounds and
        descriptions the model routes on.

        The replay shape differs too. Groq pairs a result to its call by
        `tool_call_id`; Gemini has no such id and pairs by *function name*,
        carried in a `functionResponse` part. Both facts live here rather than
        in the loop, which is what M2.6 put a Protocol in front of these two for.
        """
        client = self._load()
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": user}]}
        ]
        for exchange in exchanges:
            contents.append(
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "name": exchange.call.name,
                                "args": exchange.call.arguments,
                            }
                        }
                    ],
                }
            )
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": exchange.call.name,
                                "response": {"result": exchange.result},
                            }
                        }
                    ],
                }
            )

        # A `GenerateContentConfigDict` rather than a bare dict, because the
        # SDK's own type is what says whether a key is spelled right — and this
        # config carries the tool declarations, which is the part that fails
        # loudly at the API rather than quietly at the call site.
        config: GenerateContentConfigDict = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "temperature": 0.0,
        }
        if tools:
            config["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "parameters": _gemini_schema(spec.parameters),
                        }
                        for spec in tools
                    ]
                }
            ]

        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self._model_name, contents=contents, config=config
                ),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            raise TransientError(
                f"{self._model_name} did not respond within {self._timeout}s"
            ) from exc
        except Exception as exc:
            raise _classify(exc) from exc

        calls: list[ToolCall] = []
        texts: list[str] = []
        for candidate in response.candidates or []:
            for part in (candidate.content.parts if candidate.content else None) or []:
                if part.function_call is not None and part.function_call.name:
                    calls.append(
                        ToolCall(
                            # Gemini mints no call id. The name is what it pairs
                            # a response by, so the name is what is carried —
                            # and `ToolExchange` reads `call.id` nowhere in this
                            # adapter for exactly that reason.
                            id=part.function_call.id or part.function_call.name,
                            name=part.function_call.name,
                            arguments=dict(part.function_call.args or {}),
                        )
                    )
                elif part.text:
                    texts.append(part.text)

        text = "".join(texts).strip()
        if not text and not calls:
            reason = _finish_reason(response)
            raise PermanentError(
                f"{self._model_name} returned neither text nor a tool call"
                + (f" ({reason})" if reason else "")
            )

        logger.info(
            "llm.turn",
            model=self._model_name,
            tools_offered=len(tools),
            tool_calls=len(calls),
            answer_chars=len(text),
        )
        return ModelTurn(text=text, tool_calls=tuple(calls))

    def _load(self) -> "genai.Client":
        if self._client is None:
            from google import genai

            logger.info("llm.client_created", model=self._model_name)
            self._client = genai.Client(api_key=self._api_key)
        return self._client


def _classify(exc: Exception) -> Exception:
    """Transient or permanent, from whatever the SDK raised.

    Message matching rather than exception types: the SDK collapses most HTTP
    failures into one class, and treating everything as permanent would
    dead-letter a job for a rate limit that a ten-second wait would clear.
    """
    message = str(exc).lower()
    if any(marker in message for marker in _TRANSIENT_MARKERS):
        return TransientError(f"language model unavailable: {exc}")
    if "safety" in message or "blocked" in message or "prohibited" in message:
        return PermanentError(f"language model refused to answer: {exc}")
    # Unknown failures are permanent. Retrying something nobody has classified
    # burns the attempt budget and hides the error behind three more of them.
    return PermanentError(f"language model failed: {exc}")


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            return str(reason)
    return None


# The one JSON Schema key Gemini refuses, measured rather than assumed.
#
# `additionalProperties: false` is what a pydantic model configured
# `extra="forbid"` emits, and it is the key that makes an argument the tool did
# not declare an error rather than something silently dropped. Gemini answers a
# 400 — `Unknown name "additional_properties"` — for its presence alone, while
# accepting `title`, `default`, `minimum` and `maximum` without complaint. Groq
# accepts all of them.
#
# So the strict schema stays strict everywhere it is validated, and one key is
# removed on the way to one provider. The alternative — weakening the schema at
# the source — would trade a real guarantee on the way in for a provider's
# parser on the way out.
_UNSUPPORTED_SCHEMA_KEYS = frozenset({"additionalProperties"})


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The spec's JSON Schema with the keys this provider refuses removed."""
    cleaned = {
        key: value for key, value in schema.items() if key not in _UNSUPPORTED_SCHEMA_KEYS
    }
    properties = cleaned.get("properties")
    if isinstance(properties, dict):
        cleaned["properties"] = {
            name: (
                {k: v for k, v in spec.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
                if isinstance(spec, dict)
                else spec
            )
            for name, spec in properties.items()
        }
    return cleaned
