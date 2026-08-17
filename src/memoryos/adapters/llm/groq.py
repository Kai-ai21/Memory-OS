"""Groq, and the second implementation of `LanguageModel`.

M2.6 put this behind a Protocol on the argument that a second provider should be
a new adapter rather than a refactor. This is the milestone that tests the
claim, and the claim held: the port is unchanged, nothing above it knows which
provider answered, and the swap is one setting.

**The failure taxonomy has to match Gemini's exactly**, because the worker's
retry machinery is downstream of both and knows nothing about either. Rate
limits and timeouts are `TransientError` so the existing backoff applies; auth
failures, unknown models, refusals and empty responses are `PermanentError`
because retrying produces the same result and burns the attempt budget doing it.

Where the two adapters genuinely differ is in how that taxonomy is *derived*.
The Gemini SDK collapses most HTTP failures into one exception class, so
`gemini.py` has to match on substrings of the message. Groq's SDK raises a typed
exception per status — `RateLimitError`, `AuthenticationError`, `NotFoundError`
— so this one matches on types, with the status code as the fallback for
anything the SDK adds later. Type matching is strictly better and is the reason
this file is not a copy of the other: a substring rule here would be pretending
not to have information the SDK is handing over.
"""

import asyncio
import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import structlog

from memoryos.adapters.llm.errors import MissingApiKey, RateLimited
from memoryos.application.ports import (
    LanguageModel,
    ModelTurn,
    ToolCall,
    ToolExchange,
)
from memoryos.domain.jobs import PermanentError, TransientError

if TYPE_CHECKING:  # pragma: no cover
    from groq import AsyncGroq

    from memoryos.application.agent.tools import ToolSpec

logger = structlog.get_logger(__name__)

# "Please try again in 23.145s" — the wait Groq states in the body of a 429 when
# the limit is tokens per minute. Matched rather than assumed, and a miss simply
# falls back to exponential backoff; see `_retry_after`.
_RETRY_AFTER_IN_MESSAGE = re.compile(r"try again in ([0-9]+(?:\.[0-9]+)?)\s*s")

__all__ = ["DEFAULT_MODEL", "GroqLanguageModel", "MissingApiKey"]

# Verified against the live API during this milestone rather than trusted from a
# specification: Groq deprecates and renames models often enough that a model id
# copied from documentation is a plausible-looking string that fails at the first
# real call. See the milestone report for the response this returned.
#
# **And that is exactly what happened between M10.0 and M10.1.**
# `llama-3.3-70b-versatile` answered questions during M10.0's session and was
# returning `404 model_not_found` within the hour, on the same key, with nothing
# in this repository having changed. The failure is loud and its message names the
# cause, which is the best that can be arranged for a dependency that can be
# withdrawn underneath a running deployment — but it does mean the *default* here
# has a shelf life, and this comment is the record of one expiring.
#
# Re-verified against `GET /v1/models` on the live account before being written
# down, which is the only check worth doing on a value like this.
DEFAULT_MODEL = "openai/gpt-oss-120b"

# 5xx means the provider broke, not the request. Anything at or above this is
# retried; anything below it is the caller's fault and is not.
_SERVER_ERROR = 500

# `finish_reason` values that mean the model stopped for a reason other than
# having finished. Surfaced in the error message, because "returned no text" and
# "was cut off at the token limit" send a reader to completely different places.
_TRUNCATED = "length"
_FILTERED = "content_filter"


class GroqLanguageModel(LanguageModel):
    def __init__(
        self,
        api_key: str | None,
        *,
        model_name: str = DEFAULT_MODEL,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise MissingApiKey(
                "MEMOS_GROQ_API_KEY is not set. Answering needs a language "
                "model; get a free key at https://console.groq.com/keys. "
                "Retrieval and search work without one."
            )
        self._api_key = api_key
        self._model_name = model_name
        self._timeout = timeout_seconds
        self._client: AsyncGroq | None = None

    @property
    def model_id(self) -> str:
        return self._model_name

    async def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        client = self._load()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._model_name,
                    # The one structural difference from Gemini, and it is
                    # cosmetic: Gemini takes the system prompt as its own
                    # `system_instruction` field, Groq takes it as the first
                    # message. Same two strings either way.
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    # Zero, for the reason `gemini.py` gives: the job is to
                    # restate what the passages say, and variety in a grounded
                    # answer is another word for drift away from the source.
                    temperature=0.0,
                ),
                timeout=self._timeout,
            )
        except TimeoutError as exc:
            # `asyncio.wait_for`'s own timeout, which is the outer bound. The
            # SDK's `APITimeoutError` is the inner one and is classified below;
            # both are transient.
            raise TransientError(
                f"{self._model_name} did not respond within {self._timeout}s"
            ) from exc
        except Exception as exc:
            raise _classify(exc) from exc

        text = _text_of(response)
        if not text or not text.strip():
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
        return text

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

        **Groq takes the tool schema exactly as generated.** Verified against the
        live API during this milestone rather than trusted from a specification:
        a pydantic `model_json_schema()` — `title`, `additionalProperties: false`,
        `minimum`/`maximum` and all — is accepted unchanged, which is why nothing
        here rewrites it. `gemini.py` has to, and says so.

        The replay shape is the provider's, and it is the part that does not
        travel: an assistant message carrying the `tool_calls`, then one message
        per result with `role="tool"` and the matching `tool_call_id`. Sending
        the result without the assistant message that requested it is rejected,
        which is the mistake worth naming because the error says only that the
        message order is wrong.
        """
        client = self._load()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for exchange in exchanges:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": exchange.call.id,
                            "type": "function",
                            "function": {
                                "name": exchange.call.name,
                                "arguments": json.dumps(exchange.call.arguments),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": exchange.call.id,
                    "content": exchange.result,
                }
            )

        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in tools
            ]
            # "auto" rather than "required": a question the corpus cannot help
            # with should be answerable with a sentence saying so, and forcing a
            # call would make the model pick the least bad tool instead.
            payload["tool_choice"] = "auto"

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(**payload), timeout=self._timeout
            )
        except TimeoutError as exc:
            raise TransientError(
                f"{self._model_name} did not respond within {self._timeout}s"
            ) from exc
        except Exception as exc:
            raise _classify(exc) from exc

        message = response.choices[0].message
        calls = tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_arguments(call.function.arguments),
            )
            for call in (message.tool_calls or [])
        )
        text = (message.content or "").strip()
        if not text and not calls:
            reason = _finish_reason(response)
            raise PermanentError(
                f"{self._model_name} returned neither text nor a tool call"
                + (f" ({reason})" if reason else "")
            )

        usage = response.usage
        logger.info(
            "llm.turn",
            model=self._model_name,
            tools_offered=len(tools),
            tool_calls=len(calls),
            answer_chars=len(text),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        return ModelTurn(
            text=text,
            tool_calls=calls,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def _load(self) -> "AsyncGroq":
        if self._client is None:
            from groq import AsyncGroq

            logger.info("llm.client_created", model=self._model_name)
            # `max_retries=0`: the SDK retries some failures itself, and this
            # system already has a retry policy with backoff, attempt budgets
            # and dead-lettering in the worker. Two independent retry loops
            # multiply — five worker attempts over two SDK retries is fifteen
            # calls against a quota the free tier measures in tens per minute.
            self._client = AsyncGroq(api_key=self._api_key, max_retries=0)
        return self._client


def _arguments(raw: str) -> dict[str, Any]:
    """The model's arguments as a dict, or an empty one.

    **Malformed JSON is not raised here**, and that is deliberate: the registry
    validates arguments against the schema and hands back a correctable sentence,
    so an empty dict reaches it as "you sent no arguments" — which is a message
    the model can act on. Raising would abort the turn over the model's mistake.
    """
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _classify(exc: Exception) -> Exception:
    """Transient or permanent, from the SDK's own exception types.

    Imported here rather than at module scope to keep the SDK off the import
    path until something actually talks to it — and by the time this runs, the
    client has been constructed, so the import is already paid for.
    """
    import groq

    if isinstance(exc, groq.RateLimitError):
        # The expected steady state on a free tier, not an exception. Backoff is
        # exactly the right response — and the server says how much of it, which
        # is worth carrying rather than guessing. See `errors.RateLimited`.
        return RateLimited(
            f"language model rate limited: {exc}", retry_after=_retry_after(exc)
        )
    if isinstance(exc, groq.APITimeoutError | groq.APIConnectionError):
        return TransientError(f"language model unreachable: {exc}")
    if isinstance(exc, groq.InternalServerError):
        return TransientError(f"language model unavailable: {exc}")
    if isinstance(
        exc,
        groq.AuthenticationError
        | groq.PermissionDeniedError
        | groq.NotFoundError
        | groq.BadRequestError,
    ):
        # A bad key, a revoked key, a model that does not exist, a malformed
        # request. Every one of these returns the same thing on the next attempt.
        # `NotFoundError` is the one worth naming: it is what a stale model id
        # looks like, and it is the failure this milestone was told not to
        # trust a specification about.
        return PermanentError(f"language model rejected the request: {exc}")
    if isinstance(exc, groq.APIStatusError):
        # A status the SDK has no dedicated class for. Split on the code rather
        # than guessing, so a future 5xx is retried and a future 4xx is not.
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status >= _SERVER_ERROR:
            return TransientError(f"language model unavailable ({status}): {exc}")
        return PermanentError(f"language model failed ({status}): {exc}")
    # Unknown failures are permanent, matching Gemini: retrying something nobody
    # has classified burns the attempt budget and hides the error behind three
    # more of them.
    return PermanentError(f"language model failed: {exc}")


def _retry_after(exc: Exception) -> float | None:
    """How long the server said to wait, in seconds, or None if it did not.

    Two places to look and neither is guaranteed. The `retry-after` header is the
    HTTP convention; Groq also states it in the error body ("Please try again in
    23.145s"), and on a token-per-minute limit that number is the one that
    matters, because it counts down as the window slides rather than naming a
    fixed cooldown.

    Anything unparseable returns None, which puts the caller back on exponential
    backoff. A wrong number here would be worse than no number: too short and it
    burns the attempt budget against a window that has not moved, too long and a
    corpus run stalls on a limit that lifted a minute ago.
    """
    response = getattr(exc, "response", None)
    header = getattr(getattr(response, "headers", None), "get", lambda _: None)(
        "retry-after"
    )
    if header is not None:
        try:
            return max(0.0, float(header))
        except (TypeError, ValueError):
            pass

    match = _RETRY_AFTER_IN_MESSAGE.search(str(exc))
    if match is None:
        return None
    try:
        return max(0.0, float(match.group(1)))
    except ValueError:
        return None


def _text_of(response: Any) -> str | None:
    """The completion text, or None if the response carries none.

    Defensive about the shape because a refusal and a truncation both arrive as
    a well-formed response with nothing useful in it, and an `AttributeError`
    here would be reported as a bug in this adapter rather than as the provider
    declining to answer.
    """
    choices = getattr(response, "choices", None) or []
    for choice in choices:
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if content:
            return str(content)
    return None


def _finish_reason(response: Any) -> str | None:
    choices = getattr(response, "choices", None) or []
    for choice in choices:
        reason = getattr(choice, "finish_reason", None)
        if reason is None:
            continue
        if reason == _TRUNCATED:
            return "truncated at max_tokens"
        if reason == _FILTERED:
            return "blocked by the provider's content filter"
        return str(reason)
    return None
