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
from typing import TYPE_CHECKING, Any

import structlog

from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.application.ports import LanguageModel
from memoryos.domain.jobs import PermanentError, TransientError

if TYPE_CHECKING:  # pragma: no cover
    from groq import AsyncGroq

logger = structlog.get_logger(__name__)

__all__ = ["DEFAULT_MODEL", "GroqLanguageModel", "MissingApiKey"]

# Verified against the live API during this milestone rather than trusted from a
# specification: Groq deprecates and renames models often enough that a model id
# copied from documentation is a plausible-looking string that fails at the first
# real call. See the milestone report for the response this returned.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

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


def _classify(exc: Exception) -> Exception:
    """Transient or permanent, from the SDK's own exception types.

    Imported here rather than at module scope to keep the SDK off the import
    path until something actually talks to it — and by the time this runs, the
    client has been constructed, so the import is already paid for.
    """
    import groq

    if isinstance(exc, groq.RateLimitError):
        # The expected steady state on a free tier, not an exception. Backoff is
        # exactly the right response.
        return TransientError(f"language model rate limited: {exc}")
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
