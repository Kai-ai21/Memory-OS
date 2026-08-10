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
from typing import TYPE_CHECKING, Any

import structlog

from memoryos.application.ports import LanguageModel
from memoryos.domain.jobs import PermanentError, TransientError

if TYPE_CHECKING:  # pragma: no cover
    from google import genai

logger = structlog.get_logger(__name__)

# Free tier, and fast enough that generation is not the dominant cost of an
# answer. Flash rather than Pro deliberately: this task is extraction and
# summary over supplied passages, not reasoning, and the guardrails do the work
# a larger model would otherwise be asked to do on trust.
DEFAULT_MODEL = "gemini-2.0-flash"

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


class MissingApiKey(RuntimeError):
    """No credential for the language model, so answering is impossible."""


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
