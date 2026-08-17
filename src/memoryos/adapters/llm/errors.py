"""Failures shared by every `LanguageModel` adapter.

One class, not one per provider, and that is forced rather than tidy. `cli.py`
catches `MissingApiKey` to print a usable message and exit 2 instead of dumping a
traceback. With a class per adapter, that `except` would silently stop matching
the moment `MEMOS_LLM_PROVIDER` selected a different provider — the CLI would
keep compiling, keep passing its tests, and start showing a stack trace to
anybody who had not set the key for the provider they had just switched to.

The provider-specific taxonomy stays in the adapters, where it belongs:
`TransientError` and `PermanentError` come from `domain.jobs`, and deciding which
of the two a given SDK failure is remains each adapter's job.
"""

from memoryos.domain.jobs import PermanentError, TransientError


class RateLimited(TransientError):
    """A provider that refused for quota reasons and said when to come back.

    A `TransientError` subclass, so every existing `except TransientError` keeps
    working unchanged — the worker's backoff, the CLI's retry loops, the chunk-level
    retry in relationship extraction — and none of them has to know this exists.

    What it adds is `retry_after`, which the provider states outright and which
    guessing gets badly wrong in both directions. Groq's free tier limits *tokens*
    per minute and answers a 429 with "try again in 23.145s"; exponential backoff
    from a 2s base spends its first four attempts inside a window the server has
    already told us the length of, exhausts its budget, and fails a batch that
    would have succeeded. Measured on this corpus: eight successful calls in ten
    minutes, against a limit that allows roughly two or three a minute.

    Jitter is still applied on top, because the reason for jitter is unrelated to
    the reason for the wait: a hundred callers told "come back in 23s" all come
    back in the same millisecond.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class MissingApiKey(RuntimeError):
    """No credential for the selected provider, so answering is impossible.

    Raised when the language model is constructed rather than when it is first
    called, and constructed lazily — see `Container.answer`. A deployment with no
    key keeps working search, ingestion and replay; only answering fails, and it
    fails naming the variable to set.
    """


class ModelNotAvailable(PermanentError):
    """The configured model does not exist, or this key cannot reach it.

    **A distinct class because of what a caller should do differently**, which is
    stop. Every other `PermanentError` describes one request going wrong — a
    malformed prompt, a safety block, a document the model refused — and the right
    response is the one the batch loops already have: count it, report it, step
    over it, keep going. A withdrawn model is not about the request at all. It will
    refuse item twenty-seven exactly as it refused item one, so a loop that steps
    over it does nothing except print the same sentence once per row.

    M10.1 watched that happen. `llama-3.3-70b-versatile` answered questions during
    M10.0's session and was returning `404 model_not_found` within the hour, on the
    same key, with nothing in the repository changed — and `extract-entities`
    reported twenty-six identical failures and exit 0, which reads like twenty-six
    difficult documents rather than one setting.

    Carries the model id and the provider, because the message has to name the
    thing to change. "The model does not exist" is the provider's sentence and it
    is true; "the model *you configured* does not exist, here is the variable"
    is the one that ends the problem.
    """

    def __init__(self, message: str, *, model_id: str, provider: str) -> None:
        super().__init__(message)
        self.model_id = model_id
        self.provider = provider

    @property
    def guidance(self) -> str:
        """What to do about it, in one line a terminal can print."""
        variable = (
            "MEMOS_GROQ_MODEL" if self.provider == "groq" else "MEMOS_LLM_MODEL"
        )
        return (
            f"{self.provider} has no model {self.model_id!r} for this key. "
            f"Providers withdraw and rename models without notice, so this is "
            f"usually a stale setting rather than a fault: list what the account "
            f"can actually reach and set {variable} to one of those."
        )
