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

from memoryos.domain.jobs import TransientError


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
