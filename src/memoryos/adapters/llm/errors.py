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


class MissingApiKey(RuntimeError):
    """No credential for the selected provider, so answering is impossible.

    Raised when the language model is constructed rather than when it is first
    called, and constructed lazily — see `Container.answer`. A deployment with no
    key keeps working search, ingestion and replay; only answering fails, and it
    fails naming the variable to set.
    """
