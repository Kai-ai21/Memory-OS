"""The pair must fit the window before the model sees it.

This is M1.6.1 applied to a second model. That defect was a chunker sized to 512
tokens against a model that read 256: nothing errored, nothing failed, and half
of every long chunk was silently discarded for a milestone and a half. A
cross-encoder has the same failure available and a worse version of it — the
pair is `[CLS] query [SEP] document [SEP]`, so an over-long document does not
merely lose its tail, it can push the *query* out of the window and leave the
model scoring a document against nothing.

No model is loaded. The tokenizer and `max_length` are stubbed, because what is
under test is the arithmetic that decides what to cut, not the weights.
"""

from typing import Any

from memoryos.adapters.reranking.cross_encoder import CrossEncoderReranker


class StubTokenizer:
    """One token per whitespace-separated word. Enough to measure a budget."""

    model_max_length = 32

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"w{index}" for index in token_ids)


class StubbedReranker(CrossEncoderReranker):
    """The real truncation logic over a stubbed window and tokenizer.

    A subclass rather than a patched property: `max_length` normally reads
    through the loaded model, and overriding it here keeps `_load` — and the
    several hundred megabytes behind it — out of a unit test while leaving
    `fit` exactly as it ships.
    """

    def __init__(self, max_length: int = 32) -> None:
        super().__init__("stub/model")
        self._max = max_length
        self._tokenizer = StubTokenizer()

    @property
    def max_length(self) -> int:
        return self._max


def stubbed(max_length: int = 32) -> CrossEncoderReranker:
    return StubbedReranker(max_length)


def test_a_pair_longer_than_the_window_is_truncated_before_the_model_sees_it() -> None:
    reranker = stubbed(max_length=32)
    query = "why does the worker hold a lease"  # 7 tokens
    document = " ".join(f"word{index}" for index in range(200))

    fitted = reranker.fit(query, document)

    assert len(fitted.split()) < len(document.split()), "it was actually cut"
    # 32 window - 7 query - 4 special = 21 tokens of document.
    assert len(fitted.split()) == 32 - 7 - 4

    # The whole pair now fits, which is the property that matters — not the
    # exact budget.
    assert len(query.split()) + len(fitted.split()) + 4 <= reranker.max_length


def test_a_document_that_already_fits_is_passed_through_unchanged() -> None:
    """The common case must cost one token count and no rewriting."""
    reranker = stubbed(max_length=32)
    document = "short enough to fit beside the query"

    assert reranker.fit("a query", document) is document


def test_a_query_that_fills_the_window_leaves_no_room_and_says_so() -> None:
    """Degenerate, but silent truncation to nothing is exactly the M1.6.1 shape.

    Returning the empty string is the honest outcome — there is no room for a
    document — and the adapter logs a warning rather than scoring every
    candidate against a truncated question and reporting the ranking as real.
    """
    reranker = stubbed(max_length=8)
    query = " ".join(f"q{index}" for index in range(20))

    assert reranker.fit(query, "any document at all") == ""


def test_the_window_falls_back_when_a_tokenizer_reports_a_sentinel() -> None:
    """Some tokenizers ship `model_max_length` as 1e30 meaning "unset"."""
    reranker = CrossEncoderReranker("stub/model")

    class Sentinel:
        model_max_length = int(1e30)

    class FakeModel:
        max_length = None
        tokenizer: Any = Sentinel()

    reranker._model = FakeModel()  # type: ignore[assignment]
    reranker._tokenizer = Sentinel()

    assert reranker.max_length == 512
