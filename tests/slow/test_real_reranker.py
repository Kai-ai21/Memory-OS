"""The assertion a fake cross-encoder cannot make.

Every other reranking test uses `FakeReranker`, which proves the pipeline
honours whatever the model says — and would pass identically against a model
that scored by string length. This one loads the real weights and asserts the
thing that actually matters: that the model can tell an answer from a distractor
when both are plausible-looking prose.

Marked slow because it downloads and holds a model. It is the same bargain as
`test_real_model.py`: one test that establishes the component is doing its job,
so the fast suite is free to assume it.
"""

import pytest

from memoryos.adapters.reranking.cross_encoder import (
    DEFAULT_MODEL,
    CrossEncoderReranker,
)

pytestmark = pytest.mark.slow

QUERY = "how does a worker keep hold of a task it is already running"

RELEVANT = (
    "Each claimed task carries a lease with an expiry. While the handler runs it "
    "renews that lease periodically, which is how a long-running task keeps its "
    "hold on the work it started. If the process dies the lease lapses and a "
    "sweeper returns the task to the pending pool."
)
DISTRACTOR = (
    "A wild yeast starter is fed flour and water on a fixed schedule until it "
    "doubles reliably. The dough is then folded gently at intervals and given a "
    "long cold rest in the refrigerator before baking."
)
# Deliberately the harder negative: same vocabulary as the query, wrong topic.
# A bag-of-words scorer would rank this first.
HARD_DISTRACTOR = (
    "The task of keeping a running record of who holds which library book is "
    "handled by a worker at the front desk, who renews a loan when a reader asks "
    "to keep it longer."
)


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def test_the_relevant_document_scores_higher_than_a_distractor(
    reranker: CrossEncoderReranker,
) -> None:
    relevant, distractor = reranker.rerank(QUERY, [RELEVANT, DISTRACTOR])

    assert relevant > distractor, (relevant, distractor)


def test_it_beats_a_distractor_that_shares_the_query_vocabulary(
    reranker: CrossEncoderReranker,
) -> None:
    """The case a lexical retriever gets wrong, which is why this model exists.

    `worker`, `task`, `keep`, `holds`, `renews` all appear in the wrong
    document. Term overlap ranks it first; reading the pair does not.
    """
    scores = reranker.rerank(QUERY, [DISTRACTOR, HARD_DISTRACTOR, RELEVANT])
    bread, library, lease = scores

    assert lease > library, (lease, library)
    assert lease > bread, (lease, bread)
    # Order is returned as given, not sorted — the caller does the sorting.
    assert len(scores) == 3


def test_the_model_reports_a_real_window_and_truncates_to_it(
    reranker: CrossEncoderReranker,
) -> None:
    """Measured, not assumed. The M1.6.1 defect was a window taken on faith."""
    assert reranker.model_id.startswith(DEFAULT_MODEL)
    assert 64 <= reranker.max_length <= 512

    enormous = " ".join(f"token{index}" for index in range(5_000))
    fitted = reranker.fit(QUERY, enormous)

    assert len(fitted) < len(enormous)
    # The whole pair now fits the window the model actually reports.
    pair_tokens = reranker._count(QUERY) + reranker._count(fitted)
    assert pair_tokens <= reranker.max_length

    # And an over-long pair still scores rather than raising.
    (score,) = reranker.rerank(QUERY, [enormous])
    assert isinstance(score, float)
