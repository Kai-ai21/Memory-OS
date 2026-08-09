"""Is the query instruction prefix worth applying on this corpus?

M1.6.1 recorded the topical margin here as a known weakness — the third on-topic
document beat the first off-topic one by about 0.006 — and noted that bge's
query prefix would widen it roughly 5x. This is the measurement that was meant
to confirm that. It does not: on this corpus the prefix makes the queue question
*worse*, inverting the ranking it was supposed to widen.

bge-v1.5's model card is not ambiguous about how to settle this — "the best
method to decide whether to add instructions for queries is choosing the setting
that achieves better performance on your task" — so the adapter applies the
prefix only for models in `APPLY_QUERY_PREFIX`, and this file is the measurement
that decides membership.

The assertion is deliberately written against the *configured* setting rather
than against a fixed winner. Turning the prefix on for a model that this corpus
says should not have it fails here, and so does leaving it off if the corpus
ever changes its mind. That is what keeps this a decision the project rechecks
instead of one it made once and forgot.
"""

import pytest

from memoryos.adapters.embedding.sentence_transformers import (
    DEFAULT_MODEL,
    DOCUMENTED_QUERY_PREFIXES,
    SentenceTransformerEmbedder,
    query_prefix_for,
)
from tests.slow.test_real_search import BAKING_DOCS, QUEUE_DOCS

pytestmark = pytest.mark.slow

# One question per topic, phrased the way somebody would actually ask it —
# sharing no distinctive vocabulary with the documents it should retrieve.
QUESTIONS = {
    "how does a worker take a task and hold onto it": set(QUEUE_DOCS),
    "how do I get a crisp crust on a loaf of bread": set(BAKING_DOCS),
}

DOCUMENTS = {**QUEUE_DOCS, **BAKING_DOCS}


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    """The embedder as configured — whatever `APPLY_QUERY_PREFIX` currently says."""
    return SentenceTransformerEmbedder()


@pytest.fixture(scope="module")
def passages(embedder: SentenceTransformerEmbedder) -> dict[str, list[float]]:
    """The stored side, embedded once and shared by both arms.

    Shared deliberately: only the query side differs between the arms, so
    whatever the window does to these documents it does identically to both and
    cannot account for any difference in margin.
    """
    names = list(DOCUMENTS)
    vectors = embedder.embed_passage([DOCUMENTS[name] for name in names])
    return dict(zip(names, vectors, strict=True))


def margin(
    query_vector: list[float], passages: dict[str, list[float]], on_topic: set[str]
) -> float:
    """Worst on-topic score minus best off-topic score.

    Negative means the ranking is wrong. Near zero means it is right by an
    amount no corpus change should be trusted to preserve.
    """
    scores = {
        name: sum(a * b for a, b in zip(query_vector, vector, strict=True))
        for name, vector in passages.items()
    }
    worst_on = min(scores[name] for name in on_topic)
    best_off = max(score for name, score in scores.items() if name not in on_topic)
    return worst_on - best_off


def arms(
    question: str, passages: dict[str, list[float]]
) -> tuple[float, float]:
    """(configured, alternative) margins for one question.

    Both arms load the same weights and score against the same stored vectors.
    The only difference is whether the query carried the instruction, which is
    why any difference is attributable to the prefix and to nothing else.
    """
    documented = DOCUMENTED_QUERY_PREFIXES[DEFAULT_MODEL]
    configured = query_prefix_for(DEFAULT_MODEL)
    alternative = "" if configured else documented

    on_topic = QUESTIONS[question]
    with_configured = SentenceTransformerEmbedder(query_prefix=configured)
    with_alternative = SentenceTransformerEmbedder(query_prefix=alternative)

    (a,) = with_configured.embed_query([question])
    (b,) = with_alternative.embed_query([question])
    return margin(a, passages, on_topic), margin(b, passages, on_topic)


def test_the_documented_prefix_is_recorded_verbatim() -> None:
    """Applied or not, losing the exact string would make the A/B meaningless."""
    assert DOCUMENTED_QUERY_PREFIXES[DEFAULT_MODEL] == (
        "Represent this sentence for searching relevant passages: "
    )


def test_passages_are_always_encoded_bare(
    embedder: SentenceTransformerEmbedder,
) -> None:
    """The card is unconditional about this: no instruction on passages, ever.

    True whether or not the prefix is applied to queries — an asymmetric model
    used symmetrically is a different wrong answer that also looks fine.
    """
    prefix = DOCUMENTED_QUERY_PREFIXES[DEFAULT_MODEL]
    (bare,) = embedder.embed_passage(["leases expire when a worker dies"])
    (prefixed,) = embedder.embed_passage([prefix + "leases expire when a worker dies"])
    assert bare != prefixed


def test_the_configured_setting_is_the_one_that_ranks_correctly(
    passages: dict[str, list[float]],
) -> None:
    """Whatever is configured must at least get the ordering right.

    A wider margin around a wrong ordering is not an improvement, so this is
    checked before any comparison between the arms.
    """
    for question in QUESTIONS:
        configured, _ = arms(question, passages)
        assert configured > 0, (
            f"{question!r}: the configured setting ranks the wrong topic first "
            f"(margin {configured:+.4f})"
        )


def test_the_configured_setting_ranks_more_questions_correctly(
    passages: dict[str, list[float]],
) -> None:
    """The measurement that decides `APPLY_QUERY_PREFIX`.

    As measured on 2026-08-09, per question:

        queue question:   bare +0.0060   prefixed -0.0007
        baking question:  bare +0.1344   prefixed +0.1556

    Note what that does and does not say. On mean margin the prefix wins
    (+0.0775 against +0.0702), because the baking question gains more than the
    queue question loses. On *rankings* it loses 1-2: the queue question goes
    negative, meaning a bread document is now retrieved above a queue document.

    Correct rankings are the primary criterion and the margin is only the
    tie-break, because a ranking is what a user sees and a margin is not. An
    arm that answers one question wrongly and the other more confidently has
    not improved retrieval; it has traded a correct answer for a number.

    The same direction shows up independently on the real corpus — 719 chunks of
    this repository, the four M1.6 assessment queries — where the prefix changes
    the top result on two of four and costs "why do we store two timestamps" the
    file where the answer is actually written. Two corpora, same failure mode,
    which is why this criterion is not a metric chosen to get an answer.
    """
    correct = {"configured": 0, "alternative": 0}
    totals = {"configured": 0.0, "alternative": 0.0}
    detail: list[str] = []

    for question in QUESTIONS:
        configured, alternative = arms(question, passages)
        correct["configured"] += configured > 0
        correct["alternative"] += alternative > 0
        totals["configured"] += configured
        totals["alternative"] += alternative
        detail.append(
            f"{question[:38]!r}: configured {configured:+.4f} "
            f"alternative {alternative:+.4f}"
        )

    count = len(QUESTIONS)
    report = (
        f"rankings correct: configured {correct['configured']}/{count} vs "
        f"alternative {correct['alternative']}/{count}; mean margin "
        f"{totals['configured'] / count:+.4f} vs {totals['alternative'] / count:+.4f}\n"
        + "\n".join(detail)
        + "\nThe setting in APPLY_QUERY_PREFIX is no longer the better one on "
        "this corpus — re-measure and change it rather than adjusting this test."
    )

    assert correct["configured"] >= correct["alternative"], report
    if correct["configured"] == correct["alternative"]:
        # Only then does the margin get a say.
        assert totals["configured"] >= totals["alternative"], report
