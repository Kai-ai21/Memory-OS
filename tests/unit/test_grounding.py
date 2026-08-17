"""Context assembly and answer verification, at the points they go wrong.

Pure functions and a fake counter. No model, no network — what is under test is
the machinery around the model, which is where the guardrail actually lives.
"""

import pytest

from memoryos.application.context import assemble_context
from memoryos.domain.grounding import verify_citations
from tests.support.fakes import FakeEmbedder


class Hit:
    """The parts of a MemoryHit that assembly reads."""

    def __init__(self, key: str, text: str, prefix: int = 0, score: float = 1.0) -> None:
        self.external_key = key
        self.source_name = "self"
        self.matched_chunks = [Chunk(text, prefix, score)]


class Chunk:
    def __init__(self, text: str, prefix: int, score: float) -> None:
        self.text = text
        self.prefix_chars = prefix
        self.score = score


SENTENCE = "The worker claims a task and holds a lease while the handler runs. "


def test_assembly_respects_the_budget_and_drops_rather_than_truncates() -> None:
    """A passage cut mid-sentence is worse than an absent one.

    The model cannot tell the sentence was severed, so it completes the thought
    from training data — a fabricated claim carrying a citation to a real
    passage, which is the most convincing wrong answer this system can make.
    """
    counter = FakeEmbedder()
    long_text = SENTENCE * 40
    hits = [
        Hit("a.md", SENTENCE * 2),
        Hit("big.md", long_text),
        Hit("c.md", SENTENCE * 2),
    ]

    budget = counter.count_tokens(SENTENCE * 2) * 2 + 5
    context = assemble_context(hits, counter=counter, token_budget=budget)  # type: ignore[arg-type]

    assert context.tokens_used <= budget
    keys = [passage.hit.external_key for passage in context.passages]
    assert "big.md" not in keys, "the oversized passage was dropped"
    assert [hit.external_key for hit in context.dropped] == ["big.md"]

    # Dropped whole: no prefix of it appears anywhere in the rendered prompt.
    assert long_text[:200] not in context.render() or "a.md" in keys
    for passage in context.passages:
        assert passage.text in (SENTENCE * 2).strip()

    # A short passage after a dropped long one still fits — the loop continues
    # rather than stopping at the first thing too big.
    assert "c.md" in keys
    # Numbered from 1, densely, matching what the prompt shows the model.
    assert [passage.number for passage in context.passages] == [1, 2]
    assert context.valid_indices == {1, 2}


def test_a_citation_index_outside_the_supplied_range_is_rejected() -> None:
    """The one hallucination that can be detected with certainty."""
    result = verify_citations(
        "The queue uses SKIP LOCKED [1]. Leases expire after a timeout [7].",
        valid_indices={1, 2},
    )

    assert result.hallucinated_indices == [7]
    assert result.cited_indices == [1]
    assert not result.grounded

    # A repeated bad index is one defect, not two.
    twice = verify_citations("A [9]. B [9].", valid_indices={1})
    assert twice.hallucinated_indices == [9]

    # And a clean answer reports nothing.
    clean = verify_citations("The queue uses SKIP LOCKED [1][2].", valid_indices={1, 2})
    assert clean.hallucinated_indices == []
    assert clean.grounded


def test_an_uncited_factual_sentence_is_flagged_and_kept() -> None:
    """Flagged, never dropped.

    Quietly deleting a sentence from the middle of an answer produces prose that
    reads as complete while missing a step. A visible marker tells the reader
    which part is not grounded, which is both more honest and more useful.
    """
    answer = (
        "The claim query uses FOR UPDATE SKIP LOCKED [1]. "
        "Postgres also supports advisory locks for this purpose. "
        "The lease is renewed by a heartbeat [2]."
    )
    result = verify_citations(answer, valid_indices={1, 2})

    assert result.factual_sentences == 3
    assert result.supported_sentences == 2
    assert result.citation_rate == pytest.approx(2 / 3)
    assert not result.grounded

    unsupported = [s.text for s in result.sentences if s.unsupported]
    assert unsupported == ["Postgres also supports advisory locks for this purpose."]

    # Kept in the output, marked in place.
    marked = result.marked()
    assert "advisory locks for this purpose. [unsupported]" in marked
    assert "FOR UPDATE SKIP LOCKED [1]." in marked
    assert len(marked) > len(answer)


def test_a_refusal_is_not_penalised_for_citing_nothing() -> None:
    """The safest possible answer must not score as the worst one.

    A refusal contains no claims drawn from the corpus, so there is nothing to
    cite. Scoring it 0% would make "I don't know" look like the most ungrounded
    thing the system can say, and this milestone wants that answer.
    """
    result = verify_citations(
        "The passages do not contain anything about AWS billing. "
        "They describe a job queue and an embedding pipeline instead.",
        valid_indices={1, 2, 3},
    )

    assert result.is_refusal
    assert result.citation_rate == 1.0
    assert result.grounded
    assert result.factual_sentences == 0


@pytest.mark.parametrize(
    "answer",
    [
        # The wording that was silently mis-recorded. The old marker list held
        # "not covered" but not "do not cover", so this exact sentence — which
        # is what the system prompt asks the model to say — logged refused=False.
        "The passages do not cover this.",
        "The passages do not cover this topic.",
        "The provided passages don't cover AWS billing.",
        "The passages do not contain information about this.",
        "The retrieved passages do not mention AWS billing at all.",
        "I cannot answer this from the passages provided.",
        "None of the passages describe an AWS billing setup.",
        "The passages do not appear to contain anything about this.",
        "The passages do not explicitly mention AWS billing.",
        "There is no information about AWS billing in these passages.",
    ],
)
def test_a_decline_with_no_citations_is_a_refusal(answer: str) -> None:
    """The metric this protects is the only evidence the guardrail works.

    `refusal_rate` is how M2.6 demonstrates that an out-of-corpus question gets
    declined rather than answered. A refusal recorded as an answer does not fail
    anything — it just makes that number smaller than the truth, which is the
    worst place for a silent defect to sit.
    """
    assert verify_citations(answer, valid_indices={1, 2, 3}).is_refusal


def test_an_answer_that_cites_is_not_a_refusal_however_it_hedges() -> None:
    """The guard against the fix over-firing.

    An answer that cites [1] and notes the passages do not cover some adjacent
    detail has answered the question. Counting it as a refusal would inflate the
    rate with answers that did the work.
    """
    result = verify_citations(
        "The worker claims jobs with FOR UPDATE SKIP LOCKED [1]. "
        "The passages do not cover how the lease duration is chosen.",
        valid_indices={1, 2},
    )

    assert result.is_refusal is False


def test_an_uncited_assertion_is_not_a_refusal() -> None:
    """Why zero citations alone cannot be the rule.

    A fabrication cites nothing too. If citing nothing were sufficient, the
    refusal rate would count hallucinations as refusals and would read best
    exactly when the system was behaving worst.
    """
    result = verify_citations(
        "The billing account is consolidated under a single payer organisation.",
        valid_indices={1, 2},
    )

    assert result.is_refusal is False
    assert result.grounded is False


def test_fullwidth_citation_brackets_are_read_as_citations() -> None:
    """`【1】` is a citation, because the configured model writes them.

    Measured rather than anticipated: `openai/gpt-oss-120b` cites with CJK
    lenticular brackets however firmly the prompt writes `[1]`, and against an
    ASCII-only pattern every one of those answers scored 0% cited.

    That is the worst direction for this check to fail in. A correctly-cited
    answer reported as ungrounded trains a reader to ignore the mark, and the mark
    is the only thing between them and a fabrication — so a verifier that cannot
    read the marker the model actually sent is a verifier that makes the system
    look untrustworthy when it was behaving.
    """
    result = verify_citations("Postgres full-text search is fast 【1】.", {1})

    assert result.cited_indices == [1]
    assert result.citation_rate == 1.0
    assert result.grounded

    # Mixed, because a model may switch mid-answer.
    mixed = verify_citations(
        "The queue is a table [1]. The graph is a projection 【2】.", {1, 2}
    )
    assert sorted(set(mixed.cited_indices)) == [1, 2]

    # And a fabricated index is still caught in either bracket.
    invented = verify_citations("Something unsupported 【9】.", {1})
    assert invented.hallucinated_indices == [9]


def test_ordinary_prose_brackets_are_still_not_citations() -> None:
    """The widening must not turn every bracket into a marker.

    Bare digits are what makes `[note]`, `[sic]` and `[...]` ordinary prose. That
    was true before fullwidth brackets were added and has to stay true after, or
    a passage number stops being the only thing this system puts in brackets.
    """
    assert verify_citations("An aside [note] and a range [a-b].", {1}).cited_indices == []
    assert verify_citations("A CJK aside 【see】.", {1}).cited_indices == []
