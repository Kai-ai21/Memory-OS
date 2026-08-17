"""The classifier, and particularly the cases where it must refuse to fire.

Detection is the easy half. The failure that costs something is the reverse —
a statement read as a question is a thought that was never written down — so
roughly half of what follows is statements that contain interrogative words,
question-shaped verbs, or an unlucky full stop, and asserts they are stored.
"""

import pytest

from memoryos.domain.message_intent import MessageIntent, classify

STATEMENTS = [
    "postgres full-text search is faster than I expected",
    # An interrogative in the middle of a clause. The commonest trap by far, and
    # the one that would silently eat notes about how anything works.
    "I finally worked out how the lease fencing stops a stale worker writing",
    "the chunker sizes itself from the model window, which is why 512 appears",
    "spent the afternoon on why the graph diverged and it was the predicate",
    # `when` mid-clause, and a date, and neither makes it a question.
    "I noticed when the timeline is grouped by week the gaps stop looking real",
    # An imperative that is not a query imperative.
    "remember to re-run the extraction after changing the prompt",
    "have a look at the SKIP LOCKED path before touching the queue",
    # Exclamatory openings that borrow an interrogative's first word.
    "what a mess the entity duplicates are",
    "how odd that the mtimes all cluster on one afternoon",
    # A file name splits into fragments on the full stop. Neither fragment opens
    # a question, so the whole thing is still a statement.
    "graph_expand.py is where the hub suppression lives",
    "pgvector 0.8.0 is what the container reports",
    # Short, and a perfectly good thought. Nothing here requires four words.
    "reranking helps",
]

QUESTIONS = [
    "what did I say about postgres?",
    # No question mark, and still a question.
    "what did I say about postgres",
    "why does the worker use SKIP LOCKED",
    "how many chunks are unembedded?",
    "which sources have never been synced",
    "who wrote the fusion weights",
    "when did I last touch the reranker",
    # Imperative query forms.
    "show me what I said about indexing",
    "find everything about the entity merge threshold",
    "remind me what I decided about chunk overlap",
    # Auxiliary-fronted yes/no.
    "did I ever write down why the graph is a projection",
    "is there anything in here about neo4j",
    "have I said anything about declared dates",
    # Filler in front of a question does not make it a statement.
    "so what did I decide about the overlap",
    "hmm, what did I say about postgres?",
]

BOTH = [
    # The case the enum exists for: a dash, a claim, a question.
    "I think the queue is fine — does that match what I said before?",
    "the reranker is doing most of the work here. what did I say about it before?",
    # Two clauses, a semicolon, one of each.
    "declared dates are the first honest ones in this corpus; what does the "
    "timeline look like now?",
]


@pytest.mark.parametrize("text", STATEMENTS)
def test_a_statement_is_stored(text: str) -> None:
    assert classify(text) is MessageIntent.STATEMENT


@pytest.mark.parametrize("text", QUESTIONS)
def test_a_question_is_answered_and_not_stored(text: str) -> None:
    assert classify(text) is MessageIntent.QUESTION


@pytest.mark.parametrize("text", BOTH)
def test_a_claim_beside_a_question_is_both(text: str) -> None:
    assert classify(text) is MessageIntent.BOTH


def test_filler_in_front_of_a_question_is_not_a_claim() -> None:
    """"ok." is not a thought, and storing it would be worse than dropping it.

    The substance threshold exists for exactly this shape and for nothing else:
    it can only ever move a message from `QUESTION` to `BOTH`, never from
    `STATEMENT` to anything.
    """
    assert classify("ok. what did I say about postgres?") is MessageIntent.QUESTION
    assert classify("hmm. why is the graph empty?") is MessageIntent.QUESTION


def test_a_long_paste_ending_in_a_question_is_stored_as_well_as_answered() -> None:
    pasted = (
        "The worker claims the oldest pending job and holds a lease on it.\n"
        "Fencing is what stops a worker whose lease expired writing terminal state.\n"
        "Does that match what I wrote down earlier?"
    )
    assert classify(pasted) is MessageIntent.BOTH


def test_ambiguity_resolves_towards_storing() -> None:
    """The bias, asserted directly rather than left to the cases above.

    None of these are confidently statements. All of them are stored, because
    the alternative is losing them.
    """
    for text in (
        "wondering whether the graph is worth the operational cost",
        "not sure how the hub ratio was picked",
        "the question is whether declared dates should outrank mtimes",
    ):
        assert classify(text) is MessageIntent.STATEMENT, text


def test_an_empty_message_is_not_a_question() -> None:
    """Emptiness is the caller's error to raise, not a classification.

    `STATEMENT` here means "nothing recognised", and `Chat` rejects the message
    on its own terms before this answer is ever acted on.
    """
    assert classify("") is MessageIntent.STATEMENT
    assert classify("   \n  ") is MessageIntent.STATEMENT
