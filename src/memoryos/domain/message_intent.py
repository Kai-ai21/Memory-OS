"""Whether what you typed is a claim, a question, or both.

Pure Python, rules over a regex, for exactly the reasons `temporal_intent`
gives and one more that is specific to this milestone. A model call here would
sit on the critical path of *every keystroke's worth of typing that gets sent* —
the one operation this phase promises is instant — and it would make the decision
non-reproducible, so a message routed the wrong way could not be explained by
reading anything. What is being detected is a closed set of English openings that
fits on one screen.

**The asymmetry is the whole design.** The two mistakes this classifier can make
are not equal:

* Reading a question as a statement stores a line of text nobody wanted stored.
  It is visible in the message list, it is one click to correct, and the corpus
  carries one junk memory.
* Reading a statement as a question throws the thought away. There is no row, no
  event, no artifact — nothing to correct later, because nothing was written.

So every rule below is biased towards `STATEMENT`, and `classify` returns it for
anything it cannot place. That bias is not a hedge against weak rules; it is the
rule. A thought that took a minute to form is worth more than the cost of an
answer nobody asked for.

`BOTH` exists because the interesting messages are mixed. "I think the queue is
fine — does that match what I said before?" is a claim worth keeping and a
question worth answering, and a classifier with two outcomes has to discard one
of them. It resolves the same way everything else here does: the claim is stored
*and* the question is answered.

Nothing in this module is aware of storage, retrieval or the shape of a chat
turn. It takes a string and returns an enum.
"""

import re
from enum import StrEnum, auto


class MessageIntent(StrEnum):
    """What to do with a message, which is the only question this answers."""

    # Store it. The default, and what an unrecognised message gets.
    STATEMENT = auto()
    # Answer it, and store nothing. An answer is derived from the corpus, so
    # storing the question that produced it would add a row that is evidence for
    # nothing.
    QUESTION = auto()
    # Store the claim, answer the question. One message, both behaviours.
    BOTH = auto()


# Clause boundaries, not sentence boundaries.
#
# A full stop is here, and so are the dashes and the semicolon, because the
# mixed message this milestone cares about is punctuated with a dash rather than
# a full stop: "I think X — does that match what I said before?" is one sentence
# and two clauses, and a splitter that only knew about sentences would see a
# trailing question mark and throw the claim away.
#
# Splitting on `.` mangles `sync.py` and `0.8.0` into fragments. That is
# harmless here and worth stating plainly rather than defending against: the
# fragments are only ever inspected for a question-shaped *opening*, and `py` is
# not one. A tokenizer that got file names right would cost more than the
# failure it prevents.
#
# The dashes are written as escapes rather than literally: an em dash and an en
# dash are one pixel apart on screen and indistinguishable in a diff, and this
# expression depends on both being present.
_EM_DASH = "\u2014"
_EN_DASH = "\u2013"
_BOUNDARY = re.compile(
    rf"(?<=[.!?])\s+|\s+[{_EM_DASH}{_EN_DASH}]\s+|\s+--\s+|[;\n]+"
)

# The mark that settles it. A clause ending in one is a question whatever it
# opens with, which is the only rule here that needs no judgement.
_TERMINAL = "?"

# Leading interrogatives. Only ever matched at the *start* of a clause, because
# they are ordinary words in the middle of one: "I finally worked out how the
# lease fencing works" is a statement, and a rule that scanned for `how`
# anywhere would answer it instead of keeping it.
_INTERROGATIVE = (
    "what",
    "whats",
    "what's",
    "why",
    "whys",
    "why's",
    "how",
    "hows",
    "how's",
    "when",
    "whens",
    "when's",
    "where",
    "wheres",
    "where's",
    "which",
    "who",
    "whos",
    "who's",
    "whom",
    "whose",
)

# `what a mess`, `how odd that the mtimes all cluster`. Exclamations wearing an
# interrogative's opening word, and common in exactly the register somebody types
# into a box like this one. Two constructions rather than a general rule, because
# both are recognisable by their shape and a general rule would have to guess:
# `what` or `how` followed by an article, and `how` followed by an adjective and
# `that`. Each one excluded is a thought that gets kept.
_EXCLAMATORY = re.compile(
    r"^(?:(?:what|how)\s+(?:a|an)\b|how\s+\w+\s+that\b)", re.IGNORECASE
)

# Imperative query forms. A question does not have to be shaped like one — "show
# me what I said about indexing" asks for a retrieval as plainly as "what did I
# say about indexing" does, and a person who has used a search box will type
# both.
_QUERY_IMPERATIVE = (
    "find",
    "show me",
    "show the",
    "list",
    "search",
    "look up",
    "look for",
    "tell me",
    "remind me",
    "give me",
    "pull up",
    "recall",
    "summarise",
    "summarize",
)

# Auxiliary-fronted yes/no questions: `did I say`, `is there anything`, `have I
# written`. The auxiliary alone is not enough — "have a look at the queue" is an
# instruction and "can do" is agreement — so it has to be followed by something
# that could be a subject. That pairing is what keeps the rule from firing on
# ordinary imperative prose.
_AUXILIARY = (
    "do",
    "does",
    "did",
    "is",
    "isnt",
    "isn't",
    "are",
    "arent",
    "aren't",
    "was",
    "were",
    "have",
    "has",
    "had",
    "can",
    "could",
    "should",
    "would",
    "will",
    "am",
)

_SUBJECT = (
    "i",
    "we",
    "you",
    "it",
    "he",
    "she",
    "they",
    "there",
    "this",
    "that",
    "these",
    "those",
    "the",
    "my",
    "our",
    "any",
    "anything",
    "anyone",
    "something",
)

# How much text a non-question clause needs before it counts as a claim worth
# storing alongside a question.
#
# **This threshold only ever decides between `QUESTION` and `BOTH`.** A short
# message with no question in it is a statement regardless — "postgres is fast"
# is three words and a perfectly good thought. What this stops is the leading
# "ok." or "hmm," in front of a question turning that question into a stored
# memory containing the word "ok", which is a row nobody would ever want back.
_SUBSTANTIAL_WORDS = 4


def classify(text: str) -> MessageIntent:
    """What to do with `text`.

    Returns `STATEMENT` for anything unrecognised, including the empty string.
    Callers that cannot store an empty message reject it on its own terms rather
    than relying on a classification to do it for them — "this is not storable"
    and "this is a question" are different answers and only one of them is this
    function's to give.
    """
    clauses = [clause for clause in _BOUNDARY.split(text) if clause and clause.strip()]

    questions = [clause for clause in clauses if _is_question(clause)]
    if not questions:
        return MessageIntent.STATEMENT

    claims = [
        clause
        for clause in clauses
        if not _is_question(clause) and _is_substantial(clause)
    ]
    return MessageIntent.BOTH if claims else MessageIntent.QUESTION


def _is_question(clause: str) -> bool:
    stripped = clause.strip()
    if not stripped:
        return False
    if stripped.endswith(_TERMINAL):
        return True
    if _EXCLAMATORY.match(stripped):
        return False

    lowered = _opening(stripped)
    if not lowered:
        return False

    words = lowered.split()
    if words[0] in _INTERROGATIVE:
        return True
    if any(lowered.startswith(form) for form in _QUERY_IMPERATIVE):
        return True
    return len(words) >= 2 and words[0] in _AUXILIARY and words[1] in _SUBJECT


def _opening(clause: str) -> str:
    """The clause lowercased, with leading punctuation and filler removed.

    "so, what did I decide" opens with `so,` and asks a question. Stripping the
    punctuation rather than splitting on it keeps the filler out of the way
    without turning one clause into two — a comma is not a clause boundary here,
    because "hmm, what did I say" is one thought and splitting it would leave
    "hmm" behind as a claim.
    """
    lowered = clause.strip().lower()
    lowered = re.sub(r"^[^\w]+", "", lowered)
    for filler in ("so ", "ok ", "okay ", "hmm ", "well ", "and ", "but ", "also "):
        if lowered.startswith(filler):
            lowered = lowered[len(filler) :]
    return re.sub(r"^[^\w]+", "", lowered)


def _is_substantial(clause: str) -> bool:
    return len(clause.split()) >= _SUBSTANTIAL_WORDS
