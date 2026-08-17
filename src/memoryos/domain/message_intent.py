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


class ChatRole(StrEnum):
    """Who said one turn.

    Two, and there is deliberately no `system`. Nothing in this product shows a
    system message to anybody — the grounding prompt lives in
    `application/answering.py` and is not a turn in anybody's conversation — and a
    role nobody renders is a branch nobody tests.
    """

    USER = auto()
    ASSISTANT = auto()


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


# Words that point at something already said rather than naming it. A question
# built only out of these cannot be retrieved on: there is nothing in it for an
# embedding to be about.
_DEICTIC = frozenset(
    {
        "it",
        "its",
        "that",
        "this",
        "those",
        "these",
        "them",
        "they",
        "one",
        "ones",
        "other",
        "others",
        "another",
        "same",
        "previous",
        "latter",
        "former",
        "second",
        "third",
        "last",
        "next",
        "above",
        "below",
        "instead",
        "again",
        "more",
        "else",
    }
)

# Function words. Not a general stoplist — only what is needed to tell a question
# that names something from one that points at something.
_FUNCTION = frozenset(
    {
        *_INTERROGATIVE,
        *_AUXILIARY,
        "a",
        "an",
        "and",
        "about",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "i",
        "if",
        "in",
        "into",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "the",
        "then",
        "there",
        "to",
        "too",
        "us",
        "was",
        "we",
        "with",
        "you",
        "your",
    }
)

# How many words a question has to name before it is judged able to stand alone.
#
# Two, and the number is doing something specific. One content word — "what about
# latency?" — is a question that means much more in context than out of it, and
# folding the conversation in costs little because the one word still dominates
# the query. Three would start folding the conversation into questions that
# clearly name their own subject.
_STANDALONE_WORDS = 2

_WORD = re.compile(r"[a-z0-9_]+")


def refers_back(text: str) -> bool:
    """Whether this question needs the conversation to be retrievable at all.

    **Measured rather than assumed, and the measurement is why this exists.**
    M10.0 first folded the last three turns into every retrieval query, on the
    argument that "what about the other one?" is unanswerable without them. That
    is true, and the cost of applying it unconditionally was larger than the
    benefit: asked "why did I use external_key instead of a memory id on the
    transcript?" with three turns of unrelated conversation attached, retrieval
    returned passages the model declined to answer from — and the identical
    question asked alone put the two thoughts that answer it at ranks one and
    two. The conversation did not add context; it diluted a query that already
    had all the context it needed.

    So the rule is narrow: a question that *names* things is retrieved on its own
    words, and only one that points at something without naming it borrows the
    conversation. The prompt gets the turns either way — that is free, and it is
    where a reference gets resolved for the model reading the passages.

    Deliberately conservative in the same direction `classify` is. A false
    "stands alone" costs a follow-up its context, which is visible immediately as
    a bad answer to an obviously referential question. A false "refers back"
    quietly degrades a query that was fine, which is the failure nobody sees.
    """
    named = [
        word
        for word in _WORD.findall(text.lower())
        if word not in _FUNCTION and word not in _DEICTIC
    ]
    return len(named) < _STANDALONE_WORDS


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
