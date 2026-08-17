"""When a conversation ends, and what to call it.

Two pure rules, and both of them are guesses that have to be *legible* guesses.

**A session is a view, not a container.** M10.0 stored each message as its own
memory so that a thought from Tuesday connects to one from last month through the
entities they share rather than through having been typed in the same sitting.
That is still the arrangement: nothing here changes what a memory is, and nothing
in this module is consulted by retrieval, by the graph, or by anything that
decides meaning. A session decides what to draw in a list and which three turns
to carry into the next question.

Which is exactly why the boundary can be a clock. Getting it wrong costs a
conversation drawn in two pieces, or two drawn as one — a navigation annoyance,
correctable by looking at the other session. If a session were a container the
same mistake would put a thought in the wrong bucket for good.
"""

import re
from datetime import datetime, timedelta

# Silence that ends a conversation.
#
# Thirty minutes, and the number is a boundary rather than a measurement. The
# thing being guessed at is "did you walk away", which nothing observable
# answers: a person thinking hard about one sentence and a person who went to
# lunch look identical from here, and no threshold separates them.
#
# So it is set by which error is cheaper. Too short splits one train of thought
# into two sessions, which is visible immediately — the second session opens with
# a follow-up to something that is not in it — and costs the follow-up its
# context. Too long joins yesterday evening to this morning, which costs three
# turns of unrelated conversation carried into the first question of the day and
# is the failure M10.0 measured: a diluted retrieval query degrades silently.
#
# Both are recoverable and neither loses anything. Thirty minutes is long enough
# to think and short enough that a break reads as one.
SESSION_GAP = timedelta(minutes=30)

# How long a derived title may be. Long enough for the subject of a thought,
# short enough to sit in a rail beside a date without wrapping.
TITLE_CHARS = 60

# Where a title may be cut. A sentence end first, then a clause, then a word —
# never mid-word, because a truncated identifier reads as a different identifier.
_CLAUSE = re.compile("[.!?;:]|\\s[\u2014\u2013]\\s")


def starts_new_session(now: datetime, last_activity: datetime | None) -> bool:
    """Whether this message opens a conversation rather than continuing one.

    `None` means there is no session to continue, which is the first message this
    system has ever received and every message after an explicit new session.
    """
    if last_activity is None:
        return True
    return now - last_activity >= SESSION_GAP


def title_for(text: str) -> str | None:
    """A session's name, from the first thing typed into it.

    Derived rather than asked for, because a prompt for a title is a prompt
    nobody answers and a conversation nobody names is a conversation nobody can
    find again. Derived from the *first* message specifically: it is the one that
    says what the session was opened to talk about, and a title that tracked the
    latest message would rename a conversation out from under somebody halfway
    through reading the list.

    Cut at a sentence or clause boundary where there is one within the budget,
    because the first clause of a thought is usually the subject of it. Falls back
    to a word boundary, never to a character one — half an identifier is a
    different identifier.

    `None` for text with nothing in it, which the session list renders as
    "untitled" rather than as an empty row. Inventing "Conversation 4" here would
    be a name that says nothing and cannot be searched for.
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return None

    if len(collapsed) <= TITLE_CHARS:
        return collapsed

    head = collapsed[: TITLE_CHARS + 1]
    if (boundary := _CLAUSE.search(head)) is not None and boundary.start() >= 20:
        return head[: boundary.start()].rstrip()

    cut = head.rfind(" ")
    return (head[:cut] if cut > 20 else collapsed[:TITLE_CHARS]).rstrip() + "…"
