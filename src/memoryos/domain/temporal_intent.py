"""Whether a query is asking about *time*, and what it is asking.

Pure Python, rules over a regex. **Not a language model, and that is a decision
rather than a shortcut.** A model call here would cost money and a round trip on
every search, would make retrieval non-reproducible — the same query could parse
differently on two runs and nobody could tell why a result moved — and would buy
nothing, because the thing being detected is a closed set of English phrases that
fit on one screen. The rules are testable in microseconds and the failures are
readable.

**The hard part is not detection. It is refusal.** Most queries carry no temporal
signal at all, and the failure mode of this whole milestone is a parser that fires
when it should not: a query mentioning a date is not necessarily a query about
time. Three specific traps, all of which this corpus contains:

* **`may` is a modal verb**, and by far the commonest month name in ordinary
  prose. "what may cause a chunk to be dropped" is a question about chunking.
* **`march`, `august` and `mark` are ordinary words** — a verb, an adjective, a
  name. So is `first` in "the first argument", and `last` in "the last chunk".
* **A month can name a thing rather than a time**: "the May release notes" is a
  document, if one exists.

So a month name only counts when a *temporal preposition* precedes it, or a year
or day number sits beside it. `in August`, `during July`, `since March 2026`,
`on 10 August` — but never a bare `August`. The preposition is what turns a word
into a date, and requiring it is what makes the trap query in the golden set come
back `None`.

The same conservatism applies to ordering. `first` and `latest` only count when
they modify the *subject of the question* rather than a noun the corpus contains,
which is approximated by requiring them near the start of the query or attached to
a temporal noun (`version`, `change`, `commit`). This is the weakest rule here and
is documented as such.

Returning `None` is the common case and is deliberately the cheap one: three
lowercase scans over a short string, and no allocation when nothing matches.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum, auto


class IntentKind(StrEnum):
    """What the query is asking of time, which decides what is done about it."""

    # "in August", "on 10 August", "last month" — a bounded window. Becomes a
    # hard filter, because a question about a period is not answered by a
    # document from a different one however relevant it looks.
    RANGE = auto()
    # "recently", "lately" — an unbounded preference for newer. Becomes a
    # weight, not a filter: there is no boundary to cut at, and inventing one
    # would drop the answer whenever the guess was wrong.
    RELATIVE = auto()
    # "first", "latest", "original" — a request for an extreme. Reorders what
    # relevance already selected.
    ORDERING = auto()


class Ordering(StrEnum):
    EARLIEST = auto()
    LATEST = auto()


@dataclass(frozen=True, slots=True)
class TemporalIntent:
    """What was detected, and the phrase that caused it.

    `phrase` is not decoration. A query silently reinterpreted as temporal is the
    most confusing failure this milestone can produce — results change, the
    reason is invisible, and the user's own words are the only thing that can
    explain it. It goes into `ScoreBreakdown` and out through `--explain`.
    """

    kind: IntentKind
    phrase: str
    start: datetime | None = None
    end: datetime | None = None
    ordering: Ordering | None = None

    @property
    def is_range(self) -> bool:
        return self.kind is IntentKind.RANGE

    def describe(self) -> str:
        """One line, for the breakdown and the explain output."""
        if self.kind is IntentKind.RANGE and self.start and self.end:
            return f"range {self.start:%Y-%m-%d}..{self.end:%Y-%m-%d} (from {self.phrase!r})"
        if self.kind is IntentKind.ORDERING and self.ordering:
            return f"ordering {self.ordering.value} (from {self.phrase!r})"
        return f"{self.kind.value} (from {self.phrase!r})"


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# The words that turn a month name into a date. Without one of these in front,
# `August` is a word — and on this corpus `may` is a modal verb far more often
# than it is a month.
#
# Which preposition it is decides the *bounds*, not just whether there are any.
# `in August` is the month; `since August` is everything from its first day
# onward; `before August` is everything up to it; `after August` starts where the
# month ends. Collapsing all four to "the month of August" would answer a
# different question than the one asked, silently, and would look right whenever
# the corpus happened to sit inside the month.
_CONTAINING = ("in", "on", "during", "around", "back in")
_ONWARD = ("since", "from")
_AFTER = ("after",)
_UNTIL = ("before", "until", "till")

_PREPOSITION = r"(?:in|on|during|since|from|before|after|until|till|back\s+in|around)"

_MONTH_NAMES = "|".join(MONTHS)

# `in August`, `during august 2026`, `since March`.
_PREPOSED_MONTH = re.compile(
    rf"\b(?P<prep>{_PREPOSITION})\s+(?P<month>{_MONTH_NAMES})\b(?:\s+(?P<year>\d{{4}}))?",
    re.IGNORECASE,
)

# `on 10 August`, `on August 10`, with or without a year. A day number beside a
# month is unambiguous enough to stand without the preposition test above, but
# the preposition is still required — `August 10` alone is as likely to be a
# heading as a date.
_PREPOSED_DAY = re.compile(
    rf"\b(?P<prep>{_PREPOSITION})\s+"
    rf"(?:(?P<day1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month1>{_MONTH_NAMES})"
    rf"|(?P<month2>{_MONTH_NAMES})\s+(?P<day2>\d{{1,2}})(?:st|nd|rd|th)?)"
    rf"\b(?:,?\s+(?P<year>\d{{4}}))?",
    re.IGNORECASE,
)

# `last month`, `this week`, `past 3 days`, `last two weeks`.
_WINDOW = re.compile(
    r"\b(?P<det>last|past|previous|this)\s+"
    r"(?:(?P<count>\d+|a|one|two|three|four|five|six)\s+)?"
    r"(?P<unit>day|week|fortnight|month|quarter|year)s?\b",
    re.IGNORECASE,
)

_TODAY = re.compile(r"\b(?P<phrase>today|yesterday)\b", re.IGNORECASE)

# An unbounded lean towards the newer. No boundary is implied by any of these,
# which is exactly why they become a weight rather than a filter.
_RELATIVE = re.compile(
    r"\b(?P<phrase>recently|lately|most\s+recent(?:ly)?|these\s+days|nowadays"
    r"|of\s+late|just\s+now|the\s+other\s+day|current(?:ly)?)\b",
    re.IGNORECASE,
)

# Nouns that make `first`/`latest` temporal rather than positional. "the first
# argument" and "the last chunk" are positions in a structure; "the first
# version" is a point in time.
#
# **Deliberately narrow, and narrowed once already by measurement.** The list
# started with `time`, `thing`, `work`, `state`, `shape` and `form` on it, and
# `time` cost a real regression: "how does the system know a file changed since
# last time" parsed as *ordering: latest*, and the date sort dropped its nDCG
# from 0.963 to 0.868. "The last time" is idiomatic English for "the previous
# occasion" far more often than it is a request for the newest thing, and the
# same is true of every vague noun that was beside it.
#
# So this is now only nouns that denote a *versioned artifact* — things a corpus
# can hold several of, in order. Anything looser reads temporal intent into
# ordinary phrasing, which is this milestone's whole failure mode.
_TEMPORAL_NOUN = (
    r"(?:version|revision|change|edit|commit|iteration|draft|release)"
)

_EARLIEST = re.compile(
    rf"\b(?P<phrase>(?:the\s+)?(?:first|earliest|original|initial)\s+{_TEMPORAL_NOUN}"
    rf"|originally|to\s+begin\s+with|at\s+first)\b",
    re.IGNORECASE,
)

_LATEST = re.compile(
    rf"\b(?P<phrase>(?:the\s+)?(?:latest|newest|last|most\s+recent)\s+{_TEMPORAL_NOUN}"
    rf"|so\s+far|up\s+to\s+now)\b",
    re.IGNORECASE,
)

_UNIT_DAYS = {
    "day": 1,
    "week": 7,
    "fortnight": 14,
    "month": 30,
    "quarter": 91,
    "year": 365,
}

_COUNT_WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def parse_temporal_intent(
    query: str, *, now: datetime | None = None
) -> TemporalIntent | None:
    """What this query asks of time, or `None` if it asks nothing.

    Checked most-specific first. A query saying "in August" *and* "recently" is
    asking about August; the range is the stronger claim and the one with a
    boundary, so it wins rather than the two combining into something neither
    phrase said.

    `now` is injected rather than read from the clock so the same query parses
    identically in a test and in a replay. Every window here is relative to it.
    """
    if not query.strip():
        return None

    moment = (now or datetime.now(UTC)).astimezone(UTC)

    # 1. An explicit range. A boundary beats a preference, always.
    explicit = _explicit_range(query, moment)
    if explicit is not None:
        return explicit

    # 2. An extreme. Checked before `relative` because "the most recent change"
    # is a request for one thing, not a lean towards newer ones — and the
    # `_RELATIVE` pattern would otherwise swallow "most recent".
    ordering = _ordering(query)
    if ordering is not None:
        return ordering

    # 3. An unbounded preference.
    match = _RELATIVE.search(query)
    if match is not None:
        return TemporalIntent(
            kind=IntentKind.RELATIVE, phrase=match.group("phrase").lower()
        )

    return None


def _explicit_range(query: str, now: datetime) -> TemporalIntent | None:
    day = _PREPOSED_DAY.search(query)
    if day is not None:
        month_name = (day.group("month1") or day.group("month2") or "").lower()
        day_number = int(day.group("day1") or day.group("day2"))
        year = int(day.group("year")) if day.group("year") else None
        start = _day_start(MONTHS[month_name], day_number, year, now)
        if start is not None:
            bounds = _bounds(day.group("prep"), start, start + timedelta(days=1), now)
            return TemporalIntent(
                kind=IntentKind.RANGE,
                phrase=day.group(0).strip().lower(),
                start=bounds[0],
                end=bounds[1],
            )

    month = _PREPOSED_MONTH.search(query)
    if month is not None:
        name = month.group("month").lower()
        year = int(month.group("year")) if month.group("year") else None
        start = _month_start(MONTHS[name], year, now)
        bounds = _bounds(month.group("prep"), start, _next_month(start), now)
        return TemporalIntent(
            kind=IntentKind.RANGE,
            phrase=month.group(0).strip().lower(),
            start=bounds[0],
            end=bounds[1],
        )

    today = _TODAY.search(query)
    if today is not None:
        phrase = today.group("phrase").lower()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if phrase == "yesterday":
            start -= timedelta(days=1)
        return TemporalIntent(
            kind=IntentKind.RANGE, phrase=phrase, start=start, end=start + timedelta(days=1)
        )

    window = _WINDOW.search(query)
    if window is not None:
        raw = window.group("count")
        count = _COUNT_WORDS.get((raw or "1").lower(), 0) or int(raw) if raw else 1
        days = _UNIT_DAYS[window.group("unit").lower()] * max(count, 1)
        return TemporalIntent(
            kind=IntentKind.RANGE,
            phrase=window.group(0).strip().lower(),
            start=now - timedelta(days=days),
            # Open at the future end rather than clamped to `now`: a corpus can
            # legitimately hold an item dated tomorrow — a clock problem, or a
            # calendar — and a window that excluded it would hide the thing the
            # question is most likely about.
            end=now + timedelta(days=1),
        )

    return None


def _ordering(query: str) -> TemporalIntent | None:
    earliest = _EARLIEST.search(query)
    if earliest is not None:
        return TemporalIntent(
            kind=IntentKind.ORDERING,
            phrase=earliest.group("phrase").lower(),
            ordering=Ordering.EARLIEST,
        )
    latest = _LATEST.search(query)
    if latest is not None:
        return TemporalIntent(
            kind=IntentKind.ORDERING,
            phrase=latest.group("phrase").lower(),
            ordering=Ordering.LATEST,
        )
    return None


def _bounds(
    preposition: str, start: datetime, end: datetime, now: datetime
) -> tuple[datetime | None, datetime | None]:
    """The window a preposition names around a period.

    Open-ended on one side for three of the four, and `None` is the honest way
    to say that: "before August" has no lower bound in a corpus, and inventing
    one — the epoch, the first memory — would be a filter nobody asked for that
    silently drops anything older than the guess.
    """
    word = " ".join(preposition.lower().split())
    if word in _ONWARD:
        return start, None
    if word in _AFTER:
        return end, None
    if word in _UNTIL:
        return None, start
    return start, end


def _month_start(month: int, year: int | None, now: datetime) -> datetime:
    """The most recent occurrence of this month at or before `now`.

    Backwards rather than nearest. A corpus is a record of what has happened, so
    "in March" asked in February means last March — reading it as next month
    would name a window that cannot contain anything.
    """
    if year is not None:
        return datetime(year, month, 1, tzinfo=UTC)
    candidate = datetime(now.year, month, 1, tzinfo=UTC)
    if candidate > now:
        candidate = datetime(now.year - 1, month, 1, tzinfo=UTC)
    return candidate


def _day_start(month: int, day: int, year: int | None, now: datetime) -> datetime | None:
    """A specific day, or None if the numbers do not name one.

    `on 31 February` is a typo, not a date, and the whole query parses to nothing
    rather than being widened to February. Widening looks charitable and is not:
    it applies a hard filter nobody asked for, derived from a phrase the parser
    has just admitted it could not read. A parser that does not understand a
    query should not narrow it.
    """
    start = _month_start(month, year, now)
    try:
        return start.replace(day=day)
    except ValueError:
        return None


def _next_month(moment: datetime) -> datetime:
    year, month = divmod(moment.month, 12)
    return moment.replace(year=moment.year + year, month=month + 1, day=1)
