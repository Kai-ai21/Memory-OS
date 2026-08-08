"""Sentence splitting.

A regex with an abbreviation list, not nltk or spacy. Both are large
dependencies carrying model downloads, and the accuracy they buy is accuracy in
placing a boundary that is already approximate — chunk edges are a heuristic,
not a semantic claim.
"""

import re

# The abbreviations that actually appear in the kind of text this ingests.
# Not exhaustive by design: a missed one splits a sentence early, which costs a
# slightly odd chunk boundary and nothing else.
_ABBREVIATIONS = frozenset(
    [
        # Titles
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "rev", "hon",
        # Organisations
        "inc", "ltd", "co", "corp", "dept", "univ",
        # Latin and general
        "e.g", "i.e", "etc", "vs", "cf", "al", "viz", "approx",
        # Document references
        "fig", "eq", "ref", "sec", "ch", "vol", "no", "p", "pp",
        # Dates
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sept", "sep",
        "oct", "nov", "dec",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        # Initialisms that end in a period
        "a.m", "p.m", "u.s", "u.k", "e.u",
    ]
)  # fmt: skip

# A terminator, optional closing quotes or brackets, whitespace, then something
# that looks like a new sentence: a capital, a digit, or an opening quote.
_BOUNDARY = re.compile(r'(?<=[.!?])(["\')\]]*)(\s+)(?=[A-Z0-9"\'(\[])')


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans as `(start, end)` offsets into `text`.

    Offsets rather than strings, because everything downstream needs to map a
    chunk back to its exact position in the document — that is what lets a
    citation highlight the matched span rather than the whole file.
    """
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    start = 0

    for match in _BOUNDARY.finditer(text):
        end = match.end(1)
        if _ends_with_abbreviation(text, end):
            continue
        spans.append((start, end))
        start = match.end()

    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _ends_with_abbreviation(text: str, end: int) -> bool:
    """Whether the '.' at `end` closes an abbreviation rather than a sentence."""
    if end == 0 or text[end - 1] != ".":
        return False

    # Walk back over the word before the period, allowing internal dots so that
    # "e.g." and "U.S." are recognised whole.
    index = end - 1
    while index > 0 and (text[index - 1].isalnum() or text[index - 1] == "."):
        index -= 1

    word = text[index : end - 1].lower().strip(".")
    if not word:
        return False
    # A single letter followed by a period is nearly always an initial.
    return word in _ABBREVIATIONS or len(word) == 1
