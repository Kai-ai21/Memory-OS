"""One canonical form per entity name. Pure Python, no I/O.

This is the cheap first pass of resolution and emphatically not the answer. It
catches names that are the same string wearing different punctuation — "React",
"react", "React.js", "ReactJS" — and it cannot catch "Postgres" versus
"PostgreSQL", because those are different strings by every rule that does not
already know they are the same thing. Embedding similarity is what reaches
those, and an alias table is what reaches the rest.

**Type-specific, because the rules contradict each other across types.**
Stripping a trailing `js` is right for TECHNOLOGY and destroys a PERSON named
"Js" and a FILE called `index.js`. Stripping `.py` is right for a FILE and wrong
for the PROJECT named after it. There is no type-blind normalisation that is
correct for all seven, and pretending otherwise is how "React" and "Preact"
end up in the same bucket.

**The asymmetry that matters: a false merge is far worse than a missed one.**
Two entities that should be one leave a traversal with a path it cannot walk —
bad, visible, and fixable by a later merge. Two entities wrongly made one
produce a path that does not exist in the corpus, and every traversal through it
reports a connection nobody wrote. So every rule here is conservative: it strips
decoration it can prove is decoration, and it never guesses.
"""

import re
import unicodedata

from memoryos.domain.values import EntityType

# Honorifics and post-nominals, stripped from PERSON only. Deliberately short:
# these are the forms that appear in front of a name and carry no identity, and
# a longer list starts eating words that do ("Chair", "Lead", "Principal" are
# roles, and two different people can hold them).
_TITLES = (
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor",
    "sir", "dame", "lord", "lady", "rev", "fr", "st",
)
_SUFFIXES = ("jr", "sr", "phd", "md", "esq", "ii", "iii", "iv")

# Framework and runtime decoration, stripped from TECHNOLOGY only. `js` covers
# both "React.js" and "ReactJS" once punctuation is gone.
#
# **`sql` is deliberately absent**, and it was in this list until it was tried.
# It turns "PostgreSQL" into "postgre" and "MySQL" into "my" — forms that match
# neither the other spelling nor anything else, so the rule does not merely fail
# to help, it manufactures a canonical form no other name will ever share.
# "Postgres"/"PostgreSQL" is embedding similarity's job, which is what Step 2 is
# for; a character rule cannot reach it without mangling the names it is given.
_TECH_SUFFIXES = ("js", "javascript", "lang", "db")

# Generic nouns that follow a technology's name without being part of it: "the
# React library" and "React" are one thing. Only stripped as a whole trailing
# word, never glued, because "Bookshelf" must not lose its "shelf".
_TECH_NOUNS = ("library", "framework", "package", "module", "tool", "toolkit")

# File extensions, stripped from FILE only. The full name is kept when stripping
# would leave nothing — a file literally called `.gitignore` is its extension.
_FILE_EXTENSIONS = (
    "py", "md", "txt", "json", "yml", "yaml", "toml", "ini", "cfg",
    "js", "ts", "tsx", "jsx", "html", "css", "sh", "sql", "rs", "go",
)

# Leading articles, stripped from every type. "the React library" and "React"
# are the same thing, and the article is never the identity.
_ARTICLES = ("the", "a", "an")

_NOT_ALNUM = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


def canonicalize(name: str, entity_type: EntityType) -> str:
    """The form two names must share to be the same entity by exact match.

    Order matters and is not arbitrary:

    1. **NFKC first.** It folds compatibility characters — full-width Latin
       letters, the "fi" ligature, a non-breaking space — onto their ordinary
       equivalents. Done later, the punctuation strip would already have deleted
       the evidence.
    2. **Case and articles**, which are decoration in every type.
    3. **Type-specific suffixes**, while word boundaries still exist. `.js` is
       findable as a suffix now and indistinguishable from the stem once
       punctuation is gone: "reactjs" and "react" both end in "js" as far as a
       character comparison goes, but only one of them ends in a *suffix*.
    4. **Punctuation last**, collapsing what remains to `[a-z0-9]`.

    Returns the lowercased original when every rule would leave nothing. A name
    that canonicalises to the empty string would collide with every other such
    name, which is the worst possible merge — so an unrepresentable name keeps
    its own identity instead.
    """
    folded = unicodedata.normalize("NFKC", name)
    lowered = _WHITESPACE.sub(" ", folded).strip().lower()
    if not lowered:
        return ""

    words = lowered.split(" ")
    words = _strip_articles(words)
    words = _strip_for_type(words, entity_type)

    stripped = _NOT_ALNUM.sub("", " ".join(words))
    if stripped:
        return stripped

    # Everything was decoration. Fall back to the whole name with punctuation
    # removed, and then to the lowercased original, rather than returning "".
    whole = _NOT_ALNUM.sub("", lowered)
    return whole or lowered


def _strip_articles(words: list[str]) -> list[str]:
    if len(words) > 1 and words[0] in _ARTICLES:
        return words[1:]
    return words


def _strip_for_type(words: list[str], entity_type: EntityType) -> list[str]:
    if entity_type is EntityType.PERSON:
        return _strip_person(words)
    if entity_type is EntityType.TECHNOLOGY:
        return _strip_tech(words)
    if entity_type is EntityType.FILE:
        return _strip_file(words)
    return words


def _strip_person(words: list[str]) -> list[str]:
    """Drop honorifics and post-nominals. "Dr. Chen" and "Chen" are one person.

    Only ever from the ends, and never the last remaining word: somebody
    referred to only as "Dr" keeps that as their name rather than vanishing.
    """
    result = list(words)
    while len(result) > 1 and _bare(result[0]) in _TITLES:
        result = result[1:]
    while len(result) > 1 and _bare(result[-1]) in _SUFFIXES:
        result = result[:-1]
    return result


def _strip_tech(words: list[str]) -> list[str]:
    """Drop framework decoration: "React.js", "ReactJS" and "React" are one.

    Handles the suffix as a separate word ("react js"), as a dotted suffix
    ("react.js"), and as a glued one ("reactjs") — which are three spellings of
    the same claim and arrive from the extractor in all three forms.

    The length guard is what keeps "Preact" out of "React" territory: a stem
    shorter than three characters is not a technology name with decoration
    removed, it is a different word that happened to end in these letters.
    """
    result = list(words)
    # A trailing generic noun, as a whole word only. "the React library" has
    # already lost its article by now, so this is what closes the gap to "React".
    while len(result) > 1 and _bare(result[-1]) in _TECH_NOUNS:
        result = result[:-1]
    if len(result) > 1 and _bare(result[-1]) in _TECH_SUFFIXES:
        result = result[:-1]

    last = _bare(result[-1]) if result else ""
    for suffix in _TECH_SUFFIXES:
        if last.endswith(suffix) and len(last) - len(suffix) >= 3:
            result = [*result[:-1], last[: -len(suffix)]]
            break
    return result


def _strip_file(words: list[str]) -> list[str]:
    """Drop a file extension, keeping the path that identifies the file.

    `memoryos/config.py` and `config.py` are deliberately *not* the same thing
    here: this strips the extension, not the directories. Two files with the
    same basename in different packages are two files, and collapsing them would
    invent a merge nobody can undo by looking at the name.
    """
    if not words:
        return words
    last = words[-1]
    head, _, extension = last.rpartition(".")
    if head and extension.lower() in _FILE_EXTENSIONS:
        return [*words[:-1], head]
    return words


def _bare(word: str) -> str:
    """The word with punctuation removed, for comparing against a suffix list."""
    return _NOT_ALNUM.sub("", word)
