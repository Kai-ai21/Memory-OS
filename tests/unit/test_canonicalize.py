"""Canonicalization, which is decidable without a database.

The first of M3.2's four required tests. It is a unit test because
`canonicalize` is a pure function, and because the interesting cases are the
ones where two names must *not* collapse — a false merge invents a path the
corpus does not contain, and every traversal through it reports a connection
nobody wrote.
"""

import pytest

from memoryos.domain.canonicalize import canonicalize
from memoryos.domain.values import EntityType


@pytest.mark.parametrize(
    "name",
    ["React", "React.js", "ReactJS", "react", "REACT", "the React library", " react "],
)
def test_the_spellings_of_react_share_one_canonical_form(name: str) -> None:
    """The milestone's own example, exactly.

    Four spellings of one library are four nodes in an unresolved graph, and the
    path a traversal needs runs through whichever one it did not pick.
    """
    assert canonicalize(name, EntityType.TECHNOLOGY) == "react"


def test_react_and_preact_stay_apart() -> None:
    """The case that matters more than the one above.

    A missed merge leaves a traversal short a path. A false merge invents one.
    "Preact" ends in the same five letters as "React" and is a different
    library, and any rule loose enough to join them is loose enough to join most
    of the corpus.
    """
    assert canonicalize("React", EntityType.TECHNOLOGY) != canonicalize(
        "Preact", EntityType.TECHNOLOGY
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # `sql` is deliberately not a stripped suffix: stripping it turns
        # "PostgreSQL" into "postgre" and "MySQL" into "my", forms that match
        # nothing at all. Embedding similarity is what reaches this pair.
        ("PostgreSQL", "postgresql"),
        ("MySQL", "mysql"),
        ("SQLite", "sqlite"),
        # Real decoration, correctly removed.
        ("Node.js", "node"),
        ("Vue.js", "vue"),
    ],
)
def test_technology_suffixes_strip_decoration_and_not_stems(
    name: str, expected: str
) -> None:
    assert canonicalize(name, EntityType.TECHNOLOGY) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Dr. Chen", "chen"),
        ("Chen", "chen"),
        ("Prof Ada Lovelace", "adalovelace"),
        ("Ada Lovelace Jr.", "adalovelace"),
        # A title is only decoration when something follows it. Somebody
        # recorded only as "Dr" keeps that rather than vanishing to the empty
        # string, which would collide with every other unnameable entity.
        ("Dr", "dr"),
    ],
)
def test_person_titles_are_stripped_but_never_the_whole_name(
    name: str, expected: str
) -> None:
    assert canonicalize(name, EntityType.PERSON) == expected


def test_the_same_rule_is_not_applied_across_types() -> None:
    """Type-specific, because the rules contradict each other.

    Stripping a trailing `js` is right for a TECHNOLOGY and destroys a FILE
    called `index.js` — which keeps its identity as `index` only because the
    FILE rule removes a known extension, not any two letters that look like one.
    """
    assert canonicalize("index.js", EntityType.FILE) == "index"
    # A PERSON named "Js" is not decoration.
    assert canonicalize("Js", EntityType.PERSON) == "js"


def test_paths_are_not_collapsed_to_basenames() -> None:
    """Two files with one basename in different packages are two files.

    Collapsing them would invent a merge that cannot be undone by reading the
    name, which is the only evidence a reviewer has.
    """
    assert canonicalize("memoryos/config.py", EntityType.FILE) != canonicalize(
        "web/config.py", EntityType.FILE
    )


def test_unicode_is_folded_before_punctuation_is_stripped() -> None:
    """NFKC first, or the strip deletes the evidence it would have folded.

    Written as escapes rather than literal full-width characters: they are
    indistinguishable from ASCII in most editors, which is exactly why they
    reach a corpus unnoticed and exactly why they would be unreadable here.
    """
    fullwidth_react = "\uff32\uff25\uff21\uff23\uff34"
    assert canonicalize(fullwidth_react, EntityType.TECHNOLOGY) == "react"


def test_a_name_of_pure_punctuation_keeps_its_own_identity() -> None:
    """The empty string is the worst possible canonical form.

    Every unrepresentable name would share it, so a single bucket would merge
    entities that have nothing in common at all.
    """
    assert canonicalize("...", EntityType.CONCEPT) != ""
    assert canonicalize("???", EntityType.CONCEPT) != canonicalize(
        "...", EntityType.CONCEPT
    )
