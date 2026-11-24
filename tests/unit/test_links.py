"""`[[wiki links]]` as entities: what counts as a link, and where it is (M12.2)."""

from collections.abc import Sequence

import pytest

from memoryos.adapters.extraction.links import LinkedExtractor, links_in
from memoryos.application.ports import EntityRef, ExtractedEntity, ExtractedRelationship
from memoryos.domain.values import EntityType, MemoryKind


def names(text: str) -> list[str]:
    return [entity.name for entity in links_in(text)]


def test_every_form_of_link_yields_its_target() -> None:
    text = "See [[Harbour]], [[Ada Lovelace|Ada]], [[Roadmap#Q3]] and ![[diagram]]."
    assert names(text) == ["Harbour", "Ada Lovelace", "Roadmap", "diagram"]


def test_offsets_slice_to_exactly_the_name() -> None:
    """The invariant `ExtractEntities` re-checks before storing anything."""
    text = "Talked to [[ Ada Lovelace |Ada]] about it."
    for entity in links_in(text):
        assert text[entity.char_start : entity.char_end] == entity.name
    assert names(text) == ["Ada Lovelace"]


def test_a_list_of_lists_is_not_a_link() -> None:
    assert names("matrix = [[1, 2], [3, 4]]") == []


def test_links_inside_a_code_fence_are_quoted_not_meant() -> None:
    text = "Real: [[Harbour]]\n\n```\nfake = [[NotALink]]\n```\n"
    assert names(text) == ["Harbour"]


def test_a_link_says_the_name_and_not_the_type() -> None:
    (entity,) = links_in("[[Harbour]]")
    assert entity.type_is_guess is True


class Fixed:
    """A primary extractor that always finds the same entities."""

    version = "fixed@1"

    def __init__(self, found: list[ExtractedEntity]) -> None:
        self._found = found

    async def extract(self, text: str, *, kind: MemoryKind) -> list[ExtractedEntity]:
        return list(self._found)

    async def extract_relationships(
        self, text: str, entities: Sequence[EntityRef]
    ) -> list[ExtractedRelationship]:
        return []


async def test_links_are_added_to_notes_only() -> None:
    combined = LinkedExtractor(None)
    assert await combined.extract("[[Harbour]]", kind=MemoryKind.CODE) == []
    assert [e.name for e in await combined.extract("[[Harbour]]", kind=MemoryKind.NOTE)] == [
        "Harbour"
    ]


async def test_a_link_takes_the_type_the_model_gave_the_same_name() -> None:
    text = "Harbour is a project. See [[Harbour]]."
    model = Fixed([ExtractedEntity("Harbour", EntityType.PROJECT, 0.9, 0, 7)])
    found = await LinkedExtractor(model).extract(text, kind=MemoryKind.NOTE)

    link = next(entity for entity in found if entity.char_start > 7)
    assert link.type is EntityType.PROJECT
    assert link.type_is_guess is False


async def test_a_span_the_model_already_claimed_is_not_counted_twice() -> None:
    text = "See [[Harbour]]."
    model = Fixed([ExtractedEntity("Harbour", EntityType.PROJECT, 0.9, 6, 13)])
    found = await LinkedExtractor(model).extract(text, kind=MemoryKind.NOTE)
    assert len(found) == 1


@pytest.mark.parametrize(
    ("primary", "expected"),
    [(None, "links-v1"), (Fixed([]), "fixed@1+links-v1")],
)
def test_the_version_names_both_halves(primary: Fixed | None, expected: str) -> None:
    assert LinkedExtractor(primary).version == expected
