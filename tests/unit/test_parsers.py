"""Parsers: one output shape, whatever went in."""

import pytest

from memoryos.adapters.parsers.code import CodeParser
from memoryos.adapters.parsers.markdown import MarkdownParser
from memoryos.adapters.parsers.registry import build_default_registry
from memoryos.adapters.parsers.text import TextParser, decode
from memoryos.application.ports import ParsedDocument
from memoryos.domain.values import MemoryKind


def parse_md(source: str) -> ParsedDocument:
    return MarkdownParser().parse(
        source.encode(), media_type="text/markdown", external_key="notes/doc.md"
    )


def parse_py(source: str) -> ParsedDocument:
    return CodeParser().parse(
        source.encode(), media_type="text/x-python", external_key="mod.py"
    )


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("external_key", "expected"),
    [
        ("notes/a.md", "MarkdownParser"),
        ("notes/a.markdown", "MarkdownParser"),
        ("src/mod.py", "CodeParser"),
        ("src/app.ts", "CodeParser"),
        ("doc.pdf", "PdfParser"),
        ("notes/a.txt", "TextParser"),
        ("some.unknown-extension", "TextParser"),
    ],
)
def test_the_registry_picks_the_first_matching_parser(
    external_key: str, expected: str
) -> None:
    parser = build_default_registry().for_item(None, external_key)
    assert type(parser).__name__ == expected


def test_the_text_parser_accepts_anything() -> None:
    # It is registered last for exactly this reason: anything after it would
    # never be reached.
    assert TextParser().can_parse(None, "whatever.xyz") is True


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def test_utf8_is_decoded_as_utf8() -> None:
    text, codec = decode("héllo".encode())
    assert text == "héllo"
    assert codec == "utf-8"


def test_a_bom_is_handled_and_recorded() -> None:
    text, codec = decode("hello".encode("utf-8-sig"))
    assert text.lstrip("﻿") == "hello"
    assert codec in {"utf-8", "utf-8-sig"}


def test_invalid_utf8_falls_back_without_raising() -> None:
    # One mis-encoded file must not make itself unsearchable, but the codec is
    # recorded so that mojibake later has an explanation.
    data = b"caf\xe9 latte"  # latin-1 'é'
    document = TextParser().parse(data, media_type=None, external_key="notes/a.txt")

    assert "caf" in document.text
    assert document.metadata["codec"] == "latin-1"
    assert document.metadata["decode_fallback"] is True


def test_the_text_parser_claims_no_structure() -> None:
    document = TextParser().parse(b"one\ntwo", media_type=None, external_key="a.txt")
    assert document.structure == []
    assert document.kind is MemoryKind.NOTE


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def test_front_matter_is_extracted_and_removed_from_the_text() -> None:
    document = parse_md("---\ntitle: My Note\ntags: [a, b]\n---\n\n# Heading\n\nBody.\n")

    assert document.title == "My Note"
    assert document.metadata["front_matter"]["tags"] == ["a", "b"]
    assert "---" not in document.text
    assert document.text.startswith("# Heading")


def test_the_title_falls_back_to_the_first_h1() -> None:
    document = parse_md("# The Heading\n\nBody.\n")
    assert document.title == "The Heading"


def test_front_matter_title_wins_over_the_h1() -> None:
    document = parse_md("---\ntitle: Front Matter Wins\n---\n\n# Other\n\nBody.\n")
    assert document.title == "Front Matter Wins"


def test_a_document_with_no_headings_has_no_title() -> None:
    assert parse_md("Just prose.\n").title is None


def test_heading_markers_carry_level_and_offset() -> None:
    source = "# One\n\ntext\n\n## Two\n\nmore\n\n### Three\n"
    document = parse_md(source)

    assert [(m.level, m.label) for m in document.structure] == [
        (1, "One"),
        (2, "Two"),
        (3, "Three"),
    ]
    # The offsets must actually point at the headings, because the chunker
    # slices the text at them.
    for marker in document.structure:
        assert document.text[marker.char_offset :].startswith("#" * marker.level + " ")


def test_a_hash_inside_a_code_fence_is_not_a_heading() -> None:
    # A shell comment is not a section boundary.
    document = parse_md("# Real\n\n```sh\n# not a heading\necho hi\n```\n\n## Also real\n")

    assert [marker.label for marker in document.structure] == ["Real", "Also real"]


def test_markdown_source_is_preserved() -> None:
    # Heading syntax and code fences tell an embedding model what kind of text
    # it is looking at; stripping them to plain prose throws that away.
    document = parse_md("# Title\n\n- a list item\n\n```py\nx = 1\n```\n")

    assert "# Title" in document.text
    assert "- a list item" in document.text
    assert "```py" in document.text


def test_non_json_front_matter_values_are_coerced() -> None:
    # The column is JSONB, so a YAML date has to become something storable
    # rather than being dropped.
    document = parse_md("---\ndate: 2024-01-15\n---\n\nBody.\n")
    assert document.metadata["front_matter"]["date"] == "2024-01-15"


# --------------------------------------------------------------------------
# Code
# --------------------------------------------------------------------------


def test_one_marker_per_top_level_python_definition() -> None:
    source = "import os\n\n\ndef first():\n    pass\n\n\nclass Second:\n    pass\n"
    document = parse_py(source)

    assert [marker.label for marker in document.structure] == ["first", "Second"]
    assert document.kind is MemoryKind.CODE
    for marker in document.structure:
        assert marker.kind == "definition"


def test_python_definition_offsets_point_at_the_definition() -> None:
    source = "import os\n\n\ndef first():\n    pass\n\n\nclass Second:\n    pass\n"
    document = parse_py(source)

    starts = [document.text[m.char_offset :].split("\n", 1)[0] for m in document.structure]
    assert starts == ["def first():", "class Second:"]


def test_nested_functions_are_not_marked() -> None:
    # A nested helper belongs to the function containing it. Marking it would
    # invite the chunker to cut a function at the one place it must not.
    source = "def outer():\n    def inner():\n        pass\n    return inner\n"
    document = parse_py(source)

    assert [marker.label for marker in document.structure] == ["outer"]


def test_methods_are_not_marked_separately() -> None:
    source = "class Thing:\n    def method(self):\n        pass\n"
    document = parse_py(source)
    assert [marker.label for marker in document.structure] == ["Thing"]


def test_a_decorated_definition_starts_above_its_decorator() -> None:
    source = "@decorator\ndef decorated():\n    pass\n"
    document = parse_py(source)

    (marker,) = document.structure
    assert document.text[marker.char_offset :].startswith("@decorator")


def test_unparseable_python_falls_back_to_the_regex() -> None:
    # A syntax error is a reason to chunk a file worse, not to refuse it.
    source = "def broken(:\n    pass\n\ndef other():\n    pass\n"
    document = parse_py(source)
    assert [marker.label for marker in document.structure] == ["broken", "other"]


def test_other_languages_use_the_regex() -> None:
    source = "export function alpha() {}\n\nclass Beta {}\n"
    document = CodeParser().parse(
        source.encode(), media_type=None, external_key="app.ts"
    )
    assert [marker.label for marker in document.structure] == ["alpha", "Beta"]


def test_indented_definitions_are_not_marked() -> None:
    # The outer class is a boundary; the method inside it is not, for the same
    # reason a nested Python function is not.
    source = "class Outer {\n    function inner() {}\n}\n"
    document = CodeParser().parse(
        source.encode(), media_type=None, external_key="app.js"
    )
    assert [marker.label for marker in document.structure] == ["Outer"]


def test_code_text_is_the_source_verbatim() -> None:
    source = "def f():\n    return 1\n"
    assert parse_py(source).text == "def f():\n    return 1"
