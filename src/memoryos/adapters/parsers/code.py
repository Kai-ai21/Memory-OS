"""Source code: text verbatim, with a marker per top-level definition."""

import ast
import re

from memoryos.adapters.parsers.text import decode
from memoryos.application.ports import ParsedDocument, Parser, StructureMarker
from memoryos.domain.normalization import normalize_text
from memoryos.domain.values import MemoryKind

_SUFFIXES = (
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".rb",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".cs",
    ".swift",
    ".php",
    ".sh",
    ".sql",
)

# Deliberately not a parser. A regex that finds most top-level definitions in
# most languages is enough to give the chunker boundaries; tree-sitter would be
# a per-language grammar dependency for a marginal gain over this.
_DEFINITION = re.compile(
    r"^(?:export\s+|public\s+|private\s+|protected\s+|static\s+|async\s+)*"
    r"(?P<keyword>def|class|function|func|fn|interface|type|struct|impl)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


class CodeParser(Parser):
    def can_parse(self, media_type: str | None, external_key: str) -> bool:
        return external_key.lower().endswith(_SUFFIXES)

    def parse(
        self, data: bytes, *, media_type: str | None, external_key: str
    ) -> ParsedDocument:
        raw, codec = decode(data)
        text = normalize_text(raw)

        if external_key.lower().endswith(".py"):
            structure = python_definitions(text)
        else:
            structure = regex_definitions(text)

        return ParsedDocument(
            text=text,
            title=external_key.rsplit("/", 1)[-1],
            metadata={"codec": codec, "language": _language_of(external_key)},
            structure=structure,
            kind=MemoryKind.CODE,
        )


def python_definitions(text: str) -> list[StructureMarker]:
    """Top-level defs and classes, via `ast`.

    Only top-level: a nested helper is part of the function that contains it,
    and marking it would invite the chunker to cut a function in half at the
    one place it must not.

    Falls back to the regex when the file does not parse — a syntax error is a
    reason to chunk a file worse, not a reason to refuse it.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return regex_definitions(text)

    lines = _line_offsets(text)
    markers: list[StructureMarker] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        # Decorators are part of the definition, so the boundary goes above
        # them rather than between a decorator and the thing it decorates.
        first_line = min(
            [node.lineno, *[decorator.lineno for decorator in node.decorator_list]]
        )
        markers.append(
            StructureMarker(
                kind="definition",
                level=0,
                char_offset=lines[first_line - 1],
                label=node.name,
            )
        )
    return markers


def regex_definitions(text: str) -> list[StructureMarker]:
    """Definitions that start at column zero, for languages without a parser here."""
    markers: list[StructureMarker] = []
    for match in _DEFINITION.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        # Column zero only: an indented `def` is nested, and nesting is not a
        # boundary for the same reason it is not in the Python case.
        if match.start() != line_start:
            continue
        markers.append(
            StructureMarker(
                kind="definition",
                level=0,
                char_offset=match.start(),
                label=match.group("name"),
            )
        )
    return markers


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    for index, char in enumerate(text):
        if char == "\n":
            offsets.append(index + 1)
    return offsets


def _language_of(external_key: str) -> str:
    suffix = external_key.rsplit(".", 1)[-1].lower()
    return suffix if f".{suffix}" in _SUFFIXES else "unknown"
