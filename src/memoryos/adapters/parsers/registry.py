"""Parser selection: first match wins, so registration order is the priority."""

from memoryos.adapters.parsers.code import CodeParser
from memoryos.adapters.parsers.markdown import MarkdownParser
from memoryos.adapters.parsers.pdf import PdfParser
from memoryos.adapters.parsers.text import TextParser
from memoryos.application.ports import ParsedDocument, Parser


class ParserRegistry:
    def __init__(self, parsers: list[Parser]) -> None:
        self._parsers = parsers

    def for_item(self, media_type: str | None, external_key: str) -> Parser:
        for parser in self._parsers:
            if parser.can_parse(media_type, external_key):
                return parser
        # Unreachable while TextParser is registered: it accepts everything.
        raise LookupError(f"no parser for {external_key!r}")

    def parse(
        self, data: bytes, *, media_type: str | None, external_key: str
    ) -> ParsedDocument:
        parser = self.for_item(media_type, external_key)
        return parser.parse(data, media_type=media_type, external_key=external_key)


def build_default_registry() -> ParserRegistry:
    """Specific parsers first, the catch-all last.

    TextParser accepts anything, so anything registered after it would never be
    reached. That ordering is the whole selection algorithm.
    """
    return ParserRegistry([MarkdownParser(), CodeParser(), PdfParser(), TextParser()])
