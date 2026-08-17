"""PDF text extraction."""

import io
from typing import Any

import structlog
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from memoryos.application.ports import ParsedDocument, Parser, StructureMarker
from memoryos.domain.jobs import PermanentError
from memoryos.domain.normalization import normalize_text
from memoryos.domain.values import MemoryKind

logger = structlog.get_logger(__name__)

# Below this, across the whole document, the PDF is almost certainly scanned
# images with no text layer. Storing the handful of characters that came out
# would create a memory that retrieves for nothing and looks, to anyone reading
# the table, like the document simply had no content.
MIN_EXTRACTED_CHARS = 50

# Page separator. A form feed is the character that already means "page break"
# and it survives normalization untouched.
PAGE_BREAK = "\f"


# Public, because the upload allow-list in `application/attachments.py` is composed
# from these rather than restating them. One definition of "this parser handles
# it", read by the parser and by the thing that decides what may be uploaded — two
# lists would let a file be accepted at the door and rejected by the pipeline, or
# the reverse, which is worse.
SUFFIXES = (".pdf",)
MEDIA_TYPES = frozenset({"application/pdf"})


class PdfParser(Parser):
    def can_parse(self, media_type: str | None, external_key: str) -> bool:
        return external_key.lower().endswith(SUFFIXES) or media_type in MEDIA_TYPES

    def parse(
        self, data: bytes, *, media_type: str | None, external_key: str
    ) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
        except (PdfReadError, ValueError, OSError) as exc:
            # A malformed PDF will still be malformed on the fifth attempt.
            raise PermanentError(f"cannot read PDF {external_key!r}: {exc}") from exc

        text = normalize_text(PAGE_BREAK.join(pages))

        if len(text.replace(PAGE_BREAK, "").strip()) < MIN_EXTRACTED_CHARS:
            # OCR is out of scope, and retrying cannot add a text layer that is
            # not there. Failing loudly beats storing an empty memory that
            # silently pollutes every future search result set.
            raise PermanentError(
                f"PDF {external_key!r} yielded {len(text)} characters across "
                f"{len(pages)} page(s); it is almost certainly scanned and needs OCR"
            )

        metadata: dict[str, Any] = {"pages": len(pages)}
        info = _document_info(reader)
        if info:
            metadata["pdf_info"] = info

        return ParsedDocument(
            text=text,
            title=info.get("title") or external_key.rsplit("/", 1)[-1],
            metadata=metadata,
            structure=page_markers(pages),
            kind=MemoryKind.DOCUMENT,
        )


def page_markers(pages: list[str]) -> list[StructureMarker]:
    """One marker per page boundary.

    A page break is the only structural signal a text-layer PDF reliably
    offers. It is a weak boundary — a sentence can straddle one — but it is
    better than an arbitrary character offset.
    """
    markers: list[StructureMarker] = []
    offset = 0
    for number, page in enumerate(pages, start=1):
        markers.append(
            StructureMarker(
                kind="heading", level=1, char_offset=offset, label=f"page {number}"
            )
        )
        offset += len(page) + len(PAGE_BREAK)
    return markers


def _document_info(reader: PdfReader) -> dict[str, Any]:
    try:
        info = reader.metadata
    except Exception:
        return {}
    if info is None:
        return {}
    return {
        key.lstrip("/").lower(): str(value)
        for key, value in info.items()
        if value is not None
    }
