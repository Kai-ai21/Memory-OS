"""Normalization, and the property the whole milestone rests on."""

import unicodedata

import pytest

from memoryos.domain.normalization import normalize_text, normalized_hash


def test_crlf_becomes_lf() -> None:
    assert normalize_text("a\r\nb\r\nc") == "a\nb\nc"


def test_lone_cr_becomes_lf() -> None:
    # Classic Mac line endings. Rare, but a file that has them is not a
    # different document from the same file without them.
    assert normalize_text("a\rb\rc") == "a\nb\nc"


def test_mixed_line_endings_all_become_lf() -> None:
    assert normalize_text("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_text_is_nfc_normalized() -> None:
    decomposed = "café"  # e + combining acute
    composed = "café"  # precomposed é

    assert decomposed != composed
    assert normalize_text(decomposed) == composed
    assert unicodedata.is_normalized("NFC", normalize_text(decomposed))


def test_trailing_whitespace_is_stripped_per_line() -> None:
    assert normalize_text("a   \nb\t\nc  ") == "a\nb\nc"


def test_leading_indentation_survives() -> None:
    # Only *trailing* whitespace is noise. Indentation is content, and code
    # would be destroyed by stripping it.
    assert normalize_text("def f():\n    return 1\n") == "def f():\n    return 1"


@pytest.mark.parametrize("blanks", [3, 4, 10])
def test_runs_of_blank_lines_collapse_to_two(blanks: int) -> None:
    text = "a" + "\n" * (blanks + 1) + "b"
    assert normalize_text(text) == "a\n\nb"


def test_two_blank_lines_are_left_alone() -> None:
    # A paragraph break is meaningful in every convention worth honouring.
    assert normalize_text("a\n\nb") == "a\n\nb"


def test_a_leading_bom_is_stripped() -> None:
    assert normalize_text("﻿hello") == "hello"


def test_a_bom_elsewhere_is_kept() -> None:
    # Mid-document, the same code point is a zero-width no-break space that
    # somebody may have meant.
    assert normalize_text("a﻿b") == "a﻿b"


def test_leading_and_trailing_blank_lines_go() -> None:
    assert normalize_text("\n\n\nhello\n\n\n") == "hello"


def test_empty_input_stays_empty() -> None:
    assert normalize_text("") == ""
    assert normalize_text("\n\n\n") == ""


# --------------------------------------------------------------------------
# The central property
# --------------------------------------------------------------------------


def test_line_endings_alone_produce_the_same_normalized_hash() -> None:
    """The milestone's central property.

    The same document saved on Windows and on Linux is genuinely different
    bytes, so it is genuinely a new artifact and a new memory version. Its
    normalized text is identical, so chunking — and, in M1.5, embedding — has
    nothing to do.
    """
    unix = "# Title\n\nA paragraph.\n\nAnother.\n"
    windows = "# Title\r\n\r\nA paragraph.\r\n\r\nAnother.\r\n"

    assert unix.encode() != windows.encode()
    assert normalized_hash(normalize_text(unix)) == normalized_hash(normalize_text(windows))


def test_cosmetic_differences_collapse_to_one_hash() -> None:
    variants = [
        "# Title\n\nBody text.\n",
        "# Title\r\n\r\nBody text.\r\n",
        "﻿# Title\n\nBody text.\n",
        "# Title   \n\n\n\nBody text.  \n\n",
        "# Title\r\r\rBody text.\r",
    ]

    hashes = {normalized_hash(normalize_text(variant)).value for variant in variants}
    assert len(hashes) == 1


def test_a_real_edit_changes_the_hash() -> None:
    # The flip side: the hash must not be so forgiving that it misses content.
    first = normalized_hash(normalize_text("# Title\n\nBody text.\n"))
    second = normalized_hash(normalize_text("# Title\n\nBody text edited.\n"))
    assert first != second


def test_normalization_is_idempotent() -> None:
    text = "﻿# T\r\n\r\n\r\n\r\nbody   \r\n"
    once = normalize_text(text)
    assert normalize_text(once) == once
