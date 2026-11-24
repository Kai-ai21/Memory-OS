"""How a note is dated, without a database (M12.2)."""

from datetime import UTC, datetime

import pytest

from memoryos.adapters.connectors.vault import date_from_front_matter, date_from_name


@pytest.mark.parametrize(
    ("front_matter", "expected"),
    [
        ("date: 2024-03-15", datetime(2024, 3, 15, tzinfo=UTC)),
        ("created: 2024-03-15T10:20", datetime(2024, 3, 15, 10, 20, tzinfo=UTC)),
        ("created: '2024-03-15 10:20'", datetime(2024, 3, 15, 10, 20, tzinfo=UTC)),
        ("updated: 2024-03-15T10:20:00+05:30", datetime(2024, 3, 15, 4, 50, tzinfo=UTC)),
        # Most specific wins, whatever order the keys are written in.
        ("updated: 2025-01-01\ndate: 2024-03-15", datetime(2024, 3, 15, tzinfo=UTC)),
        # A date that is not a date falls through to the next key.
        ("date: someday\ncreated: 2024-03-15", datetime(2024, 3, 15, tzinfo=UTC)),
        ("date: 0001-01-01", None),
        ("title: no date at all", None),
    ],
)
def test_front_matter(front_matter: str, expected: datetime | None) -> None:
    parsed = date_from_front_matter(f"---\n{front_matter}\n---\nbody\n")
    assert parsed == expected


def test_text_without_front_matter_has_no_declared_date() -> None:
    assert date_from_front_matter("date: 2024-03-15\n") is None


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("journals/2024_02_07.md", datetime(2024, 2, 7, tzinfo=UTC)),
        ("daily/2024-02-07.md", datetime(2024, 2, 7, tzinfo=UTC)),
        ("2024-02-07 standup.md", datetime(2024, 2, 7, tzinfo=UTC)),
        ("2024-13-40.md", None),
        ("202402071030 zettel.md", None),
        ("notes/ideas.md", None),
    ],
)
def test_daily_note_names(key: str, expected: datetime | None) -> None:
    assert date_from_name(key) == expected


def test_malformed_front_matter_does_not_fail_the_note() -> None:
    """Until M12.2 the parser raised, and the note was never chunked."""
    from memoryos.adapters.parsers.markdown import MarkdownParser

    parsed = MarkdownParser().parse(
        b"---\ndate: [unclosed\n---\nThe body survives.\n",
        media_type="text/markdown",
        external_key="broken.md",
    )
    assert "The body survives." in parsed.text
    assert parsed.metadata.get("front_matter_error") is True
