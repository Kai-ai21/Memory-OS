"""Loading the golden set: the two ways it is allowed to lose a query.

Both of these are about the same failure mode — a harness that reports a better
number because it quietly scored less. A query that can only score zero has to
leave loudly, and an answer-key entry that no longer names anything has to be
named rather than dropped.

No database: `build_golden_set` takes the corpus as a mapping, because what is
being tested is a rule about identity rather than a query.
"""

from typing import Any

import pytest

from memoryos.application.golden import CorpusIndex, Item, build_golden_set

CORPUS = CorpusIndex(
    ordinals={
        ("self", "README.md"): frozenset({0, 1, 2}),
        ("self", "src/entities.py"): frozenset({0, 1, 2, 3, 4}),
    }
)


def item(external_key: str, verdict: str, ordinal: int | None = None) -> dict[str, Any]:
    return {
        "source_name": "self",
        "external_key": external_key,
        "chunk_ordinal": ordinal,
        "verdict": verdict,
    }


def payload(*queries: dict[str, Any]) -> dict[str, Any]:
    return {"generated_at": "2026-08-10T00:00:00+00:00", "queries": list(queries)}


def test_a_query_with_no_relevant_judgements_is_excluded_with_a_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """It can only score zero, so leaving it in is a constant drag on every mean."""
    golden = build_golden_set(
        payload(
            {
                "query_text": "nothing here is any good",
                "filters": {},
                "items": [
                    item("README.md", "not_relevant"),
                    item("src/entities.py", "not_relevant"),
                ],
            },
            {
                "query_text": "this one has an answer",
                "filters": {},
                "items": [item("README.md", "relevant")],
            },
        ),
        CORPUS,
    )

    assert [query.query_text for query in golden.queries] == ["this one has an answer"]
    assert [(e.query_text, e.reason) for e in golden.excluded] == [
        ("nothing here is any good", "no relevant or missing judgements")
    ]

    # Asserted on the emitted output rather than through `structlog.capture_logs`,
    # because the module logger is bound once and cached — by the time this test
    # runs, another one has configured structlog and the capture would silently
    # observe nothing. Checking what was actually written is the point anyway:
    # the requirement is that a person running this sees the exclusion.
    written = capsys.readouterr()
    assert "golden.query_excluded" in written.out + written.err


def test_an_unresolvable_triple_is_reported_rather_than_dropped() -> None:
    """A golden set that shrinks quietly as files move reports a rising score."""
    golden = build_golden_set(
        payload(
            {
                "query_text": "why two clocks are recorded",
                "filters": {},
                "items": [
                    # Still there, and still the answer.
                    item("src/entities.py", "relevant", 3),
                    # The file was renamed out from under the judgement.
                    item("src/old_entities.py", "relevant"),
                    # The file is there; that chunk is not.
                    item("README.md", "missing", 9),
                ],
            }
        ),
        CORPUS,
    )

    (query,) = golden.queries
    assert query.relevant == frozenset({Item("self", "src/entities.py", 3)})

    reported = {(u.item.key, u.reason) for u in golden.unresolved}
    assert reported == {
        (
            "self::src/old_entities.py",
            "no current memory with this source and external key",
        ),
        ("self::README.md#9", "chunk 9 does not exist; the memory has 3"),
    }

    # And the surviving pin is what makes "right file, wrong chunk" a miss: the
    # same memory projects to a scoring key that is only in the answer set when
    # the pinned chunk actually matched.
    assert query.project("self", "src/entities.py", [3, 4]) == "self::src/entities.py#3"
    assert query.project("self", "src/entities.py", [0, 1]) == "self::src/entities.py"
