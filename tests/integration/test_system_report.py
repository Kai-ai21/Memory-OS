"""What `report --full` counts, and the one thing it must never do.

The report is the artifact somebody is shown, which makes it the place where an
optimistic number would do the most damage. The properties worth a test are not
the arithmetic — a `COUNT(*)` is right or the database is broken — but the two
judgements the gatherer makes on top of it:

* **an empty capability is reported as empty**, with the reason, rather than
  omitted so the page looks complete;
* **extraction coverage counts memories, not mentions**, because one memory with
  forty mentions is one memory and the difference is exactly the number M8.0's
  `workflows` dimension declines on.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from memoryos.application import system_report, user_model
from memoryos.application.decisions import (
    AssumptionInput,
    DecisionDraft,
    OptionInput,
    record,
)
from memoryos.domain.values import Dimension, TimeProvenance
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration


async def test_an_ingested_corpus_with_no_behaviour_reports_the_reasons(
    harness: Harness,
) -> None:
    """**A report that omitted its empty sections would describe a different
    system.**

    The harness ingests and embeds a small corpus and does nothing else — no
    extraction, no decisions, no patterns — which is the shape a real deployment
    has for most of its life. Everything downstream of ingestion comes back
    empty, with the reason, rather than being left off a page that would then
    look complete.
    """
    assessments = await user_model.assess(harness.sessions)
    stability = await user_model.stability(harness.sessions)
    corpus, decisions, behaviour, ran_at = await system_report.gather(
        harness.sessions, assessments=assessments, stability=stability
    )

    assert ran_at <= datetime.now(UTC)
    # Ingestion ran, so these are the parts that work.
    assert corpus.memories == corpus.current_memories > 0
    assert corpus.coverage == 1.0
    # Extraction did not, and the coverage says so rather than dividing by zero
    # or reporting the mentions table's emptiness as full coverage.
    assert corpus.extracted_memories == 0
    assert corpus.extraction_coverage == 0.0
    assert decisions.decisions == 0
    assert behaviour.patterns == 0
    assert behaviour.facets == 0

    # All seven, with a stated reason each.
    assert len(behaviour.assessments) == len(Dimension)
    assert len(behaviour.stability) == len(Dimension)
    for assessment in behaviour.assessments:
        assert not assessment.has_evidence
        assert "insufficient evidence" in assessment.render()
    for entry in behaviour.stability:
        assert entry.verdict() == "no facets: nothing to measure"


async def test_the_counts_follow_the_data_that_was_actually_recorded(
    harness: Harness,
) -> None:
    """The other side, so the first test is a report rather than one that always
    says zero."""
    await record(
        harness.sessions,
        DecisionDraft(
            question="Postgres or a dedicated vector store?",
            chosen="Postgres with pgvector",
            options=(OptionInput(description="a dedicated vector store"),),
            assumptions=(
                AssumptionInput(statement="the index stays small enough to rebuild"),
                AssumptionInput(statement="one datastore is easier to operate"),
            ),
        ),
        decided_at=datetime(2026, 5, 2, 9, 0, tzinfo=UTC),
        decided_at_source=TimeProvenance.DECLARED,
    )

    assessments = await user_model.assess(harness.sessions)
    stability = await user_model.stability(harness.sessions)
    _, decisions, _, _ = await system_report.gather(
        harness.sessions, assessments=assessments, stability=stability
    )

    assert decisions.decisions == 1
    assert decisions.options == 2
    assert decisions.assumptions == 2
    # Nobody has evaluated or grouped them, and both are reported as the zero
    # they are. These two numbers are the ones M8.1 found explain why three of
    # its four detectors can never fire on this corpus.
    assert decisions.evaluated_assumptions == 0
    assert decisions.grouped_assumptions == 0
    assert decisions.groups == 0


def test_every_named_baseline_is_reported_even_when_the_file_is_absent(
    tmp_path: Path,
) -> None:
    """**A missing baseline is a row, not a shorter list.**

    The set of configurations this system has been measured against is itself a
    claim the report makes. Discovering baselines by globbing would let a
    deleted file quietly become a configuration that had never been measured,
    which is the more flattering of the two mistakes.
    """
    loaded = system_report.read_baselines(tmp_path)

    assert [name for name, _, _ in loaded] == [name for name, _ in system_report.BASELINES]
    assert all(payload is None for _, _, payload in loaded)
    assert system_report.agent_baseline(tmp_path) is None
