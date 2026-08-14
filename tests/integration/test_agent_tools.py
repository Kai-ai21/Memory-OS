"""The four properties M7.0 rests on, and one the milestone did not ask for.

* an unknown tool name is a `PermanentError`,
* invalid arguments come back as something a model can correct, not an exception,
* every tool result carries citations,
* a result over the cap says it was truncated.

The fifth is `test_every_schema_survives_both_providers`, which is not in the
list because the list assumes the schema travels. It does not: one provider
rejects the key that makes a schema strict, and that was found by sending one.

No model is called anywhere in this file. The registry, the tools and the
validation are all reachable without one, and a test that needed a network round
trip to assert that an unknown name raises would be a test nobody runs.
"""

import dataclasses
import re
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.llm.gemini import _gemini_schema
from memoryos.application.agent.library import (
    MAX_MEMORIES,
    FindGapsTool,
    GetDecisionsTool,
    GetMemoryTool,
    QueryTimelineTool,
    SearchMemoriesTool,
    TraverseGraphTool,
    build_registry,
)
from memoryos.application.agent.tools import (
    InvalidArguments,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    UnknownTool,
    spec_for,
    validate,
)
from memoryos.application.decisions import DecisionDraft, EvidenceInput, OptionInput
from memoryos.application.decisions import record as record_decision
from memoryos.application.search import FusionWeights, SearchMemories
from memoryos.domain.jobs import PermanentError
from memoryos.domain.values import EvidenceRelation, Period, TimeProvenance
from tests.integration.conftest import Harness

pytestmark = pytest.mark.integration

DECIDED_AT = datetime(2026, 5, 2, 9, 0, tzinfo=UTC)


def tools(harness: Harness) -> ToolRegistry:
    """The real six, wired the way the container wires them.

    `expand=None` is not an option here — the graph tool needs one — so it gets
    a real `ExpandThroughGraph` over whatever Neo4j the developer has. It
    degrades to an empty ranking when there is none, which M3.5 designed for and
    which is the behaviour the tool reports rather than fails on.
    """
    from memoryos.application.graph_expand import ExpandThroughGraph
    from tests.integration.conftest import Harness as _H  # noqa: F401

    search = SearchMemories(
        harness.sessions,
        harness.embedder,
        PgVectorStore(harness.sessions, harness.embedder, default_ef_search=100),
        PostgresKeywordStore(harness.sessions),
    )
    registry = ToolRegistry()
    for tool in build_registry(
        sessions=harness.sessions,
        search=search,
        expand=ExpandThroughGraph(harness.sessions, _NoGraph()),
        weights=FusionWeights(),
    ):
        registry.register(tool)
    return registry


class _NoGraph:
    """A graph store that is not there.

    M3.5's expansion catches everything and returns an empty ranking, so this
    asserts the tool's *degraded* path — which on this corpus is also its real
    one, since entity extraction has never been run over it.
    """

    async def neighbours(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("no graph in this test")

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


# --------------------------------------------------------------------------
# 1. An unknown name is permanent
# --------------------------------------------------------------------------


async def test_an_unknown_tool_name_is_permanent(harness: Harness) -> None:
    """**Retrying cannot make a tool appear.**

    The registry is fixed at startup, so the second attempt reads the same
    dictionary as the first. A `TransientError` here would spend the worker's
    five attempts on a name that will never resolve and dead-letter it with the
    message it had at the first.
    """
    registry = tools(harness)

    with pytest.raises(UnknownTool) as raised:
        await registry.call("summarise_everything", {})

    # The classification, not just the type: the worker branches on this.
    assert isinstance(raised.value, PermanentError)
    # And the message names what does exist, because the caller is usually one
    # word away from a tool that would have worked.
    assert "search_memories" in str(raised.value)


# --------------------------------------------------------------------------
# 2. Bad arguments are correctable, not fatal
# --------------------------------------------------------------------------


async def test_invalid_arguments_come_back_as_something_correctable(
    harness: Harness,
) -> None:
    """A model that passed a string where an integer belongs has made a mistake
    it can fix. What it needs is a sentence naming the field; what it cannot use
    is a traceback."""
    registry = tools(harness)

    result = await registry.call(
        "search_memories", {"query": "leases", "limit": "as many as possible"}
    )

    assert isinstance(result, ToolResult)
    assert "limit" in result.content
    assert "integer" in result.content
    # It says what to do next, which is the difference between an error and a
    # correction.
    assert "again" in result.content.lower()
    # And it is a result rather than a raise, so the turn survives it.
    assert result.citations == []


async def test_an_argument_no_tool_declared_is_refused(harness: Harness) -> None:
    """`additionalProperties: false` is what makes this an error rather than a
    silently dropped value — and a dropped argument is how a model comes to
    believe it filtered something it did not."""
    registry = tools(harness)

    result = await registry.call(
        "search_memories", {"query": "leases", "sort_by": "date"}
    )

    assert "sort_by" in result.content


async def test_two_bad_arguments_are_reported_together(harness: Harness) -> None:
    """Otherwise the model fixes one, calls again, and learns about the other —
    two turns and two tool calls for what one sentence could say."""
    registry = tools(harness)

    result = await registry.call("query_timeline", {"start": 5, "end": 9})

    assert "start" in result.content
    assert "end" in result.content


def test_validate_raises_for_callers_that_want_the_exception() -> None:
    """The registry converts it; the type still exists so a caller can tell "the
    model got it wrong" from "the tool broke"."""

    class Args(BaseModel):
        count: int

    with pytest.raises(InvalidArguments) as raised:
        validate(Args, "example", {"count": "many"})

    assert raised.value.tool == "example"
    assert "count" in raised.value.problems


# --------------------------------------------------------------------------
# 3. Every result is attributable
# --------------------------------------------------------------------------


async def test_every_tool_result_carries_citations(harness: Harness) -> None:
    """**A tool result a model can read but not attribute is how the
    no-fabrication guardrail dies quietly.**

    Every tool that returned corpus-derived content is checked, and each is
    called with arguments the fixture corpus can actually satisfy. Two are
    exercised elsewhere and named here rather than skipped silently:
    `traverse_graph`, which returns nothing without extracted entities, and
    `find_gaps`, which needs a corpus spanning months. Both have their own tests
    below asserting that they say why they are empty.
    """
    await harness.ingest()
    registry = tools(harness)

    search = await registry.call("search_memories", {"query": "fox"})
    assert search.citations, "search returned content with nothing to check it against"
    assert all(citation.excerpt for citation in search.citations)
    # A citation has to resolve, not merely exist: this is the identity
    # `verify-citations` asserts corpus-wide, checked here for a tool result.
    memory_id = search.citations[0].memory_id
    detail = await registry.call("get_memory", {"memory_id": str(memory_id)})
    assert detail.citations
    for citation in detail.citations:
        assert citation.source_name
        assert citation.external_key
        assert citation.version >= 1

    # A window wide enough to hold the fixture tree, whose dates come from the
    # filesystem rather than from a fixed constant.
    timeline = await registry.call(
        "query_timeline", {"start": "2000-01-01", "end": "2100-01-01", "period": "month"}
    )
    assert timeline.citations, "a timeline with memories in it cited none of them"


async def test_decisions_cite_their_evidence_and_say_when_there_is_none(
    harness: Harness,
) -> None:
    """**A decision is not in the corpus, so it cannot be cited like a passage.**

    It is a row somebody wrote: no chunk, no offsets, no version. Inventing a
    span for it would produce a citation that looks checkable and is not, so the
    tool cites the corpus evidence the decision is linked to — and when a
    decision has none, the content says so rather than being quietly uncited.
    """
    await harness.ingest()
    registry = tools(harness)

    async with harness.sessions() as session:
        from sqlalchemy import select

        from memoryos.adapters.db import models

        key = (
            await session.execute(select(models.Memory.external_key).limit(1))
        ).scalar_one()

    await record_decision(
        harness.sessions,
        DecisionDraft(
            question="Which storage engine for vectors?",
            chosen="Postgres with pgvector",
            reasoning="One database to operate.",
            options=(OptionInput(description="A dedicated vector index"),),
            evidence=(
                EvidenceInput(
                    source_name=harness.source.name,
                    external_key=key,
                    relation=EvidenceRelation.INFORMED,
                ),
            ),
        ),
        decided_at=DECIDED_AT,
        decided_at_source=TimeProvenance.DECLARED,
    )

    cited = await registry.call("get_decisions", {"about": "storage vectors"})
    assert "Postgres with pgvector" in cited.content
    assert cited.citations, "a decision with linked evidence cited nothing"

    await record_decision(
        harness.sessions,
        DecisionDraft(
            question="Which colour should the shed be?",
            chosen="Blue",
            options=(OptionInput(description="Green"),),
        ),
        decided_at=DECIDED_AT,
        decided_at_source=TimeProvenance.DECLARED,
    )
    bare = await registry.call("get_decisions", {"about": "shed colour"})
    assert bare.citations == []
    # Uncited, and it says so, which is the property that keeps a model from
    # writing a sourced-sounding sentence about it.
    assert "none linked" in bare.content


async def test_an_empty_result_says_why_rather_than_returning_nothing(
    harness: Harness,
) -> None:
    """Two tools are empty on this corpus for two different reasons, and a model
    that could not tell them apart would report both as "nothing found"."""
    await harness.ingest()
    registry = tools(harness)

    async with harness.sessions() as session:
        from sqlalchemy import select

        from memoryos.adapters.db import models

        memory_id = (await session.execute(select(models.Memory.id).limit(1))).scalar_one()

    graph = await registry.call("traverse_graph", {"memory_id": str(memory_id)})
    assert graph.citations == []
    assert "entit" in graph.content.lower()

    gaps = await registry.call("find_gaps", {"min_days": 3650})
    assert "no silences" in gaps.content.lower()


async def test_a_result_hands_the_next_tool_the_id_it_needs(harness: Harness) -> None:
    """**Two tools take an id "as returned by another tool", and until M7.1 no
    tool returned one.**

    The id was in the citations and never in the content, and the model reads
    the content. So the chain search → get_memory could not be walked at all:
    the model had a filename, and `get_memory` does not take filenames.

    Asserted on the *rendered text* rather than on the citations, because the
    text is the whole interface. A test against `result.citations` would have
    passed for every one of the six months this was broken.
    """
    await harness.ingest()
    registry = tools(harness)

    search = await registry.call("search_memories", {"query": "fox"})
    found = re.findall(r"id: ([0-9a-f-]{36})", search.content)
    assert found, f"search printed no id a second hop could use:\n{search.content}"

    # The id is not merely present, it resolves — which is the only version of
    # this property worth having.
    detail = await registry.call("get_memory", {"memory_id": found[0]})
    assert "There is no memory" not in detail.content
    assert "is not a memory id" not in detail.content

    timeline = await registry.call(
        "query_timeline", {"start": "2000-01-01", "end": "2100-01-01", "period": "month"}
    )
    assert re.search(r"id: [0-9a-f-]{36}", timeline.content), timeline.content


async def test_a_decision_hands_over_its_evidences_ids_and_not_its_own(
    harness: Harness,
) -> None:
    """**A decision id and a memory id are both UUIDs, and one of them works.**

    Found by running it: asked to find a decision and then read what it cited,
    the model passed the `DECISION <uuid>` it had just been shown to
    `get_memory`, was told there was no such memory, and went off searching for
    the text instead — three hops to arrive back where it started.

    The block named a count and no ids, so the id the model needed was the one
    thing the result did not contain. Now it lists them, the header says outright
    which kind of id it is carrying, and `get_memory` names the confusion rather
    than reporting a missing memory.
    """
    await harness.ingest()
    registry = tools(harness)

    async with harness.sessions() as session:
        from sqlalchemy import select

        from memoryos.adapters.db import models

        key = (
            await session.execute(select(models.Memory.external_key).limit(1))
        ).scalar_one()

    await record_decision(
        harness.sessions,
        DecisionDraft(
            question="What do chunk offsets index into?",
            chosen="The memory's text",
            options=(OptionInput(description="The stored chunk text"),),
            evidence=(
                EvidenceInput(
                    source_name=harness.source.name,
                    external_key=key,
                    relation=EvidenceRelation.INFORMED,
                ),
            ),
        ),
        decided_at=DECIDED_AT,
        decided_at_source=TimeProvenance.DECLARED,
    )

    decisions = await registry.call("get_decisions", {"about": "chunk offsets"})
    ids = re.findall(r"ids: ([0-9a-f-]{36})", decisions.content)
    assert ids, decisions.content
    assert "NOT a memory id" in decisions.content

    read = await registry.call("get_memory", {"memory_id": ids[0]})
    assert "There is no memory" not in read.content

    # And the decision's own id, offered to the tool that does not take it, says
    # what went wrong rather than that something is missing.
    decision_id = re.search(r"DECISION ([0-9a-f-]{36})", decisions.content)
    assert decision_id is not None
    wrong = await registry.call("get_memory", {"memory_id": decision_id.group(1)})
    assert "decision id" in wrong.content


def test_every_option_a_description_offers_is_one_the_tool_takes(
    harness: Harness,
) -> None:
    """**`query_timeline` advertised five bucket sizes against an enum with three.**

    Asked for a year, a model got `'year' is not a period` and lost a hop to a
    sentence this project wrote itself. Under M7.0 that cost a whole question;
    under M7.1 it costs a hop out of six, which is worse in aggregate and harder
    to see.

    The description is now generated from `Period`, so this test is checking that
    it still is rather than checking the string.
    """
    spec = next(
        spec for spec in tools(harness).specs() if spec.name == "query_timeline"
    )
    offered = spec.parameters["properties"]["period"]["description"]
    known = {member.value for member in Period}
    quoted = set(re.findall(r"[a-z]+", offered.split(":", 1)[1]))
    assert quoted - {"one", "of"} == known, offered


# --------------------------------------------------------------------------
# 4. A cut result says it was cut
# --------------------------------------------------------------------------


async def test_a_result_over_the_cap_sets_truncated(harness: Harness) -> None:
    """**Silently returning the first five of fifty leaves the model reasoning
    about a complete picture it never had**, and the answer is then confidently
    wrong in a way nothing downstream can detect.

    Asserted both ways round, because a `truncated` flag that is always true is
    as useless as one that is never set.
    """
    await harness.ingest()
    registry = tools(harness)

    everything = await registry.call("search_memories", {"query": "fox", "limit": 1})
    assert everything.truncated is True
    # The model reads the content, not the field, so the content has to say it.
    assert "more" in everything.content

    at_the_cap = await registry.call(
        "search_memories", {"query": "fox", "limit": MAX_MEMORIES}
    )
    assert len(at_the_cap.citations) <= MAX_MEMORIES * 4
    assert at_the_cap.truncated is False


async def test_asking_past_the_cap_is_clamped_rather_than_refused(
    harness: Harness,
) -> None:
    """**A bound in the published schema turned "the model asked for ten" into a
    dead turn**, so the bound is not published and the tool clamps instead.

    Groq validates the model's generated arguments against the declared schema
    on its own side and answers, for the whole request rather than the one call:

        400 tool_use_failed: tool call validation failed: parameters for tool
        search_memories did not match schema:
        errors: [`/limit`: must be <= 2 but found 10]

    Nothing in this system gets to correct that, because there is no result left
    to correct from. With the bound in the description instead, the same model
    asks for ten, gets the cap, and is told it was truncated.

    So a model asking for fifty gets the five best and is told the result was
    cut, which is what the milestone specifies caps to do anyway.
    """
    await harness.ingest()
    registry = tools(harness)

    result = await registry.call("search_memories", {"query": "fox", "limit": 50})

    assert result.truncated is True
    assert result.citations, "clamping must still return results, not an error"
    assert len(result.content.split("[1]")) == 2

    spec = next(
        spec for spec in registry.specs() if spec.name == "search_memories"
    )
    # The bound is described rather than declared, so the provider cannot
    # pre-empt the tool's own handling of it.
    assert "maximum" not in spec.parameters["properties"]["limit"]
    assert "5" in spec.parameters["properties"]["limit"]["description"]


# --------------------------------------------------------------------------
# 5. The schema has to survive both providers
# --------------------------------------------------------------------------


def test_every_schema_survives_both_providers() -> None:
    """**Gemini rejects `additionalProperties`; Groq accepts everything.**

    Measured against both live APIs during this milestone — a 400 reading
    `Unknown name "additional_properties"` — and pinned here so the translation
    cannot be dropped as tidy-up. The strict schema stays strict where it is
    validated, and exactly one key is removed on the way to one provider.
    """

    class Args(BaseModel):
        model_config = {"extra": "forbid"}
        query: str

    spec = spec_for(Args, name="example", description="An example.")

    assert spec.parameters["additionalProperties"] is False
    assert "additionalProperties" not in _gemini_schema(spec.parameters)
    # And nothing else is stripped: the bounds and descriptions are what the
    # model routes on, and a broader strip would be guessing.
    assert _gemini_schema(spec.parameters)["properties"] == spec.parameters["properties"]
    # The class name never reaches a prompt.
    assert "title" not in spec.parameters


def test_the_six_tools_are_registered_once_each(harness: Harness) -> None:
    """The registry refuses a duplicate name, because two tools under one name
    means the model's choice no longer identifies which code runs."""
    registry = tools(harness)

    assert registry.names() == [
        "search_memories",
        "get_decisions",
        "query_timeline",
        "find_gaps",
        "traverse_graph",
        "get_memory",
    ]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(GetMemoryTool(sessions=harness.sessions))


def test_every_spec_describes_what_it_is_for(harness: Harness) -> None:
    """**The descriptions are the interface**, so their shape is a property
    worth pinning rather than a matter of taste.

    Not a style check: each has to say when to prefer it, because the routing
    failure this milestone can produce is two tools that both plausibly answer
    and no way for a model to choose. The two that find memories must name each
    other.
    """
    specs = {spec.name: spec for spec in tools(harness).specs()}

    for spec in specs.values():
        assert len(spec.description) > 120, f"{spec.name} is described too thinly"
        assert spec.parameters["type"] == "object"

    assert "traverse_graph" in specs["search_memories"].description
    assert "search_memories" in specs["traverse_graph"].description
    assert "search_memories" in specs["get_memory"].description


def test_a_spec_is_the_models_schema(harness: Harness) -> None:
    """The schema a provider reads and the model that validates the reply are
    the same declaration, so they cannot drift."""
    registry = tools(harness)
    spec = next(spec for spec in registry.specs() if spec.name == "search_memories")

    assert isinstance(spec, ToolSpec)
    assert set(spec.parameters["properties"]) == {"query", "limit"}
    assert spec.parameters["required"] == ["query"]
    # No `maximum`: see `test_asking_past_the_cap_is_clamped_rather_than_refused`.
    assert "maximum" not in spec.parameters["properties"]["limit"]


def test_each_tool_wraps_exactly_one_use_case() -> None:
    """The milestone's own claim, asserted as a shape.

    Every tool holds collaborators and no state of its own — they are frozen
    dataclasses — so a tool that grew logic would have to grow a field to keep it
    in. This is the cheapest available check that the wrapping stayed wrapping.
    """
    for tool in (
        SearchMemoriesTool,
        TraverseGraphTool,
        QueryTimelineTool,
        FindGapsTool,
        GetDecisionsTool,
        GetMemoryTool,
    ):
        assert dataclasses.is_dataclass(tool)
        assert dataclasses.fields(tool), f"{tool.__name__} holds no collaborators"
