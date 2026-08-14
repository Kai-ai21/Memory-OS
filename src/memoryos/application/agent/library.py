"""The six tools, each an existing use case with a schema around it.

**How much of this is new logic is the milestone's actual question**, so the
answer is written here rather than only in the report. Each tool does three
things: declare its arguments, call one use case, and render the result as text
with citations attached. The rendering is the only code with any content in it,
and it is rendering.

Two places needed something that did not exist, and both were the same shape —
a capability that had been written for a transport rather than for the domain:

* `get_memory` had no use case at all. Its three queries lived in the body of
  `GET /memories/{id}`, unreachable from anywhere that was not HTTP. Extracted
  to `application/memories.py`; the route now calls it too.
* `find_gaps` and `query_timeline` take a source *id* and every caller has a
  *name*. That translation existed twice already — once in the timeline route,
  once inline in the CLI — and this would have been the third. Extracted to
  `application/sources.py`.

Neither is new behaviour. Both are existing behaviour moved to where a second
caller can reach it, which is what "the layering earned its cost" has to mean if
it means anything.

### Descriptions are the interface

A model picks a tool by reading its description and nothing else. Two of these
find memories and the difference between them is the whole game: `search_memories`
finds text that *means* something, `traverse_graph` finds memories *connected* to
one you already have. Each description says what it is for, when to prefer it,
and what it cannot do — that last part being the one that stops a model reaching
for a timeline when it wants a passage.

### Caps

Every tool caps what it returns and says so when it cuts. A `search_memories`
that returned fifty memories at full text would fill a context window in one
call, and the model would have no way to know that it had.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.application import decisions as decisions_app
from memoryos.application import memories as memories_app
from memoryos.application import sources as sources_app
from memoryos.application import temporal
from memoryos.application.agent.tools import Tool, ToolResult, ToolSpec, spec_for
from memoryos.application.citations import (
    citations_for_chunks,
    citations_for_memories,
    explain_hits,
)
from memoryos.application.graph_expand import ExpandThroughGraph
from memoryos.application.ports import ScoredChunk
from memoryos.application.search import FusionWeights, SearchMemories
from memoryos.domain.citation import Citation
from memoryos.domain.fusion import DEFAULT_RRF_K
from memoryos.domain.values import AssumptionVerdict, Period

# The caps this milestone specifies. Named rather than inlined, because each is
# a judgement about a context window and the next person to change one should
# see what the others are.
MAX_MEMORIES = 5
MAX_GRAPH_NODES = 20
MAX_BUCKETS = 12
MAX_DECISIONS = 5
MAX_GAPS = 10

# How much of a memory's text one result may spend.
#
# The excerpt builder produces a matched span with context either side; this
# bounds the whole thing. A code file's chunk can be a screen of imports, and
# five of those is the context window gone.
EXCERPT_CHARS = 600


class _Args(BaseModel):
    """Base for every tool's arguments.

    `extra="forbid"` is what puts `additionalProperties: false` in the generated
    schema, which is what makes a hallucinated argument an error the model is
    told about rather than a value silently dropped.
    """

    model_config = ConfigDict(extra="forbid")


def _capped(value: int, cap: int) -> tuple[int, bool]:
    """A limit the model asked for, brought inside the tool's own cap.

    **Clamped rather than refused, and the bound is deliberately not in the
    published schema.** Both halves of that were forced by a real 400 from Groq:

        tool call validation failed: parameters for tool search_memories
        did not match schema

    Groq validates the model's generated arguments against the declared schema
    *server-side* and fails the entire request when they miss — so a
    `"maximum": 5` in the schema turns "the model asked for ten" into a dead
    turn rather than into something correctable. The milestone's own design says
    the cap is the tool's job and `truncated` is how the model learns about it,
    which is exactly what this does: ask for fifty, get five, and be told.

    The range still appears in the description, so the model is informed rather
    than merely constrained.
    """
    wanted = max(1, value)
    return min(wanted, cap), wanted > cap


def _fmt(moment: datetime | None) -> str:
    return "undated" if moment is None else moment.strftime("%Y-%m-%d")


def _when(value: object) -> datetime | None:
    """`MemoryHit.occurred_at` is typed `object`, so it is narrowed rather than
    cast: a `# type: ignore` here would be this module asserting something about
    a field it does not own."""
    return value if isinstance(value, datetime) else None


def _clip(text: str, limit: int = EXCERPT_CHARS) -> str:
    stripped = " ".join(text.split())
    return stripped if len(stripped) <= limit else stripped[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Phase 2 — search
# --------------------------------------------------------------------------


class SearchArgs(_Args):
    query: str = Field(
        description=(
            "What to look for, in natural language. A question or a phrase, not "
            "keywords: this is matched by meaning as well as by wording."
        )
    )
    limit: int = Field(
        default=MAX_MEMORIES,
        description=(
            f"How many memories to return, at most {MAX_MEMORIES}. Asking for "
            "more returns the best ones and says the result was truncated."
        ),
    )


@dataclass(frozen=True, slots=True)
class SearchMemoriesTool:
    """Phase 2's hybrid retrieval, unchanged, with excerpts and citations."""

    search: SearchMemories
    sessions: async_sessionmaker[AsyncSession]
    weights: FusionWeights

    arguments: type[BaseModel] = SearchArgs

    @property
    def spec(self) -> ToolSpec:
        return spec_for(
            SearchArgs,
            name="search_memories",
            description=(
                "Find passages in the corpus that are ABOUT a topic, by meaning "
                "and by wording together. This is the default tool and the right "
                "one for almost any question that names a subject rather than a "
                "specific item: 'how does X work', 'what did we say about Y', "
                "'where is Z handled'. It searches the text of every memory and "
                "returns the passages that matched, each with the file it came "
                "from. Prefer it over traverse_graph when you have a topic rather "
                "than a specific memory, and over query_timeline when you want "
                "what was said rather than when things happened."
            ),
        )

    async def call(self, **kwargs: Any) -> ToolResult:
        args = SearchArgs.model_validate(kwargs)
        limit, over_cap = _capped(args.limit, MAX_MEMORIES)
        # One more than the cap, so "there were others" is a fact rather than an
        # inference from having returned exactly the cap. `SearchResult` carries
        # no total — retrieval stops at k — and the extra row is the cheapest
        # honest way to know whether the cut was real.
        result = await self.search(args.query, k=limit + 1)
        if not result.hits:
            return ToolResult(
                content=(
                    f"No memories in the corpus matched {args.query!r}. "
                    "The corpus may not contain this subject at all."
                )
            )

        truncated = over_cap or len(result.hits) > limit
        explained = await explain_hits(
            self.sessions,
            result.hits[:limit],
            weights=self.weights,
            rrf_k=DEFAULT_RRF_K,
        )
        lines: list[str] = []
        citations: list[Citation] = []
        for position, item in enumerate(explained, start=1):
            hit = item.hit
            best = max(item.citations, key=lambda c: len(c.excerpt), default=None)
            passage = _clip(
                best.context.text if best and best.context else (best.excerpt if best else "")
            )
            lines.append(
                f"[{position}] {hit.source_name}::{hit.external_key} "
                f"({hit.kind}, {_fmt(_when(hit.occurred_at))}, score {hit.score:.3f})\n"
                f"    {passage}"
            )
            citations.extend(item.citations)

        header = f"{len(explained)} memories"
        if truncated:
            header += " (the best ones; there were more)"
        return ToolResult(
            content=f"{header} for {args.query!r}:\n\n" + "\n\n".join(lines),
            citations=citations,
            truncated=truncated,
        )


# --------------------------------------------------------------------------
# Phase 3 — the graph
# --------------------------------------------------------------------------


class TraverseArgs(_Args):
    memory_id: str = Field(
        description=(
            "The id of a memory to start from, as returned by another tool. "
            "Traversal walks outwards from this memory's entities."
        )
    )
    limit: int = Field(
        default=MAX_GRAPH_NODES,
        description=(
            f"How many connected memories to return, at most {MAX_GRAPH_NODES}."
        ),
    )


@dataclass(frozen=True, slots=True)
class TraverseGraphTool:
    """M3.5's expansion, seeded by a memory rather than by a search."""

    expand: ExpandThroughGraph
    sessions: async_sessionmaker[AsyncSession]

    arguments: type[BaseModel] = TraverseArgs

    @property
    def spec(self) -> ToolSpec:
        return spec_for(
            TraverseArgs,
            name="traverse_graph",
            description=(
                "Find memories STRUCTURALLY connected to one specific memory, by "
                "walking the entities they share — the same person, system or "
                "concept named in both. This is not a text search: it finds "
                "related items that may share no wording at all with the "
                "starting memory, which is exactly what search_memories cannot "
                "do. It needs a memory id to start from, so use it only when you "
                "already have one; if you have a topic rather than an id, use "
                "search_memories instead."
            ),
        )

    async def call(self, **kwargs: Any) -> ToolResult:
        args = TraverseArgs.model_validate(kwargs)
        try:
            seed = UUID(args.memory_id)
        except ValueError:
            return ToolResult(
                content=(
                    f"{args.memory_id!r} is not a memory id. Memory ids are UUIDs "
                    "and are returned by search_memories and get_memory."
                )
            )

        candidates = await self.expand([seed])
        if not candidates.chunks:
            # An empty expansion is an answer, not a failure — M3.5 designed it
            # that way and the reason is usually knowable, so it is stated.
            reason = (
                "the corpus has no extracted entities to walk"
                if candidates.seeds == 0
                else "nothing shares an entity with it"
            )
            return ToolResult(
                content=(
                    f"No memories are connected to {seed} through the graph: "
                    f"{reason}. Entity extraction has to have been run over the "
                    "corpus for this tool to return anything."
                )
            )

        limit, over_cap = _capped(args.limit, MAX_GRAPH_NODES)
        shown = candidates.chunks[:limit]
        citations = await citations_for_chunks(self.sessions, shown)
        lines = [
            f"[{position}] {citation.source_name}::{citation.external_key} "
            f"(via {_route(candidates.routes, chunk.chunk_id)})\n"
            f"    {_clip(citation.excerpt)}"
            for position, (chunk, citation) in enumerate(
                zip(shown, citations, strict=False), start=1
            )
        ]
        truncated = over_cap or len(candidates.chunks) > len(shown)
        return ToolResult(
            content=(
                f"{len(shown)} memories connected to {seed}"
                + (f" (of {len(candidates.chunks)} reached)" if truncated else "")
                + ":\n\n"
                + "\n\n".join(lines)
            ),
            citations=citations,
            truncated=truncated,
        )


# --------------------------------------------------------------------------
# Phase 4 — time
# --------------------------------------------------------------------------


class TimelineArgs(_Args):
    start: str = Field(
        description="Start of the window, as YYYY-MM-DD. Inclusive."
    )
    end: str = Field(description="End of the window, as YYYY-MM-DD. Exclusive.")
    period: str = Field(
        default="week",
        description="Bucket size: one of day, week, month, quarter, year.",
    )
    source: str = Field(
        default="",
        description="Restrict to one source by name. Empty means every source.",
    )


@dataclass(frozen=True, slots=True)
class QueryTimelineTool:
    """M4.1's activity histogram, plus the memories that make it up."""

    sessions: async_sessionmaker[AsyncSession]

    arguments: type[BaseModel] = TimelineArgs

    @property
    def spec(self) -> ToolSpec:
        return spec_for(
            TimelineArgs,
            name="query_timeline",
            description=(
                "Show WHEN things happened: how much activity there was in each "
                "week or month of a date range, and which memories fall in it. "
                "Use it for questions about a period rather than a subject — "
                "'what was I working on in August', 'how busy was last quarter', "
                "'what changed between March and May'. It answers with counts "
                "over time and a sample of the memories behind them; it does not "
                "search text, so use search_memories when the question names a "
                "topic instead of a time."
            ),
        )

    async def call(self, **kwargs: Any) -> ToolResult:
        args = TimelineArgs.model_validate(kwargs)
        try:
            start = _day(args.start)
            end = _day(args.end)
        except ValueError as exc:
            return ToolResult(content=f"{exc} Dates look like 2026-08-01.")
        if end <= start:
            return ToolResult(
                content=f"The end date {args.end} is not after the start {args.start}."
            )
        try:
            period = Period(args.period)
        except ValueError:
            allowed = ", ".join(member.value for member in Period)
            return ToolResult(
                content=f"{args.period!r} is not a period. Use one of: {allowed}."
            )
        try:
            source_id = await sources_app.resolve(self.sessions, args.source or None)
        except sources_app.UnknownSource as exc:
            return ToolResult(content=str(exc))

        buckets = await temporal.activity_by_period(
            self.sessions, start, end, period=period, source_id=source_id
        )
        found = await temporal.memories_in_range(
            self.sessions, start, end, source_id=source_id
        )
        if not found:
            return ToolResult(
                content=(
                    f"No memories are dated between {args.start} and {args.end}"
                    + (f" in source {args.source}" if args.source else "")
                    + ". Nothing was recorded then, or nothing from then carries a date."
                )
            )

        # Empty buckets are dropped rather than rendered as zeroes. A year of
        # "0" rows spends the budget saying nothing happened, which the counts
        # that are shown already imply.
        occupied = [bucket for bucket in buckets if bucket.count]
        shown_buckets = occupied[:MAX_BUCKETS]
        sample = found[:MAX_MEMORIES]
        rows = "\n".join(
            f"  {_fmt(bucket.start)}  {bucket.count:>4}  "
            + ", ".join(f"{kind} {count}" for kind, count in bucket.by_kind.items())
            for bucket in shown_buckets
        )
        listed = "\n".join(
            f"  {_fmt(memory.occurred_at)}  {memory.external_key}" for memory in sample
        )
        citations = await citations_for_memories(
            self.sessions, [memory.id for memory in sample]
        )
        # Either cut counts as truncation: the model is told once, and a reader
        # of the answer cannot be left believing they saw every bucket *or*
        # every memory.
        truncated = len(found) > len(sample) or len(occupied) > len(shown_buckets)
        return ToolResult(
            content=(
                f"{len(found)} memories are dated between {args.start} and "
                f"{args.end}, by {period.value}:\n{rows}\n\n"
                f"Showing {len(sample)} of them:\n{listed}"
                + (
                    "\n\nThis is a sample; the counts above cover the whole range."
                    if truncated
                    else ""
                )
            ),
            citations=citations,
            truncated=truncated,
        )


class GapsArgs(_Args):
    min_days: float = Field(
        default=30.0,
        description="The shortest silence worth reporting, in days. Must be positive.",
    )
    source: str = Field(
        default="",
        description="Restrict to one source by name. Empty means every source.",
    )


@dataclass(frozen=True, slots=True)
class FindGapsTool:
    """M4.3's gap detection: the capability no retriever can provide."""

    sessions: async_sessionmaker[AsyncSession]

    arguments: type[BaseModel] = GapsArgs

    @property
    def spec(self) -> ToolSpec:
        return spec_for(
            GapsArgs,
            name="find_gaps",
            description=(
                "Find PERIODS OF SILENCE — stretches of time with activity "
                "either side and none during. Use it for questions about "
                "something stopping or being abandoned: 'when did I stop working "
                "on this', 'was there a break', 'what went quiet'. No search tool "
                "can answer these, because an absence is not written down "
                "anywhere and there is no document to retrieve; only counting "
                "over time can see a hole."
            ),
        )

    async def call(self, **kwargs: Any) -> ToolResult:
        args = GapsArgs.model_validate(kwargs)
        if args.min_days <= 0:
            return ToolResult(
                content=(
                    f"min_days has to be positive; {args.min_days:g} is not a "
                    "length of time. Call it again with a positive number."
                )
            )
        try:
            wanted = await sources_app.resolve(self.sessions, args.source or None)
        except sources_app.UnknownSource as exc:
            return ToolResult(content=str(exc))

        # Per source rather than corpus-wide, which is M4.3's rule: a silence in
        # one source that another was busy through is not a silence.
        scopes = [
            source
            for source in await sources_app.listed(self.sessions)
            if wanted is None or source.id == wanted
        ]
        found: list[tuple[str, temporal.Gap]] = []
        for source in scopes:
            for gap in await temporal.find_gaps(
                self.sessions,
                temporal.SourceScope(source.id),
                min_gap=timedelta(days=args.min_days),
            ):
                found.append((source.name, gap))

        if not found:
            return ToolResult(
                content=(
                    f"No silences of {args.min_days:g} days or more"
                    + (f" in source {args.source}" if args.source else "")
                    + ". Activity is continuous at that resolution."
                )
            )

        found.sort(key=lambda pair: pair[1].start)
        shown = found[:MAX_GAPS]
        lines = [
            f"  {name}: {_fmt(gap.start)} -> {_fmt(gap.end)} "
            f"({gap.duration.days} days), between {gap.before.external_key} "
            f"and {gap.after.external_key}"
            for name, gap in shown
        ]
        citations = await citations_for_memories(
            self.sessions,
            [memory.id for _, gap in shown for memory in (gap.before, gap.after)],
        )
        return ToolResult(
            content=(
                f"{len(found)} silence(s) of {args.min_days:g} days or more:\n"
                + "\n".join(lines)
            ),
            citations=citations,
            truncated=len(found) > len(shown),
        )


# --------------------------------------------------------------------------
# Phase 5 — decisions
# --------------------------------------------------------------------------


class DecisionArgs(_Args):
    about: str = Field(
        default="",
        description=(
            "Words to match against the question a decision answered. Empty "
            "returns the most recent decisions."
        )
    )
    limit: int = Field(
        default=MAX_DECISIONS,
        description=f"How many decisions to return, at most {MAX_DECISIONS}.",
    )


@dataclass(frozen=True, slots=True)
class GetDecisionsTool:
    """M5.0's decisions with their alternatives and assumptions."""

    sessions: async_sessionmaker[AsyncSession]

    arguments: type[BaseModel] = DecisionArgs

    @property
    def spec(self) -> ToolSpec:
        return spec_for(
            DecisionArgs,
            name="get_decisions",
            description=(
                "Retrieve recorded DECISIONS: what was chosen, what else was "
                "considered, why, and what had to be true for it to be right. "
                "Use it whenever a question asks why something is the way it is "
                "— 'why did we choose X', 'what were the alternatives to Y', "
                "'what were we assuming'. A decision's reasoning and its rejected "
                "options are recorded here and appear nowhere in the corpus text, "
                "so search_memories cannot find them."
            ),
        )

    async def call(self, **kwargs: Any) -> ToolResult:
        args = DecisionArgs.model_validate(kwargs)
        summaries = await decisions_app.list_decisions(self.sessions, limit=200)
        wanted = args.about.strip().lower()
        if wanted:
            # Substring over the question, which is what a decision is *about*.
            # Deliberately not embedded: decisions are tens of rows, and adding a
            # second retrieval path for them would be a ranking to explain.
            terms = [term for term in wanted.split() if len(term) > 2]
            summaries = [
                summary
                for summary in summaries
                if any(term in summary.question.lower() for term in terms)
            ]
        if not summaries:
            return ToolResult(
                content=(
                    f"No recorded decision matches {args.about!r}."
                    if wanted
                    else "No decisions have been recorded."
                )
            )

        limit, over_cap = _capped(args.limit, MAX_DECISIONS)
        shown = summaries[:limit]
        blocks: list[str] = []
        cited: list[UUID] = []
        for summary in shown:
            detail = await decisions_app.show(self.sessions, summary.id)
            rejected = "; ".join(option.description for option in detail.rejected)
            assumptions = "; ".join(
                f"{row.statement} ({_held(row.held)})" for row in detail.assumptions
            )
            evidence = [row for row in detail.evidence if row.memory_id is not None]
            cited.extend(row.memory_id for row in evidence if row.memory_id)
            blocks.append(
                f"DECISION {detail.id} ({_fmt(detail.decided_at)}, {detail.status.value})\n"
                f"  Question: {detail.question}\n"
                f"  Chose: {detail.chosen}\n"
                + (f"  Because: {_clip(detail.reasoning)}\n" if detail.reasoning else "")
                + (f"  Rejected: {rejected}\n" if rejected else "")
                + (f"  Assumed: {assumptions}\n" if assumptions else "")
                + (
                    f"  Evidence: {len(evidence)} linked memories\n"
                    if evidence
                    else "  Evidence: none linked in the corpus\n"
                )
            )

        # **The citations are the decisions' evidence, not the decisions.** A
        # decision is a row somebody wrote; it has no chunk, no offsets and no
        # place in the corpus, so a citation to one would be an invented span.
        # What is citable is the corpus evidence linked to it — and where a
        # decision has none, the block above says so rather than being silently
        # uncited.
        citations = await citations_for_memories(self.sessions, cited)
        return ToolResult(
            content="\n".join(blocks),
            citations=citations,
            truncated=over_cap or len(summaries) > len(shown),
        )


# --------------------------------------------------------------------------
# Phase 1 — one memory
# --------------------------------------------------------------------------


class MemoryArgs(_Args):
    memory_id: str = Field(
        description="The id of the memory to read, as returned by another tool."
    )


@dataclass(frozen=True, slots=True)
class GetMemoryTool:
    """One memory in full, with its version history."""

    sessions: async_sessionmaker[AsyncSession]

    arguments: type[BaseModel] = MemoryArgs

    @property
    def spec(self) -> ToolSpec:
        return spec_for(
            MemoryArgs,
            name="get_memory",
            description=(
                "Read ONE specific memory in full by its id, with how many "
                "versions of it exist and when it last changed. Use it after "
                "another tool has returned an id and the excerpt was not enough "
                "— it returns the whole item rather than a matched passage. It "
                "cannot find anything: without an id already in hand, use "
                "search_memories."
            ),
        )

    async def call(self, **kwargs: Any) -> ToolResult:
        args = MemoryArgs.model_validate(kwargs)
        try:
            memory_id = UUID(args.memory_id)
        except ValueError:
            return ToolResult(
                content=(
                    f"{args.memory_id!r} is not a memory id. Memory ids are UUIDs "
                    "and are returned by search_memories."
                )
            )
        try:
            detail = await memories_app.show(self.sessions, memory_id)
        except memories_app.UnknownMemory:
            return ToolResult(content=f"There is no memory with id {memory_id}.")

        body = detail.content or ""
        # Whole-item tools are the ones most likely to blow a context window, so
        # the cap is per chunk rather than per memory: enough chunks to answer
        # from, each clipped, and the count said out loud.
        shown_chunks = detail.chunks[:MAX_MEMORIES]
        excerpt = "\n\n".join(
            f"  [chunk {chunk.ordinal}] {_clip(chunk.content[chunk.prefix_chars :])}"
            for chunk in shown_chunks
        )
        truncated = len(detail.chunks) > len(shown_chunks) or len(body) > EXCERPT_CHARS
        history = (
            f"{len(detail.versions)} versions, this is v{detail.version}"
            f"{' (current)' if detail.is_current else ' (superseded)'}"
        )
        citations = await citations_for_chunks(
            self.sessions,
            [
                ScoredChunk(
                    chunk_id=chunk.id,
                    memory_id=detail.id,
                    ordinal=chunk.ordinal,
                    text=chunk.content,
                    score=0.0,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    prefix_chars=chunk.prefix_chars,
                    metadata=chunk.metadata,
                )
                for chunk in shown_chunks
            ],
        )
        return ToolResult(
            content=(
                f"{detail.source_name}::{detail.external_key} ({detail.kind}, "
                f"{_fmt(detail.occurred_at)})\n"
                f"  {history}\n"
                f"  {len(detail.chunks)} chunks"
                + (f", showing the first {len(shown_chunks)}" if truncated else "")
                + f"\n\n{excerpt}"
            ),
            citations=citations,
            truncated=truncated,
        )


def _route(routes: dict[str, tuple[str, ...]], chunk_id: UUID) -> str:
    """The entity path that reached a chunk, or an honest shrug.

    M3.5 carries the route precisely so "the graph promoted this" is an argument
    rather than an assertion, and dropping it here would undo that at the last
    step.
    """
    path = routes.get(str(chunk_id), ())
    return " -> ".join(path) if path else "a shared entity"


def _held(value: AssumptionVerdict | None) -> str:
    """An assumption's verdict as a word a model can read.

    Three states and not two: M5.2 widened this from a boolean because almost
    nothing anybody assumes is cleanly right or wrong, and collapsing
    `partially` into either neighbour here would undo that at the last step —
    the model would read "held" and write a sentence the record does not
    support. `None` is "nobody has judged this", which is not the same as
    broken and must not read like it.
    """
    if value is None:
        return "not yet evaluated"
    return {
        AssumptionVerdict.HELD: "held",
        AssumptionVerdict.FAILED: "broke",
        AssumptionVerdict.PARTIALLY: "held partially",
    }[value]


def _day(value: str) -> datetime:
    """A `YYYY-MM-DD` string as midnight UTC.

    UTC rather than local, for M4.0's reason: a bare date resolved against the
    caller's zone makes the same question return different rows on two machines,
    and the difference is a few hours — exactly the size nobody notices.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not a date.") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def build_registry(
    *,
    sessions: async_sessionmaker[AsyncSession],
    search: SearchMemories,
    expand: ExpandThroughGraph,
    weights: FusionWeights,
) -> tuple[Tool, ...]:
    """The six tools, in the order they are offered to the model.

    Search first, deliberately. A model reaching for a tool on a vague question
    takes the first plausible one it reads, and search is the right default for
    almost everything; the specialised four are the ones a question has to
    *earn*.
    """
    return (
        SearchMemoriesTool(search=search, sessions=sessions, weights=weights),
        GetDecisionsTool(sessions=sessions),
        QueryTimelineTool(sessions=sessions),
        FindGapsTool(sessions=sessions),
        TraverseGraphTool(expand=expand, sessions=sessions),
        GetMemoryTool(sessions=sessions),
    )
