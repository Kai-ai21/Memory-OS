"""Standing checks for the class of defect that caused M1.6.1.

The window misalignment was invisible: no error, no failing test, just
retrieval that was quietly worse than it looked. The startup assertion stops
that specific pair from drifting again; this reports the state of data already
written, which an assertion cannot see.
"""

from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import Embedder


@dataclass(slots=True)
class Finding:
    check: str
    count: int
    detail: str
    examples: list[str] = field(default_factory=list)
    # Reported and counted, but does not make the report unhealthy.
    #
    # The distinction is between damage and an absence. Every other check here
    # names something that is *wrong* with data already written — a chunk past
    # the model's window, a vector from the wrong model — and a non-zero count
    # is a defect. Entity extraction not having run is a capability that has not
    # been exercised: it needs an API key, a keyless deployment is legitimate
    # and fully working for everything Phase 1 and Phase 2 do, and `doctor`
    # exiting non-zero on a fresh corpus would be reporting an optional feature
    # as corpus damage. Same judgement `GraphStatus.healthy` makes about an
    # unreachable Neo4j.
    #
    # It is still a finding rather than nothing, because the state it reports is
    # invisible otherwise and something downstream depends on it.
    advisory: bool = False

    @property
    def healthy(self) -> bool:
        return self.advisory or self.count == 0


@dataclass(slots=True)
class GraphStatus:
    """What the graph projection reports about itself.

    Its own type rather than another `Finding`, because `Finding` means "count
    of a bad thing, zero is healthy" and none of this fits that shape. Node
    counts are information, not defects — a graph with 4,000 entities is not
    four thousand times unhealthier than one with one.
    """

    reachable: bool
    expected_version: int
    schema_version: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def current(self) -> bool:
        """Whether the applied schema matches the code.

        `None` — never applied — counts as current, because the next connect
        applies it. A *stale* number does not: it means somebody has been
        writing to this database under constraints the code no longer assumes.
        """
        return self.schema_version in (None, self.expected_version)

    @property
    def healthy(self) -> bool:
        """Unreachable is not unhealthy here, and that is the milestone's rule.

        Everything Phase 1 and Phase 2 do keeps working without the graph, so a
        `doctor` that exited non-zero on a machine with no Neo4j running would be
        reporting an outage of something optional as corpus damage. Drift is
        different: a reachable database whose schema is behind the code is a
        real inconsistency, and one nothing else will notice.
        """
        return self.current if self.reachable else True


@dataclass(slots=True)
class DoctorReport:
    findings: list[Finding] = field(default_factory=list)
    graph: GraphStatus | None = None

    @property
    def healthy(self) -> bool:
        corpus = all(finding.healthy for finding in self.findings)
        return corpus and (self.graph is None or self.graph.healthy)


class GraphDiagnostics(Protocol):
    """The slice of the graph store `doctor` needs.

    Narrower than `GraphStore` on purpose: this reports on the graph and must
    not be able to write to it, and a protocol that offered `clear` to a
    diagnostic would be an invitation.
    """

    async def verify(self) -> None: ...

    async def schema_version(self) -> int | None: ...

    async def counts_by_label(self) -> dict[str, int]: ...


async def inspect_graph(graph: GraphDiagnostics, *, expected_version: int) -> GraphStatus:
    """Reachability, schema version, and node counts — or why not.

    Every failure is caught and reported rather than raised. `doctor` runs when
    something is already suspected to be wrong, so it has to survive the thing
    it is diagnosing being broken.
    """
    try:
        await graph.verify()
    except Exception as exc:
        return GraphStatus(
            reachable=False, expected_version=expected_version, error=str(exc)
        )

    try:
        version = await graph.schema_version()
        counts = await graph.counts_by_label()
    except Exception as exc:
        # Reachable but unqueryable — a wrong database name, or a user without
        # read privileges. Distinct from unreachable, and worth saying so.
        return GraphStatus(
            reachable=True, expected_version=expected_version, error=str(exc)
        )

    return GraphStatus(
        reachable=True,
        expected_version=expected_version,
        schema_version=version,
        counts=counts,
    )


async def run_doctor(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    *,
    graph: GraphDiagnostics | None = None,
    graph_schema_version: int = 0,
    sample_limit: int = 5,
) -> DoctorReport:
    findings: list[Finding] = []
    window = embedder.max_sequence_tokens

    async with session_factory() as session:
        # Counting every chunk with the real tokenizer would mean tokenizing the
        # whole corpus, so the stored token_count is used as the filter and the
        # tokenizer only confirms the suspects.
        candidates = (
            await session.execute(
                select(models.MemoryChunk.id, models.MemoryChunk.content)
                .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
                .where(
                    models.Memory.is_current.is_(True),
                    models.MemoryChunk.token_count >= window // 2,
                )
            )
        ).all()

        oversized = [
            str(chunk_id)
            for chunk_id, content in candidates
            if embedder.count_tokens(content) > window
        ]
        findings.append(
            Finding(
                check="chunks_over_model_window",
                count=len(oversized),
                detail=(
                    f"chunks whose text exceeds {window} model tokens; everything "
                    f"past the window is discarded before embedding"
                ),
                examples=oversized[:sample_limit],
            )
        )

        stale = (
            await session.execute(
                select(models.MemoryChunk.embedding_model, func.count())
                .where(
                    models.MemoryChunk.embedding.is_not(None),
                    models.MemoryChunk.embedding_model.is_distinct_from(
                        embedder.model_id
                    ),
                )
                .group_by(models.MemoryChunk.embedding_model)
            )
        ).all()
        findings.append(
            Finding(
                check="chunks_from_another_model",
                count=sum(count for _, count in stale),
                detail=(
                    f"chunks embedded by a model other than {embedder.model_id}; "
                    f"vectors from different models are not comparable"
                ),
                examples=[f"{model or '(none)'}: {count}" for model, count in stale],
            )
        )

        unembedded = (
            await session.execute(
                select(func.count())
                .select_from(models.MemoryChunk)
                .where(models.MemoryChunk.embedding.is_(None))
            )
        ).scalar_one()
        findings.append(
            Finding(
                check="chunks_without_embeddings",
                count=unembedded,
                detail="chunks with no vector; invisible to search",
            )
        )

        empty = (
            await session.execute(
                select(models.Memory.external_key)
                .outerjoin(
                    models.MemoryChunk,
                    models.MemoryChunk.memory_id == models.Memory.id,
                )
                .where(
                    models.Memory.is_current.is_(True),
                    models.Memory.deleted_at.is_(None),
                    models.Memory.content.is_not(None),
                    func.length(func.trim(models.Memory.content)) > 0,
                )
                .group_by(models.Memory.id, models.Memory.external_key)
                .having(func.count(models.MemoryChunk.id) == 0)
            )
        ).all()
        findings.append(
            Finding(
                check="memories_with_content_but_no_chunks",
                count=len(empty),
                detail="normalized text exists but produced no chunks",
                examples=[row[0] for row in empty[:sample_limit]],
            )
        )

        # The condition a replay creates and nothing reported until M5.1 needed
        # it. `entities`, `entity_mentions` and `entity_relationships` are
        # derived-and-not-rebuilt: a full replay truncates them and leaves them
        # empty, because rebuilding means an LLM call per chunk against offsets
        # that no longer point anywhere. `application/replay.py` says so and
        # says to re-run extraction afterwards — and nothing checked whether
        # anybody had.
        #
        # It matters more than a missing convenience. M5.1 narrows outcome
        # candidates to memories sharing entities with a decision's evidence, so
        # a corpus with no mentions makes that filter return nothing while
        # looking exactly like a corpus where nothing shares an entity. Two
        # full replays during M5.0 left this at zero and the only thing that
        # noticed was a hand-written query.
        unextracted = (
            await session.execute(
                select(func.count())
                .select_from(models.Memory)
                .where(
                    models.Memory.is_current.is_(True),
                    models.Memory.deleted_at.is_(None),
                    models.Memory.entity_extractor_version.is_(None),
                )
            )
        ).scalar_one()
        mentions = (
            await session.execute(select(func.count()).select_from(models.EntityMention))
        ).scalar_one()
        findings.append(
            Finding(
                check="memories_without_entity_extraction",
                count=unextracted,
                detail=(
                    "current memories no extractor has run over; the graph and "
                    "any entity-scoped query see nothing of them. A full replay "
                    "truncates the entity tables and does not rebuild them — "
                    "re-run `extract-entities`"
                ),
                examples=[f"{mentions} mention rows in the corpus"],
                advisory=True,
            )
        )

    status = (
        await inspect_graph(graph, expected_version=graph_schema_version)
        if graph is not None
        else None
    )
    return DoctorReport(findings=findings, graph=status)
