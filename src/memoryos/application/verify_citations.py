"""Assert that every citation points at the text it claims to.

The property, exactly:

    memory.content[char_start:char_end] == chunk.content[prefix_chars:]

That identity is what makes a citation a citation rather than a plausible
gesture at a file. It is also easy to break invisibly — M1.4a broke it, and
nothing failed: row counts were right, offsets were within bounds, every test
passed, and highlights pointed a few hundred characters from the answer. The
only way that gets caught is by asserting the identity itself, on real data, as
a command somebody runs.

**A verification that cannot fail proves nothing**, so a test corrupts a chunk's
offsets on purpose and requires this to exit non-zero.

Two scopes. By default it checks the chunks behind the golden set's results,
which is what the milestone asks for and what exercises the retrieval path.
`--all` sweeps every current chunk in the corpus, which is strictly stronger and
takes seconds at this size — a citation is only ever as trustworthy as the
weakest row it could have quoted.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models

logger = structlog.get_logger(__name__)

# How much of a mismatch to show. Enough to see where the offsets drifted to
# without printing a chunk into a terminal.
_PREVIEW = 60


@dataclass(frozen=True, slots=True)
class Mismatch:
    memory_id: UUID
    external_key: str
    ordinal: int
    reason: str
    expected: str
    actual: str

    def render(self) -> str:
        return (
            f"  {self.external_key}#{self.ordinal}  ({self.reason})\n"
            f"      chunk says:  {self.expected[:_PREVIEW]!r}\n"
            f"      memory has:  {self.actual[:_PREVIEW]!r}"
        )


@dataclass(frozen=True, slots=True)
class CitationReport:
    checked: int
    memories: int
    mismatches: list[Mismatch] = field(default_factory=list)
    # Chunks whose parent memory has no normalized content to check against.
    # Not a failure — a memory can legitimately be un-normalized — but counted,
    # because a run that checked nothing must not read as a run that passed.
    unverifiable: int = 0

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def render(self) -> str:
        lines = [
            f"chunks checked   {self.checked}",
            f"memories         {self.memories}",
            f"unverifiable     {self.unverifiable} (parent memory has no normalized text)",
            f"mismatches       {len(self.mismatches)}",
        ]
        if self.mismatches:
            lines += ["", "citations pointing at the wrong text:"]
            lines += [item.render() for item in self.mismatches]
        return "\n".join(lines)


async def verify_citations(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    memory_ids: Sequence[UUID] | None = None,
) -> CitationReport:
    """Check the offset identity for every chunk of the given memories.

    `memory_ids=None` sweeps every current memory.
    """
    stmt = (
        select(
            models.Memory.id,
            models.Memory.external_key,
            models.Memory.content,
            models.MemoryChunk.ordinal,
            models.MemoryChunk.content,
            models.MemoryChunk.char_start,
            models.MemoryChunk.char_end,
            models.MemoryChunk.prefix_chars,
        )
        .join(models.MemoryChunk, models.MemoryChunk.memory_id == models.Memory.id)
        .where(models.Memory.is_current.is_(True), models.Memory.deleted_at.is_(None))
        .order_by(models.Memory.external_key, models.MemoryChunk.ordinal)
    )
    if memory_ids is not None:
        stmt = stmt.where(models.Memory.id.in_(list(memory_ids)))

    async with session_factory() as session:
        rows = list(await session.execute(stmt))

    mismatches: list[Mismatch] = []
    unverifiable = 0
    seen: set[UUID] = set()

    for (
        memory_id,
        external_key,
        memory_text,
        ordinal,
        chunk_text,
        char_start,
        char_end,
        prefix_chars,
    ) in rows:
        seen.add(memory_id)
        if memory_text is None:
            unverifiable += 1
            continue

        expected = chunk_text[prefix_chars:]
        actual = memory_text[char_start:char_end]
        if expected == actual:
            continue

        # Naming *why* rather than only that it differs. The three failures have
        # different causes: offsets past the end mean the text was rewritten
        # under the chunk, a length mismatch means the span moved, and equal
        # lengths with different content means the offsets drifted.
        if char_end > len(memory_text):
            reason = (
                f"span {char_start}..{char_end} runs past the memory's "
                f"{len(memory_text)} chars"
            )
        elif len(expected) != len(actual):
            reason = f"span is {len(actual)} chars, chunk body is {len(expected)}"
        else:
            reason = "same length, different text: offsets point elsewhere"

        mismatches.append(
            Mismatch(
                memory_id=memory_id,
                external_key=external_key,
                ordinal=ordinal,
                reason=reason,
                expected=expected,
                actual=actual,
            )
        )

    report = CitationReport(
        checked=len(rows),
        memories=len(seen),
        mismatches=mismatches,
        unverifiable=unverifiable,
    )
    logger.info(
        "citations.verified",
        checked=report.checked,
        memories=report.memories,
        mismatches=len(report.mismatches),
        unverifiable=report.unverifiable,
    )
    return report
