"""Which sources exist, and which one a name means.

**The second thing M7.0 went looking for and did not find.** Every scope in
Phase 4 is a source id — `SourceScope(source_id)`, `activity_by_period(...,
source_id=...)` — and every caller receives a source *name*, because that is
what a person types and what a model writes. The translation existed three
times: once in `api/routes/timeline.py` as `_resolve_source`, once inline in the
CLI's gaps command, and it would have been a fourth here.

Three copies of six lines is not a maintenance problem at this size. It is a
layering signal, and the milestone asked for those to be reported rather than
quietly made four: a use case that takes an id and a transport that has a name
means the resolution belongs between them, and nothing was between them.

`SqlAlchemySourceRepository.get_by_name` is the neighbouring function and cannot
be used for this: it takes a `SourceKind` as well, because it answers the
ingestion path's question — "is this filesystem source already registered" —
where a name alone is ambiguous by design. A reader naming a source does not
know its kind and should not have to.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models


class UnknownSource(LookupError):
    """A name no source has. Carries the known names, because the caller is
    usually one keystroke away and a list is the difference between correcting
    it and guessing again."""

    def __init__(self, name: str, known: list[str]) -> None:
        listed = ", ".join(sorted(known)) or "none registered"
        super().__init__(f"no source named {name!r}; known sources are: {listed}")
        self.name = name
        self.known = known


@dataclass(frozen=True, slots=True)
class SourceRef:
    id: UUID
    name: str


async def names(sessions: async_sessionmaker[AsyncSession]) -> dict[UUID, str]:
    """Every source id to its name, for labelling rows that carry only the id."""
    async with sessions() as session:
        return {
            row[0]: row[1]
            for row in await session.execute(
                select(models.Source.id, models.Source.name)
            )
        }


async def listed(sessions: async_sessionmaker[AsyncSession]) -> list[SourceRef]:
    """Every source, by name, for a caller offering a choice."""
    async with sessions() as session:
        rows = await session.execute(
            select(models.Source.id, models.Source.name).order_by(models.Source.name)
        )
    return [SourceRef(id=row[0], name=row[1]) for row in rows]


async def resolve(
    sessions: async_sessionmaker[AsyncSession], name: str | None
) -> UUID | None:
    """One source id from a name. `None` means "all sources", not "not found".

    The distinction is the reason this returns an optional rather than raising
    on an empty name: every caller has a corpus-wide mode, and making them each
    write the `if name is None` is how one of them ends up asking for a source
    called `""`.
    """
    if name is None:
        return None
    async with sessions() as session:
        found = (
            await session.execute(
                select(models.Source.id).where(models.Source.name == name)
            )
        ).scalars().first()
    if found is None:
        raise UnknownSource(name, [source.name for source in await listed(sessions)])
    return found
