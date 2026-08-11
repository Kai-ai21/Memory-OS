"""What extraction produced, and how badly it needs M3.2.

The duplicate count is the reason this module exists. Everything else here is a
`GROUP BY` that could live in the CLI; the duplicate analysis is a measurement
with a definition, and the definition is the argument for the next milestone.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models

# Suffixes that almost never distinguish two entities in this corpus: a trailing
# possessive, a plural, or a file extension picked up from a path. Stripped only
# for the *measurement*, never at write time — canonicalising this hard on the
# way in would shrink the number this exists to report.
_SUFFIXES = ("'s", "s", ".py", ".md", ".txt", ".json", ".yml", ".yaml")

# Everything that is not a letter or a digit. Punctuation is the single largest
# source of near-duplicates: "neo4j", "neo4j.", "(neo4j)" and "neo4j," are four
# rows and one thing.
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def collapse_key(canonical_name: str) -> str:
    """The stricter normalisation the duplicate count is measured against.

    Deliberately more aggressive than `canonical_entity_name`, which only
    casefolds and collapses whitespace. Two entities sharing a `collapse_key`
    differ *only* by case, punctuation, or one of the suffixes above — which is
    the milestone's definition of the duplicates M3.2 has to resolve.

    This is a measuring instrument, not a resolver. It cannot see that "Dr.
    Chen" and "Chen" are one person, or that "Postgres" and "PostgreSQL" are one
    technology; those need M3.2's judgement and are *not* counted here. So the
    number this produces is a floor on the duplicate problem, not an estimate of
    it — the easy half, countable exactly.
    """
    stripped = _NOT_ALNUM.sub("", canonical_name.lower())
    for suffix in _SUFFIXES:
        bare = _NOT_ALNUM.sub("", suffix.lower())
        if bare and stripped.endswith(bare) and len(stripped) > len(bare) + 2:
            return stripped[: -len(bare)]
    return stripped


@dataclass(slots=True)
class DuplicateGroup:
    key: str
    names: list[str]
    mentions: int

    @property
    def surplus(self) -> int:
        """Rows that would disappear if this group were one entity."""
        return len(self.names) - 1


@dataclass(slots=True)
class EntityStats:
    entities: int = 0
    mentions: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    top: list[tuple[str, str, int]] = field(default_factory=list)
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    extractor_versions: dict[str, int] = field(default_factory=dict)

    @property
    def duplicate_surplus(self) -> int:
        """How many entity rows are redundant under the collapse rule.

        The headline number for M3.2. Reported as surplus rows rather than as a
        group count because that is the size of the problem: 40 groups of two is
        40 wasted rows, and one group of 40 is 39.
        """
        return sum(group.surplus for group in self.duplicate_groups)

    @property
    def duplicate_rate(self) -> float:
        return self.duplicate_surplus / self.entities if self.entities else 0.0


async def gather_entity_stats(
    session_factory: async_sessionmaker[AsyncSession], *, top: int = 20
) -> EntityStats:
    async with session_factory() as session:
        stats = EntityStats(
            entities=(
                await session.execute(
                    select(func.count())
                    .select_from(models.Entity)
                    .where(models.Entity.merged_into_id.is_(None))
                )
            ).scalar_one(),
            mentions=(
                await session.execute(
                    select(func.count()).select_from(models.EntityMention)
                )
            ).scalar_one(),
        )

        stats.by_type = {
            row[0]: row[1]
            for row in await session.execute(
                select(models.Entity.type, func.count())
                .where(models.Entity.merged_into_id.is_(None))
                .group_by(models.Entity.type)
                .order_by(func.count().desc())
            )
        }

        stats.extractor_versions = {
            row[0]: row[1]
            for row in await session.execute(
                select(models.EntityMention.extractor_version, func.count())
                .group_by(models.EntityMention.extractor_version)
            )
        }

        # Most-mentioned, which is the list a person has to read and judge.
        stats.top = [
            (row[0], row[1], row[2])
            for row in await session.execute(
                select(
                    models.Entity.name,
                    models.Entity.type,
                    func.count(models.EntityMention.id).label("mentions"),
                )
                .join(
                    models.EntityMention,
                    models.EntityMention.entity_id == models.Entity.id,
                )
                .where(models.Entity.merged_into_id.is_(None))
                .group_by(models.Entity.id, models.Entity.name, models.Entity.type)
                .order_by(func.count(models.EntityMention.id).desc(), models.Entity.name)
                .limit(top)
            )
        ]

        stats.duplicate_groups = await _duplicates(session)

    return stats


async def _duplicates(session: AsyncSession) -> list[DuplicateGroup]:
    """Entities that collapse together under the stricter key.

    Grouped in Python rather than SQL because `collapse_key` is the definition
    of the measurement and belongs somewhere a reader can check it, not spread
    across a chain of `regexp_replace` calls in a query.
    """
    rows = await session.execute(
        select(
            models.Entity.canonical_name,
            models.Entity.type,
            func.count(models.EntityMention.id),
        )
        .outerjoin(
            models.EntityMention, models.EntityMention.entity_id == models.Entity.id
        )
        .where(models.Entity.merged_into_id.is_(None))
        .group_by(models.Entity.id, models.Entity.canonical_name, models.Entity.type)
    )

    # Keyed by type as well as name: a PERSON and a PROJECT sharing a name are
    # two different things, and collapsing them would be a resolution error
    # rather than a duplicate.
    grouped: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for canonical_name, entity_type, mentions in rows:
        grouped[(entity_type, collapse_key(canonical_name))].append(
            (canonical_name, mentions)
        )

    return sorted(
        (
            DuplicateGroup(
                key=f"{entity_type}:{key}",
                names=sorted(name for name, _ in members),
                mentions=sum(count for _, count in members),
            )
            for (entity_type, key), members in grouped.items()
            if len(members) > 1
        ),
        key=lambda group: (-group.surplus, -group.mentions, group.key),
    )
