"""Filing a memory under a word, using the vocabulary the corpus already has.

**A tag is a `CONCEPT` entity, not a tag.** That is the whole design and it is the
one decision here worth arguing about, so: the obvious implementation of tagging is
a `tags` table with names in it, and the obvious implementation is wrong. It would
create a second classification of the corpus that happens to use the same words as
the first. `#postgres` would sit in one vocabulary while the `postgres` concept
M3.1 extracted from eleven documents sat in another, nothing would connect them,
and the person who tagged three notes `#postgres` would find three notes — not the
eleven the corpus already knew about.

So tagging upserts into `entities`, through the same conflict-then-follow path
extraction uses, and M3.2's resolution then merges `#postgres` with `PostgreSQL`
without being told which of them was typed by a person. By the time resolution
runs there is no difference, because there was never a difference to preserve.

**What is stored is only the link, and it is stored by name on both ends.**
`memory_tags` carries `(source_id, external_key, tag)` and no foreign key into
either `memories` or `entities`, which looks like a missed constraint and is the
opposite:

* `memories` is truncated by every replay and its ids are minted fresh, and — more
  immediately — a correction creates version 2, so a tag keyed on a version id
  would stop applying the moment somebody fixed a typo.
* `entities` is truncated by every replay *and* refilled only by extraction, which
  costs money per chunk. A cascading key there would delete every tag anybody had
  applied on the next rebuild; a nullable one would leave rows pointing nowhere.

Names outlive ids. `reconcile` re-upserts the concepts from the names after a
rebuild, which is possible precisely because the name is what was kept.

**Where the connection is visible, and where it is not yet.** Because a tag and an
extracted concept are one entity row, `chat.status` reports a tagged memory as
connected to everything that *mentions* the concept — tag three notes `#postgres`
and the connection line names the eleven documents the corpus already knew about,
which is the whole point of not building a second vocabulary. What a tag does not
have is an edge in the Neo4j projection. `MENTIONS` carries a chunk and character
offsets, written only after the extractor confirmed the text really says the name,
and a tag appears nowhere in the text; a mention with a fabricated offset or no span
would weaken that chain for every real mention. A `TAGGED` edge is the right shape
and needs `graph_sync.expand` to close scopes over tags too, which M10.4 did not
do. Named in `EdgeType` as an absence rather than declared and left unwritten.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.domain.ids import new_id
from memoryos.domain.normalization import canonical_entity_name
from memoryos.domain.values import EntityType

logger = structlog.get_logger(__name__)

# What `#…` may contain. Letters, digits, dash and underscore, and deliberately no
# punctuation beyond those two.
#
# Conservative on purpose, twice over. A tag is stored casefolded and the
# `memory_tags` CHECK asserts it, so the character set has to be one where
# casefolding cannot produce something `lower()` disagrees with. And the sigil is
# typed in the middle of prose — "#idea, and see #queue-depth." — so anything that
# swallowed trailing punctuation would file two notes under `idea,` and `idea`.
#
# Unicode letters are in, because a tag is somebody's own word for something and
# restricting it to ASCII would be an English-only product decision made in a
# regex.
_TAG = re.compile(r"#(\w[\w-]*)", re.UNICODE)

# How many tags one message may apply at once. Not a storage limit — the table
# would hold thousands — but the point where tagging stops being organisation and
# becomes a second copy of the text. Ten words about one thought is a description;
# fifty is noise that makes every filter useless.
MAX_TAGS = 10


class NoTags(ValueError):
    """A tag command with nothing tag-shaped in it."""


class TooManyTags(ValueError):
    """More tags than one item can usefully carry."""


@dataclass(frozen=True, slots=True)
class Tag:
    """One tag: the word as typed, and the word as stored.

    Both, because they answer different questions. `label` is what somebody wants
    to read back — `#PostgreSQL` — and `name` is what joins to
    `entities.canonical_name`, so `#PostgreSQL` and `#postgresql` are one tag with
    whichever label arrived first.
    """

    name: str
    label: str

    @property
    def display(self) -> str:
        return f"#{self.label}"


@dataclass(frozen=True, slots=True)
class TagReport:
    """What one tagging did."""

    applied: tuple[Tag, ...]
    # Tags that were already on this item. Reported rather than silently absorbed:
    # the unique constraint makes re-tagging free, and a person who typed the same
    # tag twice should be told it was already there rather than shown a success
    # indistinguishable from the first time.
    already: tuple[Tag, ...] = ()
    # Concept entities this created, as opposed to joined. The interesting number
    # is the other one — a tag that resolved to an existing concept is a tag that
    # connected to something.
    entities_created: int = 0


def parse(text: str) -> list[Tag]:
    """Every `#tag` in a string, deduplicated, in the order they were typed.

    Order preserved because a person listing tags is listing them by importance
    more often than not, and deduplicated because `#idea #idea` is one tag. Both
    are `dict.fromkeys` rather than a set, which is the only ordered-dedup idiom
    that does not need a comment saying so.
    """
    found: dict[str, Tag] = {}
    for match in _TAG.finditer(text):
        label = match.group(1)
        name = canonical_entity_name(label)
        if not name:
            continue
        found.setdefault(name, Tag(name=name, label=label))
    return list(found.values())


def parse_required(text: str) -> list[Tag]:
    """`parse`, refusing an empty result and an unusable one.

    Named separately so the read paths — filtering, rendering — can parse
    permissively while the write path refuses. A `/tag` command with no tags in it
    is a typo, and doing nothing quietly is how somebody discovers next week that
    none of it was filed.
    """
    tags = parse(text)
    if not tags:
        raise NoTags(
            "no tags found. A tag is a word after a '#', like '#idea' or "
            "'#queue-depth'."
        )
    if len(tags) > MAX_TAGS:
        raise TooManyTags(
            f"{len(tags)} tags in one message, and {MAX_TAGS} is the limit. Past "
            f"that a tag list is a second copy of the text rather than a way to "
            f"find it again."
        )
    return tags


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


async def apply(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    source_id: UUID,
    external_key: str,
    tags: Sequence[Tag],
) -> TagReport:
    """Tag one item, creating the concepts the tags name.

    One transaction for the entities and the links, so a tag row can never point at
    a concept that was rolled back.

    No graph sync is enqueued here, and that is a considered omission rather than
    an oversight. A `TAGGED` edge is projected from these rows, so the graph is
    behind until the next sync — and the next sync is cheap and frequent, while
    the cost of being wrong is that a tag connects a moment late. The alternative
    is a job per tag on a path somebody uses interactively, which would put the
    graph on the critical path of typing `#idea`.
    """
    applied: list[Tag] = []
    already: list[Tag] = []
    created = 0

    async with session_factory.begin() as session:
        for tag in tags:
            _, is_new = await ensure_concept(session, tag)
            created += int(is_new)
            inserted = await session.execute(
                pg_insert(models.MemoryTag)
                .values(
                    id=new_id(),
                    source_id=source_id,
                    external_key=external_key,
                    tag=tag.name,
                    label=tag.label,
                )
                # Tagging the same item twice is one tag. The constraint is what
                # makes this idempotent without a check-then-insert, and the
                # returning clause is what tells the two cases apart.
                .on_conflict_do_nothing(constraint="uq_memory_tags_item_tag")
                .returning(models.MemoryTag.id)
            )
            (applied if inserted.scalar_one_or_none() else already).append(tag)

    logger.info(
        "tags.applied",
        key=external_key,
        applied=[tag.name for tag in applied],
        already=[tag.name for tag in already],
        entities_created=created,
    )
    return TagReport(
        applied=tuple(applied), already=tuple(already), entities_created=created
    )


async def remove(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    source_id: UUID,
    external_key: str,
    tags: Sequence[Tag],
) -> tuple[Tag, ...]:
    """Untag one item.

    The concept entity is left alone, and deliberately. It may have been created by
    this tag, but it may equally be mentioned in forty documents — and even when it
    is not, an entity with no mentions is already invisible to every read in the
    system, which joins through mentions. Deleting it would be this function
    reaching into the shared vocabulary to remove a word because one person stopped
    filing things under it.
    """
    if not tags:
        return ()
    async with session_factory.begin() as session:
        removed = (
            await session.execute(
                delete(models.MemoryTag)
                .where(
                    models.MemoryTag.source_id == source_id,
                    models.MemoryTag.external_key == external_key,
                    models.MemoryTag.tag.in_([tag.name for tag in tags]),
                )
                .returning(models.MemoryTag.tag, models.MemoryTag.label)
            )
        ).all()
    logger.info("tags.removed", key=external_key, tags=[row[0] for row in removed])
    return tuple(Tag(name=row[0], label=row[1]) for row in removed)


async def ensure_concept(
    session: AsyncSession, tag: Tag
) -> tuple[UUID, bool]:
    """The `CONCEPT` entity for this tag, creating it if the corpus has none.

    A copy of extraction's `_upsert_entity` in shape, and the shape is the part
    that matters rather than the code: insert with `ON CONFLICT DO NOTHING`, then
    select, then *follow `merged_into_id`*. That last step is the one that is easy
    to leave out and expensive to omit — M3.2 merges duplicates, and an upsert that
    stopped at the loser row would re-attach to an entity resolution had already
    retired, silently undoing the merge and drifting the entity count upwards.

    `confidence` is null rather than 1.0. The column is the extractor's confidence
    that the entity *exists*, and a person typing `#postgres` is not an extractor
    making an estimate — writing a maximum there would put a fabricated measurement
    in a column that feeds ranking.
    """
    inserted = await session.execute(
        pg_insert(models.Entity)
        .values(
            id=new_id(),
            # The surface form as first seen, which for a tag is what was typed.
            name=tag.label,
            canonical_name=tag.name,
            type=EntityType.CONCEPT.value,
            confidence=None,
        )
        .on_conflict_do_nothing(constraint="uq_entities_canonical_type")
        .returning(models.Entity.id)
    )
    entity_id = inserted.scalar_one_or_none()
    if entity_id is not None:
        return entity_id, True

    existing = (
        await session.execute(
            select(models.Entity.id, models.Entity.merged_into_id).where(
                models.Entity.canonical_name == tag.name,
                models.Entity.type == EntityType.CONCEPT.value,
            )
        )
    ).one()
    if existing[1] is None:
        return existing[0], False
    return await _follow(session, existing[1]), False


async def _follow(session: AsyncSession, entity_id: UUID, depth: int = 8) -> UUID:
    """The winner at the end of a merge chain.

    Bounded rather than recursive-until-null. The schema forbids a self-merge and
    resolution writes chains rather than cycles, but a bound is what keeps a
    corrupt pointer from hanging an interactive request — and eight is far past any
    chain resolution produces.
    """
    current = entity_id
    for _ in range(depth):
        found = (
            await session.execute(
                select(models.Entity.merged_into_id).where(
                    models.Entity.id == current
                )
            )
        ).scalar_one_or_none()
        if found is None:
            return current
        current = found
    logger.warning("tags.merge_chain_too_long", entity_id=str(entity_id))
    return current


async def reconcile(session_factory: async_sessionmaker[AsyncSession]) -> int:
    """Re-create the `CONCEPT` entity behind every tag that has one missing.

    **This exists because `memory_tags` survives a replay and `entities` does
    not.** A rebuild truncates the entity tables and does not refill them —
    extraction is an LLM call per chunk and a replay does not pay it — so after one,
    every tag is a name whose concept is gone. The tag still filters, because the
    filter is by name; what it stops doing is connecting, because the edge is
    projected through the entity.

    Idempotent, and cheap enough to run unconditionally: one upsert per *distinct*
    tag, which is the size of somebody's vocabulary rather than the size of the
    corpus.
    """
    async with session_factory.begin() as session:
        rows = (
            await session.execute(
                select(models.MemoryTag.tag, func.min(models.MemoryTag.label))
                .group_by(models.MemoryTag.tag)
                .order_by(models.MemoryTag.tag)
            )
        ).all()
        created = 0
        for name, label in rows:
            _, is_new = await ensure_concept(session, Tag(name=name, label=label))
            created += int(is_new)
    if created:
        logger.info("tags.reconciled", created=created, distinct=len(rows))
    return created


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


async def for_items(
    session: AsyncSession, items: Sequence[tuple[UUID, str]]
) -> dict[tuple[UUID, str], list[Tag]]:
    """Tags on each of these items, in one query.

    Keyed by `(source_id, external_key)` because that is what a tag is attached to,
    and returned as a dict so a caller rendering a hundred messages does a query
    for the page rather than one per row.
    """
    if not items:
        return {}
    rows = (
        await session.execute(
            select(
                models.MemoryTag.source_id,
                models.MemoryTag.external_key,
                models.MemoryTag.tag,
                models.MemoryTag.label,
            )
            .where(
                models.MemoryTag.source_id.in_({item[0] for item in items}),
                models.MemoryTag.external_key.in_({item[1] for item in items}),
            )
            .order_by(models.MemoryTag.tag)
        )
    ).all()
    grouped: dict[tuple[UUID, str], list[Tag]] = {}
    wanted = set(items)
    for source_id, key, name, label in rows:
        # Filtered here rather than in SQL. The two `IN`s above are a cross product
        # of sources and keys, which over-matches only when two sources share an
        # external key — real, and rare enough that one round trip beats a tuple
        # `IN` with a hundred pairs in it.
        if (source_id, key) in wanted:
            grouped.setdefault((source_id, key), []).append(
                Tag(name=name, label=label)
            )
    return grouped


async def all_tags(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[tuple[Tag, int]]:
    """Every tag in use and how many items carry it, most-used first.

    The vocabulary somebody has actually built, which is what a filter control
    offers rather than a free-text box: a tag misremembered as `#ideas` returns
    nothing and looks like a broken filter.
    """
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(
                    models.MemoryTag.tag,
                    func.min(models.MemoryTag.label),
                    func.count(models.MemoryTag.id),
                )
                .group_by(models.MemoryTag.tag)
                .order_by(func.count(models.MemoryTag.id).desc(), models.MemoryTag.tag)
            )
        ).all()
    return [(Tag(name=row[0], label=row[1]), int(row[2])) for row in rows]


async def items_with(
    session: AsyncSession, tags: Sequence[Tag]
) -> list[tuple[UUID, str]]:
    """The items carrying *every* one of these tags.

    Conjunction rather than disjunction, and that is the useful default: two tags
    is a person narrowing, not widening. `#idea #postgres` asks for the ideas about
    Postgres, and returning their union would return more than either tag alone —
    the opposite of what adding a second filter is for.
    """
    if not tags:
        return []
    stmt = (
        select(models.MemoryTag.source_id, models.MemoryTag.external_key)
        .where(models.MemoryTag.tag.in_([tag.name for tag in tags]))
        .group_by(models.MemoryTag.source_id, models.MemoryTag.external_key)
        .having(func.count(func.distinct(models.MemoryTag.tag)) == len(tags))
    )
    return [(row[0], str(row[1])) for row in (await session.execute(stmt)).all()]
