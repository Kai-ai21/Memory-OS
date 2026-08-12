"""Merge entities that refer to the same thing.

**This is where the phase succeeds or fails.** Extraction is the easy half — a
model does it. An unresolved graph looks impressive in a picture and is useless
to traverse, because the path you need runs through a node that exists four
times under four spellings and therefore does not exist at all.

Three strategies, cheapest first, and the ordering is not just efficiency:

1. **Exact canonical match.** Free, and as close to certain as this gets: two
   names that reduce to the same canonical form under a type-specific rule are
   the same string wearing different punctuation.
2. **Embedding similarity.** Reuses the `Embedder` the corpus is already built
   with, and reaches pairs no character rule can — "Postgres"/"PostgreSQL"
   cannot be joined by any suffix rule that does not also mangle one of them.
   It **proposes and never decides**: measured on this corpus, no similarity
   threshold separates its true matches from its false ones. See
   `AUTO_MERGE_STRATEGIES`.
3. **Aliases.** A hand-written table for the pairs the first two cannot see.
   Small on purpose: an alias list that grows without bound is a resolver
   somebody is maintaining by hand instead of fixing.

**Blocking keeps this out of O(n²).** Only entities of the same type are ever
compared, and embedding comparison runs within a type over a single matrix
product rather than pair by pair. Comparing a PERSON against a FILE is work
spent to reach a conclusion the type system already had.

**The asymmetry that decides every threshold here: a false merge is worse than
a missed one.** A missed merge leaves a traversal short a path — visible, and
fixed by a later merge. A false merge invents a path that is not in the corpus,
and every traversal through it reports a connection nobody wrote. So the
auto-merge bar is high, and everything below it is recorded for a person rather
than guessed at.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from uuid import UUID

import structlog
from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.graph_sync import enqueue_sync
from memoryos.application.ports import Embedder, JobQueue
from memoryos.domain.canonicalize import canonicalize
from memoryos.domain.ids import new_id
from memoryos.domain.values import (
    EntityType,
    MergeStatus,
    MergeStrategy,
)

logger = structlog.get_logger(__name__)

# The confidence a candidate needs to be applied without review. Binding for
# `exact` and `alias`; embedding candidates are excluded by strategy regardless
# of what they score, so raising or lowering this does not admit them.
DEFAULT_THRESHOLD = 0.93

# Below this, a pair is not worth a person's attention. Everything between here
# and the threshold becomes a pending proposal.
REVIEW_FLOOR = 0.86

# Hand-curated pairs the first two strategies cannot reach. Keyed by type,
# because "go" is a technology and a verb and a great deal else.
#
# Deliberately tiny. Every entry here is a rule that could not be expressed, and
# a long list is a sign the expressible rules are wrong rather than that the
# corpus is unusual.
ALIASES: dict[EntityType, tuple[tuple[str, str], ...]] = {
    EntityType.TECHNOLOGY: (
        ("postgres", "postgresql"),
        ("py", "python"),
        # `sa` is the conventional import alias for SQLAlchemy and names the
        # same library. `op` is *not* the equivalent for Alembic — it is
        # Alembic's operations module, a part of the thing rather than the
        # thing — and it was in this list until the dry run showed it merging
        # them. Likewise `pgvector`, which is an extension Postgres loads, not
        # Postgres.
        ("sqlalchemy", "sa"),
    ),
}

# Strategies trusted to merge without a person looking.
#
# **Embedding similarity is deliberately absent, and that is a measured
# decision rather than caution.** The milestone proposes auto-merging above a
# high similarity threshold. On this corpus no such threshold exists: the dry
# run put `ck_memory_chunks_char_start_non_negative` and
# `ck_memory_chunks_prefix_chars_non_negative` — two different constraints — at
# 0.952, above `ingestion_events`/`IngestionEvent` at 0.939, which is a real
# match. A bi-encoder embeds identifier-like names by their shared prefix, and
# names that share a prefix are not names that share a referent.
#
# Any threshold therefore either admits false merges or excludes true ones, and
# given that a false merge invents a path the corpus does not contain, the
# resolution is to let embedding *propose* and never decide. It remains the
# strategy that reaches "Postgres"/"PostgreSQL" — it just reaches them into the
# review queue.
AUTO_MERGE_STRATEGIES: frozenset[MergeStrategy] = frozenset(
    {MergeStrategy.EXACT, MergeStrategy.ALIAS, MergeStrategy.MANUAL}
)

# Names too short or too generic to resolve on similarity alone. A two-character
# name has almost no signal in an embedding, and these collide with everything.
MIN_EMBEDDING_NAME_CHARS = 3


@dataclass(frozen=True, slots=True)
class MergeCandidate:
    """A proposal that two entities are one thing.

    `evidence` is a sentence a person can act on, not a number they have to
    trust. The pending queue is a review queue, and a reviewer shown only
    "0.91" has been asked to rubber-stamp rather than to judge.
    """

    left_id: UUID
    right_id: UUID
    strategy: MergeStrategy
    confidence: float
    evidence: str


@dataclass(slots=True)
class EntityRow:
    """An entity as the resolver sees it: identity, name, and weight."""

    id: UUID
    name: str
    canonical_name: str
    type: EntityType
    mentions: int
    first_seen_at: object


@dataclass(slots=True)
class ResolutionReport:
    candidates: int = 0
    auto_merged: int = 0
    pending: int = 0
    already_pending: int = 0
    by_strategy: dict[str, int] = field(default_factory=dict)
    mentions_moved: int = 0
    duration_ms: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return {
            "candidates": self.candidates,
            "auto_merged": self.auto_merged,
            "pending": self.pending,
            "mentions_moved": self.mentions_moved,
            "duration_ms": self.duration_ms,
        }


class ResolveEntities:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        queue: JobQueue | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        review_floor: float = REVIEW_FLOOR,
    ) -> None:
        self._sessions = session_factory
        self._embedder = embedder
        # A merge changes the graph, and this is where the change is announced.
        # Optional for the same reason it is everywhere else in Phase 3: the graph
        # is a projection, and a deployment without one still resolves entities
        # correctly — `graph verify` is what reports the projection is behind.
        self._queue = queue
        self._threshold = threshold
        self._review_floor = min(review_floor, threshold)

    async def propose(self) -> list[MergeCandidate]:
        """Every candidate the three strategies find, best first.

        Read-only. `--dry-run` is exactly this, printed — which is the point:
        the thing a person reviews before a live run must be the same thing the
        live run acts on, not a separate code path that resembles it.
        """
        rows = await self._active_entities()
        by_type: dict[EntityType, list[EntityRow]] = defaultdict(list)
        for row in rows:
            by_type[row.type].append(row)

        candidates: list[MergeCandidate] = []
        for entity_type, group in by_type.items():
            candidates.extend(_exact_candidates(group))
            candidates.extend(_alias_candidates(group, entity_type))
            candidates.extend(self._embedding_candidates(group))

        return _deduplicated(candidates)

    async def __call__(self, *, dry_run: bool = False) -> ResolutionReport:
        started = time.monotonic()
        report = ResolutionReport()

        candidates = await self.propose()
        report.candidates = len(candidates)
        for candidate in candidates:
            report.by_strategy[candidate.strategy.value] = (
                report.by_strategy.get(candidate.strategy.value, 0) + 1
            )

        if dry_run:
            report.auto_merged = sum(1 for c in candidates if self.would_auto_merge(c))
            report.pending = len(candidates) - report.auto_merged
            report.duration_ms = int((time.monotonic() - started) * 1000)
            return report

        # Applied one at a time, in confidence order, because merges interact:
        # merging A into B changes what C should merge into. Highest confidence
        # first means the surest decisions set the winners.
        for candidate in candidates:
            if self.would_auto_merge(candidate):
                moved = await self.apply(candidate)
                if moved is None:
                    continue
                report.auto_merged += 1
                report.mentions_moved += moved
            else:
                created = await self.record_pending(candidate)
                report.pending += int(created)
                report.already_pending += int(not created)

        report.duration_ms = int((time.monotonic() - started) * 1000)
        logger.info("resolve.finished", **report.as_dict())
        return report

    def would_auto_merge(self, candidate: MergeCandidate) -> bool:
        """Whether this candidate may be applied without review.

        Two conditions, and the strategy one does the real work: confidence
        alone cannot separate a true embedding match from a false one on this
        corpus. See `AUTO_MERGE_STRATEGIES`.
        """
        return (
            candidate.strategy in AUTO_MERGE_STRATEGIES
            and candidate.confidence >= self._threshold
        )

    # ------------------------------------------------------------------
    # Applying and undoing
    # ------------------------------------------------------------------

    async def apply(self, candidate: MergeCandidate) -> int | None:
        """Merge `right` into `left`, repointing mentions. Returns mentions moved.

        None when the merge is no longer applicable — either side may already
        have been merged away by an earlier, higher-confidence candidate in the
        same run. Skipping is correct: chains are resolved by following the
        pointer, not by merging a loser twice.
        """
        async with self._sessions.begin() as session:
            winner, loser = await self._resolve_pair(
                session, candidate.left_id, candidate.right_id
            )
            if winner is None or loser is None or winner == loser:
                return None

            moved = await _repoint(session, winner=winner, loser=loser)
            await session.execute(
                update(models.Entity)
                .where(models.Entity.id == loser)
                .values(merged_into_id=winner)
            )
            session.add(
                models.EntityMerge(
                    id=new_id(),
                    winner_id=winner,
                    loser_id=loser,
                    strategy=candidate.strategy.value,
                    status=MergeStatus.APPLIED.value,
                    confidence=candidate.confidence,
                    evidence=candidate.evidence,
                    moved_mention_ids=[str(value) for value in moved],
                    merged_at=func.now(),
                )
            )

        # After the commit. The sync reads Postgres, so a job queued inside the
        # transaction could be claimed by a worker before the merge was visible
        # and would project the state the merge was about to replace.
        #
        # Both ids, and the loser is the one that matters: the winner only gains
        # mentions, but the loser has to *leave* the graph, and only a payload
        # naming it will prune its node and the edges into it.
        await self._queue_sync(winner, loser)
        return len(moved)

    async def revert(self, merge_id: UUID) -> int:
        """Undo a merge, restoring exactly the mentions it moved.

        Exactly those, from `moved_mention_ids`, and not "every mention of the
        winner that looks like it came from the loser" — which is not
        recoverable information once the repoint has happened, and would take
        the winner's own mentions with it on any entity that had some.
        """
        async with self._sessions.begin() as session:
            merge = await session.get(models.EntityMerge, merge_id)
            if merge is None:
                raise LookupError(f"no such merge: {merge_id}")
            if merge.status != MergeStatus.APPLIED.value:
                raise ValueError(
                    f"merge {merge_id} is {merge.status}, so there is nothing to undo"
                )

            mention_ids = [UUID(str(value)) for value in merge.moved_mention_ids]
            if mention_ids:
                await session.execute(
                    update(models.EntityMention)
                    .where(models.EntityMention.id.in_(mention_ids))
                    .values(entity_id=merge.loser_id)
                )
            await session.execute(
                update(models.Entity)
                .where(models.Entity.id == merge.loser_id)
                .values(merged_into_id=None)
            )
            merge.status = MergeStatus.REVERTED.value
            merge.reverted_at = func.now()
            winner_id, loser_id = merge.winner_id, merge.loser_id

        # An unmerge is as much a graph change as a merge: the loser becomes a
        # node again and takes back the mentions that moved.
        await self._queue_sync(winner_id, loser_id)
        return len(mention_ids)

    async def _queue_sync(self, *entity_ids: UUID) -> None:
        if self._queue is None:
            return
        await enqueue_sync(self._queue, entity_ids=list(entity_ids))

    async def record_pending(self, candidate: MergeCandidate) -> bool:
        """Queue a candidate for review. False if it was already queued."""
        async with self._sessions.begin() as session:
            winner, loser = await self._resolve_pair(
                session, candidate.left_id, candidate.right_id
            )
            if winner is None or loser is None or winner == loser:
                return False

            result = await session.execute(
                insert(models.EntityMerge)
                .values(
                    id=new_id(),
                    winner_id=winner,
                    loser_id=loser,
                    strategy=candidate.strategy.value,
                    status=MergeStatus.PENDING.value,
                    confidence=candidate.confidence,
                    evidence=candidate.evidence,
                )
                # The index is partial, and Postgres will not infer a partial
                # index unless the predicate is repeated here — without it the
                # statement fails with "no unique or exclusion constraint
                # matching the ON CONFLICT specification" rather than silently
                # doing the wrong thing, which is the good version of this bug.
                .on_conflict_do_nothing(
                    index_elements=["winner_id", "loser_id"],
                    # A literal, not a bound parameter. Postgres matches
                    # `ON CONFLICT` to a partial index by comparing the
                    # predicate to the index's own, and it cannot prove
                    # `status = $8` equals `status = 'pending'` — the statement
                    # fails to plan at all. Comparing a column to an enum value
                    # produces exactly that bound parameter, which is why this
                    # is spelled out instead.
                    index_where=text(f"status = '{MergeStatus.PENDING.value}'"),
                )
                .returning(models.EntityMerge.id)
            )
            return result.scalar_one_or_none() is not None

    async def _resolve_pair(
        self, session: AsyncSession, left: UUID, right: UUID
    ) -> tuple[UUID | None, UUID | None]:
        """Follow both sides to their current winners and order them.

        Following the pointer is what makes a run over stale candidates safe: by
        the time a low-confidence pair is reached, either side may already have
        been merged by a surer one, and merging a loser again would build a
        chain nothing follows.
        """
        left_id = await _follow(session, left)
        right_id = await _follow(session, right)
        if left_id is None or right_id is None or left_id == right_id:
            return None, None

        winner, loser = await _pick_winner(session, left_id, right_id)
        return winner, loser

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _embedding_candidates(self, group: list[EntityRow]) -> list[MergeCandidate]:
        """Pairs whose names are close in the embedding space.

        One batched `embed_passage` per type and a matrix product, rather than a
        call per pair: the vectors are unit length, so cosine similarity is the
        inner product, and the whole comparison for a type is one multiplication.

        Names shorter than three characters are excluded. A two-character name
        carries almost no signal and sits near everything, which turns the
        threshold into a coin toss for exactly the entities — `op`, `sa`, `os` —
        that a code corpus produces most of.
        """
        eligible = [
            row for row in group if len(row.canonical_name) >= MIN_EMBEDDING_NAME_CHARS
        ]
        if len(eligible) < 2:
            return []

        vectors = self._embedder.embed_passage([row.name for row in eligible])
        candidates: list[MergeCandidate] = []

        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                score = sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
                if score < self._review_floor:
                    continue
                left, right = eligible[i], eligible[j]
                candidates.append(
                    MergeCandidate(
                        left_id=left.id,
                        right_id=right.id,
                        strategy=MergeStrategy.EMBEDDING,
                        confidence=min(1.0, max(0.0, score)),
                        evidence=(
                            f"cosine {score:.3f} between {left.name!r} and "
                            f"{right.name!r} ({left.type.value})"
                        ),
                    )
                )
        return candidates

    async def _active_entities(self) -> list[EntityRow]:
        """Entities that have not been merged away, with their mention counts.

        The count is what decides winners, so it is fetched here rather than
        per candidate — one aggregate instead of two queries per pair.
        """
        stmt = (
            select(
                models.Entity.id,
                models.Entity.name,
                models.Entity.canonical_name,
                models.Entity.type,
                models.Entity.first_seen_at,
                func.count(models.EntityMention.id),
            )
            .outerjoin(
                models.EntityMention,
                models.EntityMention.entity_id == models.Entity.id,
            )
            .where(models.Entity.merged_into_id.is_(None))
            .group_by(
                models.Entity.id,
                models.Entity.name,
                models.Entity.canonical_name,
                models.Entity.type,
                models.Entity.first_seen_at,
            )
        )
        async with self._sessions() as session:
            return [
                EntityRow(
                    id=row[0],
                    name=row[1],
                    canonical_name=row[2],
                    type=EntityType(row[3]),
                    first_seen_at=row[4],
                    mentions=row[5],
                )
                for row in await session.execute(stmt)
            ]


# ----------------------------------------------------------------------
# Strategies that need no I/O
# ----------------------------------------------------------------------


def _exact_candidates(group: list[EntityRow]) -> list[MergeCandidate]:
    """Entities of one type sharing a canonical form.

    Confidence 1.0, and it is not hedging: these are the same string once
    punctuation, case and type-specific decoration are removed. If this rule is
    wrong, `canonicalize` is wrong, and the fix belongs there rather than in a
    threshold.
    """
    buckets: dict[str, list[EntityRow]] = defaultdict(list)
    for row in group:
        buckets[canonicalize(row.name, row.type)].append(row)

    candidates: list[MergeCandidate] = []
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        anchor = members[0]
        for other in members[1:]:
            candidates.append(
                MergeCandidate(
                    left_id=anchor.id,
                    right_id=other.id,
                    strategy=MergeStrategy.EXACT,
                    confidence=1.0,
                    evidence=(
                        f"{anchor.name!r} and {other.name!r} both canonicalise to "
                        f"{key!r} ({anchor.type.value})"
                    ),
                )
            )
    return candidates


def _alias_candidates(
    group: list[EntityRow], entity_type: EntityType
) -> list[MergeCandidate]:
    """Hand-curated pairs, matched on canonical form so spelling varies freely."""
    pairs = ALIASES.get(entity_type, ())
    if not pairs:
        return []

    by_key: dict[str, list[EntityRow]] = defaultdict(list)
    for row in group:
        by_key[canonicalize(row.name, row.type)].append(row)

    candidates: list[MergeCandidate] = []
    for left_key, right_key in pairs:
        left_rows = by_key.get(left_key, [])
        right_rows = by_key.get(right_key, [])
        for left in left_rows:
            for right in right_rows:
                candidates.append(
                    MergeCandidate(
                        left_id=left.id,
                        right_id=right.id,
                        strategy=MergeStrategy.ALIAS,
                        # Below 1.0 deliberately: a human wrote this pair down
                        # once, for a corpus, and the entry can outlive the
                        # reason it was true.
                        confidence=0.99,
                        evidence=(
                            f"alias {left_key!r} = {right_key!r} matched "
                            f"{left.name!r} and {right.name!r}"
                        ),
                    )
                )
    return candidates


def _deduplicated(candidates: list[MergeCandidate]) -> list[MergeCandidate]:
    """One candidate per unordered pair, keeping the most confident.

    Strategies overlap by design — an alias pair is usually also embedding-close
    — and without this the same merge would be proposed twice and the review
    queue would show a person the same decision under two headings.
    """
    best: dict[frozenset[UUID], MergeCandidate] = {}
    for candidate in candidates:
        key = frozenset({candidate.left_id, candidate.right_id})
        current = best.get(key)
        if current is None or candidate.confidence > current.confidence:
            best[key] = candidate
    return sorted(best.values(), key=lambda c: (-c.confidence, c.evidence))


# ----------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------


async def _follow(session: AsyncSession, entity_id: UUID, depth: int = 8) -> UUID | None:
    """The entity this one was merged into, transitively.

    Bounded rather than trusted. The schema forbids self-merges and the resolver
    never merges a loser, but a chain is data and data can be wrong — an
    unbounded follow on a cycle is an infinite loop inside a transaction.
    """
    current = entity_id
    for _ in range(depth):
        row = await session.get(models.Entity, current)
        if row is None:
            return None
        if row.merged_into_id is None:
            return current
        current = row.merged_into_id
    logger.warning("resolve.merge_chain_too_deep", entity_id=str(entity_id))
    return None


async def _pick_winner(
    session: AsyncSession, left: UUID, right: UUID
) -> tuple[UUID, UUID]:
    """Which of two entities survives.

    Most mentions wins, because that entity is the one most of the corpus
    already points at and the merge that follows moves the fewest rows. Ties go
    to the older entity, then to the lower id — not because age is meaningful,
    but because the rule has to be total: a tie broken arbitrarily makes the
    whole resolution non-reproducible, and two runs over the same corpus would
    disagree about which name survived.
    """
    counts = {
        row[0]: row[1]
        for row in await session.execute(
            select(models.EntityMention.entity_id, func.count())
            .where(models.EntityMention.entity_id.in_([left, right]))
            .group_by(models.EntityMention.entity_id)
        )
    }
    rows = {
        entity_id: await session.get(models.Entity, entity_id)
        for entity_id in (left, right)
    }

    def rank(entity_id: UUID) -> tuple[int, float, str]:
        row = rows[entity_id]
        seen = getattr(row, "first_seen_at", None)
        return (
            -counts.get(entity_id, 0),
            seen.timestamp() if seen is not None else 0.0,
            str(entity_id),
        )

    winner, loser = sorted([left, right], key=rank)
    return winner, loser


async def _repoint(session: AsyncSession, *, winner: UUID, loser: UUID) -> list[UUID]:
    """Move the loser's mentions to the winner, returning the ids that moved.

    The returned ids are what makes the merge reversible, so they are collected
    before the update rather than inferred afterwards — afterwards, the
    information is gone.

    A mention that would collide with one the winner already has (same chunk,
    same offset) is deleted rather than moved, because `UNIQUE (entity_id,
    chunk_id, char_start)` forbids the row and the two mentions are genuinely
    the same sighting seen under two names.
    """
    moving = [
        row[0]
        for row in await session.execute(
            select(models.EntityMention.id).where(
                models.EntityMention.entity_id == loser
            )
        )
    ]
    if not moving:
        return []

    existing = {
        (row[0], row[1])
        for row in await session.execute(
            select(models.EntityMention.chunk_id, models.EntityMention.char_start).where(
                models.EntityMention.entity_id == winner
            )
        )
    }

    moved: list[UUID] = []
    for mention_id in moving:
        mention = await session.get(models.EntityMention, mention_id)
        if mention is None:
            continue
        if (mention.chunk_id, mention.char_start) in existing:
            await session.delete(mention)
            continue
        mention.entity_id = winner
        existing.add((mention.chunk_id, mention.char_start))
        moved.append(mention_id)

    return moved



# The graph rebuild a merge requires used to live here, and it is gone rather
# than moved: it projected memories, entities and `MENTIONS`, and nothing else.
#
# That was an M3.3 defect with no symptom short of reading the graph. Resolution
# clears the projection and rebuilds it — correct, because a merge removes a node
# and moves every edge that pointed at it — but the rebuild it ran predated
# `entity_relationships`, so every `RELATES_TO` edge M3.3 had extracted was
# deleted and never re-projected. Running `resolve-entities` silently emptied the
# relationship half of the graph, reported success, and left `doctor`'s node
# counts unchanged, because nothing there counts edges.
#
# `application/graph_projection.rebuild` is the one definition now, and it is
# shared with the sync, the divergence check and the replay. That is what makes
# the omission structurally impossible rather than fixed: a projection that
# forgot an edge type would fail `graph verify` against every corpus.
