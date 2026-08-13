"""Grouping assumptions that say the same thing in different words.

**This is what makes M5.3 possible.** A pattern is the same assumption failing
repeatedly, and "the same assumption" is not a string comparison: "this will take
two days", "the deploy is straightforward" and "integration should be quick" are
one recurring belief wearing three sentences. Ungrouped, each is a sample of size
one and no pattern can exist.

M3.2's machinery over a different column, and M3.2's caveats with it. One batched
`embed_passage` and a matrix product rather than a call per pair; the vectors are
unit length, so cosine similarity is the inner product.

**The asymmetry that sets every threshold here: a false grouping is worse than a
missed one**, and worse here than it was for entities. A missed group leaves two
beliefs looking unrelated — visible, and fixed by accepting a pending candidate.
A false group *invents a recurrence*: four members, one hold rate, and a
confident finding about how somebody estimates, assembled out of assumptions that
have nothing to do with each other. M5.3 reads exactly this table, so the auto
bar is high and everything under it waits for a person.

**And the failure mode is specific to this data.** Assumption statements are
full sentences in one voice about one project, so they are *all* fairly close in
the embedding space — far closer than entity names were. A threshold tuned by
intuition would group the corpus into one blob. The numbers below are set high
for that reason and the report prints the distribution, so somebody can see
whether they are separating anything.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import Embedder
from memoryos.domain.ids import new_id
from memoryos.domain.values import MergeStatus, MergeStrategy

logger = structlog.get_logger(__name__)

# Group without asking. Higher than M3.2's 0.93 for entities, and the reason is
# in the module docstring: these are full sentences in one voice about one
# project, so the whole population sits closer together than entity names did.
# A pair has to be near-paraphrase to clear this.
AUTO_THRESHOLD = 0.95

# Below this, a pair is not worth a person's attention. Everything between here
# and the threshold becomes a pending candidate.
REVIEW_FLOOR = 0.88

# Statements shorter than this are excluded from comparison entirely. A
# four-word assumption carries almost no signal and sits near everything, which
# turns the threshold into a coin toss — the same reason M3.2 excludes
# two-character entity names.
MIN_STATEMENT_CHARS = 20


@dataclass(frozen=True, slots=True)
class GroupCandidate:
    left_id: UUID
    right_id: UUID
    left_statement: str
    right_statement: str
    similarity: float

    @property
    def evidence(self) -> str:
        """What a reviewer is actually shown.

        The two sentences and the number, because "0.91" is not something a
        person can judge and "0.91 between 'deployment will take two days' and
        'the migration is a morning's work'" is exactly what they can.
        """
        return (
            f"cosine {self.similarity:.3f} between {self.left_statement!r} and "
            f"{self.right_statement!r}"
        )


@dataclass(slots=True)
class GroupingReport:
    assumptions: int = 0
    compared: int = 0
    auto_grouped: int = 0
    groups_created: int = 0
    queued: int = 0
    already_queued: int = 0
    # The top few scores that cleared neither bar. Printed rather than logged
    # away, because on a corpus where nothing groups the interesting question is
    # how close it came — and "the highest pair scored 0.83" is a much more
    # useful report than "0 groups".
    near_misses: list[tuple[float, str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "assumptions": self.assumptions,
            "compared": self.compared,
            "auto_grouped": self.auto_grouped,
            "groups_created": self.groups_created,
            "queued": self.queued,
            "already_queued": self.already_queued,
        }


@dataclass(frozen=True, slots=True)
class _Row:
    id: UUID
    statement: str
    group_id: UUID | None


class GroupAssumptions:
    """Propose and apply groupings over assumption statements."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        *,
        threshold: float = AUTO_THRESHOLD,
        review_floor: float = REVIEW_FLOOR,
    ) -> None:
        self._sessions = session_factory
        self._embedder = embedder
        self._threshold = threshold
        self._review_floor = review_floor

    async def propose(self) -> tuple[list[GroupCandidate], int]:
        """Every pair above the review floor, and how many were compared.

        Pairwise over the whole set rather than blocked by anything. M3.2 blocks
        by entity type because comparing a PERSON against a FILE spends work to
        reach a conclusion the type system already had; assumptions have no such
        partition, and with a corpus this size the full triangle is a few
        hundred comparisons over one matrix.
        """
        rows = await self._eligible()
        if len(rows) < 2:
            return [], len(rows)

        vectors = self._embedder.embed_passage([row.statement for row in rows])
        candidates: list[GroupCandidate] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                score = sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
                if score < self._review_floor:
                    continue
                candidates.append(
                    GroupCandidate(
                        left_id=rows[i].id,
                        right_id=rows[j].id,
                        left_statement=rows[i].statement,
                        right_statement=rows[j].statement,
                        similarity=min(1.0, max(0.0, score)),
                    )
                )
        candidates.sort(key=lambda candidate: -candidate.similarity)
        return candidates, len(rows)

    async def __call__(self, *, dry_run: bool = False) -> GroupingReport:
        """Group what clears the bar, queue what does not."""
        report = GroupingReport()
        rows = await self._eligible()
        report.assumptions = len(rows)
        report.compared = len(rows) * (len(rows) - 1) // 2

        candidates, _ = await self.propose()
        report.near_misses = await self._near_misses(rows)

        if dry_run:
            for candidate in candidates:
                if candidate.similarity >= self._threshold:
                    report.auto_grouped += 1
                else:
                    report.queued += 1
            logger.info("assumptions.grouping_dry_run", **report.as_dict())
            return report

        # Union-find over the pairs that cleared the bar, so a chain of three
        # near-paraphrases becomes one group rather than two overlapping pairs.
        # Applied in descending similarity, which only matters for the label:
        # the group takes its name from the strongest pair's left-hand side.
        parent: dict[UUID, UUID] = {}

        def find(node: UUID) -> UUID:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: UUID, right: UUID) -> None:
            parent[find(left)] = find(right)

        statements = {row.id: row.statement for row in rows}
        for candidate in candidates:
            if candidate.similarity >= self._threshold:
                union(candidate.left_id, candidate.right_id)

        clusters: dict[UUID, list[UUID]] = {}
        for node in list(parent):
            clusters.setdefault(find(node), []).append(node)

        async with self._sessions.begin() as session:
            for members in clusters.values():
                if len(members) < 2:
                    continue
                # The longest statement as the label: of several paraphrases the
                # fullest one is the most legible handle, and it is a real
                # sentence somebody wrote rather than a synthesised summary.
                label = max((statements[member] for member in members), key=len)
                group_id = new_id()
                session.add(
                    models.AssumptionGroup(
                        id=group_id,
                        label=label,
                        strategy=MergeStrategy.EMBEDDING.value,
                    )
                )
                await session.execute(
                    update(models.DecisionAssumption)
                    .where(models.DecisionAssumption.id.in_(members))
                    .values(group_id=group_id)
                )
                report.groups_created += 1
                report.auto_grouped += len(members)

        for candidate in candidates:
            if candidate.similarity >= self._threshold:
                continue
            if await self._queue(candidate):
                report.queued += 1
            else:
                report.already_queued += 1

        logger.info("assumptions.grouped", **report.as_dict())
        return report

    async def _near_misses(self, rows: Sequence[_Row]) -> list[tuple[float, str, str]]:
        """The highest-scoring pairs below the review floor.

        Only computed when nothing cleared it, which is the case where the
        number matters: a run reporting zero groups should say whether the
        corpus was close or nowhere near, because those call for different next
        steps — a lower threshold, or more decisions.
        """
        if len(rows) < 2:
            return []
        vectors = self._embedder.embed_passage([row.statement for row in rows])
        scored: list[tuple[float, str, str]] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                score = sum(a * b for a, b in zip(vectors[i], vectors[j], strict=True))
                scored.append((score, rows[i].statement, rows[j].statement))
        scored.sort(key=lambda item: -item[0])
        return scored[:5]

    async def _queue(self, candidate: GroupCandidate) -> bool:
        """Record a pending pair, or do nothing if it is already outstanding."""
        stmt = (
            insert(models.AssumptionGroupCandidate)
            .values(
                id=new_id(),
                left_id=candidate.left_id,
                right_id=candidate.right_id,
                similarity=candidate.similarity,
                status=MergeStatus.PENDING.value,
                model_id=self._embedder.model_id,
            )
            .on_conflict_do_nothing(
                index_elements=["left_id", "right_id"],
                index_where=text("status = 'pending'"),
            )
            .returning(models.AssumptionGroupCandidate.id)
        )
        async with self._sessions.begin() as session:
            return (await session.execute(stmt)).scalar_one_or_none() is not None

    async def _eligible(self) -> list[_Row]:
        stmt = (
            select(
                models.DecisionAssumption.id,
                models.DecisionAssumption.statement,
                models.DecisionAssumption.group_id,
            )
            .where(
                func.length(models.DecisionAssumption.statement) >= MIN_STATEMENT_CHARS
            )
            .order_by(models.DecisionAssumption.statement)
        )
        async with self._sessions() as session:
            return [
                _Row(id=row[0], statement=row[1], group_id=row[2])
                for row in await session.execute(stmt)
            ]


# --------------------------------------------------------------------------
# The review queue
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PendingPair:
    id: UUID
    left_id: UUID
    right_id: UUID
    left_statement: str
    right_statement: str
    left_question: str
    right_question: str
    similarity: float
    status: MergeStatus
    model_id: str
    proposed_at: datetime


async def list_candidates(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    status: MergeStatus | None = MergeStatus.PENDING,
    limit: int = 50,
) -> list[PendingPair]:
    """The queue, with both statements and both decisions.

    Both decisions, because that is what the reviewer is actually judging:
    whether two people-in-two-moments were expressing the same belief. Two
    identical sentences from one decision are a duplicate; from two decisions
    six weeks apart they are a recurrence, and only the second means anything.
    """
    left = models.DecisionAssumption.__table__.alias("left_assumption")
    right = models.DecisionAssumption.__table__.alias("right_assumption")
    left_decision = models.Decision.__table__.alias("left_decision")
    right_decision = models.Decision.__table__.alias("right_decision")

    stmt = (
        select(
            models.AssumptionGroupCandidate,
            left.c.statement,
            right.c.statement,
            left_decision.c.question,
            right_decision.c.question,
        )
        .join(left, left.c.id == models.AssumptionGroupCandidate.left_id)
        .join(right, right.c.id == models.AssumptionGroupCandidate.right_id)
        .join(left_decision, left_decision.c.id == left.c.decision_id)
        .join(right_decision, right_decision.c.id == right.c.decision_id)
        .order_by(models.AssumptionGroupCandidate.similarity.desc())
        .limit(limit)
    )
    if status is not None:
        stmt = stmt.where(models.AssumptionGroupCandidate.status == status.value)

    async with session_factory() as session:
        rows = list(await session.execute(stmt))

    return [
        PendingPair(
            id=row[0].id,
            left_id=row[0].left_id,
            right_id=row[0].right_id,
            left_statement=row[1],
            right_statement=row[2],
            left_question=row[3],
            right_question=row[4],
            similarity=row[0].similarity,
            status=MergeStatus(row[0].status),
            model_id=row[0].model_id,
            proposed_at=row[0].proposed_at,
        )
        for row in rows
    ]


class AlreadyReviewed(ValueError):
    """A candidate that has already been accepted or rejected."""


class UnknownCandidate(LookupError):
    """No candidate with that id."""


async def accept(
    session_factory: async_sessionmaker[AsyncSession], candidate_id: UUID
) -> UUID:
    """Put both assumptions in one group, creating it if neither has one.

    Formulated over the *pair* rather than over a group because at the first run
    there are no groups to join. Four cases and they all reduce to two: if
    either side is already grouped, the other joins that group; if both are
    grouped and differ, the smaller group's members move into the larger, which
    is the only choice that does not silently drop a membership.
    """
    async with session_factory.begin() as session:
        row = await session.get(models.AssumptionGroupCandidate, candidate_id)
        if row is None:
            raise UnknownCandidate(f"no candidate {candidate_id}")
        if row.status != MergeStatus.PENDING.value:
            raise AlreadyReviewed(f"candidate {candidate_id} was already {row.status}")

        left = await session.get(models.DecisionAssumption, row.left_id)
        right = await session.get(models.DecisionAssumption, row.right_id)
        if left is None or right is None:
            raise UnknownCandidate("one side of this pair no longer exists")

        group_id = left.group_id or right.group_id
        if group_id is None:
            group_id = new_id()
            session.add(
                models.AssumptionGroup(
                    id=group_id,
                    label=max(left.statement, right.statement, key=len),
                    # A person said yes. Recorded as `manual` even though an
                    # embedding proposed it, because what makes the group
                    # trustworthy is the judgement rather than the score — the
                    # same reason `MergeStrategy.MANUAL` outranks the rest in
                    # M3.2.
                    strategy=MergeStrategy.MANUAL.value,
                )
            )
        elif left.group_id and right.group_id and left.group_id != right.group_id:
            # Two existing groups the reviewer has just said are one. Everything
            # in the right-hand group moves left; the emptied group is left in
            # place rather than deleted, because deleting it would take its id
            # out of any candidate row still referencing it.
            await session.execute(
                update(models.DecisionAssumption)
                .where(models.DecisionAssumption.group_id == right.group_id)
                .values(group_id=left.group_id)
            )
            group_id = left.group_id

        left.group_id = group_id
        right.group_id = group_id
        row.status = MergeStatus.APPLIED.value
        row.reviewed_at = datetime.now(UTC)

    logger.info(
        "assumptions.group_accepted",
        candidate_id=str(candidate_id),
        group_id=str(group_id),
    )
    return group_id


async def reject(
    session_factory: async_sessionmaker[AsyncSession], candidate_id: UUID
) -> None:
    """Mark a pair as not the same belief. The row stays.

    Kept for the reasons every rejection in Phase 5 is kept: the pair is then
    excluded from the next run, and the count of rejections is the only
    measurement of how often the embedder proposes two beliefs that are not the
    same one.
    """
    async with session_factory.begin() as session:
        row = await session.get(models.AssumptionGroupCandidate, candidate_id)
        if row is None:
            raise UnknownCandidate(f"no candidate {candidate_id}")
        if row.status != MergeStatus.PENDING.value:
            raise AlreadyReviewed(f"candidate {candidate_id} was already {row.status}")
        # `reverted` is the enum's name for "considered and not in force", which
        # is what a rejected proposal is. A fourth status meaning the same thing
        # would split every query that asks what is outstanding.
        row.status = MergeStatus.REVERTED.value
        row.reviewed_at = datetime.now(UTC)
    logger.info("assumptions.group_rejected", candidate_id=str(candidate_id))
