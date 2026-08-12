"""How one item changed, reconstructed from versions that were never overwritten.

**This milestone is nearly free because M1.1 decided not to update rows.** A
modified file produces a new artifact, a new event and a new memory version; the
old version keeps its bytes, its normalized text and both hashes. So the history
this reconstructs was already in the database, unread, from the first sync. The
work here is querying and presenting it.

Three things are worth stating before the code, because each one shapes what this
layer can honestly claim.

**The diff is over normalized text, never bytes.** `memories.content` is what
M1.4 produced, so a file saved with CRLF line endings — different bytes, new
artifact, new version, genuinely new row — diffs to nothing. That is the correct
answer and it is also a live check on normalization: if it ever stopped
collapsing line endings, every line of the file would report as changed.

**Superseded versions hold no chunks, by design.** `NormalizeMemory._store`
deletes the chunks of every earlier version of an item when it writes the new
ones, because chunks belonging to a version nobody can retrieve stay in the
vector index and keep surfacing stale text. The consequence for this module is
concrete: a diff can say which chunks of the *newer* version a change landed in,
and cannot say anything about the older one's, because they do not exist. A
chunk-count delta between a superseded version and its successor is therefore not
a measurement of chunking, and is reported as unavailable rather than as a number
that looks like one.

**Adoption is recoverable after the fact.** When two consecutive versions share a
`normalized_hash`, M1.4 moved the chunks across rather than rebuilding them —
the file changed, the text did not. Saying "chunks adopted, no semantic change"
is more honest than presenting an empty diff and letting a reader wonder whether
the diff failed.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import LanguageModel
from memoryos.domain.diffing import ChangeKind, Span, diff_spans, unified
from memoryos.domain.grounding import SummaryCheck, check_summary
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import PermanentError

logger = structlog.get_logger(__name__)

# Bumped when the prompt or the shape of the evidence changes. Part of the cache
# key, so an improved prompt is a re-run of the rows that carry the old value
# rather than an emptied table.
#
# **v2 exists because v1 was already wrong on this corpus.** The first summary
# this system generated claimed two import lines had been reordered; they appear
# in the diff only as unchanged context. Rule 4 was added to say so explicitly,
# and the version was *not* bumped at the time — so every read kept serving the
# v1 text from the cache, and the fix looked like it had not worked. That is
# exactly the failure this constant exists to prevent, met on the first
# opportunity. Change the prompt, change this line.
SUMMARIZER_VERSION = "change-v2"

# What a summary says when there is nothing to say. Returned *without calling the
# model*, which is the point: a model shown an empty diff and asked what changed
# will find something, because that is what it is for.
NO_SUBSTANTIVE_CHANGE = "No substantive change."

SYSTEM_PROMPT = """\
You describe what changed between two versions of one file, given only a diff.

Rules, in order of importance:

1. Describe ONLY what the diff shows. The diff is everything you have. You have \
not seen the rest of the file and must not write as though you had.
2. Do NOT explain why the change was made. You cannot know that. No "to improve", \
no "for clarity", no "as part of". Say what changed, not what it was for.
3. Use the diff's own names. Where it shows a function, a column, a constant or a \
path, use that exact name rather than a description of it.
4. Lines starting with `+` were added and lines starting with `-` were removed. \
Every other line is UNCHANGED context, shown only so you can locate the change. \
Never say an unchanged line was moved, reordered, or edited.
5. If the diff shows nothing of substance — whitespace, reordering, a comment \
reflow — reply with exactly: No substantive change.
6. One or two sentences. No preamble, no bullet list, no restating the filename.

Write plain prose."""

USER_TEMPLATE = """\
{diff}

---

In one or two sentences, what changed between these two versions?"""


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryVersion:
    """One version of one item, with what is knowable about it now.

    `chunks` is the count *currently stored against this version*, which for
    every superseded version is zero and is not a fact about how the text was
    chunked at the time. `holds_chunks` names that distinction so a caller does
    not have to know the deletion rule to read the number.
    """

    id: UUID
    version: int
    is_current: bool
    kind: str
    title: str | None
    content_hash: str
    normalized_hash: str | None
    occurred_at: datetime | None
    occurred_at_source: str
    ingested_at: datetime
    deleted_at: datetime | None
    chunks: int
    chunker_versions: tuple[str, ...]
    characters: int
    # Set against the immediately preceding version. None on the first.
    adopted: bool | None = None
    text_changed: bool | None = None
    bytes_changed: bool | None = None

    @property
    def holds_chunks(self) -> bool:
        return self.chunks > 0

    @property
    def summary_of_change(self) -> str:
        """The one-word account of this version's relationship to its predecessor."""
        if self.adopted is None:
            return "first version"
        if self.adopted:
            return "chunks adopted"
        return "rechunked"


@dataclass(frozen=True, slots=True)
class AffectedChunk:
    """A chunk of the newer version that a change landed inside."""

    id: UUID
    ordinal: int
    char_start: int
    char_end: int
    definition: str | None
    # How many of the diff's spans touch it. A chunk hit by four separate edits
    # is a different thing from one clipped at its edge by a neighbour's.
    spans: int


@dataclass(frozen=True, slots=True)
class VersionDiff:
    before: MemoryVersion
    after: MemoryVersion
    spans: list[Span] = field(default_factory=list)
    affected_chunks: list[AffectedChunk] = field(default_factory=list)
    # The text the model is shown, and the text the grounding check is run
    # against. Kept on the result rather than regenerated, so the two can never
    # be run against different evidence.
    unified_diff: str = ""

    @property
    def is_empty(self) -> bool:
        """No change in the normalized text.

        True for a line-ending-only edit, which is a real new version of a real
        new artifact whose text is identical.
        """
        return not self.spans

    @property
    def added_chars(self) -> int:
        return sum(span.added_chars for span in self.spans)

    @property
    def removed_chars(self) -> int:
        return sum(span.removed_chars for span in self.spans)

    @property
    def chunk_delta(self) -> int | None:
        """Change in chunk count, or None when it is not measurable.

        None whenever either side no longer holds its chunks, which in practice
        means every pair involving a superseded version. Returning
        `after.chunks - 0` there would print `+50` for a two-line edit.
        """
        if not self.before.holds_chunks or not self.after.holds_chunks:
            return None
        return self.after.chunks - self.before.chunks


@dataclass(frozen=True, slots=True)
class ChangeSummary:
    text: str
    model_id: str
    summarizer_version: str
    grounding: SummaryCheck
    # Whether this came out of the cache rather than out of the model. Surfaced
    # because "the summary did not change" and "the model was not asked" look
    # identical otherwise.
    cached: bool = False

    @property
    def trivial(self) -> bool:
        return self.grounding.trivial


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------


async def version_history(
    sessions: async_sessionmaker[AsyncSession], source_id: UUID, external_key: str
) -> list[MemoryVersion]:
    """Every version of one item, oldest first.

    Ascending, which is the order the history is read in and the reverse of what
    `/memories/{id}` returns for its list. The relationship fields — `adopted`,
    `text_changed`, `bytes_changed` — are each computed against the immediately
    preceding version and are therefore only meaningful in this order.
    """
    async with sessions() as session:
        rows = list(
            (
                await session.execute(
                    select(models.Memory)
                    .where(
                        models.Memory.source_id == source_id,
                        models.Memory.external_key == external_key,
                    )
                    .order_by(models.Memory.version)
                )
            ).scalars()
        )
        counts = await _chunk_counts(session, [row.id for row in rows])
        chunkers = await _chunker_versions(session, [row.id for row in rows])

    history: list[MemoryVersion] = []
    for index, row in enumerate(rows):
        previous = rows[index - 1] if index else None
        history.append(
            MemoryVersion(
                id=row.id,
                version=row.version,
                is_current=row.is_current,
                kind=row.kind,
                title=row.title,
                content_hash=row.content_hash,
                normalized_hash=row.normalized_hash,
                occurred_at=row.occurred_at,
                occurred_at_source=row.occurred_at_source,
                ingested_at=row.ingested_at,
                deleted_at=row.deleted_at,
                chunks=counts.get(row.id, 0),
                chunker_versions=chunkers.get(row.id, ()),
                characters=len(row.content or ""),
                # Identical normalized text across a version boundary is exactly
                # the condition `_adopt_from_previous_version` acts on.
                adopted=(
                    None
                    if previous is None
                    else row.normalized_hash is not None
                    and row.normalized_hash == previous.normalized_hash
                ),
                text_changed=(
                    None
                    if previous is None
                    else row.normalized_hash != previous.normalized_hash
                ),
                bytes_changed=(
                    None if previous is None else row.content_hash != previous.content_hash
                ),
            )
        )
    return history


async def diff_versions(
    sessions: async_sessionmaker[AsyncSession], memory_id_a: UUID, memory_id_b: UUID
) -> VersionDiff:
    """What changed between two versions of the same item.

    Refuses a pair from two different items rather than diffing them. Two
    unrelated files produce a large, well-formed, entirely meaningless diff, and
    the error is far more useful than the output.
    """
    async with sessions() as session:
        rows = {
            row.id: row
            for row in (
                await session.execute(
                    select(models.Memory).where(models.Memory.id.in_([memory_id_a, memory_id_b]))
                )
            ).scalars()
        }

    before = rows.get(memory_id_a)
    after = rows.get(memory_id_b)
    if before is None or after is None:
        missing = memory_id_a if before is None else memory_id_b
        raise PermanentError(f"no such memory: {missing}")
    if (before.source_id, before.external_key) != (after.source_id, after.external_key):
        raise PermanentError(
            "cannot diff two different items: "
            f"{before.external_key!r} against {after.external_key!r}"
        )

    history = {
        version.id: version
        for version in await version_history(sessions, before.source_id, before.external_key)
    }

    # `content` is null until M1.4 has normalized the row. Treated as empty
    # rather than refused, so a history containing an unnormalized version still
    # renders — with that version's diff showing the whole text as added, which
    # is what actually happened from this layer's point of view.
    a_text = before.content or ""
    b_text = after.content or ""
    spans = diff_spans(a_text, b_text)

    async with sessions() as session:
        affected = await _affected_chunks(session, after.id, spans)

    return VersionDiff(
        before=history[before.id],
        after=history[after.id],
        spans=spans,
        affected_chunks=affected,
        unified_diff=unified(
            a_text,
            b_text,
            a_label=f"{before.external_key}@v{before.version}",
            b_label=f"{after.external_key}@v{after.version}",
        ),
    )


# --------------------------------------------------------------------------
# Summarization
# --------------------------------------------------------------------------


class SummarizeChange:
    """A short description of a diff, cached on the version pair.

    **The trivial case never reaches the model, and that is the guardrail.** A
    model handed an empty diff and asked what changed will produce something,
    because producing something is what it does — and the something will be
    fluent and wrong. Deciding "nothing changed" from the spans, in code, is the
    only version of that answer that cannot be fabricated. Rule 4 of the prompt
    asks for the same string, and that is a second line of defence for the
    near-trivial diffs this cannot detect, not the first.

    The grounding check afterwards is M2.6's shape applied to different evidence.
    There are no passage numbers here, so what gets verified is vocabulary: every
    identifier the summary names must appear in the diff it was shown. See
    `domain.grounding.check_summary` for what that can and cannot catch.
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        model: LanguageModel,
        *,
        version: str = SUMMARIZER_VERSION,
        max_diff_chars: int = 12_000,
    ) -> None:
        self._sessions = sessions
        self._model = model
        self._version = version
        self._max_diff_chars = max_diff_chars

    async def __call__(self, diff: VersionDiff, *, refresh: bool = False) -> ChangeSummary:
        log = logger.bind(
            from_memory_id=str(diff.before.id), to_memory_id=str(diff.after.id)
        )

        if diff.is_empty:
            # Not cached. There is no model call to save, and a row here would
            # be a stored answer to a question answerable in constant time.
            log.info("evolution.trivial_diff")
            return ChangeSummary(
                text=NO_SUBSTANTIVE_CHANGE,
                model_id=self._model.model_id,
                summarizer_version=self._version,
                grounding=SummaryCheck(trivial=True),
            )

        if not refresh:
            cached = await self._cached(diff)
            if cached is not None:
                return cached

        evidence = self._evidence(diff)
        text = (
            await self._model.complete(
                SYSTEM_PROMPT, USER_TEMPLATE.format(diff=evidence), max_tokens=200
            )
        ).strip()

        grounding = check_summary(text, evidence)
        if grounding.context_only:
            # Not a failure, and worth seeing. The one observed fabrication in
            # this milestone was a claim about lines that appear only as
            # context — see `domain.grounding`.
            log.info("evolution.summary_names_context", terms=grounding.context_only)
        if not grounding.grounded:
            # Logged rather than suppressed. A summary naming something the diff
            # does not contain is the failure this check exists for, and hiding
            # it would leave the rate looking perfect.
            log.warning("evolution.summary_ungrounded", terms=grounding.unsupported)

        await self._store(diff, text, grounding, replace=refresh)
        return ChangeSummary(
            text=text,
            model_id=self._model.model_id,
            summarizer_version=self._version,
            grounding=grounding,
        )

    def _evidence(self, diff: VersionDiff) -> str:
        """The diff as the model sees it, truncated at a stated boundary.

        Truncation is announced in the text rather than done silently. A model
        shown the first half of a diff with no indication of that will describe
        it as the whole change, and the summary will be confidently incomplete.
        """
        text = diff.unified_diff
        if len(text) <= self._max_diff_chars:
            return text
        return (
            text[: self._max_diff_chars]
            + f"\n[diff truncated at {self._max_diff_chars} characters; "
            "more changed below this point]\n"
        )

    async def _cached(self, diff: VersionDiff) -> ChangeSummary | None:
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(models.ChangeSummary).where(
                        models.ChangeSummary.from_memory_id == diff.before.id,
                        models.ChangeSummary.to_memory_id == diff.after.id,
                        models.ChangeSummary.summarizer_version == self._version,
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return ChangeSummary(
            text=row.summary,
            model_id=row.model_id,
            summarizer_version=row.summarizer_version,
            grounding=SummaryCheck(
                terms=[],
                unsupported=[str(term) for term in row.unsupported_terms],
            ),
            cached=True,
        )

    async def _store(
        self, diff: VersionDiff, text: str, grounding: SummaryCheck, *, replace: bool
    ) -> None:
        """Write the summary, replacing an existing one only when asked to.

        The two paths differ for a reason. On the ordinary path a conflict means
        two concurrent requests for the same pair produced two equally valid
        summaries, and the first written is as good as the second — overwriting
        would change the text under a reader for no gain.

        `--refresh` is the opposite instruction, and this took a second attempt
        to get right: with `DO NOTHING` on both paths, a refresh regenerated the
        summary, paid for the call, returned the new text to its caller, and left
        the old row in place — so the next read served the stale one and the
        refresh appeared to have done nothing.
        """
        values = {
            "id": new_id(),
            "from_memory_id": diff.before.id,
            "to_memory_id": diff.after.id,
            "summarizer_version": self._version,
            "model_id": self._model.model_id,
            "summary": text,
            "grounded": grounding.grounded,
            "unsupported_terms": list(grounding.unsupported),
        }
        statement = insert(models.ChangeSummary).values(**values)
        async with self._sessions.begin() as session:
            await session.execute(
                statement.on_conflict_do_update(
                    constraint="uq_change_summaries_pair_version",
                    set_={
                        "model_id": statement.excluded.model_id,
                        "summary": statement.excluded.summary,
                        "grounded": statement.excluded.grounded,
                        "unsupported_terms": statement.excluded.unsupported_terms,
                        "created_at": func.now(),
                    },
                )
                if replace
                else statement.on_conflict_do_nothing(
                    constraint="uq_change_summaries_pair_version"
                )
            )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


async def _chunk_counts(session: AsyncSession, memory_ids: list[UUID]) -> dict[UUID, int]:
    if not memory_ids:
        return {}
    rows = await session.execute(
        select(models.MemoryChunk.memory_id, func.count())
        .where(models.MemoryChunk.memory_id.in_(memory_ids))
        .group_by(models.MemoryChunk.memory_id)
    )
    return {memory_id: count for memory_id, count in rows}


async def _chunker_versions(
    session: AsyncSession, memory_ids: list[UUID]
) -> dict[UUID, tuple[str, ...]]:
    if not memory_ids:
        return {}
    rows = await session.execute(
        select(models.MemoryChunk.memory_id, models.MemoryChunk.chunker_version)
        .where(models.MemoryChunk.memory_id.in_(memory_ids))
        .distinct()
    )
    found: dict[UUID, list[str]] = {}
    for memory_id, version in rows:
        found.setdefault(memory_id, []).append(version)
    return {memory_id: tuple(sorted(versions)) for memory_id, versions in found.items()}


async def _affected_chunks(
    session: AsyncSession, memory_id: UUID, spans: list[Span]
) -> list[AffectedChunk]:
    """Chunks of the newer version whose claimed span overlaps a change.

    Overlap is computed against `char_start`/`char_end`, which bound the span a
    chunk *claims* — not against its stored `content`, which additionally carries
    the overlap head borrowed from its predecessor. Using the text would report
    the chunk before an edit as affected too, every time, because it contains a
    copy of the edited region's opening.

    An `ADDED` span has zero width in the new text only when nothing was added,
    so a point insertion still overlaps the chunk it landed in. A zero-width span
    is treated as touching the chunk that contains its position.
    """
    if not spans:
        return []

    chunks = list(
        (
            await session.execute(
                select(models.MemoryChunk)
                .where(models.MemoryChunk.memory_id == memory_id)
                .order_by(models.MemoryChunk.ordinal)
            )
        ).scalars()
    )

    affected: list[AffectedChunk] = []
    for chunk in chunks:
        touching = sum(
            1
            for span in spans
            if _overlaps(span.b_start, span.b_end, chunk.char_start, chunk.char_end)
        )
        if touching:
            definition = chunk.meta.get("definition")
            affected.append(
                AffectedChunk(
                    id=chunk.id,
                    ordinal=chunk.ordinal,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    definition=definition if isinstance(definition, str) else None,
                    spans=touching,
                )
            )
    return affected


def _overlaps(start: int, end: int, chunk_start: int, chunk_end: int) -> bool:
    if start == end:
        # A pure deletion has no width in the new text. It still happened
        # somewhere, and the chunk containing that point is the one affected.
        return chunk_start <= start < chunk_end or (start == chunk_end == chunk_start)
    return start < chunk_end and end > chunk_start


__all__ = [
    "NO_SUBSTANTIVE_CHANGE",
    "SUMMARIZER_VERSION",
    "AffectedChunk",
    "ChangeKind",
    "ChangeSummary",
    "MemoryVersion",
    "SummarizeChange",
    "VersionDiff",
    "diff_versions",
    "version_history",
]
