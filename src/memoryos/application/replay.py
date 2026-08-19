"""Rebuild the corpus from the log and the blobs.

The claim M1.1 made in a docstring and every milestone since has repeated:
`ingestion_events` plus the blob store is the whole truth, and everything else is
a projection that can be thrown away. This is where that claim is either true or
it is not.

Which tables are which is written down below as data rather than described in
prose, because a table added to the wrong set breaks the guarantee silently —
nothing errors, the rebuild simply loses a category of information. A test
asserts every table in `Base.metadata` appears in exactly one of the three sets,
so adding a table without classifying it fails the build.

There are three sets and not two. M1.7 shipped with two and said in its own
report that the binary hid a third category; `query_judgements` is what made that
concrete. Human-authored data is neither rebuildable nor part of ingestion, and
the rule it needs — never truncated, never written by a replay — is not the rule
either other set carries.

Determinism is the property that makes a rebuild worth anything. Nothing derived
here reads the clock: `ingested_at` and `deleted_at` come from the causing
event's `recorded_at`, versions come from the order of the log, and the memory
itself comes from `projection.memory_from_event` — the same function `sync` uses,
so "exactly as sync would have" is structural rather than a promise. The one
exception is `memory_chunks.embedded_at`, which records when a vector was
computed rather than anything about the data, and legitimately differs after a
recomputation. It is not compared by `verify-replay` for that reason.
"""

import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import (
    ColumnElement,
    Select,
    Text,
    cast,
    delete,
    func,
    select,
    text,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import BlobNotFound
from memoryos.adapters.db import models
from memoryos.adapters.db.mappers import to_event
from memoryos.adapters.db.repositories import SqlAlchemyMemoryRepository
from memoryos.application import graph_projection
from memoryos.application.embed import EmbedMemory
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.ports import BlobStore, GraphStore, ShadowWorkspace
from memoryos.application.projection import memory_from_event, recorded_at_of
from memoryos.domain.entities import IngestionEvent
from memoryos.domain.ids import new_id
from memoryos.domain.values import EventType

logger = structlog.get_logger(__name__)

# --------------------------------------------------------------------------
# What is derived, explicitly
# --------------------------------------------------------------------------

# Never truncated, by anything, ever. These hold information that exists nowhere
# else: the bytes observed at a source, and the ordered record of observing them.
# `sources` is here rather than in the derived set even though its `cursor` and
# `last_sync_at` are written by the sync pipeline — those describe a connector's
# progress through the outside world, which no amount of replaying the log can
# reconstruct, because the outside world is not in the log.
SOURCE_OF_TRUTH_TABLES: frozenset[str] = frozenset(
    {
        "sources",
        "raw_artifacts",
        "ingestion_events",
    }
)

# Written by a person, reconstructible by nobody.
#
# M1.7 shipped with two sets and noted in its report that the binary was one set
# short — `jobs` is classified derived and truncating it works, but it is
# *discardable*, not reconstructible, which is a different property that happened
# to be safe. `query_judgements` is where that gap stops being theoretical. It is
# not derived, because no amount of replaying produces somebody's opinion about a
# search result. It is not source-of-truth-for-ingestion either: it describes the
# corpus rather than feeding it, and a replay does not read it.
#
# The operational rule is stronger than for either other set: replay must not
# truncate it *and* must not write it. It is the input to M2.0's evaluation
# harness, so losing it means losing the labelled data the next milestone is
# measured against.
#
# M5.0 adds four more, and they are the same category rather than a new one. A
# decision is somebody's account of a choice they made — the question, the
# alternatives, the confidence they held at the time — and no amount of
# replaying the log produces it. `decision_suggestions` belongs here for the
# reason `entity_merges` was argued about and lost: it carries a person's accept
# or reject, and unlike the merge ledger it has no foreign key forcing it into
# the derived set. Its provenance is a natural key plus id snapshots, exactly as
# `query_judgements` is, which is what lets it be classified by argument.
#
# `decision_evidence` is the uncomfortable member, and its discomfort is stated
# rather than hidden. It *does* have foreign keys into `memories` and
# `memory_chunks`, with ON DELETE CASCADE, because a link to a document that no
# longer exists is a citation to nothing. So `TRUNCATE memories CASCADE` takes
# it, whatever this set says — the same finding M1.7 made when the golden set
# was specified with a foreign key. The difference is that this table was
# designed knowing it: every row also carries `(source_name, external_key,
# chunk_ordinal)`, and `_preserve_evidence` below reads those out before the
# truncation and writes them back afterwards. Classified user-authored because
# that is what it is; protected by a snapshot because a classification cannot
# outvote a foreign key.
# The tables M11.1 gave a `user_id` and a row-level policy. Named here rather
# than only in migration 0032 because the shadow workspace has to re-apply the
# policies after a swap-in, and a second hand-kept copy of this list is a list
# that goes stale the first time somebody adds a table.
SCOPED_TABLES: frozenset[str] = frozenset(
    {
        "sources",
        "memories",
        "memory_chunks",
        "ingestion_events",
        "entities",
        "entity_mentions",
        "entity_relationships",
        "decisions",
        "query_judgements",
        "chat_sessions",
        "chat_messages",
        "jobs",
        "patterns",
        "user_model_facets",
    }
)


USER_AUTHORED_TABLES: frozenset[str] = frozenset(
    {
        # M11.0's two, and the least arguable entries in this set. Nothing in
        # the ingestion log produces an account or a login — no amount of
        # replaying documents mints a password hash — so a replay that truncated
        # these would lock the operator out of their own system and leave no way
        # back in except the CLI. Neither table has a foreign key into anything
        # derived, so unlike the evidence tables they need no snapshot: `users`
        # reaches nothing, and `sessions` reaches only `users`.
        "users",
        "sessions",
        "query_judgements",
        "decisions",
        "decision_options",
        "decision_assumptions",
        "decision_evidence",
        "decision_suggestions",
        # M5.1's three, and the same argument a third time. An outcome is
        # somebody's account of what happened, or a reading somebody accepted;
        # neither is in the log. `outcome_evidence` is `decision_evidence`'s twin
        # in every respect including the discomfort — it holds cascading foreign
        # keys into `memories`, so it is protected by the snapshot below rather
        # than by its classification.
        "decision_outcomes",
        "outcome_evidence",
        "outcome_suggestions",
        # M5.2's three. An evaluation is somebody's judgement about whether a
        # belief held; a group is somebody's judgement that two beliefs are the
        # same one. Neither is in the log. `assumption_groups` and
        # `assumption_group_candidates` reach `decision_assumptions` rather than
        # anything derived, so unlike the evidence tables they need no snapshot.
        "assumption_groups",
        "assumption_group_candidates",
        "assumption_evidence",
        # M5.3's two, and the most arguable classification in this file since
        # `entity_merges`.
        #
        # Most of `patterns` is genuinely derived: the statements, the counts
        # and the confidences are a pure function of the decisions, assumptions
        # and outcomes, and `patterns discover` reproduces them exactly. By that
        # test alone it belongs in the derived set.
        #
        # It is here anyway because of two columns. `dismissed_at` and
        # `dismissed_reason` are a person having read a behavioural claim about
        # themselves and refused it, and nothing in the log reproduces that —
        # a replay that truncated this table would quietly un-reject every
        # pattern somebody had rejected, and the next `discover` would put them
        # all back. That is precisely the property `USER_AUTHORED` exists for.
        #
        # Unlike the three evidence tables, no snapshot is needed: every foreign
        # key `pattern_evidence` holds points at `decisions`,
        # `decision_assumptions` or `decision_outcomes`, all of which are
        # user-authored and never truncated. Nothing a replay does can reach it.
        "patterns",
        "pattern_evidence",
        # M8.0's two, and the argument is `patterns`' argument with one addition.
        #
        # Most of `user_model_facets` is derived: the statements are templates
        # with counts in them and `model derive` reproduces them exactly. Three
        # things in it are not, and each alone would be enough.
        #
        # `origin = 'asserted'` rows are somebody's own words. A goal is stated
        # rather than inferred — that is the whole reason the dimension has no
        # deriver — and a replay that truncated this table would delete every
        # goal anybody had entered and regenerate none of them.
        #
        # `dismissed_at` and `dismissed_reason` are a person having read a claim
        # about themselves and refused it, which is the property this set exists
        # for and which `patterns` is here for already.
        #
        # And `superseded_by` is a *history*: which statement replaced which, and
        # when. A rebuild produces today's facets, so a truncation would collapse
        # the chain to its current end — losing exactly the part that makes a
        # model of a person worth keeping, since "you used to believe this" is
        # only answerable while the superseded row exists.
        #
        # No snapshot needed. `facet_evidence` holds no foreign key at all — see
        # the note on `ref_id` — so nothing a replay does to `memories` can reach
        # it, and its one real foreign key points at `user_model_facets`, which
        # is never truncated.
        "user_model_facets",
        "facet_evidence",
        # M5.4's two, and the least arguable classification in this file since
        # `query_judgements`.
        #
        # Nothing about a reflection is derived. Its text is what a language
        # model wrote on one occasion, which no rebuild reproduces — not even
        # approximately, and a replay that regenerated it would quietly replace
        # sentences somebody had already read with different ones. Its
        # `dismissed_at` is a person having read a behavioural claim about
        # themselves and refused it, and `application/reflections.py` reads that
        # column to decide never to generate for the pattern again. Truncating
        # this table would un-reject every rejected claim and the next `reflect`
        # would write them all back.
        #
        # `reflection_citations` follows its parent: it is the frozen numbering
        # of one generated text, meaningless apart from it. Both reach only
        # `patterns` and `decisions`, which are themselves user-authored and
        # never truncated, so neither needs the snapshot the evidence tables do.
        "reflections",
        "reflection_citations",
        # M6.3, and the first table in this set whose *origin* is operational.
        #
        # Everything else here was written by a person on purpose. A surfacing
        # decision is written by the gate, unasked, at the rate an editor fires
        # events — which is exactly the description `OPERATIONAL_TABLES` opens
        # with, and where the majority of these rows belong on their own merits.
        #
        # Two columns move the whole table. `dismissed_at` and `acted_on_at` are
        # somebody having judged what they were shown, they exist nowhere else,
        # and `application/surfacing.py` *reads* them: a focus whose context is
        # dismissed repeatedly gets a higher bar. So a replay that truncated this
        # would un-dismiss every dismissal and reset every adapted threshold to
        # its default — and the next trigger would volunteer precisely what
        # somebody had already refused, which is the failure M6.3 exists to
        # prevent, arriving by way of the rebuild.
        #
        # The same shape as `patterns` in M5.3: most of it derivable, two columns
        # not, and the two decide. The refusal rows come along for the ride.
        #
        # No snapshot needed. `trigger_id` points at `events` and is deliberately
        # not a foreign key — see the model — so nothing here is reachable by a
        # cascade from anything a replay truncates.
        "surfacing_log",
        # M10.1's conversations. Nothing about a session is derived: its
        # boundaries are where somebody stopped typing for half an hour, its title
        # is cut from the first thing they said, and its `archived_at` is a person
        # having decided they were done with it. A replay that truncated this
        # would un-archive every archived conversation and lose every boundary,
        # and the next message would land in whatever the clock guessed.
        #
        # It reaches nothing derived — `chat_messages` points *into* it, not the
        # other way round — so it needs no snapshot and cannot be reached by a
        # cascade from `memories`.
        "chat_sessions",
        # M10.0's transcript, and the first table in this set that survives a
        # replay by *schema* rather than by classification or by snapshot.
        #
        # What is irreplaceable here is not the message text — that is in the
        # memory, and the memory comes back from the log — but how the message
        # was *read* and, for a question, the answer that was given. A rules
        # classifier will misread things, so the reading that was acted on has to
        # stay recoverable; recomputing it under a newer rule set would rewrite
        # history to agree with the present. And an answer is one model's output
        # on one occasion, which no rebuild reproduces — the same argument
        # `reflections` makes, and a replay that regenerated them would replace
        # sentences somebody had already read with different ones.
        #
        # It needs no `EVIDENCE_TABLES` snapshot, and that is the interesting
        # part. The obvious column here was `memory_id`, which would have put
        # this table in the cascade's path exactly as `decision_evidence` is —
        # and the snapshot machinery is built for link rows, so protecting a
        # transcript with it would mean deleting and reinserting the whole thing
        # on every rebuild. It carries `external_key` instead: the log records
        # it, so the memory comes back with the same one and the join finds it
        # again having preserved nothing. The classification decided the schema
        # rather than the other way round.
        #
        # M10.1 made that requirement explicit — sessions must survive a replay
        # while their memories are rebuilt — and the column was already right.
        # Its only foreign key is to `chat_sessions`, which is in this set beside
        # it, so the whole conversation graph is outside the truncation.
        "chat_messages",
        # M10.2's attachments, and the same argument a third time. What is
        # irreplaceable is not the file — the bytes are in the blob store and the
        # memory is rebuilt from the log — but the record that *somebody dropped
        # it here, with this note, in this conversation*. Nothing in the log says
        # which turn a memory was attached to, because a turn is not an ingestion
        # event.
        #
        # `deduplicated` is the column that settles it: it records whether these
        # bytes were already in the corpus at the moment of upload, which is not
        # re-derivable afterwards at all. Once two uploads of one file exist,
        # every artifact looks like it was always there.
        #
        # Keyed on `external_key` and foreign-keyed only to `chat_messages`, so
        # the cascade from `memories` cannot reach it.
        "chat_attachments",
        # M10.4's tags, and the least arguable member of this set since
        # `query_judgements`. Nothing in the ingestion log says how somebody filed a
        # thought. A tag is an assertion *about* an item, made by a person, and no
        # rebuild produces one.
        #
        # Its schema was decided by that classification rather than the other way
        # round, and twice. The obvious columns were `memory_id` and `entity_id`, and
        # each would have destroyed the table on the next rebuild: `memories` is
        # truncated, so the first is a cascade; `entities` is truncated *and* refilled
        # only by extraction, which costs money, so the second is a cascade that
        # would not even be repaired by re-running the pipeline. It carries
        # `(source_id, external_key)` and a canonical name instead — the durable
        # identity on both ends — so a replay leaves it entirely alone and
        # `tags.reconcile` re-upserts the concepts from the names it kept.
        #
        # `sources` is its only foreign key, and that table is source of truth.
        "memory_tags",
    }
)

# The user-authored tables that reach into the derived ones, and the columns a
# rebuild re-links them by.
#
# Declared as data rather than handled case by case, because the next table like
# this is inevitable and the failure mode of forgetting one is silent: the rows
# vanish in a cascade nobody watched and the decisions they belonged to are left
# looking as though nobody ever cited anything. A test asserts every table with
# an inbound foreign key into `memories` appears here.
EVIDENCE_TABLES: tuple[str, ...] = (
    "decision_evidence",
    "outcome_evidence",
    "assumption_evidence",
)

# Per table: the column naming what the evidence belongs to, and the columns
# carried across a rebuild without being interpreted here.
#
# `linked_at` is in both because when somebody attached a memory is a fact about
# them rather than about the rebuild — restamping it would move every link to
# whenever the corpus was last replayed. `occurred_at` is in the outcome one
# because it is a *snapshot* of the evidence memory's clock taken at link time,
# and the gap between it and the decision's date is the claim being made: a
# rebuild that re-derived it would let a changed mtime silently alter how strong
# a link somebody already reviewed and accepted.
_EVIDENCE_SHAPE: dict[str, tuple[str, tuple[str, ...]]] = {
    "decision_evidence": ("decision_id", ("relation", "linked_at")),
    "outcome_evidence": ("outcome_id", ("occurred_at", "linked_at")),
    "assumption_evidence": ("assumption_id", ("occurred_at", "linked_at")),
}

# Reconstructible from the tables above plus the blob store. Ordered
# child-before-parent, so truncating or dropping them in sequence never fights a
# foreign key.
DERIVED_TABLES: tuple[str, ...] = (
    # M3.1's two, and the pair that stretches the word "derived" hardest.
    #
    # They are rebuildable — re-running `extract-entities` produces them from
    # the memories, which the log produces — but a replay does *not* rebuild
    # them, because doing so would mean an LLM call per chunk on every rebuild.
    # So a full replay empties these and leaves them empty until extraction is
    # run again, which is the honest outcome rather than a gap: every chunk id
    # is new after a rebuild, so every mention's `chunk_id` would dangle and
    # every offset would point into text that no longer exists at that row.
    #
    # Classifying them source-of-truth instead would be worse in a way that is
    # easy to miss: it would assert that a language model's output at a
    # particular moment is irreplaceable input, and it would make the replay
    # guarantee false, since nothing in the log can reconstruct them.
    # M3.2's ledger, and the most uncomfortable classification in this file.
    #
    # It is derived *by force* rather than by argument: it has a foreign key to
    # `entities`, which is derived and truncated, so leaving this table behind
    # would leave every row referencing an entity that no longer exists. The FK
    # decides it, not the reasoning.
    #
    # **And that costs something real.** An automatic merge is genuinely
    # rebuildable — re-run `resolve-entities`. A *manual* merge is not: it is a
    # person's judgement on a pair the resolver was unsure about, which is
    # exactly the property `USER_AUTHORED_TABLES` exists to protect, and a full
    # replay destroys it along with the pending review queue.
    #
    # The fix is the one M1.7 found for `query_judgements`: key merges on
    # `(canonical_name, type)` — stable across rebuilds — instead of on entity
    # ids, which are minted per write. That is a schema change rather than a
    # classification change, so it is written down here rather than done
    # quietly, and `resolve-entities` must be re-run after any full replay.
    # M3.3, and first in the list because it references entities, memories and
    # chunks all three. Derived without the discomfort `entity_merges` carries:
    # nothing here is anybody's judgement, so re-running the extractor rebuilds
    # it exactly as re-running the embedder rebuilds a vector.
    "entity_relationships",
    "entity_merges",
    "entity_mentions",
    "entities",
    # M4.2, and classified by its foreign keys the same way `entity_merges` is:
    # both ends point at `memories`, which is truncated, so a retained row would
    # describe a diff between two versions that no longer exist.
    #
    # **The same discomfort, for a different reason.** Nothing here is anybody's
    # judgement, so this is not user-authored — but it is not cheaply
    # rebuildable either, because every row cost a model call and a replay throws
    # all of them away. That is unlike `memory_chunks`, which a rebuild
    # reproduces exactly for free, and unlike `entity_mentions`, which a rebuild
    # at least *could* reproduce by re-running extraction over the same text.
    #
    # It is also the one derived table whose input survives a replay perfectly:
    # the versions come back from the log with the same normalized text, so the
    # diffs are identical and only the *description* of them is gone. Keying the
    # cache on the pair of normalized hashes rather than on memory ids would
    # make these survivable, exactly as M1.7 proposed keying merges on
    # `(canonical_name, type)`. Written down rather than done quietly: it is a
    # schema change, and this corpus has seven version pairs.
    "change_summaries",
    "memory_chunks",
    "memories",
    "embedding_cache",
)

# **The fourth category, and M1.7 already knew it was missing.**
#
# That milestone shipped with two sets and reported in its own retrospective that
# the binary was hiding a third; `query_judgements` made it concrete and
# `USER_AUTHORED_TABLES` was the answer. This is the one after it, and the note
# that predicted it is still in the comment above `USER_AUTHORED_TABLES`: "`jobs`
# is classified derived and truncating it works, but it is *discardable*, not
# reconstructible, which is a different property that happened to be safe."
#
# Operational rows are the exhaust of running the system rather than any part of
# what it knows. Test each against the other three and all three refuse it:
#
# * Not source-of-truth — nothing is ever rebuilt from them.
# * Not derived — a replay does not reproduce them, and could not. No amount of
#   replaying the ingestion log produces a keystroke somebody made in an editor,
#   or the retry history of a job that failed twice.
# * Not user-authored — nobody wrote them. An event is a plugin firing; a job is
#   this system talking to itself.
#
# What they are is *discardable*: emptying them loses information that existed
# nowhere else and costs nothing, because nothing downstream depends on it. That
# is a fourth property, not a shade of the other three, and writing it down is
# what stops the next table like this being filed under "derived" because the
# truncation happens to work.
#
# `jobs` moves here from `DERIVED_TABLES`, where it never belonged. Nothing about
# the replay changes: both sets are emptied, in this order, and `SHADOW_TABLES`
# still copies them. Only the reason is now correct.
#
# Ordered child-before-parent like `DERIVED_TABLES`, though neither holds a
# foreign key to the other today — the order is a property the truncation relies
# on, so it is maintained rather than assumed absent.
OPERATIONAL_TABLES: tuple[str, ...] = (
    # M6.1's cache, and the clearest member this set has. Nothing is rebuilt
    # from a cached context, no replay reproduces one, nobody wrote it, and
    # dropping the table costs a re-assembly of a few hundred milliseconds. It
    # is also the one table where *keeping* stale rows would be actively wrong —
    # a context is a function of the whole corpus, and a replay changes the
    # corpus fingerprint every key is built from anyway.
    "context_cache",
    "events",
    "jobs",
)

# The cache is the interesting member of that list. Truncating it makes a replay
# honest — every vector is recomputed, so the embedding half of the pipeline is
# actually exercised — and slow. Keeping it makes the replay fast but only proves
# the pipeline downstream of embedding.
#
# The default is to keep it, because the cache is content-addressed: an entry is
# a pure function of (model, role, text), so a retained entry is correct by
# construction rather than by trust. A `--clear-cache` run is the stronger
# periodic check, not the everyday one.
CACHE_TABLE = "embedding_cache"

# The derived tables a shadow workspace builds its own copy of — everything
# except the cache.
#
# The cache is deliberately shared with the live schema, and that follows from
# what it is: an entry is a pure function of (model, role, text), so reading one
# written by the live pipeline is correct by construction rather than by trust.
# Giving the workspace its own empty cache would instead force every vector to be
# recomputed on every verification run, which would make the routine check
# expensive enough to stop being routine.
#
# `jobs` *is* copied, and is swapped in empty. After a full replay every memory
# id is new, so any job still pending refers to a row that no longer exists and
# would fail permanently. Replacing the queue with an empty one is the honest
# outcome, not collateral damage.
SHADOW_TABLES: tuple[str, ...] = tuple(
    name for name in (*DERIVED_TABLES, *OPERATIONAL_TABLES) if name != CACHE_TABLE
)

# Derived state that is not a Postgres table at all.
#
# A separate set rather than another name in `DERIVED_TABLES`, because
# everything that reads that tuple puts its contents in a `TRUNCATE`, and a
# Neo4j database is not truncatable by SQL. The classification is the same —
# rebuildable, disposable, never source of truth — and it is written down here
# for the same reason the tables are: a projection nobody classified is a
# projection a replay silently leaves stale.
#
# The rule this set carries: **Postgres wins on disagreement.** A full replay
# empties the graph and rebuilds it, so anything the graph knew that Postgres
# did not is gone by design. That is what makes it safe for no use case to write
# here directly — and as of M3.4 nothing does: extraction and resolution enqueue
# a `SYNC_GRAPH` job, and `application/graph_sync.py` is the only writer.
#
# "Rebuilds it" is new in M3.4 and is the half M3.0 left out. Clearing alone left
# the graph empty until somebody re-ran extraction, and empty is not the same as
# correct — a divergence check cannot tell a projection nobody built from one that
# has drifted. What the rebuild produces after a full replay is every memory and
# its source and no entities, because the entity tables are derived-and-not-rebuilt
# for the reasons given above them. That projection matches Postgres, which is the
# only thing the graph has ever promised.
DERIVED_PROJECTIONS: frozenset[str] = frozenset({"neo4j_graph"})


def cleared_tables(*, clear_cache: bool) -> tuple[str, ...]:
    """The tables a replay empties, given the cache decision.

    Two classifications, one truncation, and the names are kept separate because
    the *reasons* differ: derived tables are emptied because they will be
    rebuilt, operational ones because nothing needs them back. A single set would
    make the next `jobs` — a table that is safe to truncate for the wrong reason
    — invisible again.
    """
    ordered = (*DERIVED_TABLES, *OPERATIONAL_TABLES)
    if clear_cache:
        return ordered
    return tuple(name for name in ordered if name != CACHE_TABLE)


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


class ReplayStage(StrEnum):
    """How far upstream to start over.

    `EMBED` is the one that gets used in practice: swap the model, keep the
    chunks, recompute the vectors. Minutes of compute instead of a full pipeline
    run, and it is only safe because the chunker version and the model id are
    both recorded per chunk.
    """

    ALL = auto()
    NORMALIZE = auto()
    EMBED = auto()


@dataclass(frozen=True, slots=True)
class ReplayScope:
    """Which events to replay, and from which stage."""

    source_name: str | None = None
    after_seq: int = 0
    stage: ReplayStage = ReplayStage.ALL

    @property
    def rebuilds_memories(self) -> bool:
        return self.stage is ReplayStage.ALL

    @property
    def rechunks(self) -> bool:
        return self.stage in (ReplayStage.ALL, ReplayStage.NORMALIZE)

    @property
    def is_complete(self) -> bool:
        """Whether this scope rebuilds the entire derived corpus.

        Only a complete scope may be built in a workspace and swapped in, because
        a swap *replaces* the derived tables rather than merging into them. A
        workspace holding one source's rebuild is a complete replacement for that
        source and an empty one for every other, and swapping it in would delete
        them. An `embed`-stage workspace is worse: it replays no events, so it
        would swap in nothing at all.
        """
        return (
            self.stage is ReplayStage.ALL
            and self.source_name is None
            and self.after_seq == 0
        )

    def describe(self) -> str:
        parts = [f"stage={self.stage.value}"]
        parts.append(f"source={self.source_name}" if self.source_name else "source=all")
        parts.append(
            f"after_seq={self.after_seq}" if self.after_seq else "from=beginning"
        )
        return "  ".join(parts)


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    """One evidence row, as its durable identity rather than its ids.

    What survives a rebuild, and nothing else. `memory_id` and `chunk_id` are
    deliberately absent: they are exactly the columns a replay invalidates, and
    carrying them in the snapshot would tempt a later reader into writing them
    back.

    `table` and `owner` generalise this over `decision_evidence` and M5.1's
    `outcome_evidence`, which are the same problem twice — `owner` is the
    decision id or the outcome id, whichever this row hangs off. `columns`
    carries whatever else that table requires and this one does not interpret:
    `relation` for one, `occurred_at` for the other. Interpreting them here
    would mean this module growing a branch per table, which is how the third
    one gets forgotten.
    """

    table: str
    owner: UUID
    source_name: str
    external_key: str
    chunk_ordinal: int | None
    columns: dict[str, Any]


@dataclass(slots=True)
class ReplayReport:
    events: int = 0
    observed: int = 0
    deleted: int = 0
    # M10.4. Purge events found in the log, and the events they caused this rebuild
    # to skip. The second number is the one worth reporting: it says how much of the
    # log deliberately did not become a corpus, which is otherwise indistinguishable
    # from a rebuild that lost rows.
    purged: int = 0
    skipped_purged: int = 0
    memories: int = 0
    normalized: int = 0
    embedded: int = 0
    chunks: int = 0
    vectors_computed: int = 0
    cache_hits: int = 0
    # What the graph projection holds after the rebuild. Zero on a scoped replay,
    # which does not touch the graph at all, and zero-entity on a full one — see
    # `_rebuild_graph` for why that is the honest outcome rather than a gap.
    graph_nodes: int = 0
    graph_edges: int = 0
    # M5.0. Evidence links carried across the rebuild, and how many found their
    # memory again on the other side. A gap between the two is not an error —
    # a memory can legitimately have left the corpus — but it is the number that
    # says how much of Phase 5's provenance a replay cost.
    evidence_preserved: int = 0
    evidence_relinked: int = 0
    duration_ms: int = 0
    into_shadow: bool = False
    cache_cleared: bool = False

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "events": self.events,
            "observed": self.observed,
            "deleted": self.deleted,
            "purged": self.purged,
            "skipped_purged": self.skipped_purged,
            "memories": self.memories,
            "normalized": self.normalized,
            "embedded": self.embedded,
            "chunks": self.chunks,
            "vectors_computed": self.vectors_computed,
            "cache_hits": self.cache_hits,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "evidence_preserved": self.evidence_preserved,
            "evidence_relinked": self.evidence_relinked,
            "duration_ms": self.duration_ms,
            "into_shadow": self.into_shadow,
            "cache_cleared": self.cache_cleared,
        }


class MissingBlob(RuntimeError):
    """An artifact promises bytes the blob store does not have."""


class UnreplayableEvent(RuntimeError):
    """An event kind the projection has no rule for."""


class PartialShadowReplay(ValueError):
    """A workspace can only hold a rebuild of the whole corpus."""


# Every event kind `_rebuild` knows how to apply.
#
# An event type that is not in `EventType` at all is refused one layer earlier,
# by the mapper that turns a row into an entity — loudly, which is right. The
# gap this guards is the likelier mistake: adding a member to `EventType`,
# writing a producer for it, and never teaching replay to apply it. Then every
# rebuild silently omits that fact forever. A unit test asserts this set covers
# the enum, so the omission fails the build instead.
REPLAYABLE_EVENT_TYPES: frozenset[EventType] = frozenset(
    {EventType.ARTIFACT_OBSERVED, EventType.ITEM_DELETED, EventType.ITEM_PURGED}
)


# --------------------------------------------------------------------------
# The use case
# --------------------------------------------------------------------------

# Rows the streaming cursor buffers. The point of streaming at all is that the
# log does not have to fit in memory.
STREAM_CHUNK = 1000

MakeNormalize = Callable[[async_sessionmaker[AsyncSession]], NormalizeMemory]
MakeEmbed = Callable[[async_sessionmaker[AsyncSession]], EmbedMemory]


class ReplayCorpus:
    """Truncate the derived tables and rebuild them from the log.

    The use cases it runs are the real ones — `NormalizeMemory` and
    `EmbedMemory`, built against whichever session factory the rebuild is writing
    through. A replay that used its own simplified reimplementation of the
    pipeline would prove that the reimplementation works.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        make_normalize: MakeNormalize,
        make_embed: MakeEmbed,
        make_shadow: Callable[[], ShadowWorkspace] | None = None,
        blobs: BlobStore | None = None,
        graph: GraphStore | None = None,
    ) -> None:
        self._sessions = session_factory
        self._make_normalize = make_normalize
        self._make_embed = make_embed
        self._make_shadow = make_shadow
        # Only for the pre-flight check; the rebuild itself reads blobs through
        # `NormalizeMemory`. Optional so a caller that has not wired one still
        # works — it simply skips the check rather than failing to construct.
        self._blobs = blobs
        # Optional for the same reason, and with a sharper consequence: a replay
        # run without one rebuilds Postgres and leaves the graph holding memory
        # ids that no longer exist. `doctor` is where that shows up.
        self._graph = graph

    async def __call__(
        self,
        scope: ReplayScope | None = None,
        *,
        into_shadow: bool = False,
        clear_cache: bool = False,
    ) -> ReplayReport:
        resolved = scope or ReplayScope()
        started = time.monotonic()
        log = logger.bind(
            scope=resolved.describe(), into_shadow=into_shadow, clear_cache=clear_cache
        )
        log.info("replay.started")

        # Before anything is destroyed, and outside the branch, because both
        # paths cost the links: an in-place replay truncates `memories` and the
        # cascade takes them, and a shadow swap drops the live `memories` table
        # out from under the foreign key. See `_preserve_evidence`.
        preserved = await self._preserve_evidence()

        try:
            if into_shadow:
                make_shadow = self._require_shadow(resolved)
                report = await self._into_shadow(
                    make_shadow, resolved, clear_cache=clear_cache
                )
            else:
                report = await self._in_place(
                    self._sessions,
                    resolved,
                    clear_cache=clear_cache,
                    # Only a whole-corpus replay. A scoped one rebuilds part of the
                    # corpus, and the graph has no equivalent of `--source notes`
                    # until M3.1 gives its nodes a source to be narrowed by; clearing
                    # all of it would discard the other sources' projection to
                    # rebuild one.
                    clear_graph=resolved.is_complete,
                )
        except BaseException:
            # A shadow replay that failed left the live corpus untouched, so the
            # links still resolve and putting them back is a complete rollback.
            # An in-place one that failed has a broken corpus either way, and
            # restoring what still resolves is strictly better than leaving the
            # decisions with no provenance at all.
            await self._restore_evidence(preserved)
            raise

        report.evidence_preserved = len(preserved)
        report.evidence_relinked = await self._restore_evidence(preserved)

        report.into_shadow = into_shadow
        report.cache_cleared = clear_cache
        report.duration_ms = int((time.monotonic() - started) * 1000)
        log.info("replay.finished", **report.as_dict())
        return report

    # ----------------------------------------------------------------------
    # Evidence, across the rebuild
    # ----------------------------------------------------------------------

    async def _preserve_evidence(self) -> list[EvidenceLink]:
        """Read every evidence table out by natural key, then empty them.

        **The one place a classification is not enough.** `decisions`, its
        options and its assumptions are `USER_AUTHORED` and survive a replay
        because nothing truncates them and nothing references a table that is
        truncated. `decision_evidence` cannot be protected that way: it holds
        foreign keys into `memories` and `memory_chunks` with ON DELETE CASCADE,
        deliberately, because a citation that resolves to nothing is worse than
        no citation. `TRUNCATE memories CASCADE` therefore takes it whatever set
        it is in — the same finding M1.7 made when the golden set was specified
        with a foreign key, arriving at a table designed knowing it. M5.1's
        `outcome_evidence` is the identical problem, which is why this walks a
        declared list rather than naming one table.

        So the row does not survive; the *link* does. `(source_name,
        external_key, chunk_ordinal)` is stable across a rebuild for the reason
        the golden set's key is: names outlive ids, and chunking is
        deterministic, so chunk 4 of a file is chunk 4 again afterwards.

        Emptied here rather than left to the cascade, because the shadow path
        does not truncate anything — it drops the live `memories` table and moves
        the workspace's in, and a foreign key pointing at a table that is going
        away blocks the drop. `PostgresShadowSchema.swap_in` takes the
        constraints off and puts them back; it can only put them back onto rows
        that no longer reference anything, which is what this delete guarantees.
        """
        preserved: list[EvidenceLink] = []
        async with self._sessions.begin() as session:
            for name in EVIDENCE_TABLES:
                table = models.Base.metadata.tables[name]
                owner, extra = _EVIDENCE_SHAPE[name]
                rows = list(await session.execute(select(table)))
                for row in rows:
                    mapping = row._mapping
                    preserved.append(
                        EvidenceLink(
                            table=name,
                            owner=mapping[owner],
                            source_name=mapping["source_name"],
                            external_key=mapping["external_key"],
                            chunk_ordinal=mapping["chunk_ordinal"],
                            # Everything the table needs that this module does
                            # not interpret. Carried opaquely so a new column on
                            # either table survives without a branch here.
                            columns={column: mapping[column] for column in extra},
                        )
                    )
                if rows:
                    await session.execute(delete(table))

        if preserved:
            logger.info("replay.evidence_preserved", links=len(preserved))
        return preserved

    async def _restore_evidence(self, preserved: Sequence[EvidenceLink]) -> int:
        """Re-link the decisions to the rebuilt corpus, by natural key.

        A link whose memory is no longer in the corpus is dropped rather than
        written with a null pointer, and that is the honest outcome: the memory
        genuinely left, the decision genuinely survived it, and a row asserting
        evidence that cannot be opened is the thing M2.5 spent a milestone
        making impossible. The count of dropped links is reported rather than
        logged away.

        `linked_at` is carried over rather than restamped. When somebody
        attached a memory to a decision is a fact about them, not about the
        rebuild, and re-dating it would move every link to whenever the corpus
        was last replayed.
        """
        if not preserved:
            return 0

        restored = 0
        async with self._sessions.begin() as session:
            memories = {
                (row[0], row[1]): row[2]
                for row in await session.execute(
                    select(
                        models.Source.name,
                        models.Memory.external_key,
                        models.Memory.id,
                    )
                    .join(models.Source, models.Source.id == models.Memory.source_id)
                    .where(
                        models.Memory.is_current.is_(True),
                        models.Memory.deleted_at.is_(None),
                    )
                )
            }
            chunks = {
                (row[0], row[1]): row[2]
                for row in await session.execute(
                    select(
                        models.MemoryChunk.memory_id,
                        models.MemoryChunk.ordinal,
                        models.MemoryChunk.id,
                    )
                )
            }

            for link in preserved:
                memory_id = memories.get((link.source_name, link.external_key))
                if memory_id is None:
                    continue
                chunk_id = None
                if link.chunk_ordinal is not None:
                    chunk_id = chunks.get((memory_id, link.chunk_ordinal))
                    if chunk_id is None:
                        # The memory came back and that chunk did not, which
                        # happens when a re-chunk changed the boundaries. Widening
                        # to the whole memory would silently move the citation, so
                        # the link is dropped instead.
                        continue
                table = models.Base.metadata.tables[link.table]
                owner, _ = _EVIDENCE_SHAPE[link.table]
                await session.execute(
                    table.insert().values(
                        id=new_id(),
                        **{owner: link.owner},
                        memory_id=memory_id,
                        chunk_id=chunk_id,
                        source_name=link.source_name,
                        external_key=link.external_key,
                        chunk_ordinal=link.chunk_ordinal,
                        **link.columns,
                    )
                )
                restored += 1

        logger.info(
            "replay.evidence_relinked",
            restored=restored,
            dropped=len(preserved) - restored,
        )
        return restored

    def _require_shadow(self, scope: ReplayScope) -> Callable[[], ShadowWorkspace]:
        """The workspace factory, if a workspace can legitimately serve this scope.

        The scope check is a guardrail against a genuinely destructive operation
        rather than a tidiness rule. A swap replaces the derived tables; a
        workspace built from a partial scope is a complete replacement for the
        part that was replayed and an empty one for everything else, so
        `--source notes --into-shadow` would delete every other source's corpus
        and `--stage embed --into-shadow` would delete all of it. Both would
        report success.
        """
        if self._make_shadow is None:
            raise RuntimeError(
                "a shadow replay was requested but no workspace was configured"
            )
        if not scope.is_complete:
            raise PartialShadowReplay(
                f"a shadow workspace holds a complete rebuild, but this scope is "
                f"partial ({scope.describe()}). Swapping it in would replace the "
                f"derived tables with a corpus containing only what was replayed, "
                f"deleting the rest. Run it in place instead, or widen the scope "
                f"to the whole log at stage 'all'."
            )
        return self._make_shadow

    async def _into_shadow(
        self, make_shadow: Callable[[], ShadowWorkspace],
        scope: ReplayScope,
        *,
        clear_cache: bool,
    ) -> ReplayReport:
        """Build alongside the live tables, then replace them in one step."""
        shadow = make_shadow()
        await shadow.create()
        try:
            sessions = await shadow.sessions()
            # Not during the rebuild. The point of a workspace is that the live
            # corpus stays usable while it runs, and the graph is part of the
            # live corpus until the swap makes the workspace's tables the live
            # ones.
            report = await self._in_place(
                sessions, scope, clear_cache=clear_cache, clear_graph=False
            )
            # After loading, never during: an index built on an empty table and
            # maintained through every insert is slower and worse-connected.
            await shadow.build_indexes()
            await shadow.swap_in()
            # The moment the memory ids the graph referenced stopped existing —
            # and, immediately after, the first moment a projection can be built
            # from ids that will still be there. Through `self._sessions`, not the
            # workspace's: the swap has made those tables the live ones, and the
            # workspace factory is about to be discarded.
            await self._clear_graph()
            await self._project_graph(self._sessions)
            await self._count_graph(self._sessions, report)
        except BaseException:
            # The live tables were never touched, so throwing the workspace away
            # is a complete rollback.
            await shadow.discard()
            raise
        return report

    @asynccontextmanager
    async def rebuild_into_shadow(
        self, scope: ReplayScope | None = None, *, clear_cache: bool = False
    ) -> AsyncIterator[tuple[ReplayReport, async_sessionmaker[AsyncSession]]]:
        """Rebuild alongside the live corpus and hand it over to be read.

        Never swapped in, always discarded — the caller gets a session factory
        pointed at the rebuild and can compare it against the original with both
        present at once. That is what makes `verify-replay` safe to run against a
        corpus somebody is using: the live tables are not touched even while the
        comparison says they should be.

        The index build is skipped, because nothing here searches by vector; the
        comparison reads rows by natural key.
        """
        resolved = scope or ReplayScope()
        make_shadow = self._require_shadow(resolved)

        started = time.monotonic()
        shadow = make_shadow()
        await shadow.create()
        try:
            sessions = await shadow.sessions()
            # Never, on this path: `verify-replay` builds a rebuild in order to
            # compare it and then throws it away. Touching the live graph here
            # would make a read-only verification destructive.
            report = await self._in_place(
                sessions, resolved, clear_cache=clear_cache, clear_graph=False
            )
            report.into_shadow = True
            report.cache_cleared = clear_cache
            report.duration_ms = int((time.monotonic() - started) * 1000)
            yield report, sessions
        finally:
            await shadow.discard()

    async def _in_place(
        self,
        sessions: async_sessionmaker[AsyncSession],
        scope: ReplayScope,
        *,
        clear_cache: bool,
        clear_graph: bool,
    ) -> ReplayReport:
        report = ReplayReport()

        source_id = await self._resolve_source(scope.source_name)

        if scope.rebuilds_memories:
            # Before anything is destroyed. See `_preflight_blobs`.
            await self._preflight_blobs(scope, source_id)

        await self._clear(sessions, scope, source_id, clear_cache=clear_cache)
        if clear_graph:
            # Alongside the truncation, for the same reason: every memory id
            # about to be written is new, so every node the graph holds is about
            # to refer to a row that no longer exists.
            await self._clear_graph()

        if scope.rebuilds_memories:
            # One pass, interleaved: apply an event, then run the pipeline for
            # what it produced, then move to the next event. See `_rebuild`.
            await self._rebuild(sessions, scope, source_id, report)
        else:
            # The downstream stages work on what is already there, so they are a
            # pass over the current memories rather than over the log.
            if scope.rechunks:
                await self._normalize(sessions, source_id, report)
            await self._embed(sessions, source_id, report)

        await self._count_chunks(sessions, report)
        if clear_graph:
            # Last, and only for the scope that cleared it. Every memory id is
            # new, so this is the first moment the projection can be built from
            # ids that will still exist when somebody queries them.
            await self._project_graph(sessions)
            await self._count_graph(sessions, report)
        return report

    async def _count_graph(
        self, sessions: async_sessionmaker[AsyncSession], report: ReplayReport
    ) -> None:
        """Record what the projection now holds, from Postgres rather than Neo4j.

        Derived from the corpus rather than counted in the graph, which sounds
        backwards for a report about the graph and is not: the projection is a
        pure function of Postgres, so this is the number that has to be true, and
        a count read back from Neo4j would report what the projection *managed*
        rather than what it owes. `graph verify` is where the two are compared.
        """
        projection = await graph_projection.read(sessions)
        report.graph_nodes = projection.nodes
        report.graph_edges = len(projection.edges)

    async def _preflight_blobs(self, scope: ReplayScope, source_id: UUID | None) -> None:
        """Refuse to start if the bytes the log references are not reachable.

        The rebuild already fails loudly on a missing blob — but it did so *after*
        truncating, which meant discovering an unreachable blob store cost you the
        corpus. Found the hard way: running `replay` from a subdirectory resolved
        the default relative `blob_root` to an empty path, and the run truncated
        119 memories before failing on the first document.

        The corpus was rebuildable afterwards, from the same log, once the command
        was run from the right place — which is the system working as designed.
        But "destroys your corpus, then tells you why" is not the failure mode a
        destructive operation should have when the check costs one stat per
        distinct artifact.

        Only for scopes that rebuild memories; the downstream stages read no blobs.

        **Purged keys are exempt, as of M10.4, and the exemption is the feature
        rather than a hole in the check.** A permanent deletion shreds the bytes on
        purpose; the log keeps the observation that they existed. So an artifact that
        is missing *because somebody deleted it* is the system working, and refusing
        to start would mean the first permanent deletion made the corpus
        unrebuildable — turning the guarantee this milestone delivers into one that
        costs replay. The rebuild skips those events entirely, so it never asks for
        the bytes.

        The check is otherwise unchanged, and still fails first. An artifact missing
        for any other reason is a blob store that lost a write, which is exactly
        what this exists to catch before the truncate.
        """
        if self._blobs is None:
            return

        purged = await self._purged_keys(scope, source_id)
        seen: set[str] = set()
        missing: list[tuple[str, str]] = []
        async for event in self._stream_events(scope, source_id):
            if (event.source_id, event.external_key) in purged:
                continue
            if event.content_hash is None or event.content_hash.value in seen:
                continue
            seen.add(event.content_hash.value)
            if not await self._blobs.exists(event.content_hash):
                missing.append((event.content_hash.value, event.external_key))

        if missing:
            # The whole digest, not a prefix: the operator's next move is to look
            # for that file in the blob store, and a truncated hash cannot be grepped.
            shown = "; ".join(f"{key!r} ({digest})" for digest, key in missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise MissingBlob(
                f"{len(missing)} of {len(seen)} artifacts are not in the blob "
                f"store, so the corpus cannot be rebuilt: {shown}{more}. Nothing "
                f"has been changed. Check that MEMOS_BLOB_ROOT points at the right "
                f"store — a relative default resolves against the current "
                f"directory."
            )
        logger.info("replay.blobs_verified", artifacts=len(seen))

    async def _resolve_source(self, name: str | None) -> UUID | None:
        if name is None:
            return None
        # Read from the live tables always: `sources` is source of truth and is
        # not part of any workspace.
        async with self._sessions() as session:
            source_id = (
                await session.execute(
                    select(models.Source.id).where(models.Source.name == name)
                )
            ).scalar_one_or_none()
        if source_id is None:
            raise LookupError(f"no source named {name!r}")
        return source_id

    # ----------------------------------------------------------------------
    # Clearing
    # ----------------------------------------------------------------------

    async def _clear(
        self,
        sessions: async_sessionmaker[AsyncSession],
        scope: ReplayScope,
        source_id: UUID | None,
        *,
        clear_cache: bool,
    ) -> None:
        """Empty exactly what this scope is going to rebuild, and nothing more."""
        if scope.stage is ReplayStage.ALL and source_id is None:
            # The whole-corpus case, which is the one the guarantee is about.
            # TRUNCATE by name rather than a blanket sweep, so a table that has
            # drifted into the wrong set shows up as rows left behind instead of
            # being quietly emptied along with everything else.
            await truncate_derived(sessions, clear_cache=clear_cache)
            return

        # Narrowed to the memories the downstream pass will actually reprocess,
        # which is not the same as "every chunk in scope". A tombstoned memory
        # keeps the chunks it had when it was deleted, and `_targets` — rightly —
        # will not re-chunk or re-embed a tombstone. Clearing its vectors anyway
        # would strip them with nothing to restore them, so a routine
        # `--stage embed` model swap would quietly leave a corpus with permanently
        # unembedded chunks behind every deleted file.
        chunks_in_scope = _chunks_of_reprocessed(source_id)
        async with sessions.begin() as session:
            if scope.rechunks:
                # Only the jobs that are about to refer to nothing. A scoped
                # replay used to clear the whole queue, which would have thrown
                # away valid pending work for every source it was not replaying.
                # Run before the memories are deleted, because it selects through
                # them.
                await session.execute(
                    delete(models.Job).where(
                        models.Job.payload["memory_id"].astext.in_(
                            select(cast(models.Memory.id, Text)).where(
                                *_memories_of(source_id)
                            )
                        )
                    )
                )

            if scope.stage is ReplayStage.EMBED:
                # Chunk rows and their ids survive; only the vectors go. That is
                # what makes a model swap cheap, and it is why a chunk's identity
                # owes nothing to its vector.
                await session.execute(
                    update(models.MemoryChunk)
                    .where(*chunks_in_scope)
                    .values(embedding=None, embedding_model=None, embedded_at=None)
                )
            else:
                # ON DELETE CASCADE would take the chunks with the memories, but
                # they are deleted explicitly so the intent is legible here rather
                # than inferred from a constraint.
                await session.execute(
                    delete(models.MemoryChunk).where(*chunks_in_scope)
                )
                if scope.stage is ReplayStage.ALL:
                    await session.execute(
                        delete(models.Memory).where(*_memories_of(source_id))
                    )

            if clear_cache:
                await session.execute(delete(models.EmbeddingCacheEntry))

    async def _clear_graph(self) -> None:
        """Empty the graph projection, if one is wired.

        Not caught, and that is the deliberate half of this. A replay whose graph
        clear failed has rebuilt Postgres and left Neo4j holding nodes keyed by
        memory ids that no longer exist — every one of them dangling, and nothing
        downstream able to tell the difference between a stale node and a real
        one. Reporting success there would be reporting a rebuild that did not
        happen.

        This is also why a graph is not required to construct a `ReplayCorpus`:
        the choice is between failing loudly when a configured graph cannot be
        cleared and not configuring one at all, never between the two silently.
        """
        if self._graph is None:
            logger.debug("replay.graph_not_configured")
            return
        await self._graph.clear()
        logger.info("replay.graph_cleared")

    async def _project_graph(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        """Re-project the graph from the corpus that has just been rebuilt.

        Projects without clearing, because `_clear_graph` has already run — before
        the tables were rebuilt rather than after, so that the graph never holds
        nodes keyed by memory ids Postgres has already deleted. Calling
        `graph_projection.rebuild` here instead would clear a second time, which
        is harmless and would also mean the window of unavailability spans the
        whole table rebuild rather than ending with it.

        Clearing alone was M3.0's answer and it left the graph empty until
        somebody happened to run extraction again. Empty is not the same as
        correct: `graph verify` cannot tell a projection nobody has built from one
        that has diverged, and a graph that is *supposed* to be empty for hours
        after every replay makes the check useless exactly when it matters.

        What this projects is not what was there before, and the difference is the
        honest part of this milestone. `entities`, `entity_mentions` and
        `entity_relationships` are derived tables a replay truncates and does not
        rebuild — see `DERIVED_TABLES`, which explains why: rebuilding them means
        an LLM call per chunk, and every mention's offsets point into chunk text
        that no longer exists at that row. So the projection that follows a full
        replay holds every memory and its source and *no entities at all*, and
        `resolve-entities` and the two extraction passes have to be re-run to get
        them back.

        Reporting that as a rebuild is right. The projection matches Postgres,
        which is the only guarantee the graph ever offered.
        """
        if self._graph is None:
            return
        projection = await graph_projection.read(sessions)
        await graph_projection.write(self._graph, projection)
        logger.info("replay.graph_rebuilt", **projection.counts)

    # ----------------------------------------------------------------------
    # Memories, from the log
    # ----------------------------------------------------------------------

    async def _rebuild(
        self,
        sessions: async_sessionmaker[AsyncSession],
        scope: ReplayScope,
        source_id: UUID | None,
        report: ReplayReport,
    ) -> None:
        """Apply every event in `seq` order, running the pipeline as it goes.

        Versions are not assumed, they are derived: each `artifact_observed` for a
        key supersedes whatever is current and becomes version N+1, so a key with
        four events ends with versions 1 to 4 and exactly one `is_current`.

        The pipeline runs **per event, before the next one is applied**, and that
        interleaving is load-bearing rather than stylistic. A first attempt here
        applied every event and then normalized the surviving memories, which is
        cheaper and wrong: a superseded version had been normalized before it was
        superseded, and a deleted one had been normalized *and embedded* before it
        was tombstoned. Rebuilding only the current state left `normalized_hash`
        null on every historical version and dropped the tombstoned version's
        chunk entirely.

        Neither of those is cosmetic. `normalize` finds a previous version by its
        `normalized_hash` in order to move chunks onto a cosmetically-changed
        file, so a rebuild that nulls those hashes silently costs the next sync a
        full re-chunk and re-embed of anything it touches. And the difference is
        invisible to row counts on the memories table — exactly the failure this
        milestone predicted.

        The cost is real and worth stating: an item with fifty revisions is parsed,
        chunked and embedded fifty times, because that is what the log says
        happened. A replay is as expensive as the history it is replaying.

        One transaction per event, for the reason `sync` uses one per item: the
        whole log in a single transaction holds locks for minutes and throws
        everything away on any failure.
        """
        normalize = self._make_normalize(sessions)
        embed = self._make_embed(sessions)
        # Read before the first event is applied, because a purge at the end of the
        # log has to suppress observations at the beginning of it. Streaming the log
        # once and reacting to the purge when it arrives would rebuild the memory,
        # chunk it, embed it and then delete it — which works, costs the full
        # pipeline for content nobody kept, and requires the blob that was
        # deliberately shredded. Two passes over one indexed column is the cheaper
        # and the only correct order.
        purged = await self._purged_keys(scope, source_id)

        async for event in self._stream_events(scope, source_id):
            report.events += 1

            if (event.source_id, event.external_key) in purged:
                # Everything about this key, not only the purge itself. The item was
                # permanently deleted, so a faithful rebuild is one that does not
                # contain it — and the log still says it was here, which is exactly
                # the record a purge keeps and this loop reads.
                if event.event_type is EventType.ITEM_PURGED:
                    report.purged += 1
                else:
                    report.skipped_purged += 1
                continue

            if event.event_type is EventType.ARTIFACT_OBSERVED:
                memory_id = new_id()
                async with sessions.begin() as session:
                    await SqlAlchemyMemoryRepository(session).add_version(
                        memory_from_event(event, memory_id=memory_id)
                    )
                report.observed += 1
                report.memories += 1

                await self._normalize_one(
                    normalize, memory_id, event.external_key, report
                )
                await self._embed_one(embed, memory_id, report)
                continue

            if event.event_type is EventType.ITEM_DELETED:
                await self._tombstone(sessions, event, report)
                continue

            if event.event_type is EventType.ITEM_PURGED:
                # Unreachable: the pre-scan above catches every purge, including
                # this one, so a purge reaching here means the two disagree about
                # which keys were purged. Counted rather than raised, because the
                # rebuild is still correct — the key simply was not skipped — and
                # named so the branch is not mistaken for one nothing produces.
                report.purged += 1
                logger.warning(
                    "replay.purge_outside_prescan",
                    key=event.external_key,
                    seq=event.seq,
                )
                continue

            # An event type this projection does not know how to apply. Silently
            # ignoring it would mean a future event kind quietly not being
            # replayed, and the corpus differing from the log in a way only a
            # careful reader of both would ever notice.
            raise UnreplayableEvent(
                f"replay does not know how to apply {event.event_type.value!r} "
                f"(event {event.id}, seq {event.seq}); add it to "
                f"ReplayCorpus._rebuild or the rebuild is no longer faithful"
            )

    async def _tombstone(
        self,
        sessions: async_sessionmaker[AsyncSession],
        event: IngestionEvent,
        report: ReplayReport,
    ) -> None:
        async with sessions.begin() as session:
            memories = SqlAlchemyMemoryRepository(session)
            current = await memories.get_current(event.source_id, event.external_key)
            if current is None:
                # A deletion for a key with no current version. Reachable when the
                # scope starts partway through the log, and not an error: the
                # tombstone has nothing to apply to.
                logger.debug(
                    "replay.tombstone_without_target",
                    key=event.external_key,
                    seq=event.seq,
                )
                return
            await memories.tombstone(current.id, recorded_at_of(event))
        report.deleted += 1

    async def _normalize_one(
        self,
        normalize: NormalizeMemory,
        memory_id: UUID,
        external_key: str,
        report: ReplayReport,
    ) -> None:
        try:
            await normalize(memory_id)
        except BlobNotFound as exc:
            # The whole rebuild rests on the blobs being there. Skipping one would
            # produce a corpus quietly missing a document, and no count would show
            # it — a memory row with no chunks is also what an empty file looks
            # like.
            raise MissingBlob(
                f"blob {exc.args[0] if exc.args else '?'} is referenced by "
                f"{external_key!r} (memory {memory_id}) but is not in the blob "
                f"store; the corpus cannot be rebuilt without it"
            ) from exc
        report.normalized += 1

    async def _embed_one(
        self, embed: EmbedMemory, memory_id: UUID, report: ReplayReport
    ) -> None:
        outcome = await embed(memory_id)
        report.embedded += 1
        report.vectors_computed += outcome.cache_misses
        report.cache_hits += outcome.cache_hits

    async def _purged_keys(
        self, scope: ReplayScope, source_id: UUID | None
    ) -> set[tuple[UUID, str]]:
        """Every item a purge event names, within this scope.

        **The set, not a stream, and that asymmetry is the point.** Everything else
        about a replay is streamed because the log does not fit in memory; this is
        one row per permanently deleted item, which is bounded by how much somebody
        has deliberately deleted rather than by the size of the corpus. A corpus
        with a million events and forty purges holds forty pairs here.

        Scoped identically to `_stream_events`, so a source-scoped replay does not
        suppress a key in another source that happens to share an external key —
        `notes/queue.md` exists in more than one source, and the pair is the
        identity for exactly that reason.
        """
        stmt = select(
            models.IngestionEvent.source_id, models.IngestionEvent.external_key
        ).where(
            models.IngestionEvent.event_type == EventType.ITEM_PURGED.value,
            models.IngestionEvent.seq > scope.after_seq,
        )
        if source_id is not None:
            stmt = stmt.where(models.IngestionEvent.source_id == source_id)
        async with self._sessions() as session:
            rows = (await session.execute(stmt.distinct())).all()
        keys = {(row[0], str(row[1])) for row in rows}
        if keys:
            logger.info("replay.purged_keys", items=len(keys))
        return keys

    async def _stream_events(
        self, scope: ReplayScope, source_id: UUID | None
    ) -> AsyncIterator[IngestionEvent]:
        """Server-side cursor over the log, in `seq` order.

        Streamed rather than fetched. A mature log does not fit in memory, and
        `SELECT * FROM ingestion_events` works perfectly on a fixture and takes
        the process down on a real corpus.

        Its own session, separate from the one doing the writing: a cursor held
        open across writes on the same connection is a different set of problems
        than this milestone needs.
        """
        stmt = (
            select(models.IngestionEvent)
            .where(models.IngestionEvent.seq > scope.after_seq)
            .order_by(models.IngestionEvent.seq)
        )
        if source_id is not None:
            stmt = stmt.where(models.IngestionEvent.source_id == source_id)

        async with self._sessions() as session:
            result = await session.stream(
                stmt.execution_options(yield_per=STREAM_CHUNK)
            )
            async for row in result.scalars():
                yield to_event(row)

    # ----------------------------------------------------------------------
    # Downstream stages, for the scopes that keep their memories
    # ----------------------------------------------------------------------

    async def _normalize(
        self,
        sessions: async_sessionmaker[AsyncSession],
        source_id: UUID | None,
        report: ReplayReport,
    ) -> None:
        normalize = self._make_normalize(sessions)
        for memory_id, external_key in await self._targets(sessions, source_id):
            await self._normalize_one(normalize, memory_id, external_key, report)

    async def _embed(
        self,
        sessions: async_sessionmaker[AsyncSession],
        source_id: UUID | None,
        report: ReplayReport,
    ) -> None:
        embed = self._make_embed(sessions)
        for memory_id, _ in await self._targets(sessions, source_id):
            await self._embed_one(embed, memory_id, report)

    async def _targets(
        self, sessions: async_sessionmaker[AsyncSession], source_id: UUID | None
    ) -> Sequence[tuple[UUID, str]]:
        """Current, undeleted memories, in a stable order.

        Only current versions: a superseded version's chunks were deleted with
        it, and re-chunking one would put text back into the retrieval set that
        the item no longer says.
        """
        stmt = (
            select(models.Memory.id, models.Memory.external_key)
            .where(models.Memory.id.in_(_reprocessed(source_id)))
            .order_by(models.Memory.external_key)
        )
        async with sessions() as session:
            return [(row[0], row[1]) for row in await session.execute(stmt)]

    async def _count_chunks(
        self, sessions: async_sessionmaker[AsyncSession], report: ReplayReport
    ) -> None:
        async with sessions() as session:
            report.chunks = (
                await session.execute(select(func.count()).select_from(models.MemoryChunk))
            ).scalar_one()


def _memories_of(source_id: UUID | None) -> list[ColumnElement[bool]]:
    """Narrow a memories statement to one source, if one was named."""
    if source_id is None:
        return []
    return [models.Memory.source_id == source_id]


def _reprocessed(source_id: UUID | None) -> Select[tuple[UUID]]:
    """The memories a downstream stage will reprocess.

    Deliberately the same predicate `_targets` uses. The two have to agree: what a
    partial replay clears must be exactly what it is going to rebuild, or the
    difference is destroyed.
    """
    stmt = select(models.Memory.id).where(
        models.Memory.is_current.is_(True), models.Memory.deleted_at.is_(None)
    )
    if source_id is not None:
        stmt = stmt.where(models.Memory.source_id == source_id)
    return stmt


def _chunks_of_reprocessed(source_id: UUID | None) -> list[ColumnElement[bool]]:
    """Chunks belonging to those memories, which reach their source through them."""
    return [models.MemoryChunk.memory_id.in_(_reprocessed(source_id))]


async def truncate_derived(
    session_factory: async_sessionmaker[AsyncSession], *, clear_cache: bool
) -> None:
    """Empty the derived tables outright. Used by tests and by `--from-beginning`.

    `TRUNCATE` rather than `DELETE`, and named tables rather than every table:
    naming them means a table that has drifted into the wrong set shows up here
    as data left behind, instead of being quietly swept away with everything else.
    """
    names = ", ".join(f'"{name}"' for name in cleared_tables(clear_cache=clear_cache))
    async with session_factory.begin() as session:
        await session.execute(text(f"TRUNCATE {names} CASCADE"))
