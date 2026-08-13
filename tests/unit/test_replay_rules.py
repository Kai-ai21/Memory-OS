"""The parts of replay that are decidable without a database.

The table classification is the one that matters most here. It is the only test in
the suite that fails when somebody adds a table and does not say whether it can be
rebuilt — and that omission is invisible otherwise, because a table in neither
set is simply never replayed and never reported.
"""

from datetime import UTC, datetime
from itertools import combinations
from uuid import UUID

import pytest

from memoryos.adapters.db.models import Base
from memoryos.application.projection import (
    MalformedEvent,
    kind_for,
    memory_from_event,
    recorded_at_of,
)
from memoryos.application.replay import (
    CACHE_TABLE,
    DERIVED_PROJECTIONS,
    DERIVED_TABLES,
    REPLAYABLE_EVENT_TYPES,
    SHADOW_TABLES,
    SOURCE_OF_TRUTH_TABLES,
    USER_AUTHORED_TABLES,
    ReplayScope,
    ReplayStage,
    derived_tables,
)
from memoryos.application.verification import digest_of
from memoryos.domain.entities import IngestionEvent
from memoryos.domain.ids import new_id
from memoryos.domain.values import ContentHash, EventType, MemoryKind, TimeProvenance

RECORDED_AT = datetime(2024, 3, 4, 10, 0, tzinfo=UTC)
OCCURRED_AT = datetime(2023, 1, 2, 8, 0, tzinfo=UTC)
HASH = "a" * 64


def observed(**overrides: object) -> IngestionEvent:
    fields: dict[str, object] = {
        "id": new_id(),
        "event_type": EventType.ARTIFACT_OBSERVED,
        "source_id": new_id(),
        "external_key": "notes/example.md",
        "occurred_at": OCCURRED_AT,
        "occurred_at_source": TimeProvenance.FILESYSTEM,
        "content_hash": ContentHash(HASH),
        "payload": {"media_type": "text/markdown", "byte_size": 12},
        "recorded_at": RECORDED_AT,
        "seq": 7,
    }
    fields.update(overrides)
    return IngestionEvent(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Table classification
# --------------------------------------------------------------------------


def test_every_table_is_classified_exactly_once() -> None:
    """The test that fails when a new table is not classified.

    A table in no set is never rebuilt and never mentioned; a table in two is a
    contradiction about whether it holds anything irreplaceable. Either way
    nothing errors at runtime — the replay guarantee just quietly stops covering
    part of the schema. So the check lives here, against the real metadata.

    Three sets, not two. M1.7 shipped with two and reported that the binary was
    one short; `query_judgements` is the table that made that concrete.
    """
    known = {table.name for table in Base.metadata.sorted_tables}
    sets = {
        "SOURCE_OF_TRUTH_TABLES": SOURCE_OF_TRUTH_TABLES,
        "DERIVED_TABLES": frozenset(DERIVED_TABLES),
        "USER_AUTHORED_TABLES": USER_AUTHORED_TABLES,
    }
    classified = frozenset().union(*sets.values())

    unclassified = known - classified
    assert unclassified == set(), (
        f"{sorted(unclassified)} are in none of {sorted(sets)}. Decide whether "
        f"each one is irreplaceable input, rebuildable from the log and blobs, or "
        f"written by a person, and add it to the right set in "
        f"memoryos/application/replay.py."
    )

    for left, right in combinations(sorted(sets), 2):
        overlap = sets[left] & sets[right]
        assert overlap == set(), f"{sorted(overlap)} is in both {left} and {right}"

    # And no set names a table that does not exist, which is how one goes stale
    # after a rename.
    assert classified - known == set()


def test_human_authored_data_is_neither_derived_nor_ingestion_input() -> None:
    """The classification the milestone turns on.

    A judgement cannot be rebuilt — no amount of replaying the log produces
    somebody's opinion — so it is not derived. It also does not feed ingestion,
    so it is not source of truth in the sense the other set means. Putting it in
    either would attach the wrong rule to it: derived tables get truncated, and
    that would destroy the labelled data M2.0 is measured against.
    """
    assert "query_judgements" in USER_AUTHORED_TABLES
    assert "query_judgements" not in DERIVED_TABLES
    assert "query_judgements" not in SOURCE_OF_TRUTH_TABLES


def test_decisions_are_user_authored_like_judgements() -> None:
    """M5.0's addition to the same set, and the same argument.

    No amount of replaying the log produces somebody's account of a choice they
    made — the question they were answering, what else was on the table, or how
    confident they were at the time. So a decision is not derived. It does not
    feed ingestion either, so it is not source of truth in the sense that set
    means. Classifying any of these derived would attach the truncation rule to
    the corpus every later Phase 5 milestone reads.

    `decision_suggestions` is here rather than in the derived set even though a
    language model produced its drafts, because the row also carries a person's
    accept or reject — and unlike `entity_merges` it has no foreign key forcing
    the classification. Its provenance is a natural key plus id snapshots, which
    is what lets it be classified by argument.
    """
    for name in (
        "decisions",
        "decision_options",
        "decision_assumptions",
        "decision_evidence",
        "decision_suggestions",
    ):
        assert name in USER_AUTHORED_TABLES, name
        assert name not in DERIVED_TABLES, name
        assert name not in SOURCE_OF_TRUTH_TABLES, name


def test_evidence_is_user_authored_and_still_reachable_by_a_cascade() -> None:
    """The one classification a set membership cannot enforce on its own.

    `decision_evidence` holds ON DELETE CASCADE foreign keys into `memories` and
    `memory_chunks`, deliberately — a citation that resolves to nothing is worse
    than an absent one — so `TRUNCATE memories CASCADE` reaches it however it is
    classified here. That is the finding M1.7 made when the golden set was
    specified with a foreign key, and this table was designed knowing it: every
    row carries the natural key too, and `ReplayCorpus._preserve_evidence` reads
    the links out before the truncation and writes them back afterwards.

    This test asserts the shape that makes that possible, so that a later change
    dropping the natural key from the row fails here rather than losing every
    link on the next rebuild.
    """
    evidence = Base.metadata.tables["decision_evidence"]
    columns = set(evidence.columns.keys())
    assert {"source_name", "external_key", "chunk_ordinal"} <= columns, (
        "decision_evidence must carry the durable identity as well as the ids, "
        "or a replay cannot re-link it after TRUNCATE ... CASCADE takes the row."
    )
    referenced = {
        key.column.table.name for key in evidence.foreign_keys
    }
    assert {"memories", "memory_chunks"} <= referenced


def test_the_shadow_swap_knows_about_the_constraints_pointing_into_it() -> None:
    """The collision M5.0 was the first table to cause.

    Until now nothing outside the derived set referenced anything inside it, so
    `DROP TABLE public.memories` in the swap worked. `decision_evidence` breaks
    that, and the tempting fix — `DROP TABLE ... CASCADE` — makes the error go
    away by taking the constraints with it, silently, leaving a live schema that
    no longer matches the models and an `alembic check` that fails on the next
    run.

    Imported here rather than in `test_replay_rules`' own imports because this
    is the one assertion in this file that reaches into an adapter: the rule is
    declared in `application/replay.py` and the mechanism lives beside the SQL
    that needs it.
    """
    from memoryos.adapters.db.shadow import inbound_foreign_key_names

    names = inbound_foreign_key_names()
    assert "fk_decision_evidence_memory_id" in names
    assert "fk_decision_evidence_chunk_id" in names


def test_no_user_authored_table_is_ever_truncated() -> None:
    # `truncate_derived` names its tables explicitly, so this asserts the two
    # lists cannot overlap however the cache flag is set.
    for clear_cache in (False, True):
        assert not (set(derived_tables(clear_cache=clear_cache)) & USER_AUTHORED_TABLES)


def test_a_user_authored_table_is_never_built_in_a_workspace() -> None:
    # A shadow swap replaces the tables it holds. Holding judgements would mean
    # swapping in an empty copy of them.
    assert not (set(SHADOW_TABLES) & USER_AUTHORED_TABLES)


def test_the_derived_tables_are_ordered_child_before_parent() -> None:
    # Truncation and dropping walk this list in order, so a parent listed before
    # its child would fail on a foreign key.
    order = list(DERIVED_TABLES)
    assert order.index("memory_chunks") < order.index("memories")


def test_the_event_log_is_never_derived() -> None:
    """The one classification that would invalidate the whole milestone.

    Everything else is an opinion; this is the premise. If the log were
    truncatable there would be nothing to replay from.
    """
    assert "ingestion_events" in SOURCE_OF_TRUTH_TABLES
    assert "ingestion_events" not in DERIVED_TABLES
    assert "raw_artifacts" in SOURCE_OF_TRUTH_TABLES
    assert "sources" in SOURCE_OF_TRUTH_TABLES


def test_every_event_type_has_a_replay_rule() -> None:
    """The sibling of the table check, for the log's vocabulary.

    An event kind outside `EventType` is refused by the mapper. The gap worth a
    test is the other one: adding a member to the enum, writing something that
    emits it, and never teaching replay to apply it — after which every rebuild
    silently omits that fact and no count changes.
    """
    unhandled = set(EventType) - REPLAYABLE_EVENT_TYPES
    assert unhandled == set(), (
        f"{sorted(e.value for e in unhandled)} can be written to the log but "
        f"replay has no rule for them. Add a branch in ReplayCorpus._rebuild and "
        f"list the type in REPLAYABLE_EVENT_TYPES."
    )
    # And nothing claimed here has been removed from the enum.
    assert set(EventType) >= REPLAYABLE_EVENT_TYPES


def test_the_cache_is_kept_by_default_and_cleared_on_request() -> None:
    assert CACHE_TABLE not in derived_tables(clear_cache=False)
    assert CACHE_TABLE in derived_tables(clear_cache=True)
    # Keeping the cache must not spare anything else.
    assert set(derived_tables(clear_cache=False)) == set(DERIVED_TABLES) - {CACHE_TABLE}


def test_the_graph_is_classified_as_derived_and_is_not_a_table() -> None:
    """M3.0's addition to the classification, and why it needed its own set.

    The graph is derived by every test the other sets apply: rebuildable from
    the log and the blobs, never source of truth, safe to destroy. What it is
    not is a Postgres table, and everything reading `DERIVED_TABLES` puts its
    contents inside a `TRUNCATE` — so listing it there would produce a replay
    that fails on a syntax error rather than one that clears the graph.
    """
    assert "neo4j_graph" in DERIVED_PROJECTIONS
    # Kept out of the table sets, which the classification test above checks
    # against real SQLAlchemy metadata. A name in there that is not a table
    # would fail that test, which is the intended safety net working.
    tables = set(DERIVED_TABLES) | SOURCE_OF_TRUTH_TABLES | USER_AUTHORED_TABLES
    assert not (DERIVED_PROJECTIONS & tables)
    assert not (DERIVED_PROJECTIONS & set(derived_tables(clear_cache=True)))


def test_the_shadow_workspace_copies_everything_except_the_cache() -> None:
    # The cache is content-addressed, so sharing the live one is correct rather
    # than merely convenient — and it keeps a verification run cheap.
    assert set(SHADOW_TABLES) == set(DERIVED_TABLES) - {CACHE_TABLE}


# --------------------------------------------------------------------------
# Scope semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "rebuilds", "rechunks"),
    [
        (ReplayStage.ALL, True, True),
        (ReplayStage.NORMALIZE, False, True),
        (ReplayStage.EMBED, False, False),
    ],
)
def test_a_stage_decides_which_work_is_redone(
    stage: ReplayStage, rebuilds: bool, rechunks: bool
) -> None:
    scope = ReplayScope(stage=stage)
    assert scope.rebuilds_memories is rebuilds
    assert scope.rechunks is rechunks


def test_the_default_scope_is_the_full_proof() -> None:
    scope = ReplayScope()
    assert scope.stage is ReplayStage.ALL
    assert scope.after_seq == 0
    assert scope.source_name is None
    assert "from=beginning" in scope.describe()


@pytest.mark.parametrize(
    "scope",
    [
        ReplayScope(source_name="notes"),
        ReplayScope(after_seq=10),
        ReplayScope(stage=ReplayStage.NORMALIZE),
        ReplayScope(stage=ReplayStage.EMBED),
    ],
)
def test_a_partial_scope_is_not_a_complete_rebuild(scope: ReplayScope) -> None:
    """What decides whether a scope may be swapped in from a workspace.

    A swap replaces the derived tables rather than merging into them, so a
    workspace built from a partial scope is a complete replacement for the part
    replayed and an empty one for everything else. `--source notes --into-shadow`
    would delete every other source; `--stage embed --into-shadow` would delete
    the whole corpus, having replayed no events at all.
    """
    assert scope.is_complete is False


def test_the_default_scope_is_complete() -> None:
    assert ReplayScope().is_complete is True


def test_a_scope_describes_itself_for_the_log() -> None:
    described = ReplayScope(source_name="notes", after_seq=42).describe()
    assert "source=notes" in described
    assert "after_seq=42" in described


# --------------------------------------------------------------------------
# The projection, which is pure
# --------------------------------------------------------------------------


def test_a_memory_is_stamped_from_the_event_not_the_clock() -> None:
    """The determinism rule, at the one place it is decided.

    `ingested_at` is the only timestamp a rebuild is allowed to write, and it has
    to come from the log. Sourcing it from `now()` would move every memory to the
    day the replay ran, and no row count would notice.
    """
    memory = memory_from_event(observed(), memory_id=new_id())

    assert memory.ingested_at == RECORDED_AT
    assert memory.occurred_at == OCCURRED_AT
    assert memory.occurred_at_source is TimeProvenance.FILESYSTEM


def test_the_projection_carries_the_events_payload_through() -> None:
    memory = memory_from_event(observed(), memory_id=new_id())
    assert memory.meta == {"media_type": "text/markdown"}
    assert memory.content_hash.value == HASH


def test_the_projection_leaves_versioning_to_the_repository() -> None:
    # Versions come from what is already stored, so a rebuild derives 1..N from
    # the order of the log rather than from anything a caller believed.
    memory = memory_from_event(observed(), memory_id=new_id())
    assert memory.version == 1
    assert memory.is_current is True


def test_the_same_id_is_used_that_was_asked_for() -> None:
    memory_id: UUID = new_id()
    assert memory_from_event(observed(), memory_id=memory_id).id == memory_id


def test_a_deletion_event_cannot_be_projected_to_a_memory() -> None:
    """Routing a tombstone down the wrong branch should be loud.

    It carries no content hash, so the memory it would produce could not
    reference an artifact — and a silent skip would drop a version.
    """
    event = observed(event_type=EventType.ITEM_DELETED, content_hash=None)
    with pytest.raises(MalformedEvent, match="no content_hash"):
        memory_from_event(event, memory_id=new_id())


def test_an_unrecorded_event_cannot_stamp_anything() -> None:
    with pytest.raises(MalformedEvent, match="no recorded_at"):
        recorded_at_of(observed(recorded_at=None))


def test_recorded_at_is_returned_when_present() -> None:
    assert recorded_at_of(observed()) == RECORDED_AT


@pytest.mark.parametrize(
    ("key", "kind"),
    [
        ("notes/a.md", MemoryKind.NOTE),
        ("notes/a.txt", MemoryKind.NOTE),
        ("src/a.py", MemoryKind.CODE),
        ("docs/a.pdf", MemoryKind.DOCUMENT),
        ("Makefile", MemoryKind.OTHER),
        ("a.MD", MemoryKind.NOTE),
    ],
)
def test_kind_is_guessed_from_the_suffix(key: str, kind: MemoryKind) -> None:
    # A pure function of the key, which is what lets a rebuild reproduce it.
    assert kind_for(key) == kind


# --------------------------------------------------------------------------
# The embedding digest the comparison relies on
# --------------------------------------------------------------------------


def test_the_digest_is_stable_for_the_same_vector() -> None:
    assert digest_of([0.1, 0.2, 0.3]) == digest_of([0.1, 0.2, 0.3])


def test_the_digest_separates_vectors_that_differ_below_printed_precision() -> None:
    """Why the digest is over packed bytes rather than formatted text.

    A text rendering rounds, and two vectors differing in the sixteenth decimal
    would compare equal — which is exactly the "close enough" that would let a
    changed model pass a verification.
    """
    assert digest_of([0.1, 0.2, 0.3]) != digest_of([0.1, 0.2, 0.3 + 1e-16])


def test_a_missing_vector_has_no_digest() -> None:
    # Distinct from any real digest, so "unembedded" cannot compare equal to
    # "embedded".
    assert digest_of(None) is None
