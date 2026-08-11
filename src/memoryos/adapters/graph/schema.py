"""Constraints, indexes, and the version marker that makes drift detectable.

Neo4j has no Alembic. There is no migration table, no revision graph, and no
`upgrade head` — the schema is whatever statements somebody happened to run
against this database, and nothing records which ones. That is the gap this
module fills, and it fills it with the two things Alembic actually provides:

* **Idempotent application.** Every statement carries `IF NOT EXISTS`, so the
  whole set can be applied on every connect. Applying the schema is not an
  event that has to be scheduled, remembered, or ordered against a deployment.
* **A recorded version.** A `:SchemaVersion` node says which revision of these
  statements was last applied. Without it, a database that predates a new
  constraint is indistinguishable from one that has it, and the difference only
  surfaces as duplicate nodes months later.

What this deliberately does *not* provide is migration. There is no downgrade
and no transformation of existing data, because at M3.0 there is no data to
transform. When a schema change needs one, it belongs here as an explicit
numbered step, and `SCHEMA_VERSION` is what tells it whether to run.
"""

from typing import LiteralString

import structlog
from neo4j import AsyncDriver

from memoryos.domain.values import IDENTITY_PROPERTY, GraphLabel

logger = structlog.get_logger(__name__)

# Bump whenever `STATEMENTS` changes. `doctor` compares this against what the
# database reports, so a bump that nobody applied shows up as drift rather than
# as a constraint that silently is not there.
SCHEMA_VERSION = 1

# Singleton key for the version node. A property rather than a bare label so
# `MERGE` has something to match on — `MERGE (v:SchemaVersion)` with no
# properties matches any existing node with that label, which is the behaviour
# wanted here but not the behaviour that is obvious to the next reader.
SCHEMA_SCOPE = "memoryos"

# The label the version marker carries. Excluded from `clear`, and from the
# counts `doctor` reports, because it describes the schema rather than the
# projection.
VERSION_LABEL = "SchemaVersion"

# `LiteralString` is not decoration: the driver's own signatures require it, and
# that is what makes a Cypher statement built by string formatting a type error
# rather than an injection. Everything here is a constant for exactly that
# reason, and the one place a query is assembled at runtime — the variable-depth
# traversal in `neo4j_store.py` — has to say so explicitly.
STATEMENTS: tuple[LiteralString, ...] = (
    # Uniqueness on the identity property of every label, which does two jobs at
    # once. It rejects a duplicate, and — the reason it matters more here than
    # in SQL — it backs `MERGE` with an index. An unconstrained `MERGE` on a
    # property scans, and under concurrency two transactions can both find
    # nothing and both create, which is how a graph acquires two nodes for one
    # entity without anything failing.
    "CREATE CONSTRAINT memory_id IF NOT EXISTS "
    "FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    # Not in the milestone's three statements, and here for the reason the other
    # two are: `link` merges a `Source` endpoint by `source_id`, and a merge on
    # an unconstrained property is the duplicate-creating case described above.
    # A declared label that anything merges on needs the constraint.
    "CREATE CONSTRAINT source_id IF NOT EXISTS "
    "FOR (s:Source) REQUIRE s.source_id IS UNIQUE",
    # Not unique, deliberately. Resolution is many-to-one — several entities
    # legitimately share a canonical name until M3.2 decides they are the same
    # thing — so uniqueness here would reject the state the resolver needs to
    # work from. This is a lookup index, not a constraint.
    "CREATE INDEX entity_canonical IF NOT EXISTS "
    "FOR (e:Entity) ON (e.canonical_name)",
)

_RECORD_VERSION: LiteralString = (
    "MERGE (v:SchemaVersion {scope: $scope}) "
    "SET v.version = $version, v.applied_at = datetime() "
    "RETURN v.version AS version"
)

_READ_VERSION: LiteralString = (
    "MATCH (v:SchemaVersion {scope: $scope}) RETURN v.version AS version"
)


def identity_property(label: GraphLabel) -> str:
    """The property a node of this label is merged and matched on."""
    return IDENTITY_PROPERTY[label]


async def apply_schema(driver: AsyncDriver, *, database: str | None = None) -> int:
    """Apply every constraint and index, then record the version. Idempotent.

    Statements run one at a time rather than in one transaction, and that is
    forced rather than chosen: Neo4j refuses to mix schema changes with other
    work in a single transaction. Running them separately is safe here because
    each is independently `IF NOT EXISTS` — a run that dies halfway leaves a
    subset applied, and the next run finishes the job.

    The version is written last, so a partial application never claims to be a
    complete one.
    """
    for statement in STATEMENTS:
        await driver.execute_query(statement, database_=database)

    await driver.execute_query(
        _RECORD_VERSION,
        {"scope": SCHEMA_SCOPE, "version": SCHEMA_VERSION},
        database_=database,
    )
    logger.debug("graph.schema_applied", version=SCHEMA_VERSION)
    return SCHEMA_VERSION


async def read_schema_version(
    driver: AsyncDriver, *, database: str | None = None
) -> int | None:
    """The version this database reports, or None if the schema was never applied.

    `None` is a distinct answer from a stale number and is treated as one by
    `doctor`: an empty database is a normal state that the next connect fixes,
    whereas a version behind the code is a database somebody has been writing to
    under constraints that no longer match what the code assumes.
    """
    result = await driver.execute_query(
        _READ_VERSION, {"scope": SCHEMA_SCOPE}, database_=database
    )
    if not result.records:
        return None
    version = result.records[0]["version"]
    return int(version) if version is not None else None
