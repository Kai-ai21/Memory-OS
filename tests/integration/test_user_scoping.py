"""Two accounts, and the boundary between them.

**These are the milestone.** Everything else M11.1 did — a column on fourteen
tables, an index rebuild, a role that is not a superuser, a `user_id` on every
Cypher pattern — exists so that these pass. A scoping change with no cross-user
test is a scoping change nobody has checked.

Every test here drives *two* users, which is the only way to observe the
property: a suite that runs as one account cannot tell isolation from a
coincidence, and until this milestone the whole suite was one account.

The last one is the important one. Five tests show that the code we wrote does
not cross the boundary; the sixth shows that code we *forget* to write cannot
either, which is the entire argument for choosing row-level security over a
repository base class.
"""

import hashlib
from collections.abc import AsyncIterator
from uuid import UUID

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.scoping import CURRENT_USER_ID, scoped_to
from memoryos.domain.ids import new_id
from memoryos.domain.values import SourceKind

pytestmark = pytest.mark.integration


@pytest.fixture
async def second_user(engine: AsyncEngine, clean_database: None) -> AsyncIterator[UUID]:
    """A second account, beside the one `clean_database` already made.

    Inserted with the engine rather than a scoped session because `users` is one
    of the two tables with no policy on it — a login has to read a user row
    before it knows which user it is, so identity cannot be scoped by identity.
    """
    async with engine.begin() as connection:
        other = (
            await connection.execute(
                text(
                    "INSERT INTO users (id, email, password_hash) "
                    "VALUES (gen_random_uuid(), 'b@example.invalid', 'x') RETURNING id"
                )
            )
        ).scalar_one()
    yield other


def sessions_for(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def make_memory(
    engine: AsyncEngine, owner: UUID, *, external_key: str, content: str
) -> tuple[UUID, UUID]:
    """A source, a memory and a chunk owned by `owner`. Returns their ids.

    Written through the ordinary session path with nothing but `scoped_to`
    around it — no explicit `user_id` anywhere — because that is the claim: the
    column default and the policy do the work, and application code that has
    never heard of a user still writes rows with the right owner.
    """
    # Content-addressed and shared: `raw_artifacts` is deliberately *not*
    # scoped, because a hash is a hash — two users who store the same bytes
    # store them once. `memories.content_hash` is a foreign key into it, so it
    # has to exist before a memory can.
    digest = hashlib.sha256(content.encode()).hexdigest()
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "INSERT INTO raw_artifacts (content_hash, byte_size) "
                "VALUES (:hash, :size) ON CONFLICT DO NOTHING"
            ),
            {"hash": digest, "size": len(content)},
        )

    with scoped_to(owner):
        async with sessions_for(engine).begin() as session:
            source = models.Source(
                id=new_id(), kind=SourceKind.FILESYSTEM.value, name="corpus", config={}
            )
            session.add(source)
            await session.flush()
            memory = models.Memory(
                id=new_id(),
                source_id=source.id,
                external_key=external_key,
                kind="note",
                content=content,
                content_hash=digest,
                occurred_at_source="unknown",
                version=1,
                is_current=True,
            )
            session.add(memory)
            await session.flush()
            return source.id, memory.id


async def test_user_a_cannot_read_user_bs_memory_by_id(
    engine: AsyncEngine, second_user: UUID
) -> None:
    """Knowing the id is not authorisation.

    The most direct form of the leak and the one an API is most likely to have:
    a route that takes an id from the URL and looks it up. Under RLS the lookup
    returns nothing, so the route 404s without having been written to check.
    """
    owner = CURRENT_USER_ID.get()
    assert owner is not None
    _, theirs = await make_memory(
        engine, second_user, external_key="notes/theirs.md", content="B's private note"
    )

    with scoped_to(owner):
        async with sessions_for(engine)() as session:
            found = (
                await session.execute(
                    select(models.Memory).where(models.Memory.id == theirs)
                )
            ).scalar_one_or_none()

    assert found is None

    # And it is genuinely there — the query is right, the row is hidden.
    with scoped_to(second_user):
        async with sessions_for(engine)() as session:
            assert (
                await session.execute(
                    select(models.Memory).where(models.Memory.id == theirs)
                )
            ).scalar_one_or_none() is not None


async def test_search_as_a_never_returns_bs_content(
    engine: AsyncEngine, second_user: UUID
) -> None:
    """The unqualified list, which is what most read paths actually are.

    `SELECT * FROM memories` is the shape of a dozen queries in this codebase.
    None of them mention a user; all of them are scoped, because the policy is
    not in the query.
    """
    owner = CURRENT_USER_ID.get()
    assert owner is not None
    await make_memory(engine, owner, external_key="mine.md", content="postgres notes")
    await make_memory(
        engine, second_user, external_key="theirs.md", content="postgres notes"
    )

    with scoped_to(owner):
        async with sessions_for(engine)() as session:
            mine = (await session.execute(select(models.Memory))).scalars().all()

    assert [memory.external_key for memory in mine] == ["mine.md"]
    assert all(memory.user_id == owner for memory in mine)


async def test_the_graph_as_a_never_returns_bs_entities(
    engine: AsyncEngine, second_user: UUID, graph: object
) -> None:
    """Neo4j has no policies, so isolation is a property of every pattern.

    Written against the real store rather than the fake, because the fake is the
    thing that would not have the bug: what is under test is the Cypher.
    """
    from memoryos.application.ports import EntityNode

    owner = CURRENT_USER_ID.get()
    assert owner is not None
    store = graph.store  # type: ignore[attr-defined]

    mine, theirs = new_id(), new_id()
    with scoped_to(owner):
        await store.upsert_entity(
            EntityNode(entity_id=mine, name="Postgres", canonical_name="postgres",
                       type="TECHNOLOGY", confidence=1.0)
        )
    with scoped_to(second_user):
        await store.upsert_entity(
            EntityNode(entity_id=theirs, name="Postgres", canonical_name="postgres",
                       type="TECHNOLOGY", confidence=1.0)
        )

    with scoped_to(owner):
        nodes = await store.all_nodes()
    keys = {node.key for node in nodes}
    assert str(mine) in keys
    assert str(theirs) not in keys

    # The same name and the same canonical form on both sides: two nodes, not
    # one shared one, because `user_id` is part of the merge identity.
    with scoped_to(second_user):
        assert {node.key for node in await store.all_nodes()} == {str(theirs)}


async def test_a_replay_for_a_does_not_touch_bs_data(
    engine: AsyncEngine, second_user: UUID
) -> None:
    """Truncation is the one operation RLS does *not* filter.

    `TRUNCATE` ignores policies entirely — it is a table-level operation, not a
    row-level one — so a replay that truncated `memories` would take both users'
    rows however well scoped the rest of it is. Replay therefore deletes rather
    than truncates when it is scoped, and this is the test that says so.
    """
    owner = CURRENT_USER_ID.get()
    assert owner is not None
    await make_memory(engine, owner, external_key="mine.md", content="mine")
    _, theirs = await make_memory(
        engine, second_user, external_key="theirs.md", content="theirs"
    )

    # What a scoped rebuild does: delete this user's rows. The policy turns an
    # unqualified DELETE into a scoped one, which is the property under test.
    with scoped_to(owner):
        async with sessions_for(engine).begin() as session:
            await session.execute(text("DELETE FROM memories"))

    with scoped_to(second_user):
        async with sessions_for(engine)() as session:
            survivors = (await session.execute(select(models.Memory))).scalars().all()

    assert [memory.id for memory in survivors] == [theirs]


async def test_two_users_can_have_a_file_at_the_same_path(
    engine: AsyncEngine, second_user: UUID
) -> None:
    """The unique constraint that had to be rebuilt, from the outside.

    `(source_id, external_key, version)` was globally unique until M11.1, so the
    second person to have a `README.md` would have collided with the first. It
    is `(user_id, source_id, external_key, version)` now.
    """
    owner = CURRENT_USER_ID.get()
    assert owner is not None
    _, mine = await make_memory(engine, owner, external_key="README.md", content="mine")
    _, theirs = await make_memory(
        engine, second_user, external_key="README.md", content="theirs"
    )

    assert mine != theirs
    for who, expected in ((owner, "mine"), (second_user, "theirs")):
        with scoped_to(who):
            async with sessions_for(engine)() as session:
                found = (
                    await session.execute(
                        select(models.Memory).where(
                            models.Memory.external_key == "README.md"
                        )
                    )
                ).scalar_one()
        assert found.content == expected


async def test_a_deliberately_unscoped_query_is_caught_by_the_database(
    engine: AsyncEngine, second_user: UUID
) -> None:
    """**The test that justifies the mechanism.**

    Five tests above show that the code we wrote does not cross the boundary.
    This one shows that code we *forget* to write cannot either — which is the
    whole argument for row-level security over a repository base class, since a
    base class only binds the queries that go through it.

    Three forms, because they are three different ways to forget:

      * a `SELECT` with no `WHERE user_id`, issued on a connection that never
        said who it was — returns nothing rather than everything;
      * an `INSERT` on that connection — refused, because the column default
        resolves to NULL and the column is NOT NULL;
      * an `INSERT` naming *somebody else's* id explicitly — refused by the
        policy's `WITH CHECK`, which is the deliberate-malice case rather than
        the forgetful one.
    """
    owner = CURRENT_USER_ID.get()
    assert owner is not None
    await make_memory(engine, owner, external_key="mine.md", content="mine")
    await make_memory(engine, second_user, external_key="theirs.md", content="theirs")

    # 1. Forgotten scope: an unqualified read sees nothing at all.
    with scoped_to(None):
        async with sessions_for(engine)() as session:
            rows = (await session.execute(text("SELECT id FROM memories"))).all()
    assert rows == [], "an unscoped connection could read rows"

    # 2. Forgotten scope on a write: refused rather than silently unowned.
    with scoped_to(None):
        async with sessions_for(engine)() as session:
            with pytest.raises(DBAPIError):
                await session.execute(
                    text(
                        "INSERT INTO sources (id, kind, name, config) "
                        "VALUES (gen_random_uuid(), 'filesystem', 'x', '{}')"
                    )
                )
                await session.commit()

    # 3. Writing into somebody else's account, on purpose.
    with scoped_to(owner):
        async with sessions_for(engine)() as session:
            with pytest.raises(DBAPIError):
                await session.execute(
                    text(
                        "INSERT INTO sources (id, user_id, kind, name, config) "
                        "VALUES (gen_random_uuid(), :other, 'filesystem', 'y', '{}')"
                    ),
                    {"other": second_user},
                )
                await session.commit()


async def test_the_application_role_is_not_a_superuser(engine: AsyncEngine) -> None:
    """The one-line check that the rest of this file is not theatre.

    Row-level security is skipped entirely for superusers and for BYPASSRLS
    roles, and `FORCE ROW LEVEL SECURITY` does not change that. Until M11.1 the
    application connected as the owning superuser — every policy above would
    have been enabled, listed in `pg_policies`, and enforcing nothing, and every
    test in this file would have failed in a way that looked like a bug in the
    tests.
    """
    async with engine.begin() as connection:
        role = (
            await connection.execute(
                text(
                    "SELECT rolsuper, rolbypassrls FROM pg_roles "
                    "WHERE rolname = current_user"
                )
            )
        ).one()
    assert role.rolsuper is False, "the application connects as a superuser; RLS is off"
    assert role.rolbypassrls is False, "the application role holds BYPASSRLS; RLS is off"
