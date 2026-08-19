"""`memoryos auth`, driven through the same entry point a terminal uses.

**The password is patched in at `getpass`, which is the point of the test.**
There is no `--password` flag to pass one through, and there will not be: an
argument is written to shell history and is visible in `ps` to every other user
on the machine for as long as the process runs. Patching the prompt is the only
way to exercise this without introducing the hole.
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from memoryos import cli
from memoryos.adapters.db import models

pytestmark = pytest.mark.integration

EMAIL = "cli@example.invalid"
GOOD = "a perfectly adequate password"


def run_cli(argv: list[str]) -> int:
    """Run `cli.main` off the test's event loop.

    `main` calls `asyncio.run`, which refuses to start inside a thread that
    already has a running loop — and under `asyncio_mode = auto` every test
    does. The same shape as `run_alembic` in the root conftest, and for the same
    reason.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(cli.main, argv).result()


@pytest.fixture
def answers(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Queue the replies `getpass` will hand back, in order."""
    queued: list[str] = []
    # Patched by name on the module rather than through `cli.getpass`, which
    # mypy correctly refuses: a re-exported import is not part of a module's
    # interface, and reaching through one is how a test comes to depend on an
    # import somebody is free to remove.
    monkeypatch.setattr("getpass.getpass", lambda _prompt="": queued.pop(0))
    yield queued


async def test_create_user_prompts_twice_and_creates_the_account(
    settings: object, clean_database: None, answers: list[str], session: AsyncSession
) -> None:
    answers.extend([GOOD, GOOD])

    code = run_cli(["auth", "create-user", "--email", EMAIL])

    assert code == 0
    assert not answers, "both prompts should have been consumed"
    user = (await session.execute(select(models.User))).scalar_one()
    assert user.email == EMAIL
    # The hash, not the password. Asserted because it is the whole point.
    assert GOOD not in user.password_hash
    assert user.password_hash.startswith("$argon2id$")


async def test_mismatched_confirmation_creates_nothing(
    settings: object, clean_database: None, answers: list[str], session: AsyncSession
) -> None:
    """The prompt is invisible, so a typo is undetectable without this.

    The failure it prevents is an account whose password nobody knows — which on
    a single-user system with no email reset is a reinstall.
    """
    answers.extend([GOOD, "something else entirely"])

    assert run_cli(["auth", "create-user", "--email", EMAIL]) == 1
    assert (await session.execute(select(models.User))).first() is None


async def test_a_short_password_is_refused_before_anything_is_written(
    settings: object, clean_database: None, answers: list[str], session: AsyncSession
) -> None:
    answers.append("short")

    assert run_cli(["auth", "create-user", "--email", EMAIL]) == 1
    assert (await session.execute(select(models.User))).first() is None


async def test_a_second_account_is_refused(
    settings: object, clean_database: None, answers: list[str], session: AsyncSession
) -> None:
    answers.extend([GOOD, GOOD, GOOD, GOOD])
    assert run_cli(["auth", "create-user", "--email", EMAIL]) == 0

    assert run_cli(["auth", "create-user", "--email", "other@example.invalid"]) == 1
    assert len((await session.execute(select(models.User))).scalars().all()) == 1


async def test_reset_password_changes_the_hash_and_revokes_sessions(
    settings: object, clean_database: None, answers: list[str], session: AsyncSession
) -> None:
    answers.extend([GOOD, GOOD, "a different long password", "a different long password"])
    assert run_cli(["auth", "create-user", "--email", EMAIL]) == 0
    before = (await session.execute(select(models.User))).scalar_one().password_hash

    assert run_cli(["auth", "reset-password", "--email", EMAIL]) == 0

    session.expire_all()
    after = (await session.execute(select(models.User))).scalar_one().password_hash
    assert after != before


async def test_there_is_no_password_argument(capsys: pytest.CaptureFixture[str]) -> None:
    """The absence is the feature, so it is pinned.

    If somebody adds `--password` for convenience, this fails and the reason is
    two lines above it.
    """
    with pytest.raises(SystemExit):
        # `main` directly rather than through the thread helper: argparse exits
        # during parsing, before any event loop is involved.
        cli.main(["auth", "create-user", "--email", EMAIL, "--password", "anything"])
    assert "unrecognized arguments: --password" in capsys.readouterr().err
