"""What reaches the API from a tree somebody is working in, and what does not.

Two of M6.2's three required properties, both decidable without a filesystem or
a network: a burst of file events collapses to one, and an excluded path never
emits at all. The third — that the extension survives an API error — is
`clients/vscode/src/test/client.test.ts`, because it is a property of the
TypeScript.

Every test drives `WatchTree.handle` directly with an injected clock and a
stubbed HTTP client. The alternative is writing files and sleeping, which tests
watchdog's ability to notice a write — somebody else's code — and does it
slowly and flakily.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from memoryos.application.watcher import (
    WatchTree,
    external_key_for,
    is_watchable,
)
from memoryos.domain.debounce import Debounce

NOW = datetime(2026, 8, 14, 10, 0, tzinfo=UTC)
ROOT = Path("/repo")


def transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def accepting() -> httpx.AsyncClient:
    return transport(lambda request: httpx.Response(202, json={"created": True}))


def watcher(client: httpx.AsyncClient, **kwargs: Any) -> WatchTree:
    return WatchTree(ROOT, api_url="http://localhost:8000", client=client, **kwargs)


# --------------------------------------------------------------------------
# 1. A burst collapses to one event
# --------------------------------------------------------------------------


async def test_a_burst_of_saves_produces_one_event() -> None:
    """Ten events for one file inside the window is one unit of work.

    An editor writing a file produces several filesystem events for a single
    save — truncate, write, rename, attribute change, depending on the editor
    and the platform — and a person editing saves every few seconds. Forwarding
    all of it would put hundreds of events an hour into a queue whose purpose is
    to trigger a second of compute each time.
    """
    posted: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(202, json={"created": True})

    watch = watcher(transport(record), window=timedelta(seconds=30))

    for offset in range(10):
        await watch.handle(
            "/repo/src/a.py", now=NOW + timedelta(seconds=offset * 2)
        )

    assert watch.report.emitted == 1
    assert watch.report.debounced == 9
    assert len(posted) == 1
    # The path is relative, because that is the corpus's `external_key` — an
    # absolute path would be a focus that matches nothing by name.
    assert posted[0]["payload"]["path"] == "src/a.py"
    # And the dedupe key is the path, so M6.0's index is the backstop behind
    # this one.
    assert posted[0]["dedupe_key"] == "src/a.py"


async def test_the_first_event_of_a_burst_emits_immediately() -> None:
    """Leading edge, and this is the test that pins it.

    A trailing debounce would deliver context thirty seconds after you stopped
    editing, which is after you have moved on — the one thing M6.1 said makes
    the whole feature worthless. The first save emits and the window is silent
    afterwards, which is still one event per burst from the other end.
    """
    watch = watcher(accepting(), window=timedelta(seconds=30))

    assert await watch.handle("/repo/src/a.py", now=NOW) is True


async def test_the_window_reopens_once_it_passes() -> None:
    # Coming back to a file after half a minute is genuinely new work, and a
    # debounce that never reopened would make the watcher fire once per file per
    # process lifetime.
    watch = watcher(accepting(), window=timedelta(seconds=30))

    await watch.handle("/repo/src/a.py", now=NOW)
    await watch.handle("/repo/src/a.py", now=NOW + timedelta(seconds=31))

    assert watch.report.emitted == 2


async def test_two_files_saved_together_are_two_bursts() -> None:
    # Keyed per path, because two files are two units of work and M6.0's index
    # would not collapse them either.
    watch = watcher(accepting(), window=timedelta(seconds=30))

    await watch.handle("/repo/src/a.py", now=NOW)
    await watch.handle("/repo/src/b.py", now=NOW)

    assert watch.report.emitted == 2


def test_the_debounce_forgets_paths_once_their_window_passes() -> None:
    """Otherwise the map grows for the life of the process.

    One entry per file ever touched is a slow leak that only shows up on the
    machine where the tool is left running for a week, which is the machine
    nobody is debugging on.
    """
    debounce = Debounce(window=timedelta(seconds=30))
    for index in range(100):
        debounce.should_emit(f"src/{index}.py", NOW)
    assert debounce.tracked == 100

    debounce.should_emit("src/later.py", NOW + timedelta(seconds=31))

    assert debounce.tracked == 1


# --------------------------------------------------------------------------
# 2. Excluded paths never emit
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/repo/.git/objects/ab/cdef",
        "/repo/node_modules/left-pad/index.py",
        "/repo/.venv/lib/python3.12/site.py",
        "/repo/src/__pycache__/a.cpython-312.pyc",
        "/repo/var/blobs/ab/cdef",
        # In the tree and not in the corpus: the include list is markdown, text,
        # Python and PDF, so a lock file is not watchable even though nothing
        # excludes it by name.
        "/repo/package-lock.json",
        # Outside the root entirely.
        "/elsewhere/src/a.py",
    ],
)
async def test_an_excluded_path_never_emits(path: str) -> None:
    """A watcher firing on `.git` internals is worse than no watcher.

    Worse rather than merely noisy: a rebase rewrites hundreds of objects in a
    second, each one would clear the rate limit's budget for the whole minute,
    and the events that mattered — the files you then edit — would be the ones
    refused.
    """
    posted: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        posted.append(str(request.url))
        return httpx.Response(202, json={})

    watch = watcher(transport(record))

    assert await watch.handle(path, now=NOW) is False
    assert posted == []
    assert watch.report.filtered == 1
    assert watch.report.emitted == 0


async def test_an_included_path_does_emit() -> None:
    # The positive control. Without it every exclusion test above passes on a
    # watcher that emits nothing at all.
    watch = watcher(accepting())

    assert await watch.handle("/repo/src/memoryos/cli.py", now=NOW) is True


def test_exclusions_win_over_inclusions() -> None:
    """`**/var/**` and `**/*.py` both match, and the exclusion decides.

    It has to: the connector applies the same precedence, and a watcher that
    disagreed would fire on files the sync will never ingest — every one of them
    assembling context about a file the corpus does not contain.
    """
    assert not is_watchable(
        "var/hf/models/tokenizer.py",
        include=["**/*.py"],
        exclude=["**/var/**"],
    )


def test_a_path_outside_the_root_has_no_key() -> None:
    assert external_key_for(Path("/elsewhere/a.py"), ROOT) is None
    assert external_key_for(Path("/repo/src/a.py"), ROOT) == "src/a.py"


# --------------------------------------------------------------------------
# Quiet failure
# --------------------------------------------------------------------------


async def test_the_api_being_down_is_not_an_error() -> None:
    """The constraint the whole module exists for.

    A dev tool that throws while you are trying to work is uninstalled the same
    day, and the API being stopped is the normal state of a laptop rather than
    an exception.
    """

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    watch = watcher(transport(refuse))

    assert await watch.handle("/repo/src/a.py", now=NOW) is False
    assert watch.report.failed == 1


@pytest.mark.parametrize("status", [400, 422, 429, 500, 503])
async def test_every_refusal_is_absorbed(status: int) -> None:
    watch = watcher(transport(lambda request: httpx.Response(status, json={})))

    assert await watch.handle("/repo/src/a.py", now=NOW) is False
    assert watch.report.failed == 1


async def test_a_failure_still_consumes_the_debounce_window() -> None:
    """Otherwise a stopped API turns the watcher into a retry loop.

    Every save would re-try immediately, at whatever rate the editor writes,
    against a server that is not there. Consuming the window on failure means a
    dead API costs one attempt per file per window — which is what the backoff
    is then layered on top of.
    """
    attempts = 0

    def refuse(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("connection refused")

    watch = watcher(transport(refuse), window=timedelta(seconds=30))

    for offset in range(5):
        await watch.handle("/repo/src/a.py", now=NOW + timedelta(seconds=offset))

    assert attempts == 1
    assert watch.report.debounced == 4
