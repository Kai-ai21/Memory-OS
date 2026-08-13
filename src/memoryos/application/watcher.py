"""Watch a tree, and tell the API when a file is worked on.

The first thing in this system that runs beside you rather than being run by
you, and the constraint that shapes every line of it is not correctness — it is
that **a dev tool which throws errors while you are trying to work gets
uninstalled within a day.**

So nothing here raises. The API being down, the network being slow, a 500, a
malformed response, a rate limit: all of them are logged and backed off from, and
the watcher keeps watching. A missed event costs a context nobody asked for. A
crashed watcher costs the feature.

Three filters between a filesystem event and an HTTP request, in the order that
discards the most for the least work:

1. **The source's own globs.** Reused from the connector rather than
   reimplemented — `**/.git/**`, `**/node_modules/**`, `**/var/**` — because a
   watcher firing on git internals is worse than no watcher, and because two
   implementations of "which files are in this corpus" is two chances to
   disagree about it.
2. **The debounce**, leading edge, one event per burst per path. See
   `domain/debounce`.
3. **M6.0's dedupe index**, which is the backstop rather than the plan: the
   dedupe key is the path, so a burst that slips through the first two collapses
   to one unprocessed row anyway.

The path sent is *relative to the source root*, because that is the corpus's
`external_key`. Sending an absolute path would produce a focus that matches
nothing by name, and the by-name half of M6.1's temporal source — the only one
that can find the file you are actually looking at — would silently contribute
nothing.
"""

import asyncio
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import structlog
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from memoryos.adapters.connectors.filesystem import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    matches,
)
from memoryos.domain.debounce import DEFAULT_WINDOW, Debounce
from memoryos.domain.events import EventKind

logger = structlog.get_logger(__name__)

# How long to wait after a failed POST before trying again, and the ceiling.
#
# Doubling from a second to a minute. A watcher that retried every second
# against a dead API would fill a log file overnight and do nothing useful; one
# that gave up entirely would need restarting by hand after every `docker
# compose restart`, which is the same as not surviving one.
INITIAL_BACKOFF = timedelta(seconds=1)
MAX_BACKOFF = timedelta(seconds=60)

# How long to wait for the API. Deliberately short: the endpoint stores a row
# and enqueues, so anything slower than this is a symptom rather than a slow
# query — measured at 7ms p50 against the running stack.
REQUEST_TIMEOUT = 5.0


@dataclass(slots=True)
class WatchReport:
    """What the watcher did, for the run summary and for the tests."""

    observed: int = 0
    filtered: int = 0
    debounced: int = 0
    emitted: int = 0
    failed: int = 0
    # Paths that produced an event, newest last. Bounded, because a watcher
    # running for a week must not accumulate its own history in memory.
    recent: list[str] = field(default_factory=list)

    def note(self, path: str, *, limit: int = 20) -> None:
        self.recent.append(path)
        if len(self.recent) > limit:
            del self.recent[0]

    def as_dict(self) -> dict[str, int]:
        return {
            "observed": self.observed,
            "filtered": self.filtered,
            "debounced": self.debounced,
            "emitted": self.emitted,
            "failed": self.failed,
        }


def external_key_for(path: Path, root: Path) -> str | None:
    """The corpus key for a path, or None when it is outside the root.

    Resolved on both sides before comparing, because a watcher started at `.`
    and a source configured with an absolute path describe the same tree through
    different strings — and on macOS `/tmp` is a symlink to `/private/tmp`, so
    an unresolved comparison fails on exactly the directory tests use.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def is_watchable(
    external_key: str, *, include: Sequence[str], exclude: Sequence[str]
) -> bool:
    """Whether this key is in the corpus at all.

    Exclusions are checked first and win. A file matching both an include and an
    exclude is excluded — which is the same precedence the connector applies, and
    has to be, or the watcher would fire on files the sync will never ingest and
    every one of those events would assemble context about a file the corpus does
    not contain.
    """
    if matches(external_key, list(exclude)):
        return False
    return matches(external_key, list(include))


class _Handler(FileSystemEventHandler):
    """Bridges watchdog's thread into the event loop.

    watchdog calls this from its own thread, so nothing here may touch the loop
    directly. `call_soon_threadsafe` is the whole of the bridge, and the queue is
    unbounded on purpose: dropping a path here to bound memory would drop it
    before the debounce, which is the one place that can drop cheaply and
    knowingly.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue[str]) -> None:
        self._loop = loop
        self._queue = queue

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # `moved` carries the new name in `dest_path`; everything else in
        # `src_path`. A rename is a file being worked on under its new name.
        raw = getattr(event, "dest_path", "") or event.src_path
        path = raw.decode() if isinstance(raw, bytes) else str(raw)
        if not path:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, path)


class WatchTree:
    """Watch a directory and POST `FILE_FOCUSED` for real file activity."""

    def __init__(
        self,
        root: Path,
        *,
        api_url: str,
        source: str = "watch",
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        window: timedelta = DEFAULT_WINDOW,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.root = root.resolve()
        self._api_url = api_url.rstrip("/")
        self._source = source
        self._include = list(include if include is not None else DEFAULT_INCLUDE)
        self._exclude = list(exclude if exclude is not None else DEFAULT_EXCLUDE)
        self._debounce = Debounce(window=window)
        self._client = client
        self.report = WatchReport()
        self._backoff = INITIAL_BACKOFF

    async def handle(self, path: str, *, now: datetime | None = None) -> bool:
        """Filter, debounce and emit one raw path. True when an event was sent.

        The whole decision path, separated from the watchdog plumbing so it can
        be tested without a filesystem, a thread, or a sleep.
        """
        moment = now or datetime.now(UTC)
        self.report.observed += 1

        key = external_key_for(Path(path), self.root)
        if key is None or not is_watchable(
            key, include=self._include, exclude=self._exclude
        ):
            self.report.filtered += 1
            return False

        if not self._debounce.should_emit(key, moment):
            self.report.debounced += 1
            return False

        sent = await self._emit(key)
        if sent:
            self.report.emitted += 1
            self.report.note(key)
        else:
            self.report.failed += 1
        return sent

    async def _emit(self, key: str) -> bool:
        """POST the event. Never raises, whatever the API does.

        A `429` is counted as a failure and backed off from like any other,
        rather than being retried immediately with the `Retry-After` the API
        sends. The header says a minute; the debounce window is thirty seconds;
        and an event that is a minute stale is one the next save will replace
        anyway. Sleeping a whole minute inside the emit path would block every
        *other* file's event behind one rate-limited path.
        """
        if self._client is None:
            return False
        try:
            response = await self._client.post(
                f"{self._api_url}/events",
                json={
                    "kind": EventKind.FILE_FOCUSED.value,
                    "source": self._source,
                    "payload": {"path": key},
                    # M6.0's index collapses repeats of this key while the first
                    # is still unprocessed — the backstop behind the debounce.
                    "dedupe_key": key,
                },
                timeout=REQUEST_TIMEOUT,
            )
        except (httpx.HTTPError, OSError) as exc:
            # The API being down is the normal condition on a laptop, not an
            # exception. Logged at info for that reason: a warning per save
            # while the stack is stopped is a log nobody reads afterwards.
            logger.info("watch.unreachable", path=key, error=str(exc))
            await self._back_off()
            return False

        if response.status_code >= 400:
            logger.info(
                "watch.refused", path=key, status=response.status_code
            )
            await self._back_off()
            return False

        self._backoff = INITIAL_BACKOFF
        logger.info("watch.emitted", path=key, status=response.status_code)
        return True

    async def _back_off(self) -> None:
        await asyncio.sleep(self._backoff.total_seconds())
        self._backoff = min(self._backoff * 2, MAX_BACKOFF)

    async def run(self, *, stop: asyncio.Event | None = None) -> WatchReport:
        """Watch until interrupted. Returns what it did.

        The observer runs in its own thread and is always joined, including on
        the exception path — a watcher that left a daemon thread behind would
        keep an inotify handle open for the life of the process, and on a tree
        this size that is a file descriptor per directory.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()
        observer = Observer()
        observer.schedule(_Handler(loop, queue), str(self.root), recursive=True)
        observer.start()
        logger.info(
            "watch.started",
            root=str(self.root),
            window_seconds=int(self._debounce.window.total_seconds()),
            include=self._include,
            exclude=self._exclude,
        )

        halt = stop or asyncio.Event()
        try:
            while not halt.is_set():
                try:
                    path = await asyncio.wait_for(queue.get(), timeout=0.5)
                except TimeoutError:
                    # The wake-up that lets `stop` be noticed. Without it the
                    # watcher would sit in `queue.get()` until the next file
                    # changed, which on a quiet tree is never.
                    continue
                await self.handle(path)
        finally:
            observer.stop()
            with contextlib.suppress(RuntimeError):
                observer.join(timeout=5)
            logger.info("watch.stopped", **self.report.as_dict())
        return self.report
