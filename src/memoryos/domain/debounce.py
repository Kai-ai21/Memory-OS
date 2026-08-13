"""One event per burst of file activity, not one per keystroke.

A watcher on a directory somebody is actually working in fires constantly. An
editor writing a file produces several filesystem events for one save — a
truncate, a write, a rename, an attribute change, depending on the editor and the
platform — and a person editing produces a save every few seconds. Forwarding all
of that would put hundreds of events an hour into a queue whose whole purpose is
to trigger expensive work.

**Leading edge, not trailing edge, and that is the decision worth recording.**
The obvious debounce waits for the activity to stop and then emits, which is
right when the thing being debounced is expensive and the *last* value is the
one that matters. Here it is exactly wrong. What the event triggers is context
assembly, and context is worth having when you *start* working on a file — a
trailing debounce would deliver it thirty seconds after you stopped, which is
after you have moved on, which is the one thing M6.1 said makes the whole feature
worthless.

So the first touch of a burst emits immediately and the rest of the window is
silent. That is still one event per burst; it just picks the other end of it.

Pure, with the clock passed in. The alternative — reading `time.monotonic()`
inside — makes every test of this either sleep or monkeypatch, and a debounce
tested with sleeps is a test that is slow and flaky in proportion to how
carefully it is written.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

DEFAULT_WINDOW = timedelta(seconds=30)


@dataclass(slots=True)
class Debounce:
    """Which paths may emit now, given what has emitted recently.

    Keyed on whatever string the caller uses — a path, in practice. Two files
    saved together are two bursts, not one, because they are two units of work
    and M6.0's dedupe index would not collapse them either.
    """

    window: timedelta = DEFAULT_WINDOW
    _last: dict[str, datetime] = field(default_factory=dict)

    def should_emit(self, key: str, now: datetime) -> bool:
        """True on the first touch of a burst, False for the rest of the window.

        Records the emission, so a caller that ignores the answer and emits
        anyway has still consumed the window. That is the safer way round: the
        failure mode of a caller forgetting to check is one extra event, not an
        unbounded stream.
        """
        self._prune(now)
        last = self._last.get(key)
        if last is not None and now - last < self.window:
            return False
        self._last[key] = now
        return True

    def _prune(self, now: datetime) -> None:
        """Forget keys whose window has passed.

        Without this the map grows for the lifetime of the process — one entry
        per file ever touched — which on a large tree is a slow leak in a
        long-running watcher, and the sort that only shows up on the machine
        where the tool is left running for a week.
        """
        cutoff = now - self.window
        stale = [key for key, when in self._last.items() if when <= cutoff]
        for key in stale:
            del self._last[key]

    @property
    def tracked(self) -> int:
        """How many paths are currently inside their window. For the logs."""
        return len(self._last)
