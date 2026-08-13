"""External events, and the rule that stops one client filling the queue.

**Phase 6 inverts the direction of this system and this file is where that
starts.** Everything through Phase 5 is pull: somebody asks, the system answers,
and a bad answer wastes one query. From here it is push: something happens, the
system decides work is worth doing, and it does it without being asked. A bad
push wastes compute continuously and teaches its reader to ignore the output,
which is the more expensive failure by a wide margin — the first costs a query,
the second costs the product.

Nothing in M6.0 predicts anything, so nothing here can be wrong in that way yet.
What this milestone can get wrong is volume, and the two rules below are the
whole defence.

**Deduplication.** An editor firing a focus event on every keystroke is not ten
units of work; it is one, repeated. The `events` table carries a partial unique
index on `(kind, dedupe_key)` restricted to unprocessed rows, so the tenth
delivery of the same focus collides with the first and the queue never sees it.
Restricted to *unprocessed* rows deliberately: the same file focused again
tomorrow is genuinely new work, and an index over all rows would refuse it
forever.

**Rate limiting.** Deduplication only helps a client that sets a dedupe key. A
misbehaving plugin that sets none, or sets a fresh one each time, is stopped by
counting instead — see `RateLimit`.

Bitemporal again, and M1.1's rules apply unchanged. `occurred_at` is when the
thing happened, `received_at` is when this system heard about it, and an editor
event delivered after a network hiccup happened earlier than it arrived. The gap
between them is the queue's real input latency, and M6.1 depends on it: context
that arrives after the meeting has started is worth nothing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum, auto
from typing import Any
from uuid import UUID


class EventKind(StrEnum):
    """What happened outside, in the four shapes this system accepts.

    Closed, and validated at the edge rather than stored and sorted out later.
    An unknown kind is refused with the list of known ones rather than written —
    a row nothing subscribes to is a row that sits unprocessed forever, and a
    queue full of those is indistinguishable from a queue that is behind.

    `EDITOR_OPENED` and `FILE_FOCUSED` are the high-volume pair and the reason
    both defences above exist. `MEETING_UPCOMING` is the one with a deadline
    attached, and the one whose latency actually matters. `MANUAL` is a person
    saying "do it now", which is also what makes the pipeline testable without
    a plugin.
    """

    EDITOR_OPENED = auto()
    FILE_FOCUSED = auto()
    MEETING_UPCOMING = auto()
    MANUAL = auto()


class UnknownEventKind(ValueError):
    """A kind that is not in the enum. Refused at the edge, never stored."""

    def __init__(self, kind: str) -> None:
        known = ", ".join(member.value for member in EventKind)
        super().__init__(f"unknown event kind {kind!r}; expected one of: {known}")
        self.kind = kind


def parse_kind(raw: str) -> EventKind:
    """`raw` as an `EventKind`, or a refusal naming every kind that exists.

    Written here rather than left to a schema validator because the *message*
    is the point. "value is not a valid enumeration member" tells a plugin
    author nothing; the list of four tells them what to send.
    """
    try:
        return EventKind(raw)
    except ValueError as exc:
        raise UnknownEventKind(raw) from exc


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened outside this system.

    `processed_at` is null until a handler has run to completion. It is *not*
    set when the event is dispatched: dispatch only enqueues, and an event whose
    jobs are all still pending has had nothing done about it. Marking it
    processed at dispatch would make the latency number in `events stats`
    measure the speed of an INSERT, which is the one number nobody needs.
    """

    id: UUID
    kind: EventKind
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime | None = None
    received_at: datetime | None = None
    processed_at: datetime | None = None
    dedupe_key: str | None = None

    @property
    def delivery_lag(self) -> timedelta | None:
        """How long the event took to reach this system.

        Distinct from the processing latency and worth separating. A slow queue
        is this system's fault; a slow delivery is the network's, and a report
        that added them together would blame the wrong component.
        """
        if self.occurred_at is None or self.received_at is None:
            return None
        return self.received_at - self.occurred_at


@dataclass(frozen=True, slots=True)
class RateLimit:
    """How many events one source may deliver in a window.

    **Counted against the stored rows rather than held in memory, and that is
    the decision worth recording.** A token bucket in the API process is per
    replica and resets on every deploy, while the queue it protects is shared by
    all of them — so two replicas would allow twice the limit and a restart
    would allow it again immediately. Counting rows costs one indexed query per
    POST and cannot be wrong about what was actually accepted.

    Per source, not global. The point is to stop one misbehaving plugin, and a
    global limit would let that plugin lock out every well-behaved client.
    """

    limit: int
    window: timedelta

    def exceeded(self, received_in_window: int) -> bool:
        """Whether a source that has already delivered this many must wait.

        `>=` rather than `>`: the count is of events already stored, so a source
        that has delivered exactly `limit` has used its allowance.
        """
        return received_in_window >= self.limit

    def retry_after(self) -> int:
        """Seconds to tell the client to wait, rounded up to the whole window.

        Deliberately coarse. The exact moment the oldest event leaves the window
        is computable, and publishing it invites a client to poll for it — which
        is the same load the limit exists to refuse.
        """
        return max(1, int(self.window.total_seconds()))
