"""How an event becomes a memory.

One function, called by both writers. `sync` uses it when it records an
observation; `replay` uses it when it rebuilds one from the log months later.

That is the whole reason this module exists. M1.7's requirement is that replay
reconstruct a memory "exactly as sync would have", and two implementations of
that rule would agree right up until somebody changed one of them — a drift
nothing would report, because both would keep producing plausible rows. Sharing
the function makes the guarantee structural instead of aspirational.

Everything here is a pure function of the event. No clock, no database, no
blob: `ingested_at` comes from the event's `recorded_at`, which is why a rebuild
lands on the same value rather than on the day it happened to run.
"""

from datetime import datetime
from uuid import UUID

from memoryos.domain.entities import IngestionEvent, Memory
from memoryos.domain.values import MemoryKind

# Deliberately small. Classifying content properly is a parsing concern, and the
# parser that actually reads the bytes overwrites this during normalization; a
# suffix is an honest first guess and nothing more.
_KIND_BY_SUFFIX = {
    ".md": MemoryKind.NOTE,
    ".txt": MemoryKind.NOTE,
    ".py": MemoryKind.CODE,
    ".pdf": MemoryKind.DOCUMENT,
}


def kind_for(external_key: str) -> MemoryKind:
    dot = external_key.rfind(".")
    suffix = external_key[dot:].lower() if dot > 0 else ""
    return _KIND_BY_SUFFIX.get(suffix, MemoryKind.OTHER)


class MalformedEvent(ValueError):
    """An event that cannot be projected, with enough detail to find it."""


def recorded_at_of(event: IngestionEvent) -> datetime:
    """When the log recorded this event.

    Typed optional on the entity because an unsaved event has not been recorded
    yet, and narrowed here rather than asserted at each call site. Every value
    derived from an event is stamped with this, so a null reaching a projection
    is worth a named error rather than a `None` propagating into a column.
    """
    if event.recorded_at is None:
        raise MalformedEvent(
            f"event {event.id} for {event.external_key!r} has no recorded_at, so "
            f"nothing derived from it can be stamped reproducibly"
        )
    return event.recorded_at


def memory_from_event(event: IngestionEvent, *, memory_id: UUID) -> Memory:
    """The memory one `artifact_observed` event implies.

    `version` and `is_current` are left at their defaults on purpose: the
    repository assigns the version from what is already stored, so that a rebuild
    derives 1..N from the order of the log rather than from anything a caller
    believed.
    """
    if event.content_hash is None:
        # Only deletion events carry no hash, and they do not project to a new
        # version. Reaching here means the caller routed an event by the wrong
        # branch, which is worth failing loudly over.
        raise MalformedEvent(
            f"event {event.id} ({event.event_type.value}) at seq {event.seq} for "
            f"{event.external_key!r} has no content_hash, so no memory can be "
            f"built from it"
        )

    return Memory(
        id=memory_id,
        source_id=event.source_id,
        external_key=event.external_key,
        content_hash=event.content_hash,
        kind=kind_for(event.external_key),
        occurred_at=event.occurred_at,
        occurred_at_source=event.occurred_at_source,
        # When this system learned about the item — which is precisely what the
        # log recorded, not what the clock says during a rebuild.
        ingested_at=event.recorded_at,
        meta={"media_type": event.payload.get("media_type")},
    )
