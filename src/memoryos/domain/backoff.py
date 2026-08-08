"""Retry backoff."""

import random
from collections.abc import Callable


def compute_backoff(
    attempts: int,
    *,
    base_seconds: float = 2.0,
    cap_seconds: float = 600.0,
    rand: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait before the next attempt.

    Exponential in `attempts`, capped, then multiplied by a random factor in
    [0.5, 1.0).

    The jitter is not decoration. A thousand jobs that failed at the same moment
    — because one dependency went down — retry at the same moment without it,
    then again together, and again: a perfectly synchronised thundering herd
    that can hold a recovering dependency down indefinitely. Spreading the
    retries is what lets it recover.

    `rand` is injected so the jitter is deterministic under test.
    """
    delay = min(base_seconds * (2.0**attempts), cap_seconds)
    return delay * (0.5 + rand() * 0.5)
