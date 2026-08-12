"""Retry backoff."""

import random
from collections.abc import Callable

# The longest a caller will wait on a provider's own advice. A 429 that asks for
# ten minutes is a daily quota rather than a sliding window, and sleeping through
# it inside one command would look exactly like a hang.
MAX_ADVISED_WAIT_SECONDS = 120.0


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


def wait_for(
    exc: Exception,
    attempts: int,
    *,
    rand: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait, preferring what the provider said over what we would guess.

    `retry_after` on the exception wins when it is there, because it is
    information and `compute_backoff` is an estimate. A token-per-minute limit is
    the case that makes the difference stark: the window slides, the server knows
    how far, and exponential backoff from a 2s base spends four attempts inside a
    window it was told the length of — then fails work that would have succeeded.

    Jitter is applied either way, because the reason for jitter is unrelated to the
    reason for the wait: a hundred callers told "come back in 23s" all come back in
    the same millisecond otherwise. A small floor is added on top of an advised
    wait, since a server saying "0s" means "the window has just moved", not "now".

    Capped, because a provider asking for ten minutes is describing a daily quota,
    and sleeping through one inside a foreground command is indistinguishable from
    a hang. Above the cap the caller is better off failing and being re-run.
    """
    advised = getattr(exc, "retry_after", None)
    if advised is None:
        return compute_backoff(attempts, rand=rand)
    bounded = min(float(advised), MAX_ADVISED_WAIT_SECONDS)
    return bounded + 0.5 + rand()
