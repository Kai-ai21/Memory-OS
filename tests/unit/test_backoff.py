import pytest

from memoryos.domain.backoff import compute_backoff


def fixed(value: float) -> object:
    return lambda: value


def undithered(attempts: int, base: float = 2.0, cap: float = 600.0) -> float:
    return min(base * (2.0**attempts), cap)


@pytest.mark.parametrize("attempts", range(6))
def test_delay_doubles_with_each_attempt(attempts: int) -> None:
    # rand() == 1.0 removes the jitter, leaving the undithered curve.
    delay = compute_backoff(attempts, rand=lambda: 1.0)
    assert delay == pytest.approx(undithered(attempts))


def test_delay_grows_monotonically_below_the_cap() -> None:
    delays = [compute_backoff(n, rand=lambda: 1.0) for n in range(8)]
    assert delays == sorted(delays)
    assert delays[0] < delays[-1]


def test_delay_is_capped() -> None:
    assert compute_backoff(1000, rand=lambda: 1.0) == pytest.approx(600.0)
    assert compute_backoff(50, cap_seconds=30.0, rand=lambda: 1.0) == pytest.approx(30.0)


@pytest.mark.parametrize("roll", [0.0, 0.25, 0.5, 0.75, 0.999999])
@pytest.mark.parametrize("attempts", [0, 3, 9])
def test_jitter_stays_within_half_to_full_of_the_undithered_delay(
    attempts: int, roll: float
) -> None:
    # The window matters more than the exact value: too little spread and a
    # thousand jobs that failed together retry together.
    delay = compute_backoff(attempts, rand=lambda: roll)
    ceiling = undithered(attempts)
    assert 0.5 * ceiling <= delay <= ceiling


def test_jitter_actually_varies() -> None:
    rolls = iter([0.0, 1.0])
    delays = [compute_backoff(4, rand=lambda: next(rolls)) for _ in range(2)]
    assert delays[0] != delays[1]


def test_base_seconds_scales_the_whole_curve() -> None:
    assert compute_backoff(3, base_seconds=1.0, rand=lambda: 1.0) == pytest.approx(8.0)
    assert compute_backoff(3, base_seconds=5.0, rand=lambda: 1.0) == pytest.approx(40.0)
