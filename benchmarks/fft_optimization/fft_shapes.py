from __future__ import annotations

import itertools
import math
from typing import Iterable


SMALL_PRIMES = (2, 3, 5, 7)


def is_smooth(value: int, primes: Iterable[int] = SMALL_PRIMES) -> bool:
    value = int(value)
    if value < 1:
        return False
    for prime in primes:
        while value % prime == 0:
            value //= prime
    return value == 1


def next_smooth(value: int) -> int:
    candidate = max(1, int(value))
    while not is_smooth(candidate):
        candidate += 1
    return candidate


def next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def named_shapes(min_shape: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    return {
        "exact": tuple(int(v) for v in min_shape),
        "smooth": tuple(next_smooth(v) for v in min_shape),
        "power2": tuple(next_power_of_two(v) for v in min_shape),
    }


def candidate_shapes(
    min_shape: tuple[int, ...],
    *,
    max_padding: float = 0.15,
    max_shapes: int = 24,
) -> list[tuple[int, ...]]:
    axes: list[list[int]] = []
    for minimum in min_shape:
        stop = max(minimum, int(math.ceil(minimum * (1.0 + max_padding))))
        values = [v for v in range(minimum, stop + 1) if is_smooth(v)]
        if not values:
            values = [next_smooth(minimum)]
        axes.append(values)
    tuples = list(itertools.product(*axes))
    tuples.sort(key=lambda shape: (math.prod(shape), shape))
    if len(tuples) <= max_shapes:
        return tuples
    # Preserve the volume range rather than returning only the smallest tuples.
    indices = sorted({round(i * (len(tuples) - 1) / (max_shapes - 1)) for i in range(max_shapes)})
    return [tuples[i] for i in indices]


def padding_ratio(shape: tuple[int, ...], minimum: tuple[int, ...]) -> float:
    return math.prod(shape) / math.prod(minimum) - 1.0

