"""Greedy baselines for the cutting stock comparison.

FFD-decreasing is the standard reference heuristic: sort parts by size
descending and place each into the first bar with enough room, opening
a new bar only when none fits. It is deterministic, runs in
O(n^2), and is what a shop would do by hand.
"""

from __future__ import annotations

from typing import Sequence

__all__ = ["first_fit_decreasing"]


def first_fit_decreasing(sizes: Sequence[float], capacity: float) -> list[list[float]]:
    """Pack ``sizes`` into the fewest bars FFD finds.

    Returns one bar per bin, each a list of the part sizes placed on
    it. Deterministic: ties in size keep their input order.
    """
    bins: list[list[float]] = []
    room: list[float] = []
    for size in sorted(sizes, reverse=True):
        placed = False
        for j, free in enumerate(room):
            if free >= size:
                bins[j].append(size)
                room[j] = free - size
                placed = True
                break
        if not placed:
            bins.append([size])
            room.append(capacity - size)
    return bins
