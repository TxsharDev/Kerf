"""First-fit decreasing over bar contexts.

The greedy plan does two jobs. Its patterns seed column generation, so
the master LP is feasible from the first round instead of waiting for
pricing to discover patterns that fit the bars on hand. Its bar counts
are a feasible integer plan, so the solver always has an answer at
least as good as the greedy one, however column generation ends.

Parts are placed longest first. Each part goes to the first open bar
with room for it; when no open bar has room, the next unopened bar it
fits on is opened. Bars open cheapest per millimetre first, so free
remnants are cut before purchased stock, and in context order (oldest
remnant first) within a cost class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from constraint_engine.geometry.master import MasterColumn
from constraint_engine.geometry.patterns import Context
from constraint_engine.geometry.types import CuttingProblem1D

__all__ = ["GreedyPlan", "first_fit_decreasing"]


@dataclass(frozen=True)
class GreedyPlan:
    """A feasible plan: ``counts[j]`` bars cut with ``columns[j]``."""

    columns: tuple[MasterColumn, ...]
    counts: tuple[int, ...]
    value: float


def first_fit_decreasing(
    problem: CuttingProblem1D, contexts: Sequence[Context]
) -> Optional[GreedyPlan]:
    """Pack every demanded part, or return ``None`` when the bars run out.

    ``None`` means only that the greedy order failed; a better
    arrangement may still exist.
    """
    kerf = problem.kerf_mm
    order = sorted(
        range(len(contexts)),
        key=lambda c: (contexts[c].cost / contexts[c].length_mm, c),
    )
    # One entry per physical bar in opening order.
    closed = [
        (ci, contexts[ci].length_mm + kerf)
        for ci in order
        for _ in range(contexts[ci].bar_count)
    ]
    open_bars: list[int] = []
    room: list[int] = []
    patterns: list[list[int]] = []

    n_parts = len(problem.part_lengths_mm)
    parts = sorted(range(n_parts), key=lambda i: -problem.part_lengths_mm[i])
    for i in parts:
        weight = problem.part_lengths_mm[i] + kerf
        for _ in range(problem.demands[i]):
            slot = next((k for k, free in enumerate(room) if free >= weight), None)
            if slot is None:
                fresh = next(
                    (k for k, (_, capacity) in enumerate(closed) if capacity >= weight),
                    None,
                )
                if fresh is None:
                    return None
                ci, capacity = closed.pop(fresh)
                open_bars.append(ci)
                room.append(capacity)
                patterns.append([0] * n_parts)
                slot = len(room) - 1
            room[slot] -= weight
            patterns[slot][i] += 1

    counts: dict[MasterColumn, int] = {}
    for ci, pattern in zip(open_bars, patterns):
        column = MasterColumn(context_index=ci, pattern=tuple(pattern))
        counts[column] = counts.get(column, 0) + 1
    return GreedyPlan(
        columns=tuple(counts),
        counts=tuple(counts.values()),
        value=float(sum(contexts[ci].cost for ci in open_bars)),
    )
