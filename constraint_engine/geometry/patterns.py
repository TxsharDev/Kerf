"""Patterns and contexts for the 1D cutting stock model.

A pattern is a tuple of integer part counts, one entry per part type,
describing everything cut from a single bar. The physical model counts
one kerf loss between each pair of adjacent cuts:

    used = sum(length_i * count_i) + kerf * max(0, total_cuts - 1)

A pattern fits when ``used <= bar_length``; the leftover is the
offcut. A lone part needs no cut and therefore no kerf.

A context groups the physical bars that are interchangeable from the
solver's point of view: same length and same cost class (fresh stock
at one price, or remnants of one length). The master problem then uses
one variable per pattern per context instead of one per bar. Inside a
context, bars are ordered so the oldest remnants are consumed first;
the ordering is what assembly uses when it assigns patterns to
physical bar ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from constraint_engine.geometry.types import BarSpec, CuttingProblem1D

__all__ = [
    "Context",
    "pattern_cuts",
    "pattern_fits",
    "pattern_offcut",
    "patterns_from_problem",
]


@dataclass(frozen=True)
class Context:
    """An interchangeable group of bars: one length, one cost class."""

    length_mm: int
    cost: float
    bar_ids: Tuple[str, ...]

    @property
    def bar_count(self) -> int:
        return len(self.bar_ids)


def _validate_pattern(
    pattern: Sequence[int], part_lengths_mm: Sequence[int]
) -> None:
    if len(pattern) != len(part_lengths_mm):
        raise ValueError(
            "pattern has "
            f"{len(pattern)} entries but the problem has "
            f"{len(part_lengths_mm)} part types"
        )
    for i, count in enumerate(pattern):
        if count < 0:
            raise ValueError(f"pattern count for part {i} is negative")


def pattern_cuts(pattern: Sequence[int]) -> int:
    """Total number of parts in the pattern."""
    return sum(pattern)


def pattern_used_mm(
    pattern: Sequence[int],
    part_lengths_mm: Sequence[int],
    kerf_mm: int,
) -> int:
    """Physical length consumed: parts plus kerf between adjacent cuts."""
    _validate_pattern(pattern, part_lengths_mm)
    total = pattern_cuts(pattern)
    used = sum(count * length for count, length in zip(pattern, part_lengths_mm))
    if total > 1:
        used += kerf_mm * (total - 1)
    return used


def pattern_fits(
    pattern: Sequence[int],
    part_lengths_mm: Sequence[int],
    kerf_mm: int,
    bar_length_mm: int,
) -> bool:
    """Whether the pattern fits on one bar of the given length."""
    return pattern_used_mm(pattern, part_lengths_mm, kerf_mm) <= bar_length_mm


def pattern_offcut(
    pattern: Sequence[int],
    part_lengths_mm: Sequence[int],
    kerf_mm: int,
    bar_length_mm: int,
) -> int:
    """Remaining physical length on the bar after cutting the pattern."""
    return bar_length_mm - pattern_used_mm(pattern, part_lengths_mm, kerf_mm)


def patterns_from_problem(problem: CuttingProblem1D) -> Tuple[Context, ...]:
    """Group the problem's bars into solver contexts.

    Bars are grouped by ``(length_mm, cost)``: fresh stock and
    remnants of the same length land in different contexts because
    their costs differ. Within a context, bars are ordered oldest
    remnant first (age descending), with stable input order breaking
    ties. Contexts are returned in a deterministic order: length
    descending, then cost ascending, then first bar id.
    """
    groups: dict[tuple[int, float], list[tuple[int, BarSpec]]] = {}
    for index, bar in enumerate(problem.bars):
        groups.setdefault((bar.length_mm, bar.cost), []).append((index, bar))

    contexts: list[Context] = []
    for (length_mm, cost), members in groups.items():
        # Oldest remnant first; ties keep the original bar order.
        ordered = sorted(members, key=lambda pair: (-pair[1].age_days, pair[0]))
        contexts.append(
            Context(
                length_mm=length_mm,
                cost=cost,
                bar_ids=tuple(bar.id or f"bar_{index}" for index, bar in ordered),
            )
        )
    contexts.sort(key=lambda c: (-c.length_mm, c.cost, c.bar_ids[0]))
    return tuple(contexts)
