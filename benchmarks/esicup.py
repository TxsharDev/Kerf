"""Loader for ESICUP Falkenauer 1D cutting stock instances.

The files are distributed by ESICUP (Falkenauer, 1996) in a plain text
format:

    line 1: number of instances
    per instance:
        a label line (e.g. `` u250_00 ``)
        a parameter line: capacity  part_count  lower_bound
        then ``part_count`` size lines, one part size per line

Sizes and capacities are integers or decimal numbers. The lower bound
column is a published lower bound on the optimal number of bars; the
arithmetic bound ``ceil(sum(sizes) / capacity)`` is the weakest valid
one, and the published value is sometimes tighter. The loader
self-checks the file on load: the declared instance count, the
per-instance part count, that the published bound is at least as strong
as the arithmetic bound, and that every part fits on one bar. A file
that fails any check is refused, not loaded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Instance", "load"]


@dataclass(frozen=True)
class Instance:
    """One cutting instance: cut ``sizes`` from bars of ``capacity``."""

    label: str
    capacity: float
    sizes: tuple[float, ...]
    lower_bound: int

    @property
    def part_count(self) -> int:
        return len(self.sizes)

    @property
    def total_length(self) -> float:
        return sum(self.sizes)

    @property
    def is_integer(self) -> bool:
        """True when capacity and every size are whole numbers."""
        return self.capacity.is_integer() and all(s.is_integer() for s in self.sizes)


def load(path: str | Path) -> list[Instance]:
    """Parse one ESICUP instance file and self-check it.

    Raises ``ValueError`` on any structural problem so a corrupt or
    truncated file can never reach a benchmark run.
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path}: empty instance file")

    declared = int(lines[0])
    instances: list[Instance] = []
    i = 1
    for _ in range(declared):
        if i + 2 > len(lines):
            raise ValueError(f"{path}: file ends mid-instance after {len(instances)} of {declared}")
        label = lines[i]
        fields = lines[i + 1].split()
        if len(fields) != 3:
            raise ValueError(
                f"{path}: parameter line for {label!r} must be "
                f"'capacity count lower_bound', got {lines[i + 1]!r}"
            )
        capacity, count, lb = float(fields[0]), int(fields[1]), int(fields[2])
        size_lines = lines[i + 2 : i + 2 + count]
        if len(size_lines) != count:
            raise ValueError(
                f"{path}: instance {label!r} declares {count} parts, "
                f"file contains {len(size_lines)}"
            )
        sizes = tuple(float(s) for s in size_lines)
        _check_instance(label, capacity, sizes, lb)
        instances.append(
            Instance(label=label, capacity=capacity, sizes=sizes, lower_bound=lb)
        )
        i += 2 + count

    if i != len(lines):
        raise ValueError(
            f"{path}: {len(lines) - i} trailing lines after {declared} instances"
        )
    return instances


def _check_instance(label: str, capacity: float, sizes: tuple[float, ...], lb: int) -> None:
    """Refuse instances that contradict their own parameter line."""
    if capacity <= 0:
        raise ValueError(f"{label}: capacity must be positive, got {capacity}")
    for size in sizes:
        if size <= 0:
            raise ValueError(f"{label}: part sizes must be positive, got {size}")
        if size > capacity:
            raise ValueError(
                f"{label}: part size {size} exceeds the bar capacity {capacity}"
            )
    computed_lb = math.ceil(sum(sizes) / capacity - 1e-9)
    if lb < computed_lb:
        raise ValueError(
            f"{label}: published lower bound {lb} is below the arithmetic "
            f"bound ceil(sum/capacity) = {computed_lb}"
        )
