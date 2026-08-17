"""Data types for the 1D cutting stock geometry model.

A problem is a set of part lengths with integer demands plus a set of
physical bars. Bars come from two origins: fresh stock (cost is the
price of a full bar) and remnants from earlier cuts (cost zero, an age
in days, and a stable id so the ledger can trace them).

Types here are frozen dataclasses validated at construction. Structural
errors (mismatched lengths, negative values) raise ``ValueError``;
feasibility is not a construction error, an uncuttable problem can
still be built, solved, and diagnosed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "BarAssignment",
    "BarSpec",
    "CuttingProblem1D",
    "CuttingResult1D",
]

ORIGIN_NEW = "new"
ORIGIN_REMNANT = "remnant"


@dataclass(frozen=True)
class BarSpec:
    """One physical bar available for cutting."""

    length_mm: int
    cost: float
    origin: str
    id: str = ""
    age_days: float = 0.0


@dataclass(frozen=True)
class CuttingProblem1D:
    """A 1D cutting stock problem over a single material.

    Part ``i`` has length ``part_lengths_mm[i]`` and demand
    ``demands[i]``. ``kerf_mm`` is the blade loss between adjacent
    cuts. ``min_offcut_mm`` decides which leftovers become remnants in
    the ledger; it never affects pattern feasibility.
    """

    part_lengths_mm: list[int]
    demands: list[int]
    bars: tuple[BarSpec, ...]
    kerf_mm: int = 3
    min_offcut_mm: int = 50
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.part_lengths_mm) != len(self.demands):
            raise ValueError(
                "part_lengths_mm and demands must have equal length "
                f"(got {len(self.part_lengths_mm)} and {len(self.demands)})"
            )
        for i, length in enumerate(self.part_lengths_mm):
            if length <= 0:
                raise ValueError(f"part {i} length must be positive, got {length}")
        for i, demand in enumerate(self.demands):
            if demand < 0:
                raise ValueError(f"part {i} demand must be non-negative, got {demand}")
        if self.kerf_mm < 0:
            raise ValueError(f"kerf_mm must be non-negative, got {self.kerf_mm}")
        if self.min_offcut_mm < 0:
            raise ValueError(
                f"min_offcut_mm must be non-negative, got {self.min_offcut_mm}"
            )
        for bar in self.bars:
            if bar.length_mm <= 0:
                raise ValueError(f"bar {bar.id or 'unnamed'} has non-positive length")
            if bar.cost < 0:
                raise ValueError(f"bar {bar.id or 'unnamed'} has negative cost")
        # Frozen dataclass: post-init mutation is allowed via object.
        object.__setattr__(self, "part_lengths_mm", list(self.part_lengths_mm))
        object.__setattr__(self, "demands", list(self.demands))
        object.__setattr__(self, "bars", tuple(self.bars))


@dataclass(frozen=True)
class BarAssignment:
    """The pattern assigned to one physical bar in a solution."""

    bar_id: str
    pattern: tuple[int, ...]
    offcut_mm: int


@dataclass(frozen=True)
class CuttingResult1D:
    """Outcome of solving a :class:`CuttingProblem1D`.

    ``lp_value`` and ``ip_value`` are the objective values of the LP
    relaxation and the integer solution (bar cost, zero when infeasible).
    ``gap`` is ``(ip - lp) / lp`` clamped to zero. ``stats`` carries
    diagnostics: column-generation rounds, columns kept, waste, and
    remnant usage.
    """

    feasible: bool
    assignments: tuple[BarAssignment, ...]
    lp_value: float
    ip_value: float
    gap: float
    solve_time_s: float
    stats: dict = field(default_factory=dict)
    infeasible_reasons: tuple[str, ...] = ()
