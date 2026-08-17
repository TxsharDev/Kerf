"""Master linear program and its dual for 1D cutting stock.

The master problem relaxes the set-partitioning view of cutting: one
continuous variable per (pattern, context) pair.

    min  sum_j cost(c_j) * x_j
    s.t. sum_j a_ji * x_j >= d_i       every part type's demand
         sum_{j in c} x_j <= Q_c       every context's bar count
         x_j >= 0

Column generation runs on this: a dual prices the part types, and a
candidate pattern earns a column when its dual value exceeds the cost
of its context (negative reduced cost). The dual used for pricing is
the classical capacity-free master dual: a lower bound on the optimum
and a price signal, not a convergence certificate. Convergence is
declared by the pricing round itself, when it finds no pattern with
negative reduced cost. The dual is solved as its own linear program
rather than read from solver marginals, so the values are trustworthy
even when the primal is degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linprog

from constraint_engine.geometry.patterns import Context
from constraint_engine.geometry.types import CuttingProblem1D

__all__ = [
    "INFEASIBLE",
    "OPTIMAL",
    "MasterColumn",
    "MasterSolution",
    "reduced_cost",
    "solve_master_dual",
    "solve_master_lp",
]

OPTIMAL = "optimal"
INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class MasterColumn:
    """One column of the master problem: a pattern cut on one context."""

    context_index: int
    pattern: tuple[int, ...]


@dataclass(frozen=True)
class MasterSolution:
    """Outcome of the master LP. ``x`` aligns with the column order."""

    status: str
    x: tuple[float, ...]
    value: float


def _demand_matrix(columns: Sequence[MasterColumn], n_parts: int) -> np.ndarray:
    """Pattern-count matrix: row i is part i, one column per master column."""
    matrix = np.zeros((n_parts, len(columns)), dtype=float)
    for j, col in enumerate(columns):
        for i, count in enumerate(col.pattern):
            matrix[i, j] = count
    return matrix


def solve_master_lp(
    columns: Sequence[MasterColumn],
    problem: CuttingProblem1D,
    contexts: Sequence[Context],
) -> MasterSolution:
    """Solve the master LP with the current columns.

    Returns ``status`` as ``"optimal"`` or ``"infeasible"``. Solver
    trouble that is neither (numerical failure) raises ``RuntimeError``
    instead of being silently mislabeled.
    """
    n_parts = len(problem.part_lengths_mm)
    n_cols = len(columns)
    demands = np.asarray(problem.demands, dtype=float)

    if n_cols == 0:
        if demands.any():
            return MasterSolution(status=INFEASIBLE, x=(), value=0.0)
        return MasterSolution(status=OPTIMAL, x=(), value=0.0)

    a_dem = _demand_matrix(columns, n_parts)
    n_ctx = len(contexts)
    a_cap = np.zeros((n_ctx, n_cols), dtype=float)
    for j, col in enumerate(columns):
        a_cap[col.context_index, j] = 1.0

    a_ub = np.vstack([-a_dem, a_cap])
    b_ub = np.concatenate(
        [
            -demands,
            [float(len(c.bar_ids)) for c in contexts],
        ]
    )
    c_vec = np.array(
        [contexts[col.context_index].cost for col in columns], dtype=float
    )

    result = linprog(
        c=c_vec,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * n_cols,
        method="highs",
    )
    if result.status == 0:
        return MasterSolution(
            status=OPTIMAL, x=tuple(result.x), value=float(result.fun)
        )
    if result.status == 2:
        return MasterSolution(status=INFEASIBLE, x=(), value=0.0)
    raise RuntimeError(f"master LP failed: {result.message}")


def solve_master_dual(
    columns: Sequence[MasterColumn],
    problem: CuttingProblem1D,
    contexts: Sequence[Context],
) -> Optional[np.ndarray]:
    """Solve the capacity-free master dual: ``max d.y s.t. A_dem . y <= costs, y >= 0``.

    This prices the part types and gives a lower bound on the master
    optimum. It is a pricing signal, not a certificate: the
    capacity-free relaxation drops the bar-count constraints, so its
    value sits below (or at) the true LP optimum, and convergence is
    judged by the pricing round finding no pattern with negative
    reduced cost.

    Returns the part-type prices, or ``None`` when no finite dual exists
    (typically because some demanded part has no column that cuts it).
    With zero demand the dual is trivially zero everywhere.
    """
    n_parts = len(problem.part_lengths_mm)
    demands = np.asarray(problem.demands, dtype=float)
    if not demands.any():
        return np.zeros(n_parts, dtype=float)
    if not columns:
        return None

    a_dem = _demand_matrix(columns, n_parts)
    costs = np.array(
        [contexts[col.context_index].cost for col in columns], dtype=float
    )
    # One dual variable per part type; one constraint per column
    # (the column's dual value must not exceed its context cost).
    result = linprog(
        c=-demands,
        A_ub=a_dem.T,
        b_ub=costs,
        bounds=[(0.0, None)] * n_parts,
        method="highs",
    )
    if result.status == 0:
        return np.asarray(result.x, dtype=float)
    return None


def reduced_cost(context: Context, pattern: Sequence[int], dual: np.ndarray) -> float:
    """``context.cost - sum(pattern_i * dual_i)`` for one candidate pattern.

    Negative means the pattern is worth a column: it covers demand at
    less than the price the dual charges for those parts.
    """
    value = sum(count * y for count, y in zip(pattern, dual))
    return context.cost - value
