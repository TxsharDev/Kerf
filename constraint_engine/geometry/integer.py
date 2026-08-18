"""Integer master problem for 1D cutting stock.

The master LP allows fractional bars, so its value is only a lower
bound on the cost of a real plan. The integer program decides how
many bars of each context are cut with each pattern, and its solution
is the plan itself:

    min  sum_j cost(c_j) * x_j
    s.t. sum_j a_ji * x_j >= d_i        every part type's demand
         sum_{j in c} x_j <= Q_c        every context's bar count
         x_j integer, 0 <= x_j <= Q_{c_j}

Solved with HiGHS through scipy.optimize.milp. The reported status is
the strongest claim the evidence supports:

- ``optimal``: HiGHS proved the incumbent optimal.
- ``incumbent``: the time limit hit, but HiGHS returned a feasible
  solution that passed constraint validation.
- ``heuristic``: HiGHS returned nothing, or proved infeasibility only
  over the priced columns, so a deterministic fallback floored the LP
  solution and packed the remaining free bars with the exact bounded
  knapsack. The fallback may add a column for a pattern column
  generation never priced, in which case the result's columns are a
  superset of the input.
- ``infeasible``: no integer plan covers the demand with the bars on
  hand, typically because there are too few bars.
- ``no_incumbent``: HiGHS returned nothing and the fallback ran out
  of bars before covering the demand. The result says so instead of
  pretending the demand was met.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from constraint_engine.geometry.knapsack import bounded_knapsack
from constraint_engine.geometry.master import MasterColumn
from constraint_engine.geometry.patterns import Context
from constraint_engine.geometry.types import CuttingProblem1D

__all__ = [
    "IP_HEURISTIC",
    "IP_INFEASIBLE",
    "IP_INCUMBENT",
    "IP_NO_INCUMBENT",
    "IP_OPTIMAL",
    "IntegerSolution",
    "solve_master_ip",
]

IP_OPTIMAL = "optimal"
IP_INCUMBENT = "incumbent"
IP_HEURISTIC = "heuristic"
IP_INFEASIBLE = "infeasible"
IP_NO_INCUMBENT = "no_incumbent"

_TOL = 1e-6


@dataclass(frozen=True)
class IntegerSolution:
    """Outcome of the integer master problem.

    ``x`` aligns with ``columns``. ``columns`` may be a superset of
    the input column set when the fallback fill had to add a pattern
    column generation never priced.
    """

    status: str
    columns: tuple[MasterColumn, ...]
    x: tuple[int, ...]
    value: float


def _demand_matrix(columns: Sequence[MasterColumn], n_parts: int) -> np.ndarray:
    """Pattern-count matrix: row i is part i, one column per master column."""
    matrix = np.zeros((n_parts, len(columns)), dtype=float)
    for j, col in enumerate(columns):
        for i, count in enumerate(col.pattern):
            matrix[i, j] = count
    return matrix


def _capacity_matrix(columns: Sequence[MasterColumn], n_contexts: int) -> np.ndarray:
    """Bar-use matrix: row c counts the columns that cut on context c."""
    matrix = np.zeros((n_contexts, len(columns)), dtype=float)
    for j, col in enumerate(columns):
        matrix[col.context_index, j] = 1.0
    return matrix


def solve_master_ip(
    columns: Sequence[MasterColumn],
    problem: CuttingProblem1D,
    contexts: Sequence[Context],
    lp_x: Optional[Sequence[float]] = None,
    time_limit_s: float = 5.0,
) -> IntegerSolution:
    """Solve the integer master problem with the current columns.

    ``lp_x`` is the LP solution over the same column order. It seeds
    the fallback when HiGHS returns no incumbent: the floor of the LP
    solution is usually close to a real plan, so the fill only has to
    make up the difference.
    """
    n_parts = len(problem.part_lengths_mm)
    n_cols = len(columns)
    demands = np.asarray(problem.demands, dtype=float)

    if not demands.any():
        return IntegerSolution(
            status=IP_OPTIMAL,
            columns=tuple(columns),
            x=tuple(0 for _ in columns),
            value=0.0,
        )
    if n_cols == 0:
        return IntegerSolution(
            status=IP_INFEASIBLE, columns=tuple(columns), x=(), value=0.0
        )

    a_dem = _demand_matrix(columns, n_parts)
    a_cap = _capacity_matrix(columns, len(contexts))
    c_vec = np.array(
        [contexts[col.context_index].cost for col in columns], dtype=float
    )
    ub_per_col = np.array(
        [contexts[col.context_index].bar_count for col in columns], dtype=float
    )
    bar_counts = np.array([c.bar_count for c in contexts], dtype=float)

    result = milp(
        c=c_vec,
        constraints=[
            LinearConstraint(a_dem, lb=demands, ub=np.inf),
            LinearConstraint(a_cap, lb=-np.inf, ub=bar_counts),
        ],
        integrality=np.ones(n_cols, dtype=int),
        bounds=Bounds(lb=np.zeros(n_cols), ub=ub_per_col),
        options={"time_limit": time_limit_s, "mip_rel_gap": 0.0, "presolve": True},
    )
    if result.status == 2:
        # HiGHS proved infeasibility only over the priced columns. A
        # pattern column generation never priced may still cover the
        # demand on the bars on hand, so run the deterministic fill
        # before declaring the problem infeasible.
        filled = _fallback_fill(tuple(columns), problem, contexts, lp_x)
        if filled.status == IP_HEURISTIC:
            return filled
        return IntegerSolution(
            status=IP_INFEASIBLE, columns=tuple(columns), x=(), value=0.0
        )
    if result.status not in (0, 1):
        raise RuntimeError(f"integer solve failed: {result.message}")

    incumbent = _valid_incumbent(result.x, a_dem, a_cap, demands, bar_counts)
    if incumbent is not None:
        status = IP_OPTIMAL if result.status == 0 else IP_INCUMBENT
        return IntegerSolution(
            status=status,
            columns=tuple(columns),
            x=incumbent,
            value=float(result.fun),
        )
    return _fallback_fill(tuple(columns), problem, contexts, lp_x)


def _valid_incumbent(
    x: Optional[np.ndarray],
    a_dem: np.ndarray,
    a_cap: np.ndarray,
    demands: np.ndarray,
    bar_counts: np.ndarray,
) -> Optional[tuple[int, ...]]:
    """Round the solver's answer to integers and verify every constraint.

    Returns the integer tuple when the rounded answer satisfies the
    demand and capacity constraints within tolerance, or ``None`` when
    the solver returned nothing or the answer does not validate.
    """
    if x is None:
        return None
    x_arr = np.asarray(x, dtype=float)
    rounded = np.rint(x_arr)
    if np.max(np.abs(x_arr - rounded)) > _TOL:
        return None
    iv = rounded.astype(int)
    if (iv < 0).any():
        return None
    if ((a_dem @ iv) < demands - _TOL).any():
        return None
    if ((a_cap @ iv) > bar_counts + _TOL).any():
        return None
    return tuple(int(v) for v in iv)


def _fallback_fill(
    columns: tuple[MasterColumn, ...],
    problem: CuttingProblem1D,
    contexts: Sequence[Context],
    lp_x: Optional[Sequence[float]],
) -> IntegerSolution:
    """Floor the LP solution, then pack free bars with the knapsack.

    Deterministic: the LP solution is floored in column order, usage
    is clipped per context to the bars on hand, and every remaining
    free bar is packed with the exact bounded knapsack over the still
    unmet demand. A packed pattern that matches an input column is
    reused; a novel pattern becomes a new column. When the bars run
    out before demand is met, the result is ``no_incumbent``: the
    shortfall is reported instead of hidden.
    """
    n_parts = len(problem.part_lengths_mm)
    n_cols = len(columns)
    demands = np.asarray(problem.demands, dtype=int)
    bar_counts = np.array([c.bar_count for c in contexts], dtype=int)
    length_mm = np.asarray(problem.part_lengths_mm, dtype=int)

    columns_list = list(columns)
    if lp_x is None:
        x_vec = np.zeros(n_cols, dtype=int)
    else:
        padded = np.zeros(n_cols, dtype=float)
        length = min(n_cols, len(lp_x))
        padded[:length] = np.asarray(lp_x, dtype=float)[:length]
        x_vec = np.floor(padded).astype(int)

    # Clip per-context usage to the available bars, column order.
    used_per_ctx = np.zeros(len(contexts), dtype=int)
    for j, col in enumerate(columns_list):
        ci = col.context_index
        if used_per_ctx[ci] + x_vec[j] > bar_counts[ci]:
            x_vec[j] = max(0, bar_counts[ci] - used_per_ctx[ci])
        used_per_ctx[ci] += x_vec[j]

    a_dem = _demand_matrix(tuple(columns_list), n_parts)
    covered = (a_dem @ x_vec).astype(int)
    unmet = np.maximum(0, demands - covered)
    col_index = {
        (col.context_index, col.pattern): j for j, col in enumerate(columns_list)
    }

    for ci, context in enumerate(contexts):
        free = bar_counts[ci] - used_per_ctx[ci]
        while free > 0 and unmet.max() > 0:
            capacity = context.length_mm + problem.kerf_mm
            # Each part takes its length plus one kerf; the last cut
            # on a bar owes no trailing kerf, hence capacity + kerf.
            weights = (length_mm + problem.kerf_mm).astype(int).tolist()
            bounds = [
                min(int(unmet[i]), capacity // weights[i])
                if weights[i] <= capacity
                else 0
                for i in range(n_parts)
            ]
            profits = length_mm.astype(float).tolist()
            counts, _ = bounded_knapsack(profits, weights, bounds, capacity)
            if counts.sum() == 0:
                break
            pattern = tuple(int(v) for v in counts)
            idx = col_index.get((ci, pattern))
            if idx is None:
                idx = len(columns_list)
                columns_list.append(MasterColumn(ci, pattern))
                col_index[(ci, pattern)] = idx
                x_vec = np.append(x_vec, 0)
            x_vec[idx] += 1
            used_per_ctx[ci] += 1
            free -= 1
            covered = covered + np.asarray(pattern, dtype=int)
            unmet = np.maximum(0, demands - covered)

    if unmet.max() > 0:
        return IntegerSolution(
            status=IP_NO_INCUMBENT, columns=tuple(columns_list), x=(), value=0.0
        )
    value = float(
        sum(
            contexts[col.context_index].cost * count
            for col, count in zip(columns_list, x_vec)
        )
    )
    return IntegerSolution(
        status=IP_HEURISTIC,
        columns=tuple(columns_list),
        x=tuple(int(v) for v in x_vec),
        value=value,
    )
