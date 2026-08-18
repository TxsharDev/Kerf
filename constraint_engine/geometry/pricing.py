"""Pricing round and column generation for the 1D cutting stock master.

Pricing asks, for each context: does any pattern have a dual value that
exceeds the context cost? That is exactly a bounded knapsack. Part type
``i`` has profit equal to its dual price, weight equal to its effective
weight, and a bound of ``min(demand_i, capacity // weight_i)``.

Effective weights turn the physical kerf model into a plain capacity
check: a part's weight is ``length + kerf`` and a bar's capacity is
``length + kerf``, so a pattern fits on a bar exactly when its total
effective weight stays within the effective capacity.

The price signal comes from the capacity-free master dual, a lower
bound on the master optimum and a signal, not a certificate. It does
not depend on the master LP being feasible, so pricing keeps working
in rounds where the master LP itself is infeasible.

Column generation runs as one loop: single-type columns are seeded for
every context up front, then each round solves the master LP, solves
the capacity-free dual, prices, and adds any new columns. When pricing
adds nothing the run stops: as "converged" if the master LP is optimal
at that point, as "infeasible" otherwise.

Known limitation: if the master LP stays infeasible, the true problem
is feasible, and no priced pattern beats its context cost, the loop
stops as "infeasible" without proof. The prechecks in ``solve`` catch
every realistic instance of this corner, and the property test exists
to surface a real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from constraint_engine.geometry.knapsack import bounded_knapsack
from constraint_engine.geometry.master import (
    OPTIMAL,
    MasterColumn,
    MasterSolution,
    solve_master_dual,
    solve_master_lp,
)
from constraint_engine.geometry.patterns import Context
from constraint_engine.geometry.types import CuttingProblem1D

__all__ = [
    "STOP_CONVERGED",
    "STOP_INFEASIBLE",
    "STOP_MAX_ROUNDS",
    "STOP_NO_DEMAND",
    "ColumnGenerationResult",
    "price_columns",
    "run_column_generation",
]

STOP_NO_DEMAND = "no_demand"
STOP_INFEASIBLE = "infeasible"
STOP_CONVERGED = "converged"
STOP_MAX_ROUNDS = "max_rounds"


@dataclass(frozen=True)
class ColumnGenerationResult:
    """Outcome of a column generation run.

    ``solution`` is the final master LP solution when one is available,
    ``stop`` names the reason the loop ended, and ``infeasible`` is set
    only when the run proves the master infeasible.
    """

    columns: tuple[MasterColumn, ...]
    solution: Optional[MasterSolution]
    rounds: int
    stop: str
    infeasible: bool


def _seed_columns(
    problem: CuttingProblem1D, contexts: Sequence[Context]
) -> tuple[MasterColumn, ...]:
    """One single-type column per (context, part) pair that fits.

    The pattern takes as many copies of the part as the effective
    capacity allows. Seeding every context up front is what lets pricing
    keep working when the master LP is infeasible: the capacity-free
    dual only needs some column covering each demanded part.
    """
    n_parts = len(problem.part_lengths_mm)
    columns: list[MasterColumn] = []
    for context_index, context in enumerate(contexts):
        capacity = context.length_mm + problem.kerf_mm
        for i, (length_mm, demand) in enumerate(
            zip(problem.part_lengths_mm, problem.demands)
        ):
            if demand <= 0:
                continue
            weight = length_mm + problem.kerf_mm
            count = capacity // weight
            if count < 1:
                continue
            pattern = [0] * n_parts
            pattern[i] = count
            columns.append(
                MasterColumn(context_index=context_index, pattern=tuple(pattern))
            )
    return tuple(columns)


def price_columns(
    problem: CuttingProblem1D,
    contexts: Sequence[Context],
    dual: Sequence[float],
    tolerance: float = 1e-4,
) -> tuple[MasterColumn, ...]:
    """One pricing round: the best pattern per context, or nothing.

    A context contributes a column when its best pattern's dual value
    exceeds the context cost by more than ``tolerance``. Parts with
    zero demand are excluded from the knapsack entirely, so a high
    dual price on an undemanded part cannot pull it into a pattern.
    """
    profits = [float(price) for price in dual]
    weights = [length_mm + problem.kerf_mm for length_mm in problem.part_lengths_mm]
    new_columns: list[MasterColumn] = []
    for context_index, context in enumerate(contexts):
        capacity = context.length_mm + problem.kerf_mm
        bounds = [
            min(int(demand), capacity // weight) if weight <= capacity else 0
            for demand, weight in zip(problem.demands, weights)
        ]
        counts, value = bounded_knapsack(profits, weights, bounds, capacity)
        if value > context.cost + tolerance:
            pattern = tuple(int(count) for count in counts)
            new_columns.append(
                MasterColumn(context_index=context_index, pattern=pattern)
            )
    return tuple(new_columns)


def run_column_generation(
    problem: CuttingProblem1D,
    contexts: Sequence[Context],
    max_rounds: int = 100,
    tolerance: float = 1e-4,
) -> ColumnGenerationResult:
    """Seed, then loop: master LP, capacity-free dual, pricing.

    The loop stops when pricing adds no new column: "converged" if the
    master LP is optimal at that point, "infeasible" otherwise. A run
    that exhausts ``max_rounds`` without either stops "max_rounds" and
    reports the last optimal master solution, if any.
    """
    if not any(demand > 0 for demand in problem.demands):
        return ColumnGenerationResult(
            columns=(),
            solution=None,
            rounds=0,
            stop=STOP_NO_DEMAND,
            infeasible=False,
        )

    columns = _seed_columns(problem, contexts)
    if not columns:
        # No part type fits in any context, so demand can never be met.
        return ColumnGenerationResult(
            columns=(),
            solution=None,
            rounds=0,
            stop=STOP_INFEASIBLE,
            infeasible=True,
        )

    seen = {(col.context_index, col.pattern) for col in columns}
    last_optimal: Optional[MasterSolution] = None

    for round_no in range(1, max_rounds + 1):
        solution = solve_master_lp(columns, problem, contexts)
        if solution.status == OPTIMAL:
            last_optimal = solution

        dual = solve_master_dual(columns, problem, contexts)
        if dual is None:
            # A demanded part is covered by no column: the capacity-free
            # dual is unbounded and no amount of pricing can help.
            return ColumnGenerationResult(
                columns=columns,
                solution=None,
                rounds=round_no,
                stop=STOP_INFEASIBLE,
                infeasible=True,
            )

        fresh: list[MasterColumn] = []
        for candidate in price_columns(problem, contexts, dual, tolerance):
            key = (candidate.context_index, candidate.pattern)
            if key not in seen:
                seen.add(key)
                fresh.append(candidate)

        if not fresh:
            converged = solution.status == OPTIMAL
            return ColumnGenerationResult(
                columns=columns,
                solution=last_optimal if converged else None,
                rounds=round_no,
                stop=STOP_CONVERGED if converged else STOP_INFEASIBLE,
                infeasible=not converged,
            )

        columns = columns + tuple(fresh)

    return ColumnGenerationResult(
        columns=columns,
        solution=last_optimal,
        rounds=max_rounds,
        stop=STOP_MAX_ROUNDS,
        infeasible=False,
    )
