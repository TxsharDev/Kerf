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
from constraint_engine.geometry.patterns import Context, pattern_fits
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


def _validate_initial_columns(
    problem: CuttingProblem1D,
    contexts: Sequence[Context],
    initial_columns: Sequence[MasterColumn],
) -> tuple[MasterColumn, ...]:
    """Check each warm-start column against the problem and contexts.

    Warm-start columns arrive from a caller (the engine's replan) and
    are untrusted input: a bad context index, a pattern of the wrong
    length, a negative count, or a pattern that does not physically fit
    the context it names is a caller error and raises ``ValueError``
    before the generator starts.
    """
    n_parts = len(problem.part_lengths_mm)
    checked: list[MasterColumn] = []
    for column in initial_columns:
        if not 0 <= column.context_index < len(contexts):
            raise ValueError(
                f"initial column context_index {column.context_index} is "
                f"outside the context range (0..{len(contexts) - 1})"
            )
        if len(column.pattern) != n_parts:
            raise ValueError(
                f"initial column pattern length {len(column.pattern)} does "
                f"not match the {n_parts} part types"
            )
        if any(count < 0 for count in column.pattern):
            raise ValueError(
                f"initial column pattern {column.pattern} must have "
                "non-negative counts"
            )
        context = contexts[column.context_index]
        if not pattern_fits(
            column.pattern,
            problem.part_lengths_mm,
            problem.kerf_mm,
            context.length_mm,
        ):
            raise ValueError(
                f"initial column pattern {column.pattern} does not fit "
                f"the {context.length_mm}mm context {column.context_index}"
            )
        checked.append(column)
    return tuple(checked)


def run_column_generation(
    problem: CuttingProblem1D,
    contexts: Sequence[Context],
    max_rounds: int = 100,
    tolerance: float = 1e-4,
    initial_columns: Sequence[MasterColumn] = (),
) -> ColumnGenerationResult:
    """Seed, then loop: master LP, capacity-free dual, pricing.

    The loop stops when pricing adds no new column: "converged" if the
    master LP is optimal at that point, "infeasible" otherwise. A run
    that exhausts ``max_rounds`` without either stops "max_rounds" and
    reports the last optimal master solution, if any.

    ``initial_columns`` are warm-start patterns from a previous plan
    (the engine's replan hands them in so a replan reuses what already
    worked). They are validated, then merged into the column pool
    behind the seeds; the pool dedups them against the seeds and
    against anything pricing rediscovers later.
    """
    initial = _validate_initial_columns(problem, contexts, initial_columns)

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
        # A valid initial column cannot change this: a positive count
        # on a demanded part requires that part to fit some context,
        # which the seed would have found.
        return ColumnGenerationResult(
            columns=(),
            solution=None,
            rounds=0,
            stop=STOP_INFEASIBLE,
            infeasible=True,
        )

    seen = {(col.context_index, col.pattern) for col in columns}
    for column in initial:
        key = (column.context_index, column.pattern)
        if key not in seen:
            seen.add(key)
            columns = columns + (column,)
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
