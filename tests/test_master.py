"""Tests for the master linear program and its dual.

Master LP (one variable per pattern per context):

    min  sum_j cost(c_j) * x_j
    s.t. sum_j a_ji * x_j >= d_i        for every part type i
         sum_{j in c} x_j <= Q_c        for every context c
         x_j >= 0

Dual (one variable per part type, one per column for reduced costs):

    max  sum_i d_i * y_i
    s.t. sum_i a_ji * y_i <= cost(c_j)  for every column j
         y_i >= 0

The dual is what pricing uses: a candidate pattern has negative reduced
cost exactly when its dual value exceeds its context cost. This is the
capacity-free master dual, a lower bound and a price signal, not a
convergence certificate; the pricing round declares convergence when it
finds no pattern with negative reduced cost.
"""

import numpy as np
import pytest

from constraint_engine.geometry.master import (
    MasterColumn,
    reduced_cost,
    solve_master_dual,
    solve_master_lp,
)
from constraint_engine.geometry.patterns import Context
from constraint_engine.geometry.types import BarSpec, CuttingProblem1D


def two_part_problem():
    """Two part types, one fresh-stock context, kerf zero.

    Parts: 4000mm (demand 2), 2000mm (demand 2). Bars: 6000mm at 10.0,
    so the exact-fit pattern (1, 1) is the obvious winner.
    """
    return CuttingProblem1D(
        part_lengths_mm=[4000, 2000],
        demands=[2, 2],
        bars=(
            BarSpec(length_mm=6000, cost=10.0, origin="new", id="b0"),
            BarSpec(length_mm=6000, cost=10.0, origin="new", id="b1"),
        ),
        kerf_mm=0,
    )


FRESH = Context(length_mm=6000, cost=10.0, bar_ids=("b0", "b1"))

# All five patterns that fit a 6000mm bar with zero kerf.
ALL_COLUMNS = (
    MasterColumn(context_index=0, pattern=(1, 0)),
    MasterColumn(context_index=0, pattern=(0, 1)),
    MasterColumn(context_index=0, pattern=(0, 2)),
    MasterColumn(context_index=0, pattern=(1, 1)),
    MasterColumn(context_index=0, pattern=(0, 3)),
)


class TestSolveMasterLP:
    def test_optimum_uses_exact_fit_columns(self):
        problem = two_part_problem()
        sol = solve_master_lp(ALL_COLUMNS, problem, (FRESH,))
        assert sol.status == "optimal"
        # Two exact-fit (1, 1) bars cover everything at cost 20.
        assert sol.value == pytest.approx(20.0)
        assert sol.x[3] == pytest.approx(2.0)

    def test_infeasible_when_capacity_short(self):
        # One bar cannot cover demand 3 of the 4000mm part when the only
        # column that cuts it is (1, 0).
        problem = CuttingProblem1D(
            part_lengths_mm=[4000, 2000],
            demands=[3, 2],
            bars=(BarSpec(length_mm=6000, cost=10.0, origin="new", id="b0"),),
            kerf_mm=0,
        )
        columns = (MasterColumn(context_index=0, pattern=(1, 0)),)
        sol = solve_master_lp(columns, problem, (FRESH,))
        assert sol.status == "infeasible"
        assert sol.value == 0.0

    def test_zero_demand_is_free(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[4000], demands=[0],
            bars=(BarSpec(length_mm=6000, cost=10.0, origin="new", id="b0"),),
            kerf_mm=0,
        )
        # Width-1 column to match the single part type.
        columns = (MasterColumn(context_index=0, pattern=(1,)),)
        sol = solve_master_lp(columns, problem, (FRESH,))
        assert sol.status == "optimal"
        assert sol.value == pytest.approx(0.0)

    def test_no_columns_with_demand_is_infeasible(self):
        problem = two_part_problem()
        sol = solve_master_lp((), problem, (FRESH,))
        assert sol.status == "infeasible"

    def test_no_columns_without_demand_is_free(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[4000], demands=[0], bars=(), kerf_mm=0,
        )
        sol = solve_master_lp((), problem, ())
        assert sol.status == "optimal"
        assert sol.value == 0.0

    def test_fresh_and_remnant_contexts_share_columns(self):
        # A remnant context (zero cost) should absorb the demand first.
        problem = CuttingProblem1D(
            part_lengths_mm=[2000],
            demands=[3],
            bars=(
                BarSpec(length_mm=6000, cost=10.0, origin="new", id="n0"),
                BarSpec(length_mm=6000, cost=0.0, origin="remnant", id="r0"),
            ),
            kerf_mm=0,
        )
        fresh = Context(length_mm=6000, cost=10.0, bar_ids=("n0",))
        rem = Context(length_mm=6000, cost=0.0, bar_ids=("r0",))
        columns = (
            MasterColumn(context_index=0, pattern=(3,)),
            MasterColumn(context_index=1, pattern=(3,)),
        )
        sol = solve_master_lp(columns, problem, (fresh, rem))
        assert sol.status == "optimal"
        assert sol.value == pytest.approx(0.0)
        # Three parts off the free remnant, nothing off fresh stock.
        assert sol.x[0] == pytest.approx(0.0)
        assert sol.x[1] == pytest.approx(1.0)


class TestSolveMasterDual:
    def test_dual_strong_duality(self):
        problem = two_part_problem()
        lp = solve_master_lp(ALL_COLUMNS, problem, (FRESH,))
        dual = solve_master_dual(ALL_COLUMNS, problem, (FRESH,))
        assert dual is not None
        d = np.asarray(problem.demands, dtype=float)
        assert float(d @ dual) == pytest.approx(lp.value, abs=1e-6)

    def test_dual_feasible(self):
        problem = two_part_problem()
        dual = solve_master_dual(ALL_COLUMNS, problem, (FRESH,))
        assert dual is not None
        assert np.all(dual >= -1e-9)
        # Each column: sum_i a_ji y_i <= cost of its context.
        for col in ALL_COLUMNS:
            value = float(sum(a * y for a, y in zip(col.pattern, dual)))
            assert value <= FRESH.cost + 1e-6

    def test_reduced_cost_nonnegative_for_master_columns(self):
        # At the optimum every master column has nonnegative reduced
        # cost (negative reduced cost is what pricing hunts for).
        problem = two_part_problem()
        dual = solve_master_dual(ALL_COLUMNS, problem, (FRESH,))
        assert dual is not None
        for col in ALL_COLUMNS:
            assert reduced_cost(FRESH, col.pattern, dual) >= -1e-6

    def test_reduced_cost_formula(self):
        # cost - sum(pattern_i * dual_i), hand computed.
        dual = np.array([2.0, 3.0])
        assert reduced_cost(FRESH, (2, 1), dual) == pytest.approx(10.0 - 7.0)

    def test_no_columns_returns_none_when_demand_positive(self):
        problem = two_part_problem()
        assert solve_master_dual((), problem, (FRESH,)) is None

    def test_no_columns_zero_demand_returns_zeros(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[4000], demands=[0], bars=(), kerf_mm=0,
        )
        dual = solve_master_dual((), problem, ())
        assert dual is not None
        assert np.all(dual == 0.0)
