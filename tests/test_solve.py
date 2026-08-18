"""Tests for the 1D cutting orchestrator (prechecks, solve, assembly).

What gets verified: a feasible end-to-end solve, honest infeasibility
with a human-readable reason for each precheck class, the empty zero
demand case, kerf accounting in the offcut, remnant-first consumption
at zero cost, and the shape of the diagnostics dict.
"""

import numpy as np
import pytest

from constraint_engine.geometry.master import MasterColumn
from constraint_engine.geometry.solve import solve
from constraint_engine.geometry.types import BarSpec, CuttingProblem1D


def bar(length_mm: int, cost: float, bar_id: str, origin: str = "new", age: float = 0.0) -> BarSpec:
    return BarSpec(length_mm=length_mm, cost=cost, origin=origin, id=bar_id, age_days=age)


def two_part_problem() -> CuttingProblem1D:
    """Two part types, two fresh 6000mm bars at cost 10, kerf zero."""
    return CuttingProblem1D(
        part_lengths_mm=[4000, 2000],
        demands=[2, 2],
        bars=(
            bar(6000, 10.0, "b0"),
            bar(6000, 10.0, "b1"),
        ),
        kerf_mm=0,
    )


class TestSolveFeasible:
    def test_end_to_end_hand_instance(self):
        result = solve(two_part_problem())
        assert result.feasible
        assert result.assignments
        # Demand covered exactly: one 4000 and one 2000 per bar.
        counts = np.zeros(2, dtype=int)
        for assignment in result.assignments:
            counts += np.asarray(assignment.pattern, dtype=int)
        assert list(counts) == [2, 2]
        # Every assignment names a real bar and its offcut is consistent.
        for assignment in result.assignments:
            assert assignment.bar_id in ("b0", "b1")
            assert assignment.offcut_mm >= 0
        used = sum(assignment.offcut_mm for assignment in result.assignments)
        total = sum(6000 for _ in result.assignments)
        parts = 2 * 4000 + 2 * 2000
        assert total - parts == used
        # Two bars of cost 10, no waste, LP below or at the IP value.
        assert result.ip_value == pytest.approx(20.0)
        assert result.lp_value <= result.ip_value + 1e-6
        assert result.gap >= 0.0
        assert result.stats["waste_mm"] == 0
        assert result.stats["bars_used"] == 2
        assert result.stats["remnant_bars_used"] == 0
        assert result.stats["ip_status"] in ("optimal", "incumbent", "heuristic")

    def test_stats_carry_column_generation_diagnostics(self):
        result = solve(two_part_problem())
        assert result.stats["cg_rounds"] >= 1
        assert result.stats["columns_kept"] >= 1
        assert result.stats["bars_total"] == 2

    def test_single_bar_multi_part_is_feasible(self):
        # One bar must carry both part types: 2500 + kerf + 1500 fits
        # in 6000, but the (1, 1) pattern has positive reduced cost, so
        # column generation never prices it. The restricted integer
        # master is infeasible over the seeds alone and must be
        # repaired by the fill instead of reported as infeasible.
        problem = CuttingProblem1D(
            part_lengths_mm=[2500, 1500],
            demands=[1, 1],
            bars=(bar(6000, 60.0, "b0"),),
            kerf_mm=3,
        )
        result = solve(problem)
        assert result.feasible
        assert len(result.assignments) == 1
        assert tuple(result.assignments[0].pattern) == (1, 1)
        assert result.stats["ip_status"] in ("optimal", "incumbent", "heuristic")


class TestSolveInfeasible:
    def test_part_longer_than_any_bar(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[7000],
            demands=[1],
            bars=(bar(6000, 10.0, "b0"),),
            kerf_mm=0,
        )
        result = solve(problem)
        assert not result.feasible
        assert result.assignments == ()
        assert result.infeasible_reasons
        assert any("longer" in reason for reason in result.infeasible_reasons)
        assert result.stats["bars_used"] == 0

    def test_not_enough_bars(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[4000],
            demands=[3],
            bars=(bar(6000, 10.0, "b0"),),
            kerf_mm=0,
        )
        result = solve(problem)
        assert not result.feasible
        assert result.infeasible_reasons
        assert result.assignments == ()

    def test_infeasible_result_reports_zero_values(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[4000],
            demands=[3],
            bars=(bar(6000, 10.0, "b0"),),
            kerf_mm=0,
        )
        result = solve(problem)
        assert not result.feasible
        assert result.lp_value == 0.0
        assert result.ip_value == 0.0
        assert result.gap == 0.0

    def test_no_bars_at_all(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[2000],
            demands=[1],
            bars=(),
            kerf_mm=0,
        )
        result = solve(problem)
        assert not result.feasible
        assert result.infeasible_reasons


class TestZeroDemand:
    def test_zero_demand_is_feasible_and_empty(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[2000],
            demands=[0],
            bars=(bar(6000, 10.0, "b0"),),
            kerf_mm=0,
        )
        result = solve(problem)
        assert result.feasible
        assert result.assignments == ()
        assert result.lp_value == 0.0
        assert result.ip_value == 0.0
        assert result.gap == 0.0
        assert result.stats["bars_used"] == 0
        assert result.stats["ip_status"] == "optimal"


class TestKerfAndRemnants:
    def test_offcut_accounts_for_kerf(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[1999],
            demands=[2],
            bars=(bar(6000, 10.0, "b0"),),
            kerf_mm=3,
        )
        result = solve(problem)
        assert result.feasible
        assert len(result.assignments) == 1
        assignment = result.assignments[0]
        assert assignment.pattern == (2,)
        # 2 x 1999 plus one 3mm kerf, offcut is the rest.
        assert assignment.offcut_mm == 6000 - (2 * 1999 + 3)
        assert result.stats["waste_mm"] == assignment.offcut_mm + 3

    def test_remnant_consumed_at_zero_cost(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[2000],
            demands=[4],
            bars=(
                bar(6000, 10.0, "b_new"),
                bar(3000, 0.0, "b_rem", origin="remnant", age=5.0),
            ),
            kerf_mm=0,
        )
        result = solve(problem)
        assert result.feasible
        assert result.ip_value == pytest.approx(10.0)
        by_id = {assignment.bar_id: assignment for assignment in result.assignments}
        assert set(by_id) == {"b_new", "b_rem"}
        assert by_id["b_new"].pattern == (3,)
        assert by_id["b_new"].offcut_mm == 0
        assert by_id["b_rem"].pattern == (1,)
        assert by_id["b_rem"].offcut_mm == 1000
        assert result.stats["remnant_bars_used"] == 1

    def test_kerf_tightens_the_feasibility_bound(self):
        # Three 1999mm cuts need 6003mm with kerf 3; one 6000mm bar
        # cannot do it even though 5997mm of part length would fit.
        problem = CuttingProblem1D(
            part_lengths_mm=[1999],
            demands=[3],
            bars=(bar(6000, 10.0, "b0"),),
            kerf_mm=3,
        )
        result = solve(problem)
        assert not result.feasible
        assert result.infeasible_reasons


class TestInitialColumns:
    def test_warm_start_column_is_used_from_round_one(self):
        # The exact-fit (1, 1) pattern is what pricing discovers on a
        # cold start (two rounds: price in round one, converge in
        # round two). Handed in as a warm start, the master is already
        # optimal in round one.
        problem = two_part_problem()
        result = solve(problem, initial_columns=(MasterColumn(0, (1, 1)),))
        assert result.feasible
        assert result.ip_value == pytest.approx(20.0)
        assert result.stats["cg_rounds"] == 1

    def test_warm_start_deduped_against_seed(self):
        # (0, 3) is exactly the seed for part 1; passing it back must
        # not grow the pool. The cold-start pool ends at 3 columns
        # (two seeds plus the priced (1, 1)).
        problem = two_part_problem()
        result = solve(problem, initial_columns=(MasterColumn(0, (0, 3)),))
        assert result.feasible
        assert result.stats["columns_kept"] == 3

    def test_warm_start_rejects_bad_columns(self):
        problem = two_part_problem()
        with pytest.raises(ValueError, match="context_index"):
            solve(problem, initial_columns=(MasterColumn(7, (1, 0)),))
