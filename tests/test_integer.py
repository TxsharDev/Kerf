"""Tests for the integer master problem (final IP) for 1D cutting.

The IP decides how many bars of each context are cut with each pattern;
its solution is the plan itself. What gets verified: optimality on a
hand instance, the integrality gap on a fractional LP case, capacity
shortage, zero demand, and the deterministic fallback for when the MIP
solver hands back no incumbent.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from constraint_engine.geometry.integer import (
    IP_HEURISTIC,
    IP_INFEASIBLE,
    IP_INCUMBENT,
    IP_NO_INCUMBENT,
    IP_OPTIMAL,
    solve_master_ip,
)
from constraint_engine.geometry.master import MasterColumn
from constraint_engine.geometry.patterns import Context
from constraint_engine.geometry.types import BarSpec, CuttingProblem1D


def two_part_problem():
    """Two part types, two fresh 6000mm bars, kerf zero."""
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

HAND_COLUMNS = (
    MasterColumn(0, (1, 0)),
    MasterColumn(0, (0, 3)),
    MasterColumn(0, (1, 1)),
)


class TestSolveMasterIp:
    def test_ip_optimal_on_hand_instance(self):
        problem = two_part_problem()
        result = solve_master_ip(HAND_COLUMNS, problem, (FRESH,))
        assert result.status == IP_OPTIMAL
        assert result.value == pytest.approx(20.0)
        # The exact-fit pattern twice: demand covered, two bars used.
        counts = _covered(2, result.columns, result.x)
        assert list(counts) == [2, 2]
        assert sum(result.x) <= FRESH.bar_count

    def test_ip_pays_integrality_gap(self):
        # Bar 10mm, parts 6/4/4, demand one of each. The LP relaxes to
        # 15 (1.5 fractional bars); the IP needs two full bars: 20.
        problem = CuttingProblem1D(
            part_lengths_mm=[6, 4, 4],
            demands=[1, 1, 1],
            bars=(
                BarSpec(length_mm=10, cost=10.0, origin="new", id="b0"),
                BarSpec(length_mm=10, cost=10.0, origin="new", id="b1"),
            ),
            kerf_mm=0,
        )
        ctx = Context(length_mm=10, cost=10.0, bar_ids=("b0", "b1"))
        columns = tuple(
            MasterColumn(0, p)
            for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 2, 0))
        )
        result = solve_master_ip(columns, problem, (ctx,))
        assert result.status == IP_OPTIMAL
        assert result.value == pytest.approx(20.0)
        counts = _covered(3, result.columns, result.x)
        assert (counts >= np.array([1, 1, 1])).all()
        assert sum(result.x) <= ctx.bar_count

    def test_capacity_shortage_is_infeasible(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[4000],
            demands=[2],
            bars=(BarSpec(length_mm=6000, cost=10.0, origin="new", id="b0"),),
            kerf_mm=0,
        )
        ctx = Context(length_mm=6000, cost=10.0, bar_ids=("b0",))
        columns = (MasterColumn(0, (1,)),)
        result = solve_master_ip(columns, problem, (ctx,))
        assert result.status == IP_INFEASIBLE
        assert result.x == ()

    def test_zero_demand_is_free(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[2000],
            demands=[0],
            bars=(BarSpec(length_mm=6000, cost=10.0, origin="new", id="b0"),),
            kerf_mm=0,
        )
        ctx = Context(length_mm=6000, cost=10.0, bar_ids=("b0",))
        columns = (MasterColumn(0, (1,)),)
        result = solve_master_ip(columns, problem, (ctx,))
        assert result.status == IP_OPTIMAL
        assert result.value == 0.0
        assert result.x == (0,)


class TestFallback:
    def test_floor_then_greedy_fill_when_solver_gives_nothing(self, monkeypatch):
        import constraint_engine.geometry.integer as integer_mod

        def fake_milp(*args, **kwargs):
            return SimpleNamespace(status=1, x=None, fun=None, message="iteration limit")

        monkeypatch.setattr(integer_mod, "milp", fake_milp)
        problem = two_part_problem()
        # Flooring the LP solution leaves one exact-fit bar's worth of
        # demand unmet; the greedy fill must finish the job on bar two.
        lp_x = (0.0, 0.0, 1.5)
        result = solve_master_ip(HAND_COLUMNS, problem, (FRESH,), lp_x=lp_x)
        assert result.status == IP_HEURISTIC
        counts = _covered(2, result.columns, result.x)
        assert (counts >= np.array([2, 2])).all()
        used = sum(x for c, x in zip(result.columns, result.x) if c.context_index == 0)
        assert used <= FRESH.bar_count

    def test_timed_out_solver_with_valid_incumbent_is_accepted(self, monkeypatch):
        import constraint_engine.geometry.integer as integer_mod

        def fake_milp(*args, **kwargs):
            return SimpleNamespace(
                status=1, x=np.array([0.0, 0.0, 2.0]), fun=20.0, message="time limit"
            )

        monkeypatch.setattr(integer_mod, "milp", fake_milp)
        problem = two_part_problem()
        result = solve_master_ip(HAND_COLUMNS, problem, (FRESH,))
        assert result.status == IP_INCUMBENT
        assert result.value == pytest.approx(20.0)
        assert result.x == (0, 0, 2)

    def test_fallback_admits_defeat_honestly(self, monkeypatch):
        import constraint_engine.geometry.integer as integer_mod

        def fake_milp(*args, **kwargs):
            return SimpleNamespace(status=1, x=None, fun=None, message="iteration limit")

        monkeypatch.setattr(integer_mod, "milp", fake_milp)
        problem = CuttingProblem1D(
            part_lengths_mm=[4000],
            demands=[3],
            bars=(
                BarSpec(length_mm=6000, cost=10.0, origin="new", id="b0"),
                BarSpec(length_mm=6000, cost=10.0, origin="new", id="b1"),
            ),
            kerf_mm=0,
        )
        ctx = Context(length_mm=6000, cost=10.0, bar_ids=("b0", "b1"))
        columns = (MasterColumn(0, (1,)),)
        result = solve_master_ip(columns, problem, (ctx,), lp_x=(0.0,))
        assert result.status == IP_NO_INCUMBENT
        assert result.x == ()


def _covered(n_parts, columns, x):
    counts = np.zeros(n_parts)
    for col, count in zip(columns, x):
        counts += np.asarray(col.pattern, dtype=float) * count
    return counts
