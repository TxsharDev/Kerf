"""Tests for the 1D cutting geometry types.

These are the data contracts the solver and the Kerf product build on:
immutable, structural, validated at construction. Feasibility (can the
parts actually be cut) is a *result* property, not a construction error,
so uncuttable problems can still be built, solved, and diagnosed.
"""

import dataclasses

import pytest

from constraint_engine.geometry.types import (
    BarAssignment,
    BarSpec,
    CuttingProblem1D,
    CuttingResult1D,
)


class TestBarSpec:
    def test_fields(self):
        bar = BarSpec(length_mm=6000, cost=24.0, origin="new", id="b1", age_days=12.0)
        assert bar.length_mm == 6000
        assert bar.cost == 24.0
        assert bar.origin == "new"
        assert bar.id == "b1"
        assert bar.age_days == 12.0

    def test_defaults(self):
        bar = BarSpec(length_mm=6000, cost=0.0, origin="remnant")
        assert bar.id == ""
        assert bar.age_days == 0.0

    def test_frozen(self):
        bar = BarSpec(length_mm=6000, cost=1.0, origin="new")
        with pytest.raises(dataclasses.FrozenInstanceError):
            bar.length_mm = 100


class TestCuttingProblem1D:
    def test_valid_problem(self):
        p = CuttingProblem1D(
            part_lengths_mm=[1000, 2000],
            demands=[3, 1],
            bars=(BarSpec(length_mm=6000, cost=24.0, origin="new", id="b1"),),
        )
        assert p.part_lengths_mm == [1000, 2000]
        assert p.demands == [3, 1]
        assert len(p.bars) == 1

    def test_defaults(self):
        p = CuttingProblem1D(
            part_lengths_mm=[1000], demands=[1], bars=(),
        )
        assert p.kerf_mm == 3
        assert p.min_offcut_mm == 50
        assert p.name == ""

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            CuttingProblem1D(
                part_lengths_mm=[1000, 2000], demands=[1], bars=(),
            )

    def test_negative_demand_rejected(self):
        with pytest.raises(ValueError):
            CuttingProblem1D(
                part_lengths_mm=[1000], demands=[-1], bars=(),
            )

    def test_zero_part_length_rejected(self):
        with pytest.raises(ValueError):
            CuttingProblem1D(
                part_lengths_mm=[0], demands=[1], bars=(),
            )

    def test_negative_kerf_rejected(self):
        with pytest.raises(ValueError):
            CuttingProblem1D(
                part_lengths_mm=[1000], demands=[1], bars=(), kerf_mm=-1,
            )

    def test_zero_length_bar_rejected(self):
        with pytest.raises(ValueError):
            CuttingProblem1D(
                part_lengths_mm=[1000],
                demands=[1],
                bars=(BarSpec(length_mm=0, cost=0.0, origin="new"),),
            )

    def test_frozen(self):
        p = CuttingProblem1D(
            part_lengths_mm=[1000], demands=[1], bars=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.demands = [99]

    def test_input_lists_copied_not_shared(self):
        # The problem owns its own copies: mutating the caller's list
        # after construction does not reach into the problem.
        lengths = [1000]
        demands = [1]
        p = CuttingProblem1D(part_lengths_mm=lengths, demands=demands, bars=())
        lengths[0] = 5000
        demands[0] = 99
        assert p.part_lengths_mm == [1000]
        assert p.demands == [1]


class TestBarAssignment:
    def test_fields(self):
        a = BarAssignment(bar_id="b1", pattern=(2, 1), offcut_mm=350)
        assert a.bar_id == "b1"
        assert a.pattern == (2, 1)
        assert a.offcut_mm == 350

    def test_frozen(self):
        a = BarAssignment(bar_id="b1", pattern=(2, 1), offcut_mm=350)
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.offcut_mm = 0


class TestCuttingResult1D:
    def test_infeasible_result_defaults(self):
        r = CuttingResult1D(
            feasible=False,
            assignments=(),
            lp_value=0.0,
            ip_value=0.0,
            gap=0.0,
            solve_time_s=0.01,
            stats={},
            infeasible_reasons=("part 0 too long",),
        )
        assert not r.feasible
        assert r.infeasible_reasons == ("part 0 too long",)

    def test_infeasible_reasons_default_empty(self):
        r = CuttingResult1D(
            feasible=True,
            assignments=(),
            lp_value=0.0,
            ip_value=0.0,
            gap=0.0,
            solve_time_s=0.0,
            stats={},
        )
        assert r.infeasible_reasons == ()

    def test_frozen(self):
        r = CuttingResult1D(
            feasible=True,
            assignments=(),
            lp_value=0.0,
            ip_value=0.0,
            gap=0.0,
            solve_time_s=0.0,
            stats={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.feasible = False
