"""Tests for the 1D cutting stock pricing round and column generation.

Pricing asks: given dual prices on the part types, which pattern is
worth adding to the master? A pattern is worth adding when its dual
value exceeds the cost of the context it cuts on, i.e. when it has
negative reduced cost beyond a small tolerance.

Effective weights make the physical kerf model a plain knapsack:
weight of part i is ``length_i + kerf``, capacity of a context is
``bar_length + kerf``. Then a pattern fits exactly when its total
weight is within capacity.
"""

import pytest

from constraint_engine.geometry.knapsack import bounded_knapsack
from constraint_engine.geometry.master import MasterColumn
from constraint_engine.geometry.pricing import (
    price_columns,
    run_column_generation,
)
from constraint_engine.geometry.patterns import Context, patterns_from_problem
from constraint_engine.geometry.types import BarSpec, CuttingProblem1D


def two_part_problem():
    """Two part types, two fresh 6000mm bars, kerf zero.

    Parts: 4000mm (demand 2), 2000mm (demand 2). The exact-fit
    pattern (1, 1) per bar is the optimum at cost 20, but it is not a
    single-type pattern, so seeding alone cannot see it.
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


class TestPricingMath:
    def test_effective_weight_round_trip(self):
        # The knapsack fit check must agree with the physical fit check:
        # a pattern fits on a bar iff its effective weight fits the
        # effective capacity.
        problem = two_part_problem()
        context = FRESH
        kerf = problem.kerf_mm
        capacity = context.length_mm + kerf
        weights = [l + kerf for l in problem.part_lengths_mm]
        for pattern in ((1, 0), (0, 3), (1, 1), (0, 4)):
            total_weight = sum(c * w for c, w in zip(pattern, weights))
            assert (total_weight <= capacity) == (
                sum(c * l for c, l in zip(pattern, problem.part_lengths_mm))
                + kerf * max(0, sum(pattern) - 1)
                <= context.length_mm
            )


class TestPriceColumns:
    def test_zero_duals_add_nothing(self):
        problem = two_part_problem()
        dual = [0.0, 0.0]
        assert price_columns(problem, (FRESH,), dual) == ()

    def test_mixed_pattern_beats_seed_patterns(self):
        # Duals that make the exact-fit (1, 1) pattern the best:
        # (1,0) scores 10, (0,3) scores 3, (1,1) scores 11 > cost 10.
        problem = two_part_problem()
        dual = [10.0, 1.0]
        new = price_columns(problem, (FRESH,), dual)
        assert new == (MasterColumn(context_index=0, pattern=(1, 1)),)

    def test_tolerance_suppresses_marginally_unprofitable(self):
        # Pattern value exactly at cost is not a negative reduced cost.
        problem = two_part_problem()
        dual = [10.0, 0.0]
        new = price_columns(problem, (FRESH,), dual, tolerance=1e-4)
        assert new == ()

    def test_zero_demand_parts_never_enter_patterns(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[2000, 1000],
            demands=[1, 0],
            bars=(BarSpec(length_mm=3000, cost=5.0, origin="new", id="b0"),),
            kerf_mm=0,
        )
        context = Context(length_mm=3000, cost=5.0, bar_ids=("b0",))
        dual = [10.0, 100.0]  # the zero-demand part would be tempting
        new = price_columns(problem, (context,), dual)
        assert new == (MasterColumn(context_index=0, pattern=(1, 0)),)

    def test_remnant_context_costs_nothing(self):
        # A zero-cost context adds any pattern with positive dual value.
        problem = two_part_problem()
        remnant = Context(length_mm=6000, cost=0.0, bar_ids=("r0",))
        dual = [10.0, 1.0]
        new = price_columns(problem, (remnant,), dual)
        assert new == (MasterColumn(context_index=0, pattern=(1, 1)),)


class TestRunColumnGeneration:
    def test_hand_instance_converges_on_exact_fit(self):
        problem = two_part_problem()
        contexts = patterns_from_problem(problem)
        result = run_column_generation(problem, contexts)
        assert result.stop == "converged"
        assert result.infeasible is False
        # The exact-fit column must have been discovered by pricing.
        assert (0, (1, 1)) in {(c.context_index, c.pattern) for c in result.columns}
        assert result.solution is not None
        assert result.solution.status == "optimal"
        assert result.solution.value == pytest.approx(20.0)

    def test_no_demand_stops_immediately(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[2000],
            demands=[0],
            bars=(BarSpec(length_mm=6000, cost=10.0, origin="new", id="b0"),),
            kerf_mm=0,
        )
        contexts = patterns_from_problem(problem)
        result = run_column_generation(problem, contexts)
        assert result.stop == "no_demand"
        assert result.columns == ()
        assert result.infeasible is False

    def test_part_fitting_nowhere_is_infeasible(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[9000],
            demands=[1],
            bars=(BarSpec(length_mm=6000, cost=10.0, origin="new", id="b0"),),
            kerf_mm=0,
        )
        contexts = patterns_from_problem(problem)
        result = run_column_generation(problem, contexts)
        assert result.stop == "infeasible"
        assert result.infeasible is True
        assert result.solution is None

    def test_capacity_shortage_is_infeasible(self):
        # One bar cannot hold two 4000mm parts; no pricing can fix that.
        problem = CuttingProblem1D(
            part_lengths_mm=[4000],
            demands=[2],
            bars=(BarSpec(length_mm=6000, cost=10.0, origin="new", id="b0"),),
            kerf_mm=0,
        )
        contexts = patterns_from_problem(problem)
        result = run_column_generation(problem, contexts)
        assert result.stop == "infeasible"
        assert result.infeasible is True

    def test_remnant_context_used_before_fresh(self):
        # A free remnant covering all demand drives the LP value to zero.
        problem = CuttingProblem1D(
            part_lengths_mm=[2000],
            demands=[3],
            bars=(
                BarSpec(length_mm=6000, cost=10.0, origin="new", id="n0"),
                BarSpec(length_mm=6000, cost=0.0, origin="remnant", id="r0"),
            ),
            kerf_mm=0,
        )
        contexts = patterns_from_problem(problem)
        result = run_column_generation(problem, contexts)
        assert result.stop == "converged"
        assert result.solution.value == pytest.approx(0.0)

    def test_max_rounds_stops_honestly(self):
        problem = two_part_problem()
        contexts = patterns_from_problem(problem)
        result = run_column_generation(problem, contexts, max_rounds=100)
        # The hand instance converges in a handful of rounds, nowhere
        # near the guard, so the stop reason must be "converged".
        assert result.stop == "converged"
        assert result.rounds < 100

    def test_knapsack_bounds_respected_in_pricing(self):
        # Pricing must never propose more copies of a part than the
        # demand: bounds cap the knapsack at min(demand, fit count).
        problem = two_part_problem()
        context = FRESH
        weights = [4000, 2000]
        capacity = 6000
        counts, value = bounded_knapsack(
            [10.0, 1.0], weights, [2, 2], capacity
        )
        assert int(counts[0]) <= 2
        assert int(counts[1]) <= 2


class TestInitialColumns:
    """Warm-start columns feed the generator instead of being dropped.

    A caller (the engine's replan) may hand in patterns from a previous
    plan. They must enter the column pool, dedup against the seeds and
    against later pricing rounds, and be rejected with a clear error
    when they are malformed.
    """

    def test_initial_column_is_kept_and_deduped(self):
        problem = two_part_problem()
        contexts = patterns_from_problem(problem)
        result = run_column_generation(
            problem,
            contexts,
            initial_columns=(MasterColumn(0, (1, 1)),),
        )
        assert result.stop == "converged"
        assert not result.infeasible
        keys = [(c.context_index, c.pattern) for c in result.columns]
        # The warm-start column is in the pool, and pricing rediscovers
        # the same (1, 1) pattern without adding a duplicate.
        assert keys.count((0, (1, 1))) == 1
        assert result.solution is not None
        assert result.solution.value == pytest.approx(20.0)

    def test_initial_column_deduped_against_seed(self):
        # (0, 3) is exactly the seed for part 1 on context 0
        # (6000 // 2000 = 3 copies); passing it back must not
        # create a second copy of the column.
        problem = two_part_problem()
        contexts = patterns_from_problem(problem)
        result = run_column_generation(
            problem,
            contexts,
            initial_columns=(MasterColumn(0, (0, 3)),),
        )
        keys = [(c.context_index, c.pattern) for c in result.columns]
        assert keys.count((0, (0, 3))) == 1

    def test_rejects_context_index_out_of_range(self):
        problem = two_part_problem()
        contexts = patterns_from_problem(problem)
        with pytest.raises(ValueError, match="context_index"):
            run_column_generation(
                problem, contexts, initial_columns=(MasterColumn(5, (1, 0)),)
            )

    def test_rejects_pattern_with_wrong_length(self):
        problem = two_part_problem()
        contexts = patterns_from_problem(problem)
        with pytest.raises(ValueError, match="pattern"):
            run_column_generation(
                problem,
                contexts,
                initial_columns=(MasterColumn(0, (1, 1, 0)),),
            )

    def test_rejects_pattern_that_does_not_fit(self):
        # Four 4000mm parts need 16000mm; context 0 holds 6000mm.
        problem = two_part_problem()
        contexts = patterns_from_problem(problem)
        with pytest.raises(ValueError, match="fit"):
            run_column_generation(
                problem, contexts, initial_columns=(MasterColumn(0, (4, 0)),)
            )

    def test_rejects_negative_pattern_count(self):
        problem = two_part_problem()
        contexts = patterns_from_problem(problem)
        with pytest.raises(ValueError, match="non-negative"):
            run_column_generation(
                problem,
                contexts,
                initial_columns=(MasterColumn(0, (-1, 2)),),
            )
