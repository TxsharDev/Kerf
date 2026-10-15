"""Tests for the first-fit decreasing plan over bar contexts.

The greedy plan seeds column generation and is the floor every solve
must match or beat, so it has to be a real plan: every demanded part
placed exactly as often as demanded, every pattern physically fitting
its bar, no context asked for more bars than it holds.
"""

from benchmarks.baselines import first_fit_decreasing as ffd_baseline
from constraint_engine.geometry.greedy import first_fit_decreasing
from constraint_engine.geometry.master import MasterColumn
from constraint_engine.geometry.patterns import pattern_fits, patterns_from_problem
from constraint_engine.geometry.types import BarSpec, CuttingProblem1D


def bar(length_mm, cost, bar_id, origin="new"):
    return BarSpec(length_mm=length_mm, cost=cost, origin=origin, id=bar_id)


def solve_greedy(problem):
    contexts = patterns_from_problem(problem)
    return contexts, first_fit_decreasing(problem, contexts)


class TestFirstFitDecreasing:
    def test_places_exact_demand_on_fitting_patterns(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[1200, 2500, 700],
            demands=[3, 2, 5],
            bars=tuple(bar(6000, 6.0, f"b{i}") for i in range(4)),
            kerf_mm=4,
        )
        contexts, plan = solve_greedy(problem)
        produced = [0, 0, 0]
        used = [0] * len(contexts)
        for column, count in zip(plan.columns, plan.counts):
            assert pattern_fits(
                column.pattern,
                problem.part_lengths_mm,
                problem.kerf_mm,
                contexts[column.context_index].length_mm,
            )
            used[column.context_index] += count
            for i, pieces in enumerate(column.pattern):
                produced[i] += pieces * count
        assert produced == problem.demands
        assert all(n <= c.bar_count for n, c in zip(used, contexts))
        assert plan.value == sum(used) * 6.0

    def test_matches_the_benchmark_baseline_bar_count(self):
        sizes = [42, 63, 67, 57, 22, 31, 49, 49, 70, 18, 55, 38, 61, 27]
        lengths = sorted(set(sizes))
        problem = CuttingProblem1D(
            part_lengths_mm=lengths,
            demands=[sizes.count(length) for length in lengths],
            bars=tuple(bar(100, 100.0, f"b{i}") for i in range(12)),
            kerf_mm=0,
        )
        _, plan = solve_greedy(problem)
        assert sum(plan.counts) == len(ffd_baseline(sizes, 100))

    def test_free_remnant_opens_before_purchased_stock(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[1500],
            demands=[2],
            bars=(
                bar(6000, 6.0, "n0"),
                bar(3000, 0.0, "r0", origin="remnant"),
            ),
            kerf_mm=0,
        )
        contexts, plan = solve_greedy(problem)
        remnant = next(i for i, c in enumerate(contexts) if c.cost == 0.0)
        assert plan.columns == (MasterColumn(remnant, (2,)),)
        assert plan.value == 0.0

    def test_short_bar_skipped_then_used_for_a_shorter_part(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[3000, 900],
            demands=[2, 1],
            bars=(
                bar(1000, 0.0, "r0", origin="remnant"),
                bar(6000, 6.0, "n0"),
            ),
            kerf_mm=0,
        )
        contexts, plan = solve_greedy(problem)
        by_length = {
            contexts[c.context_index].length_mm: c.pattern for c in plan.columns
        }
        assert by_length == {6000: (2, 0), 1000: (0, 1)}

    def test_returns_none_when_bars_run_out(self):
        problem = CuttingProblem1D(
            part_lengths_mm=[4000],
            demands=[2],
            bars=(bar(6000, 6.0, "b0"),),
            kerf_mm=0,
        )
        _, plan = solve_greedy(problem)
        assert plan is None
