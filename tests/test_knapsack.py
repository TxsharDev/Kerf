"""Tests for the bounded knapsack pricing solver.

The bounded knapsack is the pricing oracle for 1D cutting stock column
generation: given per-item profits, weights, and integer bounds, find the
feasible count vector with maximum total value. Correctness here is the
foundation for the whole engine, so these tests cross-check the solver
against an independent brute-force enumeration on random instances.

Contract under test (derived from the spec, not the implementation):

- ``bounded_knapsack(profits, weights, bounds, capacity)`` returns
  ``(counts, value)`` where ``counts`` is a numpy integer array of length
  ``len(profits)`` and ``value`` is a float.
- The returned counts always satisfy every constraint:
  ``0 <= counts[i] <= bounds[i]`` and ``sum(counts * weights) <= capacity``.
- The reported ``value`` equals ``sum(counts * profits)`` exactly.
- ``value`` is the optimal (maximum) feasible value.
- All inputs empty or capacity 0 yields zero value and zero counts.
- The solver is deterministic: same inputs give the same counts.
"""

import itertools

import numpy as np
import pytest

from constraint_engine.geometry.knapsack import bounded_knapsack, decompose_bound


def _brute_force_optimum(profits, weights, bounds, capacity):
    """Enumerate every feasible count vector; return the maximum value.

    Independent reference implementation, written from the problem
    statement. Small instances only.
    """
    best = 0.0
    ranges = [range(b + 1) for b in bounds]
    for combo in itertools.product(*ranges):
        total_w = sum(c * w for c, w in zip(combo, weights))
        if total_w > capacity:
            continue
        total_p = sum(c * p for c, p in zip(combo, profits))
        if total_p > best:
            best = total_p
    return best


def _assert_feasible_and_consistent(counts, value, profits, weights, bounds, capacity):
    assert counts.shape == (len(profits),)
    assert np.issubdtype(counts.dtype, np.integer)
    assert (counts >= 0).all()
    assert (counts <= np.asarray(bounds)).all()
    assert sum(int(c) * int(w) for c, w in zip(counts, weights)) <= capacity
    expected = float(sum(int(c) * p for c, p in zip(counts, profits)))
    assert value == pytest.approx(expected, abs=1e-9)


class TestBoundedKnapsack:
    def test_single_item_fits(self):
        counts, value = bounded_knapsack([6.0], [2], [1], 10)
        assert list(counts) == [1]
        assert value == pytest.approx(6.0)

    def test_single_item_too_heavy(self):
        counts, value = bounded_knapsack([6.0], [7], [3], 5)
        assert list(counts) == [0]
        assert value == pytest.approx(0.0)

    def test_bound_limits_count(self):
        # Capacity fits all 3 copies, but the bound is 2.
        counts, value = bounded_knapsack([5.0], [1], [2], 100)
        assert list(counts) == [2]
        assert value == pytest.approx(10.0)

    def test_capacity_limits_count(self):
        # Bound allows 10, but capacity only fits 3.
        counts, value = bounded_knapsack([5.0], [4], [10], 12)
        assert list(counts) == [3]
        assert value == pytest.approx(15.0)

    def test_hand_computed_optimum(self):
        # Optimum: 2 of item 2 (w=4, p=13) + 1 of item 0 (w=2, p=6)
        # = weight 10, value 32. Unique optimum.
        profits = [6.0, 10.0, 13.0]
        weights = [2, 5, 4]
        bounds = [3, 2, 2]
        capacity = 10
        counts, value = bounded_knapsack(profits, weights, bounds, capacity)
        assert list(counts) == [1, 0, 2]
        assert value == pytest.approx(32.0)

    def test_greedy_by_ratio_fails(self):
        # Best ratio is item 0 (100/51), but taking it blocks two
        # slightly-cheaper items that fit together for more value.
        profits = [100.0, 99.0, 99.0]
        weights = [51, 50, 50]
        bounds = [1, 1, 1]
        capacity = 100
        counts, value = bounded_knapsack(profits, weights, bounds, capacity)
        assert list(counts) == [0, 1, 1]
        assert value == pytest.approx(198.0)

    def test_bound_forced_into_optimum(self):
        # Value 30 is reachable several ways; the mid item's bound of 1
        # must be honored whichever way the solver picks.
        profits = [10.0, 10.0, 10.0]
        weights = [4, 4, 4]
        bounds = [2, 1, 3]
        capacity = 12
        counts, value = bounded_knapsack(profits, weights, bounds, capacity)
        assert value == pytest.approx(30.0)
        assert counts[1] <= 1

    def test_empty_inputs(self):
        counts, value = bounded_knapsack([], [], [], 10)
        assert list(counts) == []
        assert value == pytest.approx(0.0)

    def test_zero_capacity(self):
        counts, value = bounded_knapsack([1.0, 2.0], [1, 1], [3, 3], 0)
        assert list(counts) == [0, 0]
        assert value == pytest.approx(0.0)

    def test_zero_profit_items(self):
        counts, value = bounded_knapsack([0.0, 5.0], [1, 1], [3, 1], 4)
        _assert_feasible_and_consistent(counts, value, [0.0, 5.0], [1, 1], [3, 1], 4)
        assert value == pytest.approx(5.0)

    def test_exact_capacity_fill(self):
        counts, value = bounded_knapsack([1.0, 2.0, 3.0], [2, 3, 4], [5, 5, 5], 9)
        _assert_feasible_and_consistent(counts, value, [1.0, 2.0, 3.0], [2, 3, 4], [5, 5, 5], 9)
        assert value == pytest.approx(_brute_force_optimum([1.0, 2.0, 3.0], [2, 3, 4], [5, 5, 5], 9))

    def test_returns_numpy_integer_counts_and_float_value(self):
        counts, value = bounded_knapsack([1.0], [1], [1], 1)
        assert isinstance(counts, np.ndarray)
        assert np.issubdtype(counts.dtype, np.integer)
        assert isinstance(value, float)

    def test_deterministic(self):
        args = ([3.0, 7.0, 2.0], [2, 3, 5], [4, 2, 3], 12)
        first = bounded_knapsack(*args)
        second = bounded_knapsack(*args)
        assert np.array_equal(first[0], second[0])
        assert first[1] == second[1]

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            bounded_knapsack([1.0, 2.0], [1], [1, 1], 5)
        with pytest.raises(ValueError):
            bounded_knapsack([1.0], [1, 2], [1, 1], 5)

    def test_negative_inputs_rejected(self):
        with pytest.raises(ValueError):
            bounded_knapsack([1.0], [-1], [1], 5)
        with pytest.raises(ValueError):
            bounded_knapsack([1.0], [1], [-1], 5)
        with pytest.raises(ValueError):
            bounded_knapsack([1.0], [1], [1], -1)


class TestBruteForceCrossCheck:
    """Randomized differential testing against full enumeration."""

    def test_matches_brute_force_on_random_instances(self):
        rng = np.random.default_rng(42)
        for trial in range(25):
            n = int(rng.integers(1, 6))  # 1..5 items
            profits = [round(float(rng.uniform(0.0, 10.0)), 3) for _ in range(n)]
            weights = [int(rng.integers(1, 21)) for _ in range(n)]
            bounds = [int(rng.integers(0, 5)) for _ in range(n)]
            capacity = int(rng.integers(0, 60))

            counts, value = bounded_knapsack(profits, weights, bounds, capacity)
            _assert_feasible_and_consistent(
                counts, value, profits, weights, bounds, capacity
            )
            expected = _brute_force_optimum(profits, weights, bounds, capacity)
            assert value == pytest.approx(expected, abs=1e-9), (
                f"trial {trial}: value {value} != optimum {expected} "
                f"for n={n} profits={profits} weights={weights} "
                f"bounds={bounds} capacity={capacity}"
            )

    def test_matches_brute_force_with_zero_bounds_and_ties(self):
        # Deliberate corner mix: zero bounds, weight 1 items, equal profits.
        cases = [
            ([5.0, 5.0], [1, 1], [0, 3], 2),
            ([0.0, 0.0, 0.0], [2, 2, 2], [3, 3, 3], 6),
            ([2.5], [1], [0], 1),
            ([1.0, 2.0, 1.0], [3, 3, 3], [1, 1, 1], 9),
            ([10.0, 10.0, 10.0], [6, 6, 6], [2, 2, 2], 12),
        ]
        for i, (profits, weights, bounds, capacity) in enumerate(cases):
            counts, value = bounded_knapsack(profits, weights, bounds, capacity)
            _assert_feasible_and_consistent(counts, value, profits, weights, bounds, capacity)
            expected = _brute_force_optimum(profits, weights, bounds, capacity)
            assert value == pytest.approx(expected, abs=1e-9), f"case {i}: {value} != {expected}"

    def test_larger_random_instances_feasible_and_optimal(self):
        # Bigger items and capacities; brute force still affordable
        # because bounds stay small.
        rng = np.random.default_rng(7)
        for trial in range(10):
            n = int(rng.integers(2, 6))
            profits = [round(float(rng.uniform(1.0, 50.0)), 2) for _ in range(n)]
            weights = [int(rng.integers(5, 60)) for _ in range(n)]
            bounds = [int(rng.integers(1, 4)) for _ in range(n)]
            capacity = int(rng.integers(50, 300))

            counts, value = bounded_knapsack(profits, weights, bounds, capacity)
            _assert_feasible_and_consistent(counts, value, profits, weights, bounds, capacity)
            expected = _brute_force_optimum(profits, weights, bounds, capacity)
            assert value == pytest.approx(expected, abs=1e-6), f"trial {trial}: {value} != {expected}"


class TestDecomposeBound:
    """Binary decomposition must let subset sums represent every count.

    The DP expands a bounded item of bound ``b`` into unit-power split
    items; the expansion is valid only if every integer in ``0..b`` is
    representable as a subset sum of the sections.
    """

    def test_known_decomposition(self):
        assert decompose_bound(0) == []
        assert decompose_bound(1) == [1]
        assert decompose_bound(3) == [1, 2]
        assert decompose_bound(13) == [1, 2, 4, 6]

    def test_sections_sum_to_bound(self):
        for b in range(0, 65):
            assert sum(decompose_bound(b)) == b

    def test_subset_sums_cover_every_count(self):
        for b in range(0, 65):
            sections = decompose_bound(b)
            reachable = set()
            for r in range(len(sections) + 1):
                for combo in itertools.combinations(sections, r):
                    reachable.add(sum(combo))
            assert reachable == set(range(b + 1)), f"bound {b} not fully coverable"

    def test_sections_are_powers_of_two_except_last(self):
        for b in range(2, 65):
            sections = decompose_bound(b)
            for s in sections[:-1]:
                assert s & (s - 1) == 0, f"{s} is not a power of two (bound {b})"
            assert sections[-1] <= sections[-2] * 2
