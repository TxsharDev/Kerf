"""Bounded knapsack pricing solver.

Exact bounded knapsack used as the pricing oracle for 1D cutting stock
column generation. Each item type has a profit, an integer weight, and an
integer bound on how many copies may be taken; the solver maximizes total
profit subject to the total weight staying within capacity.

Method: expand every bounded item into binary-decomposed split items
(a bound of 13 becomes sections 1 + 2 + 4 + 6, whose subset sums cover
every count from 0 to 13), then run a 0/1 dynamic program over the
capacity. Backtracking through the recorded improvement masks
reconstructs the chosen counts.

Complexity: O(M * C) time and memory per solve, where M is the number of
split items and C is the capacity. Deterministic: on ties the item is not
taken, so identical inputs always yield identical counts.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bounded_knapsack", "decompose_bound"]


def decompose_bound(bound: int) -> list[int]:
    """Split a bound into sections whose subset sums cover 0..bound.

    Ascending powers of two, with the remainder as the final section:
    13 -> [1, 2, 4, 6], 3 -> [1, 2], 1 -> [1], 0 -> [].
    """
    if bound < 0:
        raise ValueError(f"bound must be non-negative, got {bound}")
    sections: list[int] = []
    remaining = bound
    power = 1
    while power < remaining:
        sections.append(power)
        remaining -= power
        power *= 2
    if remaining > 0:
        sections.append(remaining)
    return sections


def _validate(profits, weights, bounds, capacity) -> int:
    n = len(profits)
    if len(weights) != n or len(bounds) != n:
        raise ValueError(
            f"profits, weights, and bounds must have equal length "
            f"(got {n}, {len(weights)}, {len(bounds)})"
        )
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")
    for w in weights:
        if w < 0:
            raise ValueError(f"weights must be non-negative, got {w}")
    for b in bounds:
        if b < 0:
            raise ValueError(f"bounds must be non-negative, got {b}")
    return n


def bounded_knapsack(
    profits: list[float],
    weights: list[int],
    bounds: list[int],
    capacity: int,
) -> tuple[np.ndarray, float]:
    """Solve the bounded knapsack exactly.

    Args:
        profits: Profit per copy of each item type (may be zero).
        weights: Integer weight per copy of each item type.
        bounds: Maximum number of copies of each item type.
        capacity: Maximum total weight.

    Returns:
        ``(counts, value)`` where ``counts`` is an int64 array with the
        chosen number of copies of each item type and ``value`` is the
        total profit of that choice (the optimum).
    """
    n = _validate(profits, weights, bounds, capacity)
    counts = np.zeros(n, dtype=np.int64)
    if n == 0 or capacity == 0:
        return counts, 0.0

    # Expand each bounded item into binary split items.
    splits: list[tuple[int, float, int, int]] = []
    for i, (p, w, b) in enumerate(zip(profits, weights, bounds)):
        for section in decompose_bound(b):
            splits.append((section * w, section * p, i, section))

    dp = np.zeros(capacity + 1, dtype=np.float64)
    # Per split item: (weight, improvement mask, original index, section).
    decisions: list[tuple[int, np.ndarray, int, int]] = []

    for w_k, p_k, orig, section in splits:
        if w_k > capacity:
            # Too heavy to ever fit; it can be in no feasible solution.
            continue
        prev = dp
        if w_k == 0:
            cand = prev + p_k
            improved = cand > prev
            dp = np.where(improved, cand, prev)
        else:
            dp = prev.copy()
            tail = prev[w_k:]
            cand = prev[: capacity + 1 - w_k] + p_k
            improved = cand > tail
            if improved.any():
                dp[w_k:][improved] = cand[improved]
        decisions.append((w_k, improved, orig, section))

    # Backtrack: at weight budget w, item k is in some optimal solution
    # exactly when taking it strictly improved the table at that budget.
    # The mask for item k covers budgets w_k..capacity at offset w - w_k.
    w = capacity
    for w_k, improved, orig, section in reversed(decisions):
        if w >= w_k and improved[w - w_k]:
            counts[orig] += section
            w -= w_k

    return counts, float(dp[capacity])
