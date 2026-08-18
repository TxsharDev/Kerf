# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- Initial project scaffold (pyproject, pytest config, changelog).
- Constraint Engine core types: Resource, Task, Constraint, Objective, Solution, Reason.
- Bounded knapsack solver (binary decomposition + 0/1 DP) for pricing.
- 1D cutting stock types: BarSpec, CuttingProblem1D, BarAssignment, CuttingResult1D.
- Pattern helpers and bar context grouping (oldest remnant first, deterministic order).
- Master LP, capacity-free master dual, and reduced cost computation.
- Pricing round (effective-weight knapsack) and column generation loop.
- Integer master solver (HiGHS via scipy.milp) with status honesty: optimal, incumbent, heuristic, infeasible, no_incumbent.
- Deterministic floor-then-knapsack fallback fill for when the MIP solver returns no incumbent.
- 1D cutting orchestrator: feasibility prechecks, column generation, integer solve, bar assembly, and diagnostics.
- Warm-start `initial_columns` on the column generation loop: validated (context index, pattern length, non-negative counts, physical fit) and deduped against seeds and pricing.
- `initial_columns` plumbed through the 1D `solve` orchestrator for replans that reuse patterns from a previous plan.
