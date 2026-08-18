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
