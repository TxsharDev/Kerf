# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-08-17

### Added
- Initial project scaffold.
- Constraint Engine core types: Resource, Task, Constraint, Objective, Solution, Reason.
- 1D cutting stock solver via column generation with bounded knapsack pricing.
- Kerf domain model: Material, StockItem, CuttingList, Part, Plan, Cut, SavingsEvent.
- CSV importer for cutting lists.
- Solver flow mapping Kerf domain onto the engine geometry module.
- Metrics: utilization, waste, remnant consumption, dollars saved.
- Savings ledger with remnant lifecycle tracking.
- Test suite covering knapsack, cutting solver, domain model, import, and metrics.
- Benchmark harness.
