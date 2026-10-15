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
- Structured feasibility precheck: `precheck` returns `PrecheckFailure` items with machine-readable kinds (`no_bars`, `part_too_long`, `copies_exceed_supply`, `volume_exceeded`) and the numbers behind each; `solve` keeps the exact legacy reason strings.
- Engine public operations: solve, explain, replan, diagnose with model dispatch on metadata, Solution metrics for parts/bars/contexts, and precheck-driven diagnoses.
- Kerf domain model: Material, StockItem, Part, CuttingList, Cut, CutPiece, Plan, SavingsEvent with construction-time validation and string-valued enums; plain-string enum values coerce to the same member.
- `InfeasiblePlanError` carrying per-material engine reasons plus a flattened, deterministically ordered `.reasons` list.
- CSV cutting list importer (`load_cutting_list`) with a strict five-column header, row-numbered structural errors, and default `cl-<12 hex>` ids; sample fixture `tests/fixtures/sample_cutting_list.csv`.
- Pure proof metrics: utilization, waste length, new stock used, remnant length consumed, dollars saved, and `compute_all` keyed by metric name.
- Plan verification oracle (`verify_plan`): arithmetic reconciliation, no double-cut, unknown part/item, material match, and per-part demand; overproduction legal; `PlanVerificationError` aggregates all problems found in one pass.
- Kerf solver (`solve`): a cutting list, a stock room, and material prices become a PROPOSED plan with one named cut piece per part id, remnant-first bar pricing at zero cost, the five proof metrics, and every infeasible material collected before `InfeasiblePlanError` is raised.
- Kerf ledger (`record_plan_cut`): a PROPOSED or RELEASED plan becomes CUT, its consumed bars become CONSUMED, its offcuts return to the rack as AVAILABLE remnants carrying their parent plan, and a `SavingsEvent` is written with the remnant consumption and dollars avoided derived from the pure metrics; every guard runs before any mutation.
- Property tests over sixty seeded random trials: every solver plan must verify against the oracle, reconcile to the millimetre, and carry pure-recompute metrics; infeasibility must name every material with a part outgrowing every available bar; the ledger consumes each stock item exactly once and racks remnants with the parent plan.
- End-to-end round-trip test: the offcut of a cut plan returns to the rack, the next cutting list is solved against that same rack and satisfied by the remnant with no data entry, the ledger records the dollars saved, and a scrapped remnant makes the next list fail with a named material.
- Benchmark suite: forty integer ESICUP Falkenauer 1D cutting stock instances (binpack2 + binpack3) run against an FFD-decreasing baseline through both the public engine API and the Kerf product API, plus two seeded scale workloads with time budgets; the driver prints a plain-text report, names every infeasible instance, and exits non-zero when the acceptance target (aggregate kerf waste at most 92% of FFD waste with every instance resolved) is missed.
- Root README: the measured benchmark report pasted unmodified, the acceptance targets with each half stated as met or missed, the column generation round cap finding on the 11 unresolved instances, and how to run the tests and the benchmarks.

### Fixed
- The integer master runs the deterministic fill before declaring a restricted master infeasible, so a plan column generation never priced is still found instead of a false infeasible.
- Column generation no longer stalls on the ESICUP instances: the first-fit decreasing plan seeds the column pool so the master LP is feasible from round one, each round prices at Wentges-smoothed duals between the current dual and the best-bound stability center, and the loop stops once the Lagrangian/Farley lower bound and the master value round up to the same integer target. All 40 Falkenauer instances resolve well inside the 100-round cap (11 used to hit it and were reported infeasible).
- A solve never returns worse than the first-fit decreasing plan: when the integer master ends worse, with no incumbent, or infeasible over the priced columns, the greedy plan is returned, and a run cut short at the round cap still goes to the integer stage instead of being reported infeasible.
