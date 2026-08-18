"""End-to-end orchestrator for 1D cutting stock problems.

``solve`` takes a :class:`CuttingProblem1D` and returns a
:class:`CuttingResult1D` with per-bar assignments, the LP and integer
objective values, and diagnostics. The pipeline is:

1. Prechecks: necessary conditions for feasibility. Any failure
   returns an infeasible result with a human-readable reason for each
   class, so the caller knows exactly what is wrong.
2. Column generation over the bar contexts, seeded with single-type
   patterns so pricing keeps working even while the master LP is
   infeasible.
3. The integer master problem over the converged columns, with the LP
   solution seeding the fallback fill.
4. Assembly: context-level column counts are expanded to physical bar
   assignments in the context's bar order (oldest remnant first).

The model is a cover: the plan may overproduce, and overproduced
pieces are simply longer remnants in the ledger.
"""

from __future__ import annotations

import time
from typing import Sequence

from constraint_engine.geometry.integer import (
    IP_INFEASIBLE,
    IP_NO_INCUMBENT,
    IntegerSolution,
    solve_master_ip,
)
from constraint_engine.geometry.patterns import (
    Context,
    pattern_offcut,
    patterns_from_problem,
)
from constraint_engine.geometry.pricing import (
    STOP_MAX_ROUNDS,
    run_column_generation,
)
from constraint_engine.geometry.types import (
    ORIGIN_REMNANT,
    BarAssignment,
    BarSpec,
    CuttingProblem1D,
    CuttingResult1D,
)

__all__ = ["solve"]

_CG_MAX_ROUNDS = 100


def solve(
    problem: CuttingProblem1D, time_limit_s: float = 5.0
) -> CuttingResult1D:
    """Solve a 1D cutting stock problem and return the plan with diagnostics."""
    start = time.perf_counter()
    contexts = patterns_from_problem(problem)
    bar_by_id = {
        bar.id or f"bar_{index}": bar for index, bar in enumerate(problem.bars)
    }

    reasons = _precheck(problem)
    if reasons:
        return _infeasible(problem, reasons, start)

    if not any(demand > 0 for demand in problem.demands):
        return _zero_demand(problem, start)

    cg = run_column_generation(problem, contexts)
    if cg.infeasible:
        return _infeasible(
            problem,
            ("column generation proved the master problem infeasible",),
            start,
        )
    if cg.stop == STOP_MAX_ROUNDS and cg.solution is None:
        return _infeasible(
            problem,
            (
                f"column generation did not converge within "
                f"{_CG_MAX_ROUNDS} rounds"
            ),
            start,
        )

    lp_x = cg.solution.x if cg.solution is not None else None
    ip = solve_master_ip(
        cg.columns, problem, contexts, lp_x=lp_x, time_limit_s=time_limit_s
    )
    if ip.status == IP_INFEASIBLE:
        return _infeasible(
            problem,
            ("no integer plan covers the demand with the bars on hand",),
            start,
        )
    if ip.status == IP_NO_INCUMBENT:
        return _infeasible(
            problem,
            (
                "the integer solver and the fallback fill both failed "
                "to cover the demand"
            ),
            start,
        )

    assignments, parts_produced = _assemble(ip, problem, contexts)
    lp_value = cg.solution.value if cg.solution is not None else 0.0
    ip_value = ip.value
    gap = max(0.0, (ip_value - lp_value) / lp_value) if lp_value > 0 else 0.0
    waste_mm = sum(
        bar_by_id[assignment.bar_id].length_mm for assignment in assignments
    ) - parts_produced
    stats = {
        "cg_rounds": cg.rounds,
        "columns_kept": len(cg.columns),
        "bars_used": len(assignments),
        "bars_total": len(problem.bars),
        "waste_mm": waste_mm,
        "remnant_bars_used": _count_remnants(assignments, bar_by_id),
        "ip_status": ip.status,
    }
    return CuttingResult1D(
        feasible=True,
        assignments=tuple(assignments),
        lp_value=float(lp_value),
        ip_value=float(ip_value),
        gap=float(gap),
        solve_time_s=time.perf_counter() - start,
        stats=stats,
    )


def _precheck(problem: CuttingProblem1D) -> tuple[str, ...]:
    """Necessary conditions for feasibility; one reason per violated class.

    These are cheap, explainable, and cover every realistic failure.
    When they all pass, the column-generation and integer stages make
    the final call.
    """
    if sum(problem.demands) > 0 and len(problem.bars) == 0:
        return ("the problem has demand but no bars are available",)

    reasons: list[str] = []
    longest = max(bar.length_mm for bar in problem.bars)
    for i, (length_mm, demand) in enumerate(
        zip(problem.part_lengths_mm, problem.demands)
    ):
        if demand > 0 and length_mm > longest:
            reasons.append(
                f"part {i} (length {length_mm}mm) is longer than the "
                f"longest bar ({longest}mm)"
            )

    for i, (length_mm, demand) in enumerate(
        zip(problem.part_lengths_mm, problem.demands)
    ):
        if demand <= 0:
            continue
        weight = length_mm + problem.kerf_mm
        max_copies = 0
        for bar in problem.bars:
            capacity = bar.length_mm + problem.kerf_mm
            if weight <= capacity:
                max_copies += capacity // weight
        if max_copies < demand:
            reasons.append(
                f"part {i} (length {length_mm}mm): at most {max_copies} "
                f"copies fit on the bars on hand, demand is {demand}"
            )

    part_volume = sum(
        length_mm * demand
        for length_mm, demand in zip(problem.part_lengths_mm, problem.demands)
    )
    bar_volume = sum(bar.length_mm for bar in problem.bars)
    if part_volume > bar_volume:
        reasons.append(
            f"total part length ({part_volume}mm) exceeds the total "
            f"bar length ({bar_volume}mm)"
        )
    return tuple(reasons)


def _assemble(
    ip: IntegerSolution,
    problem: CuttingProblem1D,
    contexts: Sequence[Context],
) -> tuple[list[BarAssignment], int]:
    """Expand context-level column counts to physical bar assignments.

    Within each context the bars are consumed in the context's bar
    order (oldest remnant first). ``ip.columns`` is the source of
    truth: the fallback fill may have extended it with a pattern
    column generation never priced, and ``ip.x`` aligns with it.
    """
    assignments: list[BarAssignment] = []
    parts_produced = 0
    for context_index, context in enumerate(contexts):
        bar_cursor = 0
        for column, count in zip(ip.columns, ip.x):
            if column.context_index != context_index or count == 0:
                continue
            for _ in range(count):
                if bar_cursor >= context.bar_count:
                    raise RuntimeError(
                        f"context of {context.length_mm}mm bars assigned "
                        f"more bars than it holds"
                    )
                bar_id = context.bar_ids[bar_cursor]
                bar_cursor += 1
                offcut = pattern_offcut(
                    column.pattern,
                    problem.part_lengths_mm,
                    problem.kerf_mm,
                    context.length_mm,
                )
                assignments.append(
                    BarAssignment(
                        bar_id=bar_id,
                        pattern=column.pattern,
                        offcut_mm=offcut,
                    )
                )
                for i, part_count in enumerate(column.pattern):
                    parts_produced += part_count * problem.part_lengths_mm[i]
    return assignments, parts_produced


def _count_remnants(
    assignments: Sequence[BarAssignment], bar_by_id: dict[str, BarSpec]
) -> int:
    return sum(
        1
        for assignment in assignments
        if bar_by_id[assignment.bar_id].origin == ORIGIN_REMNANT
    )


def _zero_demand(
    problem: CuttingProblem1D, start: float
) -> CuttingResult1D:
    """A problem with no demand is trivially feasible and empty."""
    return CuttingResult1D(
        feasible=True,
        assignments=(),
        lp_value=0.0,
        ip_value=0.0,
        gap=0.0,
        solve_time_s=time.perf_counter() - start,
        stats={
            "cg_rounds": 0,
            "columns_kept": 0,
            "bars_used": 0,
            "bars_total": len(problem.bars),
            "waste_mm": 0,
            "remnant_bars_used": 0,
            "ip_status": "optimal",
        },
    )


def _infeasible(
    problem: CuttingProblem1D, reasons: tuple[str, ...], start: float
) -> CuttingResult1D:
    """An infeasible result with all values zeroed and the reasons attached."""
    return CuttingResult1D(
        feasible=False,
        assignments=(),
        lp_value=0.0,
        ip_value=0.0,
        gap=0.0,
        solve_time_s=time.perf_counter() - start,
        stats={
            "cg_rounds": 0,
            "columns_kept": 0,
            "bars_used": 0,
            "bars_total": len(problem.bars),
            "waste_mm": 0,
            "remnant_bars_used": 0,
            "ip_status": IP_INFEASIBLE,
        },
        infeasible_reasons=reasons,
    )
