"""End-to-end orchestrator for 1D cutting stock problems.

``solve`` takes a :class:`CuttingProblem1D` and returns a
:class:`CuttingResult1D` with per-bar assignments, the LP and integer
objective values, and diagnostics. The pipeline is:

1. Prechecks: necessary conditions for feasibility. Any failure
   returns an infeasible result with a human-readable reason for each
   class, so the caller knows exactly what is wrong.
2. A first-fit decreasing plan over the bar contexts. Its patterns
   seed column generation, so the master LP is feasible from the first
   round, and the plan itself is the floor the result must beat.
3. Column generation over the bar contexts, seeded with single-type
   patterns and the greedy patterns (and optionally warmed with
   patterns from a previous plan).
4. The integer master problem over the generated columns, with the LP
   solution seeding the fallback fill. If the integer solve ends worse
   than the greedy plan, or with no plan, the greedy plan is returned.
5. Assembly: context-level column counts are expanded to physical bar
   assignments in the context's bar order (oldest remnant first).

The model is a cover: the plan may overproduce, and overproduced
pieces are simply longer remnants in the ledger.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from constraint_engine.geometry.greedy import first_fit_decreasing
from constraint_engine.geometry.integer import (
    IP_HEURISTIC,
    IP_INFEASIBLE,
    IP_NO_INCUMBENT,
    IntegerSolution,
    solve_master_ip,
)
from constraint_engine.geometry.master import MasterColumn
from constraint_engine.geometry.patterns import (
    Context,
    pattern_offcut,
    patterns_from_problem,
)
from constraint_engine.geometry.pricing import run_column_generation
from constraint_engine.geometry.types import (
    ORIGIN_REMNANT,
    BarAssignment,
    BarSpec,
    CuttingProblem1D,
    CuttingResult1D,
)

__all__ = ["PrecheckFailure", "precheck", "solve"]


@dataclass(frozen=True)
class PrecheckFailure:
    """A machine-readable feasibility failure.

    ``kind`` is a stable identifier (``no_bars``, ``part_too_long``,
    ``copies_exceed_supply``, ``volume_exceeded``) and ``detail``
    carries the numbers that produced the failure, so callers can
    branch on the kind instead of parsing prose.
    """

    kind: str
    detail: dict = field(default_factory=dict)


def solve(
    problem: CuttingProblem1D,
    time_limit_s: float = 5.0,
    initial_columns: Sequence[MasterColumn] = (),
) -> CuttingResult1D:
    """Solve a 1D cutting stock problem and return the plan with diagnostics.

    ``initial_columns`` are warm-start patterns from a previous plan;
    they enter the column pool behind the seeds, validated and deduped
    by the generator.
    """
    start = time.perf_counter()
    contexts = patterns_from_problem(problem)
    bar_by_id = {
        bar.id or f"bar_{index}": bar for index, bar in enumerate(problem.bars)
    }

    failures = precheck(problem)
    if failures:
        return _infeasible(
            problem,
            tuple(_failure_message(failure) for failure in failures),
            start,
        )

    if not any(demand > 0 for demand in problem.demands):
        return _zero_demand(problem, start)

    greedy = first_fit_decreasing(problem, contexts)
    seeds = tuple(initial_columns) + (greedy.columns if greedy else ())
    cg = run_column_generation(problem, contexts, initial_columns=seeds)
    if cg.infeasible:
        return _infeasible(
            problem,
            ("column generation proved the master problem infeasible",),
            start,
        )

    lp_x = cg.solution.x if cg.solution is not None else None
    ip = solve_master_ip(
        cg.columns, problem, contexts, lp_x=lp_x, time_limit_s=time_limit_s
    )
    if greedy is not None and (
        ip.status in (IP_INFEASIBLE, IP_NO_INCUMBENT) or ip.value > greedy.value
    ):
        ip = IntegerSolution(
            status=IP_HEURISTIC,
            columns=greedy.columns,
            x=greedy.counts,
            value=greedy.value,
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


def precheck(problem: CuttingProblem1D) -> tuple[PrecheckFailure, ...]:
    """Necessary conditions for feasibility; one failure per violated class.

    These are cheap, explainable, and cover every realistic failure.
    When they all pass, the column-generation and integer stages make
    the final call.
    """
    if sum(problem.demands) > 0 and len(problem.bars) == 0:
        return (PrecheckFailure("no_bars"),)
    if len(problem.bars) == 0:
        return ()

    failures: list[PrecheckFailure] = []
    longest = max(bar.length_mm for bar in problem.bars)
    for i, (length_mm, demand) in enumerate(
        zip(problem.part_lengths_mm, problem.demands)
    ):
        if demand > 0 and length_mm > longest:
            failures.append(
                PrecheckFailure(
                    "part_too_long",
                    {
                        "part_index": i,
                        "length_mm": length_mm,
                        "longest_bar_mm": longest,
                    },
                )
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
            failures.append(
                PrecheckFailure(
                    "copies_exceed_supply",
                    {
                        "part_index": i,
                        "length_mm": length_mm,
                        "max_copies": max_copies,
                        "demand": demand,
                    },
                )
            )

    part_volume = sum(
        length_mm * demand
        for length_mm, demand in zip(problem.part_lengths_mm, problem.demands)
    )
    bar_volume = sum(bar.length_mm for bar in problem.bars)
    if part_volume > bar_volume:
        failures.append(
            PrecheckFailure(
                "volume_exceeded",
                {"part_volume_mm": part_volume, "bar_volume_mm": bar_volume},
            )
        )
    return tuple(failures)


def _failure_message(failure: PrecheckFailure) -> str:
    """Render a :class:`PrecheckFailure` as the user-facing reason string."""
    if failure.kind == "no_bars":
        return "the problem has demand but no bars are available"
    if failure.kind == "part_too_long":
        return (
            f"part {failure.detail['part_index']} (length "
            f"{failure.detail['length_mm']}mm) is longer than the "
            f"longest bar ({failure.detail['longest_bar_mm']}mm)"
        )
    if failure.kind == "copies_exceed_supply":
        return (
            f"part {failure.detail['part_index']} (length "
            f"{failure.detail['length_mm']}mm): at most "
            f"{failure.detail['max_copies']} copies fit on the bars "
            f"on hand, demand is {failure.detail['demand']}"
        )
    if failure.kind == "volume_exceeded":
        return (
            f"total part length ({failure.detail['part_volume_mm']}mm) "
            f"exceeds the total bar length "
            f"({failure.detail['bar_volume_mm']}mm)"
        )
    raise ValueError(f"unknown precheck failure kind: {failure.kind!r}")


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
