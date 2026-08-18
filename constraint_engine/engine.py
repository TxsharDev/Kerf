"""The four public operations every product talks to.

``solve`` turns a :class:`Problem` into a :class:`Solution`, ``explain``
reads a :class:`Solution` and produces :class:`Reason` records for one
task, ``replan`` re-solves after events with a warm start from the
previous plan, and ``diagnose`` returns the minimal conflicting
constraints for an infeasible :class:`Problem`.

The engine dispatches on ``problem.metadata["model"]``. The 1D cutting
model reads its payload from ``metadata["cutting"]`` as a
:class:`CuttingProblem1D`; other models are added behind the same four
operations without a product ever noticing.
"""

from __future__ import annotations

from typing import Any, Sequence

from constraint_engine.geometry.master import MasterColumn
from constraint_engine.geometry.patterns import patterns_from_problem
from constraint_engine.geometry.solve import (
    precheck,
    solve as solve_cutting_1d,
)
from constraint_engine.geometry.types import CuttingProblem1D
from constraint_engine.types import Diagnosis, Problem, Reason, Solution

__all__ = ["solve", "explain", "replan", "diagnose"]

_CUTTING_1D = "cutting_1d"


def solve(problem: Problem, options: dict[str, Any] | None = None) -> Solution:
    """Solve a :class:`Problem` and return a :class:`Solution`.

    Dispatches on ``metadata["model"]``. For the 1D cutting model the
    payload is ``metadata["cutting"]`` (a :class:`CuttingProblem1D`).
    """
    options = options or {}
    model = problem.metadata.get("model")
    if model is None:
        raise ValueError("problem.metadata['model'] is required to dispatch a solve")
    if model != _CUTTING_1D:
        raise NotImplementedError(f"model {model!r} is not implemented yet")
    payload = problem.metadata.get("cutting")
    if not isinstance(payload, CuttingProblem1D):
        raise ValueError("model cutting_1d requires metadata['cutting'] to be a CuttingProblem1D")
    result = solve_cutting_1d(payload, time_limit_s=float(options.get("time_limit_s", 5.0)))
    return _solution_from_cutting(result, payload)


def replan(
    problem: Problem,
    previous_solution: Solution,
    events: Sequence[str],
) -> Solution:
    """Re-solve after a set of events, warming the column pool from the
    previous plan so the solver starts where it last left off.

    Events are carried onto the result under ``stats['replan']`` along
    with the count of warm-start columns recovered.
    """
    if not isinstance(previous_solution, Solution):
        raise ValueError("previous_solution must be a Solution")
    options: dict[str, Any] = {}
    model = problem.metadata.get("model")
    if model is None:
        raise ValueError("problem.metadata['model'] is required to dispatch a replan")
    if model != _CUTTING_1D:
        raise NotImplementedError(f"model {model!r} is not implemented yet")
    payload = problem.metadata.get("cutting")
    if not isinstance(payload, CuttingProblem1D):
        raise ValueError("model cutting_1d requires metadata['cutting'] to be a CuttingProblem1D")

    warm = _warm_columns(payload, previous_solution)
    options["initial_columns"] = warm
    solution = solve(problem, options)
    solution.stats["replan"] = {
        "events": list(events),
        "warm_start_columns": len(warm),
    }
    return solution


def explain(solution: Solution, task_id: str) -> list[Reason]:
    """Structured reasons for one task, read only from the Solution.

    A task id is a part index as a string, or a part length that
    uniquely names the part. The first reason always states the binding
    constraint (how demand was met); a resource-scarcity reason follows
    when the part's context ran out with no idle substitute; an
    alternative-worse reason follows when a pricier bar could also hold
    the part.
    """
    parts = solution.metrics.get("parts") or []
    part_index = _resolve_part_index(parts, task_id)
    part = parts[part_index]
    length_mm = part["length_mm"]
    demand = part["demand"]
    produced = part["produced"]
    feasible = bool(solution.feasibility.get("feasible", False))

    reasons: list[Reason] = [
        Reason(
            "binding_constraint",
            f"part-{part_index}",
            {
                "part_index": part_index,
                "demand": demand,
                "produced": produced,
                "note": _demand_note(feasible, demand, produced),
            },
        )
    ]

    contexts = solution.metrics.get("contexts") or []
    bars = solution.metrics.get("bars") or []
    if not contexts or not bars:
        return reasons

    chosen = _chosen_context(solution.assignments, bars, contexts, part_index)
    if chosen is None:
        return reasons

    if (
        chosen["bars_used"] > 0
        and chosen["bars_used"] == chosen["bars_available"]
        and not _idle_substitute(contexts, length_mm)
    ):
        reasons.append(
            Reason(
                "resource_scarcity",
                f"{chosen['length_mm']}mm bars",
                {
                    "length_mm": chosen["length_mm"],
                    "bars_used": chosen["bars_used"],
                    "bars_available": chosen["bars_available"],
                },
            )
        )

    alternatives = [
        ctx
        for ctx in contexts
        if ctx is not chosen
        and ctx["cost"] > chosen["cost"]
        and length_mm <= ctx["length_mm"]
    ]
    if alternatives:
        worst = max(alternatives, key=lambda ctx: ctx["cost"])
        reasons.append(
            Reason(
                "alternative_worse",
                f"part-{part_index}",
                {
                    "chosen_context_mm": chosen["length_mm"],
                    "alternative_context_mm": worst["length_mm"],
                    "delta_per_part": worst["cost"] - chosen["cost"],
                },
            )
        )
    return reasons


def diagnose(problem: Problem) -> list[Diagnosis]:
    """The minimal conflicting constraints for an infeasible problem.

    Cheap prechecks run first and each violation becomes its own
    diagnosis. Only when they all pass does the solver run; a feasible
    problem diagnoses to an empty list, and a problem the solver proves
    infeasible yields one diagnosis per reported reason.
    """
    model = problem.metadata.get("model")
    if model is None:
        raise ValueError("problem.metadata['model'] is required to dispatch a diagnose")
    if model != _CUTTING_1D:
        raise NotImplementedError(f"model {model!r} is not implemented yet")
    payload = problem.metadata.get("cutting")
    if not isinstance(payload, CuttingProblem1D):
        raise ValueError("model cutting_1d requires metadata['cutting'] to be a CuttingProblem1D")

    failures = precheck(payload)
    if failures:
        return [_diagnosis_for_failure(failure) for failure in failures]

    result = solve_cutting_1d(payload)
    if result.feasible:
        return []
    return [
        Diagnosis(constraint_ids=[], explanation=reason, suggested_removals=[])
        for reason in result.infeasible_reasons
    ]


# ---------------------------------------------------------------------------
# 1D cutting payload -> core Solution
# ---------------------------------------------------------------------------


def _solution_from_cutting(result, payload: CuttingProblem1D) -> Solution:
    """Map a :class:`CuttingResult1D` onto the core :class:`Solution`."""
    bar_meta = _bar_meta(payload)
    produced = _parts_produced(payload, result.assignments)
    contexts = _context_metrics(payload, result.assignments)
    return Solution(
        assignments=[
            {
                "bar_id": a.bar_id,
                "pattern": list(a.pattern),
                "offcut_mm": a.offcut_mm,
            }
            for a in result.assignments
        ],
        objective={
            "value": result.ip_value,
            "lp_value": result.lp_value,
            "gap": result.gap,
        },
        explanations=[],
        feasibility={
            "feasible": result.feasible,
            "reasons": list(result.infeasible_reasons),
        },
        metrics={
            **result.stats,
            "parts": _part_metrics(payload, produced),
            "bars": [bar_meta[a.bar_id] for a in result.assignments],
            "contexts": contexts,
        },
        stats={
            "model": _CUTTING_1D,
            "solve_time_s": result.solve_time_s,
            "ip_status": result.stats.get("ip_status", ""),
        },
    )


def _bar_meta(payload: CuttingProblem1D) -> dict[str, dict[str, Any]]:
    """bar_id -> {bar_id, length_mm, cost, origin} for every bar."""
    meta: dict[str, dict[str, Any]] = {}
    for index, bar in enumerate(payload.bars):
        bar_id = bar.id or f"bar_{index}"
        meta[bar_id] = {
            "bar_id": bar_id,
            "length_mm": bar.length_mm,
            "cost": bar.cost,
            "origin": bar.origin,
        }
    return meta


def _parts_produced(payload: CuttingProblem1D, assignments) -> list[int]:
    produced = [0] * len(payload.part_lengths_mm)
    for a in assignments:
        for i, count in enumerate(a.pattern):
            produced[i] += count
    return produced


def _part_metrics(payload: CuttingProblem1D, produced: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "length_mm": payload.part_lengths_mm[i],
            "demand": payload.demands[i],
            "produced": produced[i],
        }
        for i in range(len(payload.part_lengths_mm))
    ]


def _context_metrics(payload: CuttingProblem1D, assignments) -> list[dict[str, Any]]:
    contexts = patterns_from_problem(payload)
    used: dict[int, int] = {}
    bar_meta = _bar_meta(payload)
    for a in assignments:
        meta = bar_meta[a.bar_id]
        for index, ctx in enumerate(contexts):
            if ctx.length_mm == meta["length_mm"] and ctx.cost == meta["cost"]:
                used[index] = used.get(index, 0) + 1
                break
    return [
        {
            "length_mm": ctx.length_mm,
            "cost": ctx.cost,
            "bars_available": ctx.bar_count,
            "bars_used": used.get(index, 0),
        }
        for index, ctx in enumerate(contexts)
    ]


# ---------------------------------------------------------------------------
# explain helpers
# ---------------------------------------------------------------------------


def _resolve_part_index(parts: list[dict[str, Any]], task_id: str) -> int:
    """A task id is a part index first, else a part length, else an error."""
    try:
        index = int(task_id)
        if 0 <= index < len(parts):
            return index
    except (TypeError, ValueError):
        pass
    for part in parts:
        if str(part["length_mm"]) == str(task_id):
            return part["index"]
    raise ValueError(f"no part matches task id {task_id!r}")


def _demand_note(feasible: bool, demand: int, produced: int) -> str:
    if not feasible:
        return "demand unmet (problem infeasible)"
    if demand == 0:
        return "no demand"
    if produced > demand:
        return "overproduced; excess pieces return as remnants"
    return "demand exactly met"


def _chosen_context(assignments, bars, contexts, part_index):
    """The cheapest used context that actually cuts the part."""
    bar_by_id = {b["bar_id"]: b for b in bars}
    chosen = None
    for a in assignments:
        if part_index >= len(a["pattern"]) or a["pattern"][part_index] == 0:
            continue
        meta = bar_by_id.get(a["bar_id"])
        if meta is None:
            continue
        ctx = _context_by_key(contexts, meta["length_mm"], meta["cost"])
        if ctx is not None and (chosen is None or ctx["cost"] < chosen["cost"]):
            chosen = ctx
    return chosen


def _context_by_key(contexts, length_mm: int, cost: float):
    for ctx in contexts:
        if ctx["length_mm"] == length_mm and ctx["cost"] == cost:
            return ctx
    return None


def _idle_substitute(contexts, length_mm: int) -> bool:
    """Whether some unused bar in any context could also hold the part."""
    return any(
        ctx["bars_used"] < ctx["bars_available"] and length_mm <= ctx["length_mm"]
        for ctx in contexts
    )


# ---------------------------------------------------------------------------
# replan helpers
# ---------------------------------------------------------------------------


def _warm_columns(payload: CuttingProblem1D, previous: Solution) -> list[MasterColumn]:
    """Recover warm-start columns from a previous plan's assignments."""
    contexts = patterns_from_problem(payload)
    context_of: dict[str, int] = {}
    bar_meta = _bar_meta(payload)
    for index, ctx in enumerate(contexts):
        for bar_id in ctx.bar_ids:
            if bar_id in bar_meta:
                context_of[bar_id] = index

    columns: list[MasterColumn] = []
    for a in previous.assignments:
        bar_id = a["bar_id"]
        pattern = tuple(a["pattern"])
        if bar_id not in context_of or len(pattern) != len(payload.part_lengths_mm):
            continue
        columns.append(
            MasterColumn(context_index=context_of[bar_id], pattern=pattern)
        )
    return columns


# ---------------------------------------------------------------------------
# diagnose helpers
# ---------------------------------------------------------------------------


def _diagnosis_for_failure(failure) -> Diagnosis:
    if failure.kind == "no_bars":
        return Diagnosis(
            constraint_ids=["bars"],
            explanation="the problem has demand but no bars are available",
            suggested_removals=[],
        )
    if failure.kind in ("part_too_long", "copies_exceed_supply"):
        part = f"part-{failure.detail['part_index']}"
        return Diagnosis(
            constraint_ids=[part],
            explanation=_failure_explanation(failure),
            suggested_removals=[part],
        )
    if failure.kind == "volume_exceeded":
        return Diagnosis(
            constraint_ids=["volume"],
            explanation=_failure_explanation(failure),
            suggested_removals=[],
        )
    raise ValueError(f"unknown precheck failure kind: {failure.kind!r}")


def _failure_explanation(failure) -> str:
    if failure.kind == "part_too_long":
        d = failure.detail
        return (
            f"part {d['part_index']} (length {d['length_mm']}mm) is longer "
            f"than the longest bar ({d['longest_bar_mm']}mm)"
        )
    if failure.kind == "copies_exceed_supply":
        d = failure.detail
        return (
            f"part {d['part_index']} (length {d['length_mm']}mm): at most "
            f"{d['max_copies']} copies fit on the bars on hand, demand is "
            f"{d['demand']}"
        )
    if failure.kind == "volume_exceeded":
        d = failure.detail
        return (
            f"total part length ({d['part_volume_mm']}mm) exceeds the total "
            f"bar length ({d['bar_volume_mm']}mm)"
        )
    return failure.kind
