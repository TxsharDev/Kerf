"""Kerf solver: a cutting list and a stock room become a verified plan.

The solver speaks two languages. The outside language is the Kerf
domain: materials, stock items, cutting lists, plans, remnants. The
inside language is the constraint engine: one 1D cutting problem per
material, solved by column generation and an integer master problem.

For every material with demand the solver:

1. gathers the unique part lengths, longest first, with the summed
   demand of every part at that length,
2. maps every available stock item of the material to an engine bar,
   purchased stock priced at the material's per-mm cost and remnants
   at zero cost with their age on the rack,
3. hands the problem to the engine and waits for a feasible answer,
4. expands the engine's bar patterns back into Kerf cuts, one named
   piece per part id so the plan can be verified and audited piece by
   piece.

Demand the engine cannot cover is collected for every material before
:class:`InfeasiblePlanError` is raised, so the report names the whole
problem instead of its first symptom.

The solver never mutates the stock room: cutting a plan is the
ledger's job. Offcuts and overproduced pieces are proposed remnants on
the plan; the ledger puts them on the rack when the plan is cut.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Mapping

from constraint_engine.engine import solve as engine_solve
from constraint_engine.geometry.types import BarSpec, CuttingProblem1D
from constraint_engine.types import Problem

from .errors import InfeasiblePlanError
from .metrics import compute_all
from .models import (
    Cut,
    CutPiece,
    CuttingList,
    Material,
    Origin,
    Part,
    Plan,
    PlanStatus,
    StockItem,
    StockStatus,
)
from .verify import verify_plan

__all__ = ["solve"]


def _new_plan_id() -> str:
    return f"plan-{uuid.uuid4().hex[:12]}"


def _bars_for_material(
    inventory: Mapping[str, StockItem],
    material_id: str,
    material: Material,
    now: datetime,
) -> tuple[BarSpec, ...]:
    """Engine bars from every AVAILABLE stock item of this material.

    Purchased stock is priced at the material's per-mm unit cost.
    Remnants cost nothing and carry their age on the rack in days,
    which is what the engine uses to clear the oldest stock first.
    """
    bars: list[BarSpec] = []
    for item in inventory.values():
        if item.material_id != material_id:
            continue
        if item.status is not StockStatus.AVAILABLE:
            continue
        if item.origin is Origin.REMNANT:
            age_days = (now - item.created_at).days if item.created_at else 0
            bars.append(
                BarSpec(
                    length_mm=item.length,
                    cost=0.0,
                    origin="remnant",
                    id=item.id,
                    age_days=float(age_days),
                )
            )
        else:
            bars.append(
                BarSpec(
                    length_mm=item.length,
                    cost=item.length * material.unit_cost,
                    origin="new",
                    id=item.id,
                )
            )
    return tuple(bars)


def _material_problem(
    material_id: str,
    parts: list[Part],
    bars: tuple[BarSpec, ...],
    kerf_mm: int,
    min_offcut_mm: int,
) -> tuple[Problem, list[int], list[list[str]]]:
    """One engine problem per material.

    Part lengths collapse to the unique lengths, longest first, with
    the summed demand of every part at that length. The pools are the
    flattened part ids per length (part id order, repeated once per
    quantity); expansion walks the pools in order, so a bar pattern of
    counts turns back into named parts.
    """
    by_length: dict[int, list[Part]] = {}
    for part in sorted(parts, key=lambda p: p.id):
        by_length.setdefault(part.length, []).append(part)
    lengths = sorted(by_length, reverse=True)
    demands = [sum(p.quantity for p in by_length[length]) for length in lengths]
    pools = [
        [p.id for p in by_length[length] for _ in range(p.quantity)]
        for length in lengths
    ]
    problem = Problem(
        metadata={
            "model": "cutting_1d",
            "cutting": CuttingProblem1D(
                part_lengths_mm=lengths,
                demands=demands,
                bars=bars,
                kerf_mm=kerf_mm,
                min_offcut_mm=min_offcut_mm,
                name=material_id,
            ),
        }
    )
    return problem, lengths, pools


def _cuts_for_material(
    assignments: list[dict],
    lengths: list[int],
    pools: list[list[str]],
    kerf_mm: int,
) -> list[Cut]:
    """Expand the engine's bar patterns into Kerf cuts.

    Pieces walk the part pools in length order, longest first, so each
    cut reconciles piece by piece against the cutting list. A bar that
    holds n pieces consumes n-1 kerfs between them.
    """
    cursors = [0] * len(lengths)
    cuts: list[Cut] = []
    sequence = 0
    for assignment in assignments:
        pattern = list(assignment["pattern"])
        n = sum(pattern)
        if n == 0:
            continue
        sequence += 1
        pieces: list[CutPiece] = []
        for length_index, count in enumerate(pattern):
            pool = pools[length_index]
            for _ in range(count):
                position = min(cursors[length_index], len(pool) - 1)
                pieces.append(
                    CutPiece(part_id=pool[position], length=lengths[length_index])
                )
                cursors[length_index] += 1
        cuts.append(
            Cut(
                stock_item_id=assignment["bar_id"],
                pieces=pieces,
                sequence=sequence,
                kerf_total=(n - 1) * kerf_mm,
                offcut_length=int(assignment["offcut_mm"]),
            )
        )
    return cuts


def _remnants_for(
    plan_id: str,
    cuts: list[Cut],
    inventory: Mapping[str, StockItem],
    parts: list[Part],
    produced: Mapping[str, int],
    min_offcut_mm: int,
    now: datetime,
) -> list[StockItem]:
    """Everything that returns to the rack when the plan is cut.

    An offcut of at least ``min_offcut_mm`` becomes a remnant stock
    item; every overproduced piece becomes a remnant of its own
    length. The ledger moves these onto the rack when it records the
    cut.
    """
    remnants: list[StockItem] = []

    def add(length: int, material_id: str) -> None:
        remnants.append(
            StockItem(
                id=f"rm-{plan_id}-{len(remnants) + 1:02d}",
                material_id=material_id,
                length=length,
                origin=Origin.REMNANT,
                parent_plan_id=plan_id,
                created_at=now,
            )
        )

    for cut in cuts:
        if cut.offcut_length >= min_offcut_mm:
            add(cut.offcut_length, inventory[cut.stock_item_id].material_id)
    for part in parts:
        over = max(0, produced.get(part.id, 0) - part.quantity)
        for _ in range(over):
            add(part.length, part.material_id)
    return remnants


def solve(
    cutting_list: CuttingList,
    inventory: Mapping[str, StockItem],
    materials: Mapping[str, Material],
    *,
    kerf_mm: int = 3,
    min_offcut_mm: int = 50,
    time_limit_s: float = 5.0,
) -> Plan:
    """Turn a cutting list and a stock room into a proposed plan.

    Every material on the list is solved independently against its
    available stock. The cuts, proposed remnants, stats, and five proof
    metrics are assembled into a plan that must pass the verification
    oracle before it is returned. Materials the engine cannot cover are
    reported together in one :class:`InfeasiblePlanError`.
    """
    plan = Plan(
        id=_new_plan_id(),
        cutting_list_id=cutting_list.id,
        status=PlanStatus.PROPOSED,
        created_at=datetime.now(),
    )

    parts = list(cutting_list.parts)
    if not parts:
        plan.metrics = compute_all([], inventory, materials)
        verify_plan(plan, cutting_list, inventory.values())
        return plan

    now = datetime.now()
    by_material: dict[str, list[Part]] = {}
    for part in parts:
        by_material.setdefault(part.material_id, []).append(part)

    failures: dict[str, list[str]] = {}
    stats: dict[str, dict] = {}
    all_cuts: list[Cut] = []

    for material_id in sorted(by_material):
        material = materials[material_id]
        material_parts = by_material[material_id]
        bars = _bars_for_material(inventory, material_id, material, now)
        problem, lengths, pools = _material_problem(
            material_id, material_parts, bars, kerf_mm, min_offcut_mm
        )
        result = engine_solve(problem, {"time_limit_s": time_limit_s})
        if not result.feasibility["feasible"]:
            failures[material_id] = list(result.feasibility["reasons"])
            continue
        all_cuts.extend(
            _cuts_for_material(result.assignments, lengths, pools, kerf_mm)
        )
        stats[material_id] = dict(result.stats)

    if failures:
        raise InfeasiblePlanError(failures)

    produced: dict[str, int] = {}
    for cut in all_cuts:
        for piece in cut.pieces:
            produced[piece.part_id] = produced.get(piece.part_id, 0) + 1

    plan.assignments = all_cuts
    plan.stats = stats
    plan.remnants_created = _remnants_for(
        plan.id, all_cuts, inventory, parts, produced, min_offcut_mm, now
    )
    plan.metrics = compute_all(all_cuts, inventory, materials)
    verify_plan(plan, cutting_list, inventory.values())
    return plan
