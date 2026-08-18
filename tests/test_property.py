"""Property tests for the Kerf solver and ledger.

Random instances are generated with a fixed seed, so this file is
deterministic: the same trials run on every machine. For each trial the
solver must end in one of two ways: a plan that survives the
independent verification oracle and reconciles to the millimetre, or
``InfeasiblePlanError`` with reasons that are structured and honest.
Where a part outgrows every available bar the precheck decides before
any heuristic runs, so the error must name that material. Every plan
that comes back also goes through the ledger: the stock is consumed
exactly once, the remnants land on the rack, the event carries the
same numbers as the plan's own metrics, and the cut cannot be recorded
twice.
"""

from copy import deepcopy
from random import Random

import pytest

from kerf.errors import InfeasiblePlanError
from kerf.ledger import record_plan_cut
from kerf.metrics import compute_all
from kerf.models import (
    CuttingList,
    Material,
    Origin,
    Part,
    PlanStatus,
    StockItem,
    StockStatus,
)
from kerf.solver import solve
from kerf.verify import verify_plan

SEED = 20260817
TRIALS = 60

MATERIAL_COSTS = (0.005, 0.01, 0.02, 0.05)
KERF_CHOICES = (0, 2, 3, 5)


def _generate_trials(rng):
    trials = []
    for t in range(TRIALS):
        materials = {}
        inventory = {}
        parts = []
        for m in range(rng.randint(1, 2)):
            material_id = f"mat-{m + 1}"
            materials[material_id] = Material(
                id=material_id,
                name=f"trial stock {m + 1}",
                unit_cost=rng.choice(MATERIAL_COSTS),
            )
            for b in range(rng.randint(0, 5)):
                bar = StockItem(
                    id=f"bar-{t}-{m}-{b}",
                    material_id=material_id,
                    length=rng.randint(2000, 8000),
                )
                inventory[bar.id] = bar
            for p in range(rng.randint(0, 6)):
                # Most parts fit typical stock; a fifth may not.
                if rng.random() < 0.8:
                    length = rng.randint(200, 3000)
                else:
                    length = rng.randint(200, 12000)
                parts.append(
                    Part(
                        id=f"part-{t}-{m}-{p}",
                        material_id=material_id,
                        length=length,
                        quantity=rng.randint(1, 3),
                    )
                )
        trials.append(
            {
                "cutting_list": CuttingList(id=f"cl-{t}", parts=parts),
                "inventory": inventory,
                "materials": materials,
                "kerf_mm": rng.choice(KERF_CHOICES),
            }
        )
    return trials


TRIALS_DATA = _generate_trials(Random(SEED))


def _too_long_materials(trial):
    """Materials with a part longer than every available bar: no plan
    can exist, and the precheck must say so without any heuristic."""
    longest: dict[str, int] = {}
    for item in trial["inventory"].values():
        if item.status is StockStatus.AVAILABLE:
            longest[item.material_id] = max(longest.get(item.material_id, 0), item.length)
    too_long = set()
    for part in trial["cutting_list"].parts:
        if part.length > longest.get(part.material_id, 0):
            too_long.add(part.material_id)
    return too_long


def test_solver_plan_or_well_named_infeasibility():
    for trial in TRIALS_DATA:
        cutting_list = trial["cutting_list"]
        inventory = trial["inventory"]
        materials = trial["materials"]
        part_lengths = {p.id: p.length for p in cutting_list.parts}
        try:
            plan = solve(
                cutting_list, inventory, materials, kerf_mm=trial["kerf_mm"]
            )
        except InfeasiblePlanError as exc:
            listed = {p.material_id for p in cutting_list.parts}
            assert set(exc.by_material) <= listed
            for reasons in exc.by_material.values():
                assert reasons
                assert all(isinstance(reason, str) and reason for reason in reasons)
            too_long = _too_long_materials(trial)
            assert too_long <= set(exc.by_material), (
                f"trial {cutting_list.id}: a part outgrows every bar for "
                f"{sorted(too_long)} but the error named "
                f"{sorted(exc.by_material)}"
            )
            continue

        # A plan came back: it must verify and reconcile to the
        # millimetre, and its metrics must be the pure recompute.
        verify_plan(plan, cutting_list, inventory.values())
        assert plan.metrics == compute_all(plan.assignments, inventory, materials)
        assert _too_long_materials(trial) == set(), (
            f"trial {cutting_list.id}: a plan exists although a part "
            "outgrows every available bar"
        )
        for cut in plan.assignments:
            piece_mm = sum(piece.length for piece in cut.pieces)
            assert (
                piece_mm + cut.kerf_total + cut.offcut_length
                == inventory[cut.stock_item_id].length
            )
            for piece in cut.pieces:
                assert part_lengths[piece.part_id] == piece.length


def test_ledger_round_trip_on_random_plans():
    for trial in TRIALS_DATA:
        cutting_list = trial["cutting_list"]
        inventory = trial["inventory"]
        materials = trial["materials"]
        try:
            plan = solve(
                cutting_list, inventory, materials, kerf_mm=trial["kerf_mm"]
            )
        except InfeasiblePlanError:
            continue

        rack = deepcopy(inventory)
        event = record_plan_cut(plan, rack, materials)

        assert plan.status is PlanStatus.CUT
        assert len({c.stock_item_id for c in plan.assignments}) == len(
            plan.assignments
        )
        for cut in plan.assignments:
            assert rack[cut.stock_item_id].status is StockStatus.CONSUMED
        consumed_ids = {cut.stock_item_id for cut in plan.assignments}
        for remnant in plan.remnants_created:
            assert rack[remnant.id] is remnant
            assert remnant.status is StockStatus.AVAILABLE
            assert remnant.origin is Origin.REMNANT
            assert remnant.parent_plan_id == plan.id
        assert not (
            {r.id for r in plan.remnants_created} & consumed_ids
        ), f"trial {cutting_list.id}: a remnant id collides with cut stock"
        assert event.remnant_length_consumed == plan.metrics["remnant_length_consumed"]
        assert event.dollars_avoided == plan.metrics["dollars_saved"]
        with pytest.raises(ValueError):
            record_plan_cut(plan, rack, materials)
