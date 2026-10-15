"""End-to-end value tests for the Kerf solver.

The solver is the product: a cutting list plus a stock room becomes a
proposed plan the shop can cut. These tests assert what the customer
can see -- demand met, bars reconciled to the millimetre, offcuts back
on the rack as remnants, infeasibility named per material, rack
remnants preferred over new purchase, unavailable stock ignored -- and
re-check every plan through the independent verification oracle.
"""

from collections import Counter
from datetime import datetime

import pytest

from kerf.errors import InfeasiblePlanError
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

METRIC_KEYS = (
    "utilization",
    "waste_length",
    "new_stock_used",
    "remnant_length_consumed",
    "dollars_saved",
)


def make_material(material_id="mat-1", unit_cost=0.01):
    return Material(id=material_id, name="test stock", unit_cost=unit_cost)


def make_bar(item_id, material_id, length, origin=Origin.PURCHASED, created_at=None):
    return StockItem(
        id=item_id,
        material_id=material_id,
        length=length,
        origin=origin,
        created_at=created_at,
    )


def make_list(list_id, parts):
    return CuttingList(id=list_id, parts=parts)


def produced_counts(plan):
    counts = Counter()
    for cut in plan.assignments:
        for piece in cut.pieces:
            counts[piece.part_id] += 1
    return counts


def test_feasible_list_returns_proposed_plan():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    parts = [
        Part(id="p-1", material_id="mat-1", length=2500, quantity=1),
        Part(id="p-2", material_id="mat-1", length=1500, quantity=1),
    ]
    cutting_list = make_list("cl-1", parts)

    plan = solve(cutting_list, inventory, {"mat-1": material})

    assert plan.status is PlanStatus.PROPOSED
    assert plan.id
    assert plan.cutting_list_id == cutting_list.id
    assert len(plan.assignments) >= 1
    verify_plan(plan, cutting_list, inventory.values())


def test_every_part_demand_is_met():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    parts = [
        Part(id="p-1", material_id="mat-1", length=1000, quantity=3),
        Part(id="p-2", material_id="mat-1", length=800, quantity=2),
    ]
    cutting_list = make_list("cl-2", parts)

    plan = solve(cutting_list, inventory, {"mat-1": material})

    counts = produced_counts(plan)
    for part in parts:
        assert counts[part.id] >= part.quantity
    verify_plan(plan, cutting_list, inventory.values())


def test_offcuts_and_overproduction_become_remnants():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    parts = [Part(id="p-1", material_id="mat-1", length=1000, quantity=1)]
    cutting_list = make_list("cl-3", parts)

    plan = solve(cutting_list, inventory, {"mat-1": material})

    # Any feasible pattern here either leaves a >= 50mm offcut or
    # overproduces, so the plan must put something back on the rack.
    assert plan.remnants_created

    expected = Counter()
    for cut in plan.assignments:
        if cut.offcut_length >= 50:
            expected[cut.offcut_length] += 1
    counts = produced_counts(plan)
    for part in parts:
        over = max(0, counts[part.id] - part.quantity)
        if over:
            expected[part.length] += over

    remnant_lengths = Counter(r.length for r in plan.remnants_created)
    for length, count in expected.items():
        assert remnant_lengths[length] >= count, (
            f"expected {count} remnant(s) of {length}mm, "
            f"have {dict(remnant_lengths)}"
        )
    for remnant in plan.remnants_created:
        assert remnant.origin is Origin.REMNANT
        assert remnant.parent_plan_id == plan.id
        assert remnant.material_id == "mat-1"


def test_part_longer_than_any_bar_is_infeasible():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    parts = [Part(id="p-1", material_id="mat-1", length=9000, quantity=1)]
    cutting_list = make_list("cl-4", parts)

    with pytest.raises(InfeasiblePlanError) as excinfo:
        solve(cutting_list, inventory, {"mat-1": material})

    assert "mat-1" in excinfo.value.by_material
    assert excinfo.value.by_material["mat-1"]


def test_infeasibility_reports_every_material():
    material_a = make_material("mat-a")
    material_b = make_material("mat-b")
    inventory = {
        "bar-a": make_bar("bar-a", "mat-a", 6000),
        "bar-b": make_bar("bar-b", "mat-b", 5000),
    }
    parts = [
        Part(id="p-a", material_id="mat-a", length=7000, quantity=1),
        Part(id="p-b", material_id="mat-b", length=6000, quantity=1),
    ]
    cutting_list = make_list("cl-5", parts)

    with pytest.raises(InfeasiblePlanError) as excinfo:
        solve(cutting_list, inventory, {"mat-a": material_a, "mat-b": material_b})

    # Both materials fail; the report must name both, not stop at first.
    assert set(excinfo.value.by_material) == {"mat-a", "mat-b"}
    assert excinfo.value.by_material["mat-a"]
    assert excinfo.value.by_material["mat-b"]


def test_empty_cutting_list_returns_empty_plan():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    cutting_list = make_list("cl-6", [])

    plan = solve(cutting_list, inventory, {"mat-1": material})

    assert plan.assignments == []
    assert plan.remnants_created == []
    assert plan.status is PlanStatus.PROPOSED
    verify_plan(plan, cutting_list, inventory.values())


def test_plan_carries_all_five_proof_metrics():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    parts = [
        Part(id="p-1", material_id="mat-1", length=2500, quantity=1),
        Part(id="p-2", material_id="mat-1", length=1500, quantity=1),
    ]
    cutting_list = make_list("cl-7", parts)

    plan = solve(cutting_list, inventory, {"mat-1": material})

    for key in METRIC_KEYS:
        assert key in plan.metrics
    assert isinstance(plan.metrics["utilization"], float)
    assert 0.0 <= plan.metrics["utilization"] <= 100.0


def test_rack_remnant_is_preferred_over_new_stock():
    material = make_material(unit_cost=0.01)
    now = datetime.now()
    inventory = {
        "bar-new": make_bar("bar-new", "mat-1", 6000),
        "rm-1": make_bar(
            "rm-1", "mat-1", 1000, origin=Origin.REMNANT, created_at=now
        ),
    }
    parts = [Part(id="p-1", material_id="mat-1", length=900, quantity=1)]
    cutting_list = make_list("cl-8", parts)

    plan = solve(cutting_list, inventory, {"mat-1": material})

    # The 900mm part fits the 1000mm remnant; the engine must prefer
    # the zero-cost bar over a fresh 6000mm purchase.
    assert {cut.stock_item_id for cut in plan.assignments} == {"rm-1"}
    assert plan.metrics["new_stock_used"] == 0
    assert plan.metrics["dollars_saved"] > 0
    verify_plan(plan, cutting_list, inventory.values())


def test_reserved_and_consumed_stock_are_not_used():
    material = make_material()
    reserved = make_bar("bar-r", "mat-1", 6000)
    reserved.status = StockStatus.RESERVED
    consumed = make_bar("bar-c", "mat-1", 6000)
    consumed.status = StockStatus.CONSUMED
    inventory = {
        "bar-r": reserved,
        "bar-c": consumed,
        "bar-open": make_bar("bar-open", "mat-1", 6000),
    }
    parts = [Part(id="p-1", material_id="mat-1", length=1000, quantity=1)]
    cutting_list = make_list("cl-9", parts)

    plan = solve(cutting_list, inventory, {"mat-1": material})

    assert {cut.stock_item_id for cut in plan.assignments} == {"bar-open"}
    verify_plan(plan, cutting_list, inventory.values())


def test_column_generation_cut_short_still_yields_a_verified_plan(monkeypatch):
    from constraint_engine.geometry import solve as engine_solve_mod

    capped = engine_solve_mod.run_column_generation

    def one_round(*args, **kwargs):
        return capped(*args, max_rounds=1, **kwargs)

    monkeypatch.setattr(engine_solve_mod, "run_column_generation", one_round)
    material = make_material()
    inventory = {
        f"bar-{i}": make_bar(f"bar-{i}", "mat-1", 6000) for i in range(6)
    }
    inventory["rm-1"] = make_bar(
        "rm-1", "mat-1", 2400, origin=Origin.REMNANT, created_at=datetime(2026, 1, 5)
    )
    parts = [
        Part(id=f"p-{i}", material_id="mat-1", length=length, quantity=quantity)
        for i, (length, quantity) in enumerate(
            [(2210, 3), (1730, 4), (1180, 5), (960, 4), (430, 6)]
        )
    ]
    cutting_list = make_list("cl-capped", parts)

    plan = solve(cutting_list, inventory, {"mat-1": material}, kerf_mm=3)

    counts = produced_counts(plan)
    for part in parts:
        assert counts[part.id] >= part.quantity
    assert "rm-1" in {cut.stock_item_id for cut in plan.assignments}
    verify_plan(plan, cutting_list, inventory.values())
