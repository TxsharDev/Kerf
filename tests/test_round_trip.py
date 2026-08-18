"""End-to-end round trip: one shop day, two cutting lists, zero data entry.

The first list is cut from a purchased bar and its offcut goes back on
the rack. The second list is solved against that same rack and must be
satisfied by the remnant the first plan created - found by the solver
itself, verified by the oracle, and recorded by the ledger with the
dollars the shop did not have to spend. A second test scraps the
remnant off the rack and checks the next list then fails honestly.
"""

import pytest

from kerf.errors import InfeasiblePlanError
from kerf.ledger import record_plan_cut
from kerf.models import (
    CuttingList,
    Material,
    Origin,
    Part,
    StockItem,
    StockStatus,
)
from kerf.solver import solve
from kerf.verify import verify_plan


def make_material():
    return Material(id="mat-1", name="test stock", unit_cost=0.01)


def make_bar(item_id, length):
    return StockItem(id=item_id, material_id="mat-1", length=length)


def make_list(list_id, part_id, length, quantity):
    return CuttingList(
        id=list_id,
        parts=[Part(id=part_id, material_id="mat-1", length=length, quantity=quantity)],
    )


def rack_remnants(inventory):
    return [item for item in inventory.values() if item.origin is Origin.REMNANT]


def test_offcuts_from_one_plan_feed_the_next_list():
    material = make_material()
    materials = {"mat-1": material}
    inventory = {"bar-1": make_bar("bar-1", 6000)}

    first_list = make_list("cl-round-1", "p-first", 2500, 2)
    plan1 = solve(first_list, inventory, materials)
    verify_plan(plan1, first_list, inventory.values())

    event1 = record_plan_cut(plan1, inventory, materials)
    assert event1.dollars_avoided == 0.0
    assert inventory["bar-1"].status is StockStatus.CONSUMED
    remnants = rack_remnants(inventory)
    assert len(remnants) == 1
    remnant = remnants[0]
    assert remnant.status is StockStatus.AVAILABLE
    assert remnant.parent_plan_id == plan1.id
    assert remnant.length >= 900

    second_list = make_list("cl-round-2", "p-second", 900, 1)
    plan2 = solve(second_list, inventory, materials)
    verify_plan(plan2, second_list, inventory.values())

    assert {cut.stock_item_id for cut in plan2.assignments} == {remnant.id}
    assert plan2.metrics["new_stock_used"] == 0
    assert plan2.metrics["remnant_length_consumed"] == 900
    assert plan2.metrics["dollars_saved"] == pytest.approx(9.0)

    event2 = record_plan_cut(plan2, inventory, materials)
    assert event2.plan_id == plan2.id
    assert event2.remnant_length_consumed == 900
    assert event2.dollars_avoided == pytest.approx(9.0)
    assert inventory[remnant.id].status is StockStatus.CONSUMED


def test_scrapped_remnant_leaves_the_next_list_infeasible():
    material = make_material()
    materials = {"mat-1": material}
    inventory = {"bar-1": make_bar("bar-1", 6000)}

    first_list = make_list("cl-round-3", "p-first", 2500, 2)
    plan1 = solve(first_list, inventory, materials)
    record_plan_cut(plan1, inventory, materials)
    remnant = rack_remnants(inventory)[0]
    remnant.status = StockStatus.SCRAPPED

    second_list = make_list("cl-round-4", "p-second", 900, 1)
    with pytest.raises(InfeasiblePlanError) as exc_info:
        solve(second_list, inventory, materials)

    assert "mat-1" in exc_info.value.by_material
