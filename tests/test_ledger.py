"""Value tests for the Kerf ledger.

The ledger is the record of the cut: a PROPOSED or RELEASED plan is
marked CUT, the stock it consumed is marked CONSUMED, its offcuts go
back on the rack as AVAILABLE remnants, and a SavingsEvent carries the
proof numbers. These tests assert the state the shop sees afterwards
and that a plan that is not cuttable raises without touching the plan
or the rack.
"""

from datetime import datetime

import pytest

from kerf.ledger import record_plan_cut
from kerf.models import (
    Cut,
    CutPiece,
    Material,
    Origin,
    Plan,
    PlanStatus,
    StockItem,
    StockStatus,
)


def make_material(material_id="mat-1", unit_cost=0.01):
    return Material(id=material_id, name="test stock", unit_cost=unit_cost)


def make_bar(
    item_id,
    material_id,
    length,
    origin=Origin.PURCHASED,
    parent_plan_id=None,
    created_at=None,
):
    return StockItem(
        id=item_id,
        material_id=material_id,
        length=length,
        origin=origin,
        parent_plan_id=parent_plan_id,
        created_at=created_at,
    )


def make_cut(stock_item_id, pieces, kerf_total=0, offcut_length=0):
    return Cut(
        stock_item_id=stock_item_id,
        pieces=[CutPiece(part_id=part_id, length=length) for part_id, length in pieces],
        kerf_total=kerf_total,
        offcut_length=offcut_length,
    )


def make_plan(plan_id, cuts=(), remnants=(), status=PlanStatus.PROPOSED):
    return Plan(
        id=plan_id,
        cutting_list_id="cl-ledger",
        assignments=list(cuts),
        remnants_created=list(remnants),
        status=status,
    )


def test_cutting_a_plan_consumes_stock_and_racks_remnants():
    material = make_material()
    now = datetime.now()
    inventory = {
        "bar-a": make_bar("bar-a", "mat-1", 6000),
        "rm-1": make_bar(
            "rm-1", "mat-1", 1000, origin=Origin.REMNANT, created_at=now
        ),
    }
    plan = make_plan(
        "plan-1",
        [
            make_cut(
                "bar-a",
                [("p-1", 2500), ("p-2", 2500)],
                kerf_total=3,
                offcut_length=997,
            ),
            make_cut("rm-1", [("p-3", 900)], offcut_length=100),
        ],
        remnants=[
            make_bar(
                "rm-plan-1-01",
                "mat-1",
                997,
                origin=Origin.REMNANT,
                parent_plan_id="plan-1",
                created_at=now,
            ),
        ],
    )

    event = record_plan_cut(plan, inventory, {"mat-1": material})

    assert event.plan_id == "plan-1"
    assert isinstance(event.occurred_at, datetime)
    assert event.remnant_length_consumed == 900
    assert event.dollars_avoided == pytest.approx(9.0)
    assert plan.status is PlanStatus.CUT
    assert inventory["bar-a"].status is StockStatus.CONSUMED
    assert inventory["rm-1"].status is StockStatus.CONSUMED
    remnant = inventory["rm-plan-1-01"]
    assert remnant.status is StockStatus.AVAILABLE
    assert remnant.origin is Origin.REMNANT
    assert remnant.parent_plan_id == "plan-1"
    assert remnant.length == 997


def test_released_plan_can_be_cut():
    material = make_material()
    inventory = {"bar-b": make_bar("bar-b", "mat-1", 1000)}
    plan = make_plan(
        "plan-2",
        [make_cut("bar-b", [("p-1", 800)], offcut_length=200)],
        status=PlanStatus.RELEASED,
    )

    event = record_plan_cut(plan, inventory, {"mat-1": material})

    assert plan.status is PlanStatus.CUT
    assert inventory["bar-b"].status is StockStatus.CONSUMED
    assert event.remnant_length_consumed == 0
    assert event.dollars_avoided == 0.0


def test_cut_plan_is_rejected():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    plan = make_plan(
        "plan-3",
        [make_cut("bar-a", [("p-1", 1000)], offcut_length=5000)],
        status=PlanStatus.CUT,
    )

    with pytest.raises(ValueError, match="can be cut"):
        record_plan_cut(plan, inventory, {"mat-1": material})

    assert plan.status is PlanStatus.CUT
    assert inventory["bar-a"].status is StockStatus.AVAILABLE


def test_abandoned_plan_is_rejected():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    plan = make_plan(
        "plan-4",
        [make_cut("bar-a", [("p-1", 1000)], offcut_length=5000)],
        status=PlanStatus.ABANDONED,
    )

    with pytest.raises(ValueError, match="can be cut"):
        record_plan_cut(plan, inventory, {"mat-1": material})

    assert plan.status is PlanStatus.ABANDONED
    assert inventory["bar-a"].status is StockStatus.AVAILABLE


def test_unknown_stock_item_is_rejected_and_state_untouched():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    plan = make_plan(
        "plan-5", [make_cut("bar-missing", [("p-1", 1000)], offcut_length=0)]
    )

    with pytest.raises(ValueError, match="bar-missing"):
        record_plan_cut(plan, inventory, {"mat-1": material})

    assert plan.status is PlanStatus.PROPOSED
    assert inventory["bar-a"].status is StockStatus.AVAILABLE
    assert "bar-missing" not in inventory


def test_unavailable_stock_is_rejected():
    material = make_material()
    for status in (StockStatus.RESERVED, StockStatus.CONSUMED, StockStatus.SCRAPPED):
        inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
        inventory["bar-a"].status = status
        plan = make_plan(
            "plan-6", [make_cut("bar-a", [("p-1", 1000)], offcut_length=5000)]
        )

        with pytest.raises(ValueError, match="bar-a"):
            record_plan_cut(plan, inventory, {"mat-1": material})

        assert plan.status is PlanStatus.PROPOSED
        assert inventory["bar-a"].status is status


def test_a_bar_cannot_be_cut_by_two_plans():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    first = make_plan(
        "plan-7a", [make_cut("bar-a", [("p-1", 3000)], offcut_length=3000)]
    )
    second = make_plan(
        "plan-7b", [make_cut("bar-a", [("p-2", 1000)], offcut_length=5000)]
    )

    record_plan_cut(first, inventory, {"mat-1": material})

    with pytest.raises(ValueError, match="bar-a"):
        record_plan_cut(second, inventory, {"mat-1": material})

    assert first.status is PlanStatus.CUT
    assert second.status is PlanStatus.PROPOSED
    assert inventory["bar-a"].status is StockStatus.CONSUMED


def test_duplicate_bar_in_one_plan_is_rejected():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    plan = make_plan(
        "plan-8",
        [
            make_cut("bar-a", [("p-1", 2500)], offcut_length=3500),
            make_cut("bar-a", [("p-2", 1000)], offcut_length=0),
        ],
    )

    with pytest.raises(ValueError, match="bar-a"):
        record_plan_cut(plan, inventory, {"mat-1": material})

    assert plan.status is PlanStatus.PROPOSED
    assert inventory["bar-a"].status is StockStatus.AVAILABLE


def test_zero_cut_plan_is_recorded_with_zero_savings():
    material = make_material()
    inventory = {"bar-a": make_bar("bar-a", "mat-1", 6000)}
    plan = make_plan("plan-9")

    event = record_plan_cut(plan, inventory, {"mat-1": material})

    assert plan.status is PlanStatus.CUT
    assert inventory["bar-a"].status is StockStatus.AVAILABLE
    assert event.plan_id == "plan-9"
    assert event.remnant_length_consumed == 0
    assert event.dollars_avoided == 0.0
