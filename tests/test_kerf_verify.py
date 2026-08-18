"""Plan verification: the never-uncuttable oracle.

A plan is valid when a shop can execute it exactly as written:

* every cut names a real stock item, and no item is cut twice;
* each cut's arithmetic reconciles the item's full length
  (pieces + kerf + offcut == stock length);
* every piece names a part from the cutting list;
* every piece's material matches the stock item's material;
* every part on the list is produced at least its demanded quantity.

Overproduction is legal (it becomes a remnant); underproduction is
not. All problems found in one pass are aggregated into a single
``PlanVerificationError``.
"""

from kerf.errors import PlanVerificationError
from kerf.models import (
    Cut,
    CutPiece,
    CuttingList,
    Part,
    Plan,
    StockItem,
)
from kerf.verify import verify_plan
import pytest


def _inventory():
    return [
        StockItem(id="b1", material_id="m1", length=6000),
        StockItem(id="r1", material_id="m1", length=2000, origin="remnant"),
    ]


def _cutting_list():
    return CuttingList(
        id="cl1",
        parts=[
            Part(id="p1", material_id="m1", length=2500, quantity=1),
            Part(id="p2", material_id="m1", length=1800, quantity=1),
            Part(id="p3", material_id="m1", length=950, quantity=1),
        ],
    )


def _valid_plan():
    # b1: 2500 + 1800 + 3 kerf + 1697 offcut = 6000
    c1 = Cut(
        stock_item_id="b1",
        pieces=[CutPiece("p1", 2500), CutPiece("p2", 1800)],
        sequence=1,
        kerf_total=3,
        offcut_length=1697,
    )
    # r1: 950 + 1050 offcut = 2000
    c2 = Cut(
        stock_item_id="r1",
        pieces=[CutPiece("p3", 950)],
        sequence=2,
        kerf_total=0,
        offcut_length=1050,
    )
    return Plan(id="plan-1", cutting_list_id="cl1", assignments=[c1, c2])


def test_valid_plan_passes():
    verify_plan(_valid_plan(), _cutting_list(), _inventory())


def test_arithmetic_mismatch_names_item():
    plan = _valid_plan()
    plan.assignments[0].offcut_length = 100
    with pytest.raises(PlanVerificationError, match="b1"):
        verify_plan(plan, _cutting_list(), _inventory())


def test_unknown_stock_item_named():
    plan = _valid_plan()
    plan.assignments[0].stock_item_id = "ghost"
    with pytest.raises(PlanVerificationError, match="ghost"):
        verify_plan(plan, _cutting_list(), _inventory())


def test_item_cut_twice_fails():
    plan = _valid_plan()
    duplicate = Cut(
        stock_item_id="b1",
        pieces=[],
        sequence=3,
        kerf_total=0,
        offcut_length=6000,
    )
    plan.assignments.append(duplicate)
    with pytest.raises(PlanVerificationError, match="b1"):
        verify_plan(plan, _cutting_list(), _inventory())


def test_unknown_part_named():
    plan = _valid_plan()
    plan.assignments[0].pieces.append(CutPiece("pX", 100))
    with pytest.raises(PlanVerificationError, match="pX"):
        verify_plan(plan, _cutting_list(), _inventory())


def test_material_mismatch_fails():
    inventory = [
        StockItem(id="b1", material_id="m1", length=6000),
        StockItem(id="r1", material_id="m2", length=2000, origin="remnant"),
    ]
    with pytest.raises(PlanVerificationError, match="r1"):
        verify_plan(_valid_plan(), _cutting_list(), inventory)


def test_underproduction_named():
    plan = _valid_plan()
    plan.assignments[1].pieces = []
    plan.assignments[1].offcut_length = 2000
    with pytest.raises(PlanVerificationError, match="p3"):
        verify_plan(plan, _cutting_list(), _inventory())


def test_overproduction_is_legal():
    plan = _valid_plan()
    # r1 2000: 950 + 950 + 3 kerf + 97 offcut; p3 demanded 1, produced 2.
    plan.assignments[1] = Cut(
        stock_item_id="r1",
        pieces=[CutPiece("p3", 950), CutPiece("p3", 950)],
        sequence=2,
        kerf_total=3,
        offcut_length=97,
    )
    verify_plan(plan, _cutting_list(), _inventory())


def test_problems_are_aggregated():
    plan = _valid_plan()
    plan.assignments[0].offcut_length = 100  # arithmetic broken
    plan.assignments[1].pieces = []  # p3 underproduced
    plan.assignments[1].offcut_length = 2000
    with pytest.raises(PlanVerificationError) as excinfo:
        verify_plan(plan, _cutting_list(), _inventory())
    assert len(excinfo.value.problems) >= 2
