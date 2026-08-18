"""Domain model tests for the Kerf product."""

from datetime import datetime, timezone

import pytest

from kerf.models import (
    Cut,
    CutPiece,
    CuttingList,
    Material,
    Origin,
    Part,
    Plan,
    PlanStatus,
    SavingsEvent,
    StockItem,
    StockStatus,
)


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


def test_material_defaults():
    material = Material(id="m1")
    assert material.name == ""
    assert material.form == "bar"
    assert material.unit_cost == 0.0
    assert material.profile_spec == ""
    assert material.density is None


def test_material_rejects_empty_id():
    with pytest.raises(ValueError, match="id"):
        Material(id="")


def test_material_rejects_negative_unit_cost():
    with pytest.raises(ValueError, match="unit_cost"):
        Material(id="m1", unit_cost=-0.5)


def test_material_rejects_unknown_form():
    with pytest.raises(ValueError, match="form"):
        Material(id="m1", form="widget")


@pytest.mark.parametrize("form", ["bar", "tube", "extrusion", "board", "profile"])
def test_material_accepts_all_forms(form):
    assert Material(id="m1", form=form).form == form


# ---------------------------------------------------------------------------
# StockItem
# ---------------------------------------------------------------------------


def test_stock_item_defaults():
    item = StockItem(id="s1", material_id="m1", length=6000)
    assert item.origin is Origin.PURCHASED
    assert item.parent_plan_id is None
    assert item.location == ""
    assert item.created_at is None
    assert item.status is StockStatus.AVAILABLE


def test_stock_item_rejects_non_positive_length():
    with pytest.raises(ValueError, match="length"):
        StockItem(id="s1", material_id="m1", length=0)
    with pytest.raises(ValueError, match="length"):
        StockItem(id="s1", material_id="m1", length=-5)


def test_stock_item_rejects_empty_id():
    with pytest.raises(ValueError, match="id"):
        StockItem(id="", material_id="m1", length=6000)


def test_stock_item_rejects_unknown_origin():
    with pytest.raises(ValueError, match="origin"):
        StockItem(id="s1", material_id="m1", length=6000, origin="salvage")


def test_stock_item_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        StockItem(id="s1", material_id="m1", length=6000, status="frozen")


def test_stock_item_accepts_string_enums():
    """Plain strings equal the enum members, so both spellings construct."""
    item = StockItem(id="s1", material_id="m1", length=100, origin="remnant")
    assert item.origin is Origin.REMNANT


def test_stock_item_remnant_with_parent():
    item = StockItem(
        id="s1",
        material_id="m1",
        length=100,
        origin=Origin.REMNANT,
        parent_plan_id="plan-1",
        created_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    assert item.parent_plan_id == "plan-1"
    assert item.created_at is not None


# ---------------------------------------------------------------------------
# Part
# ---------------------------------------------------------------------------


def test_part_defaults():
    part = Part(id="p1", material_id="m1", length=1000, quantity=2)
    assert part.label == ""
    assert part.tolerance is None


def test_part_rejects_empty_id():
    with pytest.raises(ValueError, match="id"):
        Part(id="", material_id="m1", length=1000, quantity=1)


def test_part_rejects_non_positive_length():
    with pytest.raises(ValueError, match="length"):
        Part(id="p1", material_id="m1", length=0, quantity=1)


def test_part_rejects_zero_quantity():
    with pytest.raises(ValueError, match="quantity"):
        Part(id="p1", material_id="m1", length=1000, quantity=0)


# ---------------------------------------------------------------------------
# CuttingList
# ---------------------------------------------------------------------------


def test_cutting_list_defaults():
    cutting_list = CuttingList(id="cl1")
    assert cutting_list.parts == []
    assert cutting_list.customer_ref == ""
    assert cutting_list.created_at is None
    assert cutting_list.status == "open"


def test_cutting_list_rejects_empty_id():
    with pytest.raises(ValueError, match="id"):
        CuttingList(id="")


def test_cutting_list_rejects_duplicate_part_ids():
    parts = [
        Part(id="p1", material_id="m1", length=100, quantity=1),
        Part(id="p1", material_id="m1", length=200, quantity=1),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        CuttingList(id="cl1", parts=parts)


# ---------------------------------------------------------------------------
# Cut and CutPiece
# ---------------------------------------------------------------------------


def test_cut_piece_fields():
    piece = CutPiece(part_id="p1", length=1000)
    assert piece.part_id == "p1"
    assert piece.length == 1000


def test_cut_rejects_empty_stock_item_id():
    with pytest.raises(ValueError, match="stock_item_id"):
        Cut(stock_item_id="", sequence=1, pieces=[])


def test_cut_rejects_negative_kerf_total():
    with pytest.raises(ValueError, match="kerf_total"):
        Cut(stock_item_id="s1", sequence=1, pieces=[], kerf_total=-1)


def test_cut_rejects_negative_offcut():
    with pytest.raises(ValueError, match="offcut_length"):
        Cut(stock_item_id="s1", sequence=1, pieces=[], offcut_length=-1)


def test_cut_default_sequence_is_one():
    cut = Cut(stock_item_id="s1", pieces=[])
    assert cut.sequence == 1


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def test_plan_defaults():
    plan = Plan(id="plan-1", cutting_list_id="cl1")
    assert plan.assignments == []
    assert plan.metrics == {}
    assert plan.remnants_created == []
    assert plan.stats == {}
    assert plan.explanation == []
    assert plan.status is PlanStatus.PROPOSED


def test_plan_rejects_empty_id():
    with pytest.raises(ValueError, match="id"):
        Plan(id="", cutting_list_id="cl1")


def test_plan_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        Plan(id="plan-1", cutting_list_id="cl1", status="shipped")


# ---------------------------------------------------------------------------
# SavingsEvent
# ---------------------------------------------------------------------------


def test_savings_event_fields():
    now = datetime(2026, 8, 17, tzinfo=timezone.utc)
    event = SavingsEvent(
        plan_id="plan-1",
        occurred_at=now,
        remnant_length_consumed=1234,
        dollars_avoided=12.34,
    )
    assert event.plan_id == "plan-1"
    assert event.occurred_at == now
    assert event.remnant_length_consumed == 1234
    assert event.dollars_avoided == 12.34


def test_savings_event_rejects_negative_length():
    with pytest.raises(ValueError, match="remnant_length_consumed"):
        SavingsEvent(
            plan_id="plan-1",
            occurred_at=datetime.now(timezone.utc),
            remnant_length_consumed=-1,
            dollars_avoided=0.0,
        )


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


def test_enum_values_match_spec():
    assert Origin.PURCHASED.value == "purchased"
    assert Origin.REMNANT.value == "remnant"
    assert StockStatus.AVAILABLE.value == "available"
    assert StockStatus.RESERVED.value == "reserved"
    assert StockStatus.CONSUMED.value == "consumed"
    assert StockStatus.SCRAPPED.value == "scrapped"
    assert PlanStatus.PROPOSED.value == "proposed"
    assert PlanStatus.RELEASED.value == "released"
    assert PlanStatus.CUT.value == "cut"
    assert PlanStatus.ABANDONED.value == "abandoned"
