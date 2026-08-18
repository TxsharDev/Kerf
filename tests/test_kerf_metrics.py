"""Proof metrics: the five numbers the plan owes the shop.

Pure functions over finished cuts plus the stock items they consumed.
No domain state is mutated, so the solver, the ledger, and the tests
all compute the same dollars from the same inputs.
"""

from kerf.metrics import (
    compute_all,
    dollars_saved,
    new_stock_used,
    remnant_length_consumed,
    utilization,
    waste_length,
)
from kerf.models import (
    Cut,
    CutPiece,
    Material,
    Origin,
    StockItem,
)


def _purchased(item_id="b1", material_id="m1", length=6000):
    return StockItem(id=item_id, material_id=material_id, length=length)


def _remnant(item_id="r1", material_id="m1", length=2000):
    return StockItem(
        id=item_id, material_id=material_id, length=length, origin=Origin.REMNANT
    )


def _standard_items():
    return {"b1": _purchased(), "r1": _remnant()}


def _standard_cuts():
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
    return [c1, c2]


def _materials():
    return {
        "m1": Material(id="m1", unit_cost=0.04),
        "m2": Material(id="m2", unit_cost=0.10),
    }


def test_utilization_percentage():
    cuts, items = _standard_cuts(), _standard_items()
    # part mm 5250 / stock mm 8000 = 65.625% -> rounded half to even
    assert utilization(cuts, items) == 65.62


def test_utilization_zero_without_cuts():
    assert utilization([], _standard_items()) == 0.0


def test_waste_length_is_stock_minus_parts():
    cuts, items = _standard_cuts(), _standard_items()
    assert waste_length(cuts, items) == 2750


def test_waste_length_zero_without_cuts():
    assert waste_length([], _standard_items()) == 0


def test_new_stock_used_counts_whole_purchased_items():
    cuts, items = _standard_cuts(), _standard_items()
    assert new_stock_used(cuts, items) == 6000


def test_remnant_length_consumed_counts_pieces_from_remnants():
    cuts, items = _standard_cuts(), _standard_items()
    assert remnant_length_consumed(cuts, items) == 950


def test_remnant_length_consumed_ignores_purchased():
    items = {"b1": _purchased()}
    c1 = Cut(
        stock_item_id="b1",
        pieces=[CutPiece("p1", 2500)],
        kerf_total=0,
        offcut_length=3500,
    )
    assert remnant_length_consumed([c1], items) == 0


def test_dollars_saved_is_remnant_mm_times_unit_cost():
    cuts, items = _standard_cuts(), _standard_items()
    assert dollars_saved(cuts, items, _materials()) == 38.0


def test_dollars_saved_sums_across_materials():
    items = {
        "b1": _purchased("b1", "m1"),
        "r2": _remnant("r2", "m2", 1000),
    }
    c1 = Cut(
        stock_item_id="b1",
        pieces=[CutPiece("p1", 100)],
        kerf_total=0,
        offcut_length=5900,
    )
    c2 = Cut(
        stock_item_id="r2",
        pieces=[CutPiece("p2", 200)],
        kerf_total=0,
        offcut_length=800,
    )
    assert dollars_saved([c1, c2], items, _materials()) == 20.0


def test_dollars_saved_unknown_material_raises():
    cuts, items = _standard_cuts(), _standard_items()
    try:
        dollars_saved(cuts, items, {})
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_compute_all_returns_exactly_the_five_keys():
    cuts, items = _standard_cuts(), _standard_items()
    metrics = compute_all(cuts, items, _materials())
    assert set(metrics) == {
        "utilization",
        "waste_length",
        "new_stock_used",
        "remnant_length_consumed",
        "dollars_saved",
    }
    assert metrics["utilization"] == 65.62
    assert metrics["waste_length"] == 2750
    assert metrics["new_stock_used"] == 6000
    assert metrics["remnant_length_consumed"] == 950
    assert metrics["dollars_saved"] == 38.0


def test_compute_all_empty():
    metrics = compute_all([], _standard_items(), _materials())
    assert metrics == {
        "utilization": 0.0,
        "waste_length": 0,
        "new_stock_used": 0,
        "remnant_length_consumed": 0,
        "dollars_saved": 0.0,
    }
