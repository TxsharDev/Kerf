"""Proof metrics for a finished cutting plan.

Five numbers, all computed from the same finished cuts and the stock
items they consumed:

* ``utilization``              part mm / consumed stock mm, percent.
* ``waste_length``             offcuts plus kerf, in millimetres.
* ``new_stock_used``           purchased stock consumed, whole items.
* ``remnant_length_consumed``  part mm produced from rack remnants.
* ``dollars_saved``            remnant mm priced at each material cost.

Every function is pure: no domain state is touched, so the solver,
the ledger, and the tests all derive the same dollars.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .models import Cut, Material, Origin, StockItem


def _part_mm(cuts: Sequence[Cut]) -> int:
    return sum(piece.length for cut in cuts for piece in cut.pieces)


def _stock_mm(cuts: Sequence[Cut], items: Mapping[str, StockItem]) -> int:
    return sum(items[cut.stock_item_id].length for cut in cuts)


def _is_remnant(item: StockItem) -> bool:
    return item.origin is Origin.REMNANT


def utilization(cuts: Sequence[Cut], items: Mapping[str, StockItem]) -> float:
    """Part mm as a percentage of consumed stock mm, 0.0 with no cuts."""
    if not cuts:
        return 0.0
    total = _stock_mm(cuts, items)
    if total <= 0:
        return 0.0
    return round(_part_mm(cuts) / total * 100, 2)


def waste_length(cuts: Sequence[Cut], items: Mapping[str, StockItem]) -> int:
    """Millimetres lost to offcuts and kerf across the consumed stock."""
    if not cuts:
        return 0
    return _stock_mm(cuts, items) - _part_mm(cuts)


def new_stock_used(cuts: Sequence[Cut], items: Mapping[str, StockItem]) -> int:
    """Millimetres of purchased stock consumed, whole items."""
    return sum(
        items[cut.stock_item_id].length
        for cut in cuts
        if not _is_remnant(items[cut.stock_item_id])
    )


def remnant_length_consumed(
    cuts: Sequence[Cut], items: Mapping[str, StockItem]
) -> int:
    """Part mm actually produced from rack remnants (overproduction too)."""
    total = 0
    for cut in cuts:
        if _is_remnant(items[cut.stock_item_id]):
            total += sum(piece.length for piece in cut.pieces)
    return total


def dollars_saved(
    cuts: Sequence[Cut],
    items: Mapping[str, StockItem],
    materials: Mapping[str, Material],
) -> float:
    """Remnant consumption priced at each material's per-mm unit cost."""
    total = 0.0
    for cut in cuts:
        item = items[cut.stock_item_id]
        if not _is_remnant(item):
            continue
        total += (
            sum(piece.length for piece in cut.pieces)
            * materials[item.material_id].unit_cost
        )
    return total


def compute_all(
    cuts: Sequence[Cut],
    items: Mapping[str, StockItem],
    materials: Mapping[str, Material],
) -> dict:
    """All five proof metrics keyed by name."""
    return {
        "utilization": utilization(cuts, items),
        "waste_length": waste_length(cuts, items),
        "new_stock_used": new_stock_used(cuts, items),
        "remnant_length_consumed": remnant_length_consumed(cuts, items),
        "dollars_saved": dollars_saved(cuts, items, materials),
    }
