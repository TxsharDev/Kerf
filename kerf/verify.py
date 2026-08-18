"""Plan verification: the never-uncuttable oracle.

The solver is only as trustworthy as its output. This module re-checks a
finished plan from scratch, without trusting anything the solver
computed internally, and answers one question: *can a shop execute this
plan exactly as written?*

A plan is valid when, for every cut:

* the stock item it names exists in the inventory;
* no stock item is cut more than once;
* the cut's arithmetic reconciles the item's full length,
  ``sum(pieces) + kerf_total + offcut_length == item.length``;
* every piece names a part that exists on the cutting list;
* every piece's part material matches the stock item's material.

And across the whole plan: every part on the list is produced at least
its demanded quantity. Overproduction is legal (an extra piece becomes
a remnant on the rack); underproduction is not.

The check collects *all* problems in one pass rather than stopping at
the first, so a plan that is wrong in several ways is reported in full.
Callers treat this as the ground truth the solver is graded against,
the property tests, and the E2E round-trip tests all reuse it.
"""

from __future__ import annotations

from .errors import PlanVerificationError
from .models import CuttingList, Plan, StockItem


def verify_plan(plan: Plan, cutting_list: CuttingList, inventory) -> None:
    """Verify ``plan`` against ``cutting_list`` and ``inventory``.

    Returns ``None`` when the plan is valid. Raises
    :class:`PlanVerificationError` (with every problem listed) when it
    is not. ``inventory`` is any iterable of :class:`StockItem`; items
    are matched by ``id``.
    """
    items = {item.id: item for item in inventory}
    parts_by_id = {part.id: part for part in cutting_list.parts}
    problems: list[str] = []

    seen_items: set[str] = set()
    produced: dict[str, int] = {}

    for cut in plan.assignments:
        item = items.get(cut.stock_item_id)
        if item is None:
            problems.append(
                f"cut {cut.sequence} references unknown stock item "
                f"{cut.stock_item_id!r}"
            )
            continue

        if cut.stock_item_id in seen_items:
            problems.append(
                f"stock item {cut.stock_item_id!r} is cut more than once"
            )
        seen_items.add(cut.stock_item_id)

        piece_sum = sum(piece.length for piece in cut.pieces)
        if piece_sum + cut.kerf_total + cut.offcut_length != item.length:
            problems.append(
                f"cut on {cut.stock_item_id!r} does not reconcile: "
                f"{piece_sum} pieces + {cut.kerf_total} kerf + "
                f"{cut.offcut_length} offcut != {item.length}"
            )

        for piece in cut.pieces:
            part = parts_by_id.get(piece.part_id)
            if part is None:
                problems.append(
                    f"cut on {cut.stock_item_id!r} names unknown part "
                    f"{piece.part_id!r}"
                )
                continue
            if part.material_id != item.material_id:
                problems.append(
                    f"cut on {cut.stock_item_id!r}: part {piece.part_id!r} is "
                    f"material {part.material_id!r} but stock is "
                    f"{item.material_id!r}"
                )
            produced[piece.part_id] = produced.get(piece.part_id, 0) + 1

    for part in cutting_list.parts:
        have = produced.get(part.id, 0)
        if have < part.quantity:
            problems.append(
                f"part {part.id!r}: produced {have} of {part.quantity} demanded"
            )

    if problems:
        raise PlanVerificationError(problems)
