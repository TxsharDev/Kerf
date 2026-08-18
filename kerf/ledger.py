"""The Kerf ledger: recording a cut plan against the stock room.

The solver proposes, the shop cuts, the ledger records. One entry
point:

* ``record_plan_cut`` -- the plan moves to CUT, the stock it consumed
  moves to CONSUMED, its offcuts go back on the rack as AVAILABLE
  remnants, and a :class:`SavingsEvent` is written with the proof
  numbers derived from the pure metrics, not from plan state.

Every guard runs before any mutation: a plan that is not PROPOSED or
RELEASED, a plan that cuts the same bar twice, a plan that names a bar
that is not on the rack, or a plan that touches a bar that is not
AVAILABLE all raise ``ValueError`` and leave the plan and the rack
exactly as they were.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

from .metrics import dollars_saved, remnant_length_consumed
from .models import (
    Material,
    Plan,
    PlanStatus,
    SavingsEvent,
    StockItem,
    StockStatus,
)


def _check_plan(plan: Plan) -> None:
    """Only PROPOSED or RELEASED plans can be cut."""
    if plan.status not in (PlanStatus.PROPOSED, PlanStatus.RELEASED):
        raise ValueError(
            f"plan {plan.id!r} is {plan.status.value}, only PROPOSED or "
            "RELEASED plans can be cut"
        )


def _check_items(plan: Plan, inventory: Mapping[str, StockItem]) -> None:
    """Unique, known, available stock -- checked before any mutation."""
    seen: set[str] = set()
    for cut in plan.assignments:
        if cut.stock_item_id in seen:
            raise ValueError(
                f"plan {plan.id!r} cuts stock item {cut.stock_item_id!r} more than once"
            )
        seen.add(cut.stock_item_id)
        if cut.stock_item_id not in inventory:
            raise ValueError(
                f"plan {plan.id!r} cuts unknown stock item {cut.stock_item_id!r}"
            )
        item = inventory[cut.stock_item_id]
        if item.status is not StockStatus.AVAILABLE:
            raise ValueError(
                f"plan {plan.id!r} cuts stock item {cut.stock_item_id!r} "
                f"which is {item.status.value}"
            )


def record_plan_cut(
    plan: Plan,
    inventory: dict[str, StockItem],
    materials: Mapping[str, Material],
) -> SavingsEvent:
    """Record a plan as cut and return its savings event.

    The consumed stock items move to CONSUMED, every remnant in
    ``plan.remnants_created`` is placed on the rack under its own id
    (AVAILABLE, origin REMNANT), and the plan moves to CUT.

    The event's numbers come from the pure metrics over the plan's
    assignments and the rack: ``remnant_length_consumed`` is the part
    mm produced from remnants, ``dollars_avoided`` is that length
    priced at each material's unit cost. The metrics read only origin
    and length, never status, so the consumption they measure is the
    same before and after the mutation.

    Raises ``ValueError`` when the plan is not cuttable (wrong status,
    a bar cut twice, an unknown bar, or a bar that is not AVAILABLE);
    the plan and the inventory are left untouched.
    """
    _check_plan(plan)
    _check_items(plan, inventory)

    event = SavingsEvent(
        plan_id=plan.id,
        occurred_at=datetime.now(),
        remnant_length_consumed=remnant_length_consumed(plan.assignments, inventory),
        dollars_avoided=dollars_saved(plan.assignments, inventory, materials),
    )

    for cut in plan.assignments:
        inventory[cut.stock_item_id].status = StockStatus.CONSUMED
    for remnant in plan.remnants_created:
        inventory[remnant.id] = remnant
    plan.status = PlanStatus.CUT

    return event
