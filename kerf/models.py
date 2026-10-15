"""Domain model for the Kerf product.

The records are materials, stock items, cutting lists, plans, cuts,
and savings events.

Two design points carry the product:

* A remnant is a :class:`StockItem`, not a separate table. One table
  with an ``origin`` field keeps the solver and the ledger simple.
* Statuses are plain enums so a plan can move proposed -> released ->
  cut, and a stock item can move available -> consumed or scrapped.

Structural errors (bad ids, negative lengths, unknown statuses) raise
``ValueError`` at construction. Whether a cutting list can be satisfied
is not a construction question; the solver answers that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

MATERIAL_FORMS = ("bar", "tube", "extrusion", "board", "profile")


class Origin(str, Enum):
    """Where a stock item came from."""

    PURCHASED = "purchased"
    REMNANT = "remnant"


class StockStatus(str, Enum):
    """Lifecycle of a stock item on the rack."""

    AVAILABLE = "available"
    RESERVED = "reserved"
    CONSUMED = "consumed"
    SCRAPPED = "scrapped"


class PlanStatus(str, Enum):
    """Lifecycle of a cutting plan."""

    PROPOSED = "proposed"
    RELEASED = "released"
    CUT = "cut"
    ABANDONED = "abandoned"


def _as_enum(value, enum_cls, field_name):
    """Coerce a raw string to its enum member; raise on anything else.

    Callers may pass ``Origin.REMNANT`` or the plain value ``"remnant"``;
    both land on the same member. Anything not coercible is a
    structural error and fails fast at construction.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{field_name} must be a {enum_cls.__name__} or its value, got {value!r}"
        ) from None


@dataclass(frozen=True)
class Material:
    """A material family with interchangeable profile.

    ``unit_cost`` is per millimetre of length; it prices purchased
    stock for the solver and converts remnant consumption into dollars
    saved in the ledger.
    """

    id: str
    name: str = ""
    form: str = "bar"
    profile_spec: str = ""
    unit_cost: float = 0.0
    density: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Material.id must not be empty")
        if self.form not in MATERIAL_FORMS:
            raise ValueError(
                f"Material.form must be one of {MATERIAL_FORMS}, got {self.form!r}"
            )
        if self.unit_cost < 0:
            raise ValueError(f"Material.unit_cost must be non-negative, got {self.unit_cost}")


@dataclass
class StockItem:
    """One physical piece of stock on the rack.

    Purchased stock and remnants live here together. A remnant carries
    the plan that created it (``parent_plan_id``) and the moment it was
    created (``created_at``), which drives age-based clearing order.
    """

    id: str
    material_id: str
    length: int
    origin: Origin = Origin.PURCHASED
    parent_plan_id: str | None = None
    location: str = ""
    created_at: datetime | None = None
    status: StockStatus = StockStatus.AVAILABLE

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("StockItem.id must not be empty")
        if self.length <= 0:
            raise ValueError(f"StockItem.length must be positive, got {self.length}")
        self.origin = _as_enum(self.origin, Origin, "origin")
        self.status = _as_enum(self.status, StockStatus, "status")



@dataclass
class Part:
    """One line on a cutting list: cut ``quantity`` pieces of ``length``."""

    id: str
    material_id: str
    length: int
    quantity: int
    label: str = ""
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Part.id must not be empty")
        if self.length <= 0:
            raise ValueError(f"Part.length must be positive, got {self.length}")
        if self.quantity < 1:
            raise ValueError(f"Part.quantity must be at least 1, got {self.quantity}")


@dataclass
class CuttingList:
    """A customer order expressed as parts to cut."""

    id: str
    parts: list[Part] = field(default_factory=list)
    customer_ref: str = ""
    created_at: datetime | None = None
    status: str = "open"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("CuttingList.id must not be empty")
        seen: set[str] = set()
        for part in self.parts:
            if part.id in seen:
                raise ValueError(f"duplicate part id {part.id!r} in cutting list")
            seen.add(part.id)


@dataclass(frozen=True)
class CutPiece:
    """One physical piece cut from a stock item."""

    part_id: str
    length: int


@dataclass
class Cut:
    """Everything cut from one stock item, in cutting order."""

    stock_item_id: str
    pieces: list[CutPiece] = field(default_factory=list)
    sequence: int = 1
    kerf_total: int = 0
    offcut_length: int = 0

    def __post_init__(self) -> None:
        if not self.stock_item_id:
            raise ValueError("Cut.stock_item_id must not be empty")
        if self.kerf_total < 0:
            raise ValueError(f"Cut.kerf_total must be non-negative, got {self.kerf_total}")
        if self.offcut_length < 0:
            raise ValueError(
                f"Cut.offcut_length must be non-negative, got {self.offcut_length}"
            )


@dataclass
class Plan:
    """A proposed cutting plan against a cutting list and live stock.

    ``assignments`` holds one :class:`Cut` per stock item the plan
    consumes. ``remnants_created`` lists the offcuts (and overproduced
    pieces) that return to the rack when the plan is cut. ``stats``
    carries per-material engine diagnostics, ``explanation`` notes the
    interesting outcomes (overproduced parts), and ``metrics`` holds the
    five proof numbers.
    """

    id: str
    cutting_list_id: str
    assignments: list[Cut] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    remnants_created: list[StockItem] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    explanation: list[dict] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PROPOSED
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Plan.id must not be empty")
        if not isinstance(self.status, PlanStatus):
            raise ValueError(f"Plan.status must be a PlanStatus, got {self.status!r}")


@dataclass(frozen=True)
class SavingsEvent:
    """The ledger row written when a plan is marked cut.

    ``dollars_avoided`` is the remnant length consumed times the
    material unit cost: new material the shop did not have to buy.
    """

    plan_id: str
    occurred_at: datetime
    remnant_length_consumed: int
    dollars_avoided: float

    def __post_init__(self) -> None:
        if self.remnant_length_consumed < 0:
            raise ValueError(
                "SavingsEvent.remnant_length_consumed must be non-negative, "
                f"got {self.remnant_length_consumed}"
            )
