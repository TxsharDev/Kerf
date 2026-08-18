"""Cutting list importer: shop-floor CSV into the domain model.

The file contract is deliberate and strict: a header row of exactly
``part_id,material_id,length_mm,quantity,label`` and one row per part.
Every error names the data row number (first data row is row 1) so a
shop owner can find the bad line without a diff tool.
"""

from __future__ import annotations

import csv
import secrets
from pathlib import Path
from typing import Union

from .models import CuttingList, Part

REQUIRED_COLUMNS = ("part_id", "material_id", "length_mm", "quantity", "label")


def _default_cutting_list_id() -> str:
    return f"cl-{secrets.token_hex(6)}"


def _parse_positive_int(raw: str, column: str, row: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"row {row}: {column} must be a whole number, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(f"row {row}: {column} must be at least 1, got {value}")
    return value


def load_cutting_list(
    path: Union[str, Path],
    *,
    cutting_list_id: str | None = None,
    customer_ref: str = "",
) -> CuttingList:
    """Read a cutting list CSV into a :class:`CuttingList`.

    ``cutting_list_id`` defaults to a fresh ``cl-<12 hex>`` id.
    ``customer_ref`` is passed through untouched for paper trails.
    Structural problems (bad header, bad cells, duplicate part ids)
    raise ``ValueError`` naming the offending row. A missing file
    raises the natural ``FileNotFoundError``.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))

    if not rows or not rows[0]:
        raise ValueError("empty file: expected a header row")

    header = [cell.strip() for cell in rows[0]]
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    extra = [column for column in header if column not in REQUIRED_COLUMNS]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"unexpected column(s): {', '.join(extra)}")

    parts: list[Part] = []
    seen_ids: set[str] = set()
    for row_number, cells in enumerate(rows[1:], start=1):
        if not cells or all(not cell.strip() for cell in cells):
            continue
        if len(cells) != len(REQUIRED_COLUMNS):
            raise ValueError(
                f"row {row_number}: expected {len(REQUIRED_COLUMNS)} "
                f"columns, got {len(cells)}"
            )
        part_id, material_id, length_raw, quantity_raw, label = (
            cell.strip() for cell in cells
        )
        if not part_id or not material_id or not length_raw or not quantity_raw:
            raise ValueError(
                f"row {row_number}: part_id, material_id, length_mm and "
                "quantity are required"
            )
        length = _parse_positive_int(length_raw, "length_mm", row_number)
        quantity = _parse_positive_int(quantity_raw, "quantity", row_number)
        if part_id in seen_ids:
            raise ValueError(f"row {row_number}: duplicate part id {part_id!r}")
        seen_ids.add(part_id)
        parts.append(
            Part(
                id=part_id,
                material_id=material_id,
                length=length,
                quantity=quantity,
                label=label,
            )
        )

    return CuttingList(
        id=cutting_list_id or _default_cutting_list_id(),
        parts=parts,
        customer_ref=customer_ref,
    )
