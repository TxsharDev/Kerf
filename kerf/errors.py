"""Errors for the Kerf product.

The solver raises exactly one error type for an infeasible cutting
list. It carries the per-material reasons so the UI can show a shop
owner what is short, and a flattened list for logs and tests.
"""

from __future__ import annotations

from typing import Mapping, Sequence


class InfeasiblePlanError(Exception):
    """The cutting list cannot be satisfied by the available inventory.

    ``by_material`` maps each unmet material to the engine's reasons
    (no bars, part too long, copies exceed supply, volume exceeded).
    ``reasons`` flattens the same content, one string per line, with
    the material id as prefix. Both are ordered by material id so
    output is deterministic.
    """

    def __init__(self, by_material: Mapping[str, Sequence[str]]):
        self.by_material: dict[str, list[str]] = {
            material: list(reasons)
            for material, reasons in sorted(by_material.items())
        }
        self.reasons: list[str] = [
            f"{material}: {reason}"
            for material, group in self.by_material.items()
            for reason in group
        ]
        super().__init__("infeasible cutting plan: " + "; ".join(self.reasons))
