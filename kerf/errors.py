"""Errors for the Kerf product.

Two error types, one per failure surface:

* :class:`InfeasiblePlanError` - the solver cannot satisfy a cutting
  list. It carries the per-material reasons so the UI can show a shop
  owner what is short, and a flattened list for logs and tests.
* :class:`PlanVerificationError` - a produced plan fails the
  verification oracle. It aggregates every problem found so a plan
  that is wrong in several ways is reported in a single pass.
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


class PlanVerificationError(Exception):
    """A produced plan failed the verification oracle.

    ``problems`` holds every discrepancy found in a single pass, in
    the order they were discovered: arithmetic that does not reconcile,
    unknown stock items or parts, a stock item cut more than once,
    material mismatches, and underproduced parts. Each string names the
    offending item, part, or cut so the UI can point at it. The flat
    ``str()`` joins them with a semicolon for logs and tests.
    """

    def __init__(self, problems: Sequence[str]):
        self.problems: list[str] = list(problems)
        super().__init__("plan verification failed: " + "; ".join(self.problems))
