"""Base protocol for engine constraints.

Each constraint type implements this protocol. The solver dispatches
to the appropriate implementation based on ConstraintType.
"""

from __future__ import annotations

from typing import Any

from constraint_engine.types import Constraint, Problem


class ConstraintProtocol:
    """Interface that every constraint type must satisfy.

    Implementations validate the constraint against a problem and,
    where applicable, add terms to the solver model.
    """

    @property
    def constraint_type(self) -> str:
        """The ConstraintType enum value this implementation handles."""
        raise NotImplementedError

    def validate(self, constraint: Constraint, problem: Problem) -> list[str]:
        """Check the constraint is well-formed for this problem.

        Returns a list of error strings (empty if valid).
        """
        raise NotImplementedError

    def apply(self, constraint: Constraint, problem: Problem, model: Any) -> None:
        """Add this constraint to the solver model.

        The model is solver-specific (CP-SAT model, LP, etc.).
        """
        raise NotImplementedError
