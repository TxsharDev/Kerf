"""Constraint module for the constraint engine.

Products add constraint types by contributing them to the engine,
never by implementing them privately.
"""

from constraint_engine.constraints.base import ConstraintProtocol

__all__ = ["ConstraintProtocol"]
