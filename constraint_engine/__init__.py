"""Constraint Engine.

A shared solver library for resource-constrained optimization.
Products import this as a library; it has no web framework, no database
driver, and no cloud dependency.

Public API:
    solve(problem, options)    -> Solution
    explain(solution, task_id) -> list[Reason]
    replan(problem, prev, events) -> Solution
    diagnose(problem)          -> list[Diagnosis]
"""

from constraint_engine.engine import diagnose, explain, replan, solve
from constraint_engine.types import (
    Constraint,
    ConstraintType,
    Diagnosis,
    Objective,
    ObjectiveTerm,
    Problem,
    Reason,
    Resource,
    ResourceKind,
    Solution,
    Task,
    Violation,
)

__version__ = "0.1.0"

__all__ = [
    "Constraint",
    "ConstraintType",
    "Diagnosis",
    "Objective",
    "ObjectiveTerm",
    "Problem",
    "Reason",
    "Resource",
    "ResourceKind",
    "Solution",
    "Task",
    "Violation",
    "__version__",
    "diagnose",
    "explain",
    "replan",
    "solve",
]
