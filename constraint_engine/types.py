"""Core data types for the Constraint Engine.

Five concepts form the boundary between products and the engine:
Resource, Task, Constraint, Objective, Solution.

Every product expresses its problem in these terms. Nothing else
crosses the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------


class ResourceKind(str, Enum):
    """Kind of finite resource that work consumes."""

    MACHINE = "machine"
    OPERATOR = "operator"
    TOOL = "tool"
    FIXTURE = "fixture"
    VEHICLE = "vehicle"
    MATERIAL = "material"
    SPACE = "space"


@dataclass
class Resource:
    """Something finite that work consumes.

    Material is a resource. A remnant is a resource with a quantity
    (its remaining length) and geometry attributes (profile, age).
    """

    id: str
    kind: ResourceKind
    capacity: float = 1.0
    calendar: list[dict[str, Any]] = field(default_factory=list)
    skills: set[str] = field(default_factory=set)
    attributes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A unit of work that consumes resources."""

    id: str
    demands: list[dict[str, Any]] = field(default_factory=list)
    release: float = 0.0
    due: float | None = None
    deadline: float | None = None
    priority: float = 1.0
    predecessors: list[str] = field(default_factory=list)
    group: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


class ConstraintType(str, Enum):
    """The eight constraint types supported by the engine."""

    CAPACITY = "capacity"
    PRECEDENCE = "precedence"
    SKILL = "skill"
    CALENDAR = "calendar"
    SETUP = "setup"
    EXCLUSION = "exclusion"
    GROUPING = "grouping"
    QUANTITY = "quantity"


@dataclass
class Constraint:
    """A typed, explicitly hard or soft constraint.

    Hard constraints define feasibility. Soft constraints carry a
    violation cost and appear in the objective.
    """

    id: str
    type: ConstraintType
    hard: bool = True
    cost: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


@dataclass
class ObjectiveTerm:
    """A single named term in the weighted objective."""

    name: str
    weight: float
    sense: str = "minimize"  # "minimize" | "maximize"


@dataclass
class Objective:
    """A weighted sum of named terms.

    Standard terms: tardiness, makespan, setup_time, material_waste,
    travel, overtime, cost, churn.
    """

    terms: list[ObjectiveTerm] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------


@dataclass
class Reason:
    """A structured explanation for why a task sits where it does.

    Products render this; the engine produces it.
    """

    kind: str  # binding_constraint | resource_scarcity | precedence | due_date | setup_saving | alternative_worse
    subject: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Violation:
    """A constraint violation in an infeasible or soft-violating solution."""

    constraint_id: str
    description: str
    severity: float = 1.0


@dataclass
class Solution:
    """The result of a solve operation."""

    assignments: list[dict[str, Any]] = field(default_factory=list)
    objective: dict[str, Any] = field(default_factory=dict)
    explanations: list[dict[str, Any]] = field(default_factory=list)
    feasibility: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """A complete optimization problem expressed in core types."""

    resources: list[Resource] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    objective: Objective = field(default_factory=Objective)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


@dataclass
class Diagnosis:
    """A minimal conflicting constraint set for an infeasible problem."""

    constraint_ids: list[str]
    explanation: str
    suggested_removals: list[str] = field(default_factory=list)
