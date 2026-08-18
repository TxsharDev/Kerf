"""Tests for the engine public operations: solve, explain, replan, diagnose.

These are the four operations every product talks to. The engine
dispatches on ``problem.metadata["model"]``; the 1D cutting payload
lives under ``metadata["cutting"]`` as a ``CuttingProblem1D``.
"""

import pytest

from constraint_engine import (
    Diagnosis,
    Problem,
    Solution,
    diagnose,
    explain,
    replan,
    solve,
)
from constraint_engine.geometry.types import BarSpec, CuttingProblem1D


def bar(length_mm, bar_id="b0", cost=10.0):
    return BarSpec(length_mm=length_mm, cost=cost, origin="new", id=bar_id)


def cutting(parts, demands, bars, kerf_mm=0):
    return CuttingProblem1D(
        part_lengths_mm=parts,
        demands=demands,
        bars=tuple(bar(l) if isinstance(l, int) else l for l in bars),
        kerf_mm=kerf_mm,
    )


def engine_problem(payload, model="cutting_1d"):
    return Problem(metadata={"model": model, "cutting": payload})


class TestEngineSolve:
    def test_feasible_solution_shape(self):
        p = engine_problem(
            cutting([4000, 2000], [2, 2], [bar(6000, "b0"), bar(6000, "b1")])
        )
        sol = solve(p)
        assert isinstance(sol, Solution)
        assert sol.feasibility == {"feasible": True, "reasons": []}
        assert sol.objective["value"] == pytest.approx(20.0)
        assert sol.objective["lp_value"] == pytest.approx(20.0)
        assert sol.objective["gap"] == pytest.approx(0.0)
        assert len(sol.assignments) == 2
        for entry in sol.assignments:
            assert set(entry) == {"bar_id", "pattern", "offcut_mm"}
            assert isinstance(entry["pattern"], list)
        assert {a["bar_id"] for a in sol.assignments} == {"b0", "b1"}
        total = [0, 0]
        for entry in sol.assignments:
            for i, count in enumerate(entry["pattern"]):
                total[i] += count
        assert total == [2, 2]
        assert sol.metrics["parts"] == [
            {"index": 0, "length_mm": 4000, "demand": 2, "produced": 2},
            {"index": 1, "length_mm": 2000, "demand": 2, "produced": 2},
        ]
        assert sol.metrics["bars_used"] == 2
        assert sol.metrics["waste_mm"] == 0
        assert sol.metrics["bars"] == [
            {"bar_id": "b0", "length_mm": 6000, "cost": 10.0, "origin": "new"},
            {"bar_id": "b1", "length_mm": 6000, "cost": 10.0, "origin": "new"},
        ]
        assert sol.metrics["contexts"] == [
            {"length_mm": 6000, "cost": 10.0, "bars_available": 2, "bars_used": 2},
        ]
        assert sol.stats["model"] == "cutting_1d"
        assert sol.stats["ip_status"] == "optimal"
        assert sol.stats["solve_time_s"] >= 0.0

    def test_infeasible_carries_reasons(self):
        p = engine_problem(cutting([4000], [3], [bar(6000, "b0"), bar(6000, "b1")]))
        sol = solve(p)
        assert sol.feasibility["feasible"] is False
        assert sol.feasibility["reasons"] == [
            "part 0 (length 4000mm): at most 2 copies fit on the bars "
            "on hand, demand is 3",
        ]
        assert sol.assignments == []
        assert sol.objective["value"] == 0.0
        assert sol.metrics["parts"][0]["produced"] == 0

    def test_zero_demand_is_empty(self):
        p = engine_problem(cutting([4000], [0], [bar(6000)]))
        sol = solve(p)
        assert sol.feasibility["feasible"] is True
        assert sol.assignments == []
        assert sol.objective["value"] == 0.0
        assert sol.metrics["parts"] == [
            {"index": 0, "length_mm": 4000, "demand": 0, "produced": 0},
        ]

    def test_unknown_model_not_implemented(self):
        p = engine_problem(cutting([4000], [1], [bar(6000)]), model="scheduling_1d")
        with pytest.raises(NotImplementedError):
            solve(p)

    def test_missing_model_raises(self):
        p = Problem(metadata={"cutting": cutting([4000], [1], [bar(6000)])})
        with pytest.raises(ValueError):
            solve(p)

    def test_missing_payload_raises(self):
        p = Problem(metadata={"model": "cutting_1d"})
        with pytest.raises(ValueError):
            solve(p)


class TestExplain:
    def _standard(self):
        # Three bars on hand, two consumed: the context is not
        # exhausted, so exactly-met parts explain without scarcity.
        p = engine_problem(
            cutting([4000, 2000], [2, 2], [bar(6000, "b0"), bar(6000, "b1"), bar(6000, "b2")])
        )
        return solve(p)

    def test_exactly_met_part(self):
        sol = self._standard()
        reasons = explain(sol, "0")
        assert [r.kind for r in reasons] == ["binding_constraint"]
        (reason,) = reasons
        assert reason.subject == "part-0"
        assert reason.detail == {
            "part_index": 0,
            "demand": 2,
            "produced": 2,
            "note": "demand exactly met",
        }

    def test_task_id_by_length(self):
        sol = self._standard()
        reasons = explain(sol, "2000")
        (reason,) = reasons
        assert reason.subject == "part-1"
        assert reason.detail["demand"] == 2
        assert reason.detail["produced"] == 2

    def test_unknown_task_raises(self):
        sol = self._standard()
        with pytest.raises(ValueError):
            explain(sol, "99")

    def test_alternative_worse(self):
        # A strictly pricier 8000mm bar could also hold the 4000mm
        # part; the engine says so with the price difference.
        p = engine_problem(cutting([4000], [1], [bar(6000, "b0"), bar(8000, "b1", cost=20.0)]))
        sol = solve(p)
        assert sol.feasibility["feasible"] is True
        reasons = explain(sol, "0")
        assert [r.kind for r in reasons] == ["binding_constraint", "alternative_worse"]
        alt = reasons[1]
        assert alt.subject == "part-0"
        assert alt.detail["chosen_context_mm"] == 6000
        assert alt.detail["alternative_context_mm"] == 8000
        assert alt.detail["delta_per_part"] == pytest.approx(10.0)

    def test_resource_scarcity(self):
        # Every bar on hand is consumed: the context is exhausted.
        p = engine_problem(cutting([6000, 6000], [1, 1], [bar(6000, "b0"), bar(6000, "b1")]))
        sol = solve(p)
        assert sol.feasibility["feasible"] is True
        reasons = explain(sol, "0")
        assert [r.kind for r in reasons] == ["binding_constraint", "resource_scarcity"]
        scar = reasons[1]
        assert scar.detail == {
            "length_mm": 6000,
            "bars_used": 2,
            "bars_available": 2,
        }

    def test_infeasible_unmet(self):
        p = engine_problem(cutting([4000], [2], [bar(6000)]))
        sol = solve(p)
        assert sol.feasibility["feasible"] is False
        reasons = explain(sol, "0")
        assert [r.kind for r in reasons] == ["binding_constraint"]
        assert reasons[0].detail == {
            "part_index": 0,
            "demand": 2,
            "produced": 0,
            "note": "demand unmet (problem infeasible)",
        }

    def test_no_demand_note(self):
        p = engine_problem(cutting([4000], [0], [bar(6000)]))
        sol = solve(p)
        reasons = explain(sol, "0")
        assert [r.kind for r in reasons] == ["binding_constraint"]
        assert reasons[0].detail["note"] == "no demand"

    def test_overproduced_note(self):
        # Cover semantics: the plan may produce more than the demand;
        # the excess returns to the ledger as a longer remnant.
        sol = Solution(
            feasibility={"feasible": True, "reasons": []},
            metrics={
                "parts": [
                    {"index": 0, "length_mm": 3000, "demand": 1, "produced": 2},
                ],
                "bars": [],
                "contexts": [],
            },
        )
        reasons = explain(sol, "0")
        assert [r.kind for r in reasons] == ["binding_constraint"]
        assert reasons[0].detail["note"] == (
            "overproduced; excess pieces return as remnants"
        )


class TestReplan:
    def test_warm_start_and_stamp(self):
        p = engine_problem(
            cutting([4000, 2000], [2, 2], [bar(6000, "b0"), bar(6000, "b1")])
        )
        previous = solve(p)
        again = replan(p, previous, ["restocked one bar"])
        assert again.feasibility["feasible"] is True
        assert again.objective["value"] == pytest.approx(20.0)
        assert again.stats["replan"] == {
            "events": ["restocked one bar"],
            "warm_start_columns": 2,
        }

    def test_rejects_non_solution(self):
        p = engine_problem(cutting([4000, 2000], [2, 2], [bar(6000, "b0"), bar(6000, "b1")]))
        with pytest.raises(ValueError):
            replan(p, {"assignments": []}, [])


class TestDiagnose:
    def test_no_bars(self):
        p = engine_problem(cutting([2000], [1], []))
        (diagnosis,) = diagnose(p)
        assert isinstance(diagnosis, Diagnosis)
        assert diagnosis.constraint_ids == ["bars"]
        assert diagnosis.suggested_removals == []
        assert "no bars" in diagnosis.explanation

    def test_part_failures(self):
        # A part longer than any bar defeats all three per-part and
        # volume classes; each reports its own diagnosis, in order.
        p = engine_problem(cutting([7000], [1], [bar(6000)]))
        diagnoses = diagnose(p)
        assert [d.constraint_ids for d in diagnoses] == [
            ["part-0"],
            ["part-0"],
            ["volume"],
        ]
        assert diagnoses[0].suggested_removals == ["part-0"]
        assert diagnoses[1].suggested_removals == ["part-0"]
        assert diagnoses[2].suggested_removals == []

    def test_clean_feasible_empty(self):
        p = engine_problem(
            cutting([4000, 2000], [2, 2], [bar(6000, "b0"), bar(6000, "b1")])
        )
        assert diagnose(p) == []

    def test_infeasible_beyond_prechecks(self):
        # Prechecks pass (each part fits, enough copies, volume fits)
        # but no two parts share a bar, so three parts need three
        # bars and only two exist. The solver proves it.
        p = engine_problem(
            cutting([4000, 4000, 3000], [1, 1, 1], [bar(6000, "b0"), bar(6000, "b1")])
        )
        (diagnosis,) = diagnose(p)
        assert diagnosis.constraint_ids == []
        assert diagnosis.explanation
