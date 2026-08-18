"""Tests for the structured feasibility precheck.

``precheck`` returns one ``PrecheckFailure`` per violated class, with
the numbers behind the failure in ``detail`` so the engine's diagnose
can build a human ``Diagnosis`` without re-deriving anything. The
legacy human-readable strings that ``solve`` attaches to infeasible
results are pinned here, one per failure, in order.
"""

from constraint_engine.geometry.solve import (
    PrecheckFailure,
    precheck,
    solve,
)
from constraint_engine.geometry.types import BarSpec, CuttingProblem1D


def bar(length_mm, cost=10.0, bar_id="b0"):
    return BarSpec(length_mm=length_mm, cost=cost, origin="new", id=bar_id)


def problem(parts, demands, bars, kerf_mm=0):
    return CuttingProblem1D(
        part_lengths_mm=parts,
        demands=demands,
        bars=tuple(bar(l) if isinstance(l, int) else l for l in bars),
        kerf_mm=kerf_mm,
    )


class TestPrecheck:
    def test_clean_problem_has_no_failures(self):
        p = problem([4000, 2000], [2, 2], [bar(6000, bar_id="b0"), bar(6000, bar_id="b1")])
        assert precheck(p) == ()

    def test_demand_without_bars(self):
        p = problem([2000], [1], [])
        assert precheck(p) == (PrecheckFailure("no_bars"),)

    def test_no_demand_without_bars_is_clean(self):
        p = problem([2000], [0], [])
        assert precheck(p) == ()

    def test_zero_demand_parts_are_ignored(self):
        # A part with no demand is ignored even when it cannot fit.
        p = problem([7000], [0], [bar(6000)])
        assert precheck(p) == ()

    def test_part_longer_than_any_bar(self):
        # A part that exceeds every bar also defeats the copies and
        # volume classes; all three report, in that order.
        p = problem([7000], [1], [bar(6000)])
        failures = precheck(p)
        assert [f.kind for f in failures] == [
            "part_too_long",
            "copies_exceed_supply",
            "volume_exceeded",
        ]
        assert failures[0].detail == {
            "part_index": 0,
            "length_mm": 7000,
            "longest_bar_mm": 6000,
        }

    def test_copies_exceed_supply_alone(self):
        # Two 6000mm bars hold at most two 4000mm parts, demand is
        # three. The total part length exactly fills the bars, so only
        # the copies class fails.
        p = problem([4000], [3], [bar(6000, bar_id="b0"), bar(6000, bar_id="b1")])
        (failure,) = precheck(p)
        assert failure.kind == "copies_exceed_supply"
        assert failure.detail == {
            "part_index": 0,
            "length_mm": 4000,
            "max_copies": 2,
            "demand": 3,
        }

    def test_volume_exceeded_alone(self):
        # Each part fits a bar and each has enough copies, but together
        # they exceed the total bar length.
        p = problem([4000, 3000], [1, 1], [bar(6000)])
        (failure,) = precheck(p)
        assert failure.kind == "volume_exceeded"
        assert failure.detail == {"part_volume_mm": 7000, "bar_volume_mm": 6000}


class TestLegacyMessages:
    """The exact strings ``solve`` reports, pinned one per failure."""

    def test_pins_the_no_bars_string(self):
        p = problem([2000], [1], [])
        result = solve(p)
        assert result.infeasible_reasons == (
            "the problem has demand but no bars are available",
        )

    def test_pins_the_part_too_long_strings(self):
        p = problem([7000], [1], [bar(6000)])
        result = solve(p)
        assert result.infeasible_reasons == (
            "part 0 (length 7000mm) is longer than the longest bar (6000mm)",
            "part 0 (length 7000mm): at most 0 copies fit on the bars "
            "on hand, demand is 1",
            "total part length (7000mm) exceeds the total bar length "
            "(6000mm)",
        )

    def test_pins_the_copies_string(self):
        p = problem([4000], [3], [bar(6000, bar_id="b0"), bar(6000, bar_id="b1")])
        result = solve(p)
        assert result.infeasible_reasons == (
            "part 0 (length 4000mm): at most 2 copies fit on the bars "
            "on hand, demand is 3",
        )

    def test_pins_the_volume_string(self):
        p = problem([4000, 3000], [1, 1], [bar(6000)])
        result = solve(p)
        assert result.infeasible_reasons == (
            "total part length (7000mm) exceeds the total bar length "
            "(6000mm)",
        )

    def test_pins_the_kerf_tightened_copies_string(self):
        # Three 1999mm cuts with kerf 3: two copies fit, demand is
        # three; the part length alone would still fit the bar.
        p = problem([1999], [3], [bar(6000)], kerf_mm=3)
        result = solve(p)
        assert result.infeasible_reasons == (
            "part 0 (length 1999mm): at most 2 copies fit on the bars "
            "on hand, demand is 3",
        )
