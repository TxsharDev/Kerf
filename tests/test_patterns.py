"""Tests for pattern and context handling.

A pattern is a tuple of integer part counts that fits on one bar,
accounting for kerf between adjacent cuts:

    sum(length_i * count_i) + kerf * (total_cuts - 1 if total_cuts else 0) <= bar_length

A context groups bars of the same length and cost class (fresh stock of
one price, or all remnants of one length) so the master problem can use
one variable per pattern per context instead of one per physical bar.
Remnants inside a context are consumed oldest first.
"""

import pytest

from constraint_engine.geometry.patterns import (
    Context,
    pattern_cuts,
    pattern_fits,
    pattern_offcut,
    patterns_from_problem,
)


class TestPatternFits:
    def test_exact_fit(self):
        assert pattern_fits((1, 1), [3000, 3000], kerf_mm=0, bar_length_mm=6000)

    def test_kerf_consumed_between_cuts(self):
        # Two parts of 2999 on a 6000 bar with kerf 3: 5998 + 3 = 6001, no.
        assert not pattern_fits((1, 1), [2999, 2999], kerf_mm=3, bar_length_mm=6000)
        assert pattern_fits((1, 1), [2998, 2998], kerf_mm=3, bar_length_mm=6000)

    def test_three_parts_need_two_kerfs(self):
        # 3 * 1999 + 2 * 3 = 5997 + 6 = 6003 on a 6000 bar: no.
        assert not pattern_fits((3,), [1999], kerf_mm=3, bar_length_mm=6000)
        # 3 * 1998 + 2 * 3 = 5994 + 6 = 6000: exact fit, yes.
        assert pattern_fits((3,), [1998], kerf_mm=3, bar_length_mm=6000)
        # 3 * 1996 + 6 = 5994: yes, with 6mm to spare.
        assert pattern_fits((3,), [1996], kerf_mm=3, bar_length_mm=6000)

    def test_single_part_uses_no_kerf(self):
        # A lone part needs no cut, so no kerf.
        assert pattern_fits((1,), [6000], kerf_mm=3, bar_length_mm=6000)
        assert not pattern_fits((1,), [6001], kerf_mm=3, bar_length_mm=6000)

    def test_empty_pattern_fits(self):
        assert pattern_fits((0,), [1000], kerf_mm=3, bar_length_mm=6000)

    def test_width_mismatch_rejected(self):
        with pytest.raises(ValueError):
            pattern_fits((1, 1), [1000], kerf_mm=3, bar_length_mm=6000)

    def test_negative_count_rejected(self):
        with pytest.raises(ValueError):
            pattern_fits((-1, 0), [1000, 2000], kerf_mm=3, bar_length_mm=6000)

    def test_zero_kerf(self):
        assert pattern_fits((2, 1), [2000, 2000], kerf_mm=0, bar_length_mm=6000)


class TestPatternCuts:
    def test_counts(self):
        assert pattern_cuts((1, 1)) == 2
        assert pattern_cuts((3, 0)) == 3
        assert pattern_cuts(()) == 0
        assert pattern_cuts((5,)) == 5


class TestPatternOffcut:
    def test_offcut_is_remaining_physical_length(self):
        # 3 * 1996 + 2 * 3 kerf = 5994 used, 6 remains.
        assert pattern_offcut((3,), [1996], kerf_mm=3, bar_length_mm=6000) == 6

    def test_exact_fill_zero_offcut(self):
        # 2998 + 2999 + 3 kerf = 6000: exact fill, zero offcut.
        assert pattern_offcut((1, 1), [2998, 2999], kerf_mm=3, bar_length_mm=6000) == 0


def make_problem():
    from constraint_engine.geometry.types import BarSpec, CuttingProblem1D

    return CuttingProblem1D(
        part_lengths_mm=[1000, 2000],
        demands=[4, 2],
        bars=(
            BarSpec(length_mm=6000, cost=24.0, origin="new", id="n1"),
            BarSpec(length_mm=6000, cost=24.0, origin="new", id="n2"),
            BarSpec(length_mm=6000, cost=0.0, origin="remnant", id="r1", age_days=30.0),
            BarSpec(length_mm=6000, cost=0.0, origin="remnant", id="r2", age_days=5.0),
            BarSpec(length_mm=3000, cost=0.0, origin="remnant", id="r3", age_days=2.0),
        ),
        kerf_mm=3,
    )


class TestPatternsFromProblem:
    def test_groups_bars_into_contexts(self):
        contexts = patterns_from_problem(make_problem())
        # Two contexts: 6000mm fresh (2 bars), 6000mm remnants (2 bars).
        # The 3000mm remnant forms a third context.
        assert len(contexts) == 3

    def test_context_carries_cost_and_age_order(self):
        contexts = patterns_from_problem(make_problem())
        by_id = {tuple(c.bar_ids): c for c in contexts}
        fresh = by_id[("n1", "n2")]
        rem6000 = by_id[("r1", "r2")]
        assert fresh.length_mm == 6000
        assert fresh.cost == 24.0
        assert fresh.bar_count == 2
        # Oldest remnant first inside the context.
        assert rem6000.bar_ids == ("r1", "r2")
        assert rem6000.cost == 0.0

    def test_remnants_same_length_different_age_one_context(self):
        # r1 (30 days) and r2 (5 days) share the 6000mm context.
        contexts = patterns_from_problem(make_problem())
        rem = [c for c in contexts if c.cost == 0.0 and c.length_mm == 6000]
        assert len(rem) == 1
        assert rem[0].bar_ids == ("r1", "r2")

    def test_bar_ids_preserve_physical_identity(self):
        contexts = patterns_from_problem(make_problem())
        all_ids = [bid for c in contexts for bid in c.bar_ids]
        assert sorted(all_ids) == ["n1", "n2", "r1", "r2", "r3"]

    def test_empty_bars(self):
        from constraint_engine.geometry.types import CuttingProblem1D

        p = CuttingProblem1D(part_lengths_mm=[1000], demands=[1], bars=())
        assert patterns_from_problem(p) == ()


class TestContext:
    def test_frozen(self):
        c = Context(
            length_mm=6000, cost=0.0, bar_ids=("r1", "r2"),
        )
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            c.cost = 5.0

    def test_bar_count_property(self):
        c = Context(length_mm=6000, cost=0.0, bar_ids=("a", "b", "c"))
        assert c.bar_count == 3
