"""InfeasiblePlanError: the solver's structured infeasibility signal."""

from kerf.errors import InfeasiblePlanError


def test_by_material_preserves_groups():
    err = InfeasiblePlanError({"m1": ["no bars"], "m2": ["part too long"]})
    assert err.by_material == {"m1": ["no bars"], "m2": ["part too long"]}


def test_by_material_materials_sorted_for_determinism():
    err = InfeasiblePlanError({"zeta": ["r1"], "alpha": ["r2"]})
    assert list(err.by_material) == ["alpha", "zeta"]


def test_by_material_copies_lists():
    reasons = ["no bars"]
    err = InfeasiblePlanError({"m1": reasons})
    reasons.append("more")
    assert err.by_material["m1"] == ["no bars"]


def test_reasons_flattened_with_material_prefix():
    err = InfeasiblePlanError({"m2": ["part too long"], "m1": ["no bars"]})
    assert err.reasons == ["m1: no bars", "m2: part too long"]


def test_str_lists_every_reason():
    err = InfeasiblePlanError({"m1": ["no bars"], "m2": ["part too long"]})
    message = str(err)
    assert "m1: no bars" in message
    assert "m2: part too long" in message


def test_empty_does_not_crash():
    err = InfeasiblePlanError({})
    assert err.by_material == {}
    assert err.reasons == []
    assert "infeasible" in str(err).lower()


def test_is_exception():
    err = InfeasiblePlanError({"m1": ["no bars"]})
    try:
        raise err
    except InfeasiblePlanError as caught:
        assert caught is err
