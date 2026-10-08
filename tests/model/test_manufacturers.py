"""Charger manufacturers, their outages and the per-session non-response array.

Trading contract v1 §10.1d and §10.1e (decision 0004 item 59): four
illustrative makers, assigned by exact counts and one permutation, one
outage uniform per (world, study night, maker), and a (world, night, EV)
non-response probability from the control group, the outages and the
makers' response rates.  All fixtures are synthetic.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions
from axle_studio.model.forecast import _session_non_response_probability, simulate_forecast
from axle_studio.model.sampling import assign_manufacturers, sample_manufacturer_outages
from axle_studio.model.settings import RunSettings

_SETTINGS = RunSettings(
    start_local_date=date(2026, 1, 12),
    warmup_days=1,
    study_days=3,
    vehicle_count=40,
    seed=17,
    evaluation_world_count=4,
    opening_soc_fraction=0.55,
    reserve_soc_fraction=0.2,
    home_charge_efficiency=0.92,
    weather_efficiency_sensitivity_fraction_per_c=0.01,
)


def _makers(name: str, values) -> dict[str, float]:
    return {
        f"manufacturers.{name}.{maker_id}": float(value)
        for maker_id, value in zip(assumptions.MANUFACTURER_IDS, values, strict=True)
    }


def _run(model: str = "action", trading: dict | None = None, **factor_edits):
    inputs = assumptions.forecast_inputs(None, warmup_days=1, study_days=3)
    inputs["shared_factor_assumptions"] = {**inputs["shared_factor_assumptions"], **factor_edits}
    if trading is not None:
        inputs["trading_assumptions"] = {**inputs["trading_assumptions"], **trading}
    return simulate_forecast(_SETTINGS, assumptions.cohort_fixture(), model=model, **inputs)


# --- Assignment ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vehicles", "expected"),
    [(10, (4, 3, 2, 1)), (11, (5, 3, 2, 1)), (1_000, (400, 300, 200, 100))],
)
def test_exact_counts_by_largest_remainder(vehicles: int, expected: tuple[int, ...]) -> None:
    index = assign_manufacturers(np.random.default_rng(1), vehicles, (0.4, 0.3, 0.2, 0.1))

    assert index.dtype == np.int64
    assert tuple(np.bincount(index, minlength=4)) == expected


def test_assignment_takes_one_permutation_whatever_the_shares() -> None:
    for shares in ((0.25, 0.25, 0.25, 0.25), (1.0, 0.0, 0.0, 0.0)):
        rng = np.random.default_rng(9)
        assign_manufacturers(rng, 37, shares)
        reference = np.random.default_rng(9)
        reference.permutation(37)
        assert rng.random() == reference.random()


def test_shares_are_checked() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        assign_manufacturers(np.random.default_rng(1), 10, (0.5, 0.3, 0.1, 0.0))
    with pytest.raises(ValueError, match="four|4"):
        assign_manufacturers(np.random.default_rng(1), 10, (0.5, 0.5))
    with pytest.raises(ValueError, match="between zero and one"):
        assign_manufacturers(np.random.default_rng(1), 10, (1.5, -0.5, 0.0, 0.0))


def test_units_carry_the_maker_on_both_models() -> None:
    for model in ("action", "no_action"):
        units = _run(model).units
        assert units["manufacturer_id"].isin(assumptions.MANUFACTURER_IDS).all()
        np.testing.assert_array_equal(
            np.asarray(assumptions.MANUFACTURER_IDS)[units["manufacturer_index"]],
            units["manufacturer_id"],
        )
        assert units["manufacturer_id"].value_counts().to_dict() == {
            m: 10 for m in ("m1", "m2", "m3", "m4")
        }


def test_editing_shares_moves_no_other_draw() -> None:
    base = _run(trading={"trading.control_group": 1.0})
    edited = _run(trading={"trading.control_group": 1.0}, **_makers("share", (0.4, 0.3, 0.2, 0.1)))

    assert np.bincount(edited.units["manufacturer_index"], minlength=4).tolist() == [16, 12, 8, 4]
    for column in ("cohort_id", "daily_miles_mean", "zone_id", "control_group"):
        pd.testing.assert_series_equal(base.units[column], edited.units[column])
    for key, values in base.evaluation.inputs.items():
        assert pd.Series(np.ravel(values)).equals(
            pd.Series(np.ravel(edited.evaluation.inputs[key]))
        )
    for key in ("week", "day", "skip_share"):
        np.testing.assert_array_equal(base.plug_in_factors[key], edited.plug_in_factors[key])
    pd.testing.assert_frame_equal(base.evaluation_prices, edited.evaluation_prices)
    np.testing.assert_array_equal(
        base.smart_charging.non_response_uniform, edited.smart_charging.non_response_uniform
    )
    # The outage array is per maker, not per EV, so it does not move either.
    np.testing.assert_array_equal(base.manufacturer_outage, edited.manufacturer_outage)


# --- Outages -----------------------------------------------------------------------


def test_outage_block_is_one_uniform_block_whatever_the_probabilities() -> None:
    rng = np.random.default_rng(4)
    uniforms = sample_manufacturer_outages(rng, 3, 7, 4)
    reference = np.random.default_rng(4)

    np.testing.assert_array_equal(uniforms, reference.random((3, 7, 4)))
    assert rng.random() == reference.random()


def test_outage_probability_only_switches_outages() -> None:
    never = _run(**_makers("outage_probability_per_night", (0.0,) * 4))
    always = _run(**_makers("outage_probability_per_night", (1.0, 0.0, 0.0, 0.0)))

    assert not never.manufacturer_outage.any()
    assert always.manufacturer_outage[:, :, 0].all()
    assert not always.manufacturer_outage[:, :, 1:].any()
    assert always.manufacturer_outage.shape == (4, 3, 4)
    np.testing.assert_array_equal(
        never.smart_charging.non_response_uniform, always.smart_charging.non_response_uniform
    )
    # Outages act on the smart path only (§10.1e): the normal path is unchanged.
    world = [run.fleet_world_intervals for run in (never, always)]
    normal = [frame.loc[frame["path_id"].eq("normal")] for frame in world]
    pd.testing.assert_frame_equal(normal[0], normal[1])
    assert not world[0].equals(world[1])


def test_a_larger_outage_probability_never_removes_an_outage() -> None:
    # One uniform per maker-night compared with the probability, so raising
    # it only adds outages (Compare runs on identical futures).
    low = _run(**_makers("outage_probability_per_night", (0.3,) * 4)).manufacturer_outage
    high = _run(**_makers("outage_probability_per_night", (0.7,) * 4)).manufacturer_outage
    assert not (low & ~high).any()
    assert high.sum() > low.sum()


# --- The session non-response array (§10.1e) -------------------------------------------


def test_array_rule_by_hand() -> None:
    control = np.array([False, False, True, False])
    maker = np.array([0, 1, 0, 1])
    outage = np.zeros((2, 3, 2), dtype=bool)
    outage[1, 2, 1] = True
    rate = np.array([0.9, 0.6])

    probability = _session_non_response_probability(control, maker, outage, rate)

    assert probability.shape == (2, 3, 4)
    assert probability.dtype == np.float64
    np.testing.assert_allclose(probability[0, 0], [0.1, 0.4, 1.0, 0.4])
    # Maker 1 is out in world 1, night 2: both its EVs lose control.
    np.testing.assert_allclose(probability[1, 2], [0.1, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(probability[1, 1], [0.1, 0.4, 1.0, 0.4])


def test_default_rates_without_outages_equal_the_per_archetype_rates() -> None:
    # All makers at r = 0.95, π = 0 give 0.05 for every treated session: the
    # array the kernel lane swaps in equals §9.5's six records at 0.05.
    run = _run(
        trading={"trading.control_group": 1.0},
        **_makers("outage_probability_per_night", (0.0,) * 4),
    )
    per_archetype = np.broadcast_to(run.non_response_probability, (4, 3, 40))

    np.testing.assert_allclose(run.session_non_response_probability, per_archetype, atol=1e-15)
    control = run.units["control_group"].to_numpy()
    assert control.any()
    assert (run.session_non_response_probability[:, :, control] == 1.0).all()


def test_full_and_zero_response_rates() -> None:
    no_outage = _makers("outage_probability_per_night", (0.0,) * 4)
    always = _run(**no_outage, **_makers("response_rate", (1.0,) * 4))
    never = _run(**no_outage, **_makers("response_rate", (0.0,) * 4))

    assert (always.session_non_response_probability == 0.0).all()
    assert (never.session_non_response_probability == 1.0).all()


def test_one_maker_always_out_loses_only_its_own_slice() -> None:
    run = _run(**_makers("outage_probability_per_night", (0.0, 1.0, 0.0, 0.0)))
    maker_b = run.units["manufacturer_index"].to_numpy() == 1

    assert (run.session_non_response_probability[:, :, maker_b] == 1.0).all()
    np.testing.assert_allclose(run.session_non_response_probability[:, :, ~maker_b], 0.05)


def test_no_action_run_has_no_outages_or_non_response_array() -> None:
    run = _run("no_action")
    assert run.manufacturer_outage is None
    assert run.session_non_response_probability is None
    assert run.plug_in_factors is not None


def test_invalid_factor_values_are_rejected_before_sampling() -> None:
    with pytest.raises(ValueError, match="between"):
        _run(**{"manufacturers.response_rate.m1": 1.2})
    with pytest.raises(ValueError, match="0 or 1"):
        _run(**{"behaviour.holiday_half_term_week": 0.5})
    with pytest.raises(ValueError, match="between"):
        _run(**{"behaviour.plug_in_day_factor_sd": float("nan")})
    with pytest.raises(ValueError, match="sum to one"):
        _run(**_makers("share", (0.5, 0.5, 0.5, 0.0)))
    inputs = assumptions.forecast_inputs(None, warmup_days=1, study_days=3)
    inputs["shared_factor_assumptions"].pop("behaviour.plug_in_skip_median")
    with pytest.raises(ValueError, match="missing"):
        simulate_forecast(_SETTINGS, assumptions.cohort_fixture(), **inputs)
