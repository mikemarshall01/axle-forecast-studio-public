import warnings
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from axle_studio.model import assumptions
from axle_studio.model.sampling import UNIT_COLUMNS, build_population
from axle_studio.model.settings import CohortFixture, RunSettings


def settings(vehicle_count: int = 1_000) -> RunSettings:
    return RunSettings(
        start_local_date=date(2025, 1, 7),
        warmup_days=1,
        study_days=7,
        vehicle_count=vehicle_count,
        seed=31,
        evaluation_world_count=50,
        opening_soc_fraction=0.5,
        reserve_soc_fraction=0.15,
        home_charge_efficiency=0.9,
        weather_efficiency_sensitivity_fraction_per_c=0.01,
    )


def fixture() -> CohortFixture:
    return assumptions.cohort_fixture()


def mileage_cvs() -> dict[str, float]:
    return assumptions.forecast_inputs(None, warmup_days=1, study_days=7)[
        "daily_miles_cv_by_cohort"
    ]


def personal_mileage_cvs() -> dict[str, float]:
    return assumptions.forecast_inputs(None, warmup_days=1, study_days=7)[
        "personal_mileage_cv_by_cohort"
    ]


def test_population_has_one_home_charging_power_and_no_vehicle_ac_column() -> None:
    # Decision 0004 item 34: the cohort's home charging power is the only limit.
    population = build_population(settings(60), fixture(), np.random.default_rng(1))

    assert "vehicle_ac_limit_kw" not in population.columns
    assert population["home_charger_limit_kw"].eq(7.0).all()


def test_builds_exact_thousand_vehicle_population_with_unique_stable_ids() -> None:
    population = build_population(settings(), fixture(), np.random.default_rng(123))

    assert tuple(population.columns) == UNIT_COLUMNS
    assert population["unit_id"].tolist() == [f"ev-{index:03d}" for index in range(1, 1_001)]
    assert population["unit_id"].is_unique
    assert population["population_id"].nunique() == 1
    assert population.groupby("cohort_id", sort=False).size().to_dict() == {
        "average_uk": 400,
        "intelligent_octopus": 300,
        "infrequent_charging": 100,
        "infrequent_driving": 100,
        "scheduled_charging": 90,
        "always_plugged_in": 10,
    }


def test_assignment_is_repeatable_by_seed_and_changes_with_seed() -> None:
    first = build_population(settings(), fixture(), np.random.default_rng(41))
    repeat = build_population(settings(), fixture(), np.random.default_rng(41))
    changed = build_population(settings(), fixture(), np.random.default_rng(42))

    pd.testing.assert_frame_equal(first, repeat)
    assert not first["cohort_id"].equals(changed["cohort_id"])
    assert first["population_id"].iat[0] != changed["population_id"].iat[0]


def test_preserves_current_unit_fields_cohort_traits_and_source_labels() -> None:
    run_settings = settings()
    source_fixture = fixture()
    population = build_population(run_settings, source_fixture, np.random.default_rng(7))

    specs = {cohort.cohort_id: cohort for cohort in source_fixture.cohorts}
    for cohort_id, rows in population.groupby("cohort_id", sort=False):
        spec = specs[cohort_id]
        assert rows["cohort_source_name"].eq(spec.source_name).all()
        assert rows["daily_miles_mean"].eq(spec.daily_miles_mean).all()
        assert rows["daily_miles_sd"].eq(spec.daily_miles_sd).all()
        assert rows["plug_probability"].eq(spec.plug_probability).all()
        assert rows["physical_capacity_kwh"].eq(spec.battery_capacity_kwh).all()
        assert (
            rows["efficiency_miles_per_battery_kwh"].eq(spec.efficiency_miles_per_battery_kwh).all()
        )
        assert rows["home_charger_limit_kw"].eq(spec.home_charger_limit_kw).all()
        assert rows["preferred_target_soc_fraction"].eq(spec.preferred_target_soc_fraction).all()
        assert rows["reserve_soc_fraction"].eq(run_settings.reserve_soc_fraction).all()
        assert rows["runtime_departure_local_hour"].eq(spec.departure_local_hour).all()
        assert rows["runtime_arrival_local_hour"].eq(spec.arrival_local_hour).all()
        expected_slots = (spec.arrival_local_hour - spec.departure_local_hour) % 24 * 2
        assert rows["travel_window_half_hours"].eq(expected_slots).all()


def test_applies_illustrative_cohort_daily_mileage_spreads() -> None:
    source_fixture = fixture()
    daily_miles_cv_by_cohort = mileage_cvs()
    population = build_population(
        settings(),
        source_fixture,
        np.random.default_rng(7),
        daily_miles_cv_by_cohort=daily_miles_cv_by_cohort,
    )

    specs = {cohort.cohort_id: cohort for cohort in source_fixture.cohorts}
    for cohort_id, rows in population.groupby("cohort_id", sort=False):
        source_mean = specs[cohort_id].daily_miles_mean
        expected_cv = daily_miles_cv_by_cohort[cohort_id]
        assert rows["daily_miles_mean"].eq(source_mean).all()
        assert rows["daily_miles_sd"].eq(source_mean * expected_cv).all()


def test_applies_persistent_personal_mileage_expectations() -> None:
    source_fixture = fixture()
    personal_cvs = {cohort.cohort_id: 0.0 for cohort in source_fixture.cohorts}
    personal_cvs["average_uk"] = 0.5

    population = build_population(
        settings(vehicle_count=100),
        source_fixture,
        np.random.default_rng(7),
        personal_mileage_cv_by_cohort=personal_cvs,
    )

    average_rows = population.loc[population["cohort_id"].eq("average_uk")]
    assert average_rows["daily_miles_mean"].nunique() > 1
    source_means = {cohort.cohort_id: cohort.daily_miles_mean for cohort in source_fixture.cohorts}
    non_average = population.loc[population["cohort_id"].ne("average_uk")]
    pd.testing.assert_series_equal(
        non_average["daily_miles_mean"].reset_index(drop=True),
        non_average["cohort_id"].map(source_means).reset_index(drop=True),
        check_names=False,
    )


def test_personal_mileage_mean_is_persistent_and_daily_spread_uses_it() -> None:
    source_fixture = fixture()
    personal_cvs = {cohort.cohort_id: 0.2 for cohort in source_fixture.cohorts}
    daily_cvs = {cohort.cohort_id: 0.3 for cohort in source_fixture.cohorts}

    population = build_population(
        settings(),
        source_fixture,
        np.random.default_rng(7),
        personal_mileage_cv_by_cohort=personal_cvs,
        daily_miles_cv_by_cohort=daily_cvs,
    )
    repeated_worlds = [population.loc[[0]].copy() for _ in range(25)]

    assert len({world["daily_miles_mean"].iat[0] for world in repeated_worlds}) == 1
    np.testing.assert_allclose(population["daily_miles_sd"], population["daily_miles_mean"] * 0.3)


def test_zero_personal_cv_keeps_source_daily_mean() -> None:
    source_fixture = fixture()
    personal_cvs = {cohort.cohort_id: 0.0 for cohort in source_fixture.cohorts}

    population = build_population(
        settings(),
        source_fixture,
        np.random.default_rng(7),
        personal_mileage_cv_by_cohort=personal_cvs,
    )

    expected = population["cohort_id"].map(
        {cohort.cohort_id: cohort.daily_miles_mean for cohort in source_fixture.cohorts}
    )
    pd.testing.assert_series_equal(population["daily_miles_mean"], expected, check_names=False)


def test_personal_mileage_is_reproducible_and_changes_population_identity() -> None:
    personal_cvs = {cohort.cohort_id: 0.2 for cohort in fixture().cohorts}
    first = build_population(
        settings(), fixture(), np.random.default_rng(41), personal_mileage_cv_by_cohort=personal_cvs
    )
    repeat = build_population(
        settings(), fixture(), np.random.default_rng(41), personal_mileage_cv_by_cohort=personal_cvs
    )
    changed = build_population(
        settings(),
        fixture(),
        np.random.default_rng(41),
        personal_mileage_cv_by_cohort={**personal_cvs, "average_uk": 0.25},
    )

    pd.testing.assert_frame_equal(first, repeat)
    assert first["population_id"].iat[0] != changed["population_id"].iat[0]


def test_personal_mileage_multipliers_have_mean_one() -> None:
    population = build_population(
        settings(vehicle_count=10_000),
        fixture(),
        np.random.default_rng(123),
        personal_mileage_cv_by_cohort={cohort.cohort_id: 0.2 for cohort in fixture().cohorts},
    )

    source_mean = population["cohort_id"].map(
        {cohort.cohort_id: cohort.daily_miles_mean for cohort in fixture().cohorts}
    )
    assert np.mean(population["daily_miles_mean"] / source_mean) == pytest.approx(1.0, abs=0.01)


def test_variation_does_not_change_generator_draw_order() -> None:
    actual_rng = np.random.default_rng(19)
    expected_rng = np.random.default_rng(19)

    build_population(
        settings(),
        fixture(),
        actual_rng,
        daily_miles_cv_by_cohort=mileage_cvs(),
    )
    expected_rng.permutation(np.repeat(np.arange(6), [400, 300, 100, 100, 90, 10]))
    # The zone permutation is the last population draw (trading contract v1 §3).
    expected_rng.permutation(np.repeat(np.arange(4), [250, 250, 250, 250]))

    assert actual_rng.integers(0, 2**31) == expected_rng.integers(0, 2**31)


def test_variation_is_reproducible_and_part_of_population_identity() -> None:
    daily_miles_cv_by_cohort = mileage_cvs()
    first = build_population(
        settings(),
        fixture(),
        np.random.default_rng(41),
        daily_miles_cv_by_cohort=daily_miles_cv_by_cohort,
    )
    repeat = build_population(
        settings(),
        fixture(),
        np.random.default_rng(41),
        daily_miles_cv_by_cohort=daily_miles_cv_by_cohort,
    )
    changed_cvs = {**daily_miles_cv_by_cohort, "average_uk": 0.31}
    changed = build_population(
        settings(), fixture(), np.random.default_rng(41), daily_miles_cv_by_cohort=changed_cvs
    )

    pd.testing.assert_frame_equal(first, repeat)
    assert first["population_id"].iat[0] != changed["population_id"].iat[0]


@pytest.mark.parametrize(
    "personal_mileage_cv_by_cohort",
    [
        {**{cohort.cohort_id: 0.2 for cohort in fixture().cohorts}, "unexpected": 0.2},
        {cohort.cohort_id: 0.2 for cohort in fixture().cohorts if cohort.cohort_id != "average_uk"},
        {**{cohort.cohort_id: 0.2 for cohort in fixture().cohorts}, "average_uk": -0.1},
        {**{cohort.cohort_id: 0.2 for cohort in fixture().cohorts}, "average_uk": float("nan")},
    ],
)
def test_rejects_invalid_personal_mileage_variation_map(
    personal_mileage_cv_by_cohort: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        build_population(
            settings(),
            fixture(),
            np.random.default_rng(1),
            personal_mileage_cv_by_cohort=personal_mileage_cv_by_cohort,
        )


@pytest.mark.parametrize(
    "daily_miles_cv_by_cohort",
    [
        {**mileage_cvs(), "unexpected": 0.2},
        {
            cohort_id: value
            for cohort_id, value in mileage_cvs().items()
            if cohort_id != "average_uk"
        },
        {**mileage_cvs(), "average_uk": -0.1},
        {**mileage_cvs(), "average_uk": float("nan")},
        {**mileage_cvs(), "average_uk": 1e308},
    ],
)
def test_rejects_invalid_daily_mileage_variation_map(
    daily_miles_cv_by_cohort: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        build_population(
            settings(),
            fixture(),
            np.random.default_rng(1),
            daily_miles_cv_by_cohort=daily_miles_cv_by_cohort,
        )


def test_source_hundred_vehicle_shares_remain_exact_when_requested() -> None:
    population = build_population(settings(vehicle_count=100), fixture(), np.random.default_rng(3))

    assert population.groupby("cohort_id").size().to_dict() == {
        "always_plugged_in": 1,
        "average_uk": 40,
        "infrequent_charging": 10,
        "infrequent_driving": 10,
        "intelligent_octopus": 30,
        "scheduled_charging": 9,
    }


def test_uses_caller_generator_for_exactly_the_cohort_and_zone_permutations() -> None:
    actual_rng = np.random.default_rng(19)
    expected_rng = np.random.default_rng(19)

    build_population(settings(), fixture(), actual_rng)
    expected_rng.permutation(np.repeat(np.arange(6), [400, 300, 100, 100, 90, 10]))
    # The zone permutation is the last population draw (trading contract v1 §3).
    expected_rng.permutation(np.repeat(np.arange(4), [250, 250, 250, 250]))

    assert actual_rng.integers(0, 2**31) == expected_rng.integers(0, 2**31)


def test_editable_cohort_mix_changes_only_the_counts_it_moves() -> None:
    # Decision 0004 item 54 causal check at the build_population level (the
    # forecast-level version lives in test_assumption_effects.py): moving 5
    # points from average_uk to always_plugged_in, leaving the other four
    # shares untouched, must change only those two cohorts' exact counts
    # (sampling._cohort_counts' largest-remainder apportionment), never the
    # other four or the fleet total.
    edited = assumptions.cohort_fixture(
        {
            "average_uk.population_share_percent": 35.0,
            "always_plugged_in.population_share_percent": 6.0,
        }
    )
    population = build_population(settings(), edited, np.random.default_rng(123))

    assert population.groupby("cohort_id", sort=False).size().to_dict() == {
        "average_uk": 350,
        "intelligent_octopus": 300,
        "infrequent_charging": 100,
        "infrequent_driving": 100,
        "scheduled_charging": 90,
        "always_plugged_in": 60,
    }


def test_editable_cohort_mix_draws_no_new_random_numbers() -> None:
    # CRN (module docstring "Common random numbers"): build_population's one
    # permutation call draws a fixed number of values for a given
    # vehicle_count, whatever the split between cohorts, so editing the mix
    # (item 54) never shifts the random numbers every later channel (trips,
    # connections, weather, prices) consumes -- the same guarantee editing
    # any other assumption already gets. A generator seeded the same way
    # before and after build_population must therefore land on the same next
    # value regardless of which fixture (default or edited shares) was used.
    default_rng = np.random.default_rng(77)
    build_population(settings(), fixture(), default_rng)

    edited = assumptions.cohort_fixture(
        {
            "average_uk.population_share_percent": 35.0,
            "always_plugged_in.population_share_percent": 6.0,
        }
    )
    edited_rng = np.random.default_rng(77)
    build_population(settings(), edited, edited_rng)

    assert default_rng.random() == edited_rng.random()


def test_cohort_counts_tolerate_percent_derived_rounding_noise() -> None:
    # sampling._cohort_counts' sum-to-one tolerance is 0.1 percentage points
    # (item 54), looser than the dialog's own +/-0.01 point check
    # (assumptions.validation_errors), so a fixture built directly from a
    # percent value divided by 100 -- bypassing the dialog, as this test does
    # -- is not rejected by floating-point noise. Genuinely wrong sums (for
    # example one share replaced by 0.5 with the other five untouched, 10
    # points off) are still rejected: see test_rejects_invalid_shares_and_counts.
    close_enough = replace(
        fixture(),
        cohorts=(
            replace(fixture().cohorts[0], population_share_fraction=0.4005),
            *fixture().cohorts[1:],
        ),
    )
    population = build_population(settings(), close_enough, np.random.default_rng(1))
    # The 0.05-point excess is too small to shift any exact count at 1,000 EVs.
    assert population.groupby("cohort_id", sort=False).size().to_dict() == {
        "average_uk": 400,
        "intelligent_octopus": 300,
        "infrequent_charging": 100,
        "infrequent_driving": 100,
        "scheduled_charging": 90,
        "always_plugged_in": 10,
    }


def test_callers_can_reuse_the_same_population_unchanged_across_worlds() -> None:
    population = build_population(settings(), fixture(), np.random.default_rng(5))
    before = population.copy(deep=True)

    hypothetical_world_populations = [population] * 100

    assert all(
        world_population is population for world_population in hypothetical_world_populations
    )
    pd.testing.assert_frame_equal(population, before)
    assert "world_id" not in population.columns


@pytest.mark.parametrize(
    ("run_settings", "source_fixture", "message"),
    [
        (settings(vehicle_count=0), fixture(), "vehicle_count must be a positive integer"),
        (
            settings(),
            replace(
                fixture(),
                cohorts=(
                    replace(fixture().cohorts[0], population_share_fraction=0.5),
                    *fixture().cohorts[1:],
                ),
            ),
            "population shares must sum to one",
        ),
        (
            settings(),
            replace(fixture(), cohorts=fixture().cohorts[:5]),
            "exactly six cohorts",
        ),
    ],
)
def test_rejects_invalid_shares_and_counts(
    run_settings: RunSettings,
    source_fixture: CohortFixture,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_population(run_settings, source_fixture, np.random.default_rng(1))


def test_personal_mileage_multipliers_match_lognormal_cv() -> None:
    # Mean-one lognormal via scipy.stats.lognorm (decision 0004 item 14): the
    # multiplier's CV should equal the configured personal CV, including CV > 1.
    for cv in (0.3, 1.5):
        population = build_population(
            settings(vehicle_count=20_000),
            fixture(),
            np.random.default_rng(8),
            personal_mileage_cv_by_cohort={cohort.cohort_id: cv for cohort in fixture().cohorts},
        )
        source_mean = population["cohort_id"].map(
            {cohort.cohort_id: cohort.daily_miles_mean for cohort in fixture().cohorts}
        )
        multipliers = population["daily_miles_mean"] / source_mean
        assert (multipliers > 0.0).all()
        assert multipliers.mean() == pytest.approx(1.0, abs=0.05 * max(cv, 1.0))
        assert np.log(multipliers).std() == pytest.approx(np.sqrt(np.log1p(cv**2)), rel=0.03)


def test_personal_cv_edit_changes_only_that_cohort_and_not_the_draw_count() -> None:
    # Common random numbers (plan M6): setting one cohort's personal CV to zero
    # fixes its EVs at the cohort mean but leaves every other EV's multiplier
    # and the generator position unchanged.
    base_cvs = {cohort.cohort_id: 0.2 for cohort in fixture().cohorts}
    edited_cvs = {**base_cvs, fixture().cohorts[0].cohort_id: 0.0}
    base_rng = np.random.default_rng(12)
    edited_rng = np.random.default_rng(12)
    base = build_population(settings(), fixture(), base_rng, personal_mileage_cv_by_cohort=base_cvs)
    edited = build_population(
        settings(), fixture(), edited_rng, personal_mileage_cv_by_cohort=edited_cvs
    )

    assert edited_rng.random() == base_rng.random()
    changed = edited["cohort_id"] == fixture().cohorts[0].cohort_id
    pd.testing.assert_series_equal(
        edited.loc[~changed, "daily_miles_mean"], base.loc[~changed, "daily_miles_mean"]
    )
    assert edited.loc[changed, "daily_miles_mean"].nunique() == 1


@pytest.mark.parametrize("cv", [10.5, 1e160])
def test_rejects_non_physical_personal_cv_without_warnings(cv: float) -> None:
    # Regression: CV 1e160 overflowed CV² and silently gave zero mileage.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="at most 10"):
            build_population(
                settings(),
                fixture(),
                np.random.default_rng(1),
                personal_mileage_cv_by_cohort={c.cohort_id: cv for c in fixture().cohorts},
            )


def test_personal_cv_at_bound_is_finite_and_positive_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        population = build_population(
            settings(),
            fixture(),
            np.random.default_rng(1),
            personal_mileage_cv_by_cohort={c.cohort_id: 10.0 for c in fixture().cohorts},
        )
    assert np.isfinite(population["daily_miles_mean"]).all()
    assert (population["daily_miles_mean"] > 0.0).all()
