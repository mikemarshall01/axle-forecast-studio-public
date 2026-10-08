import numpy as np
import pytest

from axle_studio.model.assumptions import forecast_inputs
from axle_studio.model.sampling import sample_daily_temperature

EXPECTED_STUDY_PREFIXES = {
    1: [8.0],
    3: [8.0, 9.0, 11.0],
    7: [8.0, 9.0, 11.0, 12.0, 10.0, 7.0, 9.0],
}


@pytest.mark.parametrize("warmup_days", [0, 1, 3])
@pytest.mark.parametrize("study_days", sorted(EXPECTED_STUDY_PREFIXES))
def test_weather_inputs_put_warmup_days_first_then_the_study_prefix(
    warmup_days: int, study_days: int
) -> None:
    inputs = forecast_inputs(None, warmup_days=warmup_days, study_days=study_days)

    # The noon-to-noon study touches one more date than it has nights: the
    # morning after the last night keeps the last study day's base
    # temperature (decision 0004 item 52).
    study = EXPECTED_STUDY_PREFIXES[study_days]
    expected = [8.0] * warmup_days + study + [study[-1]]
    assert inputs["base_temperature_c_by_day"].tolist() == expected
    assert inputs["base_temperature_c_by_day"].dtype == np.float64
    assert inputs["weather_sd_c"] == 2.0


def test_weather_inputs_reject_a_study_longer_than_the_scenario() -> None:
    with pytest.raises(ValueError, match="between 1 and 7"):
        forecast_inputs(None, warmup_days=1, study_days=8)


def test_samples_world_by_day_temperatures_in_celsius() -> None:
    base = np.array([8.0, 9.0, 11.0], dtype=np.float64)

    temperatures = sample_daily_temperature(
        np.random.default_rng(12),
        world_count=4,
        base_temperature_c_by_day=base,
        sd_c=2.0,
    )

    assert temperatures.shape == (4, 3)
    assert temperatures.dtype == np.float64
    assert np.all(np.isfinite(temperatures))


def test_zero_sd_tiles_base_exactly() -> None:
    base = np.array([8.0, 9.0, 11.0], dtype=np.float64)

    temperatures = sample_daily_temperature(
        np.random.default_rng(12),
        world_count=2,
        base_temperature_c_by_day=base,
        sd_c=0.0,
    )

    np.testing.assert_array_equal(temperatures, np.tile(base, (2, 1)))


def test_sampling_replays_with_same_seed_and_advances_generator() -> None:
    base = np.array([8.0, 9.0], dtype=np.float64)
    first_rng = np.random.default_rng(21)
    repeat_rng = np.random.default_rng(21)
    expected_rng = np.random.default_rng(21)

    first = sample_daily_temperature(
        first_rng,
        world_count=3,
        base_temperature_c_by_day=base,
        sd_c=2.0,
    )
    repeat = sample_daily_temperature(
        repeat_rng,
        world_count=3,
        base_temperature_c_by_day=base,
        sd_c=2.0,
    )
    expected_rng.normal(base, 2.0, size=(3, base.size))

    np.testing.assert_array_equal(first, repeat)
    assert first_rng.integers(0, 2**31) == expected_rng.integers(0, 2**31)


@pytest.mark.parametrize(
    "base, sd",
    [
        (np.array([], dtype=float), 2.0),
        (np.array([8.0, np.nan]), 2.0),
        (np.array([8.0, np.inf]), 2.0),
        (np.array([8.0, 9.0]), -1.0),
        (np.array([8.0, 9.0]), np.nan),
    ],
)
def test_rejects_nonfinite_or_invalid_sampling_inputs(base: np.ndarray, sd: float) -> None:
    with pytest.raises(ValueError):
        sample_daily_temperature(
            np.random.default_rng(1),
            world_count=2,
            base_temperature_c_by_day=base,
            sd_c=sd,
        )


def test_temperatures_match_normal_mean_and_sd() -> None:
    base = np.array([5.0, 12.0], dtype=np.float64)
    temperatures = sample_daily_temperature(
        np.random.default_rng(14),
        world_count=20_000,
        base_temperature_c_by_day=base,
        sd_c=2.5,
    )
    np.testing.assert_allclose(temperatures.mean(axis=0), base, atol=0.05)
    np.testing.assert_allclose(temperatures.std(axis=0), [2.5, 2.5], rtol=0.02)


def test_zero_sd_consumes_the_same_draws_as_positive_sd() -> None:
    # Common random numbers (plan M6): a weather-SD edit to zero must not move
    # the generator, or every channel drawn afterwards would change.
    base = np.array([8.0, 9.0], dtype=np.float64)
    zero_rng = np.random.default_rng(6)
    positive_rng = np.random.default_rng(6)
    sample_daily_temperature(zero_rng, world_count=3, base_temperature_c_by_day=base, sd_c=0.0)
    sample_daily_temperature(positive_rng, world_count=3, base_temperature_c_by_day=base, sd_c=2.0)
    assert zero_rng.random() == positive_rng.random()
