import numpy as np
import pytest

from axle_studio.model.physics import effective_driving_efficiency


def test_zero_sensitivity_preserves_base_efficiency_for_every_world_and_day() -> None:
    base = np.array([3.5, 4.0])
    temperature = np.array([[10.0, 20.0], [30.0, 40.0]])

    result = effective_driving_efficiency(base, temperature, sensitivity_fraction_per_c=0.0)

    assert result.shape == (2, 2, 2)
    np.testing.assert_array_equal(result, np.broadcast_to(base, result.shape))


def test_ten_and_thirty_degrees_have_symmetric_loss() -> None:
    result = effective_driving_efficiency(
        np.array([3.5]), np.array([[10.0, 30.0]]), sensitivity_fraction_per_c=0.01
    )

    np.testing.assert_allclose(result, 3.15)


def test_loss_is_capped_at_fifty_percent() -> None:
    result = effective_driving_efficiency(
        np.array([3.5]), np.array([[-100.0]]), sensitivity_fraction_per_c=0.01
    )

    np.testing.assert_allclose(result, 1.75)


def test_outputs_are_finite_and_within_half_to_full_base_bounds() -> None:
    base = np.array([3.5, 4.0, 4.5])
    temperature = np.array([[0.0, 20.0, 40.0], [10.0, 30.0, 50.0]])

    result = effective_driving_efficiency(base, temperature, sensitivity_fraction_per_c=0.01)

    assert np.all(np.isfinite(result))
    lower_bound = 0.5 * base[np.newaxis, np.newaxis, :]
    upper_bound = base[np.newaxis, np.newaxis, :]
    assert np.all(result >= lower_bound)
    assert np.all(result <= upper_bound)


def test_changed_daily_driving_energy_uses_effective_efficiency() -> None:
    efficiency = effective_driving_efficiency(
        np.array([3.5]), np.array([[10.0]]), sensitivity_fraction_per_c=0.01
    )[0, 0, 0]

    assert efficiency == pytest.approx(3.15)
    assert 31.5 / efficiency == pytest.approx(10.0)


@pytest.mark.parametrize("sensitivity", [-0.01, np.nan, np.inf])
def test_rejects_invalid_sensitivity(sensitivity: float) -> None:
    # Base efficiency and temperature come from the model's own population
    # and sampler; only the user-editable sensitivity is checked here.
    with pytest.raises(ValueError):
        effective_driving_efficiency(np.array([3.5]), np.array([[20.0]]), sensitivity)


def test_does_not_mutate_inputs_and_replays_exactly() -> None:
    base = np.array([3.5, 4.0])
    temperature = np.array([[10.0, 20.0], [30.0, 40.0]])
    base_before = base.copy()
    temperature_before = temperature.copy()

    first = effective_driving_efficiency(base, temperature, sensitivity_fraction_per_c=0.01)
    second = effective_driving_efficiency(base, temperature, sensitivity_fraction_per_c=0.01)

    np.testing.assert_array_equal(base, base_before)
    np.testing.assert_array_equal(temperature, temperature_before)
    np.testing.assert_array_equal(first, second)
